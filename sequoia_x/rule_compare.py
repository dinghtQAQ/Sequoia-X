"""规则比较入口：一次比较一个选股策略的全部交易规则。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.strategy.ma_volume import MaVolumeStrategy


class RuleCompareError(RuntimeError):
    """比较所需的本地数据不足，或请求超出当前可运行范围。"""


@dataclass(frozen=True)
class CompareParams:
    strategy: str
    db_path: str
    results_dir: Path
    run_at: datetime
    start: str
    end: str
    initial_cash: float = 50_000
    commission_rate: float = 0.00016
    min_commission: float = 5
    stamp_duty: float = 0.0005
    slippage: float = 0.001
    risk_fraction: float = 0.02
    atr_multiple: float = 2
    position_cap: float = 0.2
    max_positions: int = 4
    max_new_per_day: int = 1
    max_hold_days: int | None = None


@dataclass(frozen=True)
class BuyRule:
    key: str
    label: str


@dataclass(frozen=True)
class SellRule:
    key: str
    label: str
    stop: float | None
    take: float | None
    hold: int
    ma_exit: bool


BUY_RULES = (
    BuyRule("high", "信号日最高价"),
    BuyRule("close", "信号日收盘价"),
    BuyRule("close-down2", "收盘价下跌2%"),
    BuyRule("ma20", "20日均线"),
)
SELL_RULES = (
    SellRule("stop8-take15-hold10", "8%止损+15%止盈+10个交易日", 0.08, 0.15, 10, False),
    SellRule("stop8-hold10", "8%止损+10个交易日", 0.08, None, 10, False),
    SellRule("stop5-take10-hold5", "5%止损+10%止盈+5个交易日", 0.05, 0.10, 5, False),
    SellRule("ma20-take15-hold20", "跌破20日均线+15%止盈+20个交易日", None, 0.15, 20, True),
)


def compare_rules(
    strategy: str,
    db_path: str,
    results_dir: str | Path,
    run_at: str,
    start: str,
    end: str,
    initial_cash: float = 50_000,
    commission_rate: float = 0.00016,
    min_commission: float = 5,
    stamp_duty: float = 0.0005,
    slippage: float = 0.001,
    risk_fraction: float = 0.02,
    atr_multiple: float = 2,
    position_cap: float = 0.2,
    max_positions: int = 4,
    max_new_per_day: int = 1,
    max_hold_days: int | None = None,
) -> None:
    """逐日重跑一个选股策略，并用本地不复权日线模拟全部比较场景。"""
    params = CompareParams(
        strategy=strategy,
        db_path=db_path,
        results_dir=Path(results_dir),
        run_at=datetime.strptime(run_at, "%Y-%m-%d %H:%M:%S"),
        start=start,
        end=end,
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        min_commission=min_commission,
        stamp_duty=stamp_duty,
        slippage=slippage,
        risk_fraction=risk_fraction,
        atr_multiple=atr_multiple,
        position_cap=position_cap,
        max_positions=max_positions,
        max_new_per_day=max_new_per_day,
        max_hold_days=max_hold_days,
    )
    _require_supported(params)
    _require_local_data(params)

    settings = Settings(db_path=params.db_path, bark_key="rule-compare")
    engine = DataEngine(settings)
    strategy_impl = MaVolumeStrategy(engine=engine, settings=settings)
    dates = _dates(engine, params.start, params.end)
    if not dates:
        raise RuleCompareError("比较区间内没有本地交易日")

    scenarios = _scenarios(params)
    books = [_Book(params, scenario) for scenario in scenarios]
    for index, signal_date in enumerate(dates):
        picks = _picks_as_of(strategy_impl, signal_date) if index + 1 < len(dates) else []
        for book in books:
            book.on_day(engine, dates, index, picks)

    _capture_open_value(books)
    for book in books:
        book.close_interval(engine, dates[-1])
    _write(params, _scorecard(params, books, dates), _logs(books))


@dataclass(frozen=True)
class _Scenario:
    scenario_id: str
    buy: BuyRule
    sell: SellRule


class _Book:
    def __init__(self, params: CompareParams, scenario: _Scenario) -> None:
        self.params = params
        self.scenario = scenario
        self.cash = params.initial_cash
        self.positions: list[dict] = []
        self.trades: list[tuple[float, float]] = []
        self.equity = [params.initial_cash]
        self.logs: list[str] = []
        self.unfilled = 0
        self.opened_today = 0
        self.open_cash = params.initial_cash
        self.open_value = 0.0

    def on_day(
        self,
        engine: DataEngine,
        dates: list[str],
        index: int,
        picks: list[tuple[str, float]],
    ) -> None:
        signal_date = dates[index]
        self.opened_today = 0
        if index + 1 < len(dates):
            self._buy(engine, dates, index, picks)
        self._sell(engine, signal_date, index)
        self.equity.append(self.cash + _book_value(engine, self.positions, signal_date))
        for position in self.positions:
            bar = _raw_on(engine, position["symbol"], signal_date)
            if bar is not None:
                position["mark"] = float(bar["close"])

    def close_interval(self, engine: DataEngine, last_date: str) -> None:
        for position in list(self.positions):
            last = _raw_on(engine, position["symbol"], last_date)
            if last is None:
                raise RuleCompareError(f"区间结束缺少 {position['symbol']} 的不复权日线")
            self._close(float(last["close"]), position, "区间结束")
        if self.positions:
            self.equity.append(self.cash)

    def _buy(
        self,
        engine: DataEngine,
        dates: list[str],
        index: int,
        picks: list[tuple[str, float]],
    ) -> None:
        signal_date = dates[index]
        next_date = dates[index + 1]
        for symbol, _ranking in picks:
            if self.opened_today >= self.params.max_new_per_day:
                return
            if len(self.positions) >= self.params.max_positions:
                self._unfilled(signal_date, _output_code(symbol))
                return
            signal_raw = _raw_on(engine, symbol, signal_date)
            next_raw = _raw_on(engine, symbol, next_date)
            code = _output_code(symbol)
            if (
                signal_raw is None
                or next_raw is None
                or not _can_trade(next_raw)
                or _limit_up(signal_raw, _previous_raw(engine, symbol, signal_date))
                or _limit_up(next_raw, signal_raw)
            ):
                self._unfilled(signal_date, code)
                continue
            limit = _buy_limit(engine, symbol, signal_raw, self.scenario.buy)
            if limit is None or not _touched(next_raw, limit):
                self._unfilled(signal_date, code)
                continue
            price = limit * (1 + self.params.slippage)
            shares = _shares(engine, self, symbol, signal_date, limit)
            if shares < 100:
                self._unfilled(signal_date, code)
                continue
            cost = _buy_cost(price, shares, self.params)
            self.cash -= cost
            self.positions.append(
                {
                    "symbol": symbol,
                    "signal_date": signal_date,
                    "entry_index": index + 1,
                    "shares": shares,
                    "price": price,
                    "limit": limit,
                    "cost": cost,
                }
            )
            self.opened_today += 1
            self.logs.append(f"{self.scenario.scenario_id} {signal_date} {code} 买入")

    def _sell(self, engine: DataEngine, trade_date: str, index: int) -> None:
        for position in list(self.positions):
            if index <= position["entry_index"]:
                continue
            raw = _raw_on(engine, position["symbol"], trade_date)
            if raw is None or not _can_trade(raw):
                continue
            exit_price = _sell_price(engine, position, raw, index, self.scenario.sell, self.params)
            if exit_price is None:
                continue
            self._close(exit_price, position, "卖出")

    def _close(self, price: float, position: dict, action: str) -> None:
        proceeds = _sell_proceeds(price, position["shares"], self.params)
        self.cash += proceeds
        self.trades.append((proceeds - position["cost"], position["cost"]))
        self.logs.append(
            f"{self.scenario.scenario_id} {position['signal_date']} "
            f"{_output_code(position['symbol'])} {action}"
        )
        self.positions.remove(position)

    def _unfilled(self, signal_date: str, code: str) -> None:
        self.unfilled += 1
        self.logs.append(f"{self.scenario.scenario_id} {signal_date} {code} 无法买入")


def _scenarios(params: CompareParams) -> list[_Scenario]:
    scenarios = []
    number = 1
    for buy in BUY_RULES:
        for sell in SELL_RULES:
            scenarios.append(_Scenario(f"S{number:02d}", buy, sell))
            number += 1
    return scenarios


def _require_supported(params: CompareParams) -> None:
    if params.strategy != "MaVolume":
        raise RuleCompareError("一次比较只接受一个选股策略，当前只支持 MaVolume")


def _require_local_data(params: CompareParams) -> None:
    if not Path(params.db_path).exists():
        raise RuleCompareError(f"缺少本地数据库：{params.db_path}")
    with sqlite3.connect(params.db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if "stock_daily" not in tables or "stock_daily_raw" not in tables:
            raise RuleCompareError("缺少比较所需的本地后复权日线或不复权日线")
        adjusted = conn.execute("SELECT COUNT(*) FROM stock_daily").fetchone()[0]
        if adjusted == 0:
            raise RuleCompareError("缺少比较所需的本地后复权日线")


def _dates(engine: DataEngine, start: str, end: str) -> list[str]:
    with sqlite3.connect(engine.db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT date FROM stock_daily "
            "WHERE date >= ? AND date <= ? ORDER BY date",
            (start, end),
        ).fetchall()
    return [row[0] for row in rows]


def _picks_as_of(strategy: MaVolumeStrategy, signal_date: str) -> list[tuple[str, float]]:
    original = strategy.engine.get_ohlcv

    def limited(symbol: str) -> pd.DataFrame:
        frame = original(symbol)
        return frame[frame["date"] <= signal_date].copy()

    strategy.engine.get_ohlcv = limited
    try:
        picks = strategy.run()
    finally:
        strategy.engine.get_ohlcv = original
    if picks and not isinstance(picks[0], tuple):
        raise RuleCompareError("选股策略没有提供排序值")
    return picks


def _raw_on(engine: DataEngine, symbol: str, trade_date: str) -> pd.Series | None:
    frame = engine.get_raw_ohlcv(symbol)
    matched = frame[frame["date"] == trade_date]
    if matched.empty:
        return None
    return matched.iloc[0]


def _previous_raw(engine: DataEngine, symbol: str, trade_date: str) -> pd.Series | None:
    frame = engine.get_raw_ohlcv(symbol)
    earlier = frame[frame["date"] < trade_date]
    if earlier.empty:
        return None
    return earlier.iloc[-1]


def _can_trade(bar: pd.Series) -> bool:
    return float(bar["volume"]) > 0


def _limit_up(bar: pd.Series, previous: pd.Series | None) -> bool:
    if previous is None or float(previous["close"]) == 0:
        return False
    return float(bar["close"]) >= float(previous["close"]) * 1.095


def _limit_down(bar: pd.Series, previous: pd.Series | None) -> bool:
    if previous is None or float(previous["close"]) == 0:
        return False
    return float(bar["open"]) <= float(previous["close"]) * 0.905


def _touched(bar: pd.Series, limit: float) -> bool:
    return float(bar["low"]) <= limit <= float(bar["high"])


def _buy_limit(
    engine: DataEngine,
    symbol: str,
    signal_raw: pd.Series,
    buy: BuyRule,
) -> float | None:
    close = float(signal_raw["close"])
    if buy.key == "high":
        price = float(signal_raw["high"])
    elif buy.key == "close":
        price = close
    elif buy.key == "close-down2":
        price = close * 0.98
    else:
        ma20 = _ma(engine, symbol, str(signal_raw["date"]), 20)
        if ma20 is None:
            return None
        price = ma20
    return round(price, 2)


def _shares(
    engine: DataEngine,
    book: _Book,
    symbol: str,
    signal_date: str,
    price: float,
) -> int:
    assets = _total_assets(book)
    distance = _stop_distance(engine, book, symbol, signal_date, price)
    if distance <= 0 or price <= 0:
        return 0
    risk_shares = assets * book.params.risk_fraction / distance
    cap_shares = assets * book.params.position_cap / price
    target = min(risk_shares, cap_shares)
    affordable = book.cash / (_buy_unit_cost(price, book.params))
    return int(min(target, affordable) // 100) * 100


def _stop_distance(
    engine: DataEngine,
    book: _Book,
    symbol: str,
    signal_date: str,
    price: float,
) -> float:
    stop = book.scenario.sell.stop
    if stop is not None:
        return price * stop
    atr = _atr(engine, symbol, signal_date, 14)
    if atr is None:
        return 0
    return book.params.atr_multiple * atr


def _atr(engine: DataEngine, symbol: str, trade_date: str, window: int) -> float | None:
    frame = engine.get_raw_ohlcv(symbol)
    history = frame[frame["date"] <= trade_date].tail(window + 1)
    if len(history) < window + 1:
        return None
    previous_close = history["close"].shift(1)
    true_range = pd.concat(
        [
            history["high"] - history["low"],
            (history["high"] - previous_close).abs(),
            (history["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(true_range.iloc[1:].mean())


def _ma(engine: DataEngine, symbol: str, trade_date: str, window: int) -> float | None:
    frame = engine.get_raw_ohlcv(symbol)
    history = frame[frame["date"] <= trade_date].tail(window)
    if len(history) < window:
        return None
    return float(history["close"].mean())


def _total_assets(book: _Book) -> float:
    return book.cash + sum(position["cost"] for position in book.positions)


def _buy_unit_cost(price: float, params: CompareParams) -> float:
    return price * (1 + params.commission_rate)


def _buy_cost(price: float, shares: int, params: CompareParams) -> float:
    amount = price * shares
    commission = amount * params.commission_rate
    if params.min_commission and commission < params.min_commission:
        commission = params.min_commission
    return amount + commission


def _sell_price(
    engine: DataEngine,
    position: dict,
    bar: pd.Series,
    index: int,
    sell: SellRule,
    params: CompareParams,
) -> float | None:
    previous = _previous_raw(engine, position["symbol"], str(bar["date"]))
    if _limit_down(bar, previous):
        return float(bar["open"])
    open_price = float(bar["open"])
    low = float(bar["low"])
    high = float(bar["high"])
    entry = position["price"]
    if sell.stop is not None:
        stop = round(entry * (1 - sell.stop), 2)
        if open_price <= stop:
            return open_price
        if low <= stop:
            return stop
    if sell.take is not None:
        take = round(entry * (1 + sell.take), 2)
        if open_price >= take:
            return open_price
        if high >= take:
            return take
    if sell.ma_exit:
        ma20 = _ma(engine, position["symbol"], str(bar["date"]), 20)
        if ma20 is not None and float(bar["close"]) < ma20:
            return float(bar["close"])
    held_days = index - position["entry_index"]
    hold = params.max_hold_days if params.max_hold_days is not None else sell.hold
    if held_days >= hold:
        return open_price
    return None


def _sell_proceeds(price: float, shares: int, params: CompareParams) -> float:
    amount = price * (1 - params.slippage) * shares
    return amount * (1 - params.stamp_duty)


def _book_value(engine: DataEngine, positions: list[dict], trade_date: str) -> float:
    value = 0.0
    for position in positions:
        bar = _raw_on(engine, position["symbol"], trade_date)
        price = position["price"] if bar is None else float(bar["close"])
        value += position["shares"] * price
    return value


def _output_code(symbol: str) -> str:
    prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
    return f"{prefix}{symbol}"


def _scorecard(params: CompareParams, books: list[_Book], dates: list[str]) -> str:
    header = (
        "场景编号,买入规则,卖出规则,收益,最大回撤,收益回撤比,中位收益,交易次数,胜率,"
        "平均每笔收益,无法买入次数,期末现金,剩余持仓收盘价值,相对初始资金收益,实际区间"
    )
    rows = [_score_row(params, book, dates) for book in books]
    rows.sort(key=lambda row: (-float(row.split(",")[5]), row.split(",")[0]))
    return header + "\n" + "\n".join(rows) + "\n"


def _score_row(params: CompareParams, book: _Book, dates: list[str]) -> str:
    trade_returns = [profit / cost for profit, cost in book.trades if cost]
    returns = sum(trade_returns)
    peak = book.equity[0]
    max_drawdown = 0.0
    for value in book.equity:
        peak = max(peak, value)
        if peak:
            max_drawdown = max(max_drawdown, (peak - value) / peak)
    ratio = returns / max_drawdown if max_drawdown else 0
    wins = [item for item in trade_returns if item > 0]
    median = sorted(trade_returns)[len(trade_returns) // 2] if trade_returns else 0
    average = sum(trade_returns) / len(trade_returns) if trade_returns else 0
    win_rate = len(wins) / len(book.trades) if book.trades else 0
    remaining = book.open_value
    terminal = book.open_cash + remaining
    capital_return = terminal / params.initial_cash - 1 if params.initial_cash else 0
    return (
        f"{book.scenario.scenario_id},{book.scenario.buy.label},{book.scenario.sell.label},"
        f"{returns:.4f},{max_drawdown:.4f},{ratio:.4f},{median:.4f},{len(book.trades)},"
        f"{win_rate:.4f},{average:.4f},{book.unfilled},{book.open_cash:.4f},{remaining:.4f},"
        f"{capital_return:.4f},{dates[0]}至{dates[-1]}"
    )


def _capture_open_value(books: list[_Book]) -> None:
    for book in books:
        book.open_cash = book.cash
        book.open_value = _book_value_from_book(book)


def _book_value_from_book(book: _Book) -> float:
    return sum(position["shares"] * float(_raw_close(book, position)) for position in book.positions)


def _raw_close(book: _Book, position: dict) -> float:
    return position.get("mark", position["price"])


def _logs(books: list[_Book]) -> list[str]:
    return [line for book in books for line in book.logs]


def _write(params: CompareParams, scorecard: str, logs: list[str]) -> None:
    params.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = params.run_at.strftime("%Y%m%d%H%M%S")
    stem = f"{params.strategy}_{stamp}"
    (params.results_dir / f"{stem}.csv").write_text(scorecard, encoding="utf-8")
    (params.results_dir / f"{stem}.log").write_text(
        "\n".join(logs) + ("\n" if logs else ""),
        encoding="utf-8",
    )
