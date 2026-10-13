"""规则比较入口：一次只比较一个选股策略的一个比较场景。"""

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
    buy_rule: str
    sell_rule: str
    initial_cash: float = 50_000
    commission_rate: float = 0.00016
    min_commission: float = 5
    stamp_duty: float = 0.0005
    slippage: float = 0.001
    scenario_id: str = "S01"


def compare_rules(
    strategy: str,
    db_path: str,
    results_dir: str | Path,
    run_at: str,
    start: str,
    end: str,
    buy_rule: str = "close",
    sell_rule: str = "stop8-take15-hold10",
    initial_cash: float = 50_000,
    commission_rate: float = 0.00016,
    min_commission: float = 5,
    stamp_duty: float = 0.0005,
    slippage: float = 0.001,
) -> None:
    """逐日重跑一个选股策略，并用本地不复权日线模拟一个比较场景。"""
    params = CompareParams(
        strategy=strategy,
        db_path=db_path,
        results_dir=Path(results_dir),
        run_at=datetime.strptime(run_at, "%Y-%m-%d %H:%M:%S"),
        start=start,
        end=end,
        buy_rule=buy_rule,
        sell_rule=sell_rule,
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        min_commission=min_commission,
        stamp_duty=stamp_duty,
        slippage=slippage,
    )
    _require_supported(params)
    _require_local_data(params)

    settings = Settings(db_path=params.db_path, bark_key="rule-compare")
    engine = DataEngine(settings)
    strategy_impl = MaVolumeStrategy(engine=engine, settings=settings)
    dates = _dates(engine, params.start, params.end)
    if not dates:
        raise RuleCompareError("比较区间内没有本地交易日")

    cash = params.initial_cash
    position: dict | None = None
    trades: list[float] = []
    equity = [params.initial_cash]
    logs: list[str] = []
    unfilled = 0

    for index, signal_date in enumerate(dates):
        if position is not None and index > position["entry_index"]:
            raw = _raw_on(engine, position["symbol"], signal_date)
            if raw is not None and _can_trade(raw):
                exit_price = _sell_price(position, raw, index, params)
                if exit_price is not None:
                    proceeds = _sell_proceeds(exit_price, position["shares"], params)
                    cash += proceeds
                    trades.append(proceeds - position["cost"])
                    logs.append(
                        f"{params.scenario_id} {position['signal_date']} "
                        f"{_output_code(position['symbol'])} 卖出"
                    )
                    position = None

        if position is None and index + 1 < len(dates):
            picks = _picks_as_of(strategy_impl, signal_date)
            if picks:
                symbol = picks[0][0]
                signal_raw = _raw_on(engine, symbol, signal_date)
                next_date = dates[index + 1]
                next_raw = _raw_on(engine, symbol, next_date)
                if signal_raw is None or next_raw is None or not _can_trade(next_raw):
                    unfilled += 1
                    logs.append(
                        f"{params.scenario_id} {signal_date} {_output_code(symbol)} 无法买入"
                    )
                    continue
                else:
                    limit = round(float(signal_raw["close"]), 2)
                    filled = float(next_raw["low"]) <= limit <= float(next_raw["high"])
                    if not filled:
                        unfilled += 1
                        logs.append(
                            f"{params.scenario_id} {signal_date} {_output_code(symbol)} 无法买入"
                        )
                    else:
                        price = limit * (1 + params.slippage)
                        shares = _shares(cash, price, params)
                        if shares < 100:
                            unfilled += 1
                            logs.append(
                                f"{params.scenario_id} {signal_date} {_output_code(symbol)} 无法买入"
                            )
                        else:
                            cost = _buy_cost(price, shares, params)
                            cash -= cost
                            position = {
                                "symbol": symbol,
                                "signal_date": signal_date,
                                "entry_index": index + 1,
                                "shares": shares,
                                "price": price,
                                "cost": cost,
                            }
                            logs.append(
                                f"{params.scenario_id} {signal_date} {_output_code(symbol)} 买入"
                            )
        equity.append(cash + _position_value(engine, position, signal_date))

    if position is not None:
        last = _raw_on(engine, position["symbol"], dates[-1])
        if last is None:
            raise RuleCompareError(f"区间结束缺少 {position['symbol']} 的不复权日线")
        proceeds = _sell_proceeds(float(last["close"]), position["shares"], params)
        cash += proceeds
        trades.append(proceeds - position["cost"])
        logs.append(
            f"{params.scenario_id} {position['signal_date']} "
            f"{_output_code(position['symbol'])} 区间结束"
        )
        position = None
        equity.append(cash)

    _write(params, _scorecard(params, trades, equity, unfilled, dates), logs)


def _require_supported(params: CompareParams) -> None:
    if params.strategy != "MaVolume":
        raise RuleCompareError("一次比较只接受一个选股策略，当前只支持 MaVolume")
    if params.buy_rule != "close" or params.sell_rule != "stop8-take15-hold2":
        raise RuleCompareError("当前只跑通一个比较场景：收盘价买入，持有 2 个交易日")


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


def _can_trade(bar: pd.Series) -> bool:
    return float(bar["volume"]) > 0


def _sell_price(position: dict, bar: pd.Series, index: int, params: CompareParams) -> float | None:
    held_days = index - position["entry_index"]
    if held_days >= 2:
        return float(bar["open"])
    return None


def _shares(cash: float, price: float, params: CompareParams) -> int:
    cap_cash = min(cash, params.initial_cash / 5)
    return int(cap_cash // (price * 100)) * 100


def _buy_cost(price: float, shares: int, params: CompareParams) -> float:
    amount = price * shares
    commission = amount * params.commission_rate
    if params.min_commission and commission < params.min_commission:
        commission = params.min_commission
    return amount + commission


def _sell_proceeds(price: float, shares: int, params: CompareParams) -> float:
    amount = price * (1 - params.slippage) * shares
    return amount * (1 - params.stamp_duty)


def _position_value(engine: DataEngine, position: dict | None, trade_date: str) -> float:
    if position is None:
        return 0
    bar = _raw_on(engine, position["symbol"], trade_date)
    if bar is None:
        return position["shares"] * position["price"]
    return position["shares"] * float(bar["close"])


def _output_code(symbol: str) -> str:
    prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
    return f"{prefix}{symbol}"


def _scorecard(
    params: CompareParams,
    trades: list[float],
    equity: list[float],
    unfilled: int,
    dates: list[str],
) -> str:
    trade_returns = [
        profit / (params.initial_cash / 5) for profit in trades
    ]
    returns = sum(trade_returns)
    peak = equity[0]
    max_drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak:
            max_drawdown = max(max_drawdown, (peak - value) / peak)
    ratio = returns / max_drawdown if max_drawdown else 0
    wins = [item for item in trade_returns if item > 0]
    median = sorted(trade_returns)[len(trade_returns) // 2] if trade_returns else 0
    average = (sum(trade_returns) / len(trade_returns)) if trade_returns else 0
    win_rate = len(wins) / len(trades) if trades else 0
    header = (
        "场景编号,收益,最大回撤,收益回撤比,中位收益,交易次数,胜率,"
        "平均每笔收益,无法买入次数,实际区间"
    )
    row = (
        f"{params.scenario_id},{returns:.4f},{max_drawdown:.4f},{ratio:.4f},"
        f"{median:.4f},{len(trades)},{win_rate:.4f},{average:.4f},{unfilled},"
        f"{dates[0]}至{dates[-1]}"
    )
    return header + "\n" + row + "\n"


def _write(params: CompareParams, scorecard: str, logs: list[str]) -> None:
    params.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = params.run_at.strftime("%Y%m%d%H%M%S")
    stem = f"{params.strategy}_{stamp}"
    (params.results_dir / f"{stem}.csv").write_text(scorecard, encoding="utf-8")
    (params.results_dir / f"{stem}.log").write_text(
        "\n".join(logs) + ("\n" if logs else ""),
        encoding="utf-8",
    )
