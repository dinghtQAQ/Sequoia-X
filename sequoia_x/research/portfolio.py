"""在一个共享账户中重放固定持有期组合；不生成策略通过结论。"""

import hashlib
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from math import isfinite
from typing import Literal

from pydantic import Field

from .dataset import DatasetSnapshot, audit_dataset
from .protocol import DateWindow, FrozenModel, FrozenProtocol
from .replay import (
    DividendReceivable,
    ExecutionPolicy,
    FillAttempt,
    Signal,
    apply_corporate_actions,
    mark_position,
    quote_open_fill,
    replay_ma_signals,
    validate_execution_policy,
)

IMPLEMENTATION_VERSION = "shared-account-limit-v1"


class PortfolioPolicy(FrozenModel):
    """每日名额和中性选择必须在看结果前通过哈希绑定研究协议。"""

    daily_entry_slots: int = Field(gt=0)
    selection: Literal["symbol_order", "seeded_shuffle"]
    seed: int = Field(ge=0)

    @property
    def policy_id(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class PortfolioPosition(FrozenModel):
    trade_id: str
    signal_id: str
    symbol: str
    quantity: int = Field(ge=0)
    market_value: float
    price: float | None
    price_as_of: date | None
    valuation_status: Literal["current", "stale", "unreliable"]


class DailyAccount(FrozenModel):
    date: date
    cash: float
    available_cash: float
    frozen_cash: float
    market_value: float
    receivables_value: float
    nav: float
    daily_net_return: float
    open_exposure: float
    costs: float
    positions: tuple[PortfolioPosition, ...] = ()
    valuation_flags: tuple[str, ...] = ()


class PortfolioOrder(FrozenModel):
    order_id: str
    trade_id: str
    signal_id: str
    episode_id: str | None
    symbol: str
    side: Literal["buy", "sell"]
    created_on: date
    scheduled_for: date | None
    requested_quantity: int = Field(ge=0)
    filled_quantity: int = Field(ge=0)
    frozen_cash: float = Field(ge=0)
    cash_budget: float = Field(default=0, ge=0)
    order_type: str = "limit_at_verified_session_bounds"
    limit_price: float | None = None
    submitted_quantity: int = Field(default=0, ge=0)
    unsubmitted_quantity: int = Field(default=0, ge=0)
    reserved_cash: float = Field(default=0, ge=0)
    released_cash: float = Field(default=0, ge=0)
    reserved_quantity: int = Field(default=0, ge=0)
    released_quantity: int = Field(default=0, ge=0)
    status: Literal["pending", "filled", "partial", "expired", "rejected", "unfilled"]
    reason: str
    attempt_count: int = Field(ge=0)


class PortfolioFill(FillAttempt):
    order_id: str
    trade_id: str
    signal_id: str


class PortfolioTrade(FrozenModel):
    trade_id: str
    signal_id: str
    episode_id: str | None
    symbol: str
    strategy: str
    signal_date: date
    entry_date: date | None
    exit_date: date | None
    status: Literal["unfilled", "open", "closed"]
    entry_quantity: int = Field(ge=0)
    remaining_quantity: int = Field(ge=0)
    entry_cash: float = Field(ge=0)
    exit_cash: float
    distributions: float
    total_costs: float = Field(ge=0)
    net_pnl: float | None
    net_return: float | None
    unrealized_pnl: float
    exit_reasons: tuple[str, ...] = ()


class PortfolioCorporateAction(FrozenModel):
    trade_id: str
    signal_id: str
    symbol: str
    date: date
    quantity_before: int
    quantity_after: int
    cash_delta: float
    distribution_accrual: float
    withholding_tax: float
    receivables_value: float
    events: tuple[str, ...]
    reasons: tuple[str, ...]


class PortfolioMetrics(FrozenModel):
    closed_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    net_win_rate: float | None = None
    average_win: float | None = None
    average_loss: float | None = None
    payoff_ratio: float | None = None
    net_expectancy: float | None = None
    portfolio_net_return: float = 0
    annualized_net_return: float | None = None
    max_drawdown: float = 0
    total_costs: float = 0
    turnover: float = 0
    open_positions: int = 0
    open_exposure: float = 0
    unexitable_exposure: float = 0
    unfinished_trades: int = 0
    benchmark_net_return: float | None = None


class BenchmarkDay(FrozenModel):
    date: date
    nav: float
    daily_net_return: float


class PortfolioCoverage(FrozenModel):
    total_sessions: int
    raw_signal_count: int
    signal_dates: tuple[date, ...]
    accepted_entry_orders: int
    filled_entries: int
    unfilled_entries: int
    closed_trade_count: int
    open_lots: int
    stale_valuation_dates: tuple[date, ...]


class PortfolioResult(FrozenModel):
    protocol_id: str
    data_snapshot_id: str
    strategy_version: str
    implementation_version: str
    execution_policy_id: str
    selection_policy_id: str
    evidence_kind: Literal["synthetic", "observed"]
    data_qualified: bool
    execution_qualified: bool
    execution_reasons: tuple[str, ...]
    qualification_reasons: tuple[str, ...]
    initial_capital: float
    window: DateWindow
    signals: tuple[Signal, ...]
    daily_nav: tuple[DailyAccount, ...]
    orders: tuple[PortfolioOrder, ...]
    fills: tuple[PortfolioFill, ...]
    trades: tuple[PortfolioTrade, ...]
    metrics: PortfolioMetrics
    benchmark: tuple[BenchmarkDay, ...]
    coverage: PortfolioCoverage
    corporate_actions: tuple[PortfolioCorporateAction, ...] = ()


def run_portfolio(
    snapshot: DatasetSnapshot,
    frozen_protocol: FrozenProtocol,
    execution_policy: ExecutionPolicy,
    selection_policy: PortfolioPolicy,
    *,
    signals: tuple[Signal, ...] | None = None,
    benchmark: Mapping[date, float] | None = None,
    window: DateWindow | None = None,
) -> PortfolioResult:
    """重放公开信号；默认使用恢复版 MA，所有金额为元、数量为股。"""
    protocol = frozen_protocol.protocol
    validate_execution_policy(frozen_protocol, execution_policy)
    if f"selection_policy:{selection_policy.policy_id}" not in protocol.trading.capacity.split():
        raise ValueError("中性选择政策未通过 selection_policy 标识绑定冻结协议")
    if protocol.data_snapshot_id != snapshot.snapshot_id:
        raise ValueError("数据快照与冻结协议不符")
    if protocol.universe != snapshot.manifest.universe_id:
        raise ValueError("股票池与冻结协议不符")
    window = window or DateWindow(start=snapshot.manifest.start, end=snapshot.manifest.end)
    if window.start < snapshot.manifest.start or window.end > snapshot.manifest.end:
        raise ValueError("重放窗口超出数据样本范围")
    if window.end > protocol.data_cutoff:
        raise ValueError("重放窗口超出协议数据截止日")
    calendar = tuple(day for day in snapshot.calendar if window.start <= day <= window.end)
    if not calendar:
        raise ValueError("重放窗口没有交易日")
    if signals is None:
        signals = replay_ma_signals(snapshot, through=window.end)
    signals = tuple(
        signal for signal in signals if window.start <= signal.signal_date <= window.end
    )
    if any(signal.signal_date not in calendar for signal in signals):
        raise ValueError("外部信号日期必须属于重放窗口的全市场交易日历")
    if len({signal.signal_id for signal in signals}) != len(signals):
        raise ValueError("同一 signal_id 不可重复提交")
    reference = _benchmark_series(calendar, benchmark, protocol.trading.initial_capital)
    return _SharedAccount(
        snapshot,
        frozen_protocol,
        execution_policy,
        selection_policy,
        window,
        calendar,
        signals,
        reference,
    ).run()


def _benchmark_series(
    calendar: tuple[date, ...],
    levels: Mapping[date, float] | None,
    capital: float,
) -> tuple[BenchmarkDay, ...]:
    if levels is None:
        return ()
    if any(day not in levels for day in calendar):
        raise ValueError("基准必须覆盖完全相同的账户交易日历")
    if any(not isfinite(levels[day]) or levels[day] <= 0 for day in calendar):
        raise ValueError("基准水平必须是有限正值")
    initial = Decimal(str(levels[calendar[0]]))
    previous = _amount(capital)
    rows = []
    for day in calendar:
        nav = _amount(Decimal(str(capital)) * Decimal(str(levels[day])) / initial)
        rows.append(
            BenchmarkDay(date=day, nav=float(nav), daily_net_return=float(nav / previous - 1))
        )
        previous = nav
    return tuple(rows)


def _amount(value: float | int | Decimal) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _fill_costs(fill: FillAttempt) -> Decimal:
    return sum(
        (
            _amount(value)
            for value in (
                fill.commission,
                fill.stamp_tax,
                fill.transfer_fee,
                fill.slippage_cost,
            )
        ),
        Decimal(0),
    )


@dataclass
class _Lot:
    signal: Signal
    quantity: int = 0
    entry_quantity: int = 0
    entry_date: date | None = None
    entry_index: int | None = None
    exit_date: date | None = None
    entry_cash: Decimal = Decimal(0)
    exit_cash: Decimal = Decimal(0)
    distributions: Decimal = Decimal(0)
    costs: Decimal = Decimal(0)
    receivables: tuple[DividendReceivable, ...] = ()
    last_market_value: Decimal = Decimal(0)
    exit_reasons: list[str] = field(default_factory=list)

    @property
    def trade_id(self) -> str:
        return "lot:" + self.signal.signal_id


class _SharedAccount:
    def __init__(
        self,
        snapshot: DatasetSnapshot,
        frozen: FrozenProtocol,
        execution: ExecutionPolicy,
        selection: PortfolioPolicy,
        window: DateWindow,
        calendar: tuple[date, ...],
        signals: tuple[Signal, ...],
        benchmark: tuple[BenchmarkDay, ...],
    ) -> None:
        self.snapshot, self.frozen, self.execution, self.selection = (
            snapshot,
            frozen,
            execution,
            selection,
        )
        self.window, self.calendar, self.signals = window, calendar, signals
        self.benchmark = benchmark
        self.cash = _amount(frozen.protocol.trading.initial_capital)
        self.lots = {signal.signal_id: _Lot(signal) for signal in signals}
        self.orders: list[PortfolioOrder] = []
        self.fills: list[PortfolioFill] = []
        self.daily: list[DailyAccount] = []
        self.action_events: list[PortfolioCorporateAction] = []
        self.execution_reasons: list[str] = []
        self.bars = {(bar.symbol, bar.date): bar for bar in snapshot.bars}
        self.reasons = list(audit_dataset(snapshot).reasons)
        if not benchmark:
            self.reasons.append("缺少同日历协议基准序列；基准收益不可估计")
        self.day_costs = Decimal(0)
        self.used_volume: dict[str, int] = {}

    def _bar(self, symbol: str, session: date):
        bar = self.bars.get((symbol, session))
        return bar if bar is not None and bar.available_on <= session else None

    def _reserved(self) -> Decimal:
        return sum(
            (
                _amount(order.frozen_cash)
                for order in self.orders
                if order.side == "buy" and order.status == "pending"
            ),
            Decimal(0),
        )

    def _record_fill(self, fill: FillAttempt, order: PortfolioOrder) -> Decimal:
        self.fills.append(
            PortfolioFill(
                **fill.model_dump(),
                order_id=order.order_id,
                trade_id=order.trade_id,
                signal_id=order.signal_id,
            )
        )
        self.used_volume[order.symbol] = self.used_volume.get(order.symbol, 0) + fill.quantity
        self.cash += _amount(fill.cash_delta)
        costs = _fill_costs(fill)
        self.day_costs += costs
        if fill.reason.startswith(
            (
                "unsupported",
                "unknown",
                "missing",
                "invalid",
                "state_",
                "execution_requires",
                "execution_bar_",
                "raw_prices_",
                "risk_warning_",
            )
        ):
            reason = f"{order.order_id}:{fill.reason}"
            self.execution_reasons.append(reason)
            self.reasons.append("执行模型或输入未覆盖:" + reason)
        if self.cash < 0:
            raise ArithmeticError("账户现金守恒失败：出现借款")
        return costs

    def _prepare_opening_buys(
        self,
        session: date,
        opening_cash: Decimal,
        known_nav: Decimal,
    ) -> dict[str, tuple[Decimal, FillAttempt]]:
        """在读取本轮拍卖结果前，冻结全部申报预算及潜在证券名额。"""
        trading = self.frozen.protocol.trading
        held = {lot.signal.symbol for lot in self.lots.values() if lot.quantity}
        symbol_values: dict[str, Decimal] = {}
        for lot in self.lots.values():
            if lot.quantity:
                symbol_values[lot.signal.symbol] = (
                    symbol_values.get(lot.signal.symbol, Decimal(0)) + lot.last_market_value
                )
        free_budget = max(Decimal(0), opening_cash - self._reserved())
        prepared: dict[str, tuple[Decimal, FillAttempt]] = {}
        for position, order in enumerate(self.orders):
            if order.side != "buy" or order.status != "pending" or order.scheduled_for != session:
                continue
            own_reserve = _amount(order.frozen_cash)
            if order.symbol not in held and len(held) >= trading.max_positions:
                self.orders[position] = order.model_copy(
                    update={
                        "status": "rejected",
                        "reason": "max_positions_pre_auction",
                        "frozen_cash": 0,
                        "unsubmitted_quantity": order.requested_quantity,
                    }
                )
                free_budget += own_reserve
                continue
            allocation = max(
                Decimal(0),
                _amount(known_nav * Decimal(str(trading.max_position_fraction)))
                - symbol_values.get(order.symbol, Decimal(0)),
            )
            budget = min(own_reserve + free_budget, _amount(order.cash_budget), allocation)
            # quote公开结果的submitted/reserved只依赖事前状态、限价、费用及预算。
            # 后验open/volume只影响quantity，不参与这里的任何选择或预算决策。
            submission = quote_open_fill(
                self._bar(order.symbol, session),
                self.snapshot.state_at(order.symbol, session),
                "buy",
                order.requested_quantity,
                float(budget),
                self.execution,
            )
            reserved = _amount(submission.reserved_cash)
            free_budget += own_reserve - reserved
            if free_budget < 0:
                raise ArithmeticError("集合竞价事前预算超配")
            if submission.submitted_quantity:
                held.add(order.symbol)
                symbol_values[order.symbol] = symbol_values.get(order.symbol, Decimal(0)) + reserved
            self.orders[position] = order.model_copy(
                update={
                    "frozen_cash": float(reserved),
                    "limit_price": submission.limit_price,
                    "submitted_quantity": submission.submitted_quantity,
                    "unsubmitted_quantity": submission.unsubmitted_quantity,
                    "reserved_cash": submission.reserved_cash,
                }
            )
            prepared[order.order_id] = budget, submission
        return prepared

    def _buy_pending(
        self,
        session: date,
        index: int,
        prepared: dict[str, tuple[Decimal, FillAttempt]],
    ) -> None:
        for position, order in enumerate(self.orders):
            if order.order_id not in prepared:
                continue
            budget, submission = prepared[order.order_id]
            lot = self.lots[order.signal_id]
            fill = quote_open_fill(
                self._bar(order.symbol, session),
                self.snapshot.state_at(order.symbol, session),
                "buy",
                order.requested_quantity,
                float(budget),
                self.execution,
                used_volume=self.used_volume.get(order.symbol, 0),
            )
            if (
                fill.submitted_quantity != submission.submitted_quantity
                or fill.reserved_cash != submission.reserved_cash
            ):
                raise ArithmeticError("同批撮合结果不能改变事前申报预算")
            costs = self._record_fill(fill, order)
            if fill.quantity:
                lot.quantity = lot.entry_quantity = fill.quantity
                lot.entry_date, lot.entry_index = session, index
                lot.entry_cash = -_amount(fill.cash_delta)
                lot.costs += costs
                assert fill.market_price is not None
                lot.last_market_value = _amount(Decimal(str(fill.market_price)) * fill.quantity)
            reason = fill.reason
            if fill.quantity < fill.submitted_quantity:
                reason += ";entry_remainder_expired"
            if fill.unsubmitted_quantity:
                reason += f";unsubmitted_request:{fill.unsubmitted_quantity}"
            self.orders[position] = order.model_copy(
                update={
                    "filled_quantity": fill.quantity,
                    "frozen_cash": 0,
                    "status": fill.status,
                    "reason": reason,
                    "attempt_count": 1,
                    "released_cash": fill.released_cash,
                    "reserved_quantity": fill.reserved_quantity,
                    "released_quantity": fill.released_quantity,
                }
            )

    def _apply_actions(self, session: date) -> None:
        for lot in self.lots.values():
            if lot.entry_date is None or lot.entry_date >= session:
                continue
            if not lot.quantity and not lot.receivables:
                continue
            before = lot.quantity
            adjustment = apply_corporate_actions(
                self.snapshot,
                lot.signal.symbol,
                session,
                lot.quantity,
                lot.receivables,
            )
            lot.quantity = adjustment.quantity
            lot.receivables = adjustment.receivables
            lot.distributions += _amount(adjustment.distribution_accrual)
            lot.costs += _amount(adjustment.withholding_tax)
            self.cash += _amount(adjustment.cash_delta)
            self.day_costs += _amount(adjustment.withholding_tax)
            # 没有可靠除权后价格时，只能沿用明确标为不可靠的总市场价值。
            # 净应收与税从旧股票总值移出，不能让缺价凭空增加净值。
            lot.last_market_value = max(
                Decimal(0),
                lot.last_market_value
                - _amount(adjustment.distribution_accrual)
                - _amount(adjustment.withholding_tax),
            )
            if adjustment.events or adjustment.reasons:
                self.action_events.append(
                    PortfolioCorporateAction(
                        trade_id=lot.trade_id,
                        signal_id=lot.signal.signal_id,
                        symbol=lot.signal.symbol,
                        date=session,
                        quantity_before=before,
                        quantity_after=lot.quantity,
                        cash_delta=adjustment.cash_delta,
                        distribution_accrual=adjustment.distribution_accrual,
                        withholding_tax=adjustment.withholding_tax,
                        receivables_value=sum(item.amount for item in adjustment.receivables),
                        events=adjustment.events,
                        reasons=adjustment.reasons,
                    )
                )
            self.reasons.extend(
                f"{session.isoformat()}:{lot.signal.symbol}:{reason}"
                for reason in adjustment.reasons
            )

    def _sell_due(self, session: date, pre_auction_cash: Decimal) -> None:
        fee_cash = max(Decimal(0), pre_auction_cash - self._reserved())
        for position, order in enumerate(self.orders):
            if (
                order.side != "sell"
                or order.status == "filled"
                or order.scheduled_for is None
                or order.scheduled_for > session
            ):
                continue
            lot = self.lots[order.signal_id]
            quantity = lot.quantity
            fill = quote_open_fill(
                self._bar(order.symbol, session),
                self.snapshot.state_at(order.symbol, session),
                "sell",
                quantity,
                float(fee_cash),
                self.execution,
                acquired_on=lot.entry_date,
                used_volume=self.used_volume.get(order.symbol, 0),
            )
            costs = self._record_fill(fill, order)
            lot.exit_cash += _amount(fill.cash_delta)
            fee_cash += min(Decimal(0), _amount(fill.cash_delta))
            lot.costs += costs
            lot.quantity -= fill.quantity
            lot.last_market_value = _amount(lot.last_market_value * lot.quantity / quantity)
            if not lot.quantity:
                lot.exit_date = session
            else:
                lot.exit_reasons.append(f"{session.isoformat()}:{fill.reason}")
            self.orders[position] = order.model_copy(
                update={
                    "requested_quantity": order.filled_quantity + quantity,
                    "filled_quantity": order.filled_quantity + fill.quantity,
                    "scheduled_for": next(
                        (day for day in self.snapshot.calendar if day > session), None
                    )
                    if lot.quantity
                    else session,
                    "status": fill.status,
                    "reason": fill.reason,
                    "attempt_count": order.attempt_count + 1,
                    "submitted_quantity": fill.submitted_quantity,
                    "unsubmitted_quantity": fill.unsubmitted_quantity,
                    "limit_price": fill.limit_price,
                    "reserved_cash": fill.reserved_cash,
                    "released_cash": fill.released_cash,
                    "reserved_quantity": fill.reserved_quantity,
                    "released_quantity": fill.released_quantity,
                }
            )

    def _schedule_exits(self, session: date, index: int) -> None:
        for lot in self.lots.values():
            if (
                not lot.quantity
                or lot.entry_index is None
                or index - lot.entry_index + 1 < self.frozen.protocol.trading.holding_sessions
                or any(
                    order.side == "sell" and order.signal_id == lot.signal.signal_id
                    for order in self.orders
                )
            ):
                continue
            self.orders.append(
                PortfolioOrder(
                    order_id="sell:" + lot.signal.signal_id,
                    trade_id=lot.trade_id,
                    signal_id=lot.signal.signal_id,
                    episode_id=lot.signal.episode_id,
                    symbol=lot.signal.symbol,
                    side="sell",
                    created_on=session,
                    scheduled_for=next(
                        (day for day in self.snapshot.calendar if day > session), None
                    ),
                    requested_quantity=lot.quantity,
                    filled_quantity=0,
                    frozen_cash=0,
                    status="pending",
                    reason="after_day_60_close_next_tradable_open",
                    attempt_count=0,
                )
            )

    def _mark(self, session: date) -> tuple[tuple[PortfolioPosition, ...], tuple[str, ...]]:
        positions: list[PortfolioPosition] = []
        flags: list[str] = []
        for lot in self.lots.values():
            if not lot.quantity:
                continue
            mark = mark_position(self.snapshot, lot.signal.symbol, session, lot.quantity)
            if mark.value is not None:
                lot.last_market_value = _amount(mark.value)
            if mark.status != "current":
                flags.append(f"{lot.signal.symbol}:{mark.status}:{mark.reason}")
            positions.append(
                PortfolioPosition(
                    trade_id=lot.trade_id,
                    signal_id=lot.signal.signal_id,
                    symbol=lot.signal.symbol,
                    quantity=lot.quantity,
                    market_value=float(lot.last_market_value),
                    price=mark.price,
                    price_as_of=mark.as_of,
                    valuation_status=mark.status,
                )
            )
        return tuple(positions), tuple(flags)

    def _issue_orders(self, session: date, nav: Decimal) -> None:
        candidates = sorted(
            (signal for signal in self.signals if signal.signal_date == session),
            key=lambda signal: (signal.symbol, signal.signal_id),
        )
        if self.selection.selection == "seeded_shuffle":
            random.Random(f"{self.selection.seed}:{session.isoformat()}").shuffle(candidates)
        next_session = next((day for day in self.snapshot.calendar if day > session), None)
        selected = 0
        for signal in candidates:
            lot = self.lots[signal.signal_id]
            held = any(
                item.quantity and item.signal.symbol == signal.symbol for item in self.lots.values()
            )
            pending = any(
                order.side == "buy" and order.status == "pending" and order.symbol == signal.symbol
                for order in self.orders
            )
            bar = self._bar(signal.symbol, session)
            state = self.snapshot.state_at(signal.symbol, session)
            reason = "next_session_day"
            quantity = 0
            budget = _amount(nav * Decimal(str(self.frozen.protocol.trading.max_position_fraction)))
            if signal.symbol not in self.snapshot.manifest.symbols:
                reason = "symbol_outside_frozen_universe"
            elif selected >= self.selection.daily_entry_slots:
                reason = "daily_entry_slots"
            elif held and self.frozen.protocol.trading.repeat_signal_policy == "ignore_while_open":
                reason = "repeat_signal_while_open"
            elif (
                pending and self.frozen.protocol.trading.repeat_signal_policy == "ignore_while_open"
            ):
                reason = "repeat_signal_pending_entry"
            elif bar is None or bar.close is None or bar.price_basis != "raw" or state is None:
                reason = "missing_signal_close_or_state"
            elif not state.listed or state.suspended or bar.status != "trading":
                reason = "signal_session_not_tradable"
            else:
                quantity = (
                    int(
                        (budget / Decimal(str(bar.close)) / state.lot_size).to_integral_value(
                            rounding=ROUND_DOWN
                        )
                    )
                    * state.lot_size
                )
                if quantity == 0:
                    reason = "allocation_below_lot"
            accepted = reason == "next_session_day"
            reserved = (
                min(max(Decimal(0), self.cash - self._reserved()), budget)
                if accepted
                else Decimal(0)
            )
            if accepted:
                selected += 1
            self.orders.append(
                PortfolioOrder(
                    order_id="buy:" + signal.signal_id,
                    trade_id=lot.trade_id,
                    signal_id=signal.signal_id,
                    episode_id=signal.episode_id,
                    symbol=signal.symbol,
                    side="buy",
                    created_on=session,
                    scheduled_for=next_session,
                    requested_quantity=quantity,
                    filled_quantity=0,
                    frozen_cash=float(reserved),
                    cash_budget=float(budget if accepted else 0),
                    status="pending" if accepted else "rejected",
                    reason=reason,
                    attempt_count=0,
                )
            )

    def _unexitable_value(self, session: date) -> Decimal:
        value = Decimal(0)
        for lot in self.lots.values():
            if not lot.quantity:
                continue
            state = self.snapshot.state_at(lot.signal.symbol, session)
            bar = self._bar(lot.signal.symbol, session)
            unavailable = (
                bool(lot.exit_reasons)
                or lot.entry_date == session
                or state is None
                or not state.listed
                or state.suspended
                or state.limit_rule == "unknown"
                or bar is None
                or bar.status != "trading"
            )
            if state is not None and bar is not None and state.limit_rule == "bounded":
                unavailable = (
                    unavailable
                    or state.limit_down is None
                    or bar.open is None
                    or (bar.open <= state.limit_down)
                )
            if unavailable:
                value += lot.last_market_value
        return value

    def _trades(self) -> tuple[PortfolioTrade, ...]:
        return tuple(
            PortfolioTrade(
                trade_id=lot.trade_id,
                signal_id=lot.signal.signal_id,
                episode_id=lot.signal.episode_id,
                symbol=lot.signal.symbol,
                strategy=lot.signal.strategy,
                signal_date=lot.signal.signal_date,
                entry_date=lot.entry_date,
                exit_date=lot.exit_date,
                status="unfilled"
                if lot.entry_date is None
                else "open"
                if lot.quantity
                else "closed",
                entry_quantity=lot.entry_quantity,
                remaining_quantity=lot.quantity,
                entry_cash=float(lot.entry_cash),
                exit_cash=float(lot.exit_cash),
                distributions=float(lot.distributions),
                total_costs=float(lot.costs),
                net_pnl=float(lot.exit_cash + lot.distributions - lot.entry_cash)
                if lot.exit_date is not None
                else None,
                net_return=float((lot.exit_cash + lot.distributions) / lot.entry_cash - 1)
                if lot.exit_date is not None
                else None,
                unrealized_pnl=float(
                    lot.last_market_value + lot.exit_cash + lot.distributions - lot.entry_cash
                )
                if lot.entry_date is not None and lot.exit_date is None
                else 0,
                exit_reasons=tuple(lot.exit_reasons),
            )
            for lot in self.lots.values()
        )

    def run(self) -> PortfolioResult:
        previous_nav = _amount(self.frozen.protocol.trading.initial_capital)
        peak = previous_nav
        max_drawdown = Decimal(0)
        for index, session in enumerate(self.calendar):
            self.day_costs = Decimal(0)
            self.used_volume = {}
            opening_cash = self.cash
            self._apply_actions(session)
            prepared = self._prepare_opening_buys(
                session, opening_cash, previous_nav - self.day_costs
            )
            self._sell_due(session, opening_cash)
            self._buy_pending(session, index, prepared)
            positions, flags = self._mark(session)
            market = sum((_amount(position.market_value) for position in positions), Decimal(0))
            receivables = sum(
                (_amount(item.amount) for lot in self.lots.values() for item in lot.receivables),
                Decimal(0),
            )
            nav = self.cash + market + receivables
            self._issue_orders(session, nav)
            self._schedule_exits(session, index)
            reserved = self._reserved()
            if reserved > self.cash:
                raise ArithmeticError("账户冻结资金不能超过实际现金")
            if flags:
                self.reasons.append(f"{session.isoformat()}:持仓估值含陈旧或不可靠价格")
            self.daily.append(
                DailyAccount(
                    date=session,
                    cash=float(self.cash),
                    available_cash=float(self.cash - reserved),
                    frozen_cash=float(reserved),
                    market_value=float(market),
                    receivables_value=float(receivables),
                    nav=float(nav),
                    daily_net_return=float(nav / previous_nav - 1) if previous_nav else 0,
                    open_exposure=float((market + receivables) / nav) if nav else 0,
                    costs=float(self.day_costs),
                    positions=positions,
                    valuation_flags=flags,
                )
            )
            peak = max(peak, nav)
            max_drawdown = max(max_drawdown, 1 - nav / peak)
            previous_nav = nav
        capital = self.frozen.protocol.trading.initial_capital
        last = self.daily[-1]
        elapsed = (self.calendar[-1] - self.calendar[0]).days
        annualized = (last.nav / capital) ** (365.25 / elapsed) - 1 if elapsed else None
        trades = self._trades()
        pnls = [trade.net_pnl for trade in trades if trade.net_pnl is not None]
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        average_win = sum(wins) / len(wins) if wins else None
        average_loss = sum(losses) / len(losses) if losses else None
        audit = audit_dataset(self.snapshot)
        return PortfolioResult(
            protocol_id=self.frozen.protocol_id,
            data_snapshot_id=self.snapshot.snapshot_id,
            strategy_version=self.frozen.protocol.strategy_version,
            implementation_version=IMPLEMENTATION_VERSION,
            execution_policy_id=self.execution.policy_id,
            selection_policy_id=self.selection.policy_id,
            evidence_kind=self.snapshot.manifest.evidence_kind,
            data_qualified=audit.status == "qualified" and not self.reasons,
            execution_qualified=not self.execution_reasons,
            execution_reasons=tuple(dict.fromkeys(self.execution_reasons)),
            qualification_reasons=tuple(dict.fromkeys(self.reasons)),
            initial_capital=capital,
            window=self.window,
            signals=self.signals,
            daily_nav=tuple(self.daily),
            orders=tuple(self.orders),
            fills=tuple(self.fills),
            trades=trades,
            benchmark=self.benchmark,
            corporate_actions=tuple(self.action_events),
            metrics=PortfolioMetrics(
                closed_trades=len(pnls),
                winning_trades=len(wins),
                losing_trades=len(losses),
                breakeven_trades=sum(pnl == 0 for pnl in pnls),
                net_win_rate=len(wins) / len(pnls) if pnls else None,
                average_win=average_win,
                average_loss=average_loss,
                payoff_ratio=average_win / abs(average_loss)
                if average_win is not None and average_loss is not None
                else None,
                net_expectancy=sum(pnls) / len(pnls) if pnls else None,
                portfolio_net_return=last.nav / capital - 1,
                annualized_net_return=annualized,
                max_drawdown=float(max_drawdown),
                total_costs=sum(day.costs for day in self.daily),
                benchmark_net_return=self.benchmark[-1].nav / capital - 1
                if self.benchmark
                else None,
                turnover=sum(fill.gross for fill in self.fills)
                / (sum(day.nav for day in self.daily) / len(self.daily)),
                open_positions=len(
                    {lot.signal.symbol for lot in self.lots.values() if lot.quantity}
                ),
                open_exposure=last.open_exposure,
                unexitable_exposure=float(self._unexitable_value(last.date)) / last.nav
                if last.nav
                else 0,
                unfinished_trades=sum(trade.status == "open" for trade in trades),
            ),
            coverage=PortfolioCoverage(
                total_sessions=len(self.calendar),
                raw_signal_count=len(self.signals),
                signal_dates=tuple(sorted({s.signal_date for s in self.signals})),
                accepted_entry_orders=sum(
                    order.side == "buy" and order.status != "rejected" for order in self.orders
                ),
                filled_entries=sum(trade.entry_date is not None for trade in trades),
                unfilled_entries=sum(trade.entry_date is None for trade in trades),
                closed_trade_count=len(pnls),
                open_lots=sum(trade.status == "open" for trade in trades),
                stale_valuation_dates=tuple(day.date for day in self.daily if day.valuation_flags),
            ),
        )
