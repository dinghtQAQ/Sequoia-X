"""日截断信号与保守的单笔日线模拟；不访问券商或修改日常策略。"""

import json
from datetime import date
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from hashlib import sha256
from typing import Literal

import pandas as pd
from pydantic import Field, model_validator

from .dataset import DailyBar, DatasetSnapshot, SecurityState, audit_dataset
from .protocol import FrozenModel, FrozenProtocol

Side = Literal["buy", "sell"]


def _decimal(value: float | int) -> Decimal:
    return Decimal(str(value))


def _money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


class FeeSchedule(FrozenModel):
    """按成交额的小数费率；现金单位人民币元，起始日包含当天。"""

    effective_from: date
    commission_rate: float = Field(ge=0, lt=1)
    min_commission: float = Field(ge=0)
    stamp_tax_rate: float = Field(ge=0, lt=1)
    transfer_fee_rate: float = Field(ge=0, lt=1)


class ExecutionPolicy(FrozenModel):
    algorithm_version: Literal["daily-open-limit-v1"] = "daily-open-limit-v1"
    order_type: Literal["limit_at_verified_session_bounds"] = "limit_at_verified_session_bounds"
    fees: tuple[FeeSchedule, ...] = Field(min_length=1)
    max_order_quantity: int = Field(gt=0, le=1_000_000)
    slippage_bps: float = Field(ge=0, lt=10000)
    max_volume_participation: float = Field(gt=0, le=1)
    volume_model: Literal["daily_volume_upper_bound"]
    entry_tif: Literal["next_session_day"] = "next_session_day"
    sale_proceeds: Literal["available_after_opening_auction"] = "available_after_opening_auction"
    dividend_proceeds: Literal["available_at_session_end"] = "available_at_session_end"

    @model_validator(mode="after")
    def ordered_fees(self) -> "ExecutionPolicy":
        dates = tuple(fee.effective_from for fee in self.fees)
        if dates != tuple(sorted(set(dates))):
            raise ValueError("费用生效日必须严格递增且不能重复")
        return self

    @property
    def policy_id(self) -> str:
        return sha256(self.model_dump_json().encode()).hexdigest()


def validate_execution_policy(frozen: FrozenProtocol, policy: ExecutionPolicy) -> None:
    """数值模型必须先由协议精确引用；不解释自然语言成本。"""
    if f"execution_policy:{policy.policy_id}" not in frozen.protocol.trading.fees:
        raise ValueError("冻结协议未精确绑定 execution_policy 指纹；请先登记数值执行模型再冻结")
    if frozen.protocol.trading.entry_order_time != "next_session_open_limit":
        raise ValueError("冻结协议订单时点必须为 next_session_open_limit；旧市价协议须新建版本")


class FillAttempt(FrozenModel):
    session: date | None
    symbol: str
    side: Side
    requested_quantity: int
    submitted_quantity: int = 0
    unsubmitted_quantity: int = 0
    quantity: int = 0
    order_type: Literal["limit_at_verified_session_bounds"] = "limit_at_verified_session_bounds"
    limit_price: float | None = None
    reserved_cash: float = 0
    released_cash: float = 0
    reserved_quantity: int = 0
    released_quantity: int = 0
    market_price: float | None = None
    price: float | None = None
    gross: float = 0
    commission: float = 0
    stamp_tax: float = 0
    transfer_fee: float = 0
    slippage_cost: float = 0
    cash_delta: float = 0
    reason: str
    status: Literal["unfilled", "partial", "filled"] = "unfilled"


def quote_open_fill(
    bar: DailyBar | None,
    state: SecurityState | None,
    side: Side,
    quantity: int,
    available_cash: float,
    policy: ExecutionPolicy,
    *,
    acquired_on: date | None = None,
    used_volume: int = 0,
) -> FillAttempt:
    """事前边界限价 DAY 单；先核申报冻结，再用日线作保守的事后匹配。"""
    if side not in ("buy", "sell") or not isinstance(quantity, int) or quantity <= 0:
        raise ValueError("订单方向或数量无效")
    if used_volume < 0 or available_cash < 0:
        raise ValueError("已使用成交量或可用现金不能为负")
    session = bar.date if bar else (state.effective_date if state else None)
    base = FillAttempt(
        session=session,
        symbol=bar.symbol if bar else (state.symbol if state else "unknown"),
        side=side,
        requested_quantity=quantity,
        unsubmitted_quantity=quantity,
        reason="not_submitted",
    )

    def reject(reason: str) -> FillAttempt:
        return base.model_copy(update={"reason": reason})

    if state is None or session is None:
        return reject("missing_bar_or_state")
    if state.effective_date != session or (bar is not None and state.symbol != bar.symbol):
        return reject("state_date_or_symbol_mismatch")
    if state.available_on > session:
        return reject("state_not_yet_known")
    if not state.listed or state.suspended:
        return reject("not_tradable")
    if state.is_st:
        return reject("risk_warning_rules_not_supported")
    if state.board != "SSE_MAIN":
        return reject("unsupported_or_unverified_board")
    if state.lot_size != 100:
        return reject("unsupported_quantity_unit")
    if state.limit_rule == "unknown":
        return reject("unknown_price_limit_rule")
    if state.limit_rule != "bounded":
        return reject("unsupported_unbounded_price_limit")
    if state.limit_up is None or state.limit_down is None or state.limit_down >= state.limit_up:
        return reject("missing_price_limits")
    if quantity > min(policy.max_order_quantity, 1_000_000):
        return reject("order_exceeds_verified_quantity_limit")
    if side == "sell" and (acquired_on is None or acquired_on >= session):
        return reject("t_plus_one")
    applicable = [fee for fee in policy.fees if fee.effective_from <= session]
    if not applicable:
        return reject("missing_fee_schedule")
    fee = applicable[-1]
    limit = _decimal(state.limit_up if side == "buy" else state.limit_down)
    if _money(limit) != limit:
        return reject("unsupported_limit_price_tick")

    def amounts(price: Decimal, shares: int) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
        gross = _money(price * shares)
        commission = _money(
            max(gross * _decimal(fee.commission_rate), _decimal(fee.min_commission))
        )
        stamp = _money(gross * _decimal(fee.stamp_tax_rate)) if side == "sell" else Decimal(0)
        transfer = _money(gross * _decimal(fee.transfer_fee_rate))
        delta = (
            -gross - commission - stamp - transfer
            if side == "buy"
            else (gross - commission - stamp - transfer)
        )
        return gross, commission, stamp, transfer, delta

    submitted = quantity
    reserved_cash = Decimal(0)
    if side == "buy":
        lot = state.lot_size
        cash_cap = int((_decimal(available_cash) / limit).to_integral_value(rounding=ROUND_DOWN))
        submitted = min(quantity // lot * lot, cash_cap // lot * lot)
        while submitted > 0:
            reserved_cash = -amounts(limit, submitted)[-1]
            if reserved_cash <= _decimal(available_cash):
                break
            submitted -= lot
        if submitted <= 0:
            return reject("insufficient_cash_at_limit_or_lot")
    base = base.model_copy(
        update={
            "submitted_quantity": submitted,
            "unsubmitted_quantity": quantity - submitted,
            "limit_price": float(limit),
            "reserved_cash": float(reserved_cash),
            "released_cash": float(reserved_cash),
            "reserved_quantity": submitted if side == "sell" else 0,
            "released_quantity": submitted if side == "sell" else 0,
        }
    )
    # 以上决定仅用事前状态、限价、费率和可用预算，不能倒用日线 open/volume。
    if bar is None:
        return reject("missing_bar_or_state")
    if bar.available_on > session:
        return reject("execution_bar_not_available_on_session")
    if bar.status != "trading":
        return reject("not_tradable")
    if bar.price_basis != "raw":
        return reject("execution_requires_raw_price")
    if bar.open is None or bar.high is None or bar.low is None or bar.volume is None:
        return reject("missing_execution_data")
    if bar.volume <= 0 or not bar.low <= bar.open <= bar.high:
        return reject("invalid_execution_data")
    if any(
        value is not None and not state.limit_down <= value <= state.limit_up
        for value in (bar.open, bar.high, bar.low, bar.close)
    ):
        return reject("raw_prices_outside_verified_price_limits")
    if side == "buy" and bar.open >= state.limit_up:
        return reject("open_at_limit_up")
    if side == "sell" and bar.open <= state.limit_down:
        return reject("open_at_limit_down")
    market_price = _decimal(bar.open)
    slip = _decimal(policy.slippage_bps) / 10000
    price = _money(market_price * (1 + slip if side == "buy" else 1 - slip))
    if price <= 0 or price < _decimal(bar.low) or price > _decimal(bar.high):
        return reject("slipped_price_outside_observed_range")
    if price > _decimal(state.limit_up) or price < _decimal(state.limit_down):
        return reject("slipped_price_outside_price_limits")
    capacity = (
        int(
            (_decimal(bar.volume) * _decimal(policy.max_volume_participation)).to_integral_value(
                rounding=ROUND_DOWN
            )
        )
        - used_volume
    )
    filled = min(submitted, max(capacity, 0))
    if side == "buy" or filled < submitted:
        filled = filled // state.lot_size * state.lot_size
    if filled <= 0:
        return reject("insufficient_capacity_or_lot")
    gross, commission, stamp, transfer, delta = amounts(price, filled)
    if _decimal(available_cash) + delta < 0:
        return reject("insufficient_cash_for_execution_costs")
    return FillAttempt(
        session=session,
        symbol=bar.symbol,
        side=side,
        requested_quantity=quantity,
        submitted_quantity=submitted,
        unsubmitted_quantity=quantity - submitted,
        quantity=filled,
        limit_price=float(limit),
        reserved_cash=float(reserved_cash),
        released_cash=float(reserved_cash + delta) if side == "buy" else 0,
        reserved_quantity=submitted if side == "sell" else 0,
        released_quantity=submitted - filled if side == "sell" else 0,
        market_price=float(market_price),
        price=float(price),
        gross=float(gross),
        commission=float(commission),
        stamp_tax=float(stamp),
        transfer_fee=float(transfer),
        slippage_cost=float(_money(abs(price - market_price) * filled)),
        cash_delta=float(delta),
        reason="daily_volume_is_simulation_upper_bound",
        status="filled" if filled == submitted else "partial",
    )


class DividendReceivable(FrozenModel):
    action_id: str
    symbol: str
    ex_date: date
    pay_date: date
    amount: float = Field(ge=0)


class ActionAdjustment(FrozenModel):
    quantity: int
    cash_delta: float
    distribution_accrual: float
    withholding_tax: float
    receivables: tuple[DividendReceivable, ...]
    events: tuple[str, ...]
    reasons: tuple[str, ...]


def apply_corporate_actions(
    snapshot: "DatasetSnapshot",
    symbol: str,
    session: date,
    quantity: int,
    receivables: tuple[DividendReceivable, ...] = (),
) -> ActionAdjustment:
    """每日开盘前对昨日持仓调用一次；已卖清的应收仍须持有至到账。"""
    if quantity < 0 or any(item.symbol != symbol for item in receivables):
        raise ValueError("持仓数量或分红应收的证券不符")
    if len({item.action_id for item in receivables}) != len(receivables):
        raise ValueError("分红应收事件不能重复")
    cash = accrual = withholding = Decimal(0)
    events: list[str] = []
    reasons: list[str] = []
    pending = list(receivables)
    original_quantity = quantity
    actions = [
        item for item in snapshot.actions if item.symbol == symbol and item.ex_date == session
    ]
    share_actions = [item for item in actions if item.share_multiplier != 1]
    if len(share_actions) > 1 and quantity:
        reasons.append("multiple_same_day_share_actions_require_verified_combined_factor")
    for action in actions:
        if action.available_on > session:
            if original_quantity:
                reasons.append("corporate_action_not_yet_known")
            continue
        if not original_quantity:
            continue
        action_id = action.action_id
        if action_id in {item.action_id for item in pending}:
            reasons.append("corporate_action_already_registered")
            continue
        if action.share_multiplier != 1 and len(share_actions) == 1:
            new_quantity = _decimal(original_quantity) * _decimal(action.share_multiplier)
            if new_quantity != new_quantity.to_integral_value():
                reasons.append("fractional_share_disposal_not_supported")
            else:
                quantity = int(new_quantity)
        amount = _money(_decimal(original_quantity) * _decimal(action.cash_per_share))
        accrual += amount
        withholding += _money(
            _decimal(original_quantity) * _decimal(action.withholding_tax_per_share)
        )
        if amount:
            assert action.pay_date is not None
            pending.append(
                DividendReceivable(
                    action_id=action_id,
                    symbol=symbol,
                    ex_date=session,
                    pay_date=action.pay_date,
                    amount=float(amount),
                )
            )
        events.append(f"corporate_action:{action_id}")
    unpaid: list[DividendReceivable] = []
    for item in pending:
        if item.pay_date <= session:
            cash += _decimal(item.amount)
            events.append(f"dividend_paid:{item.action_id}")
        else:
            unpaid.append(item)
    return ActionAdjustment(
        quantity=quantity,
        cash_delta=float(cash),
        distribution_accrual=float(accrual),
        withholding_tax=float(withholding),
        receivables=tuple(unpaid),
        events=tuple(events),
        reasons=tuple(reasons),
    )


class PositionMark(FrozenModel):
    value: float | None
    price: float | None
    as_of: date | None
    status: Literal["current", "stale", "unreliable"]
    reason: str | None


def mark_position(
    snapshot: DatasetSnapshot,
    symbol: str,
    session: date,
    quantity: int,
) -> PositionMark:
    """只用可核对原始市场收盘价，跨除权缺价不拿旧价乘新股数。"""
    if quantity < 0:
        raise ValueError("持仓数量不能为负")
    if not quantity:
        return PositionMark(value=0, price=None, as_of=session, status="current", reason=None)
    prices = [
        item
        for item in snapshot.bars_as_of(session)
        if item.symbol == symbol
        and item.price_basis == "raw"
        and item.close is not None
        and item.status == "trading"
        and item.volume is not None
        and item.volume > 0
    ]
    if not prices:
        return PositionMark(
            value=None,
            price=None,
            as_of=None,
            status="unreliable",
            reason="no_verified_market_close",
        )
    last = max(prices, key=lambda item: item.date)
    if any(
        item.symbol == symbol and last.date < item.ex_date <= session for item in snapshot.actions
    ):
        return PositionMark(
            value=None,
            price=None,
            as_of=last.date,
            status="unreliable",
            reason="corporate_action_since_last_market_close",
        )
    assert last.close is not None
    known = snapshot.state_at(symbol, last.date)
    if (
        known is not None
        and known.limit_rule == "bounded"
        and (
            known.limit_down is None
            or known.limit_up is None
            or not known.limit_down <= last.close <= known.limit_up
        )
    ):
        return PositionMark(
            value=None,
            price=None,
            as_of=last.date,
            status="unreliable",
            reason="market_close_outside_verified_price_limits",
        )
    return PositionMark(
        value=float(_money(_decimal(last.close) * quantity)),
        price=last.close,
        as_of=last.date,
        status="current" if last.date == session else "stale",
        reason=None if last.date == session else "last_known_market_close_not_cost",
    )


class Signal(FrozenModel):
    symbol: str
    signal_date: date
    strategy: Literal["ma", "flag"] = "ma"
    features: dict[str, float] = Field(default_factory=dict)
    episode_id: str | None = None

    @property
    def signal_id(self) -> str:
        payload = {"schema": "baseline-signal-v1", **self.model_dump(mode="json")}
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def signal_frame(snapshot: DatasetSnapshot, symbol: str, on: date) -> "pd.DataFrame":
    """当时已知的指标帧。缺失/异常切断预热；停牌与零量沿恢复版入库行为跳过。"""
    columns = ["date", "open", "high", "low", "close", "volume"]
    empty = pd.DataFrame(columns=columns)
    if on not in snapshot.calendar or symbol not in snapshot.manifest.symbols:
        return empty
    if snapshot.manifest.indicator_price_basis == "unverified_adjusted":
        return empty
    bars = {item.date: item for item in snapshot.bars_as_of(on) if item.symbol == symbol}
    rows: list[dict[str, object]] = []
    for session in snapshot.calendar:
        if session > on:
            break
        state = snapshot.state_at(symbol, session)
        item = bars.get(session)
        if state is not None and (not state.listed or state.suspended):
            if session == on:
                return empty
            continue
        if state is None or item is None or item.status in ("missing", "anomalous"):
            rows = []
            continue
        if item.status == "suspended" or item.volume is None or item.volume <= 0:
            if session == on:
                return empty
            continue
        if snapshot.manifest.indicator_price_basis == "point_in_time_adjusted":
            prices = (
                item.indicator_open,
                item.indicator_high,
                item.indicator_low,
                item.indicator_close,
            )
            known = item.indicator_available_on
            if known is None or known > on:
                rows = []
                continue
        else:
            prices = (item.open, item.high, item.low, item.close)
            if item.price_basis != "raw":
                rows = []
                continue
        if any(value is None for value in prices):
            rows = []
            continue
        rows.append(dict(zip(columns, (session, *prices, item.volume))))
    if not rows or rows[-1]["date"] != on:
        return empty
    return pd.DataFrame(rows, columns=columns)


def replay_ma_signals(
    snapshot: DatasetSnapshot,
    *,
    through: date | None = None,
) -> tuple[Signal, ...]:
    """严格复用恢复版 MA 条件；不排序、不将信号日收盘价当成交价。"""
    cutoff = min(through or snapshot.manifest.end, snapshot.manifest.end, snapshot.manifest.as_of)
    signals: list[Signal] = []
    for session in snapshot.calendar:
        if session > cutoff:
            break
        if session < snapshot.manifest.start:
            continue
        for symbol in snapshot.manifest.symbols:
            frame = signal_frame(snapshot, symbol, session)
            if len(frame) < 21:
                continue
            close = pd.Series(frame["close"], dtype=float)
            volume = pd.Series(frame["volume"], dtype=float)
            ma5 = pd.Series(close.rolling(5).mean())
            ma20 = pd.Series(close.rolling(20).mean())
            vol_ma20 = pd.Series(volume.rolling(20).mean())
            if (
                ma5.iloc[-2] < ma20.iloc[-2]
                and ma5.iloc[-1] > ma20.iloc[-1]
                and frame["volume"].iloc[-1] > vol_ma20.iloc[-1] * 1.5
            ):
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_date=session,
                        strategy="ma",
                        features={
                            "ma5": float(ma5.iloc[-1]),
                            "ma20": float(ma20.iloc[-1]),
                            "vol_ma20": float(vol_ma20.iloc[-1]),
                        },
                    )
                )
    return tuple(signals)


class ReplayOrder(FrozenModel):
    session: date | None
    side: Side
    requested_quantity: int
    quantity: int
    unsubmitted_quantity: int = 0
    limit_price: float | None = None
    reserved_cash: float = 0
    released_cash: float = 0
    reserved_quantity: int = 0
    released_quantity: int = 0
    filled_quantity: int = 0
    status: Literal["pending", "unfilled", "partial", "filled", "expired", "cancelled"]
    reason: str


class TradeDay(FrozenModel):
    session: date
    quantity: int
    cash: float
    market_value: float | None
    receivables_value: float
    nav: float | None
    valuation_status: Literal["current", "stale", "unreliable"]


class TradeReplay(FrozenModel):
    protocol_id: str
    data_snapshot_id: str
    execution_policy_id: str
    signal: Signal
    status: Literal["not_entered", "open", "closed", "exit_blocked"]
    entry_date: date | None
    exit_date: date | None
    quantity: int
    holding_sessions: int
    orders: tuple[ReplayOrder, ...]
    fills: tuple[FillAttempt, ...]
    daily: tuple[TradeDay, ...]
    events: tuple[str, ...]
    entry_cash: float
    exit_cash: float
    distributions_accrued: float
    distributions_paid: float
    withholding_tax: float
    total_costs: float
    net_pnl: float | None
    evidence_kind: Literal["synthetic", "observed"]
    quality_status: Literal["qualified", "data_unqualified"]
    quality_reasons: tuple[str, ...]


def replay_trade(
    snapshot: DatasetSnapshot,
    frozen: FrozenProtocol,
    signal: Signal,
    quantity: int,
    policy: ExecutionPolicy,
    *,
    through: date | None = None,
    cancel_entry_on: date | None = None,
) -> TradeReplay:
    """重放外部明确提供的一笔信号；MA 信号应来自 replay_ma_signals。"""
    validate_execution_policy(frozen, policy)
    if frozen.protocol.data_snapshot_id != snapshot.snapshot_id:
        raise ValueError("冻结协议与重放数据快照指纹不符")
    if snapshot.manifest.end > frozen.protocol.data_cutoff:
        raise ValueError("数据区间晚于冻结协议截止日")
    if quantity <= 0 or signal.symbol not in snapshot.manifest.symbols:
        raise ValueError("订单数量或信号证券无效")
    if signal.signal_date < snapshot.manifest.start:
        raise ValueError("预热区间的信号不能作为研究交易")
    if signal.signal_date not in snapshot.calendar:
        raise ValueError("信号日不在数据交易日历中")
    cutoff = min(through or snapshot.manifest.end, snapshot.manifest.end, snapshot.manifest.as_of)
    if cutoff < signal.signal_date:
        raise ValueError("重放截止日不能早于信号日")
    days = tuple(day for day in snapshot.calendar if signal.signal_date <= day <= cutoff)
    next_days = [day for day in snapshot.calendar if day > signal.signal_date]
    entry_session = next_days[0] if next_days else None
    if cancel_entry_on is not None and (
        cancel_entry_on < signal.signal_date
        or (entry_session is not None and cancel_entry_on > entry_session)
    ):
        raise ValueError("只支持信号确认后、入场时点前的显式撤销")
    entry_cancelled = cancel_entry_on is not None and cancel_entry_on <= cutoff
    bars = {item.date: item for item in snapshot.bars if item.symbol == signal.symbol}
    audit = audit_dataset(snapshot)
    reasons: list[str] = []
    cash = _money(_decimal(frozen.protocol.trading.initial_capital))
    initial_position_budget = _money(cash * _decimal(frozen.protocol.trading.max_position_fraction))
    remaining = 0
    entry_date: date | None = None
    exit_date: date | None = None
    entry_cash = exit_cash = accrual = paid = withholding = costs = Decimal(0)
    receivables: tuple[DividendReceivable, ...] = ()
    orders: list[ReplayOrder] = []
    if entry_cancelled:
        orders.append(
            ReplayOrder(
                session=cancel_entry_on,
                side="buy",
                requested_quantity=quantity,
                quantity=0,
                unsubmitted_quantity=quantity,
                status="cancelled",
                reason="explicit_entry_cancellation",
            )
        )
    fills: list[FillAttempt] = []
    daily: list[TradeDay] = []
    events: list[str] = []
    exit_triggered = False
    entry_index: int | None = None
    for session in days:
        adjusted = apply_corporate_actions(
            snapshot,
            signal.symbol,
            session,
            remaining,
            receivables,
        )
        remaining = adjusted.quantity
        receivables = adjusted.receivables
        cash += _decimal(adjusted.cash_delta)
        paid += _decimal(adjusted.cash_delta)
        accrual += _decimal(adjusted.distribution_accrual)
        withholding += _decimal(adjusted.withholding_tax)
        costs += _decimal(adjusted.withholding_tax)
        events.extend(adjusted.events)
        reasons.extend(adjusted.reasons)
        index = snapshot.calendar.index(session)
        side: Side | None = None
        requested = 0
        available = cash
        if session == entry_session and not entry_cancelled:
            side, requested = "buy", quantity
            available = min(cash, initial_position_budget)
        elif (
            remaining
            and entry_index is not None
            and index - entry_index >= frozen.protocol.trading.holding_sessions
        ):
            side, requested = "sell", remaining
            exit_triggered = True
        if side is not None:
            attempt = quote_open_fill(
                bars.get(session),
                snapshot.state_at(signal.symbol, session),
                side,
                requested,
                float(available),
                policy,
                acquired_on=entry_date,
            )
            orders.append(
                ReplayOrder(
                    session=session,
                    side=side,
                    requested_quantity=requested,
                    quantity=attempt.submitted_quantity,
                    unsubmitted_quantity=attempt.unsubmitted_quantity,
                    limit_price=attempt.limit_price,
                    reserved_cash=attempt.reserved_cash,
                    released_cash=attempt.released_cash,
                    reserved_quantity=attempt.reserved_quantity,
                    released_quantity=attempt.released_quantity,
                    filled_quantity=attempt.quantity,
                    status=attempt.status,
                    reason=attempt.reason,
                )
            )
            if attempt.quantity:
                fills.append(attempt)
                cash += _decimal(attempt.cash_delta)
                costs += sum(
                    (
                        _decimal(attempt.commission),
                        _decimal(attempt.stamp_tax),
                        _decimal(attempt.transfer_fee),
                        _decimal(attempt.slippage_cost),
                    )
                )
                if side == "buy":
                    remaining += attempt.quantity
                    entry_date, entry_index = session, index
                    entry_cash -= _decimal(attempt.cash_delta)
                else:
                    remaining -= attempt.quantity
                    exit_cash += _decimal(attempt.cash_delta)
                    if remaining == 0:
                        exit_date = session
            if side == "buy" and attempt.quantity < attempt.submitted_quantity:
                orders.append(
                    ReplayOrder(
                        session=session,
                        side="buy",
                        requested_quantity=attempt.submitted_quantity - attempt.quantity,
                        quantity=attempt.submitted_quantity - attempt.quantity,
                        status="expired",
                        reason="unfilled_day_order_remainder_expires",
                    )
                )
        mark = mark_position(snapshot, signal.symbol, session, remaining)
        receivable_value = sum((_decimal(item.amount) for item in receivables), Decimal(0))
        nav = cash + receivable_value + _decimal(mark.value) if mark.value is not None else None
        if mark.status == "unreliable" and remaining:
            reasons.append(mark.reason or "unreliable_valuation")
        daily.append(
            TradeDay(
                session=session,
                quantity=remaining,
                cash=float(cash),
                market_value=mark.value,
                receivables_value=float(receivable_value),
                nav=float(nav) if nav is not None else None,
                valuation_status=mark.status,
            )
        )
    if not entry_cancelled and (entry_session is None or entry_session > cutoff):
        orders.append(
            ReplayOrder(
                session=entry_session,
                side="buy",
                requested_quantity=quantity,
                quantity=0,
                unsubmitted_quantity=quantity,
                status="pending",
                reason="awaiting_next_session_open",
            )
        )
    status: Literal["not_entered", "open", "closed", "exit_blocked"]
    if entry_date is None:
        status = "not_entered"
    elif not remaining:
        status = "closed"
    else:
        status = "exit_blocked" if exit_triggered else "open"
    elapsed = 0
    if entry_index is not None and days:
        end_index = snapshot.calendar.index(exit_date or days[-1])
        elapsed = end_index - entry_index + (0 if exit_date is not None else 1)
    reliable_model = audit.replay_eligible and not reasons
    return TradeReplay(
        protocol_id=frozen.protocol_id,
        data_snapshot_id=snapshot.snapshot_id,
        execution_policy_id=policy.policy_id,
        signal=signal,
        status=status,
        entry_date=entry_date,
        exit_date=exit_date,
        quantity=remaining,
        holding_sessions=elapsed,
        orders=tuple(orders),
        fills=tuple(fills),
        daily=tuple(daily),
        events=tuple(events),
        entry_cash=float(entry_cash),
        exit_cash=float(exit_cash),
        distributions_accrued=float(accrual),
        distributions_paid=float(paid),
        withholding_tax=float(withholding),
        total_costs=float(costs),
        net_pnl=float(exit_cash + accrual - entry_cash)
        if status == "closed" and reliable_model
        else None,
        evidence_kind=snapshot.manifest.evidence_kind,
        quality_status="qualified"
        if audit.status == "qualified" and not reasons
        else "data_unqualified",
        quality_reasons=tuple(dict.fromkeys((*audit.reasons, *reasons))),
    )
