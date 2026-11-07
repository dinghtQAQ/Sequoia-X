"""日截断信号与保守的单笔日线模拟；不访问券商或修改日常策略。"""

import json
from datetime import date
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from hashlib import sha256
from typing import Literal

import pandas as pd
from pydantic import Field, model_validator

from .dataset import DailyBar, DatasetSnapshot, SecurityState
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
    algorithm_version: Literal["daily-open-v1"] = "daily-open-v1"
    fees: tuple[FeeSchedule, ...] = Field(min_length=1)
    slippage_bps: float = Field(ge=0, lt=10000)
    max_volume_participation: float = Field(gt=0, le=1)
    volume_model: Literal["daily_volume_upper_bound"]
    entry_tif: Literal["next_session_day"] = "next_session_day"
    sale_proceeds: Literal["available_for_same_session_buys"] = (
        "available_for_same_session_buys"
    )

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


class FillAttempt(FrozenModel):
    session: date | None
    symbol: str
    side: Side
    requested_quantity: int
    quantity: int = 0
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
    """模拟开盘市价成交。日成交量仅是事后模拟上界，不声称开盘流动性可知。"""
    if side not in ("buy", "sell") or quantity <= 0 or used_volume < 0:
        raise ValueError("订单方向、数量或已使用成交量无效")
    if available_cash < 0:
        raise ValueError("可用现金不能为负")
    base = FillAttempt(
        session=bar.date if bar else None,
        symbol=bar.symbol if bar else (state.symbol if state else "unknown"),
        side=side,
        requested_quantity=quantity,
        reason="pending",
    )

    def reject(reason: str) -> FillAttempt:
        return base.model_copy(update={"reason": reason})

    if bar is None or state is None:
        return reject("missing_bar_or_state")
    if state.symbol != bar.symbol or state.effective_date != bar.date:
        return reject("state_date_or_symbol_mismatch")
    if bar.available_on > bar.date:
        return reject("execution_bar_not_available_on_session")
    if state.available_on > bar.date:
        return reject("state_not_yet_known")
    if not state.listed or state.suspended or bar.status != "trading":
        return reject("not_tradable")
    if bar.price_basis != "raw":
        return reject("execution_requires_raw_price")
    if bar.open is None or bar.high is None or bar.low is None or bar.volume is None:
        return reject("missing_execution_data")
    if bar.open <= 0 or bar.volume <= 0 or not bar.low <= bar.open <= bar.high:
        return reject("invalid_execution_data")
    if state.limit_rule == "unknown":
        return reject("unknown_price_limit_rule")
    if state.limit_rule == "bounded":
        if state.limit_up is None or state.limit_down is None:
            return reject("missing_price_limits")
        if side == "buy" and bar.open >= state.limit_up:
            return reject("open_at_limit_up")
        if side == "sell" and bar.open <= state.limit_down:
            return reject("open_at_limit_down")
    if side == "sell" and (acquired_on is None or acquired_on >= bar.date):
        return reject("t_plus_one")
    applicable = [fee for fee in policy.fees if fee.effective_from <= bar.date]
    if not applicable:
        return reject("missing_fee_schedule")
    fee = applicable[-1]
    slip = _decimal(policy.slippage_bps) / 10000
    market_price = _decimal(bar.open)
    price = _money(market_price * (1 + slip if side == "buy" else 1 - slip))
    if price <= 0 or price < _decimal(bar.low) or price > _decimal(bar.high):
        return reject("slipped_price_outside_observed_range")
    if state.limit_rule == "bounded":
        assert state.limit_up is not None and state.limit_down is not None
        if price > _decimal(state.limit_up) or price < _decimal(state.limit_down):
            return reject("slipped_price_outside_price_limits")
    capacity = int(
        (_decimal(bar.volume) * _decimal(policy.max_volume_participation)).to_integral_value(
            rounding=ROUND_DOWN
        )
    ) - used_volume
    fill_quantity = min(quantity, max(capacity, 0))
    lot_size = state.lot_size
    # 完整清仓可卖不足一手的余股；部分成交仍按整手，避免凭空拆出可售余股。
    if side == "buy" or fill_quantity < quantity:
        fill_quantity = fill_quantity // lot_size * lot_size
    if side == "buy":
        cash_cap = int((_decimal(available_cash) / price).to_integral_value(rounding=ROUND_DOWN))
        fill_quantity = min(fill_quantity, cash_cap // lot_size * lot_size)

    def amounts(shares: int) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
        gross = _money(price * shares)
        commission = _money(max(gross * _decimal(fee.commission_rate),
                                _decimal(fee.min_commission)))
        stamp = _money(gross * _decimal(fee.stamp_tax_rate)) if side == "sell" else Decimal(0)
        transfer = _money(gross * _decimal(fee.transfer_fee_rate))
        cash_delta = -gross - commission - stamp - transfer if side == "buy" else (
            gross - commission - stamp - transfer
        )
        return gross, commission, stamp, transfer, cash_delta

    while fill_quantity > 0:
        gross, commission, stamp, transfer, delta = amounts(fill_quantity)
        if _decimal(available_cash) + delta >= 0:
            return FillAttempt(
                session=base.session,
                symbol=base.symbol,
                side=side,
                requested_quantity=quantity,
                quantity=fill_quantity,
                market_price=float(market_price),
                price=float(price),
                gross=float(gross),
                commission=float(commission),
                stamp_tax=float(stamp),
                transfer_fee=float(transfer),
                slippage_cost=float(_money(abs(price - market_price) * fill_quantity)),
                cash_delta=float(delta),
                reason="daily_volume_is_simulation_upper_bound",
                status="filled" if fill_quantity == quantity else "partial",
            )
        if side == "sell":
            break
        fill_quantity -= lot_size
    return reject("insufficient_cash_or_capacity_or_lot")


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
    actions = [item for item in snapshot.actions if item.symbol == symbol
               and item.ex_date == session]
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
            pending.append(DividendReceivable(
                action_id=action_id, symbol=symbol, ex_date=session,
                pay_date=action.pay_date, amount=float(amount),
            ))
        events.append(f"corporate_action:{action_id}")
    unpaid: list[DividendReceivable] = []
    for item in pending:
        if item.pay_date <= session:
            cash += _decimal(item.amount)
            events.append(f"dividend_paid:{item.action_id}")
        else:
            unpaid.append(item)
    return ActionAdjustment(
        quantity=quantity, cash_delta=float(cash), distribution_accrual=float(accrual),
        withholding_tax=float(withholding), receivables=tuple(unpaid),
        events=tuple(events), reasons=tuple(reasons),
    )


class PositionMark(FrozenModel):
    value: float | None
    price: float | None
    as_of: date | None
    status: Literal["current", "stale", "unreliable"]
    reason: str | None


def mark_position(
    snapshot: DatasetSnapshot, symbol: str, session: date, quantity: int,
) -> PositionMark:
    """只用可核对原始市场收盘价，跨除权缺价不拿旧价乘新股数。"""
    if quantity < 0:
        raise ValueError("持仓数量不能为负")
    if not quantity:
        return PositionMark(value=0, price=None, as_of=session, status="current", reason=None)
    prices = [item for item in snapshot.bars_as_of(session) if item.symbol == symbol
              and item.price_basis == "raw" and item.close is not None
              and item.status in ("trading", "suspended")]
    if not prices:
        return PositionMark(value=None, price=None, as_of=None, status="unreliable",
                            reason="no_verified_market_close")
    last = max(prices, key=lambda item: item.date)
    if any(item.symbol == symbol and last.date < item.ex_date <= session
           for item in snapshot.actions):
        return PositionMark(value=None, price=None, as_of=last.date, status="unreliable",
                            reason="corporate_action_since_last_market_close")
    assert last.close is not None
    return PositionMark(
        value=float(_money(_decimal(last.close) * quantity)), price=last.close, as_of=last.date,
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
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
            prices = (item.indicator_open, item.indicator_high,
                      item.indicator_low, item.indicator_close)
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
    snapshot: DatasetSnapshot, *, through: date | None = None,
) -> tuple[Signal, ...]:
    """严格复用恢复版 MA 条件；不排序、不将信号日收盘价当成交价。"""
    cutoff = min(through or snapshot.manifest.end, snapshot.manifest.end, snapshot.manifest.as_of)
    signals: list[Signal] = []
    for session in snapshot.calendar:
        if session > cutoff:
            break
        for symbol in snapshot.manifest.symbols:
            frame = signal_frame(snapshot, symbol, session)
            if len(frame) < 21:
                continue
            ma5 = frame["close"].rolling(5).mean()
            ma20 = frame["close"].rolling(20).mean()
            vol_ma20 = frame["volume"].rolling(20).mean()
            if (ma5.iloc[-2] < ma20.iloc[-2] and ma5.iloc[-1] > ma20.iloc[-1]
                    and frame["volume"].iloc[-1] > vol_ma20.iloc[-1] * 1.5):
                signals.append(Signal(
                    symbol=symbol, signal_date=session, strategy="ma",
                    features={"ma5": float(ma5.iloc[-1]), "ma20": float(ma20.iloc[-1]),
                              "vol_ma20": float(vol_ma20.iloc[-1])},
                ))
    return tuple(signals)
