"""收益重放的只读数据契约；覆盖声明不是收益证明。"""

import json
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .protocol import FrozenModel


class DailyBar(FrozenModel):
    symbol: str
    date: date
    available_on: date
    open: float | None = Field(default=None, gt=0)
    high: float | None = Field(default=None, gt=0)
    low: float | None = Field(default=None, gt=0)
    close: float | None = Field(default=None, gt=0)
    volume: float | None = Field(default=None, ge=0)  # 股，不是手
    status: Literal["trading", "suspended", "missing", "anomalous"] = "trading"
    price_basis: Literal["raw", "back_adjusted", "forward_adjusted"] = "raw"
    indicator_open: float | None = Field(default=None, gt=0)
    indicator_high: float | None = Field(default=None, gt=0)
    indicator_low: float | None = Field(default=None, gt=0)
    indicator_close: float | None = Field(default=None, gt=0)
    indicator_available_on: date | None = None

    @model_validator(mode="after")
    def complete_indicator_prices(self) -> "DailyBar":
        values = (self.indicator_open, self.indicator_high, self.indicator_low, self.indicator_close)
        if any(value is not None for value in values) and any(value is None for value in values):
            raise ValueError("指标四价必须全部提供或全部省略")
        return self


class SecurityState(FrozenModel):
    """每日状态；限价是当日显式数据，不能从前日状态延续。"""

    symbol: str
    effective_date: date
    available_on: date
    listed: bool
    suspended: bool
    is_st: bool
    board: str
    lot_size: int = Field(default=100, gt=0)
    limit_rule: Literal["bounded", "unbounded", "unknown"] = "unknown"
    limit_up: float | None = Field(default=None, gt=0)
    limit_down: float | None = Field(default=None, gt=0)


class CorporateAction(FrozenModel):
    """公告可知日与除权生效日分离；现金是每原股税后实收，税另列。"""

    symbol: str
    ex_date: date
    available_on: date
    cash_per_share: float = Field(default=0, ge=0)
    withholding_tax_per_share: float = Field(default=0, ge=0)
    share_multiplier: float = Field(default=1, gt=0)
    pay_date: date | None = None

    @property
    def action_id(self) -> str:
        return "sha256:" + sha256(self.model_dump_json().encode()).hexdigest()

    @model_validator(mode="after")
    def cash_payment_date(self) -> "CorporateAction":
        if self.cash_per_share or self.withholding_tax_per_share:
            if self.pay_date is None or self.pay_date < self.ex_date:
                raise ValueError("分红必须有不早于除息日的实际入账日")
        return self


class DatasetManifest(FrozenModel):
    source: str
    source_version: str
    universe_id: str
    symbols: tuple[str, ...] = Field(min_length=1)
    start: date
    end: date
    as_of: date
    evidence_kind: Literal["synthetic", "observed"]
    universe_complete: bool = False
    states_complete: bool = False
    actions_complete: bool = False
    calendar_complete: bool = False
    price_granularity: Literal["daily"] = "daily"
    indicator_price_basis: Literal[
        "raw_as_of", "point_in_time_adjusted", "unverified_adjusted"
    ] = "raw_as_of"
    indicator_source_version: str | None = None

    @field_validator("symbols")
    @classmethod
    def ordered_symbols(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ValueError("股票池证券不能重复")
        return tuple(sorted(values))

    @model_validator(mode="after")
    def dates(self) -> "DatasetManifest":
        if not self.start <= self.end <= self.as_of:
            raise ValueError("数据日期必须满足 start <= end <= as_of")
        return self


class DatasetSnapshot(FrozenModel):
    manifest: DatasetManifest
    calendar: tuple[date, ...]
    bars: tuple[DailyBar, ...] = ()
    states: tuple[SecurityState, ...] = ()
    actions: tuple[CorporateAction, ...] = ()

    @field_validator("calendar")
    @classmethod
    def ordered_calendar(cls, values: tuple[date, ...]) -> tuple[date, ...]:
        if len(set(values)) != len(values):
            raise ValueError("交易日历不能重复")
        return tuple(sorted(values))

    @field_validator("bars")
    @classmethod
    def ordered_bars(cls, values: tuple[DailyBar, ...]) -> tuple[DailyBar, ...]:
        if len({(bar.symbol, bar.date) for bar in values}) != len(values):
            raise ValueError("同一证券交易日不能有重复行情")
        return tuple(sorted(values, key=lambda bar: (bar.date, bar.symbol)))

    @field_validator("states")
    @classmethod
    def ordered_states(cls, values: tuple[SecurityState, ...]) -> tuple[SecurityState, ...]:
        if len({(state.symbol, state.effective_date) for state in values}) != len(values):
            raise ValueError("同一证券交易日不能有重复状态")
        return tuple(sorted(values, key=lambda state: (state.effective_date, state.symbol)))

    @field_validator("actions")
    @classmethod
    def ordered_actions(cls, values: tuple[CorporateAction, ...]) -> tuple[CorporateAction, ...]:
        return tuple(sorted(values, key=lambda action: (action.ex_date, action.symbol)))

    @property
    def snapshot_id(self) -> str:
        # 模型已规范记录顺序；不含读取时间、导出路径等易变信息。
        return "sha256:" + sha256(self.model_dump_json().encode()).hexdigest()

    def bars_as_of(self, on: date) -> tuple[DailyBar, ...]:
        return tuple(bar for bar in self.bars if bar.date <= on and bar.available_on <= on)

    def state_at(self, symbol: str, on: date) -> SecurityState | None:
        return next((state for state in self.states if state.symbol == symbol
                     and state.effective_date == on and state.available_on <= on), None)

    def actions_as_of(self, on: date) -> tuple[CorporateAction, ...]:
        return tuple(action for action in self.actions if action.available_on <= on)


class DatasetAudit(FrozenModel):
    snapshot_id: str
    evidence_kind: Literal["synthetic", "observed"]
    status: Literal["qualified", "data_unqualified"]
    replay_eligible: bool
    reasons: tuple[str, ...]
    bar_counts: dict[str, int]
    missing_state_sessions: int = Field(ge=0)
    missing_bar_sessions: int = Field(ge=0)
    zero_volume_sessions: int = Field(ge=0)
    limitations: tuple[str, ...] = (
        "日线无法证明开盘撮合、排队成交或盘中路径；重放仍需保守执行政策",
        "历史全集与公司行动完整性依赖来源的显式覆盖声明，不代表已独立核实全市场",
    )


def audit_dataset(snapshot: DatasetSnapshot) -> DatasetAudit:
    manifest = snapshot.manifest
    reasons: list[str] = []
    for name, message in (
        ("universe_complete", "历史时点股票池及退市证券覆盖未核实"),
        ("states_complete", "上市、停牌、ST、板块及交易限制覆盖未核实"),
        ("actions_complete", "公司行动、股数、现金及扣税覆盖未核实"),
        ("calendar_complete", "全市场交易日历覆盖未核实"),
    ):
        if not getattr(manifest, name):
            reasons.append(message)
    sessions = tuple(day for day in snapshot.calendar if manifest.start <= day <= manifest.end)
    bars = {(bar.symbol, bar.date): bar for bar in snapshot.bars_as_of(manifest.as_of)}
    counts = {name: 0 for name in ("trading", "suspended", "missing", "anomalous")}
    missing_states = missing_bars = zero_volume = 0
    for symbol in manifest.symbols:
        for day in sessions:
            state = snapshot.state_at(symbol, day)
            if state is None:
                missing_states += 1
            bar = bars.get((symbol, day))
            if bar is not None:
                counts[bar.status] += 1
                if bar.volume == 0:
                    zero_volume += 1
            elif state is None or (state.listed and not state.suspended):
                missing_bars += 1
    scoped_bars = [bar for bar in snapshot.bars
                   if manifest.start <= bar.date <= manifest.end]
    for bar in scoped_bars:
        label = f"{bar.symbol}/{bar.date}"
        if bar.symbol not in manifest.symbols or bar.date not in sessions:
            reasons.append(f"行情不在声明股票池或交易日历内：{label}")
        if bar.price_basis != "raw":
            reasons.append(f"非原始市场价格不能用于执行和收益核算：{label}")
        if bar.status in ("missing", "anomalous"):
            reasons.append(f"行情明确为{bar.status}：{label}")
        if bar.status == "trading":
            if (bar.open is None or bar.high is None or bar.low is None
                or bar.close is None or bar.volume is None):
                reasons.append(f"交易行情缺少价格或成交量：{label}")
            elif not (bar.low <= bar.open <= bar.high and bar.low <= bar.close <= bar.high):
                reasons.append(f"行情OHLC关系异常：{label}")
        state = snapshot.state_at(bar.symbol, bar.date)
        if state is not None:
            if state.limit_rule == "unknown":
                reasons.append(f"涨跌停规则未知：{label}")
            elif state.limit_rule == "bounded" and (
                state.limit_up is None or state.limit_down is None
                or state.limit_down >= state.limit_up
            ):
                reasons.append(f"涨跌停价格缺失或异常：{label}")
            if (not state.listed or state.suspended) and bar.status == "trading" and bar.volume:
                reasons.append(f"上市/停牌状态与可交易行情冲突：{label}")
            if bar.status == "suspended" and not state.suspended:
                reasons.append(f"停牌行情与证券状态冲突：{label}")
        if manifest.indicator_price_basis == "raw_as_of":
            if bar.indicator_close is not None and (
                bar.indicator_open, bar.indicator_high, bar.indicator_low, bar.indicator_close
            ) != (bar.open, bar.high, bar.low, bar.close):
                reasons.append(f"声明原始指标口径却提供不同指标价：{label}")
        elif manifest.indicator_price_basis == "point_in_time_adjusted":
            if (bar.indicator_close is None or bar.indicator_available_on is None
                or bar.indicator_available_on > bar.date or not manifest.indicator_source_version):
                reasons.append(f"缺少当时可知的指标价格及来源：{label}")
    if manifest.indicator_price_basis == "unverified_adjusted":
        reasons.append("指标使用未核实的复权口径，不能证明当时可知")
    seen_actions: set[str] = set()
    for action in snapshot.actions:
        if action.action_id in seen_actions:
            reasons.append(f"重复公司行动可能重复计入收益：{action.symbol}/{action.ex_date}")
        seen_actions.add(action.action_id)
        if action.symbol not in manifest.symbols:
            reasons.append(f"公司行动不在声明股票池：{action.symbol}")
        if manifest.start <= action.ex_date <= manifest.end and action.available_on > action.ex_date:
            reasons.append(f"公司行动未能在生效日可知：{action.symbol}/{action.ex_date}")
    if not sessions:
        reasons.append("研究区间没有交易日历")
    if missing_states:
        reasons.append(f"缺少当时已知的逐日证券状态：{missing_states} 个证券交易日")
    if missing_bars:
        reasons.append(f"缺少应有行情：{missing_bars} 个证券交易日")
    replay_eligible = not reasons
    if manifest.evidence_kind == "synthetic":
        reasons.append("synthetic 合成样本仅验证功能，不能用作真实收益证据")
    return DatasetAudit(
        snapshot_id=snapshot.snapshot_id, evidence_kind=manifest.evidence_kind,
        status="data_unqualified" if reasons else "qualified", replay_eligible=replay_eligible,
        reasons=tuple(reasons), bar_counts=counts, missing_state_sessions=missing_states,
        missing_bar_sessions=missing_bars, zero_volume_sessions=zero_volume,
    )


def save_sample(snapshot: DatasetSnapshot, path: str | Path) -> str:
    """导出完整输入与指纹；不接触行情数据库。"""
    payload = {"snapshot_id": snapshot.snapshot_id, "snapshot": snapshot.model_dump(mode="json")}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return snapshot.snapshot_id


def load_sample(path: str | Path) -> DatasetSnapshot:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    snapshot = DatasetSnapshot.model_validate(payload["snapshot"])
    if payload["snapshot_id"] != snapshot.snapshot_id:
        raise ValueError("样本指纹不匹配；数据可能被修改")
    return snapshot
