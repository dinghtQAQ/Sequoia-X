"""实验输入与结果契约；历史模拟、纸面交易、真实成交明确分开。"""

from datetime import datetime
from math import isclose
from typing import Literal

from pydantic import Field, model_validator

from .protocol import DateWindow, FrozenModel

EvidenceKind = Literal["historical_simulation", "paper_trading", "live_fills"]
Stage = Literal["train", "validation", "final_test"]
Purpose = Literal["exploration", "confirmation"]


class Experiment(FrozenModel):
    experiment_id: str
    protocol_id: str
    study_id: str
    strategy_version: str
    comparison_version: str
    evidence_kind: EvidenceKind
    stage: Stage
    purpose: Purpose
    used_window: DateWindow
    parameters_json: str
    started_at: datetime


class AdjustedInterval(FrozenModel):
    estimate: float
    lower: float
    upper: float

    @model_validator(mode="after")
    def ordered(self) -> "AdjustedInterval":
        if not self.lower <= self.estimate <= self.upper:
            raise ValueError("区间必须满足 lower <= estimate <= upper")
        return self


class PerformanceReport(FrozenModel):
    closed_trades: int = Field(ge=0)
    winning_trades: int = Field(ge=0)
    losing_trades: int = Field(ge=0)
    breakeven_trades: int = Field(ge=0)
    net_win_rate: AdjustedInterval | None
    average_win: float | None = Field(gt=0)
    average_loss: float | None = Field(lt=0)
    payoff_ratio: float | None = Field(gt=0)
    net_expectancy: AdjustedInterval | None
    portfolio_net_return: float = Field(ge=-1)
    annualized_net_return: float = Field(ge=-1)
    max_drawdown: float = Field(ge=0, le=1)
    open_exposure: float = Field(ge=0, le=1)
    open_positions: int = Field(ge=0)
    open_valuation: str
    total_costs: float = Field(ge=0)
    sample_coverage: str

    @model_validator(mode="after")
    def estimable_metrics(self) -> "PerformanceReport":
        if self.closed_trades != self.winning_trades + self.losing_trades + self.breakeven_trades:
            raise ValueError("完整交易总数必须等于盈利、亏损和零盈亏笔数之和")
        if self.net_win_rate and not 0 <= self.net_win_rate.lower <= self.net_win_rate.upper <= 1:
            raise ValueError("胜率区间必须在 [0, 1] 内")
        if self.closed_trades == 0 and any(
            value is not None
            for value in (
                self.net_win_rate,
                self.average_win,
                self.average_loss,
                self.payoff_ratio,
                self.net_expectancy,
            )
        ):
            raise ValueError("无完整平仓交易时，交易指标必须为 null（不可估计）")
        if (
            self.average_win is None or self.average_loss is None
        ) and self.payoff_ratio is not None:
            raise ValueError("缺少盈利组或亏损组时盈亏比必须为 null")
        if (self.winning_trades > 0) != (self.average_win is not None):
            raise ValueError("有盈利交易时必须报告平均盈利；无盈利组时必须为 null")
        if (self.losing_trades > 0) != (self.average_loss is not None):
            raise ValueError("有亏损交易时必须报告平均亏损；无亏损组时必须为 null")
        if self.average_win is not None and self.average_loss is not None:
            if self.payoff_ratio is None or not isclose(
                self.payoff_ratio,
                self.average_win / abs(self.average_loss),
                abs_tol=1e-8,
            ):
                raise ValueError("盈亏比缺失或与平均盈利/平均亏损不一致")
        if self.closed_trades:
            if self.net_win_rate is not None and not isclose(
                self.net_win_rate.estimate,
                self.winning_trades / self.closed_trades,
                abs_tol=1e-8,
            ):
                raise ValueError("净胜率必须等于盈利笔数除以完整交易数；零盈亏不算胜利")
            total_pnl = (self.average_win or 0) * self.winning_trades + (
                self.average_loss or 0
            ) * self.losing_trades
            if self.net_expectancy is not None and not isclose(
                self.net_expectancy.estimate,
                total_pnl / self.closed_trades,
                abs_tol=1e-8,
            ):
                raise ValueError("每笔净期望与完整交易的净盈亏均值不一致")
        return self


class EvaluationEvidence(FrozenModel):
    experiment_id: str
    protocol_id: str
    evidence_kind: EvidenceKind
    evaluation_window: DateWindow
    data_snapshot_id: str
    candidate_version: str
    comparison_version: str
    primary_return: Literal["portfolio_net_return", "annualized_net_return"]
    candidate: PerformanceReport
    comparison: PerformanceReport
    net_return_delta: AdjustedInterval | None
    net_win_rate_delta: AdjustedInterval | None
    paired_calendar_and_capital: bool
    confidence_level: float = Field(gt=0, lt=1)
    achieved_power: float = Field(ge=0, le=1)
    dependence_method: str
    multiplicity_method: str
    family_id: str
    family_members: tuple[str, ...] = Field(min_length=1)
    report_id: str
    resampling_seed: int = Field(ge=0)


class Decision(FrozenModel):
    experiment_id: str
    protocol_id: str
    study_id: str
    evidence_kind: EvidenceKind
    status: Literal["exploratory", "insufficient_evidence", "not_adopted", "adopted"]
    active_version: str
    reasons: tuple[str, ...]
    evidence: EvaluationEvidence | None
    failure_reason: str | None
    finished_at: datetime
