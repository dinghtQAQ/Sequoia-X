"""冻结的研究契约。这里只定义口径和判据，不计算收益或统计区间。"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        str_min_length=1,
        allow_inf_nan=False,
    )


class DateWindow(FrozenModel):
    """首尾日期均包含在区间内。"""

    start: date
    end: date

    @model_validator(mode="after")
    def ordered(self) -> "DateWindow":
        if self.start > self.end:
            raise ValueError("区间开始日期不能晚于结束日期")
        return self

    def overlaps(self, other: "DateWindow") -> bool:
        return self.start <= other.end and other.start <= self.end


class TradingPolicy(FrozenModel):
    initial_capital: float = Field(gt=0)
    max_position_fraction: float = Field(gt=0, le=1)
    max_positions: int = Field(gt=0)
    entry_order_time: str
    fees: tuple[str, ...] = Field(min_length=1)
    slippage: str
    capacity: str
    repeat_signal_policy: Literal["ignore_while_open", "independent_signal_lots"]
    holding_sessions: Literal[60] = 60
    holding_count: Literal["actual_entry_session_is_day_1"] = "actual_entry_session_is_day_1"
    exit_order_time: Literal["after_day_60_close_next_tradable_open"] = (
        "after_day_60_close_next_tradable_open"
    )
    trade_attribution: Literal["signal_lot_aggregate_partial_fills_until_flat"] = (
        "signal_lot_aggregate_partial_fills_until_flat"
    )


class MetricDefinitions(FrozenModel):
    net_trade_pnl: Literal["exit_cash_plus_distributions_minus_entry_cash_and_all_costs"] = (
        "exit_cash_plus_distributions_minus_entry_cash_and_all_costs"
    )
    net_win_rate: Literal["closed_lots_with_net_pnl_gt_zero_divided_by_all_closed_lots"] = (
        "closed_lots_with_net_pnl_gt_zero_divided_by_all_closed_lots"
    )
    win_rate_interval: Literal["dependence_and_multiplicity_adjusted_interval"] = (
        "dependence_and_multiplicity_adjusted_interval"
    )
    average_win: Literal["mean_positive_net_pnl"] = "mean_positive_net_pnl"
    average_loss: Literal["mean_negative_net_pnl"] = "mean_negative_net_pnl"
    payoff_ratio: Literal["average_win_divided_by_abs_average_loss"] = (
        "average_win_divided_by_abs_average_loss"
    )
    net_expectancy: Literal["mean_net_pnl_of_all_closed_lots_including_zero"] = (
        "mean_net_pnl_of_all_closed_lots_including_zero"
    )
    portfolio_net_return: Literal["cash_flow_adjusted_nav_return_after_all_costs"] = (
        "cash_flow_adjusted_nav_return_after_all_costs"
    )
    annualized_net_return: Literal["compound_nav_return_annualized_by_actual_elapsed_years"] = (
        "compound_nav_return_annualized_by_actual_elapsed_years"
    )
    max_drawdown: Literal["maximum_peak_to_trough_nav_loss_fraction"] = (
        "maximum_peak_to_trough_nav_loss_fraction"
    )
    open_risk: Literal["unclosed_lots_excluded_from_win_rate_report_exposure_and_valuation"] = (
        "unclosed_lots_excluded_from_win_rate_report_exposure_and_valuation"
    )
    undefined_metrics: Literal["no_closed_lots_or_missing_win_loss_group_is_not_estimable"] = (
        "no_closed_lots_or_missing_win_loss_group_is_not_estimable"
    )


class StatisticsPolicy(FrozenModel):
    confidence_level: float | None = Field(default=None, gt=0, lt=1)
    power_target: float | None = Field(default=None, gt=0, lt=1)
    dependence_method: str | None = None
    multiplicity_method: str | None = None
    family_id: str | None = None
    family_members: tuple[str, ...] = ()
    review_schedule: str | None = None
    test_set_rule: Literal["single_final_use"] | None = None


class AcceptancePolicy(FrozenModel):
    primary_return: Literal["portfolio_net_return", "annualized_net_return"] | None = None
    minimum_return_delta: float | None = Field(default=None, gt=0)
    minimum_win_rate: float | None = Field(default=None, ge=0, le=1)
    minimum_win_rate_delta: float | None = Field(default=None, ge=-1, le=1)
    max_drawdown: float | None = Field(default=None, ge=0, le=1)
    max_open_exposure: float | None = Field(default=None, ge=0, le=1)
    claim: Literal["return_improvement", "return_and_win_rate_improvement"] = "return_improvement"
    joint_rule: Literal["all_requirements_must_pass"] = "all_requirements_must_pass"


class ResearchProtocol(FrozenModel):
    schema_version: Literal[1] = 1
    study_id: str
    hypothesis: str
    strategy_version: str
    comparison_version: str
    universe: str
    data_snapshot_id: str
    data_cutoff: date
    train: DateWindow
    validation: DateWindow
    final_test: DateWindow
    final_test_status: Literal["untouched", "used_for_selection", "unknown"] = "unknown"
    benchmark: str
    trading: TradingPolicy
    statistics: StatisticsPolicy = Field(default_factory=StatisticsPolicy)
    acceptance: AcceptancePolicy = Field(default_factory=AcceptancePolicy)
    metrics: MetricDefinitions = Field(default_factory=MetricDefinitions)

    @model_validator(mode="after")
    def chronological(self) -> "ResearchProtocol":
        if not (
            self.train.end < self.validation.start and self.validation.end < self.final_test.start
        ):
            raise ValueError("训练、验证、最终测试区间必须按时间排列且不重叠（含首尾日期）")
        if self.final_test.end > self.data_cutoff:
            raise ValueError("最终测试不能晚于数据截止日")
        if self.strategy_version == self.comparison_version:
            raise ValueError("候选策略版本必须与对照版本不同")
        return self

    def exploration_reasons(self) -> tuple[str, ...]:
        """结构有效但决策缺失的协议只能用于探索，不替用户填写门槛。"""
        reasons = []
        for section in ("statistics", "acceptance"):
            for name, value in getattr(self, section).model_dump().items():
                if value is None or value == ():
                    reasons.append(f"缺少 {section}.{name}")
        if self.strategy_version not in self.statistics.family_members:
            reasons.append("候选版本未列入预先冻结的多重检验家族")
        if self.final_test_status != "untouched":
            reasons.append("最终测试集未明确声明为未触碰；未知或已用于选型的数据只能探索")
        if self.acceptance.claim == "return_and_win_rate_improvement":
            delta = self.acceptance.minimum_win_rate_delta
            if delta is None or delta <= 0:
                reasons.append("同时改善的主张要求预先指定正的胜率增量")
        return tuple(reasons)


class FrozenProtocol(FrozenModel):
    protocol_id: str
    revision: int
    parent_id: str | None
    frozen_at: datetime
    protocol: ResearchProtocol
