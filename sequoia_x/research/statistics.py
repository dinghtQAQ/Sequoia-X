"""冻结计划驱动的收益统计；合成验证与实证资格分开报告。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import date
from math import ceil, isclose, isfinite, prod, sqrt
from pathlib import Path
from random import Random
from statistics import NormalDist, stdev
from typing import TYPE_CHECKING, Literal

from pydantic import Field, ValidationError, model_validator

from .dataset import DatasetSnapshot, audit_dataset
from .experiments import AdjustedInterval, Decision, EvaluationEvidence, PerformanceReport
from .registry import ResearchRegistry

if TYPE_CHECKING:
    from .portfolio import PortfolioResult
from .protocol import FrozenModel, ResearchProtocol


class ClosedTrade(FrozenModel):
    trade_id: str
    account: Literal["candidate", "comparison"] = "candidate"
    symbol: str
    signal_date: date
    entry_date: date
    exit_date: date
    net_pnl: float

    @model_validator(mode="after")
    def chronological(self) -> ClosedTrade:
        if not self.signal_date <= self.entry_date <= self.exit_date:
            raise ValueError("交易日期必须按信号、入场、退出排列")
        return self


class ResamplingPlan(FrozenModel):
    block_sessions: int = Field(ge=2)
    resamples: int = Field(ge=100)
    seed: int = Field(ge=0)


class DependenceSummary(FrozenModel):
    dependence_method: Literal["moving_block_signal_date_v1"] = "moving_block_signal_date_v1"
    multiplicity_method: Literal["bonferroni_family_metrics_v1"] = "bonferroni_family_metrics_v1"
    candidate_return: AdjustedInterval
    comparison_return: AdjustedInterval
    candidate_annualized_return: AdjustedInterval
    comparison_annualized_return: AdjustedInterval
    annualized_return_delta: AdjustedInterval
    return_delta: AdjustedInterval
    candidate_win_rate: AdjustedInterval | None
    comparison_win_rate: AdjustedInterval | None
    win_rate_delta: AdjustedInterval | None
    candidate_expectancy: AdjustedInterval | None
    comparison_expectancy: AdjustedInterval | None
    expectancy_delta: AdjustedInterval | None
    candidate_closed_trades: int
    comparison_closed_trades: int
    date_count: int
    independent_block_equivalents: float
    return_delta_standard_error: float | None
    annualized_delta_standard_error: float | None
    win_rate_standard_error: float | None
    win_delta_standard_error: float | None
    familywise_interval_count: int
    reasons: tuple[str, ...]


def _interval(estimate: float | None, samples: list[float], tail: float) -> AdjustedInterval | None:
    if estimate is None or not samples:
        return None
    ordered = sorted(samples)
    lower = ordered[int((len(ordered) - 1) * tail)]
    upper = ordered[min(len(ordered) - 1, int((len(ordered) - 1) * (1 - tail)) + 1)]
    return AdjustedInterval(
        estimate=estimate, lower=min(lower, estimate), upper=max(upper, estimate)
    )


def _trade_stats(
    pnl_by_day: list[list[float]], indices: list[int]
) -> tuple[float | None, float | None]:
    values = [pnl for i in indices for pnl in pnl_by_day[i]]
    return (
        (sum(value > 0 for value in values) / len(values), sum(values) / len(values))
        if values
        else (None, None)
    )


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def summarize_dependence(
    calendar: tuple[date, ...],
    candidate_returns: tuple[float, ...],
    comparison_returns: tuple[float, ...],
    candidate_trades: tuple[ClosedTrade, ...],
    comparison_trades: tuple[ClosedTrade, ...],
    plan: ResamplingPlan,
    *,
    confidence_level: float,
    family_size: int,
) -> DependenceSummary:
    """重采样连续日期块；同一信号日的交易全部一起进入每次抽样。

    条件于既有账户路径，不把抽样的行情重新执行成另一套交易。百分位区间
    依赖弱平稳、充分块长及渐近近似，不宣称有限样本精确覆盖。
    """
    n = len(calendar)
    if n < plan.block_sessions or tuple(sorted(set(calendar))) != calendar:
        raise ValueError("日历须唯一递增，长度不得少于一个完整连续块")
    if len(candidate_returns) != n or len(comparison_returns) != n:
        raise ValueError("两个账户必须包含相同完整日历，含空仓日")
    if not 0 < confidence_level < 1 or family_size < 1:
        raise ValueError("置信度或多重检验家族大小无效")
    if any(not isfinite(r) or r < -1 for r in (*candidate_returns, *comparison_returns)):
        raise ValueError("净日收益必须有限且不得低于 -100%")
    pnl_panels: list[list[list[float]]] = []
    lookup = {day: i for i, day in enumerate(calendar)}
    for trades in (candidate_trades, comparison_trades):
        if len({trade.trade_id for trade in trades}) != len(trades):
            raise ValueError("同一账户交易标识不能重复")
        panel: list[list[float]] = [[] for _ in calendar]
        for trade in trades:
            if any(
                day not in lookup for day in (trade.signal_date, trade.entry_date, trade.exit_date)
            ):
                raise ValueError("完整交易必须位于评估日历内，跨边界标签须先清除")
            panel[lookup[trade.signal_date]].append(trade.net_pnl)
        pnl_panels.append(panel)
    observed_indices = list(range(n))
    c_win, c_mean = _trade_stats(pnl_panels[0], observed_indices)
    b_win, b_mean = _trade_stats(pnl_panels[1], observed_indices)
    c_return = prod(1 + r for r in candidate_returns) - 1
    b_return = prod(1 + r for r in comparison_returns) - 1
    years = ((calendar[-1] - calendar[0]).days + 1) / 365.2425

    def annualized(value: float) -> float:
        return (1 + value) ** (1 / years) - 1

    def measurements(
        candidate_return: float,
        comparison_return: float,
        candidate_win: float | None,
        comparison_win: float | None,
        candidate_mean: float | None,
        comparison_mean: float | None,
    ) -> dict[str, float | None]:
        candidate_annual = annualized(candidate_return)
        comparison_annual = annualized(comparison_return)
        return {
            "candidate_return": candidate_return,
            "comparison_return": comparison_return,
            "return_delta": candidate_return - comparison_return,
            "candidate_win_rate": candidate_win,
            "comparison_win_rate": comparison_win,
            "win_rate_delta": _difference(candidate_win, comparison_win),
            "candidate_expectancy": candidate_mean,
            "comparison_expectancy": comparison_mean,
            "expectancy_delta": _difference(candidate_mean, comparison_mean),
            "candidate_annualized_return": candidate_annual,
            "comparison_annualized_return": comparison_annual,
            "annualized_return_delta": candidate_annual - comparison_annual,
        }

    observed = measurements(c_return, b_return, c_win, b_win, c_mean, b_mean)
    samples: dict[str, list[float]] = {name: [] for name in observed}
    rng = Random(plan.seed)
    for _ in range(plan.resamples):
        indices: list[int] = []
        while len(indices) < n:
            start = rng.randrange(n - plan.block_sessions + 1)
            indices.extend(range(start, start + plan.block_sessions))
        indices = indices[:n]
        cr = prod(1 + candidate_returns[i] for i in indices) - 1
        br = prod(1 + comparison_returns[i] for i in indices) - 1
        cw, cm = _trade_stats(pnl_panels[0], indices)
        bw, bm = _trade_stats(pnl_panels[1], indices)
        values = measurements(cr, br, cw, bw, cm, bm)
        for name, value in values.items():
            if value is not None:
                samples[name].append(value)
    interval_count = family_size * len(observed)
    tail = (1 - confidence_level) / (2 * interval_count)
    reasons = []
    longest_span = max(
        (
            lookup[t.exit_date] - lookup[t.signal_date] + 1
            for t in (*candidate_trades, *comparison_trades)
        ),
        default=0,
    )
    if plan.block_sessions < longest_span:
        reasons.append("日期块短于实际信号至退出持有跨度，不能忽略持仓重叠")
    if n / plan.block_sessions < 2:
        reasons.append("少于两个完整日期块，无法估计块间不确定性")
    if plan.resamples * tail < 2:
        reasons.append("重采样次数不足以分辨校正后区间尾部")
    if any(
        value is not None and len(samples[name]) != plan.resamples
        for name, value in observed.items()
    ):
        reasons.append("部分重采样无完整交易，交易区间不可用于确认性验收")
    intervals = {name: _interval(value, samples[name], tail) for name, value in observed.items()}
    errors = {
        name: stdev(samples[name]) if len(samples[name]) > 1 else None
        for name in (
            "return_delta",
            "candidate_win_rate",
            "win_rate_delta",
            "annualized_return_delta",
        )
    }
    return DependenceSummary.model_validate(
        {
            **intervals,
            "candidate_closed_trades": len(candidate_trades),
            "comparison_closed_trades": len(comparison_trades),
            "date_count": n,
            "independent_block_equivalents": n / plan.block_sessions,
            "return_delta_standard_error": errors["return_delta"],
            "win_rate_standard_error": errors["candidate_win_rate"],
            "win_delta_standard_error": errors["win_rate_delta"],
            "annualized_delta_standard_error": errors["annualized_return_delta"],
            "familywise_interval_count": interval_count,
            "reasons": tuple(reasons),
        }
    )


class ExcludedLabel(FrozenModel):
    trade_id: str
    account: Literal["candidate", "comparison"]
    reason: str


class MaturityPartition(FrozenModel):
    train: tuple[ClosedTrade, ...]
    validation: tuple[ClosedTrade, ...]
    final_test: tuple[ClosedTrade, ...]
    excluded: tuple[ExcludedLabel, ...]


def partition_mature_trades(
    trades: tuple[ClosedTrade, ...],
    protocol: ResearchProtocol,
    calendar: tuple[date, ...],
    *,
    as_of: date,
    embargo_sessions: int,
) -> MaturityPartition:
    """仅保留信号至实际平仓均在同一时间段内且当时已成熟的标签。"""
    if embargo_sessions < 0 or tuple(sorted(set(calendar))) != calendar:
        raise ValueError("隔离交易日数须非负，交易日历须唯一递增")
    groups: dict[str, list[ClosedTrade]] = {
        stage: [] for stage in ("train", "validation", "final_test")
    }
    excluded = []
    for trade in trades:
        stage = next(
            (
                name
                for name in groups
                if getattr(protocol, name).start <= trade.signal_date <= getattr(protocol, name).end
            ),
            None,
        )
        reason = None
        if stage is None:
            reason = "信号不属于任何冻结时间区间"
        elif trade.exit_date > getattr(protocol, stage).end:
            reason = "收益标签跨越冻结区间边界，已清除"
        elif trade.exit_date > as_of:
            reason = "决策时尚未成熟，不可用于滚动训练"
        else:
            stage_days = [
                d
                for d in calendar
                if getattr(protocol, stage).start <= d <= getattr(protocol, stage).end
            ]
            if (
                trade.signal_date not in stage_days
                or stage_days.index(trade.signal_date) < embargo_sessions
            ):
                reason = "信号在区间起点的预定隔离期内或缺少日历"
        if reason is None and stage is not None:
            groups[stage].append(trade)
        else:
            excluded.append(
                ExcludedLabel(
                    trade_id=trade.trade_id, account=trade.account, reason=reason or "标签不可用"
                )
            )
    return MaturityPartition(
        train=tuple(groups["train"]),
        validation=tuple(groups["validation"]),
        final_test=tuple(groups["final_test"]),
        excluded=tuple(excluded),
    )


DEPENDENCE_METHOD = "moving_block_signal_date_v1"
MULTIPLICITY_METHOD = "bonferroni_family_metrics_v1"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def implementation_fingerprint() -> str:
    """实验登记前取得实际测量实现指纹，不能只相信账户自报版本名。"""
    names = (
        "dataset.py",
        "replay.py",
        "portfolio.py",
        "flag.py",
        "statistics.py",
        "protocol.py",
        "experiments.py",
        "evaluation.py",
        "registry.py",
    )
    root = Path(__file__).parent
    return _digest(
        {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names
            if (root / name).exists()
        }
    )


class AccountContract(FrozenModel):
    protocol_id: str
    strategy_version: str
    implementation_version: str
    execution_policy_id: str
    selection_policy_id: str


class ScenarioContract(FrozenModel):
    scenario_id: str
    kind: Literal["cost_stress", "parameter_neighbor"]
    account: AccountContract
    description: str
    cost_multiplier: float | None = Field(default=None, gt=1)
    parameter_changes: dict[str, float | int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def declared_change(self) -> ScenarioContract:
        if self.kind == "cost_stress" and self.cost_multiplier is None:
            raise ValueError("成本压力场景须预先指定大于1的费率与滑点乘数")
        if self.kind == "parameter_neighbor" and not self.parameter_changes:
            raise ValueError("参数邻域须预先指定改变的公开参数")
        return self


class StatisticsPlan(FrozenModel):
    """须作为 statistics_plan 写入 start_experiment 的不可覆盖参数。"""

    strategy_family: Literal["ma", "flag"]
    resampling: ResamplingPlan
    embargo_sessions: int = Field(ge=0)
    dependence_method: str
    multiplicity_method: str
    assumption: Literal["weak_stationarity_block_length_covers_dependence"]
    implementation_fingerprint: str
    candidate: AccountContract
    comparison: AccountContract
    market_reference_id: str
    exposure_reference_id: str
    sector_map_id: str
    regime_map_id: str
    power_method: Literal["normal_block_standard_error_v1"]
    power_alternative_return_delta: float
    power_alternative_win_rate: float = Field(gt=0, le=1)
    power_alternative_win_rate_delta: float | None = Field(default=None, ge=-1, le=1)
    scenarios: tuple[ScenarioContract, ...]
    minimum_scenario_return_delta: float

    @model_validator(mode="after")
    def unique_scenarios(self) -> StatisticsPlan:
        if len({s.scenario_id for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("稳健性场景标识不能重复")
        return self


class ReferenceDay(FrozenModel):
    date: date
    net_return: float = Field(ge=-1)


class ReferenceSeries(FrozenModel):
    source_version: str
    description: str
    days: tuple[ReferenceDay, ...]

    @property
    def reference_id(self) -> str:
        return _digest(self.model_dump(mode="json"))


class SectorRecord(FrozenModel):
    symbol: str
    sector: str
    start: date
    end: date
    available_on: date


class SectorMap(FrozenModel):
    source_version: str
    records: tuple[SectorRecord, ...]

    @property
    def map_id(self) -> str:
        return _digest(self.model_dump(mode="json"))


class RegimePeriod(FrozenModel):
    label: str
    start: date
    end: date
    available_on: date


class RegimeMap(FrozenModel):
    source_version: str
    periods: tuple[RegimePeriod, ...]

    @property
    def map_id(self) -> str:
        return _digest(self.model_dump(mode="json"))


class PowerMetric(FrozenModel):
    metric: str
    alternative: float | None
    threshold: float | None
    standard_error: float | None
    estimated_power: float | None
    required_independent_block_equivalents: int | None


class PowerSummary(FrozenModel):
    method: Literal["normal_block_standard_error_v1"] = "normal_block_standard_error_v1"
    scope: Literal["joint_return_and_win_rate_gates"] = "joint_return_and_win_rate_gates"
    metrics: tuple[PowerMetric, ...]
    achieved_power: float | None
    required_independent_block_equivalents: int | None
    reasons: tuple[str, ...]
    limitation: str = (
        "以弱平稳与块间近似独立为假设的正态规划近似；所需块是固定指标、固定效应量下"
        "的独立信息等价量，不等于交易笔数、运行月份或未来功效保证。"
    )


def assess_power(
    summary: DependenceSummary, protocol: ResearchProtocol, plan: StatisticsPlan
) -> PowerSummary:
    """按预注册备择与冻结通过门槛之间的间距估计功效，不用观察到的大收益代替效应量。"""
    stats, gate = protocol.statistics, protocol.acceptance
    reasons = []
    annual = gate.primary_return == "annualized_net_return"
    tests: list[tuple[str, float | None, float | None, float | None]] = [
        (
            "return_delta",
            plan.power_alternative_return_delta,
            gate.minimum_return_delta,
            summary.annualized_delta_standard_error
            if annual
            else summary.return_delta_standard_error,
        ),
        (
            "candidate_win_rate",
            plan.power_alternative_win_rate,
            gate.minimum_win_rate,
            summary.win_rate_standard_error,
        ),
    ]
    if protocol.comparison_role != "initial_reference":
        tests.append(
            (
                "win_rate_delta",
                plan.power_alternative_win_rate_delta,
                gate.minimum_win_rate_delta,
                summary.win_delta_standard_error,
            )
        )
    metrics = []
    for label, alternative, threshold, error in tests:
        valid = (
            alternative is not None
            and threshold is not None
            and alternative > threshold
            and error is not None
            and error > 0
            and stats.power_target is not None
            and stats.confidence_level is not None
        )
        power = None
        required = None
        if valid:
            assert alternative is not None and threshold is not None and error is not None
            assert stats.power_target is not None and stats.confidence_level is not None
            alpha = (1 - stats.confidence_level) / summary.familywise_interval_count
            z = NormalDist().inv_cdf(1 - alpha / 2)
            margin = alternative - threshold
            # 只计向上通过冻结下界门槛的概率；负方向拒绝不属于成功。
            power = 1 - NormalDist().cdf(z - margin / error)
            individual_target = 1 - (1 - stats.power_target) / len(tests)
            required = max(
                1,
                ceil(
                    summary.independent_block_equivalents
                    * ((z + NormalDist().inv_cdf(individual_target)) * error / margin) ** 2
                ),
            )
            if power < stats.power_target:
                reasons.append(f"{label} 的冻结备择下统计功效不足")
        else:
            reasons.append(f"{label} 缺少有效冻结备择、门槛或可估块方差")
        metrics.append(
            PowerMetric(
                metric=label,
                alternative=alternative,
                threshold=threshold,
                standard_error=error,
                estimated_power=power,
                required_independent_block_equivalents=required,
            )
        )
    powers = [m.estimated_power for m in metrics if m.estimated_power is not None]
    requirements = [
        m.required_independent_block_equivalents
        for m in metrics
        if m.required_independent_block_equivalents is not None
    ]
    joint_lower = (
        max(0.0, 1 - sum(1 - power for power in powers)) if len(powers) == len(metrics) else None
    )
    if (
        joint_lower is not None
        and stats.power_target is not None
        and joint_lower < stats.power_target
    ):
        reasons.append("收益与胜率全部统计门槛的联合功效保守下界不足")
    return PowerSummary(
        metrics=tuple(metrics),
        achieved_power=joint_lower,
        required_independent_block_equivalents=max(requirements)
        if len(requirements) == len(metrics)
        else None,
        reasons=tuple(reasons),
    )


class AccountStatistics(FrozenModel):
    raw_trade_count: int
    raw_closed_trades: int
    purged_closed_trades: int
    closed_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    net_win_rate: float | None
    average_win: float | None
    average_loss: float | None
    payoff_ratio: float | None
    net_expectancy: float | None
    portfolio_net_return: float | None
    annualized_net_return: float | None
    max_drawdown: float | None
    open_positions: int
    open_exposure: float | None
    open_valuation: str
    total_costs: float
    unfilled_orders: int
    unfinished_trades: int
    signal_dates: tuple[date, ...]
    raw_signals: int
    independent_episodes: int
    trade_date_cluster_sizes: dict[str, int]
    lag1_daily_return_correlation: float | None
    time_concentration: dict[str, float]
    sector_concentration: dict[str, float]
    regime_concentration: dict[str, float]
    top_1_share_of_positive_pnl: float | None
    top_5_share_of_positive_pnl: float | None


class RobustnessResult(FrozenModel):
    scenario_id: str
    kind: str
    description: str
    account_hash: str | None
    portfolio_net_return: float | None
    net_win_rate: float | None
    net_expectancy: float | None
    primary_return_delta: float | None
    net_return_change_vs_candidate: float | None
    net_win_rate_change_vs_candidate: float | None
    total_cost_change_vs_candidate: float | None
    total_costs: float | None
    reasons: tuple[str, ...]


class BaselineReport(FrozenModel):
    report_id: str
    experiment_id: str
    protocol_id: str
    data_snapshot_id: str
    implementation_fingerprint: str
    candidate_account_hash: str
    comparison_account_hash: str
    strategy_family: Literal["ma", "flag"] | None
    candidate: AccountStatistics
    comparison: AccountStatistics
    maturity: MaturityPartition
    intervals: DependenceSummary | None
    power: PowerSummary | None
    market_reference_return: float | None
    exposure_reference_return: float | None
    market_excess_return: float | None
    exposure_excess_return: float | None
    robustness: tuple[RobustnessResult, ...]
    diagnostic_gaps: tuple[str, ...]
    frozen_parameters_json: str
    attempt_ids: tuple[str, ...]
    reference_ids: dict[str, str]
    decision: Decision
    b0_version: str | None
    evidence_source: Literal["synthetic", "observed"]
    limitation: str = (
        "分块百分位区间为弱平稳与足够块长下的近似。分组与相关性诊断仅作解释，"
        "未经独立确认的子组不能用于交易升级。MA与flag分别通过不证明共享资金联合策略。"
    )


def _closed_trades(
    account: PortfolioResult, role: Literal["candidate", "comparison"]
) -> tuple[ClosedTrade, ...]:
    return tuple(
        ClosedTrade(
            trade_id=t.trade_id,
            account=role,
            symbol=t.symbol,
            signal_date=t.signal_date,
            entry_date=t.entry_date,
            exit_date=t.exit_date,
            net_pnl=t.net_pnl,
        )
        for t in account.trades
        if t.status == "closed"
        and t.entry_date is not None
        and t.exit_date is not None
        and t.net_pnl is not None
    )


def _correlation(values: tuple[float, ...]) -> float | None:
    if len(values) < 3:
        return None
    left, right = values[:-1], values[1:]
    lm, rm = sum(left) / len(left), sum(right) / len(right)
    denominator = sqrt(sum((x - lm) ** 2 for x in left) * sum((y - rm) ** 2 for y in right))
    return (
        sum((x - lm) * (y - rm) for x, y in zip(left, right)) / denominator if denominator else None
    )


def _account_statistics(
    account: PortfolioResult,
    trades: tuple[ClosedTrade, ...],
    sectors: SectorMap | None,
    regimes: RegimeMap | None,
) -> tuple[AccountStatistics, tuple[str, ...]]:
    pnl = [t.net_pnl for t in trades]
    wins, losses = [v for v in pnl if v > 0], [v for v in pnl if v < 0]
    average_win = sum(wins) / len(wins) if wins else None
    average_loss = sum(losses) / len(losses) if losses else None
    groups: list[dict[str, float]] = [defaultdict(float), defaultdict(float), defaultdict(float)]
    gaps = []
    for trade in trades:
        groups[0][f"{trade.signal_date.year}-Q{(trade.signal_date.month - 1) // 3 + 1}"] += (
            trade.net_pnl
        )
        sector = (
            [
                r.sector
                for r in sectors.records
                if r.symbol == trade.symbol
                and r.start <= trade.signal_date <= r.end
                and r.available_on <= trade.signal_date
            ]
            if sectors
            else []
        )
        regime = (
            [
                r.label
                for r in regimes.periods
                if r.start <= trade.signal_date <= r.end and r.available_on <= trade.signal_date
            ]
            if regimes
            else []
        )
        if len(sector) != 1:
            gaps.append("行业时点映射缺失或不唯一")
        if len(regime) != 1:
            gaps.append("市场阶段映射缺失或不唯一")
        groups[1][sector[0] if len(sector) == 1 else "unknown"] += trade.net_pnl
        groups[2][regime[0] if len(regime) == 1 else "unknown"] += trade.net_pnl
    total_return = (
        account.daily_nav[-1].nav / account.initial_capital - 1
        if account.daily_nav and account.initial_capital > 0
        else None
    )
    years = (
        ((account.daily_nav[-1].date - account.daily_nav[0].date).days + 1) / 365.2425
        if account.daily_nav
        else None
    )
    annual = (
        (1 + total_return) ** (1 / years) - 1
        if total_return is not None and total_return >= -1 and years
        else None
    )
    peak, drawdown = account.initial_capital, 0.0
    for day in account.daily_nav:
        peak = max(peak, day.nav)
        drawdown = max(drawdown, 1 - day.nav / peak) if peak > 0 else drawdown
    flags = {flag for day in account.daily_nav for flag in day.valuation_flags}
    positives = sorted(wins, reverse=True)
    raw_closed = sum(t.status == "closed" for t in account.trades)
    result = AccountStatistics(
        raw_trade_count=len(account.trades),
        raw_closed_trades=raw_closed,
        purged_closed_trades=raw_closed - len(pnl),
        closed_trades=len(pnl),
        winning_trades=len(wins),
        losing_trades=len(losses),
        breakeven_trades=sum(v == 0 for v in pnl),
        net_win_rate=len(wins) / len(pnl) if pnl else None,
        average_win=average_win,
        average_loss=average_loss,
        payoff_ratio=average_win / abs(average_loss)
        if average_win is not None and average_loss is not None
        else None,
        net_expectancy=sum(pnl) / len(pnl) if pnl else None,
        portfolio_net_return=total_return,
        annualized_net_return=annual,
        max_drawdown=drawdown if account.daily_nav else None,
        open_positions=sum(p.quantity > 0 for p in account.daily_nav[-1].positions)
        if account.daily_nav
        else 0,
        open_exposure=account.daily_nav[-1].open_exposure if account.daily_nav else None,
        open_valuation=json.dumps(sorted(flags), ensure_ascii=False)
        if flags
        else "当日市场价格与应收权益估值",
        total_costs=sum(d.costs for d in account.daily_nav),
        unfilled_orders=sum(o.side == "buy" and o.filled_quantity == 0 for o in account.orders),
        unfinished_trades=sum(t.status == "open" for t in account.trades),
        signal_dates=tuple(sorted({s.signal_date for s in account.signals})),
        raw_signals=len(account.signals),
        independent_episodes=len({s.episode_id or s.signal_id for s in account.signals}),
        trade_date_cluster_sizes=dict(Counter(str(t.signal_date) for t in trades)),
        lag1_daily_return_correlation=_correlation(
            tuple(d.daily_net_return for d in account.daily_nav)
        ),
        time_concentration=dict(groups[0]),
        sector_concentration=dict(groups[1]),
        regime_concentration=dict(groups[2]),
        top_1_share_of_positive_pnl=positives[0] / sum(positives) if positives else None,
        top_5_share_of_positive_pnl=sum(positives[:5]) / sum(positives) if positives else None,
    )
    return result, tuple(sorted(set(gaps)))


def _account_reasons(
    account: PortfolioResult,
    contract: AccountContract | None,
    registry: ResearchRegistry,
    snapshot: DatasetSnapshot,
    window: object,
    parameters: dict,
    *,
    reproduce: bool,
) -> tuple[str, ...]:
    reasons = []
    calendar = tuple(
        d for d in snapshot.calendar if account.window.start <= d <= account.window.end
    )
    if (
        account.window != window
        or tuple(d.date for d in account.daily_nav) != calendar
        or not calendar
    ):
        reasons.append("账户窗口或完整交易日历不符")
    if (
        account.data_snapshot_id != snapshot.snapshot_id
        or account.evidence_kind != snapshot.manifest.evidence_kind
    ):
        reasons.append("账户数据快照或证据来源不符")
    if not account.data_qualified:
        reasons.extend(account.qualification_reasons or ("账户明确未取得数据资格",))
    if not account.execution_qualified:
        reasons.extend(account.execution_reasons or ("账户执行支持资格未通过",))
    if account.initial_capital <= 0:
        reasons.append("账户初始资金必须为正")
    previous_nav = account.initial_capital
    for day in account.daily_nav:
        if (
            previous_nav <= 0
            or day.nav < 0
            or min(
                day.cash,
                day.available_cash,
                day.frozen_cash,
                day.market_value,
                day.receivables_value,
            )
            < 0
            or not 0 <= day.open_exposure <= 1
            or not isclose(
                day.nav, day.cash + day.market_value + day.receivables_value, abs_tol=0.011
            )
            or not isclose(day.daily_net_return, day.nav / previous_nav - 1, abs_tol=1e-8)
            or not isclose(day.cash, day.available_cash + day.frozen_cash, abs_tol=0.011)
        ):
            reasons.append("账户净值、现金守恒或净日收益不一致")
        if day.valuation_flags:
            reasons.append("估值含陈旧价格或不可靠状态，不能给可信收益通过结论")
        previous_nav = day.nav
    for trade in account.trades:
        if trade.status == "closed" and (
            trade.entry_date is None
            or trade.exit_date is None
            or trade.entry_date <= trade.signal_date
            or trade.exit_date <= trade.entry_date
            or trade.remaining_quantity != 0
            or trade.net_pnl is None
            or trade.entry_cash <= 0
            or not isclose(
                trade.net_pnl,
                trade.exit_cash + trade.distributions - trade.entry_cash,
                abs_tol=0.011,
            )
        ):
            reasons.append("完整交易时间、数量或净现金盈亏不一致")
        if trade.status != "closed" and (trade.net_pnl is not None or trade.net_return is not None):
            reasons.append("未完成交易不能携带完整交易净盈亏标签")
    if contract is not None:
        for field in (
            "protocol_id",
            "strategy_version",
            "implementation_version",
            "execution_policy_id",
            "selection_policy_id",
        ):
            if getattr(account, field) != getattr(contract, field):
                reasons.append(f"账户与预登记契约不符：{field}")
        try:
            frozen = registry.get_protocol(contract.protocol_id)
            if (
                account.initial_capital != frozen.protocol.trading.initial_capital
                or frozen.protocol.data_snapshot_id != snapshot.snapshot_id
            ):
                reasons.append("账户资金或对应冻结协议数据不符")
            if reproduce:
                from .portfolio import PortfolioPolicy, run_portfolio
                from .replay import ExecutionPolicy

                execution = ExecutionPolicy.model_validate(
                    parameters["execution_policies"][contract.execution_policy_id]
                )
                selection = PortfolioPolicy.model_validate(
                    parameters["selection_policies"][contract.selection_policy_id]
                )
                if (
                    execution.policy_id != contract.execution_policy_id
                    or selection.policy_id != contract.selection_policy_id
                ):
                    raise ValueError("政策指纹不符")
                benchmark = {d.date: d.nav for d in account.benchmark} or None
                replayed = run_portfolio(
                    snapshot,
                    frozen,
                    execution,
                    selection,
                    signals=account.signals,
                    window=account.window,
                    benchmark=benchmark,
                )
                if replayed != account:
                    reasons.append("以实际快照和预登记数值政策重放得到不同账户账本")
        except (KeyError, ValueError, TypeError) as error:
            reasons.append(f"账户缺少可核对的冻结协议或数值政策：{error}")
    return tuple(sorted(set(reasons)))


def _reference_return(
    reference: ReferenceSeries | None, expected_id: str | None, calendar: tuple[date, ...]
) -> tuple[float | None, str | None]:
    if reference is None:
        return None, "参照数据缺失"
    if expected_id is None or reference.reference_id != expected_id:
        return None, "参照与预登记指纹不符"
    if tuple(d.date for d in reference.days) != calendar:
        return None, "参照未覆盖同一完整交易日历"
    return prod(1 + d.net_return for d in reference.days) - 1, None


def _performance(
    points: AccountStatistics, intervals: DependenceSummary, candidate: bool
) -> PerformanceReport:
    assert points.portfolio_net_return is not None and points.annualized_net_return is not None
    assert points.max_drawdown is not None and points.open_exposure is not None
    return PerformanceReport(
        closed_trades=points.closed_trades,
        winning_trades=points.winning_trades,
        losing_trades=points.losing_trades,
        breakeven_trades=points.breakeven_trades,
        net_win_rate=intervals.candidate_win_rate if candidate else intervals.comparison_win_rate,
        average_win=points.average_win,
        average_loss=points.average_loss,
        payoff_ratio=points.payoff_ratio,
        net_expectancy=intervals.candidate_expectancy
        if candidate
        else intervals.comparison_expectancy,
        portfolio_net_return=points.portfolio_net_return,
        annualized_net_return=points.annualized_net_return,
        max_drawdown=points.max_drawdown,
        open_exposure=points.open_exposure,
        open_positions=points.open_positions,
        open_valuation=points.open_valuation,
        total_costs=points.total_costs,
        sample_coverage=json.dumps(
            {
                "closed_trades": points.closed_trades,
                "signal_dates": len(points.signal_dates),
                "unfinished_trades": points.unfinished_trades,
            },
            sort_keys=True,
        ),
    )


def _scenario_change_reasons(
    contract: ScenarioContract,
    scenario: PortfolioResult,
    candidate: PortfolioResult,
    registry: ResearchRegistry,
    parameters: dict,
) -> tuple[str, ...]:
    reasons = []
    try:
        base = registry.get_protocol(candidate.protocol_id).protocol
        changed = registry.get_protocol(scenario.protocol_id).protocol
        if scenario.initial_capital != candidate.initial_capital:
            reasons.append("压力或邻域账户没有共享同一初始资金")
        if contract.kind == "cost_stress":
            from .replay import ExecutionPolicy

            original = ExecutionPolicy.model_validate(
                parameters["execution_policies"][candidate.execution_policy_id]
            )
            stressed = ExecutionPolicy.model_validate(
                parameters["execution_policies"][scenario.execution_policy_id]
            )
            if original.model_dump(exclude={"fees", "slippage_bps"}) != stressed.model_dump(
                exclude={"fees", "slippage_bps"}
            ):
                reasons.append("成本压力改变了未声明的订单、容量或资金可用政策")
            multiplier = contract.cost_multiplier
            assert multiplier is not None
            if tuple(f.effective_from for f in original.fees) != tuple(
                f.effective_from for f in stressed.fees
            ):
                reasons.append("成本压力场景改变费用日期而未保持可比")
            increased = False
            for old, new in zip(original.fees, stressed.fees):
                for field in (
                    "commission_rate",
                    "min_commission",
                    "stamp_tax_rate",
                    "transfer_fee_rate",
                ):
                    if getattr(new, field) + 1e-12 < getattr(old, field) * multiplier:
                        reasons.append("成本压力费用未达到预登记乘数")
                    increased = increased or getattr(new, field) > getattr(old, field)
            if stressed.slippage_bps + 1e-12 < original.slippage_bps * multiplier:
                reasons.append("成本压力滑点未达到预登记乘数")
            increased = increased or stressed.slippage_bps > original.slippage_bps
            if not increased:
                reasons.append("成本与滑点均未增加，不能把同一账户冒称压力场景")
            if scenario.selection_policy_id != candidate.selection_policy_id:
                reasons.append("成本压力同时改变了选择规则")
            for field in (
                "initial_capital",
                "max_position_fraction",
                "max_positions",
                "repeat_signal_policy",
                "holding_sessions",
                "holding_count",
                "exit_order_time",
            ):
                if getattr(changed.trading, field) != getattr(base.trading, field):
                    reasons.append("成本压力改变了资金、持有或重复信号约束")
            if scenario.signals != candidate.signals:
                reasons.append("成本压力改变了信号池")
        else:
            supported = {"trading.max_position_fraction", "trading.max_positions"}
            if not set(contract.parameter_changes).issubset(supported):
                reasons.append("邻域参数不受当前固定基线支持，不能声称完成未实现规则的稳定性验证")
            for key, value in contract.parameter_changes.items():
                field = key.removeprefix("trading.")
                if key in supported and (
                    getattr(changed.trading, field) != value
                    or getattr(base.trading, field) == value
                ):
                    reasons.append("实际邻域参数与预登记变化不符或根本未变化")
            declared_fields = {key.removeprefix("trading.") for key in contract.parameter_changes}
            if base.trading.model_dump(exclude=declared_fields) != changed.trading.model_dump(
                exclude=declared_fields
            ):
                reasons.append("参数邻域改变了预登记变化以外的交易政策")
            if scenario.strategy_version == candidate.strategy_version:
                reasons.append("邻域场景须使用独立登记的策略版本")
            if (
                scenario.execution_policy_id != candidate.execution_policy_id
                or scenario.selection_policy_id != candidate.selection_policy_id
            ):
                reasons.append("参数邻域同时改变了未声明执行或选择政策")
    except (KeyError, ValueError, TypeError) as error:
        reasons.append(f"无法复核预登记场景变化：{error}")
    return tuple(dict.fromkeys(reasons))


def analyze_baseline(
    registry: ResearchRegistry,
    experiment_id: str,
    candidate: PortfolioResult,
    comparison: PortfolioResult,
    *,
    snapshot: DatasetSnapshot,
    market_reference: ReferenceSeries | None = None,
    exposure_reference: ReferenceSeries | None = None,
    sector_map: SectorMap | None = None,
    regime_map: RegimeMap | None = None,
    scenarios: Mapping[str, PortfolioResult] | None = None,
) -> BaselineReport:
    """消费真实组合契约，计算并登记一次不可覆盖的实验结论。

    调用会结束已登记实验，包括失败与探索；不能看结果后复用同一最终测试。
    完整报告可用 record_report 内容寻址保存，登记簿保存相同 report_id 与门槛证据。
    """
    candidate_frozen = registry.get_protocol(candidate.protocol_id)
    runs = registry.experiments(candidate_frozen.protocol.study_id)
    run = next((r for r in runs if r.experiment_id == experiment_id), None)
    if run is None:
        raise KeyError("实验不存在于候选对应的研究登记簿")
    if any(d.experiment_id == experiment_id for d in registry.decisions(run.study_id)):
        raise ValueError("实验已有结果，不能重新计算后覆盖")
    frozen = registry.get_protocol(run.protocol_id)
    protocol = frozen.protocol
    parameters = json.loads(run.parameters_json)
    reasons = list(protocol.exploration_reasons())
    plan = None
    try:
        plan = StatisticsPlan.model_validate(parameters["statistics_plan"])
    except (KeyError, ValidationError, TypeError) as error:
        reasons.append(f"缺少完整预登记数值统计计划：{error}")
    audit = audit_dataset(snapshot)
    if audit.status != "qualified" or snapshot.manifest.evidence_kind != "observed":
        reasons.extend(audit.reasons or ("合成样例不能作为真实收益证据",))
    if protocol.data_snapshot_id != snapshot.snapshot_id:
        reasons.append("冻结协议与实际数据快照不符")
    if run.evidence_kind != "historical_simulation":
        reasons.append("历史账户不能冒充纸面交易或真实成交证据")
    if (
        candidate.protocol_id != run.protocol_id
        or candidate.strategy_version != protocol.strategy_version
    ):
        reasons.append("候选账户不是当前登记的冻结版本")
    if comparison.strategy_version != protocol.comparison_version:
        reasons.append("对照不是冻结协议指定版本")
    if (
        candidate.initial_capital != comparison.initial_capital
        or candidate.initial_capital != protocol.trading.initial_capital
    ):
        reasons.append("候选与对照未使用相同冻结初始资金")
    try:
        other = registry.get_protocol(comparison.protocol_id).protocol
        if other.trading != protocol.trading:
            reasons.append("候选与对照未使用同一资金、持仓与执行约束")
    except KeyError:
        reasons.append("对照没有可查的冻结协议")
    fingerprint = implementation_fingerprint()
    for account, contract in (
        (candidate, plan.candidate if plan else None),
        (comparison, plan.comparison if plan else None),
    ):
        reasons.extend(
            _account_reasons(
                account,
                contract,
                registry,
                snapshot,
                run.used_window,
                parameters,
                reproduce=plan is not None,
            )
        )
    if plan is not None:
        if plan.implementation_fingerprint != fingerprint:
            reasons.append("实际测量算法源码指纹与预登记版本不同")
        if (
            plan.dependence_method != DEPENDENCE_METHOD
            or protocol.statistics.dependence_method != DEPENDENCE_METHOD
            or plan.multiplicity_method != MULTIPLICITY_METHOD
            or protocol.statistics.multiplicity_method != MULTIPLICITY_METHOD
        ):
            reasons.append("冻结统计方法不受支持；不能把实际Bonferroni算法改称Holm或其他方法")
        if plan.resampling.block_sessions < protocol.trading.holding_sessions:
            reasons.append("冻结块长少于基线持有期")
        for label, mapping, expected_id in (
            ("行业", sector_map, plan.sector_map_id),
            ("市场阶段", regime_map, plan.regime_map_id),
        ):
            if mapping is None or mapping.map_id != expected_id:
                reasons.append(f"{label}映射缺失或与预登记指纹不符")
        if protocol.comparison_role == "initial_reference":
            from .replay import replay_ma_signals

            if plan.strategy_family == "flag":
                from .flag import replay_flag_signals

                expected_signals = replay_flag_signals(
                    snapshot, through=run.used_window.end
                ).signals
            else:
                expected_signals = replay_ma_signals(snapshot, through=run.used_window.end)
            expected_signals = tuple(
                s
                for s in expected_signals
                if run.used_window.start <= s.signal_date <= run.used_window.end
            )
            if candidate.signals != expected_signals:
                reasons.append("初始基线信号与恢复版逐日截断重放不一致")
    calendar = tuple(
        d for d in snapshot.calendar if run.used_window.start <= d <= run.used_window.end
    )
    try:
        candidate_trades = _closed_trades(candidate, "candidate")
        comparison_trades = _closed_trades(comparison, "comparison")
    except ValidationError as error:
        candidate_trades, comparison_trades = (), ()
        reasons.append(f"完整交易标签无效：{error}")
    maturity = partition_mature_trades(
        (*candidate_trades, *comparison_trades),
        protocol,
        snapshot.calendar,
        as_of=run.used_window.end,
        embargo_sessions=plan.embargo_sessions if plan else 0,
    )
    mature = getattr(maturity, run.stage)
    candidate_trades = tuple(t for t in mature if t.account == "candidate")
    comparison_trades = tuple(t for t in mature if t.account == "comparison")
    points, gaps = _account_statistics(candidate, candidate_trades, sector_map, regime_map)
    control, control_gaps = _account_statistics(
        comparison, comparison_trades, sector_map, regime_map
    )
    reasons.extend((*gaps, *control_gaps))
    if points.closed_trades == 0:
        reasons.append("候选没有成熟完整平仓交易，胜率和每笔净期望为NA")
    summary = None
    power = None
    if (
        plan is not None
        and protocol.statistics.confidence_level is not None
        and protocol.statistics.family_members
    ):
        try:
            summary = summarize_dependence(
                calendar,
                tuple(d.daily_net_return for d in candidate.daily_nav),
                tuple(d.daily_net_return for d in comparison.daily_nav),
                candidate_trades,
                comparison_trades,
                plan.resampling,
                confidence_level=protocol.statistics.confidence_level,
                family_size=len(protocol.statistics.family_members),
            )
            reasons.extend(summary.reasons)
            power = assess_power(summary, protocol, plan)
            reasons.extend(power.reasons)
        except (ValueError, OverflowError) as error:
            reasons.append(f"依赖调整区间无法估计：{error}")
    market, market_gap = _reference_return(
        market_reference, plan.market_reference_id if plan else None, calendar
    )
    exposure, exposure_gap = _reference_return(
        exposure_reference, plan.exposure_reference_id if plan else None, calendar
    )
    if market_gap:
        reasons.append("市场参照：" + market_gap)
    if exposure_gap:
        reasons.append("风险暴露参照：" + exposure_gap)
    robust = []
    provided_scenarios = scenarios or {}
    if plan is None or not {"cost_stress", "parameter_neighbor"}.issubset(
        {s.kind for s in plan.scenarios}
    ):
        reasons.append("缺少预登记的费用滑点压力或合理参数邻域场景")
    if plan is not None:
        if set(provided_scenarios) != {s.scenario_id for s in plan.scenarios}:
            reasons.append("场景集合与全部预登记尝试不符，不可只保留赢家")
        for contract in plan.scenarios:
            account = provided_scenarios.get(contract.scenario_id)
            scenario_reasons = []
            account_hash = None
            scenario_points = None
            delta = None
            if account is None:
                scenario_reasons.append("场景账户缺失")
            else:
                scenario_reasons.extend(
                    _scenario_change_reasons(contract, account, candidate, registry, parameters)
                )
                account_hash = _digest(account.model_dump(mode="json"))
                scenario_reasons.extend(
                    _account_reasons(
                        account,
                        contract.account,
                        registry,
                        snapshot,
                        run.used_window,
                        parameters,
                        reproduce=True,
                    )
                )
                scenario_partition = partition_mature_trades(
                    _closed_trades(account, "candidate"),
                    protocol,
                    snapshot.calendar,
                    as_of=run.used_window.end,
                    embargo_sessions=plan.embargo_sessions,
                )
                scenario_points, _ = _account_statistics(
                    account, getattr(scenario_partition, run.stage), sector_map, regime_map
                )
                primary = protocol.acceptance.primary_return
                if (
                    primary is not None
                    and getattr(scenario_points, primary) is not None
                    and getattr(control, primary) is not None
                ):
                    delta = getattr(scenario_points, primary) - getattr(control, primary)
                    if delta < plan.minimum_scenario_return_delta:
                        scenario_reasons.append("稳健性场景未达到预登记净收益增量下限")
                if (
                    contract.kind == "parameter_neighbor"
                    and account.strategy_version not in protocol.statistics.family_members
                ):
                    scenario_reasons.append("邻域策略未列入冻结多重检验家族")
            robust.append(
                RobustnessResult(
                    scenario_id=contract.scenario_id,
                    kind=contract.kind,
                    description=contract.description,
                    account_hash=account_hash,
                    portfolio_net_return=scenario_points.portfolio_net_return
                    if scenario_points
                    else None,
                    net_win_rate=scenario_points.net_win_rate if scenario_points else None,
                    net_expectancy=scenario_points.net_expectancy if scenario_points else None,
                    primary_return_delta=delta,
                    total_costs=scenario_points.total_costs if scenario_points else None,
                    net_return_change_vs_candidate=_difference(
                        scenario_points.portfolio_net_return, points.portfolio_net_return
                    )
                    if scenario_points
                    else None,
                    net_win_rate_change_vs_candidate=_difference(
                        scenario_points.net_win_rate, points.net_win_rate
                    )
                    if scenario_points
                    else None,
                    total_cost_change_vs_candidate=scenario_points.total_costs - points.total_costs
                    if scenario_points
                    else None,
                    reasons=tuple(scenario_reasons),
                )
            )
            reasons.extend(f"{contract.scenario_id}: {reason}" for reason in scenario_reasons)
    candidate_hash, comparison_hash = (
        _digest(a.model_dump(mode="json")) for a in (candidate, comparison)
    )
    report_id = _digest(
        {
            "experiment_id": experiment_id,
            "candidate": candidate_hash,
            "comparison": comparison_hash,
            "snapshot": snapshot.snapshot_id,
            "implementation": fingerprint,
            "parameters": parameters,
            "references": [
                market_reference.reference_id if market_reference else None,
                exposure_reference.reference_id if exposure_reference else None,
            ],
            "robustness": [r.model_dump(mode="json") for r in robust],
        }
    )
    evidence = None
    reasons = list(dict.fromkeys(reasons))
    if not reasons and summary is not None and plan is not None and power is not None:
        assert (
            protocol.statistics.confidence_level is not None
            and protocol.statistics.family_id is not None
        )
        assert protocol.acceptance.primary_return is not None and power.achieved_power is not None
        evidence = EvaluationEvidence(
            experiment_id=experiment_id,
            protocol_id=frozen.protocol_id,
            evidence_kind=run.evidence_kind,
            evaluation_window=run.used_window,
            data_snapshot_id=snapshot.snapshot_id,
            candidate_version=protocol.strategy_version,
            comparison_version=protocol.comparison_version,
            primary_return=protocol.acceptance.primary_return,
            candidate=_performance(points, summary, True),
            comparison=_performance(control, summary, False),
            net_return_delta=summary.annualized_return_delta
            if protocol.acceptance.primary_return == "annualized_net_return"
            else summary.return_delta,
            net_win_rate_delta=summary.win_rate_delta,
            paired_calendar_and_capital=True,
            confidence_level=protocol.statistics.confidence_level,
            achieved_power=power.achieved_power,
            dependence_method=DEPENDENCE_METHOD,
            multiplicity_method=MULTIPLICITY_METHOD,
            family_id=protocol.statistics.family_id,
            family_members=protocol.statistics.family_members,
            report_id=report_id,
            resampling_seed=plan.resampling.seed,
        )
    decision = registry.finish_experiment(
        experiment_id, evidence=evidence, failure_reason="; ".join(reasons) if reasons else None
    )
    return BaselineReport(
        report_id=report_id,
        experiment_id=experiment_id,
        protocol_id=frozen.protocol_id,
        data_snapshot_id=snapshot.snapshot_id,
        implementation_fingerprint=fingerprint,
        candidate_account_hash=candidate_hash,
        comparison_account_hash=comparison_hash,
        strategy_family=plan.strategy_family if plan else None,
        candidate=points,
        comparison=control,
        maturity=maturity,
        intervals=summary,
        power=power,
        market_reference_return=market,
        exposure_reference_return=exposure,
        market_excess_return=points.portfolio_net_return - market
        if points.portfolio_net_return is not None and market is not None
        else None,
        exposure_excess_return=points.portfolio_net_return - exposure
        if points.portfolio_net_return is not None and exposure is not None
        else None,
        robustness=tuple(robust),
        diagnostic_gaps=tuple(reasons),
        frozen_parameters_json=run.parameters_json,
        attempt_ids=tuple(r.experiment_id for r in runs),
        reference_ids={
            "market": market_reference.reference_id if market_reference else "missing",
            "exposure": exposure_reference.reference_id if exposure_reference else "missing",
        },
        decision=decision,
        b0_version=registry.latest_approved(run.study_id, run.evidence_kind),
        evidence_source=snapshot.manifest.evidence_kind,
    )


class SavedReport(FrozenModel):
    path: str
    sha256: str
    size: int


def record_report(report: BaselineReport, directory: str | Path) -> SavedReport:
    """完整JSON按内容哈希追加存档，重复同一内容幂等，不覆盖任何旧结果。"""
    payload = report.model_dump_json(indent=2).encode("utf8")
    digest = hashlib.sha256(payload).hexdigest()
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{digest}.json"
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError("报告内容寻址冲突或存档已遭修改") from None
    return SavedReport(path=str(path.resolve()), sha256=digest, size=len(payload))


def load_report(path: str | Path, sha256: str) -> BaselineReport:
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError("统计报告内容指纹不符")
    return BaselineReport.model_validate_json(payload)
