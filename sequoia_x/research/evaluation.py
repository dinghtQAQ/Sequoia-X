"""对预先计算的证据执行冻结判据，不在这里制造统计证据。"""

from math import isclose
from typing import Literal

from .experiments import EvaluationEvidence
from .protocol import ResearchProtocol


def evaluate(
    protocol: ResearchProtocol,
    protocol_id: str,
    evidence: EvaluationEvidence,
) -> tuple[Literal["insufficient_evidence", "not_adopted", "adopted"], tuple[str, ...]]:
    stats = protocol.statistics
    gate = protocol.acceptance
    reasons = list(protocol.exploration_reasons())
    for actual, expected, label in (
        (evidence.protocol_id, protocol_id, "协议标识"),
        (evidence.data_snapshot_id, protocol.data_snapshot_id, "数据快照"),
        (evidence.candidate_version, protocol.strategy_version, "候选版本"),
        (evidence.comparison_version, protocol.comparison_version, "配对对照版本"),
        (evidence.primary_return, gate.primary_return, "主要净收益指标"),
        (evidence.confidence_level, stats.confidence_level, "置信水平"),
        (evidence.dependence_method, stats.dependence_method, "相关性处理"),
        (evidence.multiplicity_method, stats.multiplicity_method, "多重检验处理"),
        (evidence.family_id, stats.family_id, "多重检验家族标识"),
        (evidence.family_members, stats.family_members, "多重检验家族成员"),
    ):
        if actual != expected:
            reasons.append(f"证据与冻结协议不符：{label}")
    if not evidence.paired_calendar_and_capital:
        reasons.append("缺少同日历、同资金约束的配对比较")
    if stats.power_target is None or evidence.achieved_power < stats.power_target:
        reasons.append("统计功效不足")
    candidate, comparison = evidence.candidate, evidence.comparison
    initial = protocol.comparison_role == "initial_reference"
    required_trade_reports = (candidate,) if initial else (candidate, comparison)
    if any(
        report.closed_trades == 0 or report.net_win_rate is None or report.net_expectancy is None
        for report in required_trade_reports
    ):
        reasons.append("完整交易的净胜率或净期望不可估计")
    return_delta, win_delta = evidence.net_return_delta, evidence.net_win_rate_delta
    if initial and comparison.closed_trades == 0 and win_delta is not None:
        reasons.append("无完整交易的初始参考不能声明胜率增量")
    if return_delta is None or (not initial and win_delta is None):
        reasons.append("缺少独立的收益与胜率增量区间")
    else:
        observed_return_delta = getattr(candidate, evidence.primary_return) - getattr(
            comparison, evidence.primary_return
        )
        if not isclose(return_delta.estimate, observed_return_delta, abs_tol=1e-8):
            reasons.append("净收益增量与两版本账户收益不一致")
        if win_delta is not None and candidate.net_win_rate and comparison.net_win_rate:
            if not isclose(
                win_delta.estimate,
                candidate.net_win_rate.estimate - comparison.net_win_rate.estimate,
                abs_tol=1e-8,
            ):
                reasons.append("净胜率增量与两版本胜率不一致")
    if reasons:
        return "insufficient_evidence", tuple(reasons)

    # 缺失值已在前面返回；每项独立检查，不用一个好结果抵消另一项失败。
    assert return_delta is not None and candidate.net_win_rate is not None
    assert gate.minimum_return_delta is not None and gate.minimum_win_rate is not None
    assert gate.max_drawdown is not None and gate.max_open_exposure is not None
    if return_delta.lower < gate.minimum_return_delta:
        reasons.append("调整后的净收益增量区间不支持最低经济增量")
    if candidate.net_win_rate.lower < gate.minimum_win_rate:
        reasons.append("调整后的净胜率区间不支持胜率目标")
    if not initial:
        assert win_delta is not None and gate.minimum_win_rate_delta is not None
        if win_delta.lower < gate.minimum_win_rate_delta:
            reasons.append("调整后的净胜率变化区间不支持允许变化")
    if candidate.max_drawdown > gate.max_drawdown:
        reasons.append("组合回撤超出风险预算")
    if candidate.open_exposure > gate.max_open_exposure:
        reasons.append("未平仓暴露超出风险预算")
    return ("not_adopted", tuple(reasons)) if reasons else ("adopted", ())
