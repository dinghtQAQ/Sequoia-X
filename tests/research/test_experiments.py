"""从登记与判定接口验证上线规则，不合成交易实证。"""

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from sequoia_x.research import EvaluationEvidence, ResearchProtocol, ResearchRegistry


def test_incomplete_protocol_only_allows_recorded_exploration(
    tmp_path: Path,
    protocol_data: dict,
) -> None:
    registry = ResearchRegistry(tmp_path / "research.sqlite")
    protocol_data["acceptance"]["minimum_return_delta"] = None
    frozen = registry.freeze(ResearchProtocol.model_validate(protocol_data))

    with pytest.raises(ValueError, match="只能探索"):
        registry.start_experiment(
            frozen.protocol_id,
            parameters={"ma_window": 20},
            stage="final_test",
            purpose="confirmation",
            evidence_kind="historical_simulation",
        )
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={"ma_window": 20},
        stage="validation",
        purpose="exploration",
        evidence_kind="historical_simulation",
    )

    assert run.protocol_id == frozen.protocol_id
    assert run.parameters_json == '{"ma_window":20}'
    assert registry.experiments("ma-baseline") == (run,)


def test_complete_adjusted_evidence_adopts_candidate_and_preserves_report(
    tmp_path: Path,
    protocol_data: dict,
    evidence_data: dict,
) -> None:
    registry = ResearchRegistry(tmp_path / "research.sqlite")
    frozen = registry.freeze(ResearchProtocol.model_validate(protocol_data))
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={"ma_window": 20},
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    evidence = EvaluationEvidence(
        experiment_id=run.experiment_id,
        protocol_id=frozen.protocol_id,
        **evidence_data,
    )

    decision = registry.finish_experiment(run.experiment_id, evidence=evidence)

    assert decision.status == "adopted"
    assert decision.active_version == "ma-v1"
    assert decision.protocol_id == frozen.protocol_id
    assert decision.evidence == evidence
    assert registry.decisions("ma-baseline") == (decision,)
    assert registry.latest_approved("ma-baseline", "historical_simulation") == "ma-v1"
    assert registry.latest_approved("ma-baseline", "live_fills") is None


@pytest.mark.parametrize("change", ["new_threshold", "new_snapshot", "new_study"])
def test_used_test_data_cannot_be_relabelled_untouched(
    tmp_path: Path,
    protocol_data: dict,
    change: str,
) -> None:
    registry = ResearchRegistry(tmp_path / "research.sqlite")
    frozen = registry.freeze(ResearchProtocol.model_validate(protocol_data))
    registry.start_experiment(
        frozen.protocol_id,
        parameters={"window": 20},
        stage="final_test",
        purpose="exploration",
        evidence_kind="historical_simulation",
    )
    if change == "new_threshold":
        protocol_data["acceptance"]["minimum_return_delta"] = 0.01
    elif change == "new_snapshot":
        protocol_data["data_snapshot_id"] = "renamed-same-observations"
    else:
        protocol_data["study_id"] = "renamed-study"
    revised = registry.freeze(ResearchProtocol.model_validate(protocol_data))

    with pytest.raises(ValueError, match="已使用"):
        registry.start_experiment(
            revised.protocol_id,
            parameters={"window": 10},
            stage="final_test",
            purpose="confirmation",
            evidence_kind="historical_simulation",
        )


@pytest.mark.parametrize(
    ("path", "value", "status", "reason"),
    [
        (("net_return_delta", "lower"), 0.019, "not_adopted", "最低经济增量"),
        (("candidate", "net_win_rate", "lower"), 0.49, "not_adopted", "胜率目标"),
        (("net_win_rate_delta", "lower"), -0.02, "not_adopted", "允许变化"),
        (("candidate", "max_drawdown"), 0.21, "not_adopted", "回撤"),
        (("candidate", "open_exposure"), 0.31, "not_adopted", "未平仓暴露"),
        (("achieved_power",), 0.79, "insufficient_evidence", "功效不足"),
        (("paired_calendar_and_capital",), False, "insufficient_evidence", "配对比较"),
        (("dependence_method",), "unadjusted", "insufficient_evidence", "相关性处理"),
        (("multiplicity_method",), "none", "insufficient_evidence", "多重检验处理"),
        (("family_id",), "post-hoc-family", "insufficient_evidence", "家族标识"),
        (("family_members",), ["ma-v1"], "insufficient_evidence", "家族成员"),
        (("confidence_level",), 0.9, "insufficient_evidence", "置信水平"),
        (("data_snapshot_id",), "other-data", "insufficient_evidence", "数据快照"),
        (("candidate_version",), "ma-v2", "insufficient_evidence", "候选版本"),
        (("comparison_version",), "weaker-control", "insufficient_evidence", "对照版本"),
        (("net_return_delta", "estimate"), 0.04, "insufficient_evidence", "收益不一致"),
        (("net_win_rate_delta",), None, "insufficient_evidence", "独立的收益与胜率"),
    ],
)
def test_frozen_requirements_cannot_be_offset_by_other_good_metrics(
    tmp_path: Path,
    protocol_data: dict,
    evidence_data: dict,
    path: tuple[str, ...],
    value: object,
    status: str,
    reason: str,
) -> None:
    registry = ResearchRegistry(tmp_path / "research.sqlite")
    frozen = registry.freeze(ResearchProtocol.model_validate(protocol_data))
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={},
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    section = evidence_data
    for key in path[:-1]:
        section = section[key]
    section[path[-1]] = value
    evidence = EvaluationEvidence(
        experiment_id=run.experiment_id,
        protocol_id=frozen.protocol_id,
        **evidence_data,
    )

    decision = registry.finish_experiment(run.experiment_id, evidence=evidence)

    assert decision.status == status
    assert any(reason in item for item in decision.reasons)
    assert decision.active_version == "cash-control"
    assert registry.latest_approved("ma-baseline", "historical_simulation") is None


def test_failure_is_retained_and_cannot_be_overwritten(
    tmp_path: Path,
    protocol_data: dict,
) -> None:
    registry = ResearchRegistry(tmp_path / "research.sqlite")
    frozen = registry.freeze(ResearchProtocol.model_validate(protocol_data))
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={"window": 20},
        stage="final_test",
        purpose="confirmation",
        evidence_kind="paper_trading",
    )
    decision = registry.finish_experiment(run.experiment_id, failure_reason="数据覆盖不足")

    assert decision.status == "insufficient_evidence"
    assert decision.failure_reason == "数据覆盖不足"
    assert ResearchRegistry(tmp_path / "research.sqlite").decisions("ma-baseline") == (decision,)
    with pytest.raises(ValueError, match="不能覆盖"):
        registry.finish_experiment(run.experiment_id, failure_reason="替换失败结果")


def test_exploration_with_good_numbers_cannot_pass(
    tmp_path: Path,
    protocol_data: dict,
    evidence_data: dict,
) -> None:
    registry = ResearchRegistry(tmp_path / "research.sqlite")
    frozen = registry.freeze(ResearchProtocol.model_validate(protocol_data))
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={},
        stage="validation",
        purpose="exploration",
        evidence_kind="historical_simulation",
    )
    decision = registry.finish_experiment(
        run.experiment_id,
        evidence=EvaluationEvidence(
            experiment_id=run.experiment_id,
            protocol_id=frozen.protocol_id,
            **evidence_data,
        ),
    )
    assert decision.status == "exploratory"
    assert registry.latest_approved("ma-baseline", "historical_simulation") is None


def test_test_data_used_after_registration_also_blocks_confirmation(
    tmp_path: Path,
    protocol_data: dict,
    evidence_data: dict,
) -> None:
    registry = ResearchRegistry(tmp_path / "research.sqlite")
    frozen = registry.freeze(ResearchProtocol.model_validate(protocol_data))
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={},
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    registry.start_experiment(
        frozen.protocol_id,
        parameters={"window": 10},
        stage="final_test",
        purpose="exploration",
        evidence_kind="historical_simulation",
    )
    decision = registry.finish_experiment(
        run.experiment_id,
        evidence=EvaluationEvidence(
            experiment_id=run.experiment_id,
            protocol_id=frozen.protocol_id,
            **evidence_data,
        ),
    )
    assert decision.status == "insufficient_evidence"
    assert any("已使用" in reason for reason in decision.reasons)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_id", "another-experiment"),
        ("evidence_kind", "live_fills"),
        ("evaluation_window", {"start": "2020-01-01", "end": "2022-12-31"}),
    ],
)
def test_evidence_must_belong_to_registered_experiment(
    tmp_path: Path,
    protocol_data: dict,
    evidence_data: dict,
    field: str,
    value: object,
) -> None:
    registry = ResearchRegistry(tmp_path / "research.sqlite")
    frozen = registry.freeze(ResearchProtocol.model_validate(protocol_data))
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={},
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    evidence_data["experiment_id"] = run.experiment_id
    evidence_data[field] = value
    evidence = EvaluationEvidence(protocol_id=frozen.protocol_id, **evidence_data)

    decision = registry.finish_experiment(run.experiment_id, evidence=evidence)

    assert decision.status == "insufficient_evidence"
    assert any("实验登记" in reason for reason in decision.reasons)


def test_next_version_must_compare_with_latest_approved_and_failure_keeps_it(
    tmp_path: Path,
    protocol_data: dict,
    evidence_data: dict,
) -> None:
    registry = ResearchRegistry(tmp_path / "research.sqlite")
    first = registry.freeze(ResearchProtocol.model_validate(protocol_data))
    run = registry.start_experiment(
        first.protocol_id,
        parameters={"window": 20},
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    registry.finish_experiment(
        run.experiment_id,
        evidence=EvaluationEvidence(
            experiment_id=run.experiment_id,
            protocol_id=first.protocol_id,
            **evidence_data,
        ),
    )
    protocol_data["strategy_version"] = "ma-v2"
    protocol_data["data_cutoff"] = "2026-06-30"
    protocol_data["final_test"] = {"start": "2026-01-01", "end": "2026-06-30"}
    stale = registry.freeze(ResearchProtocol.model_validate(protocol_data))

    with pytest.raises(ValueError, match="最近通过版本"):
        registry.start_experiment(
            stale.protocol_id,
            parameters={"window": 10},
            stage="final_test",
            purpose="confirmation",
            evidence_kind="historical_simulation",
        )

    protocol_data["comparison_version"] = "ma-v1"
    second = registry.freeze(ResearchProtocol.model_validate(protocol_data))
    next_run = registry.start_experiment(
        second.protocol_id,
        parameters={"window": 10},
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    next_evidence = deepcopy(evidence_data)
    next_evidence.update(
        candidate_version="ma-v2",
        comparison_version="ma-v1",
        evaluation_window=protocol_data["final_test"],
    )
    next_evidence["net_return_delta"]["lower"] = 0.01
    decision = registry.finish_experiment(
        next_run.experiment_id,
        evidence=EvaluationEvidence(
            experiment_id=next_run.experiment_id,
            protocol_id=second.protocol_id,
            **next_evidence,
        ),
    )

    assert decision.status == "not_adopted"
    assert decision.active_version == "ma-v1"
    assert registry.latest_approved("ma-baseline", "historical_simulation") == "ma-v1"
    assert len(registry.experiments("ma-baseline")) == 2
    assert len(registry.decisions("ma-baseline")) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"average_win": None, "average_loss": None, "payoff_ratio": None},
        {"payoff_ratio": None},
        {"payoff_ratio": 10},
        {"winning_trades": 61},
        {"breakeven_trades": 1},
        {"net_win_rate": {"estimate": 0.65, "lower": 0.53, "upper": 0.67}},
        {"net_expectancy": {"estimate": 100, "lower": 10, "upper": 110}},
    ],
)
def test_inconsistent_or_missing_trade_metrics_are_rejected(
    evidence_data: dict, change: dict
) -> None:
    evidence_data["candidate"].update(change)
    with pytest.raises(ValidationError):
        EvaluationEvidence(experiment_id="run", protocol_id="protocol", **evidence_data)


def test_breakeven_trades_are_not_wins_and_missing_loss_group_is_not_estimable(
    tmp_path: Path,
    protocol_data: dict,
    evidence_data: dict,
) -> None:
    registry = ResearchRegistry(tmp_path / "research.sqlite")
    frozen = registry.freeze(ResearchProtocol.model_validate(protocol_data))
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={},
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    evidence_data["candidate"].update(
        winning_trades=60,
        losing_trades=0,
        breakeven_trades=40,
        average_loss=None,
        payoff_ratio=None,
        net_expectancy={"estimate": 108, "lower": 10, "upper": 150},
    )
    evidence = EvaluationEvidence(
        experiment_id=run.experiment_id,
        protocol_id=frozen.protocol_id,
        **evidence_data,
    )
    decision = registry.finish_experiment(run.experiment_id, evidence=evidence)

    assert decision.status == "adopted"
    assert decision.evidence is not None
    assert decision.evidence.candidate.net_win_rate is not None
    assert decision.evidence.candidate.net_win_rate.estimate == 0.6
    assert decision.evidence.candidate.average_loss is None
    assert decision.evidence.candidate.payoff_ratio is None


def test_no_closed_trades_are_insufficient_but_open_risk_is_preserved(
    tmp_path: Path,
    protocol_data: dict,
    evidence_data: dict,
) -> None:
    registry = ResearchRegistry(tmp_path / "research.sqlite")
    frozen = registry.freeze(ResearchProtocol.model_validate(protocol_data))
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={},
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    evidence_data["candidate"].update(
        closed_trades=0,
        winning_trades=0,
        losing_trades=0,
        breakeven_trades=0,
        average_win=None,
        average_loss=None,
        payoff_ratio=None,
        net_win_rate=None,
        net_expectancy=None,
    )
    evidence_data["net_win_rate_delta"] = None
    decision = registry.finish_experiment(
        run.experiment_id,
        evidence=EvaluationEvidence(
            experiment_id=run.experiment_id,
            protocol_id=frozen.protocol_id,
            **evidence_data,
        ),
    )

    assert decision.status == "insufficient_evidence"
    assert decision.evidence is not None
    assert decision.evidence.candidate.open_positions == 2
    assert decision.evidence.candidate.open_exposure == 0.05
