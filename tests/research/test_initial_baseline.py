"""现金参考无交易时，初始 B0 与增强版的门槛语义分开。"""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from sequoia_x.research.experiments import EvaluationEvidence
from sequoia_x.research.protocol import ResearchProtocol
from sequoia_x.research.registry import ResearchRegistry


def initial_protocol(protocol_data):
    protocol_data["comparison_role"] = "initial_reference"
    protocol_data["acceptance"].pop("minimum_win_rate_delta")
    return ResearchProtocol.model_validate(protocol_data)


def cash_evidence(evidence_data, frozen, run):
    evidence_data.update(
        experiment_id=run.experiment_id,
        protocol_id=frozen.protocol_id,
        report_id="known-statistical-fixture-not-market-evidence",
        resampling_seed=42,
        evaluation_window=run.used_window.model_dump(mode="json"),
    )
    evidence_data["comparison"].update(
        closed_trades=0,
        winning_trades=0,
        losing_trades=0,
        breakeven_trades=0,
        net_win_rate=None,
        average_win=None,
        average_loss=None,
        payoff_ratio=None,
        net_expectancy=None,
        portfolio_net_return=0,
        annualized_net_return=0,
        open_exposure=0,
        open_positions=0,
    )
    evidence_data["net_win_rate_delta"] = None
    evidence_data["net_return_delta"] = {"estimate": 0.12, "lower": 0.06, "upper": 0.18}
    return EvaluationEvidence.model_validate(evidence_data)


def test_initial_cash_reference_has_na_trade_comparison_but_can_form_b0(
    tmp_path,
    protocol_data,
    evidence_data,
):
    registry = ResearchRegistry(tmp_path / "research.sqlite")
    frozen = registry.freeze(initial_protocol(protocol_data))
    assert frozen.protocol.exploration_reasons() == ()
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={},
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    decision = registry.finish_experiment(
        run.experiment_id,
        evidence=cash_evidence(evidence_data, frozen, run),
    )
    assert decision.status == "adopted"
    assert decision.active_version == "ma-v1"
    assert decision.evidence.comparison.net_win_rate is None
    assert decision.evidence.comparison.net_expectancy is None
    assert decision.evidence.net_win_rate_delta is None


def test_initial_reference_forbids_claim_of_win_rate_improvement(protocol_data):
    protocol_data["comparison_role"] = "initial_reference"
    protocol_data["acceptance"]["claim"] = "return_and_win_rate_improvement"
    with pytest.raises(ValidationError, match="初始参考"):
        ResearchProtocol.model_validate(protocol_data)


def test_enhancement_keeps_trade_delta_requirement(tmp_path, protocol_data, evidence_data):
    registry = ResearchRegistry(tmp_path / "research.sqlite")
    frozen = registry.freeze(ResearchProtocol.model_validate(protocol_data))
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={},
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    decision = registry.finish_experiment(
        run.experiment_id,
        evidence=cash_evidence(evidence_data, frozen, run),
    )
    assert decision.status == "insufficient_evidence"


def test_initial_candidate_without_closed_trades_stays_insufficient(
    tmp_path,
    protocol_data,
    evidence_data,
):
    registry = ResearchRegistry(tmp_path / "research.sqlite")
    frozen = registry.freeze(initial_protocol(protocol_data))
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={},
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    evidence = cash_evidence(evidence_data, frozen, run)
    values = evidence.model_dump(mode="json")
    values["candidate"] = values["comparison"].copy()
    values["net_return_delta"] = {"estimate": 0, "lower": 0, "upper": 0}
    decision = registry.finish_experiment(
        run.experiment_id,
        evidence=EvaluationEvidence.model_validate(values),
    )
    assert decision.status == "insufficient_evidence"


def test_existing_approval_blocks_new_and_pre_registered_initial_attempts(
    tmp_path,
    protocol_data,
    evidence_data,
):
    registry = ResearchRegistry(tmp_path / "research.sqlite")
    first_data = deepcopy(protocol_data)
    first_data["final_test"] = {"start": "2024-01-01", "end": "2024-12-31"}
    second_data = deepcopy(protocol_data)
    second_data["final_test"] = {"start": "2025-01-01", "end": "2025-12-31"}
    first = registry.freeze(initial_protocol(first_data))
    second = registry.freeze(initial_protocol(second_data))
    runs = [
        registry.start_experiment(
            f.protocol_id,
            parameters={},
            stage="final_test",
            purpose="confirmation",
            evidence_kind="historical_simulation",
        )
        for f in (first, second)
    ]
    assert (
        registry.finish_experiment(
            runs[0].experiment_id,
            evidence=cash_evidence(deepcopy(evidence_data), first, runs[0]),
        ).status
        == "adopted"
    )
    with pytest.raises(ValueError, match="初始参考"):
        registry.start_experiment(
            second.protocol_id,
            parameters={},
            stage="final_test",
            purpose="confirmation",
            evidence_kind="historical_simulation",
        )
    decision = registry.finish_experiment(
        runs[1].experiment_id,
        evidence=cash_evidence(deepcopy(evidence_data), second, runs[1]),
    )
    assert decision.status == "insufficient_evidence"
    assert decision.active_version == "ma-v1"
    assert any("初始参考" in reason for reason in decision.reasons)


def test_initial_absolute_win_rate_target_cannot_be_offset_by_return(
    tmp_path,
    protocol_data,
    evidence_data,
):
    protocol_data["acceptance"]["minimum_win_rate"] = 0.61
    registry = ResearchRegistry(tmp_path / "research.sqlite")
    frozen = registry.freeze(initial_protocol(protocol_data))
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={},
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    decision = registry.finish_experiment(
        run.experiment_id,
        evidence=cash_evidence(evidence_data, frozen, run),
    )
    assert decision.status == "not_adopted"
    assert any("胜率目标" in reason for reason in decision.reasons)


def test_cash_reference_does_not_allow_a_fabricated_win_delta(
    tmp_path,
    protocol_data,
    evidence_data,
):
    registry = ResearchRegistry(tmp_path / "research.sqlite")
    frozen = registry.freeze(initial_protocol(protocol_data))
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={},
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    values = cash_evidence(evidence_data, frozen, run).model_dump(mode="json")
    values["net_win_rate_delta"] = {"estimate": 0.5, "lower": 0.4, "upper": 0.6}
    decision = registry.finish_experiment(
        run.experiment_id,
        evidence=EvaluationEvidence.model_validate(values),
    )
    assert decision.status == "insufficient_evidence"
