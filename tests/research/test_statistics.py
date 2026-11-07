"""#14 公共统计边界的可手算合成样例，不代表市场证据。"""

from datetime import date, timedelta

import pytest

from sequoia_x.research.statistics import ClosedTrade, ResamplingPlan, summarize_dependence


def test_zero_effect_keeps_cash_days_and_zero_trade_metrics_na():
    days = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(120))
    result = summarize_dependence(
        days,
        (0.0,) * 120,
        (0.0,) * 120,
        (),
        (),
        ResamplingPlan(block_sessions=60, resamples=1000, seed=7),
        confidence_level=0.95,
        family_size=1,
    )
    assert result.candidate_return.estimate == 0
    assert result.return_delta.lower == result.return_delta.upper == 0
    assert result.candidate_win_rate is None
    assert result.candidate_expectancy is None
    assert result.candidate_closed_trades == 0
    assert result.date_count == 120
    assert result.independent_block_equivalents == 2


def test_hundreds_of_same_day_trades_remain_one_cluster():
    days = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(180))
    trades = tuple(
        ClosedTrade(
            trade_id=f"lot-{i}",
            symbol=f"S{i}",
            signal_date=days[90],
            entry_date=days[91],
            exit_date=days[151],
            net_pnl=100 if i % 2 else -100,
        )
        for i in range(400)
    )
    result = summarize_dependence(
        days,
        (0.0,) * 180,
        (0.0,) * 180,
        trades,
        trades,
        ResamplingPlan(block_sessions=61, resamples=1000, seed=7),
        confidence_level=0.95,
        family_size=1,
    )
    assert result.candidate_closed_trades == 400
    assert result.candidate_win_rate.estimate == 0.5
    assert result.candidate_expectancy.estimate == 0
    assert result.win_rate_delta.estimate == 0
    assert any("部分重采样无完整交易" in reason for reason in result.reasons)


@pytest.mark.parametrize("change", ["dates", "missing_cash_day", "nan", "duplicate_trade"])
def test_invalid_paired_inputs_are_rejected(change):
    days = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(120))
    left = (0.0,) * 120
    right = left
    trades = ()
    if change == "dates":
        days = tuple(reversed(days))
    elif change == "missing_cash_day":
        right = right[1:]
    elif change == "nan":
        left = (float("nan"),) + left[1:]
    else:
        trade = ClosedTrade(
            trade_id="same",
            symbol="S",
            signal_date=days[0],
            entry_date=days[1],
            exit_date=days[61],
            net_pnl=10,
        )
        trades = (trade, trade)
    with pytest.raises(ValueError):
        summarize_dependence(
            days,
            left,
            right,
            trades,
            (),
            ResamplingPlan(block_sessions=60, resamples=1000, seed=7),
            confidence_level=0.95,
            family_size=1,
        )


def test_block_shorter_than_actual_holding_span_cannot_confirm():
    days = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(180))
    trades = (
        ClosedTrade(
            trade_id="suspended",
            symbol="S",
            signal_date=days[0],
            entry_date=days[1],
            exit_date=days[150],
            net_pnl=-100,
        ),
    )
    result = summarize_dependence(
        days,
        (0.0,) * 180,
        (0.0,) * 180,
        trades,
        trades,
        ResamplingPlan(block_sessions=60, resamples=1000, seed=7),
        confidence_level=0.95,
        family_size=1,
    )
    assert any("持有跨度" in reason for reason in result.reasons)


def test_temporal_partition_purges_cross_boundary_and_immature_labels(protocol_data):
    from sequoia_x.research.protocol import ResearchProtocol
    from sequoia_x.research.statistics import partition_mature_trades

    protocol = ResearchProtocol.model_validate(protocol_data)
    days = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(2192))
    trades = (
        ClosedTrade(
            trade_id="mature",
            symbol="S",
            signal_date=date(2021, 1, 1),
            entry_date=date(2021, 1, 2),
            exit_date=date(2021, 3, 2),
            net_pnl=10,
        ),
        ClosedTrade(
            trade_id="cross",
            symbol="S",
            signal_date=date(2022, 12, 1),
            entry_date=date(2022, 12, 2),
            exit_date=date(2023, 2, 2),
            net_pnl=100,
        ),
        ClosedTrade(
            trade_id="not-yet-known",
            symbol="S",
            signal_date=date(2022, 10, 1),
            entry_date=date(2022, 10, 2),
            exit_date=date(2022, 12, 20),
            net_pnl=100,
        ),
    )
    result = partition_mature_trades(
        trades, protocol, days, as_of=date(2022, 12, 10), embargo_sessions=0
    )
    assert tuple(t.trade_id for t in result.train) == ("mature",)
    assert {t.trade_id for t in result.excluded} == {"cross", "not-yet-known"}
    assert result.validation == result.final_test == ()


def account_example(tmp_path, protocol_data, *, sessions=180, enter=False):
    from copy import deepcopy

    from sequoia_x.research.dataset import DatasetSnapshot
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.protocol import ResearchProtocol
    from sequoia_x.research.registry import ResearchRegistry
    from sequoia_x.research.replay import ExecutionPolicy, FeeSchedule, Signal

    days = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(sessions))

    def price(index):
        return 12 if index >= 61 else (11 if index == 60 else 10)

    sample = DatasetSnapshot.model_validate(
        {
            "manifest": {
                "source": "synthetic-statistics-example",
                "source_version": "1",
                "universe_id": "synthetic-AAA",
                "symbols": ["AAA"],
                "start": days[0],
                "end": days[-1],
                "as_of": days[-1],
                "evidence_kind": "synthetic",
                "universe_complete": True,
                "states_complete": True,
                "actions_complete": True,
                "calendar_complete": True,
            },
            "calendar": days,
            "bars": [
                {
                    "symbol": "AAA",
                    "date": day,
                    "available_on": day,
                    "open": price(i),
                    "high": price(i),
                    "low": price(i),
                    "close": price(i),
                    "volume": 1000000,
                    "price_basis": "raw",
                }
                for i, day in enumerate(days)
            ],
            "states": [
                {
                    "symbol": "AAA",
                    "effective_date": day,
                    "available_on": day,
                    "listed": True,
                    "suspended": False,
                    "is_st": False,
                    "board": "SSE_MAIN",
                    "limit_rule": "bounded",
                    "limit_up": round(price(max(0, i - 1)) * 1.1, 2),
                    "limit_down": round(price(max(0, i - 1)) * 0.9, 2),
                }
                for i, day in enumerate(days)
            ],
        }
    )
    execution = ExecutionPolicy(
        fees=(
            FeeSchedule(
                effective_from=date(2000, 1, 1),
                commission_rate=0,
                min_commission=0,
                stamp_tax_rate=0,
                transfer_fee_rate=0,
            ),
        ),
        slippage_bps=0,
        max_order_quantity=1_000_000,
        max_volume_participation=1,
        volume_model="daily_volume_upper_bound",
    )
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    values = deepcopy(protocol_data)
    values.update(
        data_snapshot_id=sample.snapshot_id,
        universe=sample.manifest.universe_id,
        data_cutoff=days[-1],
        final_test={"start": days[0], "end": days[-1]},
        comparison_role="initial_reference",
    )
    values["acceptance"].pop("minimum_win_rate_delta")
    values["trading"].update(
        initial_capital=10000,
        max_position_fraction=1,
        max_positions=1,
        entry_order_time="next_session_open_limit",
        fees=[f"execution_policy:{execution.policy_id}"],
        capacity=f"selection_policy:{selection.policy_id}",
    )
    registry = ResearchRegistry(tmp_path / "statistics.sqlite")
    frozen = registry.freeze(ResearchProtocol.model_validate(values))
    control_values = deepcopy(values)
    control_values.update(
        study_id="cash-reference", strategy_version="cash-control", comparison_version="cash-anchor"
    )
    control = registry.freeze(ResearchProtocol.model_validate(control_values))
    signals = (Signal(symbol="AAA", signal_date=days[0]),) if enter else ()
    candidate = run_portfolio(sample, frozen, execution, selection, signals=signals)
    comparison = run_portfolio(sample, control, execution, selection, signals=())
    return registry, frozen, candidate, comparison, sample, execution, selection


def test_zero_trade_report_keeps_nav_risk_and_immutable_failure(tmp_path, protocol_data):
    from sequoia_x.research.statistics import analyze_baseline, load_report, record_report

    registry, frozen, candidate, comparison, sample, _, _ = account_example(tmp_path, protocol_data)
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={},
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    report = analyze_baseline(registry, run.experiment_id, candidate, comparison, snapshot=sample)
    assert report.candidate.closed_trades == 0
    assert report.candidate.net_win_rate is report.candidate.net_expectancy is None
    assert report.candidate.portfolio_net_return == 0
    assert report.candidate.open_exposure == 0
    assert report.decision.status == "insufficient_evidence"
    assert report.b0_version is None
    assert report.evidence_source == "synthetic"
    saved = record_report(report, tmp_path / "reports")
    assert load_report(saved.path, saved.sha256) == report
    assert record_report(report, tmp_path / "reports") == saved
    with pytest.raises(ValueError, match="已有结果"):
        analyze_baseline(registry, run.experiment_id, candidate, comparison, snapshot=sample)
    with pytest.raises(ValueError, match="最终测试日期已使用"):
        registry.start_experiment(
            frozen.protocol_id,
            parameters={},
            stage="final_test",
            purpose="confirmation",
            evidence_kind="historical_simulation",
        )


def test_legacy_report_load_checks_original_bytes_and_rearchive_keeps_version(
    tmp_path, protocol_data
):
    import json
    from hashlib import sha256

    from sequoia_x.research.statistics import analyze_baseline, load_report, record_report

    registry, frozen, candidate, comparison, sample, _, _ = account_example(tmp_path, protocol_data)
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={},
        stage="final_test",
        purpose="exploration",
        evidence_kind="historical_simulation",
    )
    report = analyze_baseline(registry, run.experiment_id, candidate, comparison, snapshot=sample)
    legacy = report.model_dump(mode="json")
    legacy["implementation_fingerprint"] = "historical-implementation-v1"
    for role in ("candidate", "comparison"):
        legacy[role].pop("unexitable_exposure")
    original = json.dumps(legacy, ensure_ascii=False, indent=2).encode("utf8")
    original_sha = sha256(original).hexdigest()
    path = tmp_path / "legacy-report.json"
    path.write_bytes(original)

    loaded = load_report(path, original_sha)
    assert loaded.candidate.unexitable_exposure is None
    assert loaded.comparison.unexitable_exposure is None
    assert loaded.implementation_fingerprint == "historical-implementation-v1"
    archived = record_report(loaded, tmp_path / "new-serialization")
    assert archived.sha256 != original_sha
    assert load_report(archived.path, archived.sha256) == loaded
    assert path.read_bytes() == original
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="内容指纹不符"):
        load_report(path, original_sha)


def test_one_hand_calculated_trade_reports_expectancy_and_missing_loss_group(
    tmp_path, protocol_data
):
    from sequoia_x.research.statistics import analyze_baseline

    registry, frozen, candidate, comparison, sample, _, _ = account_example(
        tmp_path,
        protocol_data,
        sessions=64,
        enter=True,
    )
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={},
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    report = analyze_baseline(registry, run.experiment_id, candidate, comparison, snapshot=sample)
    assert report.candidate.closed_trades == 1
    assert report.candidate.net_win_rate == 1
    assert report.candidate.net_expectancy == 1800
    assert report.candidate.average_win == 1800
    assert report.candidate.average_loss is report.candidate.payoff_ratio is None
    assert report.candidate.portfolio_net_return == pytest.approx(0.18)
    assert report.candidate.annualized_net_return == pytest.approx(
        candidate.metrics.annualized_net_return
    )
    assert report.comparison.net_win_rate is None
    assert report.decision.status == "insufficient_evidence"


def test_unfinished_position_does_not_become_a_win_or_disappear(tmp_path, protocol_data):
    from sequoia_x.research.statistics import analyze_baseline

    registry, frozen, candidate, comparison, sample, _, _ = account_example(
        tmp_path,
        protocol_data,
        sessions=4,
        enter=True,
    )
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={},
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    report = analyze_baseline(registry, run.experiment_id, candidate, comparison, snapshot=sample)
    assert report.candidate.closed_trades == 0
    assert report.candidate.net_win_rate is report.candidate.net_expectancy is None
    assert report.candidate.open_positions == report.candidate.unfinished_trades == 1
    assert report.candidate.open_exposure == 0.9


def test_report_keeps_unsupported_pre_maturity_position_and_exit_risk(tmp_path, protocol_data):
    from copy import deepcopy

    from sequoia_x.research.dataset import DatasetSnapshot
    from sequoia_x.research.portfolio import run_portfolio
    from sequoia_x.research.protocol import ResearchProtocol
    from sequoia_x.research.replay import Signal
    from sequoia_x.research.statistics import analyze_baseline, load_report, record_report

    registry, frozen, _, _, sample, execution, selection = account_example(
        tmp_path, protocol_data, sessions=10
    )
    raw = sample.model_dump(mode="json")
    for state in raw["states"][7:]:
        state.update(is_st=True, limit_up=10.5, limit_down=9.5)
    sample = DatasetSnapshot.model_validate(raw)
    values = frozen.protocol.model_dump(mode="json")
    values.update(
        study_id="pre-maturity-exit-risk",
        strategy_version="ma-pre-maturity-risk",
        comparison_version="cash-risk-reference",
        data_snapshot_id=sample.snapshot_id,
    )
    values["trading"]["initial_capital"] = 11000
    values["statistics"]["family_members"] = [values["strategy_version"]]
    frozen = registry.freeze(ResearchProtocol.model_validate(values))
    cash_values = deepcopy(values)
    cash_values.update(
        study_id="cash-pre-maturity-exit-risk",
        strategy_version="cash-risk-reference",
        comparison_version="cash-anchor",
    )
    cash = registry.freeze(ResearchProtocol.model_validate(cash_values))
    candidate = run_portfolio(
        sample,
        frozen,
        execution,
        selection,
        signals=(Signal(symbol="AAA", signal_date=sample.calendar[0]),),
    )
    comparison = run_portfolio(sample, cash, execution, selection, signals=())
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters={},
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    report = analyze_baseline(registry, run.experiment_id, candidate, comparison, snapshot=sample)

    # Hand calculation: 1000 shares * 10, plus 1000 cash; no exit has become due.
    assert candidate.daily_nav[-1].market_value == 10000
    assert candidate.daily_nav[-1].nav == 11000
    assert report.candidate.open_exposure == pytest.approx(10000 / 11000)
    assert report.candidate.unexitable_exposure == pytest.approx(10000 / 11000)
    assert report.comparison.unexitable_exposure == 0
    assert report.candidate.open_positions == report.candidate.unfinished_trades == 1
    assert report.candidate.closed_trades == 0
    assert report.candidate.net_win_rate is report.candidate.net_expectancy is None
    assert not candidate.execution_qualified
    assert any("risk_warning_rules_not_supported" in gap for gap in report.diagnostic_gaps)
    assert report.decision.status == "insufficient_evidence"
    assert report.b0_version is None
    saved = record_report(report, tmp_path / "pre-maturity-reports")
    assert load_report(saved.path, saved.sha256) == report


def registered_parameters(candidate, comparison, sample, execution, selection):
    from sequoia_x.research.statistics import (
        AccountContract,
        ReferenceDay,
        ReferenceSeries,
        RegimeMap,
        RegimePeriod,
        SectorMap,
        SectorRecord,
        StatisticsPlan,
        implementation_fingerprint,
    )

    def contract(account):
        return AccountContract.model_validate(
            {
                field: getattr(account, field)
                for field in (
                    "protocol_id",
                    "strategy_version",
                    "implementation_version",
                    "execution_policy_id",
                    "selection_policy_id",
                )
            }
        )

    reference = ReferenceSeries(
        source_version="synthetic-reference-v1",
        description="人工零收益参照",
        days=tuple(ReferenceDay(date=day, net_return=0) for day in sample.calendar),
    )
    sectors = SectorMap(
        source_version="synthetic-sector-v1",
        records=(
            SectorRecord(
                symbol="AAA",
                sector="fixture",
                start=sample.calendar[0],
                end=sample.calendar[-1],
                available_on=sample.calendar[0],
            ),
        ),
    )
    regimes = RegimeMap(
        source_version="synthetic-regime-v1",
        periods=(
            RegimePeriod(
                label="controlled-fixture",
                start=sample.calendar[0],
                end=sample.calendar[-1],
                available_on=sample.calendar[0],
            ),
        ),
    )
    plan = StatisticsPlan(
        strategy_family="ma",
        resampling=ResamplingPlan(block_sessions=62, resamples=2000, seed=7),
        embargo_sessions=0,
        dependence_method="moving_block_signal_date_v1",
        multiplicity_method="bonferroni_family_metrics_v1",
        assumption="weak_stationarity_block_length_covers_dependence",
        implementation_fingerprint=implementation_fingerprint(),
        candidate=contract(candidate),
        comparison=contract(comparison),
        market_reference_id=reference.reference_id,
        exposure_reference_id=reference.reference_id,
        sector_map_id=sectors.map_id,
        regime_map_id=regimes.map_id,
        power_method="normal_block_standard_error_v1",
        power_alternative_return_delta=0.2,
        power_alternative_win_rate=0.9,
        scenarios=(),
        minimum_scenario_return_delta=0,
    )
    parameters = {
        "statistics_plan": plan.model_dump(mode="json"),
        "execution_policies": {execution.policy_id: execution.model_dump(mode="json")},
        "selection_policies": {selection.policy_id: selection.model_dump(mode="json")},
    }
    return parameters, reference, sectors, regimes


def test_unsupported_frozen_method_is_not_relabelled_and_synthetic_never_passes(
    tmp_path, protocol_data
):
    from sequoia_x.research.statistics import analyze_baseline

    registry, frozen, candidate, comparison, sample, execution, selection = account_example(
        tmp_path,
        protocol_data,
        enter=True,
    )
    parameters, reference, sectors, regimes = registered_parameters(
        candidate,
        comparison,
        sample,
        execution,
        selection,
    )
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters=parameters,
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    report = analyze_baseline(
        registry,
        run.experiment_id,
        candidate,
        comparison,
        snapshot=sample,
        market_reference=reference,
        exposure_reference=reference,
        sector_map=sectors,
        regime_map=regimes,
    )
    assert any("不受支持" in gap for gap in report.diagnostic_gaps)
    assert report.intervals.dependence_method == "moving_block_signal_date_v1"
    assert report.intervals.multiplicity_method == "bonferroni_family_metrics_v1"
    assert report.comparison.net_win_rate is report.comparison.net_expectancy is None
    assert report.intervals.win_rate_delta is None
    assert report.decision.status == "insufficient_evidence"
    assert report.b0_version is None


def test_actual_implementation_and_replayed_ledger_are_checked(tmp_path, protocol_data):
    from sequoia_x.research.statistics import analyze_baseline

    registry, frozen, candidate, comparison, sample, execution, selection = account_example(
        tmp_path, protocol_data
    )
    parameters, reference, sectors, regimes = registered_parameters(
        candidate,
        comparison,
        sample,
        execution,
        selection,
    )
    parameters["statistics_plan"]["implementation_fingerprint"] = "stale-implementation"
    altered = candidate.daily_nav[-1].model_copy(update={"nav": 10100, "daily_net_return": 0.01})
    candidate = candidate.model_copy(
        update={"daily_nav": candidate.daily_nav[:-1] + (altered,), "data_qualified": True}
    )
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters=parameters,
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    report = analyze_baseline(
        registry,
        run.experiment_id,
        candidate,
        comparison,
        snapshot=sample,
        market_reference=reference,
        exposure_reference=reference,
        sector_map=sectors,
        regime_map=regimes,
    )
    assert any("源码指纹" in gap for gap in report.diagnostic_gaps)
    assert any("不同账户账本" in gap for gap in report.diagnostic_gaps)
    assert any("守恒" in gap for gap in report.diagnostic_gaps)
    assert report.decision.status == "insufficient_evidence"


def test_identical_account_cannot_be_called_a_cost_stress(tmp_path, protocol_data):
    from sequoia_x.research.statistics import analyze_baseline

    registry, frozen, candidate, comparison, sample, execution, selection = account_example(
        tmp_path, protocol_data
    )
    parameters, reference, sectors, regimes = registered_parameters(
        candidate,
        comparison,
        sample,
        execution,
        selection,
    )
    parameters["statistics_plan"]["scenarios"] = [
        {
            "scenario_id": "fake-stress",
            "kind": "cost_stress",
            "description": "人工误标相同账户",
            "account": parameters["statistics_plan"]["candidate"],
            "cost_multiplier": 2,
        }
    ]
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters=parameters,
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    report = analyze_baseline(
        registry,
        run.experiment_id,
        candidate,
        comparison,
        snapshot=sample,
        market_reference=reference,
        exposure_reference=reference,
        sector_map=sectors,
        regime_map=regimes,
        scenarios={"fake-stress": candidate},
    )
    assert any("均未增加" in reason for reason in report.robustness[0].reasons)


@pytest.mark.parametrize(("kind", "honest_cost"), [("parameter_neighbor", 5), ("cost_stress", 10)])
@pytest.mark.parametrize(
    ("variant", "changed_input"),
    [
        ("removed_all", True),
        ("removed_one", True),
        ("added", True),
        ("date", True),
        ("symbol", True),
        ("strategy", True),
        ("features", True),
        ("episode", True),
        ("duplicated", True),
        ("reordered", True),
        ("feature_key_reordered", False),
        ("unchanged", False),
    ],
)
def test_trading_scenarios_keep_the_declared_raw_signal_sequence(
    tmp_path, protocol_data, variant, changed_input, kind, honest_cost
):
    from copy import deepcopy

    from sequoia_x.research.portfolio import run_portfolio
    from sequoia_x.research.protocol import ResearchProtocol
    from sequoia_x.research.replay import ExecutionPolicy, Signal
    from sequoia_x.research.statistics import analyze_baseline

    registry, original, _, _, sample, execution, selection = account_example(
        tmp_path, protocol_data, sessions=10
    )
    fee_values = execution.model_dump(mode="json")
    fee_values["fees"][0]["min_commission"] = 5
    execution = ExecutionPolicy.model_validate(fee_values)
    values = original.protocol.model_dump(mode="json")
    values.update(study_id="neighbor-signal-pool", comparison_version="neighbor-cash")
    values["trading"].update(
        initial_capital=11000,
        fees=[f"execution_policy:{execution.policy_id}"],
        repeat_signal_policy="ignore_while_open",
    )
    values["statistics"]["family_members"] = [values["strategy_version"], "position-neighbor"]
    frozen = registry.freeze(ResearchProtocol.model_validate(values))
    neighbor_values = deepcopy(values)
    neighbor_values.update(study_id="position-neighbor", strategy_version="position-neighbor")
    scenario_execution = execution
    if kind == "parameter_neighbor":
        neighbor_values["trading"]["max_position_fraction"] = 0.5
    else:
        stress_values = execution.model_dump(mode="json")
        stress_values["fees"][0]["min_commission"] = 10
        scenario_execution = ExecutionPolicy.model_validate(stress_values)
        neighbor_values["trading"]["fees"] = [f"execution_policy:{scenario_execution.policy_id}"]
    neighbor = registry.freeze(ResearchProtocol.model_validate(neighbor_values))
    cash_values = deepcopy(values)
    cash_values.update(
        study_id="neighbor-cash", strategy_version="neighbor-cash", comparison_version="anchor"
    )
    cash = registry.freeze(ResearchProtocol.model_validate(cash_values))
    signals = (
        Signal(
            symbol="AAA",
            signal_date=sample.calendar[0],
            episode_id="original-episode",
            features={"alpha": 1.0, "beta": 2.0},
        ),
        Signal(symbol="AAA", signal_date=sample.calendar[3], episode_id="original-episode"),
    )
    benchmark = {day: 100.0 for day in sample.calendar}
    candidate = run_portfolio(
        sample, frozen, execution, selection, signals=signals, benchmark=benchmark
    )
    comparison = run_portfolio(sample, cash, execution, selection, signals=(), benchmark=benchmark)
    variants = {
        "removed_all": (),
        "removed_one": signals[1:],
        "added": signals + (Signal(symbol="AAA", signal_date=sample.calendar[6]),),
        "date": (signals[0].model_copy(update={"signal_date": sample.calendar[1]}), signals[1]),
        "symbol": (signals[0].model_copy(update={"symbol": "BBB"}), signals[1]),
        "strategy": (signals[0].model_copy(update={"strategy": "flag"}), signals[1]),
        "features": (signals[0].model_copy(update={"features": {"source": 1.0}}), signals[1]),
        "episode": (signals[0].model_copy(update={"episode_id": "changed-episode"}), signals[1]),
        "duplicated": signals,
        "reordered": tuple(reversed(signals)),
        "feature_key_reordered": (
            signals[0].model_copy(update={"features": {"beta": 2.0, "alpha": 1.0}}),
            signals[1],
        ),
        "unchanged": signals,
    }
    supplied = run_portfolio(
        sample,
        neighbor,
        scenario_execution,
        selection,
        signals=variants[variant],
        benchmark=benchmark,
    )
    if variant == "duplicated":
        # The report boundary must reject multiplicity drift, including a forged ledger.
        supplied = supplied.model_copy(update={"signals": signals + signals[:1]})
    honest = run_portfolio(
        sample, neighbor, scenario_execution, selection, signals=signals, benchmark=benchmark
    )
    parameters, reference, sectors, regimes = registered_parameters(
        candidate, comparison, sample, execution, selection
    )
    parameters["statistics_plan"]["scenarios"] = [
        {
            "scenario_id": "position-neighbor",
            "kind": kind,
            "description": "Only the frozen position fraction or cost multiplier changes",
            "account": {
                **parameters["statistics_plan"]["candidate"],
                "protocol_id": neighbor.protocol_id,
                "strategy_version": "position-neighbor",
                "execution_policy_id": scenario_execution.policy_id,
            },
            **(
                {"parameter_changes": {"trading.max_position_fraction": 0.5}}
                if kind == "parameter_neighbor"
                else {"cost_multiplier": 2}
            ),
        }
    ]
    parameters["execution_policies"][scenario_execution.policy_id] = scenario_execution.model_dump(
        mode="json"
    )
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters=parameters,
        stage="final_test",
        purpose="exploration",
        evidence_kind="historical_simulation",
    )
    report = analyze_baseline(
        registry,
        run.experiment_id,
        candidate,
        comparison,
        snapshot=sample,
        market_reference=reference,
        exposure_reference=reference,
        sector_map=sectors,
        regime_map=regimes,
        scenarios={"position-neighbor": supplied},
    )
    # Each honest account pays the same minimum commission, regardless of share count.
    assert candidate.metrics.total_costs == 5
    assert honest.metrics.total_costs == honest_cost
    assert candidate.metrics.portfolio_net_return == pytest.approx(-5 / 11000)
    assert honest.metrics.portfolio_net_return == pytest.approx(-honest_cost / 11000)
    if variant == "removed_all":
        assert supplied.metrics.portfolio_net_return == 0
        assert report.robustness[0].primary_return_delta == 0
    if not changed_input:
        assert supplied.metrics.total_costs == honest_cost
        assert report.robustness[0].primary_return_delta == pytest.approx(-honest_cost / 11000)
        assert any(
            "未达到预登记净收益增量下限" in reason for reason in report.robustness[0].reasons
        )
    assert any("信号池" in reason for reason in report.robustness[0].reasons) is changed_input
    assert any("信号池" in reason for reason in report.diagnostic_gaps) is changed_input
    assert report.b0_version is None


def test_scenario_preserves_signal_order_when_exit_fee_cash_is_shared(tmp_path, protocol_data):
    """Controlled observed-tagged software fixture, never empirical return evidence."""
    from copy import deepcopy

    from sequoia_x.research.dataset import DatasetSnapshot
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.protocol import ResearchProtocol
    from sequoia_x.research.replay import ExecutionPolicy, FeeSchedule, Signal
    from sequoia_x.research.statistics import analyze_baseline, load_report, record_report

    registry, original, _, _, sample, _, _ = account_example(tmp_path, protocol_data, sessions=63)
    days = sample.calendar
    raw = sample.model_dump(mode="json")
    raw["manifest"].update(
        symbols=["AAA", "BBB"],
        universe_id="controlled-order-priority",
        evidence_kind="observed",
        source="Controlled software fixture, NOT market evidence",
    )
    raw["bars"], raw["states"] = [], []
    for symbol in ("AAA", "BBB"):
        previous = 10
        for index, day in enumerate(days):
            price = (
                10.5
                if symbol == "AAA" and index >= 62
                else (10.1 if symbol == "BBB" and index >= 61 else 10)
            )
            raw["bars"].append(
                dict(
                    symbol=symbol,
                    date=day,
                    available_on=day,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=100000,
                    price_basis="raw",
                )
            )
            raw["states"].append(
                dict(
                    symbol=symbol,
                    effective_date=day,
                    available_on=day,
                    listed=True,
                    suspended=False,
                    is_st=False,
                    board="SSE_MAIN",
                    lot_size=100,
                    limit_rule="bounded",
                    limit_up=round(previous * 1.1, 2),
                    limit_down=round(previous * 0.9, 2),
                )
            )
            previous = price
    sample = DatasetSnapshot.model_validate(raw)

    def execution(final_fee):
        return ExecutionPolicy(
            fees=tuple(
                FeeSchedule(
                    effective_from=effective,
                    commission_rate=0,
                    min_commission=fee,
                    stamp_tax_rate=0,
                    transfer_fee_rate=0,
                )
                for effective, fee in ((date(2000, 1, 1), 0), (days[61], final_fee))
            ),
            slippage_bps=0,
            max_order_quantity=1_000_000,
            max_volume_participation=1,
            volume_model="daily_volume_upper_bound",
        )

    ordinary, stressed = execution(5750), execution(11500)
    selection = PortfolioPolicy(daily_entry_slots=2, selection="symbol_order", seed=7)
    values = original.protocol.model_dump(mode="json")
    values.update(
        study_id="fee-order-base",
        strategy_version="fee-order-base",
        comparison_version="fee-order-cash",
        comparison_role="approved_strategy",
        data_snapshot_id=sample.snapshot_id,
        universe=sample.manifest.universe_id,
    )
    values["trading"].update(
        initial_capital=22000,
        max_position_fraction=0.5,
        max_positions=2,
        fees=[f"execution_policy:{ordinary.policy_id}"],
        capacity=f"selection_policy:{selection.policy_id}",
        repeat_signal_policy="ignore_while_open",
    )
    values["statistics"]["family_members"] = ["fee-order-base", "fee-order-stress"]
    values["acceptance"]["minimum_win_rate_delta"] = -0.01
    frozen = registry.freeze(ResearchProtocol.model_validate(values))
    cash_values = deepcopy(values)
    cash_values.update(
        study_id="fee-order-cash", strategy_version="fee-order-cash", comparison_version="anchor"
    )
    cash = registry.freeze(ResearchProtocol.model_validate(cash_values))
    stress_values = deepcopy(values)
    stress_values.update(study_id="fee-order-stress", strategy_version="fee-order-stress")
    stress_values["trading"]["fees"] = [f"execution_policy:{stressed.policy_id}"]
    stress = registry.freeze(ResearchProtocol.model_validate(stress_values))
    signals = tuple(Signal(symbol=symbol, signal_date=days[0]) for symbol in ("AAA", "BBB"))
    benchmark = {day: 22000 for day in days}
    candidate = run_portfolio(
        sample, frozen, ordinary, selection, signals=signals, benchmark=benchmark
    )
    comparison = run_portfolio(sample, cash, ordinary, selection, signals=(), benchmark=benchmark)
    forward = run_portfolio(
        sample, stress, stressed, selection, signals=signals, benchmark=benchmark
    )
    reverse = run_portfolio(
        sample, stress, stressed, selection, signals=signals[::-1], benchmark=benchmark
    )
    parameters, reference, sectors, regimes = registered_parameters(
        candidate, comparison, sample, ordinary, selection
    )
    parameters["execution_policies"][stressed.policy_id] = stressed.model_dump(mode="json")
    parameters["statistics_plan"]["resampling"]["resamples"] = 100
    parameters["statistics_plan"]["minimum_scenario_return_delta"] = -0.51
    parameters["statistics_plan"]["scenarios"] = [
        {
            "scenario_id": "fee-order-stress",
            "kind": "cost_stress",
            "description": "Only all fees x2, retaining actual execution-input priority",
            "cost_multiplier": 2,
            "account": {
                **parameters["statistics_plan"]["candidate"],
                "protocol_id": stress.protocol_id,
                "strategy_version": "fee-order-stress",
                "execution_policy_id": stressed.policy_id,
            },
        }
    ]
    reports = {}
    for name, account in (("forward", forward), ("reverse", reverse)):
        run = registry.start_experiment(
            frozen.protocol_id,
            parameters=parameters,
            stage="final_test",
            purpose="exploration",
            evidence_kind="historical_simulation",
        )
        reports[name] = analyze_baseline(
            registry,
            run.experiment_id,
            candidate,
            comparison,
            snapshot=sample,
            market_reference=reference,
            exposure_reference=reference,
            sector_map=sectors,
            regime_map=regimes,
            scenarios={"fee-order-stress": account},
        )
        assert registry.decisions(frozen.protocol.study_id)[-1] == reports[name].decision
        assert reports[name].decision.evidence is None
        assert reports[name].b0_version is None
    assert registry.latest_approved(frozen.protocol.study_id, "historical_simulation") is None
    assert forward.data_qualified and reverse.data_qualified
    assert forward.execution_qualified and reverse.execution_qualified
    assert sorted(s.signal_id for s in forward.signals) == sorted(
        s.signal_id for s in reverse.signals
    )
    # Both start the exit auction with 2000 fee cash; only one stressed sell fits.
    assert forward.daily_nav[-1].cash == 500
    assert forward.daily_nav[-1].market_value == 10100
    assert forward.daily_nav[-1].nav == 10600
    assert reverse.daily_nav[-1].cash == 600
    assert reverse.daily_nav[-1].market_value == 10500
    assert reverse.daily_nav[-1].nav == 11100
    assert reports["forward"].robustness[0].primary_return_delta == pytest.approx(10600 / 22000 - 1)
    assert any(
        "未达到预登记净收益增量下限" in reason
        for reason in reports["forward"].robustness[0].reasons
    )
    assert reports["reverse"].robustness[0].primary_return_delta == pytest.approx(11100 / 22000 - 1)
    assert any("执行顺序" in reason for reason in reports["reverse"].robustness[0].reasons)
    assert any("执行顺序" in reason for reason in reports["reverse"].diagnostic_gaps)
    assert any("执行顺序" in reason for reason in reports["reverse"].decision.reasons)
    saved = record_report(reports["reverse"], tmp_path / "ordered-risk-report")
    assert load_report(saved.path, saved.sha256) == reports["reverse"]


def test_seed_is_reproducible_and_bonferroni_larger_family_widens_intervals():
    days = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(180))
    returns = (0.001,) * 90 + (-0.0002,) * 90
    trades = tuple(
        ClosedTrade(
            trade_id=f"trade-{i}",
            symbol="S",
            signal_date=days[i],
            entry_date=days[i + 1],
            exit_date=days[i + 61],
            net_pnl=100 if i % 3 else -50,
        )
        for i in range(110)
    )
    args = (
        days,
        returns,
        (0.0,) * 180,
        trades,
        (),
        ResamplingPlan(block_sessions=62, resamples=6000, seed=17),
    )
    single = summarize_dependence(*args, confidence_level=0.95, family_size=1)
    repeat = summarize_dependence(*args, confidence_level=0.95, family_size=1)
    family = summarize_dependence(*args, confidence_level=0.95, family_size=3)
    assert single == repeat
    assert family.return_delta.lower <= single.return_delta.lower
    assert family.return_delta.upper >= single.return_delta.upper
    assert family.candidate_win_rate.lower <= single.candidate_win_rate.lower
    assert family.familywise_interval_count == 36


def test_all_requirements_power_is_a_joint_lower_bound_not_minimum(tmp_path, protocol_data):
    from sequoia_x.research.statistics import StatisticsPlan, assess_power

    _, frozen, candidate, comparison, sample, execution, selection = account_example(
        tmp_path, protocol_data
    )
    parameters, _, _, _ = registered_parameters(candidate, comparison, sample, execution, selection)
    # 标准正态表：z(1-.05/24)=2.8652602385，z(.8)=.8416212336。
    # se=.1、margin=.3706881472 时每个单方向通过概率=.8。
    parameters["statistics_plan"].update(
        power_alternative_return_delta=0.39068814721050486,
        power_alternative_win_rate=0.8706881472105049,
    )
    plan = StatisticsPlan.model_validate(parameters["statistics_plan"])
    summary = summarize_dependence(
        sample.calendar,
        (0.0,) * len(sample.calendar),
        (0.0,) * len(sample.calendar),
        (),
        (),
        ResamplingPlan(block_sessions=62, resamples=2000, seed=7),
        confidence_level=0.95,
        family_size=1,
    ).model_copy(update={"return_delta_standard_error": 0.1, "win_rate_standard_error": 0.1})
    power = assess_power(summary, frozen.protocol, plan)
    assert [m.estimated_power for m in power.metrics] == pytest.approx([0.8, 0.8])
    # 未假定两门槛独立：union bound为.6（独立时实际也只有.64），不能宣称.8。
    assert power.achieved_power == pytest.approx(0.6)
    assert any("联合" in reason for reason in power.reasons)


def test_power_does_not_count_rejection_in_the_wrong_direction(tmp_path, protocol_data):
    from sequoia_x.research.statistics import StatisticsPlan, assess_power

    _, frozen, candidate, comparison, sample, execution, selection = account_example(
        tmp_path, protocol_data
    )
    parameters, _, _, _ = registered_parameters(candidate, comparison, sample, execution, selection)
    parameters["statistics_plan"]["power_alternative_return_delta"] = 0.03
    plan = StatisticsPlan.model_validate(parameters["statistics_plan"])
    summary = summarize_dependence(
        sample.calendar,
        (0.0,) * len(sample.calendar),
        (0.0,) * len(sample.calendar),
        (),
        (),
        ResamplingPlan(block_sessions=62, resamples=2000, seed=7),
        confidence_level=0.95,
        family_size=1,
    ).model_copy(update={"return_delta_standard_error": 0.1, "win_rate_standard_error": 0.1})
    power = assess_power(summary, frozen.protocol, plan)
    actual = next(m.estimated_power for m in power.metrics if m.metric == "return_delta")
    # 标准正态表：1-Phi(2.8652602385-.1)=.002843869999。
    assert actual == pytest.approx(0.002843869999026305)


def test_scenarios_share_mature_denominator_and_exclusions_have_account(tmp_path, protocol_data):
    from sequoia_x.research.portfolio import run_portfolio
    from sequoia_x.research.replay import Signal
    from sequoia_x.research.statistics import analyze_baseline

    registry, frozen, _, comparison, sample, execution, selection = account_example(
        tmp_path, protocol_data
    )
    candidate = run_portfolio(
        sample,
        frozen,
        execution,
        selection,
        signals=(
            Signal(symbol="AAA", signal_date=sample.calendar[0]),
            Signal(symbol="AAA", signal_date=sample.calendar[70]),
        ),
    )
    parameters, reference, sectors, regimes = registered_parameters(
        candidate,
        comparison,
        sample,
        execution,
        selection,
    )
    parameters["statistics_plan"]["embargo_sessions"] = 10
    parameters["statistics_plan"]["scenarios"] = [
        {
            "scenario_id": "same-account",
            "kind": "cost_stress",
            "description": "同账户用于口径核对",
            "account": parameters["statistics_plan"]["candidate"],
            "cost_multiplier": 2,
        }
    ]
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters=parameters,
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    report = analyze_baseline(
        registry,
        run.experiment_id,
        candidate,
        comparison,
        snapshot=sample,
        market_reference=reference,
        exposure_reference=reference,
        sector_map=sectors,
        regime_map=regimes,
        scenarios={"same-account": candidate},
    )
    assert report.candidate.raw_trade_count == report.candidate.raw_closed_trades == 2
    assert report.candidate.closed_trades == report.candidate.purged_closed_trades == 1
    assert report.candidate.net_win_rate == report.robustness[0].net_win_rate == 0
    assert report.robustness[0].net_win_rate_change_vs_candidate == 0
    assert report.maturity.excluded[0].account == "candidate"


def controlled_initial_report(tmp_path, protocol_data):
    """仅覆盖observed资格与通过分支的受控构造，绝非真实市场证据。"""
    from copy import deepcopy
    from math import pi, sin

    from sequoia_x.research.dataset import DatasetSnapshot
    from sequoia_x.research.portfolio import IMPLEMENTATION_VERSION, PortfolioPolicy, run_portfolio
    from sequoia_x.research.protocol import ResearchProtocol
    from sequoia_x.research.registry import ResearchRegistry
    from sequoia_x.research.replay import ExecutionPolicy, FeeSchedule
    from sequoia_x.research.statistics import (
        AccountContract,
        ReferenceDay,
        ReferenceSeries,
        RegimeMap,
        RegimePeriod,
        ResamplingPlan,
        ScenarioContract,
        SectorMap,
        SectorRecord,
        StatisticsPlan,
        analyze_baseline,
        implementation_fingerprint,
        record_report,
    )

    calendar = tuple(date(2028, 1, 2) + timedelta(days=i) for i in range(900))
    symbols = ("600001", "600002", "600003")
    bars, states = [], []
    for number, symbol in enumerate(symbols):
        period, drift = (30, -0.001) if number == 2 else (120, 0.006)
        prices = [
            round(20 + drift * i + 1.5 * sin(2 * pi * i / period + number + 0.37), 2)
            for i in range(len(calendar))
        ]
        for i, (day, price) in enumerate(zip(calendar, prices)):
            cross = (
                i >= 20
                and sum(prices[i - 5 : i]) / 5 < sum(prices[i - 20 : i]) / 20
                and sum(prices[i - 4 : i + 1]) / 5 > sum(prices[i - 19 : i + 1]) / 20
            )
            bars.append(
                dict(
                    symbol=symbol,
                    date=day,
                    available_on=day,
                    open=price,
                    high=round(price * 1.001, 2),
                    low=round(price * 0.999, 2),
                    close=price,
                    volume=100000 if cross else 10000,
                    price_basis="raw",
                )
            )
            previous = prices[max(0, i - 1)]
            states.append(
                dict(
                    symbol=symbol,
                    effective_date=day,
                    available_on=day,
                    listed=True,
                    suspended=False,
                    is_st=False,
                    board="SSE_MAIN",
                    lot_size=100,
                    limit_rule="bounded",
                    limit_up=round(previous * 1.1, 2),
                    limit_down=round(previous * 0.9, 2),
                )
            )
    snapshot = DatasetSnapshot.model_validate(
        dict(
            manifest=dict(
                source="controlled-observed-tag-NOT-empirical",
                source_version="fixture-v2-different-calendar-and-prices",
                universe_id="controlled-three-symbols",
                symbols=symbols,
                start=calendar[0],
                end=calendar[-1],
                as_of=calendar[-1],
                evidence_kind="observed",
                universe_complete=True,
                states_complete=True,
                actions_complete=True,
                calendar_complete=True,
            ),
            calendar=calendar,
            bars=bars,
            states=states,
        )
    )

    def execution(multiplier):
        return ExecutionPolicy(
            fees=(
                FeeSchedule(
                    effective_from=date(2000, 1, 1),
                    commission_rate=0.0001 * multiplier,
                    min_commission=1 * multiplier,
                    stamp_tax_rate=0,
                    transfer_fee_rate=0,
                ),
            ),
            slippage_bps=1 * multiplier,
            max_volume_participation=1,
            volume_model="daily_volume_upper_bound",
            max_order_quantity=1000000,
            order_type="limit_at_verified_session_bounds",
        )

    base_execution, stress_execution = execution(1), execution(2)
    selection = PortfolioPolicy(daily_entry_slots=3, selection="symbol_order", seed=7)
    values = deepcopy(protocol_data)
    values.update(
        data_snapshot_id=snapshot.snapshot_id,
        universe=snapshot.manifest.universe_id,
        data_cutoff=calendar[-1],
        final_test={"start": calendar[0], "end": calendar[-1]},
        comparison_role="initial_reference",
    )
    values["statistics"].update(
        dependence_method="moving_block_signal_date_v1",
        multiplicity_method="bonferroni_family_metrics_v1",
    )
    values["acceptance"].update(
        minimum_return_delta=0.001,
        minimum_win_rate=0.05,
        minimum_win_rate_delta=None,
        max_drawdown=1,
        max_open_exposure=1,
    )
    values["trading"].update(
        initial_capital=100000,
        max_position_fraction=0.3,
        max_positions=3,
        repeat_signal_policy="ignore_while_open",
        entry_order_time="next_session_open_limit",
        fees=[f"execution_policy:{base_execution.policy_id}"],
        capacity=f"selection_policy:{selection.policy_id}",
    )
    registry = ResearchRegistry(tmp_path / "b0-branch.sqlite")
    frozen = registry.freeze(ResearchProtocol.model_validate(values))
    control_values = deepcopy(values)
    control_values.update(
        study_id="cash-reference", strategy_version="cash-control", comparison_version="cash-anchor"
    )
    control = registry.freeze(ResearchProtocol.model_validate(control_values))
    stress_values = deepcopy(values)
    stress_values["study_id"] = "stress-reference"
    stress_values["trading"]["fees"] = [f"execution_policy:{stress_execution.policy_id}"]
    stressed = registry.freeze(ResearchProtocol.model_validate(stress_values))
    neighbor_values = deepcopy(values)
    neighbor_values.update(study_id="neighbor-reference", strategy_version="ma-v2")
    neighbor_values["trading"]["max_position_fraction"] = 0.25
    neighbor = registry.freeze(ResearchProtocol.model_validate(neighbor_values))

    def contract(item, policy=base_execution):
        return AccountContract(
            protocol_id=item.protocol_id,
            strategy_version=item.protocol.strategy_version,
            implementation_version=IMPLEMENTATION_VERSION,
            execution_policy_id=policy.policy_id,
            selection_policy_id=selection.policy_id,
        )

    reference = ReferenceSeries(
        source_version="controlled-reference-v1",
        description="Controlled same-calendar reference",
        days=tuple(ReferenceDay(date=d, net_return=0) for d in calendar),
    )
    sectors = SectorMap(
        source_version="controlled-sector-v1",
        records=tuple(
            SectorRecord(
                symbol=s,
                sector=f"sector-{i}",
                start=calendar[0],
                end=calendar[-1],
                available_on=calendar[0],
            )
            for i, s in enumerate(symbols)
        ),
    )
    regimes = RegimeMap(
        source_version="controlled-regime-v1",
        periods=(
            RegimePeriod(
                label="controlled-regime",
                start=calendar[0],
                end=calendar[-1],
                available_on=calendar[0],
            ),
        ),
    )
    plan = StatisticsPlan(
        strategy_family="ma",
        resampling=ResamplingPlan(block_sessions=62, resamples=5000, seed=23),
        embargo_sessions=0,
        dependence_method="moving_block_signal_date_v1",
        multiplicity_method="bonferroni_family_metrics_v1",
        assumption="weak_stationarity_block_length_covers_dependence",
        implementation_fingerprint=implementation_fingerprint(),
        candidate=contract(frozen),
        comparison=contract(control),
        market_reference_id=reference.reference_id,
        exposure_reference_id=reference.reference_id,
        sector_map_id=sectors.map_id,
        regime_map_id=regimes.map_id,
        power_method="normal_block_standard_error_v1",
        power_alternative_return_delta=1,
        power_alternative_win_rate=0.95,
        scenarios=(
            ScenarioContract(
                scenario_id="cost-2x",
                kind="cost_stress",
                description="All fees and slippage 2x",
                account=contract(stressed, stress_execution),
                cost_multiplier=2,
            ),
            ScenarioContract(
                scenario_id="position-neighbor",
                kind="parameter_neighbor",
                description="Position fraction 0.25",
                account=contract(neighbor),
                parameter_changes={"trading.max_position_fraction": 0.25},
            ),
        ),
        minimum_scenario_return_delta=0,
    )
    parameters = dict(
        statistics_plan=plan.model_dump(mode="json"),
        execution_policies={
            p.policy_id: p.model_dump(mode="json") for p in (base_execution, stress_execution)
        },
        selection_policies={selection.policy_id: selection.model_dump(mode="json")},
    )
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters=parameters,
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    benchmark = {day: 100.0 for day in calendar}
    candidate = run_portfolio(snapshot, frozen, base_execution, selection, benchmark=benchmark)
    comparison = run_portfolio(
        snapshot, control, base_execution, selection, signals=(), benchmark=benchmark
    )
    scenarios = {
        "cost-2x": run_portfolio(
            snapshot, stressed, stress_execution, selection, benchmark=benchmark
        ),
        "position-neighbor": run_portfolio(
            snapshot, neighbor, base_execution, selection, benchmark=benchmark
        ),
    }
    report = analyze_baseline(
        registry,
        run.experiment_id,
        candidate,
        comparison,
        snapshot=snapshot,
        market_reference=reference,
        exposure_reference=reference,
        sector_map=sectors,
        regime_map=regimes,
        scenarios=scenarios,
    )
    saved = record_report(report, tmp_path / "reports")
    return registry, report, run, saved


def test_controlled_initial_b0_uses_actual_replay_and_keeps_cash_metrics_na(
    tmp_path, protocol_data
):
    from time import perf_counter

    from sequoia_x.research.statistics import load_report

    started = perf_counter()
    registry, report, run, saved = controlled_initial_report(tmp_path, protocol_data)
    elapsed = perf_counter() - started
    print(
        {
            "fixture_only_not_empirical": True,
            "elapsed_seconds": elapsed,
            "registry": str(registry.path),
            "report": saved.path,
            "report_sha256": saved.sha256,
            "experiment": run.experiment_id,
            "seed": 23,
            "sessions": 900,
            "status": report.decision.status,
            "reasons": report.decision.reasons,
        }
    )
    assert report.diagnostic_gaps == ()
    assert report.decision.status == "adopted"
    assert report.b0_version == "ma-v1"
    assert report.strategy_family == "ma"
    assert report.candidate.closed_trades > 0
    assert report.comparison.closed_trades == 0
    assert report.comparison.net_win_rate is report.comparison.net_expectancy is None
    assert report.intervals.comparison_win_rate is None
    assert report.intervals.win_rate_delta is None
    assert report.intervals.candidate_win_rate.lower > 0.05
    assert report.intervals.return_delta.lower > 0.001
    assert report.power.achieved_power >= 0.8
    assert {scenario.kind for scenario in report.robustness} == {
        "cost_stress",
        "parameter_neighbor",
    }
    assert all(scenario.reasons == () for scenario in report.robustness)
    assert load_report(saved.path, saved.sha256) == report
    with pytest.raises(ValueError, match="已有通过版本|最终测试日期已使用"):
        registry.start_experiment(
            run.protocol_id,
            parameters={},
            stage="final_test",
            purpose="confirmation",
            evidence_kind=run.evidence_kind,
        )


def test_actual_flag_portfolio_has_separate_zero_episode_report(tmp_path, protocol_data):
    from sequoia_x.research.flag import run_flag_portfolio
    from sequoia_x.research.protocol import ResearchProtocol
    from sequoia_x.research.statistics import analyze_baseline

    registry, ma_frozen, _, comparison, sample, execution, selection = account_example(
        tmp_path, protocol_data
    )
    values = ma_frozen.protocol.model_dump(mode="json")
    values.update(study_id="flag-baseline", strategy_version="flag-v1")
    values["statistics"].update(family_id="flag-family", family_members=["flag-v1", "flag-v2"])
    frozen = registry.freeze(ResearchProtocol.model_validate(values))
    flag = run_flag_portfolio(sample, frozen, execution, selection)
    parameters, reference, sectors, regimes = registered_parameters(
        flag.portfolio, comparison, sample, execution, selection
    )
    parameters["statistics_plan"]["strategy_family"] = "flag"
    run = registry.start_experiment(
        frozen.protocol_id,
        parameters=parameters,
        stage="final_test",
        purpose="confirmation",
        evidence_kind="historical_simulation",
    )
    report = analyze_baseline(
        registry,
        run.experiment_id,
        flag.portfolio,
        comparison,
        snapshot=sample,
        market_reference=reference,
        exposure_reference=reference,
        sector_map=sectors,
        regime_map=regimes,
    )
    assert flag.replay.coverage.episode_count == 0
    assert report.strategy_family == "flag"
    assert report.candidate.closed_trades == report.candidate.independent_episodes == 0
    assert report.candidate.net_win_rate is report.candidate.net_expectancy is None
    assert report.intervals.candidate_win_rate is report.intervals.candidate_expectancy is None
    assert report.b0_version is None
    assert report.decision.status == "insufficient_evidence"
