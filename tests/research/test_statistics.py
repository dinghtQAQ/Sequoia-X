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
                    "open": 12 if i >= 61 else 10,
                    "high": 12 if i >= 61 else 10,
                    "low": 12 if i >= 61 else 10,
                    "close": 12 if i >= 61 else 10,
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
                    "board": "main",
                    "limit_rule": "unbounded",
                }
                for day in days
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
        entry_order_time="signal_next_session_open_day",
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
    assert report.candidate.net_expectancy == 2000
    assert report.candidate.average_win == 2000
    assert report.candidate.average_loss is report.candidate.payoff_ratio is None
    assert report.candidate.portfolio_net_return == pytest.approx(0.2)
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
    assert report.candidate.open_exposure == 1


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
