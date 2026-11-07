"""flag 的公开观察与组合边界；行情全部为手算合成样例。"""

from datetime import date

import pandas as pd
import pytest

from sequoia_x.research.flag import observe_flag


def flag_frame(sessions: int = 42) -> pd.DataFrame:
    days = pd.bdate_range("2024-01-02", periods=sessions).date
    return pd.DataFrame(
        {
            "date": days,
            "open": [10.5 if i < 10 else 18 for i in range(sessions)],
            "high": [11 if i < 10 else 18.5 for i in range(sessions)],
            "low": [10 if i < 10 else 17.5 for i in range(sessions)],
            "close": [10.5 if i < 10 else 18 for i in range(sessions)],
            "volume": [1000 if i < 39 else 100 for i in range(sessions)],
        }
    )


def test_observer_preserves_all_four_restored_conditions_and_features() -> None:
    frame = flag_frame()
    on = frame.iloc[39]["date"]
    observation = observe_flag(frame, on=on)
    assert observation.formed is True
    assert observation.status == "formed"
    assert observation.features["high40"] == 18.5
    assert observation.features["low40"] == 10
    assert observation.features["high10"] == 18.5
    assert observation.features["low10"] == 17.5
    assert observation.features["previous_vol_ma20"] == 1000
    assert observation.features["momentum_ratio"] == 1.85
    assert observation.features["consolidation_ratio"] == pytest.approx(1.0571428571428572)
    assert observation.features["volume_ratio"] == 0.1


@pytest.mark.parametrize("column,value", [("low", 0), ("volume", 0), ("high", float("nan"))])
def test_observer_reports_invalid_recent_prices_and_volume_as_unavailable(column, value) -> None:
    frame = flag_frame()
    frame.loc[38, column] = value
    observation = observe_flag(frame, on=frame.iloc[39]["date"])
    assert observation.formed is None
    assert observation.status == "data_unavailable"


def test_observer_is_unchanged_by_future_rows_and_requires_current_data() -> None:
    frame = flag_frame()
    on = frame.iloc[39]["date"]
    past = frame.iloc[:40].copy()
    before = observe_flag(past, on=on)
    frame.loc[40:, ["high", "low", "volume"]] = [100, 1, 100000]
    assert observe_flag(frame, on=on) == before
    assert observe_flag(past.iloc[:-1], on=on).status == "data_unavailable"
    assert observe_flag(past.iloc[:39], on=past.iloc[38]["date"]).status == "warmup"


def test_price_range_does_not_require_low_before_high_or_current_breakout() -> None:
    frame = flag_frame(40)
    frame.loc[:9, ["open", "high", "low", "close"]] = [19.5, 20, 19, 19.5]
    frame.loc[10, ["open", "high", "low", "close"]] = [10.5, 11, 10, 10.5]
    observation = observe_flag(frame, on=frame.iloc[-1]["date"])
    assert observation.formed is True
    assert observation.features["high40"] == 20
    assert frame["high"].idxmax() < frame["low"].idxmin()
    assert frame.iloc[-1]["high"] < observation.features["high40"]


def test_every_daily_observation_matches_the_restored_daily_strategy(tmp_path) -> None:
    from unittest.mock import patch

    from sequoia_x.core.config import Settings
    from sequoia_x.data.engine import DataEngine
    from sequoia_x.strategy.high_tight_flag import HighTightFlagStrategy

    frame = flag_frame()
    settings = Settings(db_path=str(tmp_path / "daily.sqlite"), bark_key="fixture")
    engine = DataEngine(settings)
    strategy = HighTightFlagStrategy(engine, settings)
    for on in frame["date"]:
        truncated = frame.loc[frame["date"] <= on].copy()
        with patch.object(engine, "get_local_symbols", return_value=["600001"]):
            with patch.object(engine, "get_ohlcv", return_value=truncated):
                actual = strategy.run()
        observation = observe_flag(frame, on=on)
        assert (observation.formed is True) == bool(actual)


@pytest.mark.parametrize("boundary", ["momentum", "consolidation", "volume"])
def test_strict_formation_thresholds_do_not_accept_equality(boundary: str) -> None:
    frame = flag_frame(40)
    if boundary == "momentum":
        frame.loc[:9, ["open", "high", "low", "close"]] = [11.8, 12, 11.5625, 11.8]
    elif boundary == "consolidation":
        frame.loc[30:, "high"] = 20.125
    else:
        frame.loc[39, "volume"] = 600
    assert observe_flag(frame, on=frame.iloc[-1]["date"]).formed is False


def test_high_level_threshold_accepts_equality_but_not_lower_prices() -> None:
    frame = flag_frame(40)
    frame.loc[10, "high"] = 22
    frame.loc[30:, "low"] = 17.6
    assert observe_flag(frame, on=frame.iloc[-1]["date"]).formed is True
    frame.loc[30:, "low"] = 17.59
    assert observe_flag(frame, on=frame.iloc[-1]["date"]).formed is False


def flag_snapshot(sessions: int = 42):
    from sequoia_x.research.dataset import DatasetSnapshot

    frame = flag_frame(sessions)
    days = list(frame["date"])
    return DatasetSnapshot.model_validate({
        "manifest": {
            "source": "hand-worked flag synthetic fixture", "source_version": "1",
            "universe_id": "flag-fixture", "symbols": ["600001"],
            "start": days[0], "end": days[-1], "as_of": days[-1],
            "evidence_kind": "synthetic", "universe_complete": True,
            "states_complete": True, "actions_complete": True, "calendar_complete": True,
        },
        "calendar": days,
        "bars": [dict(symbol="600001", available_on=row["date"], **row)
                 for row in frame.to_dict(orient="records")],
        "states": [dict(symbol="600001", effective_date=day, available_on=day,
                        listed=True, suspended=False, is_st=False, board="SSE_MAIN",
                        lot_size=100, limit_rule="bounded",
                        limit_up=round(float(frame.iloc[max(i - 1, 0)]["close"]) * 1.1, 2),
                        limit_down=round(float(frame.iloc[max(i - 1, 0)]["close"]) * 0.9, 2))
                   for i, day in enumerate(days)],
    })


def test_replay_preserves_repeated_daily_signals_but_groups_the_same_episode() -> None:
    from sequoia_x.research.flag import replay_flag_signals

    sample = flag_snapshot()
    replay = replay_flag_signals(sample)
    assert len(replay.observations) == 42
    formed = [item for item in replay.observations if item.formed]
    assert len(formed) == 3
    assert [item.is_episode_start for item in formed] == [True, False, False]
    assert len({item.episode_id for item in formed}) == 1
    assert len(replay.signals) == 3
    assert {item.strategy for item in replay.signals} == {"flag"}
    assert [item.signal_id for item in replay.signals] == [item.signal_id for item in formed]
    assert replay.coverage.episode_count == 1
    assert replay.coverage.formed_observations == 3
    assert replay.coverage.repeated_observations == 2
    assert replay.coverage.episode_start_dates == (sample.calendar[39],)
    assert replay.coverage.signal_dates == sample.calendar[39:]


def test_missing_market_data_does_not_restart_an_existing_episode() -> None:
    from sequoia_x.research.dataset import DatasetSnapshot
    from sequoia_x.research.flag import replay_flag_signals

    data = flag_snapshot(84).model_dump()
    for bar in data["bars"]:
        bar["volume"] = 1000
    for i in (39, 80, 81):
        data["bars"][i]["volume"] = 100
    data["bars"][40].update(status="missing", open=None, high=None, low=None, close=None)
    for i in (41, 42):
        data["bars"][i].update(open=10.5, high=11, low=10, close=10.5)
    sample = DatasetSnapshot.model_validate(data)
    replay = replay_flag_signals(sample)
    formed = [item for item in replay.observations if item.formed]
    assert [item.date for item in formed] == [sample.calendar[i] for i in (39, 80, 81)]
    assert [item.is_episode_start for item in formed] == [True, False, False]
    assert len({item.episode_id for item in formed}) == 1
    missing = replay.observations[40]
    assert missing.formed is None
    assert missing.episode_id == formed[0].episode_id
    assert replay.coverage.episode_count == 1
    assert replay.data_status == "data_unqualified"


def test_replay_future_append_does_not_change_past_observations_or_signals() -> None:
    from sequoia_x.research.flag import replay_flag_signals

    sample = flag_snapshot(42)
    past = replay_flag_signals(sample)
    extended = replay_flag_signals(flag_snapshot(80), through=sample.calendar[-1])
    assert extended.observations == past.observations
    assert extended.signals == past.signals
    assert extended.coverage == past.coverage
    assert extended.evidence_kind == "synthetic"
    assert extended.data_status == "data_unqualified"
    assert any("synthetic" in reason for reason in extended.qualification_reasons)


def test_valid_not_formed_observation_ends_an_episode_before_a_new_pattern() -> None:
    from sequoia_x.research.dataset import DatasetSnapshot
    from sequoia_x.research.flag import replay_flag_signals

    data = flag_snapshot(43).model_dump()
    data["bars"][40]["volume"] = 1000
    sample = DatasetSnapshot.model_validate(data)
    replay = replay_flag_signals(sample)
    formed = [item for item in replay.observations if item.formed]
    assert [item.is_episode_start for item in formed] == [True, True, False]
    assert formed[0].episode_id != formed[1].episode_id
    assert formed[1].episode_id == formed[2].episode_id
    assert replay.coverage.episode_count == 2


def test_pre_window_history_warms_features_and_episode_but_cannot_trigger_entry() -> None:
    from sequoia_x.research.dataset import DatasetSnapshot
    from sequoia_x.research.flag import replay_flag_signals

    data = flag_snapshot().model_dump()
    data["manifest"]["start"] = data["calendar"][40]
    sample = DatasetSnapshot.model_validate(data)
    replay = replay_flag_signals(sample)
    assert [item.signal_date for item in replay.signals] == list(sample.calendar[40:])
    assert replay.observations[39].formed is True
    assert replay.observations[39].signal_id is None
    assert replay.observations[39].in_research_window is False
    assert replay.observations[40].episode_id == replay.observations[39].episode_id
    assert replay.observations[40].is_episode_start is False
    assert replay.coverage.formed_observations == 2
    assert replay.coverage.pre_window_observations == 40
    assert replay.coverage.episode_count == 1
    assert replay.coverage.episode_start_dates == (sample.calendar[39],)


def flag_account_inputs(tmp_path, protocol_data, sample, repeat_policy="ignore_while_open"):
    from sequoia_x.research import ResearchProtocol, ResearchRegistry
    from sequoia_x.research.portfolio import PortfolioPolicy
    from sequoia_x.research.replay import ExecutionPolicy, FeeSchedule

    execution = ExecutionPolicy(
        fees=(FeeSchedule(effective_from=date(2020, 1, 1), commission_rate=0,
                          min_commission=5, stamp_tax_rate=0, transfer_fee_rate=0),),
        max_order_quantity=1_000_000, slippage_bps=0,
        max_volume_participation=1, volume_model="daily_volume_upper_bound",
    )
    selection = PortfolioPolicy(daily_entry_slots=2, selection="symbol_order", seed=7)
    protocol = ResearchProtocol.model_validate({
        **protocol_data, "strategy_version": "flag-fixture-v1",
        "data_snapshot_id": sample.snapshot_id, "universe": sample.manifest.universe_id,
        "trading": {
            **protocol_data["trading"], "repeat_signal_policy": repeat_policy,
            "entry_order_time": "next_session_open_limit",
            "fees": [f"execution_policy:{execution.policy_id}"],
            "capacity": f"selection_policy:{selection.policy_id}",
        },
    })
    frozen = ResearchRegistry(tmp_path / "flag-protocol.sqlite").freeze(protocol)
    return frozen, execution, selection


def test_flag_uses_shared_cash_capacity_costs_and_preserves_open_exposure(
    tmp_path, protocol_data,
) -> None:
    from sequoia_x.research.flag import run_flag_portfolio

    sample = flag_snapshot()
    frozen, execution, selection = flag_account_inputs(tmp_path, protocol_data, sample)
    result = run_flag_portfolio(sample, frozen, execution, selection)
    account = result.portfolio
    filled = [item for item in account.fills if item.quantity]
    assert len(filled) == 1
    assert filled[0].requested_quantity == filled[0].submitted_quantity == 500
    assert filled[0].unsubmitted_quantity == 0
    assert filled[0].quantity == 100
    assert filled[0].limit_price == 19.8
    assert filled[0].reserved_cash == 9905
    assert filled[0].released_cash == 8100
    assert filled[0].session == sample.calendar[40]
    assert account.daily_nav[-1].cash == 98195
    assert account.daily_nav[-1].market_value == 1800
    assert account.daily_nav[-1].nav == 99995
    assert account.metrics.total_costs == 5
    assert account.metrics.open_positions == 1
    assert account.metrics.closed_trades == 0
    assert account.metrics.net_win_rate is None
    assert account.metrics.net_expectancy is None
    assert account.metrics.open_exposure == pytest.approx(1800 / 99995)
    assert account.data_qualified is False
    assert account.evidence_kind == "synthetic"
    assert account.execution_policy_id == execution.policy_id
    assert account.selection_policy_id == selection.policy_id
    assert len(result.replay.signals) == len(account.signals) == 3
    assert filled[0].signal_id == result.replay.observations[39].signal_id
    assert account.orders[0].episode_id == result.replay.observations[39].episode_id
    assert result.model_dump(mode="json")["portfolio"]["initial_capital"] == 100000


def test_independent_lots_still_share_cash_and_repeated_episode_identity(
    tmp_path, protocol_data,
) -> None:
    from sequoia_x.research.flag import run_flag_portfolio

    sample = flag_snapshot()
    frozen, execution, selection = flag_account_inputs(
        tmp_path, protocol_data, sample, repeat_policy="independent_signal_lots"
    )
    result = run_flag_portfolio(sample, frozen, execution, selection)
    account = result.portfolio
    filled = [item for item in account.fills if item.quantity]
    assert [item.quantity for item in filled] == [100, 100]
    assert account.daily_nav[-1].cash == 96390
    assert account.daily_nav[-1].market_value == 3600
    assert account.daily_nav[-1].nav == 99990
    assert account.metrics.total_costs == 10
    assert account.coverage.open_lots == 2
    assert account.metrics.net_win_rate is None
    assert result.replay.coverage.episode_count == 1
    assert len({item.episode_id for item in account.trades}) == 1


def test_same_sixty_session_exit_and_costs_produce_a_hand_checkable_closed_trade(
    tmp_path, protocol_data,
) -> None:
    from sequoia_x.research.flag import run_flag_portfolio

    sample = flag_snapshot(102)
    frozen, execution, selection = flag_account_inputs(tmp_path, protocol_data, sample)
    result = run_flag_portfolio(sample, frozen, execution, selection)
    account = result.portfolio
    closed = [item for item in account.trades if item.status == "closed"]
    assert len(closed) == 1
    trade = closed[0]
    assert trade.entry_date == sample.calendar[40]
    assert trade.exit_date == sample.calendar[100]
    assert trade.entry_cash == 1805
    assert trade.exit_cash == 1795
    assert trade.net_pnl == -10
    assert account.daily_nav[-1].cash == account.daily_nav[-1].nav == 99990
    assert account.metrics.closed_trades == account.metrics.losing_trades == 1
    assert account.metrics.winning_trades == 0
    assert account.metrics.net_win_rate == 0
    assert account.metrics.net_expectancy == -10
    assert account.metrics.average_loss == -10
    assert account.metrics.average_win is account.metrics.payoff_ratio is None
    assert account.metrics.total_costs == 10
    assert account.metrics.open_exposure == 0


def test_selected_window_retains_episode_origin_but_has_no_prior_window_orders(
    tmp_path, protocol_data,
) -> None:
    from sequoia_x.research.flag import run_flag_portfolio
    from sequoia_x.research.protocol import DateWindow

    sample = flag_snapshot()
    frozen, execution, selection = flag_account_inputs(tmp_path, protocol_data, sample)
    window = DateWindow(start=sample.calendar[40], end=sample.calendar[41])
    result = run_flag_portfolio(sample, frozen, execution, selection, window=window)
    assert [item.signal_date for item in result.replay.signals] == list(sample.calendar[40:])
    assert result.replay.observations[39].in_research_window is False
    assert result.replay.coverage.episode_start_dates == (sample.calendar[39],)
    assert result.replay.coverage.formed_observations == 2
    assert result.portfolio.signals == result.replay.signals
    assert result.portfolio.fills[0].session == sample.calendar[41]
    assert result.portfolio.daily_nav[-1].nav == 99995


@pytest.mark.parametrize(("state_changes", "execution_reason"), [
    ({}, None),
    ({"is_st": True, "limit_up": 18.9, "limit_down": 17.1},
     "risk_warning_rules_not_supported"),
    ({"board": "UNVERIFIED_BOARD"}, "unsupported_or_unverified_board"),
    ({"limit_rule": "unbounded", "limit_up": None, "limit_down": None},
     "unsupported_unbounded_price_limit"),
], ids=["supported", "st", "unverified-board", "unbounded-limits"])
def test_unsupported_held_flag_position_retains_pre_maturity_risk(
    tmp_path, protocol_data, state_changes: dict[str, object], execution_reason: str | None,
) -> None:
    from sequoia_x.research.dataset import DatasetSnapshot
    from sequoia_x.research.flag import run_flag_portfolio

    sample = flag_snapshot(52)
    data = sample.model_dump()
    for state in data["states"]:
        if state["effective_date"] >= sample.calendar[48]:
            state.update(state_changes)
    sample = DatasetSnapshot.model_validate(data)
    frozen, execution, selection = flag_account_inputs(tmp_path, protocol_data, sample)
    result = run_flag_portfolio(sample, frozen, execution, selection)
    account = result.portfolio

    # 100 shares enter at 18 plus a 5 cash fee; only 12 sessions have elapsed.
    entered = [trade for trade in account.trades if trade.entry_date is not None]
    assert len(entered) == 1
    assert entered[0].entry_date == sample.calendar[40]
    assert entered[0].status == "open"
    assert entered[0].remaining_quantity == 100
    assert entered[0].exit_date is None
    assert all(order.side == "buy" for order in account.orders)
    assert account.daily_nav[-1].cash == 98195
    assert account.daily_nav[-1].market_value == 1800
    assert account.daily_nav[-1].nav == 99995
    assert account.metrics.total_costs == 5
    assert account.metrics.closed_trades == 0
    assert account.metrics.net_win_rate is account.metrics.net_expectancy is None
    assert account.metrics.open_exposure == pytest.approx(1800 / 99995)
    # The 40-session momentum window retains the early low through session 48.
    assert result.replay.coverage.formed_observations == 10
    assert result.replay.coverage.episode_count == 1
    assert account.signals == result.replay.signals

    if execution_reason is None:
        assert account.execution_qualified is True
        assert account.execution_reasons == ()
        assert account.metrics.unexitable_exposure == 0
    else:
        assert account.metrics.unexitable_exposure == pytest.approx(1800 / 99995)
        assert account.execution_qualified is False
        assert any(
            reason.rsplit(":", 1)[-1] == execution_reason
            for reason in account.execution_reasons
        )
