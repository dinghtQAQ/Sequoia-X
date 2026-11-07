"""通过公开组合重放接口核对共享账户经济结果；全部行情为合成样例。"""

from datetime import date, timedelta
from pathlib import Path

import pytest


def _sample(sessions=4, symbols=("AAA", "BBB"), changes=None, actions=(), state_changes=None):
    from sequoia_x.research.dataset import DatasetSnapshot

    calendar = tuple(date(2024, 1, 1) + timedelta(days=index) for index in range(sessions))
    bars = []
    states = []
    for index, session in enumerate(calendar):
        for symbol in symbols:
            bar = {
                "symbol": symbol, "date": session, "available_on": session,
                "open": 10, "high": 10, "low": 10, "close": 10,
                "volume": 100000, "status": "trading", "price_basis": "raw",
            }
            bar.update((changes or {}).get((index, symbol), {}))
            bars.append(bar)
            state = {
                "symbol": symbol, "effective_date": session, "available_on": session,
                "listed": True, "suspended": False, "is_st": False, "board": "main",
                "lot_size": 100, "limit_rule": "unbounded",
            }
            state.update((state_changes or {}).get((index, symbol), {}))
            states.append(state)
    return DatasetSnapshot.model_validate({
        "manifest": {
            "source": "synthetic-account-example", "source_version": "1",
            "universe_id": "synthetic-universe", "symbols": symbols,
            "start": calendar[0], "end": calendar[-1], "as_of": calendar[-1],
            "evidence_kind": "synthetic", "universe_complete": True,
            "states_complete": True, "actions_complete": True, "calendar_complete": True,
        },
        "calendar": calendar, "bars": bars, "states": states, "actions": actions,
    })


def _execution():
    from sequoia_x.research.replay import ExecutionPolicy, FeeSchedule

    return ExecutionPolicy(
        fees=(FeeSchedule(effective_from=date(2000, 1, 1), commission_rate=0,
                          min_commission=0, stamp_tax_rate=0, transfer_fee_rate=0),),
        slippage_bps=0, max_volume_participation=1, volume_model="daily_volume_upper_bound",
    )


def _frozen(tmp_path, protocol_data, sample, execution, selection, **trading):
    from sequoia_x.research import ResearchProtocol, ResearchRegistry

    protocol_data = {
        **protocol_data, "data_snapshot_id": sample.snapshot_id,
        "universe": sample.manifest.universe_id,
        "trading": {
            **protocol_data["trading"], "initial_capital": 10000,
            "max_position_fraction": 0.6, "max_positions": 2,
            "repeat_signal_policy": "ignore_while_open",
            "entry_order_time": "signal_next_session_open_day",
            "fees": [f"execution_policy:{execution.policy_id}"],
            "capacity": f"selection_policy:{selection.policy_id}", **trading,
        },
    }
    return ResearchRegistry(tmp_path / "protocol.sqlite").freeze(
        ResearchProtocol.model_validate(protocol_data)
    )


def test_empty_signal_days_preserve_shared_cash_and_are_not_evidence(
    tmp_path: Path, protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio

    sample = _sample()
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=2, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)

    result = run_portfolio(sample, frozen, execution, selection, signals=())

    assert [row.nav for row in result.daily_nav] == [10000, 10000, 10000, 10000]
    assert [row.cash for row in result.daily_nav] == [10000, 10000, 10000, 10000]
    assert all(row.daily_net_return == row.frozen_cash == row.market_value == 0
               for row in result.daily_nav)
    assert result.orders == result.trades == ()
    assert result.metrics.net_win_rate is None
    assert result.metrics.net_expectancy is None
    assert result.metrics.max_drawdown == 0
    assert result.data_qualified is False
    assert result.evidence_kind == "synthetic"
    assert result.data_snapshot_id == sample.snapshot_id
    assert result.model_dump(mode="json")["initial_capital"] == 10000


def test_simultaneous_candidates_share_cash_and_keep_partial_fill(
    tmp_path: Path, protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(changes={
        (1, "AAA"): {"high": 12, "close": 12},
        (1, "BBB"): {"low": 8, "close": 8},
    })
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=2, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)
    signals = tuple(Signal(symbol=symbol, signal_date=sample.calendar[0])
                    for symbol in ("BBB", "AAA"))

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)

    first, second = sorted(result.trades, key=lambda trade: trade.symbol)
    assert (first.entry_quantity, second.entry_quantity) == (600, 400)
    assert (first.entry_cash, second.entry_cash) == (6000, 4000)
    assert result.daily_nav[0].frozen_cash == 10000
    assert result.daily_nav[0].available_cash == 0
    assert result.daily_nav[1].cash == 0
    assert result.daily_nav[1].nav == 10400
    assert result.daily_nav[1].daily_net_return == pytest.approx(0.04)
    assert all(day.cash >= day.frozen_cash >= 0 for day in result.daily_nav)
    assert len(result.fills) == 2
    assert result.fills[1].status == "partial"
    assert "expired" in result.orders[1].reason
    assert result.metrics.closed_trades == 0
    assert result.metrics.open_positions == 2
    assert result.signals == signals


def test_sixty_market_sessions_close_one_signal_lot_and_ignore_open_repeat(
    tmp_path: Path, protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(sessions=64, symbols=("AAA",), changes={
        (61, "AAA"): {"open": 12, "high": 12, "low": 12, "close": 12},
    })
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)
    signals = (Signal(symbol="AAA", signal_date=sample.calendar[0]),
               Signal(symbol="AAA", signal_date=sample.calendar[2]))

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)

    completed = next(trade for trade in result.trades if trade.status == "closed")
    assert completed.entry_date == sample.calendar[1]
    assert completed.exit_date == sample.calendar[61]
    assert completed.net_pnl == 1200
    assert completed.net_return == pytest.approx(0.2)
    assert result.daily_nav[60].market_value == 6000
    assert result.daily_nav[61].cash == result.daily_nav[-1].nav == 11200
    assert any(order.reason == "repeat_signal_while_open" for order in result.orders)
    assert result.metrics.closed_trades == result.metrics.winning_trades == 1
    assert result.metrics.net_win_rate == 1
    assert result.metrics.average_win == result.metrics.net_expectancy == 1200
    assert result.metrics.average_loss is result.metrics.payoff_ratio is None
    assert result.metrics.portfolio_net_return == pytest.approx(0.12)
    assert result.metrics.open_positions == 0


def test_dividend_receivable_survives_stock_exit_and_split_preserves_value(
    tmp_path: Path, protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    changes = {(index, "AAA"): {"open": 4.75, "high": 4.75, "low": 4.75, "close": 4.75}
               for index in range(2, 64)}
    sample = _sample(sessions=64, symbols=("AAA",), changes=changes, actions=({
        "symbol": "AAA", "ex_date": date(2024, 1, 3), "available_on": date(2024, 1, 1),
        "share_multiplier": 2, "cash_per_share": 0.5, "pay_date": date(2024, 3, 3),
    },))
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)
    signals = (Signal(symbol="AAA", signal_date=sample.calendar[0]),)

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)

    assert result.daily_nav[2].positions[0].quantity == 1200
    assert result.daily_nav[2].cash == 4000
    assert result.daily_nav[2].market_value == 5700
    assert result.daily_nav[2].receivables_value == 300
    assert result.daily_nav[2].nav == 10000
    assert result.daily_nav[61].cash == 9700
    assert result.daily_nav[61].receivables_value == 300
    assert result.daily_nav[61].nav == 10000
    assert result.daily_nav[62].cash == 10000
    assert result.daily_nav[62].receivables_value == 0
    assert result.trades[0].status == "closed"
    assert result.trades[0].net_pnl == 0
    assert result.metrics.net_win_rate == result.metrics.net_expectancy == 0
    assert result.metrics.breakeven_trades == 1
    assert len(result.corporate_actions) == 2


def test_limit_down_and_suspension_defer_exit_and_keep_unexitable_end_risk(
    tmp_path: Path, protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.protocol import DateWindow
    from sequoia_x.research.replay import Signal

    sample = _sample(sessions=66, symbols=("AAA",), changes={
        (61, "AAA"): {"open": 8, "high": 8, "low": 8, "close": 8},
        (62, "AAA"): {"open": 7.2, "high": 7.2, "low": 7.2, "close": 7.2},
        (63, "AAA"): {"open": None, "high": None, "low": None, "close": None,
                        "status": "suspended", "volume": 0},
        (64, "AAA"): {"open": 6, "high": 6, "low": 6, "close": 6},
    }, state_changes={
        (61, "AAA"): {"limit_rule": "bounded", "limit_down": 8, "limit_up": 12},
        (62, "AAA"): {"limit_rule": "bounded", "limit_down": 7.2, "limit_up": 8.8},
        (63, "AAA"): {"suspended": True},
    })
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)
    signals = (Signal(symbol="AAA", signal_date=sample.calendar[0]),)

    unfinished = run_portfolio(sample, frozen, execution, selection, signals=signals,
        window=DateWindow(start=sample.calendar[0], end=sample.calendar[63]))

    assert unfinished.trades[0].status == "open"
    assert unfinished.trades[0].net_pnl is None
    assert unfinished.daily_nav[-1].market_value == 4320
    assert unfinished.daily_nav[-1].nav == 8320
    assert unfinished.daily_nav[-1].valuation_flags
    assert unfinished.metrics.net_win_rate is None
    assert unfinished.metrics.unexitable_exposure == pytest.approx(4320 / 8320)
    assert len(unfinished.trades[0].exit_reasons) == 3
    assert unfinished.data_qualified is False

    finished = run_portfolio(sample, frozen, execution, selection, signals=signals)

    assert finished.trades[0].exit_date == sample.calendar[64]
    assert finished.trades[0].net_pnl == -2400
    assert finished.daily_nav[-1].nav == 7600
    assert finished.metrics.average_loss == finished.metrics.net_expectancy == -2400
    assert finished.metrics.net_win_rate == 0
    assert finished.metrics.average_win is finished.metrics.payoff_ratio is None
    assert finished.metrics.unexitable_exposure == 0
    assert next(order for order in finished.orders if order.side == "sell").attempt_count == 4


def test_sale_proceeds_fund_same_session_candidate_without_pre_spending(
    tmp_path: Path, protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(sessions=64, changes={
        (61, "AAA"): {"open": 12, "high": 12, "low": 12, "close": 12},
    })
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection,
                     max_position_fraction=1, max_positions=1)
    signals = (Signal(symbol="AAA", signal_date=sample.calendar[0]),
               Signal(symbol="BBB", signal_date=sample.calendar[60]))

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)

    assert result.daily_nav[60].cash == result.daily_nav[60].frozen_cash == 0
    assert result.trades[0].net_pnl == 2000
    assert result.trades[1].entry_date == sample.calendar[61]
    assert result.trades[1].entry_cash == 10000
    assert result.daily_nav[61].cash == 2000
    assert result.daily_nav[61].nav == 12000
    assert [(fill.symbol, fill.side) for fill in result.fills] == [
        ("AAA", "buy"), ("AAA", "sell"), ("BBB", "buy"),
    ]


def test_fee_schedule_changes_use_actual_fill_day_and_account_cash(
    tmp_path: Path, protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import ExecutionPolicy, FeeSchedule, Signal

    sample = _sample(sessions=64, symbols=("AAA",), changes={
        (61, "AAA"): {"open": 12, "high": 12, "low": 12, "close": 12},
    })
    execution = ExecutionPolicy(fees=(
        FeeSchedule(effective_from=date(2000, 1, 1), commission_rate=0.001,
                    min_commission=5, stamp_tax_rate=0, transfer_fee_rate=0),
        FeeSchedule(effective_from=sample.calendar[61], commission_rate=0.002,
                    min_commission=5, stamp_tax_rate=0.002, transfer_fee_rate=0.001),
    ), slippage_bps=0, max_volume_participation=1, volume_model="daily_volume_upper_bound")
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection,
                     max_position_fraction=1)

    result = run_portfolio(sample, frozen, execution, selection,
        signals=(Signal(symbol="AAA", signal_date=sample.calendar[0]),))

    assert result.trades[0].entry_quantity == 900
    assert result.trades[0].entry_cash == 9009
    assert result.trades[0].exit_cash == 10746
    assert result.trades[0].net_pnl == 1737
    assert result.metrics.total_costs == 63
    assert result.daily_nav[1].costs == 9
    assert result.daily_nav[61].costs == 54
    assert result.daily_nav[-1].cash == 11737
    assert result.metrics.portfolio_net_return == pytest.approx(0.1737)


def test_benchmark_uses_identical_calendar_and_explicit_missing_status(
    tmp_path: Path, protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio

    sample = _sample()
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)
    levels = dict(zip(sample.calendar, (100, 101, 99, 102)))

    result = run_portfolio(sample, frozen, execution, selection, signals=(), benchmark=levels)

    assert [day.date for day in result.benchmark] == [day.date for day in result.daily_nav]
    assert [day.nav for day in result.benchmark] == [10000, 10100, 9900, 10200]
    assert result.benchmark[0].daily_net_return == 0
    assert result.benchmark[1].daily_net_return == pytest.approx(0.01)
    assert result.metrics.benchmark_net_return == pytest.approx(0.02)
    assert result.metrics.portfolio_net_return == 0
    with pytest.raises(ValueError, match="基准"):
        run_portfolio(sample, frozen, execution, selection, signals=(),
                      benchmark={sample.calendar[0]: 100})
    missing = run_portfolio(sample, frozen, execution, selection, signals=())
    assert missing.metrics.benchmark_net_return is None
    assert any("基准" in reason for reason in missing.qualification_reasons)


@pytest.mark.parametrize("slots,max_positions,reason", [
    (1, 2, "daily_entry_slots"), (2, 1, "max_positions"),
])
def test_daily_slots_and_security_count_retain_rejected_candidates(
    tmp_path: Path, protocol_data: dict, slots: int, max_positions: int, reason: str,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample()
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=slots, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection,
                     max_positions=max_positions)
    signals = tuple(Signal(symbol=symbol, signal_date=sample.calendar[0])
                    for symbol in ("AAA", "BBB"))

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)

    assert result.metrics.open_positions == 1
    assert result.daily_nav[-1].cash == 4000
    assert len(result.orders) == len(result.trades) == 2
    assert any(order.reason == reason for order in result.orders)


def test_seeded_selection_is_frozen_reproducible_and_input_order_neutral(
    tmp_path: Path, protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(symbols=("AAA", "BBB", "CCC"))
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="seeded_shuffle", seed=73)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)
    signals = tuple(Signal(symbol=symbol, signal_date=sample.calendar[0])
                    for symbol in sample.manifest.symbols)

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)
    again = run_portfolio(sample, frozen, execution, selection, signals=signals)
    reversed_input = run_portfolio(sample, frozen, execution, selection,
                                   signals=tuple(reversed(signals)))

    assert result.model_dump(mode="json") == again.model_dump(mode="json")
    assert result.orders == reversed_input.orders
    with pytest.raises(ValueError, match="selection_policy"):
        run_portfolio(sample, frozen, execution,
                      PortfolioPolicy(daily_entry_slots=1, selection="seeded_shuffle", seed=74),
                      signals=signals)


def test_independent_lots_share_symbol_volume_and_allocation(
    tmp_path: Path, protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(symbols=("AAA",), changes={(1, "AAA"): {"volume": 200}})
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=2, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection,
                     repeat_signal_policy="independent_signal_lots")
    signals = (Signal(symbol="AAA", signal_date=sample.calendar[0], features={"case": 1}),
               Signal(symbol="AAA", signal_date=sample.calendar[0], features={"case": 2}))

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)

    assert sum(fill.quantity for fill in result.fills) == 200
    assert result.daily_nav[-1].market_value == 2000
    assert result.daily_nav[-1].cash == 8000
    assert result.metrics.open_positions == 1
    assert result.coverage.open_lots == 1
    assert len(result.trades) == 2
    assert all(order.reason != "repeat_signal_while_open" for order in result.orders)


def test_ignore_policy_also_prevents_duplicate_pending_symbol_orders(
    tmp_path: Path, protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(symbols=("AAA",))
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=2, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)
    signals = (Signal(symbol="AAA", signal_date=sample.calendar[0], features={"case": 1}),
               Signal(symbol="AAA", signal_date=sample.calendar[0], features={"case": 2}))

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)

    assert len(result.fills) == 1
    assert any(order.reason == "repeat_signal_pending_entry" for order in result.orders)
    assert len(result.trades) == 2


def test_suspended_split_and_partial_exit_without_new_close_cannot_create_nav(
    tmp_path: Path, protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    start = date(2024, 1, 1)
    sample = _sample(sessions=63, symbols=("AAA",), changes={
        (60, "AAA"): {"status": "suspended", "volume": 0},
        (61, "AAA"): {"open": 4.7, "high": 4.7, "low": 4.7,
                        "close": None, "volume": 200},
        (62, "AAA"): {"status": "suspended", "volume": 0},
    }, state_changes={
        (60, "AAA"): {"suspended": True}, (62, "AAA"): {"suspended": True},
    }, actions=({
        "symbol": "AAA", "ex_date": start + timedelta(days=60), "available_on": start,
        "share_multiplier": 2, "cash_per_share": 0.5, "withholding_tax_per_share": 0.1,
        "pay_date": start + timedelta(days=65),
    },))
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)

    result = run_portfolio(sample, frozen, execution, selection,
        signals=(Signal(symbol="AAA", signal_date=sample.calendar[0]),))

    assert result.daily_nav[60].positions[0].quantity == 1200
    assert result.daily_nav[60].market_value == 5640
    assert result.daily_nav[60].receivables_value == 300
    assert result.daily_nav[60].nav == 9940
    assert result.daily_nav[60].positions[0].valuation_status == "unreliable"
    assert result.daily_nav[61].positions[0].quantity == 1000
    assert result.daily_nav[61].market_value == result.daily_nav[62].market_value == 4700
    assert result.daily_nav[61].cash == 4940
    assert result.daily_nav[61].nav == result.daily_nav[62].nav == 9940
    assert result.daily_nav[62].valuation_flags
    assert result.metrics.total_costs == 60
    assert result.trades[0].net_pnl is None
    assert result.data_qualified is False


def test_unknown_execution_rule_is_visible_separately_from_data_coverage(
    tmp_path: Path, protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(symbols=("AAA",), state_changes={
        (1, "AAA"): {"limit_rule": "unknown"},
    })
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)

    result = run_portfolio(sample, frozen, execution, selection,
        signals=(Signal(symbol="AAA", signal_date=sample.calendar[0]),))

    assert result.fills[0].quantity == 0
    assert result.fills[0].reason == "unknown_price_limit_rule"
    assert result.execution_qualified is False
    assert any("unknown_price_limit_rule" in reason for reason in result.execution_reasons)
    assert result.trades[0].status == "unfilled"
    assert result.daily_nav[-1].nav == 10000


def test_external_signal_on_non_session_is_rejected_without_silent_ledger_loss(
    tmp_path: Path, protocol_data: dict,
) -> None:
    from sequoia_x.research.dataset import DatasetSnapshot
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    original = _sample()
    payload = original.model_dump()
    payload["calendar"] = tuple(day for day in original.calendar if day != date(2024, 1, 3))
    sample = DatasetSnapshot.model_validate(payload)
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)

    with pytest.raises(ValueError, match="交易日"):
        run_portfolio(sample, frozen, execution, selection,
            signals=(Signal(symbol="AAA", signal_date=date(2024, 1, 3)),))
