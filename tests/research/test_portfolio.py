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
                "symbol": symbol,
                "date": session,
                "available_on": session,
                "open": 10,
                "high": 10,
                "low": 10,
                "close": 10,
                "volume": 100000,
                "status": "trading",
                "price_basis": "raw",
            }
            bar.update((changes or {}).get((index, symbol), {}))
            bars.append(bar)
            state = {
                "symbol": symbol,
                "effective_date": session,
                "available_on": session,
                "listed": True,
                "suspended": False,
                "is_st": False,
                "board": "SSE_MAIN",
                "lot_size": 100,
                "limit_rule": "bounded",
                "limit_up": 11,
                "limit_down": 9,
            }
            state.update((state_changes or {}).get((index, symbol), {}))
            states.append(state)
    return DatasetSnapshot.model_validate(
        {
            "manifest": {
                "source": "synthetic-account-example",
                "source_version": "1",
                "universe_id": "synthetic-universe",
                "symbols": symbols,
                "start": calendar[0],
                "end": calendar[-1],
                "as_of": calendar[-1],
                "evidence_kind": "synthetic",
                "universe_complete": True,
                "states_complete": True,
                "actions_complete": True,
                "calendar_complete": True,
            },
            "calendar": calendar,
            "bars": bars,
            "states": states,
            "actions": actions,
        }
    )


def _execution():
    from sequoia_x.research.replay import ExecutionPolicy, FeeSchedule

    return ExecutionPolicy(
        fees=(
            FeeSchedule(
                effective_from=date(2000, 1, 1),
                commission_rate=0,
                min_commission=0,
                stamp_tax_rate=0,
                transfer_fee_rate=0,
            ),
        ),
        max_order_quantity=1000000,
        slippage_bps=0,
        max_volume_participation=1,
        volume_model="daily_volume_upper_bound",
    )


def _frozen(tmp_path, protocol_data, sample, execution, selection, **trading):
    from sequoia_x.research import ResearchProtocol, ResearchRegistry

    protocol_data = {
        **protocol_data,
        "data_snapshot_id": sample.snapshot_id,
        "universe": sample.manifest.universe_id,
        "trading": {
            **protocol_data["trading"],
            "initial_capital": 10000,
            "max_position_fraction": 0.6,
            "max_positions": 2,
            "repeat_signal_policy": "ignore_while_open",
            "entry_order_time": "next_session_open_limit",
            "fees": [f"execution_policy:{execution.policy_id}"],
            "capacity": f"selection_policy:{selection.policy_id}",
            **trading,
        },
    }
    return ResearchRegistry(tmp_path / "protocol.sqlite").freeze(
        ResearchProtocol.model_validate(protocol_data)
    )


def test_empty_signal_days_preserve_shared_cash_and_are_not_evidence(
    tmp_path: Path,
    protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio

    sample = _sample()
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=2, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)

    result = run_portfolio(sample, frozen, execution, selection, signals=())

    assert [row.nav for row in result.daily_nav] == [10000, 10000, 10000, 10000]
    assert [row.cash for row in result.daily_nav] == [10000, 10000, 10000, 10000]
    assert all(
        row.daily_net_return == row.frozen_cash == row.market_value == 0 for row in result.daily_nav
    )
    assert result.orders == result.trades == ()
    assert result.metrics.net_win_rate is None
    assert result.metrics.net_expectancy is None
    assert result.metrics.max_drawdown == 0
    assert result.data_qualified is False
    assert result.evidence_kind == "synthetic"
    assert result.data_snapshot_id == sample.snapshot_id
    assert result.model_dump(mode="json")["initial_capital"] == 10000


def test_simultaneous_candidates_share_cash_and_keep_partial_fill(
    tmp_path: Path,
    protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(
        changes={
            (1, "AAA"): {"high": 11, "close": 11},
            (1, "BBB"): {"low": 9, "close": 9, "volume": 200},
        }
    )
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=2, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)
    signals = tuple(
        Signal(symbol=symbol, signal_date=sample.calendar[0]) for symbol in ("BBB", "AAA")
    )

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)

    first, second = sorted(result.trades, key=lambda trade: trade.symbol)
    assert (first.entry_quantity, second.entry_quantity) == (500, 200)
    assert (first.entry_cash, second.entry_cash) == (5000, 2000)
    assert result.daily_nav[0].frozen_cash == 10000
    assert result.daily_nav[0].available_cash == 0
    assert result.daily_nav[1].cash == 3000
    assert result.daily_nav[1].nav == 10300
    assert result.daily_nav[1].daily_net_return == pytest.approx(0.03)
    assert all(day.cash >= day.frozen_cash >= 0 for day in result.daily_nav)
    assert len(result.fills) == 2
    assert sum(fill.reserved_cash for fill in result.fills) == 9900
    assert result.fills[0].submitted_quantity == 500
    assert result.fills[1].submitted_quantity == 400
    assert result.fills[1].status == "partial"
    assert "expired" in result.orders[1].reason
    assert result.metrics.closed_trades == 0
    assert result.metrics.open_positions == 2
    assert result.signals == signals


def test_sixty_market_sessions_close_one_signal_lot_and_ignore_open_repeat(
    tmp_path: Path,
    protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(
        sessions=64,
        symbols=("AAA",),
        changes={
            (61, "AAA"): {"open": 11, "high": 11, "low": 11, "close": 11},
        },
    )
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)
    signals = (
        Signal(symbol="AAA", signal_date=sample.calendar[0]),
        Signal(symbol="AAA", signal_date=sample.calendar[2]),
    )

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)

    completed = next(trade for trade in result.trades if trade.status == "closed")
    assert completed.entry_date == sample.calendar[1]
    assert completed.exit_date == sample.calendar[61]
    assert completed.net_pnl == 500
    assert completed.net_return == pytest.approx(0.1)
    assert result.daily_nav[60].market_value == 5000
    assert result.daily_nav[61].cash == result.daily_nav[-1].nav == 10500
    assert any(order.reason == "repeat_signal_while_open" for order in result.orders)
    assert result.metrics.closed_trades == result.metrics.winning_trades == 1
    assert result.metrics.net_win_rate == 1
    assert result.metrics.average_win == result.metrics.net_expectancy == 500
    assert result.metrics.average_loss is result.metrics.payoff_ratio is None
    assert result.metrics.portfolio_net_return == pytest.approx(0.05)
    assert result.metrics.open_positions == 0


def test_dividend_receivable_survives_stock_exit_and_split_preserves_value(
    tmp_path: Path,
    protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    changes = {
        (index, "AAA"): {"open": 4.75, "high": 4.75, "low": 4.75, "close": 4.75}
        for index in range(2, 64)
    }
    sample = _sample(
        sessions=64,
        symbols=("AAA",),
        changes=changes,
        actions=(
            {
                "symbol": "AAA",
                "ex_date": date(2024, 1, 3),
                "available_on": date(2024, 1, 1),
                "share_multiplier": 2,
                "cash_per_share": 0.5,
                "pay_date": date(2024, 3, 3),
            },
        ),
        state_changes={
            (index, "AAA"): {"limit_up": 5.23, "limit_down": 4.28} for index in range(2, 64)
        },
    )
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)
    signals = (Signal(symbol="AAA", signal_date=sample.calendar[0]),)

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)

    assert result.daily_nav[2].positions[0].quantity == 1000
    assert result.daily_nav[2].cash == 5000
    assert result.daily_nav[2].market_value == 4750
    assert result.daily_nav[2].receivables_value == 250
    assert result.daily_nav[2].nav == 10000
    assert result.daily_nav[61].cash == 9750
    assert result.daily_nav[61].receivables_value == 250
    assert result.daily_nav[61].nav == 10000
    assert result.daily_nav[62].cash == 10000
    assert result.daily_nav[62].receivables_value == 0
    assert result.trades[0].status == "closed"
    assert result.trades[0].net_pnl == 0
    assert result.metrics.net_win_rate == result.metrics.net_expectancy == 0
    assert result.metrics.breakeven_trades == 1
    assert len(result.corporate_actions) == 2


def test_limit_down_and_suspension_defer_exit_and_keep_unexitable_end_risk(
    tmp_path: Path,
    protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.protocol import DateWindow
    from sequoia_x.research.replay import Signal

    sample = _sample(
        sessions=66,
        symbols=("AAA",),
        changes={
            (61, "AAA"): {"open": 9, "high": 9, "low": 9, "close": 9},
            (62, "AAA"): {"open": 8.1, "high": 8.1, "low": 8.1, "close": 8.1},
            (63, "AAA"): {
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "status": "suspended",
                "volume": 0,
            },
            (64, "AAA"): {"open": 7.3, "high": 7.3, "low": 7.3, "close": 7.3},
        },
        state_changes={
            (61, "AAA"): {"limit_rule": "bounded", "limit_down": 9, "limit_up": 11},
            (62, "AAA"): {"limit_rule": "bounded", "limit_down": 8.1, "limit_up": 9.9},
            (64, "AAA"): {"limit_down": 7.29, "limit_up": 8.91},
            (63, "AAA"): {"suspended": True},
        },
    )
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)
    signals = (Signal(symbol="AAA", signal_date=sample.calendar[0]),)

    unfinished = run_portfolio(
        sample,
        frozen,
        execution,
        selection,
        signals=signals,
        window=DateWindow(start=sample.calendar[0], end=sample.calendar[63]),
    )

    assert unfinished.trades[0].status == "open"
    assert unfinished.trades[0].net_pnl is None
    assert unfinished.daily_nav[-1].market_value == 4050
    assert unfinished.daily_nav[-1].nav == 9050
    assert unfinished.daily_nav[-1].valuation_flags
    assert unfinished.metrics.net_win_rate is None
    assert unfinished.metrics.unexitable_exposure == pytest.approx(4050 / 9050)
    assert len(unfinished.trades[0].exit_reasons) == 3
    assert unfinished.data_qualified is False

    finished = run_portfolio(sample, frozen, execution, selection, signals=signals)

    assert finished.trades[0].exit_date == sample.calendar[64]
    assert finished.trades[0].net_pnl == -1350
    assert finished.daily_nav[-1].nav == 8650
    assert finished.metrics.average_loss == finished.metrics.net_expectancy == -1350
    assert finished.metrics.net_win_rate == 0
    assert finished.metrics.average_win is finished.metrics.payoff_ratio is None
    assert finished.metrics.unexitable_exposure == 0
    assert next(order for order in finished.orders if order.side == "sell").attempt_count == 4


def test_same_auction_sales_cannot_fund_buys_or_free_position_slots(
    tmp_path: Path,
    protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(
        sessions=64,
        changes={
            (61, "AAA"): {"open": 11, "high": 11, "low": 11, "close": 11},
        },
    )
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(
        tmp_path,
        protocol_data,
        sample,
        execution,
        selection,
        max_position_fraction=1,
        max_positions=1,
    )
    signals = (
        Signal(symbol="AAA", signal_date=sample.calendar[0]),
        Signal(symbol="BBB", signal_date=sample.calendar[60]),
        Signal(symbol="BBB", signal_date=sample.calendar[61]),
    )

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)

    assert result.trades[1].entry_date is None
    assert any(order.reason == "max_positions_pre_auction" for order in result.orders)
    assert result.trades[2].entry_date == sample.calendar[62]
    assert result.trades[0].net_pnl == 900
    assert result.trades[2].entry_cash == 9000
    assert result.daily_nav[62].cash == 1900
    assert result.daily_nav[-1].nav == 10900


def test_fee_schedule_changes_use_actual_fill_day_and_account_cash(
    tmp_path: Path,
    protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import ExecutionPolicy, FeeSchedule, Signal

    sample = _sample(
        sessions=64,
        symbols=("AAA",),
        changes={
            (61, "AAA"): {"open": 11, "high": 11, "low": 11, "close": 11},
        },
    )
    execution = ExecutionPolicy(
        fees=(
            FeeSchedule(
                effective_from=date(2000, 1, 1),
                commission_rate=0.001,
                min_commission=5,
                stamp_tax_rate=0,
                transfer_fee_rate=0,
            ),
            FeeSchedule(
                effective_from=sample.calendar[61],
                commission_rate=0.002,
                min_commission=5,
                stamp_tax_rate=0.002,
                transfer_fee_rate=0.001,
            ),
        ),
        max_order_quantity=1000000,
        slippage_bps=0,
        max_volume_participation=1,
        volume_model="daily_volume_upper_bound",
    )
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection, max_position_fraction=1)

    result = run_portfolio(
        sample,
        frozen,
        execution,
        selection,
        signals=(Signal(symbol="AAA", signal_date=sample.calendar[0]),),
    )

    assert result.trades[0].entry_quantity == 900
    assert result.trades[0].entry_cash == 9009
    assert result.trades[0].exit_cash == 9850.5
    assert result.trades[0].net_pnl == 841.5
    assert result.metrics.total_costs == 58.5
    assert result.daily_nav[1].costs == 9
    assert result.daily_nav[61].costs == 49.5
    assert result.daily_nav[-1].cash == 10841.5
    assert result.metrics.portfolio_net_return == pytest.approx(0.08415)


def test_benchmark_uses_identical_calendar_and_explicit_missing_status(
    tmp_path: Path,
    protocol_data: dict,
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
        run_portfolio(
            sample, frozen, execution, selection, signals=(), benchmark={sample.calendar[0]: 100}
        )
    missing = run_portfolio(sample, frozen, execution, selection, signals=())
    assert missing.metrics.benchmark_net_return is None
    assert any("基准" in reason for reason in missing.qualification_reasons)


@pytest.mark.parametrize(
    "slots,max_positions,reason",
    [
        (1, 2, "daily_entry_slots"),
        (2, 1, "max_positions_pre_auction"),
    ],
)
def test_daily_slots_and_security_count_retain_rejected_candidates(
    tmp_path: Path,
    protocol_data: dict,
    slots: int,
    max_positions: int,
    reason: str,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample()
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=slots, selection="symbol_order", seed=7)
    frozen = _frozen(
        tmp_path, protocol_data, sample, execution, selection, max_positions=max_positions
    )
    signals = tuple(
        Signal(symbol=symbol, signal_date=sample.calendar[0]) for symbol in ("AAA", "BBB")
    )

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)

    assert result.metrics.open_positions == 1
    assert result.daily_nav[-1].cash == 5000
    assert len(result.orders) == len(result.trades) == 2
    assert any(order.reason == reason for order in result.orders)


def test_seeded_selection_is_frozen_reproducible_and_input_order_neutral(
    tmp_path: Path,
    protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(symbols=("AAA", "BBB", "CCC"))
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="seeded_shuffle", seed=73)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)
    signals = tuple(
        Signal(symbol=symbol, signal_date=sample.calendar[0]) for symbol in sample.manifest.symbols
    )

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)
    again = run_portfolio(sample, frozen, execution, selection, signals=signals)
    reversed_input = run_portfolio(
        sample, frozen, execution, selection, signals=tuple(reversed(signals))
    )

    assert result.model_dump(mode="json") == again.model_dump(mode="json")
    assert result.orders == reversed_input.orders
    with pytest.raises(ValueError, match="selection_policy"):
        run_portfolio(
            sample,
            frozen,
            execution,
            PortfolioPolicy(daily_entry_slots=1, selection="seeded_shuffle", seed=74),
            signals=signals,
        )


def test_independent_lots_share_symbol_volume_and_allocation(
    tmp_path: Path,
    protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(symbols=("AAA",), changes={(1, "AAA"): {"volume": 200}})
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=2, selection="symbol_order", seed=7)
    frozen = _frozen(
        tmp_path,
        protocol_data,
        sample,
        execution,
        selection,
        repeat_signal_policy="independent_signal_lots",
    )
    signals = (
        Signal(symbol="AAA", signal_date=sample.calendar[0], features={"case": 1}),
        Signal(symbol="AAA", signal_date=sample.calendar[0], features={"case": 2}),
    )

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)

    assert sum(fill.quantity for fill in result.fills) == 200
    assert result.daily_nav[-1].market_value == 2000
    assert result.daily_nav[-1].cash == 8000
    assert result.metrics.open_positions == 1
    assert result.coverage.open_lots == 1
    assert len(result.trades) == 2
    assert all(order.reason != "repeat_signal_while_open" for order in result.orders)


def test_ignore_policy_also_prevents_duplicate_pending_symbol_orders(
    tmp_path: Path,
    protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(symbols=("AAA",))
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=2, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)
    signals = (
        Signal(symbol="AAA", signal_date=sample.calendar[0], features={"case": 1}),
        Signal(symbol="AAA", signal_date=sample.calendar[0], features={"case": 2}),
    )

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)

    assert len(result.fills) == 1
    assert any(order.reason == "repeat_signal_pending_entry" for order in result.orders)
    assert len(result.trades) == 2


def test_suspended_split_and_partial_exit_without_new_close_cannot_create_nav(
    tmp_path: Path,
    protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    start = date(2024, 1, 1)
    sample = _sample(
        sessions=63,
        symbols=("AAA",),
        changes={
            (60, "AAA"): {"status": "suspended", "volume": 0},
            (61, "AAA"): {"open": 4.7, "high": 4.7, "low": 4.7, "close": None, "volume": 200},
            (62, "AAA"): {"status": "suspended", "volume": 0},
        },
        state_changes={
            (60, "AAA"): {"suspended": True},
            (62, "AAA"): {"suspended": True},
            (61, "AAA"): {"limit_up": 5.17, "limit_down": 4.23},
        },
        actions=(
            {
                "symbol": "AAA",
                "ex_date": start + timedelta(days=60),
                "available_on": start,
                "share_multiplier": 2,
                "cash_per_share": 0.5,
                "withholding_tax_per_share": 0.1,
                "pay_date": start + timedelta(days=65),
            },
        ),
    )
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)

    result = run_portfolio(
        sample,
        frozen,
        execution,
        selection,
        signals=(Signal(symbol="AAA", signal_date=sample.calendar[0]),),
    )

    assert result.daily_nav[60].positions[0].quantity == 1000
    assert result.daily_nav[60].market_value == 4700
    assert result.daily_nav[60].receivables_value == 250
    assert result.daily_nav[60].nav == 9950
    assert result.daily_nav[60].positions[0].valuation_status == "unreliable"
    assert result.daily_nav[61].positions[0].quantity == 800
    assert result.daily_nav[61].market_value == result.daily_nav[62].market_value == 3760
    assert result.daily_nav[61].cash == 5940
    assert result.daily_nav[61].nav == result.daily_nav[62].nav == 9950
    assert result.daily_nav[62].valuation_flags
    assert result.metrics.total_costs == 50
    assert result.trades[0].net_pnl is None
    assert result.data_qualified is False


def test_unknown_execution_rule_is_visible_separately_from_data_coverage(
    tmp_path: Path,
    protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(
        symbols=("AAA",),
        state_changes={
            (1, "AAA"): {"limit_rule": "unknown"},
        },
    )
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)

    result = run_portfolio(
        sample,
        frozen,
        execution,
        selection,
        signals=(Signal(symbol="AAA", signal_date=sample.calendar[0]),),
    )

    assert result.fills[0].quantity == 0
    assert result.fills[0].reason == "unknown_price_limit_rule"
    assert result.execution_qualified is False
    assert any("unknown_price_limit_rule" in reason for reason in result.execution_reasons)
    assert result.trades[0].status == "unfilled"
    assert result.daily_nav[-1].nav == 10000


def test_external_signal_on_non_session_is_rejected_without_silent_ledger_loss(
    tmp_path: Path,
    protocol_data: dict,
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
        run_portfolio(
            sample,
            frozen,
            execution,
            selection,
            signals=(Signal(symbol="AAA", signal_date=date(2024, 1, 3)),),
        )


@pytest.mark.parametrize("opening_price", [9, 10, 11])
def test_one_auction_fill_cannot_rewrite_other_submissions_or_slots(
    tmp_path: Path,
    protocol_data: dict,
    opening_price: float,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(
        sessions=2,
        symbols=("AAA", "BBB", "CCC"),
        changes={
            (1, "AAA"): {
                "open": opening_price,
                "high": opening_price,
                "low": opening_price,
                "close": opening_price,
            },
        },
    )
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=3, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection, max_positions=2)
    signals = tuple(
        Signal(symbol=symbol, signal_date=sample.calendar[0]) for symbol in sample.manifest.symbols
    )

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)

    assert [fill.submitted_quantity for fill in result.fills] == [500, 400]
    assert [fill.reserved_cash for fill in result.fills] == [5500, 4400]
    assert result.orders[2].reason == "max_positions_pre_auction"
    assert result.trades[2].entry_date is None
    assert result.daily_nav[-1].nav == 10000
    if opening_price == 11:
        assert result.fills[0].quantity == 0
        assert result.fills[0].released_cash == 5500
    else:
        assert result.fills[0].quantity == 500


@pytest.mark.parametrize("opening_price", [9, 10])
def test_opening_price_cannot_resize_pre_auction_allocation(
    tmp_path: Path,
    protocol_data: dict,
    opening_price: float,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(
        sessions=3,
        changes={
            (2, "AAA"): {
                "open": opening_price,
                "high": opening_price,
                "low": opening_price,
                "close": opening_price,
            },
        },
    )
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(
        tmp_path, protocol_data, sample, execution, selection, max_position_fraction=0.33
    )
    signals = (
        Signal(symbol="AAA", signal_date=sample.calendar[0]),
        Signal(symbol="BBB", signal_date=sample.calendar[1]),
    )

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)

    assert [trade.entry_quantity for trade in result.trades] == [300, 300]
    assert result.fills[1].reserved_cash == 3300
    assert result.daily_nav[-1].cash == 4000
    assert result.daily_nav[-1].nav == 7000 + 300 * opening_price


def test_pay_date_cash_cannot_fund_that_days_opening_buy(
    tmp_path: Path,
    protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(
        sessions=3,
        changes={
            (2, "AAA"): {"open": 9, "high": 9, "low": 9, "close": 9},
        },
        state_changes={(2, "AAA"): {"limit_up": 9.9, "limit_down": 8.1}},
        actions=(
            {
                "symbol": "AAA",
                "available_on": date(2024, 1, 1),
                "ex_date": date(2024, 1, 3),
                "cash_per_share": 1,
                "pay_date": date(2024, 1, 3),
            },
        ),
    )
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)
    signals = (
        Signal(symbol="AAA", signal_date=sample.calendar[0]),
        Signal(symbol="BBB", signal_date=sample.calendar[1]),
    )

    result = run_portfolio(sample, frozen, execution, selection, signals=signals)

    assert result.trades[0].distributions == 500
    assert result.trades[1].entry_quantity == 400
    assert result.fills[1].reserved_cash == 4400
    assert result.daily_nav[-1].cash == 1500
    assert result.daily_nav[-1].receivables_value == 0
    assert result.daily_nav[-1].nav == 10000


def test_pay_date_cash_cannot_pre_fund_auction_sale_fees(
    tmp_path: Path,
    protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import ExecutionPolicy, FeeSchedule, Signal

    sample = _sample(
        sessions=63,
        symbols=("AAA",),
        changes={
            (index, "AAA"): {"open": 9, "high": 9, "low": 9, "close": 9} for index in range(2, 63)
        },
        state_changes={
            (index, "AAA"): {"limit_up": 9.9, "limit_down": 8.1} for index in range(2, 63)
        },
        actions=(
            {
                "symbol": "AAA",
                "available_on": date(2024, 1, 1),
                "ex_date": date(2024, 1, 3),
                "cash_per_share": 1,
                "pay_date": date(2024, 3, 2),
            },
        ),
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
            FeeSchedule(
                effective_from=sample.calendar[61],
                commission_rate=0,
                min_commission=10901,
                stamp_tax_rate=0,
                transfer_fee_rate=0,
            ),
        ),
        max_order_quantity=1000000,
        slippage_bps=0,
        max_volume_participation=1,
        volume_model="daily_volume_upper_bound",
    )
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(
        tmp_path,
        protocol_data,
        sample,
        execution,
        selection,
        initial_capital=11000,
        max_position_fraction=0.1,
    )

    result = run_portfolio(
        sample,
        frozen,
        execution,
        selection,
        signals=(Signal(symbol="AAA", signal_date=sample.calendar[0]),),
    )

    assert result.daily_nav[61].cash == 10100
    assert result.daily_nav[61].positions[0].quantity == 100
    assert result.trades[0].exit_date == sample.calendar[62]
    assert result.trades[0].exit_cash == -10001
    assert result.trades[0].net_pnl == -10901
    assert result.daily_nav[-1].cash == result.daily_nav[-1].nav == 99


@pytest.mark.parametrize(
    "state_change,reason",
    [
        ({"is_st": True, "limit_up": 10.5, "limit_down": 9.5},
         "risk_warning_rules_not_supported"),
        ({"board": "UNKNOWN"}, "unsupported_or_unverified_board"),
        ({"limit_rule": "unbounded", "limit_up": None, "limit_down": None},
         "unsupported_unbounded_price_limit"),
        ({"lot_size": 200}, "unsupported_quantity_unit"),
        ({"limit_rule": "unknown", "limit_up": None, "limit_down": None},
         "unknown_price_limit_rule"),
    ],
)
def test_position_becoming_unsupported_before_time_exit_retains_exit_risk(
    tmp_path: Path,
    protocol_data: dict,
    state_change: dict,
    reason: str,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(
        sessions=10,
        symbols=("AAA",),
        state_changes={(index, "AAA"): state_change for index in (8, 9)},
    )
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(
        tmp_path, protocol_data, sample, execution, selection,
        initial_capital=11000, max_position_fraction=1,
    )

    result = run_portfolio(
        sample, frozen, execution, selection,
        signals=(Signal(symbol="AAA", signal_date=sample.calendar[0]),),
        benchmark={day: 11000 for day in sample.calendar},
    )

    assert result.trades[0].remaining_quantity == 1000
    assert result.trades[0].status == "open"
    assert result.trades[0].net_pnl is None
    assert result.daily_nav[-1].market_value == 10000
    assert result.daily_nav[-1].nav == 11000
    assert result.metrics.unexitable_exposure == pytest.approx(10000 / 11000)
    assert result.execution_qualified is False
    assert any(reason in item for item in result.execution_reasons)
    assert any(reason in item for item in result.qualification_reasons)
    assert len(result.orders) == len(result.fills) == 1
    assert result.fills[0].side == "buy"


@pytest.mark.parametrize(
    "sessions,state_change,bar_change,market,nav,blocked_value",
    [
        (10, {}, {}, 10000, 11000, 0),
        (2, {}, {}, 10000, 11000, 10000),  # Entry-day T+1 remains a modelled restriction.
        (10, {"suspended": True},
         {"status": "suspended", "open": None, "high": None, "low": None,
          "close": None, "volume": 0}, 10000, 11000, 10000),
        (10, {}, {"open": 9, "high": 9, "low": 9, "close": 9}, 9000, 10000, 9000),
        (10, {}, {"volume": 500}, 10000, 11000, 5000),
    ],
)
def test_exit_preview_preserves_modelled_restrictions_and_real_ledger(
    tmp_path: Path,
    protocol_data: dict,
    sessions: int,
    state_change: dict,
    bar_change: dict,
    market: int,
    nav: int,
    blocked_value: int,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(
        sessions=sessions,
        symbols=("AAA",),
        state_changes={(sessions - 1, "AAA"): state_change},
        changes={(sessions - 1, "AAA"): bar_change},
    )
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(
        tmp_path, protocol_data, sample, execution, selection,
        initial_capital=11000, max_position_fraction=1,
    )

    result = run_portfolio(
        sample, frozen, execution, selection,
        signals=(Signal(symbol="AAA", signal_date=sample.calendar[0]),),
        benchmark={day: 11000 for day in sample.calendar},
    )

    assert result.daily_nav[-1].cash == 1000
    assert result.daily_nav[-1].market_value == market
    assert result.daily_nav[-1].nav == nav
    assert result.trades[0].remaining_quantity == 1000
    assert result.metrics.unexitable_exposure == pytest.approx(blocked_value / nav)
    assert result.execution_qualified is True
    assert result.execution_reasons == ()
    assert len(result.orders) == len(result.fills) == 1
    assert result.fills[0].side == "buy"


def test_temporary_unsupported_holding_remains_visible_after_state_recovers(
    tmp_path: Path, protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(
        sessions=10, symbols=("AAA",),
        state_changes={(8, "AAA"): {"board": "UNKNOWN"}},
    )
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(tmp_path, protocol_data, sample, execution, selection)

    result = run_portfolio(
        sample, frozen, execution, selection,
        signals=(Signal(symbol="AAA", signal_date=sample.calendar[0]),),
    )

    assert result.metrics.unexitable_exposure == 0
    assert result.execution_qualified is False
    assert any(
        sample.calendar[8].isoformat() in reason and "unsupported_or_unverified_board" in reason
        for reason in result.execution_reasons
    )
    assert len(result.orders) == len(result.fills) == 1


def test_exit_preview_shares_one_symbol_capacity_across_open_lots(
    tmp_path: Path, protocol_data: dict,
) -> None:
    from sequoia_x.research.portfolio import PortfolioPolicy, run_portfolio
    from sequoia_x.research.replay import Signal

    sample = _sample(
        sessions=6, symbols=("AAA",), changes={(5, "AAA"): {"volume": 1000}},
    )
    execution = _execution()
    selection = PortfolioPolicy(daily_entry_slots=1, selection="symbol_order", seed=7)
    frozen = _frozen(
        tmp_path, protocol_data, sample, execution, selection,
        initial_capital=33000, max_position_fraction=1,
        repeat_signal_policy="independent_signal_lots",
    )

    result = run_portfolio(
        sample, frozen, execution, selection,
        signals=tuple(Signal(symbol="AAA", signal_date=sample.calendar[i]) for i in (0, 2)),
    )

    assert [trade.remaining_quantity for trade in result.trades] == [3000, 200]
    assert result.daily_nav[-1].cash == 1000
    assert result.daily_nav[-1].market_value == 32000
    assert result.daily_nav[-1].nav == 33000
    # Only 1000 of 3200 shares fit the one shared capacity; neither lot is sold.
    assert result.metrics.unexitable_exposure == pytest.approx(22000 / 33000)
    assert result.execution_qualified is True
    assert len(result.orders) == len(result.fills) == 2
    assert all(fill.side == "buy" for fill in result.fills)
