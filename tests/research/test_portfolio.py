"""通过公开组合重放接口核对共享账户经济结果；全部行情为合成样例。"""

from datetime import date, timedelta
from pathlib import Path

import pytest


def _sample(sessions=4, symbols=("AAA", "BBB"), changes=None, actions=()):
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
            states.append({
                "symbol": symbol, "effective_date": session, "available_on": session,
                "listed": True, "suspended": False, "is_st": False, "board": "main",
                "lot_size": 100, "limit_rule": "unbounded",
            })
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
