"""合成的可手算成交样例，不代表真实策略收益。"""

from datetime import date

import pytest

from sequoia_x.research.dataset import DailyBar, SecurityState
from sequoia_x.research.replay import ExecutionPolicy, FeeSchedule, quote_open_fill


def policy(**changes):
    values = dict(
        fees=(
            FeeSchedule(
                effective_from=date(2020, 1, 1),
                commission_rate=0.0003,
                min_commission=5,
                stamp_tax_rate=0.001,
                transfer_fee_rate=0.0001,
            ),
        ),
        slippage_bps=100,
        max_volume_participation=0.01,
        volume_model="daily_volume_upper_bound",
    )
    values.update(changes)
    return ExecutionPolicy(**values)


def bar(on=date(2025, 1, 6), **changes):
    values = dict(
        symbol="000001", date=on, available_on=on, open=10, high=12, low=9, close=11, volume=100000
    )
    values.update(changes)
    return DailyBar(**values)


def state(on=date(2025, 1, 6), **changes):
    values = dict(
        symbol="000001",
        effective_date=on,
        available_on=on,
        listed=True,
        suspended=False,
        is_st=False,
        board="main",
        lot_size=100,
        limit_rule="bounded",
        limit_up=12,
        limit_down=8,
    )
    values.update(changes)
    return SecurityState(**values)


def test_open_fill_accounts_for_gap_slippage_fees_and_cent_precision():
    bought = quote_open_fill(bar(), state(), "buy", 100, 2000, policy())
    assert bought.status == "filled"
    assert bought.quantity == 100
    assert bought.market_price == 10
    assert bought.price == 10.1
    assert bought.gross == 1010
    assert bought.commission == 5
    assert bought.transfer_fee == 0.1
    assert bought.cash_delta == -1015.1
    assert bought.slippage_cost == 10

    sold = quote_open_fill(
        bar(open=11), state(), "sell", 100, 0, policy(), acquired_on=date(2025, 1, 3)
    )
    assert sold.price == 10.89
    assert sold.cash_delta == 1082.8
    assert sold.stamp_tax == 1.09
    assert sold.transfer_fee == 0.11
    assert bought.cash_delta + sold.cash_delta == pytest.approx(67.7)


def sample(days, *, bars=None, actions=(), states=None):
    from sequoia_x.research.dataset import DatasetManifest, DatasetSnapshot

    return DatasetSnapshot(
        manifest=DatasetManifest(
            source="hand-worked fixture",
            source_version="v1",
            universe_id="synthetic-single",
            symbols=("000001",),
            start=days[0],
            end=days[-1],
            as_of=days[-1],
            evidence_kind="synthetic",
            universe_complete=True,
            states_complete=True,
            actions_complete=True,
            calendar_complete=True,
        ),
        calendar=tuple(days),
        bars=tuple(bar(day) for day in days) if bars is None else tuple(bars),
        states=tuple(state(day) for day in days) if states is None else tuple(states),
        actions=tuple(actions),
    )


def test_corporate_action_registers_prior_shares_then_pays_after_sale():
    from sequoia_x.research.dataset import CorporateAction
    from sequoia_x.research.replay import apply_corporate_actions

    ex_day, pay_day = date(2025, 1, 7), date(2025, 1, 9)
    snapshot = sample(
        [date(2025, 1, 6), ex_day, date(2025, 1, 8), pay_day],
        actions=[
            CorporateAction(
                symbol="000001",
                ex_date=ex_day,
                available_on=date(2025, 1, 6),
                cash_per_share=1,
                withholding_tax_per_share=0.2,
                share_multiplier=2,
                pay_date=pay_day,
            ),
        ],
    )
    ex_result = apply_corporate_actions(snapshot, "000001", ex_day, 100)
    assert ex_result.quantity == 200
    assert ex_result.distribution_accrual == 100
    assert ex_result.withholding_tax == 20
    assert ex_result.cash_delta == 0
    assert ex_result.receivables[0].amount == 100
    paid = apply_corporate_actions(snapshot, "000001", pay_day, 0, ex_result.receivables)
    assert paid.quantity == 0
    assert paid.cash_delta == 100
    assert paid.distribution_accrual == 0
    assert paid.receivables == ()
    newcomer = apply_corporate_actions(snapshot, "000001", ex_day, 0)
    assert newcomer.quantity == 0
    assert newcomer.receivables == ()


def test_missing_price_keeps_last_market_mark_but_split_without_price_is_unreliable():
    from sequoia_x.research.dataset import CorporateAction
    from sequoia_x.research.replay import mark_position

    first, last = date(2025, 1, 6), date(2025, 1, 7)
    missing_bar = bar(
        last, open=None, high=None, low=None, close=None, volume=None, status="missing"
    )
    snapshot = sample([first, last], bars=[bar(first, close=11), missing_bar])
    mark = mark_position(snapshot, "000001", last, 100)
    assert mark.value == 1100
    assert mark.status == "stale"
    assert mark.as_of == first
    changed = snapshot.model_copy(
        update={
            "actions": (
                CorporateAction(
                    symbol="000001", ex_date=last, available_on=first, share_multiplier=2
                ),
            )
        }
    )
    mark = mark_position(changed, "000001", last, 200)
    assert mark.status == "unreliable"
    assert mark.value is None


def sessions(count):
    from datetime import timedelta

    days = []
    day = date(2025, 1, 2)
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def test_ma_replay_matches_restored_daily_strategy_and_future_data_cannot_rewrite():
    import pandas as pd

    from sequoia_x.core.config import Settings
    from sequoia_x.research.replay import replay_ma_signals, signal_frame
    from sequoia_x.strategy.ma_volume import MaVolumeStrategy

    days = sessions(25)
    closes = [10] * 16 + [9] * 5 + [16] + [20] * 3
    bars = [
        bar(
            day,
            open=value,
            high=value + 1,
            low=value - 1,
            close=value,
            volume=300 if index == 21 else 100,
        )
        for index, (day, value) in enumerate(zip(days, closes))
    ]
    snapshot = sample(days, bars=bars)
    signals = replay_ma_signals(snapshot, through=days[21])
    assert [(item.symbol, item.signal_date) for item in signals] == [("000001", days[21])]
    full_signals = replay_ma_signals(snapshot)
    assert [item.signal_id for item in full_signals if item.signal_date <= days[21]] == [
        item.signal_id for item in signals
    ]

    class DailySource:
        def get_local_symbols(self):
            return ["000001"]

        def get_ohlcv(self, symbol):
            return pd.DataFrame(
                [
                    dict(date=day, close=value, volume=(300 if i == 21 else 100))
                    for i, (day, value) in enumerate(zip(days, closes))
                    if day <= cutoff
                ]
            )

    for cutoff in days[:22]:
        restored = MaVolumeStrategy(DailySource(), Settings(bark_key="test", _env_file=None)).run()
        replayed = [
            item.symbol
            for item in replay_ma_signals(snapshot, through=cutoff)
            if item.signal_date == cutoff
        ]
        assert restored == replayed
    assert list(signal_frame(snapshot, "000001", days[0]).columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


def frozen_for(snapshot, numeric_policy, protocol_data):
    from datetime import datetime

    from sequoia_x.research.protocol import FrozenProtocol, ResearchProtocol

    values = dict(protocol_data)
    values["data_snapshot_id"] = snapshot.snapshot_id
    values["trading"] = dict(
        values["trading"],
        initial_capital=10000,
        max_position_fraction=1,
        fees=[f"execution_policy:{numeric_policy.policy_id}"],
        entry_order_time="signal_next_session_open_day",
    )
    return FrozenProtocol(
        protocol_id="test-frozen",
        revision=1,
        parent_id=None,
        frozen_at=datetime.fromisoformat("2024-01-01T00:00:00+00:00"),
        protocol=ResearchProtocol.model_validate(values),
    )


def zero_cost_policy():
    return policy(
        fees=(
            FeeSchedule(
                effective_from=date(2020, 1, 1),
                commission_rate=0,
                min_commission=0,
                stamp_tax_rate=0,
                transfer_fee_rate=0,
            ),
        ),
        slippage_bps=0,
    )


def test_trade_holds_60_market_sessions_and_preserves_unfinished_exposure(protocol_data):
    from sequoia_x.research.replay import Signal, replay_trade

    days = sessions(63)
    bars = [
        bar(day, open=(12 if i >= 61 else 10), high=13, low=9, close=(12 if i >= 61 else 10))
        for i, day in enumerate(days)
    ]
    snapshot = sample(days, bars=bars)
    numeric_policy = zero_cost_policy()
    frozen = frozen_for(snapshot, numeric_policy, protocol_data)
    signal = Signal(symbol="000001", signal_date=days[0])
    unfinished = replay_trade(snapshot, frozen, signal, 100, numeric_policy, through=days[60])
    assert unfinished.entry_date == days[1]
    assert unfinished.status == "open"
    assert unfinished.quantity == 100
    assert unfinished.net_pnl is None
    assert unfinished.daily[-1].market_value == 1000
    assert unfinished.holding_sessions == 60
    closed = replay_trade(snapshot, frozen, signal, 100, numeric_policy)
    assert closed.exit_date == days[61]
    assert closed.status == "closed"
    assert closed.entry_cash == 1000
    assert closed.exit_cash == 1200
    assert closed.net_pnl == 200
    assert closed.daily[-1].nav == 10200
    assert closed.evidence_kind == "synthetic"
    assert closed.quality_status == "data_unqualified"


@pytest.mark.parametrize(
    ("change_bar", "change_state", "side", "acquired", "reason"),
    [
        ({}, {}, "sell", date(2025, 1, 6), "t_plus_one"),
        ({"open": 12}, {}, "buy", None, "open_at_limit_up"),
        ({"open": 8, "low": 8}, {}, "sell", date(2025, 1, 3), "open_at_limit_down"),
        ({"price_basis": "back_adjusted"}, {}, "buy", None, "execution_requires_raw_price"),
        ({}, {"limit_rule": "unknown"}, "buy", None, "unknown_price_limit_rule"),
        (
            {"available_on": date(2025, 1, 7)},
            {},
            "buy",
            None,
            "execution_bar_not_available_on_session",
        ),
    ],
)
def test_illegal_or_unverifiable_open_orders_never_create_a_fill(
    change_bar,
    change_state,
    side,
    acquired,
    reason,
):
    result = quote_open_fill(
        bar(**change_bar),
        state(**change_state),
        side,
        100,
        2000,
        zero_cost_policy(),
        acquired_on=acquired,
    )
    assert result.quantity == 0
    assert result.cash_delta == 0
    assert result.reason == reason


def test_daily_volume_is_only_a_cap_and_fees_cannot_overdraw_cash():
    numeric_policy = zero_cost_policy()
    partial = quote_open_fill(bar(volume=15000), state(), "buy", 300, 10000, numeric_policy)
    assert partial.status == "partial"
    assert partial.quantity == 100
    assert partial.cash_delta == -1000
    unavailable = quote_open_fill(bar(), state(), "buy", 100, 999.99, numeric_policy)
    assert unavailable.quantity == 0
    minimum = quote_open_fill(bar(), state(), "buy", 100, 1004.99, policy(slippage_bps=0))
    assert minimum.quantity == 0
    used = quote_open_fill(
        bar(volume=15000), state(), "buy", 100, 10000, numeric_policy, used_volume=100
    )
    assert used.quantity == 0


def test_fee_schedule_changes_on_the_historical_effective_date():
    old, new = date(2025, 1, 6), date(2025, 1, 7)
    numeric_policy = zero_cost_policy().model_copy(
        update={
            "fees": (
                FeeSchedule(
                    effective_from=old,
                    commission_rate=0,
                    min_commission=0,
                    stamp_tax_rate=0.001,
                    transfer_fee_rate=0,
                ),
                FeeSchedule(
                    effective_from=new,
                    commission_rate=0,
                    min_commission=0,
                    stamp_tax_rate=0.0005,
                    transfer_fee_rate=0,
                ),
            )
        }
    )
    first = quote_open_fill(
        bar(old), state(old), "sell", 100, 0, numeric_policy, acquired_on=date(2025, 1, 3)
    )
    second = quote_open_fill(bar(new), state(new), "sell", 100, 0, numeric_policy, acquired_on=old)
    assert first.stamp_tax == 1
    assert second.stamp_tax == 0.5
    assert second.cash_delta == 999.5
    early = quote_open_fill(
        bar(date(2025, 1, 3)), state(date(2025, 1, 3)), "buy", 100, 2000, numeric_policy
    )
    assert early.reason == "missing_fee_schedule"


def test_full_replay_reconciles_split_net_dividend_tax_and_later_payment(protocol_data):
    from sequoia_x.research.dataset import CorporateAction
    from sequoia_x.research.replay import Signal, replay_trade

    days = sessions(65)
    ex_date, pay_date = days[30], days[62]
    bars = [
        bar(day, open=(4.4 if i >= 30 else 10), high=12, low=4, close=(4.4 if i >= 30 else 10))
        for i, day in enumerate(days)
    ]
    snapshot = sample(
        days,
        bars=bars,
        states=[state(day, limit_down=3) for day in days],
        actions=[
            CorporateAction(
                symbol="000001",
                ex_date=ex_date,
                available_on=days[25],
                cash_per_share=1,
                withholding_tax_per_share=0.2,
                share_multiplier=2,
                pay_date=pay_date,
            ),
        ],
    )
    numeric_policy = zero_cost_policy()
    result = replay_trade(
        snapshot,
        frozen_for(snapshot, numeric_policy, protocol_data),
        Signal(symbol="000001", signal_date=days[0]),
        100,
        numeric_policy,
    )
    assert result.status == "closed"
    assert result.entry_cash == 1000
    assert result.exit_cash == 880
    assert result.distributions_accrued == 100
    assert result.distributions_paid == 100
    assert result.withholding_tax == 20
    assert result.total_costs == 20
    assert result.net_pnl == -20
    assert result.daily[30].quantity == 200
    assert result.daily[30].market_value == 880
    assert result.daily[30].receivables_value == 100
    assert result.daily[30].nav == 9980
    assert result.daily[61].cash == 9880
    assert result.daily[61].receivables_value == 100
    assert result.daily[62].cash == 9980
    assert result.daily[62].receivables_value == 0


def test_incomplete_action_coverage_blocks_trusted_complete_trade_pnl(protocol_data):
    from sequoia_x.research.replay import Signal, replay_trade

    days = sessions(63)
    snapshot = sample(days)
    snapshot = snapshot.model_copy(
        update={
            "manifest": snapshot.manifest.model_copy(
                update={"actions_complete": False},
            )
        }
    )
    numeric_policy = zero_cost_policy()
    result = replay_trade(
        snapshot,
        frozen_for(snapshot, numeric_policy, protocol_data),
        Signal(symbol="000001", signal_date=days[0]),
        100,
        numeric_policy,
    )
    assert result.status == "closed"
    assert result.net_pnl is None
    assert any("公司行动" in reason for reason in result.quality_reasons)


def test_suspension_counts_toward_60_sessions_and_limit_down_defers_exit(protocol_data):
    from sequoia_x.research.replay import Signal, replay_trade

    days = sessions(64)
    bars = [bar(day) for day in days]
    states = [state(day) for day in days]
    bars[10] = bar(
        days[10], open=None, high=None, low=None, close=None, volume=0, status="suspended"
    )
    states[10] = state(days[10], suspended=True)
    bars[61] = bar(days[61], open=8, high=8, low=8, close=8)
    snapshot = sample(days, bars=bars, states=states)
    numeric_policy = zero_cost_policy()
    frozen = frozen_for(snapshot, numeric_policy, protocol_data)
    signal = Signal(symbol="000001", signal_date=days[0])
    blocked = replay_trade(snapshot, frozen, signal, 100, numeric_policy, through=days[61])
    assert blocked.status == "exit_blocked"
    assert blocked.quantity == 100
    assert blocked.daily[-1].market_value == 800
    assert blocked.net_pnl is None
    assert blocked.orders[-1].session == days[61]
    assert blocked.orders[-1].reason == "open_at_limit_down"
    completed = replay_trade(snapshot, frozen, signal, 100, numeric_policy)
    assert completed.exit_date == days[62]
    assert completed.net_pnl == 0


def test_entry_day_order_expires_and_replay_never_postpones_a_failed_buy(protocol_data):
    from sequoia_x.research.replay import Signal, replay_trade

    days = sessions(64)
    bars = [bar(day) for day in days]
    bars[1] = bar(days[1], open=12, high=12, low=12, close=12)
    snapshot = sample(days, bars=bars)
    numeric_policy = zero_cost_policy()
    result = replay_trade(
        snapshot,
        frozen_for(snapshot, numeric_policy, protocol_data),
        Signal(symbol="000001", signal_date=days[0]),
        100,
        numeric_policy,
    )
    assert result.status == "not_entered"
    assert result.fills == ()
    assert [order.status for order in result.orders] == ["unfilled", "expired"]
    assert result.daily[-1].cash == 10000


def test_explicit_entry_cancellation_is_auditable_without_a_fill(protocol_data):
    from sequoia_x.research.replay import Signal, replay_trade

    days = sessions(3)
    snapshot = sample(days)
    numeric_policy = zero_cost_policy()
    result = replay_trade(
        snapshot,
        frozen_for(snapshot, numeric_policy, protocol_data),
        Signal(symbol="000001", signal_date=days[0]),
        100,
        numeric_policy,
        cancel_entry_on=days[1],
    )
    assert result.status == "not_entered"
    assert result.fills == ()
    assert [(item.status, item.reason) for item in result.orders] == [
        ("cancelled", "explicit_entry_cancellation"),
    ]
    assert result.daily[-1].nav == 10000


def test_replay_rejects_unbound_numeric_costs_and_contradictory_order_time(protocol_data):
    from sequoia_x.research.replay import Signal, replay_trade

    days = sessions(3)
    snapshot = sample(days)
    numeric_policy = zero_cost_policy()
    frozen = frozen_for(snapshot, numeric_policy, protocol_data)
    changed = numeric_policy.model_copy(update={"slippage_bps": 10})
    with pytest.raises(ValueError, match="指纹"):
        replay_trade(snapshot, frozen, Signal(symbol="000001", signal_date=days[0]), 100, changed)
    contradictory = frozen.model_copy(
        update={
            "protocol": frozen.protocol.model_copy(
                update={
                    "trading": frozen.protocol.trading.model_copy(
                        update={"entry_order_time": "same_day_close"}
                    ),
                }
            )
        }
    )
    with pytest.raises(ValueError, match="订单时点"):
        replay_trade(
            snapshot,
            contradictory,
            Signal(symbol="000001", signal_date=days[0]),
            100,
            numeric_policy,
        )


def test_warmup_prices_are_available_but_warmup_signals_are_not_research_trades():
    from sequoia_x.research.replay import replay_ma_signals, signal_frame

    days = sessions(25)
    closes = [10] * 16 + [9] * 5 + [16] + [20] * 3
    snapshot = sample(
        days,
        bars=[
            bar(
                day,
                open=value,
                high=value + 1,
                low=value - 1,
                close=value,
                volume=300 if i == 21 else 100,
            )
            for i, (day, value) in enumerate(zip(days, closes))
        ],
    )
    snapshot = snapshot.model_copy(
        update={
            "manifest": snapshot.manifest.model_copy(
                update={
                    "start": days[22],
                }
            )
        }
    )
    assert len(signal_frame(snapshot, "000001", days[22])) == 23
    assert replay_ma_signals(snapshot) == ()


def test_suspended_quote_is_not_a_new_market_price_after_a_share_action():
    from sequoia_x.research.dataset import CorporateAction
    from sequoia_x.research.replay import mark_position

    first, ex_day = date(2025, 1, 6), date(2025, 1, 7)
    snapshot = sample(
        [first, ex_day],
        bars=[
            bar(first, close=10),
            bar(ex_day, open=None, high=None, low=None, close=10, volume=0, status="suspended"),
        ],
        states=[state(first), state(ex_day, suspended=True)],
    )
    unchanged = mark_position(snapshot, "000001", ex_day, 100)
    assert unchanged.status == "stale"
    assert unchanged.as_of == first
    assert unchanged.value == 1000
    changed = snapshot.model_copy(
        update={
            "actions": (
                CorporateAction(
                    symbol="000001", ex_date=ex_day, available_on=first, share_multiplier=2
                ),
            )
        }
    )
    split_mark = mark_position(changed, "000001", ex_day, 200)
    assert split_mark.status == "unreliable"
    assert split_mark.value is None
