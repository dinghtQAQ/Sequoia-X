"""合成的可手算成交样例，不代表真实策略收益。"""

from datetime import date

import pytest

from sequoia_x.research.dataset import DailyBar, SecurityState
from sequoia_x.research.replay import ExecutionPolicy, FeeSchedule, quote_open_fill


def policy(**changes):
    values = dict(
        fees=(FeeSchedule(effective_from=date(2020, 1, 1), commission_rate=0.0003,
                          min_commission=5, stamp_tax_rate=0.001, transfer_fee_rate=0.0001),),
        slippage_bps=100,
        max_volume_participation=0.01,
        volume_model="daily_volume_upper_bound",
    )
    values.update(changes)
    return ExecutionPolicy(**values)


def bar(on=date(2025, 1, 6), **changes):
    values = dict(symbol="000001", date=on, available_on=on,
                  open=10, high=12, low=9, close=11, volume=100000)
    values.update(changes)
    return DailyBar(**values)


def state(on=date(2025, 1, 6), **changes):
    values = dict(symbol="000001", effective_date=on, available_on=on, listed=True,
                  suspended=False, is_st=False, board="main", lot_size=100,
                  limit_rule="bounded", limit_up=12, limit_down=8)
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

    sold = quote_open_fill(bar(open=11), state(), "sell", 100, 0, policy(),
                           acquired_on=date(2025, 1, 3))
    assert sold.price == 10.89
    assert sold.cash_delta == 1082.8
    assert sold.stamp_tax == 1.09
    assert sold.transfer_fee == 0.11
    assert bought.cash_delta + sold.cash_delta == pytest.approx(67.7)


def sample(days, *, bars=None, actions=(), states=None):
    from sequoia_x.research.dataset import DatasetManifest, DatasetSnapshot
    return DatasetSnapshot(
        manifest=DatasetManifest(
            source="hand-worked fixture", source_version="v1", universe_id="synthetic-single",
            symbols=("000001",), start=days[0], end=days[-1], as_of=days[-1],
            evidence_kind="synthetic", universe_complete=True, states_complete=True,
            actions_complete=True, calendar_complete=True,
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
    snapshot = sample([date(2025, 1, 6), ex_day, date(2025, 1, 8), pay_day], actions=[
        CorporateAction(symbol="000001", ex_date=ex_day, available_on=date(2025, 1, 6),
                        cash_per_share=1, withholding_tax_per_share=0.2,
                        share_multiplier=2, pay_date=pay_day),
    ])
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
    missing_bar = bar(last, open=None, high=None, low=None, close=None, volume=None,
                      status="missing")
    snapshot = sample([first, last], bars=[bar(first, close=11), missing_bar])
    mark = mark_position(snapshot, "000001", last, 100)
    assert mark.value == 1100
    assert mark.status == "stale"
    assert mark.as_of == first
    changed = snapshot.model_copy(update={"actions": (
        CorporateAction(symbol="000001", ex_date=last, available_on=first, share_multiplier=2),
    )})
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
    from sequoia_x.strategy.ma_volume import MaVolumeStrategy
    from sequoia_x.research.replay import replay_ma_signals, signal_frame

    days = sessions(25)
    closes = [10] * 16 + [9] * 5 + [16] + [20] * 3
    bars = [bar(day, open=value, high=value+1, low=value-1, close=value,
                volume=300 if index == 21 else 100)
            for index, (day, value) in enumerate(zip(days, closes))]
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
            return pd.DataFrame([dict(date=day, close=value, volume=(300 if i == 21 else 100))
                                 for i, (day, value) in enumerate(zip(days, closes)) if day <= cutoff])

    for cutoff in days[:22]:
        restored = MaVolumeStrategy(DailySource(), Settings(bark_key="test", _env_file=None)).run()
        replayed = [item.symbol for item in replay_ma_signals(snapshot, through=cutoff)
                    if item.signal_date == cutoff]
        assert restored == replayed
    assert list(signal_frame(snapshot, "000001", days[0]).columns) == [
        "date", "open", "high", "low", "close", "volume"
    ]
