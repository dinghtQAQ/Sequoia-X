"""验证方向、缺失观测、排名优先级及时间边界，避免把未来行情带入筛选。"""

import numpy as np
import pandas as pd
import pytest

from sequoia_x.correlation import CorrelationConfig, CorrelationHistory, select_uncorrelated_signals


def _history() -> tuple[pd.DataFrame, list[str]]:
    dates = pd.bdate_range("2025-01-02", periods=65).strftime("%Y-%m-%d").tolist()
    returns = np.resize(np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.04]), 64)
    records = []
    for symbol, changes in (
        ("000001", returns), ("000002", returns * 1.5 + 0.001),
        ("000003", -returns), ("000004", np.zeros(64)),
    ):
        prices = 10 * np.exp(np.r_[0, np.cumsum(changes)])
        records.extend((symbol, day, price, 1000.0) for day, price in zip(dates, prices))
    return pd.DataFrame(records, columns=["symbol", "date", "close", "volume"]), dates


def test_log_return_correlation_keeps_sign_and_counts_pairwise_missing_observations() -> None:
    bars, dates = _history()
    bars = bars.loc[~((bars["symbol"] == "000002") & (bars["date"] == dates[10]))].copy()
    bars.loc[(bars["symbol"] == "000002") & (bars["date"] == dates[15]), "volume"] = 0
    history = CorrelationHistory(bars, dates, CorrelationConfig())

    rho, count = history.matrix(60, ["000001"], ["000002", "000003", "000004"])

    # 缺一根日线或停牌日均使该日和紧接其后的一日收益未知，不前向填充。
    assert count.tolist() == [[56, 60, 60]]
    assert rho[0, :2].tolist() == pytest.approx([1.0, -1.0])
    assert np.isnan(rho[0, 2])  # 零波动相关性未知，不能记成0。
    history = CorrelationHistory(bars, dates, CorrelationConfig(min_periods=57))
    rho, _ = history.matrix(60, ["000001"], ["000002"])
    assert np.isnan(rho[0, 0])


def test_full_history_and_truncated_history_give_same_signal_day_correlation() -> None:
    bars, dates = _history()
    full = CorrelationHistory(bars, dates, CorrelationConfig())
    truncated = CorrelationHistory(bars.loc[bars["date"] <= dates[50]], dates[:51], CorrelationConfig())
    expected, counts = truncated.matrix(50, ["000001"], ["000002", "000003"])
    actual, full_counts = full.matrix(50, ["000001"], ["000002", "000003"])
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(full_counts, counts)

    changed = bars.copy()
    changed.loc[changed["date"] > dates[50], "close"] *= 1000
    future_changed = CorrelationHistory(changed, dates, CorrelationConfig())
    actual, _ = future_changed.matrix(50, ["000001"], ["000002", "000003"])
    np.testing.assert_allclose(actual, expected)


def test_same_day_uses_ranking_and_negative_correlation_is_not_rejected() -> None:
    bars, dates = _history()
    signals = pd.DataFrame({
        "symbol": ["000001", "000003", "000002"],
        "date": [dates[40]] * 3,
        "ranking": [1.0, 5.0, 10.0],
    })
    history = CorrelationHistory(bars, dates, CorrelationConfig())
    selected = select_uncorrelated_signals(signals, {day: i for i, day in enumerate(dates)}, 5, 60, history)

    assert selected["corr_accepted"].tolist() == [False, True, True]
    assert selected.loc[0, "corr_reason"] == "高相关"
    assert selected.loc[0, "corr_peer"] == "000002"
    assert selected.loc[0, "corr_value"] == pytest.approx(1.0)
    assert selected.loc[1, "corr_value"] == pytest.approx(-1.0)

    signals["ranking"] = 1.0
    selected = select_uncorrelated_signals(signals, {day: i for i, day in enumerate(dates)}, 5, 60, history)
    assert selected["corr_accepted"].tolist() == [True, True, False]  # 同分代码升序。


def test_expired_observation_is_not_held_for_entire_cooldown_and_rejection_does_not_lock() -> None:
    bars, dates = _history()
    signals = pd.DataFrame({
        "symbol": ["000001", "000002", "000002", "000001", "000001"],
        "date": [dates[40], dates[42], dates[43], dates[44], dates[51]],
        "ranking": [1.0] * 5,
    })
    history = CorrelationHistory(bars, dates, CorrelationConfig())
    index = {day: i for i, day in enumerate(dates)}

    selected = select_uncorrelated_signals(signals, index, 2, 10, history)

    assert selected["corr_accepted"].tolist() == [True, False, True, False, True]
    assert selected["corr_reason"].tolist() == ["保留", "高相关", "保留", "时间窗口内重复信号", "保留"]
    long_hold = select_uncorrelated_signals(signals, index, 5, 10, history)
    assert not long_hold.loc[2, "corr_accepted"]  # 5日观察期下，旧信号仍然在观察。


def test_unknown_correlation_is_recorded_and_not_silently_treated_as_zero() -> None:
    bars, dates = _history()
    signals = pd.DataFrame({
        "symbol": ["000001", "000002"], "date": [dates[20]] * 2, "ranking": [2.0, 1.0],
    })
    history = CorrelationHistory(bars, dates, CorrelationConfig())
    selected = select_uncorrelated_signals(signals, {day: i for i, day in enumerate(dates)}, 5, 60, history)

    assert selected["corr_accepted"].tolist() == [True, True]
    assert selected.loc[1, "corr_unknown_pairs"] == 1
    assert selected.loc[1, "corr_reference_count"] == 1
    assert np.isnan(selected.loc[1, "corr_value"])
