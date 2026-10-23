"""用可手算的小型数据库验证信号收益、市场交易日与共同样本。"""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from sequoia_x.signal_study import SignalStudyError, study_signals

WARMUP = [10.0] * 17 + [9.0, 8.0, 7.0, 20.0]
DATES = pd.bdate_range("2025-01-02", periods=110).strftime("%Y-%m-%d").tolist()


def _bars(symbol: str, future: list[float]) -> list[tuple]:
    closes = WARMUP + future
    volumes = [1000.0] * 20 + [2000.0] + [1000.0] * len(future)
    return [(symbol, day, close, volume) for day, close, volume in zip(DATES, closes, volumes)]


def _seed(db_path: Path, rows: list[tuple]) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS stock_daily "
            "(symbol TEXT, date TEXT, close REAL, volume REAL, PRIMARY KEY (symbol,date))"
        )
        connection.executemany("INSERT INTO stock_daily VALUES (?,?,?,?)", rows)
        connection.commit()
    finally:
        connection.close()


def _study(tmp_path: Path, **overrides) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    arguments = {
        "strategy": "MaVolume",
        "db_path": str(tmp_path / "history.db"),
        "results_dir": tmp_path / "results",
        "run_at": "2026-10-23 12:34:56",
        "start": DATES[20],
        "end": DATES[20],
        "horizons": (20, 40, 60),
    }
    arguments.update(overrides)
    path = study_signals(**arguments)
    summary = pd.read_csv(path)
    samples = pd.read_csv(path.with_name(path.stem + "_samples.csv"))
    metadata = json.loads(path.with_name(path.stem + "_meta.json").read_text(encoding="utf-8"))
    return summary, samples, metadata


def test_fixed_horizons_include_every_signal_and_ignore_early_price_swings(tmp_path: Path) -> None:
    """五只股票全部纳入；中途跌95%、涨400%均不提前退出，终点按收盘价格算。"""
    future = [20.0] * 60
    future[0], future[1] = 1.0, 100.0
    future[19], future[39], future[59] = 22.0, 18.0, 24.0
    for symbol in ("000001", "000002", "000003", "600001", "600002"):
        _seed(tmp_path / "history.db", _bars(symbol, future))
    before = (tmp_path / "history.db").read_bytes()

    summary, samples, metadata = _study(tmp_path)

    assert (tmp_path / "history.db").read_bytes() == before
    independent = summary.loc[
        (summary["样本口径"] == "各周期有效样本") & (summary["去重口径"] == "全部信号")
    ].set_index("观察周期")
    assert independent["有效样本数"].tolist() == [5, 5, 5]
    assert independent["平均收益"].tolist() == pytest.approx([0.1, -0.1, 0.2])
    assert independent["胜率"].tolist() == [1.0, 0.0, 1.0]
    assert set(samples["股票代码"]) == {"sz000001", "sz000002", "sz000003", "sh600001", "sh600002"}
    assert set(samples.loc[samples["观察周期"] == 20, "目标日"]) == {DATES[40]}
    assert len(samples) == 15
    assert metadata["total_signals"] == metadata["common_samples"] == 5
    assert metadata["signal_end"] == DATES[20]


def test_common_cohort_intersects_exact_signals_and_missing_prices_do_not_shift(tmp_path: Path) -> None:
    """同一信号日，不同股票缺不同终点；共同样本只能留下三个周期都完整的一只。"""
    for symbol, missing_day in (("000001", DATES[40]), ("000002", DATES[80]), ("000003", None)):
        rows = [row for row in _bars(symbol, [22.0] * 60) if row[1] != missing_day]
        _seed(tmp_path / "history.db", rows)

    summary, samples, metadata = _study(tmp_path)

    baseline = summary.loc[summary["去重口径"] == "全部信号"]
    independent = baseline.loc[baseline["样本口径"] == "各周期有效样本"]
    common = baseline.loc[baseline["样本口径"] == "共同有效样本"]
    assert independent["有效样本数"].tolist() == [2, 3, 2]
    assert common["有效样本数"].tolist() == [1, 1, 1]
    assert metadata["common_samples"] == 1
    missing = samples.loc[samples["状态"] == "目标日缺少价格"]
    assert set(zip(missing["股票代码"], missing["目标日"])) == {
        ("sz000001", DATES[40]), ("sz000002", DATES[80]),
    }
    assert missing["收益率"].isna().all()


def test_incomplete_windows_are_reported_without_forced_interval_exit(tmp_path: Path) -> None:
    _seed(tmp_path / "history.db", _bars("000001", [40.0] * 10))

    summary, samples, metadata = _study(tmp_path)

    assert summary["有效样本数"].eq(0).all()
    assert summary[["胜率", "平均收益", "中位收益"]].isna().all().all()
    assert samples["状态"].eq("后续交易日不足").all()
    assert samples["收益率"].isna().all()
    assert metadata["common_samples"] == 0


def test_standard_median_and_mean_are_not_sum_of_returns(tmp_path: Path) -> None:
    _seed(tmp_path / "history.db", _bars("000001", [22.0] * 60))
    _seed(tmp_path / "history.db", _bars("000002", [26.0] * 60))

    summary, _, _ = _study(tmp_path, horizons=(20,))

    assert summary["平均收益"].tolist() == pytest.approx([0.2] * 4)
    assert summary["中位收益"].tolist() == pytest.approx([0.2] * 4)


def test_different_dates_of_same_stock_remain_independent(tmp_path: Path) -> None:
    closes = WARMUP * 2 + [20.0] * 60
    volumes = ([1000.0] * 20 + [2000.0]) * 2 + [1000.0] * 60
    rows = [("000001", day, close, volume) for day, close, volume in zip(DATES, closes, volumes)]
    _seed(tmp_path / "history.db", rows)

    summary, samples, metadata = _study(tmp_path, end=DATES[41])

    assert metadata["total_signals"] == 2
    assert summary.loc[summary["去重口径"] == "全部信号", "有效样本数"].eq(2).all()
    filtered = summary.loc[
        (summary["去重口径"] == "持有期去重") & (summary["样本口径"] == "各周期有效样本")
    ]
    assert filtered["有效样本数"].tolist() == [2, 1, 1]
    common = summary.loc[
        (summary["去重口径"] == "持有期去重") & (summary["样本口径"] == "共同有效样本")
    ]
    assert common["有效样本数"].eq(1).all()
    assert set(samples["信号日"]) == {DATES[20], DATES[41]}


def test_no_signals_still_writes_empty_details_and_all_default_horizons(tmp_path: Path) -> None:
    rows = [("000001", day, 10.0, 1000.0) for day in DATES]
    _seed(tmp_path / "history.db", rows)

    summary, samples, metadata = _study(tmp_path, horizons=tuple(range(20, 61)))

    assert summary["观察周期"].tolist() == [value for value in range(20, 61) for _ in range(4)]
    assert summary["有效样本数"].eq(0).all()
    assert samples.empty
    assert metadata["total_signals"] == 0


def test_holding_window_blocks_endpoint_and_rejected_signal_does_not_extend_it(tmp_path: Path) -> None:
    """信号间隔21日：21日观察窗口拒绝第二次，第三次已到期，不被拒绝信号续期。"""
    closes = WARMUP * 3 + [20.0] * 60
    volumes = ([1000.0] * 20 + [2000.0]) * 3 + [1000.0] * 60
    dates = pd.bdate_range("2025-01-02", periods=len(closes)).strftime("%Y-%m-%d").tolist()
    rows = [("000001", day, close, volume) for day, close, volume in zip(dates, closes, volumes)]
    _seed(tmp_path / "history.db", rows)

    summary, samples, metadata = _study(tmp_path, end=dates[62], horizons=(20, 21, 41))

    filtered = summary.loc[
        (summary["去重口径"] == "持有期去重") & (summary["样本口径"] == "各周期有效样本")
    ]
    assert filtered["接受信号数"].tolist() == [3, 2, 2]
    assert filtered["重复过滤数"].tolist() == [0, 1, 1]
    selected = samples.loc[(samples["观察周期"] == 21) & samples["纳入持有期去重"]]
    assert selected["信号日"].tolist() == [dates[20], dates[62]]
    assert metadata["accepted_signals_by_horizon"] == {"20": 3, "21": 2, "41": 2}


def test_deduplication_does_not_skip_first_signal_based_on_future_missing_price(tmp_path: Path) -> None:
    """首个已接受信号缺少终点价格，也不能事后改为接受其观察期内的第二个信号。"""
    closes = WARMUP * 2 + [20.0] * 60
    volumes = ([1000.0] * 20 + [2000.0]) * 2 + [1000.0] * 60
    rows = [
        ("000001", day, close, volume)
        for day, close, volume in zip(DATES, closes, volumes) if day != DATES[60]
    ]
    _seed(tmp_path / "history.db", rows)
    _seed(tmp_path / "history.db", [("600001", day, 10.0, 1000.0) for day in DATES])

    summary, samples, _ = _study(tmp_path, end=DATES[41], horizons=(40,))

    assert samples["纳入持有期去重"].tolist() == [True, False]
    assert samples["状态"].tolist() == ["目标日缺少价格", "完整"]
    assert summary.loc[summary["去重口径"] == "持有期去重", "有效样本数"].eq(0).all()
    assert summary.loc[summary["去重口径"] == "全部信号", "有效样本数"].eq(1).all()


@pytest.mark.parametrize("overrides, message", [
    ({"strategy": "TurtleTrade"}, "只支持 MaVolume"),
    ({"start": "2025-99-99"}, "日期参数无效"),
    ({"start": "2026-01-01", "end": "2025-01-01"}, "不能晚于"),
    ({"horizons": (0,)}, "正整数"),
    ({"horizons": (-1,)}, "正整数"),
    ({"horizons": ()}, "正整数"),
])
def test_invalid_parameters_write_nothing(tmp_path: Path, overrides: dict, message: str) -> None:
    with pytest.raises(SignalStudyError, match=message):
        _study(tmp_path, **overrides)
    assert not (tmp_path / "results").exists()


def test_missing_database_is_not_created(tmp_path: Path) -> None:
    with pytest.raises(SignalStudyError, match="缺少本地数据库"):
        _study(tmp_path)
    assert not (tmp_path / "history.db").exists()
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("flag", ["--study", "--compare"])
def test_cli_study_skips_sync_notifications_and_data_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str,
) -> None:
    import main

    calls = []
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(
        db_path=str(tmp_path / "history.db"), start_date="2024-01-01",
    ))

    def forbidden(*args, **kwargs):
        pytest.fail("研究模式不应初始化数据同步或通知")

    def fake_study(**kwargs):
        calls.append(kwargs)
        return tmp_path / "summary.csv"

    monkeypatch.setattr(main, "DataEngine", forbidden)
    monkeypatch.setattr(main, "BarkNotifier", forbidden)
    monkeypatch.setattr("sequoia_x.signal_study.study_signals", fake_study)
    monkeypatch.setattr("sys.argv", [
        "main.py", flag, "MaVolume", "--start", "2025-06-01", "--end", "2026-09-30",
        "--horizons", "20", "60",
    ])

    main.main()

    assert len(calls) == 1
    assert calls[0]["strategy"] == "MaVolume"
    assert calls[0]["horizons"] == [20, 60]
