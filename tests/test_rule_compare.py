"""规则比较入口：准备小型本地库，只断言成绩表和日志。"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.rule_compare import compare_rules


def _settings(tmp_dir: Path) -> Settings:
    return Settings(
        db_path=str(tmp_dir / "test.db"),
        start_date="2024-01-01",
        bark_key="device-key",
    )


def _seed(engine: DataEngine, table: str, symbol: str, bars: list[tuple]) -> None:
    rows = [
        (symbol, trade_date, open_, high, low, close, volume, turnover)
        for trade_date, open_, high, low, close, volume, turnover in bars
    ]
    with sqlite3.connect(engine.db_path) as conn:
        conn.executemany(
            f"INSERT INTO {table} "
            "(symbol, date, open, high, low, close, volume, turnover) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def _flat(start: str, days: int, close: float = 10.0) -> list[tuple]:
    year, month, day = (int(part) for part in start.split("-"))
    bars = []
    for offset in range(days):
        trade_date = f"{year:04d}-{month:02d}-{day + offset:02d}"
        bars.append((trade_date, close, close, close, close, 1000.0, 10_000.0))
    return bars


def _scorecard(results: Path) -> Path:
    files = list(results.glob("MaVolume_*.csv"))
    assert len(files) == 1
    return files[0]


def _log(results: Path) -> Path:
    files = list(results.glob("MaVolume_*.log"))
    assert len(files) == 1
    return files[0]


def test_compare_reruns_one_strategy_each_day_and_writes_one_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """一次比较只跑一个选股策略，每个交易日重新生成名单，并写出一个场景的成绩和日志。

    后复权日线只在 2024-01-23 形成 MaVolume 金叉。不复权日线单独定价：
    信号日收盘 10，次日开盘 10、最高 10.5，按收盘价买入规则在次日成交。
    买入后第二个交易日开盘 11 卖出。手算（费用置零，单只上限 10,000 元）：
    买入价 10.00，卖出价 11.00，收益 (11-10)/10 = 0.10。
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_name:
        tmp_dir = Path(tmp_name)
        results = tmp_dir / "results"
        engine = DataEngine(_settings(tmp_dir))
        adjusted = _flat("2024-01-02", 16) + [
            ("2024-01-18", 10.0, 10.0, 10.0, 10.0, 1000.0, 10_000.0),
            ("2024-01-19", 9.0, 9.0, 9.0, 9.0, 1000.0, 10_000.0),
            ("2024-01-20", 8.0, 8.0, 8.0, 8.0, 1000.0, 10_000.0),
            ("2024-01-21", 7.0, 7.0, 7.0, 7.0, 1000.0, 10_000.0),
            ("2024-01-22", 20.0, 20.0, 20.0, 20.0, 2000.0, 200_000_000.0),
            ("2024-01-23", 20.0, 20.0, 20.0, 20.0, 1000.0, 10_000.0),
            ("2024-01-24", 20.0, 20.0, 20.0, 20.0, 1000.0, 10_000.0),
            ("2024-01-25", 20.0, 20.0, 20.0, 20.0, 1000.0, 10_000.0),
        ]
        raw = _flat("2024-01-02", 21, close=10.0) + [
            ("2024-01-23", 10.0, 10.0, 10.0, 10.0, 1000.0, 10_000.0),
            ("2024-01-24", 10.0, 10.5, 9.8, 10.2, 1000.0, 10_000.0),
            ("2024-01-25", 11.0, 11.2, 10.8, 11.0, 1000.0, 10_000.0),
        ]
        _seed(engine, "stock_daily", "000001", adjusted)
        _seed(engine, "stock_daily_raw", "000001", raw)
        monkeypatch.setattr("sequoia_x.data.engine.BaostockSession", _NoNetwork)

        compare_rules(
            strategy="MaVolume",
            db_path=engine.db_path,
            results_dir=results,
            run_at="2026-10-13 15:04:05",
            start="2024-01-20",
            end="2024-01-25",
            buy_rule="close",
            sell_rule="stop8-take15-hold2",
            initial_cash=50_000,
            commission_rate=0,
            min_commission=0,
            stamp_duty=0,
            slippage=0,
        )

        scorecard = _scorecard(results).read_text(encoding="utf-8")
        log = _log(results).read_text(encoding="utf-8")

    assert scorecard == (
        "场景编号,收益,最大回撤,收益回撤比,中位收益,交易次数,胜率,平均每笔收益,无法买入次数,实际区间\n"
        "S01,0.1000,0.0000,0.0000,0.1000,1,1.0000,0.1000,0,2024-01-20至2024-01-25\n"
    )
    assert log == "S01 2024-01-22 sz000001 买入\nS01 2024-01-22 sz000001 卖出\n"


def test_compare_fails_when_local_bars_are_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """缺少比较所需的本地日线时明确失败，且不下载行情。"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_name:
        tmp_dir = Path(tmp_name)
        engine = DataEngine(_settings(tmp_dir))
        monkeypatch.setattr("sequoia_x.data.engine.BaostockSession", _NoNetwork)

        with pytest.raises(Exception, match="缺少"):
            compare_rules(
                strategy="MaVolume",
                db_path=engine.db_path,
                results_dir=tmp_dir / "results",
                run_at="2026-10-13 15:04:05",
                start="2024-01-20",
                end="2024-01-25",
                buy_rule="close",
                sell_rule="stop8-take15-hold2",
            )

        assert list((tmp_dir / "results").glob("*")) == []


def test_missing_raw_bar_is_unfilled_and_not_replaced_by_adjusted_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """缺少对应不复权日线时记为无法买入，不用后复权价替代。"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_name:
        tmp_dir = Path(tmp_name)
        results = tmp_dir / "results"
        engine = DataEngine(_settings(tmp_dir))
        adjusted = _flat("2024-01-02", 16) + [
            ("2024-01-18", 10.0, 10.0, 10.0, 10.0, 1000.0, 10_000.0),
            ("2024-01-19", 9.0, 9.0, 9.0, 9.0, 1000.0, 10_000.0),
            ("2024-01-20", 8.0, 8.0, 8.0, 8.0, 1000.0, 10_000.0),
            ("2024-01-21", 7.0, 7.0, 7.0, 7.0, 1000.0, 10_000.0),
            ("2024-01-22", 20.0, 20.0, 20.0, 20.0, 2000.0, 200_000_000.0),
            ("2024-01-23", 20.0, 20.0, 20.0, 20.0, 1000.0, 10_000.0),
        ]
        raw = _flat("2024-01-02", 20, close=10.0) + [
            ("2024-01-22", 10.0, 10.0, 10.0, 10.0, 1000.0, 10_000.0),
        ]
        _seed(engine, "stock_daily", "000001", adjusted)
        _seed(engine, "stock_daily_raw", "000001", raw)
        monkeypatch.setattr("sequoia_x.data.engine.BaostockSession", _NoNetwork)

        compare_rules(
            strategy="MaVolume",
            db_path=engine.db_path,
            results_dir=results,
            run_at="2026-10-13 15:04:05",
            start="2024-01-20",
            end="2024-01-23",
            buy_rule="close",
            sell_rule="stop8-take15-hold2",
            commission_rate=0,
            min_commission=0,
            stamp_duty=0,
            slippage=0,
        )

        scorecard = _scorecard(results).read_text(encoding="utf-8")
        log = _log(results).read_text(encoding="utf-8")

    assert "无法买入次数" in scorecard
    assert ",1,2024-01-20至2024-01-23\n" in scorecard
    assert log == "S01 2024-01-22 sz000001 无法买入\n"


def test_default_costs_reduce_the_close_to_open_trade(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认费用从同一笔收盘买入、持有两日后开盘卖出中扣除。

    手算：买入 10 * 1.001 = 10.01。1000 股加最低佣金超过单只上限 10,000 元，
    所以买 900 股，成本 9014 元。卖出 11 * 0.999 * 900 = 9885.15495，
    印花税按滑点后金额扣一次。净收益 871.15495 / 10000 = 0.087115495，成绩表记 0.0871。
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_name:
        tmp_dir = Path(tmp_name)
        results = tmp_dir / "results"
        engine = DataEngine(_settings(tmp_dir))
        adjusted = _flat("2024-01-02", 16) + [
            ("2024-01-18", 10.0, 10.0, 10.0, 10.0, 1000.0, 10_000.0),
            ("2024-01-19", 9.0, 9.0, 9.0, 9.0, 1000.0, 10_000.0),
            ("2024-01-20", 8.0, 8.0, 8.0, 8.0, 1000.0, 10_000.0),
            ("2024-01-21", 7.0, 7.0, 7.0, 7.0, 1000.0, 10_000.0),
            ("2024-01-22", 20.0, 20.0, 20.0, 20.0, 2000.0, 200_000_000.0),
            ("2024-01-23", 20.0, 20.0, 20.0, 20.0, 1000.0, 10_000.0),
            ("2024-01-24", 20.0, 20.0, 20.0, 20.0, 1000.0, 10_000.0),
            ("2024-01-25", 20.0, 20.0, 20.0, 20.0, 1000.0, 10_000.0),
        ]
        raw = _flat("2024-01-02", 21, close=10.0) + [
            ("2024-01-23", 10.0, 10.0, 10.0, 10.0, 1000.0, 10_000.0),
            ("2024-01-24", 10.0, 10.5, 9.8, 10.2, 1000.0, 10_000.0),
            ("2024-01-25", 11.0, 11.2, 10.8, 11.0, 1000.0, 10_000.0),
        ]
        _seed(engine, "stock_daily", "000001", adjusted)
        _seed(engine, "stock_daily_raw", "000001", raw)
        monkeypatch.setattr("sequoia_x.data.engine.BaostockSession", _NoNetwork)

        compare_rules(
            strategy="MaVolume",
            db_path=engine.db_path,
            results_dir=results,
            run_at="2026-10-13 15:04:05",
            start="2024-01-20",
            end="2024-01-25",
            buy_rule="close",
            sell_rule="stop8-take15-hold2",
        )

        scorecard = _scorecard(results).read_text(encoding="utf-8")

    assert scorecard.splitlines()[1].split(",")[1] == "0.0871"


def test_output_names_include_strategy_and_second_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """成绩表和日志的文件名包含选股策略和精确到秒的运行时间。"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_name:
        tmp_dir = Path(tmp_name)
        results = tmp_dir / "results"
        engine = DataEngine(_settings(tmp_dir))
        _seed(engine, "stock_daily", "000001", _flat("2024-01-02", 3))
        _seed(engine, "stock_daily_raw", "000001", _flat("2024-01-02", 3))
        monkeypatch.setattr("sequoia_x.data.engine.BaostockSession", _NoNetwork)

        compare_rules(
            strategy="MaVolume",
            db_path=engine.db_path,
            results_dir=results,
            run_at="2026-10-13 15:04:05",
            start="2024-01-02",
            end="2024-01-04",
            buy_rule="close",
            sell_rule="stop8-take15-hold2",
        )

        names = sorted(path.name for path in results.iterdir())

    assert names == [
        "MaVolume_20261013150405.csv",
        "MaVolume_20261013150405.log",
    ]


class _NoNetwork:
    def __init__(self, *args, **kwargs) -> None:
        raise AssertionError("规则比较不能下载行情")
