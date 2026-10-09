"""策略引擎属性测试。"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from hypothesis import given, settings as h_settings
from hypothesis import strategies as st

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.strategy.ma_volume import MaVolumeStrategy


# Feature: sequoia-x-v2, Property 9: 策略 run() 返回值类型正确
@given(
    symbols=st.lists(
        st.text(min_size=6, max_size=6, alphabet="0123456789"),
        min_size=0, max_size=3, unique=True,
    )
)
@h_settings(max_examples=30, deadline=None)
def test_strategy_run_returns_list_of_str(symbols: list[str]) -> None:
    """属性 9：run() 应返回 list[str]，每个元素为非空字符串。"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        settings = Settings(
            db_path=str(Path(tmp_dir) / "test.db"),
            start_date="2024-01-01",
            bark_key="device-key",
        )
        engine = DataEngine(settings)

        with patch.object(engine, "get_all_symbols", return_value=symbols):
            with patch.object(engine, "get_ohlcv", return_value=pd.DataFrame()):
                strategy = MaVolumeStrategy(engine=engine, settings=settings)
                result = strategy.run()

    assert isinstance(result, list)
    assert all(isinstance(s, str) and len(s) > 0 for s in result)


def _seed_adjusted_bars(engine: DataEngine, symbol: str, bars: list[tuple[float, float, float]]) -> None:
    """写入后复权日线。bars 为 (close, volume, turnover)。"""
    rows = [
        (symbol, f"2024-01-{day:02d}", close, close, close, close, volume, turnover)
        for day, (close, volume, turnover) in enumerate(bars, start=1)
    ]
    with sqlite3.connect(engine.db_path) as conn:
        conn.executemany(
            "INSERT INTO stock_daily "
            "(symbol, date, open, high, low, close, volume, turnover) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def test_ma_volume_returns_comparable_ranking_for_each_pick() -> None:
    """每只入选股票都带可比较排序值，且同一天按排序值从高到低稳定排列。

    两只股票最后一日都满足金叉和放量。排序值是信号日成交额乘以
    “收盘价 / 20 日均线 - 1”，由独立的手算窗口得出，不复用策略实现：
    000001 最近 20 个收盘价之和为 204，均线为 10.2，200_000_000 * (20 / 10.2 - 1) = 192_156_862.74509805
    000002 最近 20 个收盘价之和为 208，均线为 10.4，100_000_000 * (22 / 10.4 - 1) = 111_538_461.53846154
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        settings = Settings(
            db_path=str(Path(tmp_dir) / "test.db"),
            start_date="2024-01-01",
            bark_key="device-key",
        )
        engine = DataEngine(settings)
        _seed_adjusted_bars(
            engine,
            "000001",
            [(10.0, 1_000.0, 1_000_000.0)] * 16
            + [(10.0, 1_000.0, 1_000_000.0), (9.0, 1_000.0, 1_000_000.0),
               (8.0, 1_000.0, 1_000_000.0), (7.0, 1_000.0, 1_000_000.0),
               (20.0, 2_000.0, 200_000_000.0)],
        )
        _seed_adjusted_bars(
            engine,
            "000002",
            [(10.0, 1_000.0, 1_000_000.0)] * 16
            + [(12.0, 1_000.0, 1_000_000.0), (9.0, 1_000.0, 1_000_000.0),
               (8.0, 1_000.0, 1_000_000.0), (7.0, 1_000.0, 1_000_000.0),
               (22.0, 2_000.0, 100_000_000.0)],
        )

        selected = MaVolumeStrategy(engine=engine, settings=settings).run()

    assert selected == [
        ("000001", 192_156_862.74509805),
        ("000002", 111_538_461.53846154),
    ]


def test_ma_volume_ranking_does_not_change_selection() -> None:
    """放量不足的股票即使能算出排序值，也不会进入选股名单。"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        settings = Settings(
            db_path=str(Path(tmp_dir) / "test.db"),
            start_date="2024-01-01",
            bark_key="device-key",
        )
        engine = DataEngine(settings)
        _seed_adjusted_bars(
            engine,
            "000003",
            [(10.0, 1_000.0, 1_000_000.0)] * 16
            + [(10.0, 1_000.0, 1_000_000.0), (9.0, 1_000.0, 1_000_000.0),
               (8.0, 1_000.0, 1_000_000.0), (7.0, 1_000.0, 1_000_000.0),
               (20.0, 1_000.0, 500_000_000.0)],
        )

        selected = MaVolumeStrategy(engine=engine, settings=settings).run()

    assert selected == []
