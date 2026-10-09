"""回填同时保存后复权日线和不复权日线，两种口径互不覆盖。"""

import gc
import tempfile
from datetime import date
from pathlib import Path

import pytest

from sequoia_x.core.config import Settings
from sequoia_x.data import engine as engine_module
from sequoia_x.data.baostock_session import BaostockRequestError
from sequoia_x.data.engine import DataEngine


def make_engine(tmp_dir: Path) -> DataEngine:
    settings = Settings(
        db_path=str(tmp_dir / "test.db"),
        start_date="2024-01-01",
        bark_key="device-key",
    )
    return DataEngine(settings)


def bar(trade_date: str, close: str) -> list[str]:
    return [trade_date, close, close, close, close, "2000", "22000"]


class ScriptedSession:
    """按复权口径返回同一证券的后复权或不复权 K 线。"""

    instances: list["ScriptedSession"] = []
    scripts: dict[tuple[str, str], list[list[str] | BaseException]] = {}
    queries: list[tuple[str, str]] = []

    def __init__(self, *args, **kwargs) -> None:
        ScriptedSession.instances.append(self)

    def query_history_k_data_plus(
        self,
        code: str,
        fields: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjustflag: str = "3",
    ) -> list[list[str]]:
        ScriptedSession.queries.append((code, adjustflag))
        item = ScriptedSession.scripts[(code, adjustflag)].pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        return None


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch):
    ScriptedSession.instances = []
    ScriptedSession.scripts = {}
    ScriptedSession.queries = []
    monkeypatch.setattr(engine_module, "BaostockSession", ScriptedSession)
    return ScriptedSession


@pytest.fixture
def workspace():
    tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    try:
        yield Path(tmp.name)
    finally:
        gc.collect()
        tmp.cleanup()


def test_backfill_saves_both_adjustments_without_overwriting(
    scripted, workspace: Path
) -> None:
    """一次回填保存同一批股票的两种日线；重复回填不产生重复记录，也不改写另一种口径。"""
    today = date.today().strftime("%Y-%m-%d")
    scripted.scripts[("sz.000001", "1")] = [[bar(today, "20.0")], [bar(today, "21.0")]]
    scripted.scripts[("sz.000001", "3")] = [[bar(today, "10.0")], [bar(today, "11.0")]]
    scripted.scripts[("sh.600000", "1")] = [[bar(today, "40.0")], [bar(today, "41.0")]]
    scripted.scripts[("sh.600000", "3")] = [[bar(today, "30.0")], [bar(today, "31.0")]]

    engine = make_engine(workspace)
    engine.backfill(["000001", "600000"])
    engine.backfill(["000001", "600000"])

    adjusted = engine.get_ohlcv("000001")
    raw = engine.get_raw_ohlcv("000001")
    other_adjusted = engine.get_ohlcv("600000")
    other_raw = engine.get_raw_ohlcv("600000")

    assert adjusted["date"].tolist() == [today]
    assert raw["date"].tolist() == [today]
    assert adjusted.loc[0, "close"] == 20.0
    assert raw.loc[0, "close"] == 10.0
    assert other_adjusted.loc[0, "close"] == 40.0
    assert other_raw.loc[0, "close"] == 30.0
    assert {flag for _, flag in scripted.queries} == {"1", "3"}


def test_missing_raw_bars_are_not_stored_as_adjusted_prices(
    scripted, workspace: Path
) -> None:
    """不复权日线拉取失败时明确报错，不能把后复权价格写成不复权日线。"""
    today = date.today().strftime("%Y-%m-%d")
    scripted.scripts[("sz.000001", "1")] = [[bar(today, "20.0")]]
    scripted.scripts[("sz.000001", "3")] = [
        BaostockRequestError("不复权日线缺失", code="sz.000001")
    ]

    engine = make_engine(workspace)
    with pytest.raises(BaostockRequestError, match="不复权"):
        engine.backfill(["000001"])

    assert engine.get_raw_ohlcv("000001").empty


def test_adjusted_fetch_failure_does_not_block_or_replace_raw_bars(
    scripted, workspace: Path
) -> None:
    """后复权拉取失败时明确报错，仍保存已返回的不复权日线，不用后复权价格替代。"""
    today = date.today().strftime("%Y-%m-%d")
    scripted.scripts[("sz.000001", "1")] = [
        BaostockRequestError("后复权日线缺失", code="sz.000001")
    ]
    scripted.scripts[("sz.000001", "3")] = [[bar(today, "10.0")]]

    engine = make_engine(workspace)
    with pytest.raises(BaostockRequestError, match="后复权"):
        engine.backfill(["000001"])

    assert engine.get_ohlcv("000001").empty
    assert engine.get_raw_ohlcv("000001").loc[0, "close"] == 10.0


def test_failed_repeat_does_not_rewrite_the_other_adjustment(
    scripted, workspace: Path
) -> None:
    """同一日期再次回填失败时，已保存的另一种复权口径保持原价。"""
    trade_date = "2024-06-03"
    scripted.scripts[("sz.000001", "1")] = [
        [bar(trade_date, "20.0")],
        BaostockRequestError("后复权重跑失败", code="sz.000001"),
    ]
    scripted.scripts[("sz.000001", "3")] = [
        [bar(trade_date, "10.0")],
        BaostockRequestError("不复权重跑失败", code="sz.000001"),
    ]

    engine = make_engine(workspace)
    engine.backfill(["000001"])
    with pytest.raises(BaostockRequestError, match="不复权"):
        engine.backfill(["000001"])

    adjusted = engine.get_ohlcv("000001")
    raw = engine.get_raw_ohlcv("000001")
    assert adjusted["date"].tolist() == [trade_date]
    assert raw["date"].tolist() == [trade_date]
    assert adjusted.loc[0, "close"] == 20.0
    assert raw.loc[0, "close"] == 10.0
