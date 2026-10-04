"""日常全市场增量同步：有界并发、逐股重试、失败隔离。"""

import gc
import sqlite3
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from sequoia_x.core.config import Settings
from sequoia_x.data.baostock_session import (
    BaostockRequestError,
    ConnectionFailure,
    LoginFailure,
)
from sequoia_x.data import engine as engine_module
from sequoia_x.data.engine import DataEngine, _bs_fetch_symbol


def make_engine(tmp_dir: Path) -> DataEngine:
    settings = Settings(
        db_path=str(Path(tmp_dir) / "test.db"),
        start_date="2024-01-01",
        feishu_webhook_url="https://example.com/hook",
    )
    return DataEngine(settings)


def seed(engine: DataEngine, symbols: list[str], last_date: str) -> None:
    rows = [
        {
            "symbol": symbol,
            "date": last_date,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 1000.0,
            "turnover": 10500.0,
        }
        for symbol in symbols
    ]
    with sqlite3.connect(engine.db_path) as conn:
        pd.DataFrame(rows).to_sql("stock_daily", conn, if_exists="append", index=False)


def bar(trade_date: str, close: str = "11.2") -> list[str]:
    return [trade_date, "10.8", "11.5", "10.6", close, "2000", "22000"]


class ScriptedSession:
    """按调用脚本返回行情、连接失败或登录失败。"""

    instances: list["ScriptedSession"] = []
    scripts: dict[str, list[list[str] | BaseException]] = {}
    queried: list[str] = []
    closes = 0

    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[str] = []
        ScriptedSession.instances.append(self)

    def query_history_k_data_plus(self, code: str, *args, **kwargs) -> list[list[str]]:
        self.calls.append(code)
        ScriptedSession.queried.append(code)
        script = ScriptedSession.scripts[code]
        item = script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        ScriptedSession.closes += 1


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch):
    ScriptedSession.instances = []
    ScriptedSession.scripts = {}
    ScriptedSession.queried = []
    ScriptedSession.closes = 0
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


def test_default_workers_stay_at_most_four(scripted, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """五只待更新股票时，默认并发不超过 4，且不再固定开 8 个进程。"""
    seen: list[int] = []
    real_collector = engine_module._collect_symbol_rows

    def record(tasks, worker_count):
        seen.append(worker_count)
        return real_collector(tasks, worker_count)

    monkeypatch.setattr(engine_module, "_collect_symbol_rows", record)
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    today = date.today().strftime("%Y-%m-%d")
    symbols = ["000001", "000002", "000003", "600000", "600001"]
    for symbol in symbols:
        scripted.scripts[DataEngine._to_baostock_code(symbol)] = [[bar(today)]]

    engine = make_engine(workspace)
    seed(engine, symbols, yesterday)
    engine.sync_today_bulk()

    assert seen == [4]


def test_successful_rows_are_written_and_rerun_keeps_unique_dates(scripted, workspace: Path) -> None:
    """正常结果写入 SQLite；同一天重跑不会撞唯一约束。"""
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    today = date.today().strftime("%Y-%m-%d")
    scripted.scripts["sz.000001"] = [[bar(today, "12.0")]]

    engine = make_engine(workspace)
    seed(engine, ["000001"], yesterday)
    written = engine.sync_today_bulk()
    again = engine.sync_today_bulk()
    frame = engine.get_ohlcv("000001")

    assert written == 1
    assert again == 0
    assert frame["date"].tolist() == [yesterday, today]
    assert frame.loc[frame["date"] == today, "close"].iloc[0] == 12.0


def test_retry_rebuilds_connection_then_keeps_the_row(scripted, workspace: Path) -> None:
    """单股请求失败后按限定次数重试，重试前重建连接，成功结果仍入库。"""
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    today = date.today().strftime("%Y-%m-%d")
    scripted.scripts["sz.000001"] = [
        ConnectionFailure("连接中断", code="sz.000001"),
        [bar(today, "13.0")],
    ]

    engine = make_engine(workspace)
    seed(engine, ["000001"], yesterday)
    written = engine.sync_today_bulk()
    close = engine.get_ohlcv("000001").loc[lambda df: df["date"] == today, "close"].iloc[0]

    assert written == 1
    assert close == 13.0
    assert len(scripted.instances) >= 2
    assert scripted.closes >= 1


def test_one_symbol_failure_does_not_drop_the_rest(
    scripted, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """单只股票最终失败不影响其他股票，汇总日志报告失败数量和代码。"""
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    today = date.today().strftime("%Y-%m-%d")
    scripted.scripts["sz.000001"] = [
        ConnectionFailure("第一次", code="sz.000001"),
        ConnectionFailure("第二次", code="sz.000001"),
        ConnectionFailure("第三次", code="sz.000001"),
    ]
    scripted.scripts["sh.600000"] = [[bar(today, "21.0")]]
    messages: list[str] = []

    engine = make_engine(workspace)
    seed(engine, ["000001", "600000"], yesterday)
    monkeypatch.setattr(engine_module.logger, "error", messages.append)
    written = engine.sync_today_bulk()
    survivor = engine.get_ohlcv("600000")
    failed = engine.get_ohlcv("000001")

    assert written == 1
    assert survivor["date"].tolist() == [yesterday, today]
    assert failed["date"].tolist() == [yesterday]
    assert any("1" in message and "000001" in message for message in messages)


def test_login_failure_keeps_rows_already_fetched(scripted, workspace: Path) -> None:
    """登录失败会向上抛出；已经返回的其他股票仍写入，失败股不入库。"""
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    today = date.today().strftime("%Y-%m-%d")
    scripted.scripts["sz.000001"] = [LoginFailure("登录失败", code="sz.000001")]
    scripted.scripts["sh.600000"] = [[bar(today, "21.0")]]

    engine = make_engine(workspace)
    seed(engine, ["000001", "600000"], yesterday)
    with pytest.raises(LoginFailure):
        engine.sync_today_bulk()
    assert today not in engine.get_ohlcv("000001")["date"].tolist()
    assert engine.get_ohlcv("600000")["date"].tolist() == [yesterday, today]


def test_failed_symbol_rerun_does_not_erase_other_symbols(scripted, workspace: Path) -> None:
    """失败股补跑同一天时，只替换该股，不删掉当天已成功的其他股票。"""
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    today = date.today().strftime("%Y-%m-%d")
    scripted.scripts["sz.000001"] = [
        ConnectionFailure("第一次", code="sz.000001"),
        ConnectionFailure("第二次", code="sz.000001"),
        ConnectionFailure("第三次", code="sz.000001"),
    ]
    scripted.scripts["sh.600000"] = [[bar(today, "21.0")]]

    engine = make_engine(workspace)
    seed(engine, ["000001", "600000"], yesterday)
    engine.sync_today_bulk()
    scripted.scripts["sz.000001"] = [[bar(today, "13.0")]]
    engine.sync_today_bulk()

    assert engine.get_ohlcv("600000")["date"].tolist() == [yesterday, today]
    assert engine.get_ohlcv("000001").loc[lambda df: df["date"] == today, "close"].iloc[0] == 13.0


def test_symbol_fetch_retries_are_bounded(scripted) -> None:
    """单股拉取的重试次数是限定的，耗尽后抛出可识别失败。"""
    scripted.scripts["sz.000001"] = [ConnectionFailure("耗尽", code="sz.000001")]
    with pytest.raises(BaostockRequestError):
        _bs_fetch_symbol(("000001", "sz.000001", "2026-10-03", "2026-10-04"), max_retries=0)
