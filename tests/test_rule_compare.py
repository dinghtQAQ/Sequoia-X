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


def test_compare_reruns_one_strategy_and_writes_all_sixteen_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """一次比较只跑一个选股策略，并写出 4 条买入规则 × 4 条卖出规则共 16 个场景。

    后复权日线只在 2024-01-22 形成 MaVolume 金叉。不复权日线单独定价：
    信号日收盘 10、最高 10.20，最近 20 日收盘均线 10.20；次日开盘 10、最高 10.50。
    费用置零。总资产 50,000，风险 2% 为 1,000。收盘价买入的真实止损距离是
    10×8%=0.80，股数取整为 1,200，再被总资产五分之一 10,000 元限制为 1,000 股。
    次日开盘 11 未到 8% 止损或 15% 止盈，持有满 2 日按开盘卖出，单笔收益 0.10。
    没有成交的场景也保留，交易次数为 0。
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
        raw = _flat("2024-01-02", 20, close=10.2125) + [
            ("2024-01-22", 10.0, 12.0, 9.8, 10.0, 1000.0, 10_000.0),
            ("2024-01-23", 10.0, 10.5, 9.8, 10.2, 1000.0, 10_000.0),
            ("2024-01-24", 11.0, 11.2, 10.8, 11.0, 1000.0, 10_000.0),
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
            initial_cash=50_000,
            commission_rate=0,
            min_commission=0,
            stamp_duty=0,
            slippage=0,
            max_hold_days=2,
        )

        scorecard = _scorecard(results).read_text(encoding="utf-8")
        log = _log(results).read_text(encoding="utf-8")

    rows = scorecard.splitlines()
    assert rows[0] == (
        "场景编号,买入规则,卖出规则,收益,最大回撤,收益回撤比,中位收益,交易次数,胜率,"
        "平均每笔收益,无法买入次数,期末现金,剩余持仓收盘价值,相对初始资金收益,实际区间"
    )
    assert len(rows) == 17
    by_id = {row.split(",")[0]: row for row in rows[1:]}
    assert set(by_id) == {f"S{number:02d}" for number in range(1, 17)}
    close_stop = next(row for row in rows[1:] if ",信号日收盘价,8%止损+15%止盈+10个交易日," in row)
    assert close_stop.startswith("S05,")
    assert ",0.1000,0.0000,0.0000,0.1000,1,1.0000,0.1000,0,51000.0000,0.0000,0.0200," in close_stop
    high_rows = [row for row in rows[1:] if row.split(",")[1] == "信号日最高价"]
    assert high_rows
    assert all(",0,0.0000,0.0000,1," in row for row in high_rows)
    assert "S05 2024-01-22 sz000001 买入" in log
    assert "S05 2024-01-22 sz000001 卖出" in log


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
            commission_rate=0,
            min_commission=0,
            stamp_duty=0,
            slippage=0,
        )

        scorecard = _scorecard(results).read_text(encoding="utf-8")
        log = _log(results).read_text(encoding="utf-8")

    assert "无法买入次数" in scorecard
    assert scorecard.count(",1,") >= 1
    assert "2024-01-20至2024-01-23" in scorecard
    assert "S05 2024-01-22 sz000001 无法买入" in log


def test_default_costs_reduce_the_close_to_open_trade(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认费用从收盘买入、持有两日后开盘卖出中扣除，且四个费用参数可单独修改。

    手算：买入价 10 上浮 0.1% 为 10.01。总资产 50,000 的 2% 风险是 1,000，
    真实止损距离 10×8%=0.80，风险股数 1,200；五分之一上限 10,000 元只够 900 股。
    成本 9,009 + 最低佣金 5 = 9,014。卖出 11×0.999×900=9,890.1，印花税 4.94505，
    净回收 9,885.15495。交易成绩 (9,885.15495-9,014)/9,014 = 0.096644，四舍五入为 0.0967。
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
        raw = _flat("2024-01-02", 20, close=10.0) + [
            ("2024-01-22", 10.0, 10.0, 10.0, 10.0, 1000.0, 10_000.0),
            ("2024-01-23", 10.0, 10.5, 9.8, 10.2, 1000.0, 10_000.0),
            ("2024-01-24", 11.0, 11.2, 10.8, 11.0, 1000.0, 10_000.0),
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
            max_hold_days=2,
        )
        compare_rules(
            strategy="MaVolume",
            db_path=engine.db_path,
            results_dir=results / "free",
            run_at="2026-10-13 15:04:06",
            start="2024-01-20",
            end="2024-01-25",
            commission_rate=0,
            min_commission=0,
            stamp_duty=0,
            slippage=0,
            max_hold_days=2,
        )

        scorecard = _scorecard(results).read_text(encoding="utf-8")
        overridden = _scorecard(results / "free").read_text(encoding="utf-8")

    close_row = next(
        row for row in scorecard.splitlines() if ",信号日收盘价,8%止损+15%止盈+10个交易日," in row
    )
    overridden_row = next(
        row for row in overridden.splitlines() if ",信号日收盘价,8%止损+15%止盈+10个交易日," in row
    )
    assert close_row.split(",")[3] == "0.0967"
    assert overridden_row.split(",")[3] == "0.1000"


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
        )

        names = sorted(path.name for path in results.iterdir())

    assert names == [
        "MaVolume_20261013150405.csv",
        "MaVolume_20261013150405.log",
    ]


def test_four_buy_rules_fill_at_their_own_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    """四条买入规则各自成交，并按自己的价格计算同一卖出规则的收益。

    信号日最高 11、收盘 10、收盘下跌 2% 为 9.80。最近 20 个不复权收盘价之和
    为 200.50，20 日均线为 10.025。次日覆盖这些价格，第三日开盘全部卖在 12。
    费用置零，每条规则都买 100 股：最高价收益 0.0909，收盘价 0.2000，
    下跌 2% 为 0.2245。20 日均线 10.025 四舍五入到 10.03，收益为 0.1976。
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
        ]
        raw = _flat("2024-01-02", 20, close=10.025) + [
            ("2024-01-22", 10.0, 11.0, 9.5, 10.0, 1000.0, 10_000.0),
            ("2024-01-23", 9.0, 11.2, 9.0, 10.5, 1000.0, 10_000.0),
            ("2024-01-24", 12.0, 12.2, 11.8, 12.0, 1000.0, 10_000.0),
            ("2024-01-25", 12.0, 12.2, 11.8, 12.0, 1000.0, 10_000.0),
        ]
        _seed(engine, "stock_daily", "000001", adjusted)
        _seed(engine, "stock_daily_raw", "000001", raw)
        monkeypatch.setattr("sequoia_x.data.engine.BaostockSession", _NoNetwork)

        compare_rules(
            strategy="MaVolume",
            db_path=engine.db_path,
            results_dir=results,
            run_at="2026-10-13 15:04:05",
            start="2024-01-22",
            end="2024-01-25",
            commission_rate=0,
            min_commission=0,
            stamp_duty=0,
            slippage=0,
            max_hold_days=1,
            risk_fraction=0.002,
        )

        rows = _scorecard(results).read_text(encoding="utf-8").splitlines()

    stop_rows = [row for row in rows if "8%止损+15%止盈+10个交易日" in row]
    returns = {row.split(",")[1]: row.split(",")[3] for row in stop_rows}
    assert returns == {
        "信号日最高价": "0.0909",
        "信号日收盘价": "0.2000",
        "收盘价下跌2%": "0.2245",
        "20日均线": "0.1976",
    }


def test_scenarios_keep_independent_cash_and_size_from_total_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """每个场景从 50,000 元独立开始；股数先按 2% 风险，再受五分之一和现金限制。

    两只股票同一天入选。真实止损距离 0.80 元时，风险金额 1,000 对应 1,250 股，
    取整后 1,200 股，再被五分之一上限限制为 1,000 股。同一天只能新开排序更高的一只。
    现金只够 100 股时记为成交 100 股；连 100 股都不够时记为无法买入。
    区间结束按收盘价强制卖出，买卖价相同，所以交易成绩为 0，期末现金回到初始资金。
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_name:
        tmp_dir = Path(tmp_name)
        results = tmp_dir / "results"
        engine = DataEngine(_settings(tmp_dir))
        for symbol, turnover in (("000001", 200_000_000.0), ("000002", 100_000_000.0)):
            adjusted = _flat("2024-01-02", 16) + [
                ("2024-01-18", 10.0, 10.0, 10.0, 10.0, 1000.0, 10_000.0),
                ("2024-01-19", 9.0, 9.0, 9.0, 9.0, 1000.0, 10_000.0),
                ("2024-01-20", 8.0, 8.0, 8.0, 8.0, 1000.0, 10_000.0),
                ("2024-01-21", 7.0, 7.0, 7.0, 7.0, 1000.0, 10_000.0),
                ("2024-01-22", 20.0, 20.0, 20.0, 20.0, 2000.0, turnover),
                ("2024-01-23", 20.0, 20.0, 20.0, 20.0, 1000.0, 10_000.0),
            ]
            _seed(engine, "stock_daily", symbol, adjusted)
        raw = _flat("2024-01-02", 20, close=10.0) + [
            ("2024-01-22", 10.0, 10.0, 10.0, 10.0, 1000.0, 10_000.0),
            ("2024-01-23", 10.0, 10.0, 10.0, 10.0, 1000.0, 10_000.0),
            ("2024-01-24", 10.0, 10.2, 9.9, 10.1, 1000.0, 10_000.0),
        ]
        _seed(engine, "stock_daily_raw", "000001", raw)
        _seed(engine, "stock_daily_raw", "000002", raw)
        monkeypatch.setattr("sequoia_x.data.engine.BaostockSession", _NoNetwork)

        compare_rules(
            strategy="MaVolume",
            db_path=engine.db_path,
            results_dir=results,
            run_at="2026-10-13 15:04:05",
            start="2024-01-22",
            end="2024-01-23",
            commission_rate=0,
            min_commission=0,
            stamp_duty=0,
            slippage=0,
            max_hold_days=5,
        )
        full = _scorecard(results).read_text(encoding="utf-8")
        full_log = _log(results).read_text(encoding="utf-8")

        compare_rules(
            strategy="MaVolume",
            db_path=engine.db_path,
            results_dir=results / "tight",
            run_at="2026-10-13 15:04:06",
            start="2024-01-22",
            end="2024-01-23",
            initial_cash=6_000,
            commission_rate=0,
            min_commission=0,
            stamp_duty=0,
            slippage=0,
            max_hold_days=5,
        )
        tight = _scorecard(results / "tight").read_text(encoding="utf-8")
        tight_log = _log(results / "tight").read_text(encoding="utf-8")

        compare_rules(
            strategy="MaVolume",
            db_path=engine.db_path,
            results_dir=results / "broke",
            run_at="2026-10-13 15:04:07",
            start="2024-01-22",
            end="2024-01-23",
            initial_cash=900,
            commission_rate=0,
            min_commission=0,
            stamp_duty=0,
            slippage=0,
            max_hold_days=5,
        )
        broke = _scorecard(results / "broke").read_text(encoding="utf-8")

    full_row = next(
        row for row in full.splitlines() if ",信号日收盘价,8%止损+15%止盈+10个交易日," in row
    )
    full_cells = full_row.split(",")
    assert full_cells[7] == "1"
    assert full_cells[10] == "0"
    assert full_cells[12] == "10000.0000"
    assert "S05 2024-01-22 sz000001 买入" in full_log
    assert "sz000002 买入" not in full_log
    tight_row = next(
        row for row in tight.splitlines() if ",信号日收盘价,8%止损+15%止盈+10个交易日," in row
    )
    tight_cells = tight_row.split(",")
    assert tight_cells[7] == "1"
    assert tight_cells[10] == "0"
    assert tight_cells[11] == "5000.0000"
    assert "S05 2024-01-22 sz000001 买入" in tight_log
    broke_row = next(
        row for row in broke.splitlines() if ",信号日收盘价,8%止损+15%止盈+10个交易日," in row
    )
    assert ",0,0.0000,0.0000,2,900.0000,0.0000,0.0000," in broke_row


def test_same_day_buy_uses_cash_before_sale(monkeypatch: pytest.MonkeyPatch) -> None:
    """同一交易日先买入再卖出，当天卖出收回的现金不能再开新仓。

    初始现金 1,200 元，单只上限放开后第一只买 100 股，剩下 200 元。
    次日开盘止损卖在 9 元，回款 900 元，加上剩余现金够再买 100 股。
    第二只股票同一天入选，只能使用卖出前的 200 元，所以记为无法买入。
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_name:
        tmp_dir = Path(tmp_name)
        results = tmp_dir / "results"
        engine = DataEngine(_settings(tmp_dir))
        first = _flat("2024-01-02", 16) + [
            ("2024-01-18", 10.0, 10.0, 10.0, 10.0, 1000.0, 10_000.0),
            ("2024-01-19", 9.0, 9.0, 9.0, 9.0, 1000.0, 10_000.0),
            ("2024-01-20", 8.0, 8.0, 8.0, 8.0, 1000.0, 10_000.0),
            ("2024-01-21", 7.0, 7.0, 7.0, 7.0, 1000.0, 10_000.0),
            ("2024-01-22", 20.0, 20.0, 20.0, 20.0, 2000.0, 200_000_000.0),
            ("2024-01-23", 20.0, 20.0, 20.0, 20.0, 1000.0, 10_000.0),
            ("2024-01-24", 20.0, 20.0, 20.0, 20.0, 1000.0, 10_000.0),
            ("2024-01-25", 20.0, 20.0, 20.0, 20.0, 1000.0, 10_000.0),
        ]
        second = _flat("2024-01-02", 18) + [
            ("2024-01-20", 10.0, 10.0, 10.0, 10.0, 1000.0, 10_000.0),
            ("2024-01-21", 9.0, 9.0, 9.0, 9.0, 1000.0, 10_000.0),
            ("2024-01-22", 8.0, 8.0, 8.0, 8.0, 1000.0, 10_000.0),
            ("2024-01-23", 7.0, 7.0, 7.0, 7.0, 1000.0, 10_000.0),
            ("2024-01-24", 20.0, 20.0, 20.0, 20.0, 2000.0, 200_000_000.0),
            ("2024-01-25", 20.0, 20.0, 20.0, 20.0, 1000.0, 10_000.0),
        ]
        _seed(engine, "stock_daily", "000001", first)
        _seed(engine, "stock_daily", "000002", second)
        _seed(
            engine,
            "stock_daily_raw",
            "000001",
            _flat("2024-01-02", 20, close=10.0)
            + [
                ("2024-01-22", 10.0, 10.5, 9.5, 10.0, 1000.0, 10_000.0),
                ("2024-01-23", 10.0, 10.5, 9.8, 10.0, 1000.0, 10_000.0),
                ("2024-01-24", 10.0, 10.2, 9.8, 10.0, 1000.0, 10_000.0),
                ("2024-01-25", 9.0, 10.2, 8.8, 9.1, 1000.0, 10_000.0),
            ],
        )
        _seed(
            engine,
            "stock_daily_raw",
            "000002",
            _flat("2024-01-02", 22, close=10.0)
            + [
                ("2024-01-24", 10.0, 10.5, 9.5, 10.0, 1000.0, 10_000.0),
                ("2024-01-25", 10.0, 10.2, 9.8, 10.0, 1000.0, 10_000.0),
            ],
        )
        monkeypatch.setattr("sequoia_x.data.engine.BaostockSession", _NoNetwork)

        compare_rules(
            strategy="MaVolume",
            db_path=engine.db_path,
            results_dir=results,
            run_at="2026-10-13 15:04:05",
            start="2024-01-22",
            end="2024-01-25",
            initial_cash=1_200,
            commission_rate=0,
            min_commission=0,
            stamp_duty=0,
            slippage=0,
            risk_fraction=1,
            position_cap=1,
        )

        log = _log(results).read_text(encoding="utf-8")

    assert "S05 2024-01-22 sz000001 买入" in log
    assert "S05 2024-01-22 sz000001 卖出" in log
    assert "S05 2024-01-24 sz000002 无法买入" in log


def _return_for(results: Path, scenario_id: str) -> str:
    rows = _scorecard(results).read_text(encoding="utf-8").splitlines()
    row = next(item for item in rows if item.startswith(f"{scenario_id},"))
    return row.split(",")[3]


def test_stop_has_priority_over_gap_and_limit_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """最低价碰到止损就按止损卖；跳空越过止损按开盘价；开盘跌停也按开盘价卖出。

    信号日收盘 10 元，8% 止损价是 9.20。盘中最低 9.00、开盘仍是 10 时按 9.20 卖出。
    开盘 9.00 已经跌破止损时按开盘价卖出。前收 10、开盘 9.00 的开盘跌停也按开盘价卖出。
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_name:
        tmp_dir = Path(tmp_name)
        adjusted = _flat("2024-01-02", 16) + [
            ("2024-01-18", 10.0, 10.0, 10.0, 10.0, 1000.0, 10_000.0),
            ("2024-01-19", 9.0, 9.0, 9.0, 9.0, 1000.0, 10_000.0),
            ("2024-01-20", 8.0, 8.0, 8.0, 8.0, 1000.0, 10_000.0),
            ("2024-01-21", 7.0, 7.0, 7.0, 7.0, 1000.0, 10_000.0),
            ("2024-01-22", 20.0, 20.0, 20.0, 20.0, 2000.0, 200_000_000.0),
            ("2024-01-23", 20.0, 20.0, 20.0, 20.0, 1000.0, 10_000.0),
            ("2024-01-24", 20.0, 20.0, 20.0, 20.0, 1000.0, 10_000.0),
        ]
        exits = {
            "touch": ("2024-01-24", 10.0, 10.2, 9.0, 9.5),
            "gap": ("2024-01-24", 9.0, 9.2, 8.8, 9.1),
            "limit-down": ("2024-01-24", 9.0, 9.2, 8.8, 9.0),
        }
        monkeypatch.setattr("sequoia_x.data.engine.BaostockSession", _NoNetwork)
        returns = {}
        for name, exit_bar in exits.items():
            engine = DataEngine(_settings(tmp_dir / name))
            results = tmp_dir / name / "results"
            _seed(engine, "stock_daily", "000001", adjusted)
            _seed(
                engine,
                "stock_daily_raw",
                "000001",
                _flat("2024-01-02", 20, close=10.0)
                + [
                    ("2024-01-22", 10.0, 10.5, 9.5, 10.0, 1000.0, 10_000.0),
                    ("2024-01-23", 10.0, 10.5, 9.8, 10.0, 1000.0, 10_000.0),
                    (*exit_bar, 1000.0, 10_000.0),
                ],
            )
            compare_rules(
                strategy="MaVolume",
                db_path=engine.db_path,
                results_dir=results,
                run_at="2026-10-13 15:04:05",
                start="2024-01-22",
                end="2024-01-24",
                commission_rate=0,
                min_commission=0,
                stamp_duty=0,
                slippage=0,
                risk_fraction=0.002,
                position_cap=1,
                max_hold_days=1,
            )
            returns[name] = _return_for(results, "S05")

    assert returns == {"touch": "-0.0800", "gap": "-0.1000", "limit-down": "-0.1000"}


def test_ma_exit_sizes_with_atr_and_marks_interval_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """没有固定止损时，用 2 倍 14 日 ATR 计算股数；区间结束单独标记。

    不复权高低收都是 10，ATR 为 0。这里把信号日前 14 日的高低差固定为 1，
    ATR 为 1，2 倍 ATR 是 2。总资产 50,000 的 2% 风险对应 500 股。
    持有未触发均线卖出时，区间结束按最后收盘价强制卖出，日志标记区间结束。
    强制卖出前剩余持仓收盘价值是 500 股 × 10 元。
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
        ]
        raw = [
            (f"2024-01-{day:02d}", 10.0, 10.5, 9.5, 10.0, 1000.0, 10_000.0)
            for day in range(2, 22)
        ] + [
            ("2024-01-22", 10.0, 10.5, 9.5, 10.0, 1000.0, 10_000.0),
            ("2024-01-23", 10.0, 10.5, 9.8, 10.0, 1000.0, 10_000.0),
            ("2024-01-24", 10.0, 10.2, 9.9, 10.0, 1000.0, 10_000.0),
        ]
        _seed(engine, "stock_daily", "000001", adjusted)
        _seed(engine, "stock_daily_raw", "000001", raw)
        monkeypatch.setattr("sequoia_x.data.engine.BaostockSession", _NoNetwork)

        compare_rules(
            strategy="MaVolume",
            db_path=engine.db_path,
            results_dir=results,
            run_at="2026-10-13 15:04:05",
            start="2024-01-22",
            end="2024-01-24",
            commission_rate=0,
            min_commission=0,
            stamp_duty=0,
            slippage=0,
            max_hold_days=20,
        )

        row = next(
            item
            for item in _scorecard(results).read_text(encoding="utf-8").splitlines()
            if item.startswith("S16,")
        )
        log = _log(results).read_text(encoding="utf-8")

    assert row.split(",")[7] == "1"
    assert row.split(",")[12] == "5000.0000"
    assert "S16 2024-01-22 sz000001 区间结束" in log


def test_four_open_positions_block_the_fifth_pick(monkeypatch: pytest.MonkeyPatch) -> None:
    """最多同时持有 4 笔。第五只入选时记为无法买入，已有 4 笔不受影响。"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_name:
        tmp_dir = Path(tmp_name)
        results = tmp_dir / "results"
        engine = DataEngine(_settings(tmp_dir))
        signals = ["2024-01-22", "2024-01-23", "2024-01-24", "2024-01-25", "2024-01-26"]
        for symbol, signal in zip((f"{number:06d}" for number in range(1, 6)), signals):
            adjusted = {
                trade_date: bar
                for trade_date, *bar in _flat("2024-01-02", 16)
            }
            adjusted["2024-01-18"] = (10.0, 10.0, 10.0, 10.0, 1000.0, 10_000.0)
            adjusted["2024-01-19"] = (9.0, 9.0, 9.0, 9.0, 1000.0, 10_000.0)
            adjusted["2024-01-20"] = (8.0, 8.0, 8.0, 8.0, 1000.0, 10_000.0)
            adjusted["2024-01-21"] = (7.0, 7.0, 7.0, 7.0, 1000.0, 10_000.0)
            day = int(signal[-2:])
            adjusted[f"2024-01-{day - 4:02d}"] = (10.0, 10.0, 10.0, 10.0, 1000.0, 10_000.0)
            adjusted[f"2024-01-{day - 3:02d}"] = (9.0, 9.0, 9.0, 9.0, 1000.0, 10_000.0)
            adjusted[f"2024-01-{day - 2:02d}"] = (8.0, 8.0, 8.0, 8.0, 1000.0, 10_000.0)
            adjusted[f"2024-01-{day - 1:02d}"] = (7.0, 7.0, 7.0, 7.0, 1000.0, 10_000.0)
            adjusted[signal] = (20.0, 20.0, 20.0, 20.0, 2000.0, 200_000_000.0)
            adjusted["2024-01-27"] = (20.0, 20.0, 20.0, 20.0, 1000.0, 10_000.0)
            _seed(
                engine,
                "stock_daily",
                symbol,
                [(trade_date, *bar) for trade_date, bar in sorted(adjusted.items())],
            )
            raw = {
                trade_date: bar
                for trade_date, *bar in _flat("2024-01-02", 26, close=10.0)
            }
            raw[signal] = (10.0, 10.5, 9.5, 10.0, 1000.0, 10_000.0)
            raw[f"2024-01-{day + 1:02d}"] = (10.0, 10.5, 9.8, 10.0, 1000.0, 10_000.0)
            _seed(
                engine,
                "stock_daily_raw",
                symbol,
                [(trade_date, *bar) for trade_date, bar in sorted(raw.items())],
            )
        monkeypatch.setattr("sequoia_x.data.engine.BaostockSession", _NoNetwork)

        compare_rules(
            strategy="MaVolume",
            db_path=engine.db_path,
            results_dir=results,
            run_at="2026-10-13 15:04:05",
            start="2024-01-22",
            end="2024-01-27",
            commission_rate=0,
            min_commission=0,
            stamp_duty=0,
            slippage=0,
            risk_fraction=0.002,
            position_cap=1,
            max_hold_days=20,
        )

        log = _log(results).read_text(encoding="utf-8")

    assert "S05 2024-01-26 sz000005 无法买入" in log


def test_suspended_sell_waits_for_the_next_tradable_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """卖出日停牌时不成交，顺延到下一个可交易日的开盘价。"""
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
            ("2024-01-24", 20.0, 20.0, 20.0, 20.0, 0.0, 0.0),
            ("2024-01-25", 20.0, 20.0, 20.0, 20.0, 1000.0, 10_000.0),
        ]
        raw = _flat("2024-01-02", 20, close=10.0) + [
            ("2024-01-22", 10.0, 10.5, 9.5, 10.0, 1000.0, 10_000.0),
            ("2024-01-23", 10.0, 10.5, 9.8, 10.0, 1000.0, 10_000.0),
            ("2024-01-24", 8.0, 8.2, 7.8, 8.0, 0.0, 0.0),
            ("2024-01-25", 9.0, 9.2, 8.8, 9.1, 1000.0, 10_000.0),
        ]
        _seed(engine, "stock_daily", "000001", adjusted)
        _seed(engine, "stock_daily_raw", "000001", raw)
        monkeypatch.setattr("sequoia_x.data.engine.BaostockSession", _NoNetwork)

        compare_rules(
            strategy="MaVolume",
            db_path=engine.db_path,
            results_dir=results,
            run_at="2026-10-13 15:04:05",
            start="2024-01-22",
            end="2024-01-25",
            commission_rate=0,
            min_commission=0,
            stamp_duty=0,
            slippage=0,
            risk_fraction=0.002,
            position_cap=1,
            max_hold_days=1,
        )

        row = next(
            item
            for item in _scorecard(results).read_text(encoding="utf-8").splitlines()
            if item.startswith("S05,")
        )

    assert row.split(",")[3] == "-0.1000"


class _NoNetwork:
    def __init__(self, *args, **kwargs) -> None:
        raise AssertionError("规则比较不能下载行情")
