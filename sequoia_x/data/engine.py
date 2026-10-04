"""数据引擎模块：负责 SQLite 行情数据存储与 baostock 增量同步。"""

import queue
import sqlite3
import threading
from pathlib import Path

import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.baostock_session import (
    BaostockRequestError,
    BaostockSession,
    LoginFailure,
)

logger = get_logger(__name__)


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stock_daily (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol   TEXT    NOT NULL,
    date     TEXT    NOT NULL,
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,
    volume   REAL,
    turnover REAL,
    UNIQUE (symbol, date)
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_symbol_date ON stock_daily (symbol, date);
"""


_DEFAULT_SYNC_WORKERS = 4
_SYMBOL_FETCH_RETRIES = 2
_KLINE_FIELDS = "date,open,high,low,close,volume,amount"


def _bs_fetch_symbol(
    task: tuple[str, str, str, str],
    max_retries: int = _SYMBOL_FETCH_RETRIES,
) -> list[list]:
    """拉取单只股票。连接类失败先关闭并重建会话，再按限定次数重试。

    登录失败不重试，直接抛出，避免在未登录状态下继续查询。
    """
    symbol, bs_code, start, end = task
    attempts = max_retries + 1
    last_error: BaostockRequestError | None = None
    for attempt in range(attempts):
        session = BaostockSession(max_retries=0)
        try:
            return session.query_history_k_data_plus(
                bs_code,
                _KLINE_FIELDS,
                start,
                end,
                frequency="d",
                adjustflag="1",  # 后复权
            )
        except LoginFailure:
            raise
        except BaostockRequestError as exc:
            last_error = exc
            logger.warning(f"[{symbol}] {exc.kind}（第 {attempt + 1}/{attempts} 次）: {exc}")
            if attempt >= max_retries:
                raise
        finally:
            session.close()
    raise last_error or BaostockRequestError(f"[{symbol}] 拉取失败", code=bs_code)


def _collect_symbol_rows(
    tasks: list,
    worker_count: int,
) -> tuple[list, list[str], LoginFailure | None]:
    """有界并发拉取。单股请求失败或登录失败只记入失败列表，不丢其他股票。"""
    pending: queue.Queue = queue.Queue()
    for task in tasks:
        pending.put(task)
    results: queue.Queue = queue.Queue()

    def worker() -> None:
        while True:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            symbol = task[0]
            try:
                rows = _bs_fetch_symbol(task)
            except LoginFailure as exc:
                results.put(("login", symbol, exc))
                continue
            except BaostockRequestError as exc:
                results.put(("failed", symbol, exc))
                continue
            results.put(("ok", symbol, rows))

    threads = [threading.Thread(target=worker, name=f"sync-{index}") for index in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    rows_out: list = []
    failed: list[str] = []
    login_error: LoginFailure | None = None
    while not results.empty():
        kind, symbol, payload = results.get()
        if kind == "login":
            login_error = payload
            failed.append(symbol)
            logger.error(f"[{symbol}] 登录失败: {payload}")
            continue
        if kind == "failed":
            logger.error(f"[{symbol}] {payload.kind}: {payload}")
            failed.append(symbol)
            continue
        for row in payload:
            rows_out.append([symbol] + row)
    return rows_out, failed, login_error


class DataEngine:
    """行情数据引擎，负责 SQLite 存储和 baostock 数据同步。"""

    def __init__(self, settings: Settings) -> None:
        self.db_path: str = settings.db_path
        self.start_date: str = settings.start_date
        self._init_db()

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)
            conn.commit()
        logger.info(f"数据库初始化完成：{self.db_path}")

    def _get_last_date(self, symbol: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM stock_daily WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        return row[0] if row and row[0] else None

    def get_ohlcv(self, symbol: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(
                "SELECT * FROM stock_daily WHERE symbol = ? ORDER BY date",
                conn,
                params=(symbol,),
            )
        return df

    @staticmethod
    def _to_baostock_code(symbol: str) -> str:
        """将纯数字代码转为 baostock 格式：6/9开头 -> sh，其余 -> sz。"""
        prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
        return f"{prefix}.{symbol}"

    # ── 数据同步 ──

    def sync_today_bulk(self) -> int:
        """有界并发拉取增量数据（后复权），写入 SQLite。

        默认最多 4 个 worker。单只股票最终失败或登录失败不影响其他股票。
        若出现登录失败，已成功的结果写入后向调用方抛出。
        """
        from datetime import date, timedelta

        today_str = date.today().strftime("%Y-%m-%d")

        tasks = []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT symbol, MAX(date) FROM stock_daily GROUP BY symbol"
            ).fetchall()

        if not rows:
            logger.warning("本地无股票数据，请先执行 --backfill")
            return 0

        for symbol, last_date in rows:
            if last_date and last_date >= today_str:
                continue
            start = today_str
            if last_date:
                start = (date.fromisoformat(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
            tasks.append((symbol, self._to_baostock_code(symbol), start, today_str))

        if not tasks:
            logger.info("所有股票已是最新，无需更新")
            return 0

        n_workers = min(_DEFAULT_SYNC_WORKERS, len(tasks))
        logger.info(f"需要更新 {len(tasks)} 只股票，启动 {n_workers} 个 worker 并行拉取...")

        all_rows, failed_symbols, login_error = _collect_symbol_rows(tasks, n_workers)
        if failed_symbols:
            logger.error(
                f"sync_today_bulk: {len(failed_symbols)} 只股票拉取失败: {','.join(failed_symbols)}"
            )

        if not all_rows:
            logger.info("无新数据（可能非交易日）")
            if login_error is not None:
                raise login_error
            return 0

        df = pd.DataFrame(all_rows, columns=["symbol", "date", "open", "high", "low", "close", "volume", "turnover"])
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df[df["volume"] > 0]

        count = len(df)
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "DELETE FROM stock_daily WHERE symbol = ? AND date = ?",
                list(zip(df["symbol"], df["date"])),
            )
            df.to_sql("stock_daily", conn, if_exists="append", index=False, method="multi", chunksize=500)
            conn.commit()

        logger.info(f"sync_today_bulk: 写入 {count} 条数据")
        if login_error is not None:
            raise login_error
        return count

    def backfill(self, symbols: list[str]) -> None:
        """通过 baostock 批量回填历史日 K 线数据（后复权）。

        容错机制：
        - 单只股票超时或断连后关闭连接，最多重建 2 次再查
        - 每 200 只股票主动重连，避免长连接超时
        - 已入库的自动 skip，中断后可重跑续传
        """
        import time
        from datetime import date, timedelta

        from sequoia_x.data.baostock_session import (
            BaostockRequestError,
            BaostockSession,
            LoginFailure,
        )

        today_str = date.today().strftime("%Y-%m-%d")
        max_retries = 3
        reconnect_interval = 200  # 每处理 N 只股票重连一次
        session = BaostockSession(max_retries=max_retries - 1)

        success = 0
        skipped = 0
        failed = 0
        since_reconnect = 0

        try:
            for i, symbol in enumerate(symbols):
                last_date = self._get_last_date(symbol)
                if last_date and last_date >= today_str:
                    skipped += 1
                    if (i + 1) % 500 == 0:
                        logger.info(
                            f"已处理 {i + 1}/{len(symbols)}，"
                            f"成功 {success} 跳过 {skipped} 失败 {failed}"
                        )
                    continue

                # 定期重连，防止长连接超时
                since_reconnect += 1
                if since_reconnect >= reconnect_interval:
                    session.close()
                    time.sleep(1)
                    since_reconnect = 0

                start = last_date or self.start_date
                if last_date:
                    start = (date.fromisoformat(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")

                bs_code = self._to_baostock_code(symbol)

                try:
                    rows = session.query_history_k_data_plus(
                        bs_code,
                        "date,open,high,low,close,volume,amount",
                        start,
                        today_str,
                        frequency="d",
                        adjustflag="1",  # 后复权
                    )
                except LoginFailure as exc:
                    logger.error(f"[{symbol}] 登录失败，终止回填: {exc}")
                    return
                except BaostockRequestError as exc:
                    logger.error(f"[{symbol}] {exc.kind}: {exc}")
                    failed += 1
                    continue

                if not rows:
                    skipped += 1
                    continue

                df = pd.DataFrame(
                    rows,
                    columns=["date", "open", "high", "low", "close", "volume", "amount"],
                )
                for col in ["open", "high", "low", "close", "volume", "amount"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["close"])
                df = df[df["volume"] > 0]

                if df.empty:
                    skipped += 1
                    continue

                df["symbol"] = symbol
                df = df.rename(columns={"amount": "turnover"})
                df = df[["symbol", "date", "open", "high", "low", "close", "volume", "turnover"]]

                try:
                    with sqlite3.connect(self.db_path) as conn:
                        df.to_sql(
                            "stock_daily", conn, if_exists="append",
                            index=False, method="multi", chunksize=500,
                        )
                except sqlite3.IntegrityError:
                    pass

                success += 1

                if (i + 1) % 500 == 0:
                    logger.info(
                        f"已处理 {i + 1}/{len(symbols)}，"
                        f"成功 {success} 跳过 {skipped} 失败 {failed}"
                    )

        finally:
            session.close()

        logger.info(f"回填完成 — 成功: {success} | 跳过: {skipped} | 失败: {failed}")

    # ── 股票列表 ──

    def get_all_symbols(self) -> list[str]:
        """通过 baostock 获取全市场 A 股代码列表。"""
        from sequoia_x.data.baostock_session import BaostockRequestError, BaostockSession

        session = BaostockSession(max_retries=0)
        try:
            rows = session.query_stock_basic()
        except BaostockRequestError as exc:
            logger.error(f"获取股票列表失败: {exc.kind}: {exc}")
            return []
        finally:
            session.close()

        symbols = []
        for row in rows:
            code = row[0]           # "sh.600000" or "sz.000001"
            status = row[4]         # "1" = 上市
            stock_type = row[5]     # "1" = 股票
            if status == "1" and stock_type == "1":
                symbols.append(code.split(".")[1])  # 提取纯数字代码
        logger.info(f"获取股票列表完成，共 {len(symbols)} 只")
        return symbols

    def get_local_symbols(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM stock_daily"
            ).fetchall()
        return [row[0] for row in rows]
