"""用全部 MA 信号统计固定交易日后的收益，不模拟委托或资金账户。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from sequoia_x.strategy.ma_volume import find_ma_volume_signals

DEFAULT_HORIZONS = tuple(range(20, 61))


class SignalStudyError(ValueError):
    """研究参数无效，或本地后复权日线不足。"""


def study_signals(
    strategy: str,
    db_path: str,
    results_dir: str | Path,
    run_at: str,
    start: str,
    end: str,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    dedup_days: int | None = None,
) -> Path:
    """按信号日限定样本，用区间外的后续日线补足观察窗口，返回汇总文件。"""
    if strategy != "MaVolume":
        raise SignalStudyError("信号收益研究当前只支持 MaVolume")
    try:
        start = date.fromisoformat(start).isoformat()
        end = date.fromisoformat(end).isoformat()
        stamp = datetime.strptime(run_at, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H%M%S")
    except ValueError as exc:
        raise SignalStudyError("日期参数无效") from exc
    if start > end:
        raise SignalStudyError("信号开始日期不能晚于结束日期")
    if not horizons or any(not isinstance(item, int) or item <= 0 for item in horizons):
        raise SignalStudyError("观察周期必须为正整数交易日")
    horizons = sorted(set(horizons))
    if dedup_days is not None:
        if not isinstance(dedup_days, int) or dedup_days <= 0:
            raise SignalStudyError("固定去重窗口必须为正整数交易日")
        if dedup_days < max(horizons):
            raise SignalStudyError("固定去重窗口不能短于最长观察周期")

    bars = _read_bars(db_path)
    calendar = sorted(bars["date"].unique())
    if not any(start <= item <= end for item in calendar):
        raise SignalStudyError("信号区间内没有本地交易日")
    signals = find_ma_volume_signals(bars)
    signals = signals.loc[signals["date"].between(start, end)].reset_index(drop=True)
    observations = _observe(bars, signals, calendar, horizons, dedup_days)

    complete = observations.loc[observations["status"] == "完整"]
    dedup_policy = "持有期去重" if dedup_days is None else "固定窗口去重"
    cohorts = {
        "全部信号": complete,
        dedup_policy: complete.loc[complete["accepted"]],
    }
    common_ids = {}
    for policy, cohort in cohorts.items():
        counts = cohort.groupby("signal_id")["horizon"].nunique()
        common_ids[policy] = counts.index[counts == len(horizons)]
    accepted_counts = observations.loc[observations["accepted"]].groupby("horizon").size()
    summary = []
    for horizon in horizons:
        for policy, cohort in cohorts.items():
            rows = cohort.loc[cohort["horizon"] == horizon]
            accepted = len(signals) if policy == "全部信号" else int(accepted_counts.get(horizon, 0))
            for label, selected in (
                ("各周期有效样本", rows),
                ("共同有效样本", rows.loc[rows["signal_id"].isin(common_ids[policy])]),
            ):
                returns = selected["return"]
                summary.append({
                    "观察周期": horizon,
                    "去重口径": policy,
                    "去重窗口": 0 if policy == "全部信号" else (dedup_days or horizon),
                    "样本口径": label,
                    "总信号数": len(signals),
                    "接受信号数": accepted,
                    "重复过滤数": len(signals) - accepted,
                    "有效样本数": len(returns),
                    "未纳入样本数": len(signals) - len(returns),
                    "胜率": float((returns > 0).mean()) if len(returns) else None,
                    "平均收益": returns.mean(),
                    "中位收益": returns.median(),
                    "信号开始": start,
                    "信号结束": end,
                    "数据截至": calendar[-1],
                })

    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{strategy}_signal_study_{stamp}"
    if dedup_days is not None:
        stem += f"_dedup{dedup_days}"
    summary_path = output / f"{stem}.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False, encoding="utf-8-sig")
    details = observations.drop(columns="signal_id").rename(columns={
        "symbol": "股票代码", "date": "信号日", "close": "信号日收盘价",
        "horizon": "观察周期", "target_date": "目标日", "target_close": "目标日收盘价",
        "return": "收益率", "status": "状态", "accepted": "纳入去重样本",
        "dedup_days": "去重窗口",
    })
    details["去重说明"] = details["纳入去重样本"].map({True: "保留", False: "去重窗口内重复信号"})
    details["股票代码"] = details["股票代码"].map(
        lambda symbol: ("sh" if symbol.startswith(("6", "9")) else "sz") + symbol
    )
    details.to_csv(output / f"{stem}_samples.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "strategy": strategy,
        "signal_start": start,
        "signal_end": end,
        "data_start": calendar[0],
        "data_end": calendar[-1],
        "horizons": horizons,
        "total_signals": len(signals),
        "common_samples": len(common_ids["全部信号"]),
        "deduplicated_common_samples": len(common_ids[dedup_policy]),
        "dedup_days": dedup_days,
        "effective_dedup_days_by_horizon": {str(item): dedup_days or item for item in horizons},
        "accepted_signals_by_horizon": {str(item): int(accepted_counts.get(item, 0)) for item in horizons},
        "price_basis": "stock_daily 后复权收盘价",
        "formula": "close(D+n) / close(D) - 1",
        "signal": "昨日 MA5 < MA20，今日 MA5 > MA20，今日量 > 含当日的20日均量×1.5",
        "calendar": "stock_daily 中全市场日期的有序并集，D+n 为第 n 个后续交易日",
        "deduplication": "保留全部信号基准；各周期初始无占用。默认去重窗口n等于各自观察周期，可指定不短于最长观察周期的固定窗口。接受D日信号后，D至D+n日闭区间内不再接受同股，D+n+1起恢复；拒绝信号不延长窗口",
        "selection_timing": "先按历史信号顺序决定去重，不使用后续价格有效性或收益决定是否接受信号",
        "common_cohort": "在各自去重口径下，所有请求周期均接受且拥有有效目标价格的相同信号集合",
        "missing_prices": "不补值、不顺延、不在区间末尾提前退出；明细标注原因",
        "filters": "无排名截取、止盈止损、均线退出、仓位、资金、费用或成交过滤",
        "limitations": "研究价格变化，不代表可成交收益；原帖股票池、去重窗口和价格口径未公开",
        "run_at": run_at,
    }
    (output / f"{stem}_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary_path


def _read_bars(db_path: str) -> pd.DataFrame:
    db_file = Path(db_path).resolve()
    if not db_file.is_file():
        raise SignalStudyError(f"缺少本地数据库：{db_path}")
    try:
        with closing(sqlite3.connect(db_file.as_uri() + "?mode=ro", uri=True)) as connection:
            bars = pd.read_sql_query(
                "SELECT symbol, date, close, volume FROM stock_daily ORDER BY symbol, date",
                connection,
            )
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        raise SignalStudyError("缺少研究所需的本地后复权日线") from exc
    if bars.empty:
        raise SignalStudyError("缺少研究所需的本地后复权日线")
    return bars


def _observe(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    calendar: list[str],
    horizons: Sequence[int],
    dedup_days: int | None = None,
) -> pd.DataFrame:
    date_index = {value: index for index, value in enumerate(calendar)}
    prices = bars[["symbol", "date", "close"]].rename(
        columns={"date": "target_date", "close": "target_close"}
    )
    observations = []
    for horizon in horizons:
        rows = signals[["symbol", "date", "close"]].copy()
        rows["signal_id"] = rows.index
        rows["horizon"] = horizon
        rows["dedup_days"] = dedup_days or horizon
        rows["accepted"] = _non_overlapping_mask(signals, date_index, dedup_days or horizon)
        rows["target_date"] = rows["date"].map(
            lambda value: calendar[date_index[value] + horizon]
            if date_index[value] + horizon < len(calendar) else None
        )
        rows = rows.merge(prices, on=["symbol", "target_date"], how="left", validate="many_to_one")
        valid = (rows["close"] > 0) & (rows["target_close"] > 0)
        rows["return"] = (rows["target_close"] / rows["close"] - 1).where(valid)
        rows["status"] = "完整"
        rows.loc[~valid, "status"] = "价格无效"
        rows.loc[rows["target_close"].isna(), "status"] = "目标日缺少价格"
        rows.loc[rows["target_date"].isna(), "status"] = "后续交易日不足"
        observations.append(rows)
    return pd.concat(observations, ignore_index=True)


def _non_overlapping_mask(
    signals: pd.DataFrame, date_index: dict[str, int], horizon: int,
) -> pd.Series:
    """只用已接受信号锁定该股观察期，期满的下一交易日恢复。"""
    blocked_until: dict[str, int] = {}
    accepted = set()
    for row in signals.sort_values(["date", "symbol"]).itertuples():
        index = date_index[row.date]
        if index <= blocked_until.get(row.symbol, -1):
            continue
        accepted.add(row.Index)
        blocked_until[row.symbol] = index + horizon
    return pd.Series(signals.index.isin(accepted), index=signals.index, dtype=bool)
