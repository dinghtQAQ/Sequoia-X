"""只用信号日已知的对数收益率，筛除与观察中股票高度正相关的新信号。"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrelationConfig:
    threshold: float = 0.8
    window: int = 60
    min_periods: int = 40

    def __post_init__(self) -> None:
        if not 0 < self.threshold <= 1:
            raise ValueError("相关性阈值必须大于0且不超过1")
        if (not isinstance(self.window, int) or not isinstance(self.min_periods, int)
                or self.window < 2 or not 2 <= self.min_periods <= self.window):
            raise ValueError("相关性窗口至少2日，共同观测数须在2与窗口长度之间")


class CorrelationHistory:
    def __init__(self, bars: pd.DataFrame, calendar: list[str], config: CorrelationConfig) -> None:
        self.config = config
        prices = bars.pivot(index="date", columns="symbol", values="close").reindex(calendar)
        volumes = bars.pivot(index="date", columns="symbol", values="volume").reindex(
            index=calendar, columns=prices.columns,
        )
        prices = prices.where((prices > 0) & (volumes > 0) & np.isfinite(prices))
        self.symbol_index = {symbol: index for index, symbol in enumerate(prices.columns)}
        self.returns = np.log(prices).diff().to_numpy()

    def matrix(
        self, index: int, candidates: list[str], references: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """成对剔除缺失后估计Pearson；索引右端包含信号日，绝不取下一日。"""
        window = self.returns[max(0, index - self.config.window + 1):index + 1]
        x = window[:, [self.symbol_index[symbol] for symbol in candidates]].T
        y = window[:, [self.symbol_index[symbol] for symbol in references]]
        valid_x = np.isfinite(x).astype(float)
        valid_y = np.isfinite(y).astype(float)
        x = np.nan_to_num(x, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        counts = valid_x @ valid_y
        sum_x, sum_y = x @ valid_y, valid_x @ y
        with np.errstate(divide="ignore", invalid="ignore"):
            xx = (x * x) @ valid_y - sum_x * sum_x / counts
            yy = valid_x @ (y * y) - sum_y * sum_y / counts
            covariance = x @ y - sum_x * sum_y / counts
            rho = covariance / np.sqrt(np.maximum(xx, 0) * np.maximum(yy, 0))
        valid = (counts >= self.config.min_periods) & (xx > 1e-15) & (yy > 1e-15)
        return np.where(valid, np.clip(rho, -1, 1), np.nan), counts.astype(int)


def select_uncorrelated_signals(
    signals: pd.DataFrame,
    date_index: dict[str, int],
    horizon: int,
    dedup_days: int,
    history: CorrelationHistory,
) -> pd.DataFrame:
    """按日期及MA排名重放独立选择状态；只有真正接受的信号才产生占用。"""
    decisions = pd.DataFrame({
        "corr_accepted": False,
        "corr_reason": "时间窗口内重复信号",
        "corr_peer": "",
        "corr_value": np.nan,
        "corr_observations": 0,
        "corr_unknown_pairs": 0,
        "corr_reference_count": 0,
    }, index=signals.index)
    blocked_until: dict[str, int] = {}
    active_until: dict[str, int] = {}
    ordered = signals.sort_values(["date", "ranking", "symbol"], ascending=[True, False, True])
    for signal_date, day in ordered.groupby("date", sort=False):
        index = date_index[signal_date]
        active_until = {symbol: end for symbol, end in active_until.items() if end >= index}
        eligible = day.loc[day["symbol"].map(lambda symbol: index > blocked_until.get(symbol, -1))]
        if eligible.empty:
            continue
        candidates = eligible["symbol"].tolist()
        references = sorted(set(active_until) | set(candidates))
        reference_index = {symbol: position for position, symbol in enumerate(references)}
        rho, counts = history.matrix(index, candidates, references)
        for position, row in enumerate(eligible.itertuples()):
            peers = sorted(active_until)
            columns = [reference_index[symbol] for symbol in peers]
            values = rho[position, columns]
            finite = np.isfinite(values)
            decisions.loc[row.Index, "corr_reference_count"] = len(peers)
            decisions.loc[row.Index, "corr_unknown_pairs"] = int((~finite).sum())
            maximum = np.nan
            if finite.any():
                best = int(np.nanargmax(values))
                maximum = float(values[best])
                decisions.loc[row.Index, "corr_peer"] = peers[best]
                decisions.loc[row.Index, "corr_value"] = maximum
                decisions.loc[row.Index, "corr_observations"] = int(counts[position, columns[best]])
            if np.isfinite(maximum) and maximum >= history.config.threshold:
                decisions.loc[row.Index, "corr_reason"] = "高相关"
                continue
            decisions.loc[row.Index, "corr_accepted"] = True
            decisions.loc[row.Index, "corr_reason"] = "保留"
            blocked_until[row.symbol] = index + dedup_days
            active_until[row.symbol] = index + horizon
    return decisions
