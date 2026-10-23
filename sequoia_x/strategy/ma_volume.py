"""均线+成交量选股策略：5日均线上穿20日均线且成交量放大。"""

import pandas as pd

from sequoia_x.core.logger import get_logger
from sequoia_x.strategy.base import BaseStrategy

logger = get_logger(__name__)


def ma_volume_ranking(turnover: float, close: float, ma20: float) -> float:
    """日常展示和相关性去重共用的MA排序公式。"""
    return float(turnover) * (float(close) / float(ma20) - 1)


def find_ma_volume_signals(bars: pd.DataFrame) -> pd.DataFrame:
    """找出各股票历史上的金叉放量行，附 MA20；仅使用该行及以前的数据。"""
    history = bars.sort_values(["symbol", "date"]).reset_index(drop=True)
    groups = history.groupby("symbol", sort=False)
    ma5 = groups["close"].transform(lambda values: values.rolling(5).mean())
    ma20 = groups["close"].transform(lambda values: values.rolling(20).mean())
    volume_ma20 = groups["volume"].transform(lambda values: values.rolling(20).mean())
    previous_ma5 = ma5.groupby(history["symbol"], sort=False).shift(1)
    previous_ma20 = ma20.groupby(history["symbol"], sort=False).shift(1)
    mask = (
        (previous_ma5 < previous_ma20)
        & (ma5 > ma20)
        & (history["volume"] > volume_ma20 * 1.5)
    )
    return history.loc[mask].assign(ma20=ma20.loc[mask])


class MaVolumeStrategy(BaseStrategy):
    """均线+成交量选股策略。

    选股条件（全部向量化，严禁 iterrows）：
    1. 5日收盘均线上穿20日收盘均线（金叉）
    2. 当日成交量 > 20日均量的 1.5 倍（放量确认）

    Attributes:
        webhook_key: Bark 分组 'ma_volume'。
    """

    webhook_key: str = "ma_volume"

    def run(self) -> list[tuple[str, float]]:
        """
        遍历全市场，返回满足均线金叉+放量条件的股票及排序值。

        Returns:
            按排序值从高到低排列的 (股票代码, 排序值)。
            排序值是信号日成交额乘以收盘价相对 20 日均线的距离。
        """
        symbols = self.engine.get_local_symbols()
        selected: list[tuple[str, float]] = []

        for symbol in symbols:
            try:
                df = self.engine.get_ohlcv(symbol)
                if len(df) < 20:
                    continue

                signals = find_ma_volume_signals(df)
                if signals.empty or signals.iloc[-1]["date"] != df.iloc[-1]["date"]:
                    continue
                last = signals.iloc[-1]
                ranking = ma_volume_ranking(last["turnover"], last["close"], last["ma20"])
                selected.append((symbol, ranking))

            except Exception as exc:
                logger.warning(f"[{symbol}] 策略计算失败：{exc}")
                continue

        selected.sort(key=lambda item: (-item[1], item[0]))
        logger.info(f"MaVolumeStrategy 选出 {len(selected)} 只股票")
        return selected
