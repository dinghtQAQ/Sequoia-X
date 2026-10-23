"""Sequoia-X V2 主程序入口。

三种运行模式：
  python main.py               # 日常模式：最多 4 个 worker 增量补数据 + 跑策略 + Bark 推送
  python main.py --backfill    # 回填模式：同时保存后复权日线和不复权日线
  python main.py --study MaVolume --start 2025-06-01 --end 2026-09-30
                              # 本地 MA 信号后续收益研究，默认 20–60 个交易日
"""

import argparse
import sys
from dotenv import load_dotenv
load_dotenv()

from datetime import date

import socket
socket.setdefaulttimeout(10.0)

from sequoia_x.core.config import get_settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine
from sequoia_x.notify.bark import BarkNotifier
from sequoia_x.strategy.base import BaseStrategy
from sequoia_x.strategy.high_tight_flag import HighTightFlagStrategy
from sequoia_x.strategy.limit_up_shakeout import LimitUpShakeoutStrategy
from sequoia_x.strategy.ma_volume import MaVolumeStrategy
from sequoia_x.strategy.turtle_trade import TurtleTradeStrategy
from sequoia_x.strategy.uptrend_limit_down import UptrendLimitDownStrategy
from sequoia_x.strategy.rps_breakout import RpsBreakoutStrategy
from sequoia_x.strategy.private_placement import PrivatePlacementStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequoia-X V2 选股系统")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="回填模式：同时保存后复权日线和不复权日线",
    )
    parser.add_argument(
        "--study", "--compare", dest="study", metavar="STRATEGY",
        help="信号收益研究：统计全部 MA 信号的固定周期收益（--compare 为兼容别名）",
    )
    parser.add_argument("--start", help="信号区间开始日期")
    parser.add_argument("--end", help="信号区间结束日期")
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=tuple(range(20, 61)),
        help="观察周期，单位为交易日，默认逐一统计 20 至 60 日",
    )
    parser.add_argument(
        "--dedup-days", type=int,
        help="固定同股去重窗口，单位为交易日，不能短于最长观察周期；默认随各观察周期",
    )
    parser.add_argument("--corr-threshold", type=float, help="启用对数收益相关性去重，如0.8")
    parser.add_argument("--corr-window", type=int, default=60, help="相关性历史窗口，默认60个交易日")
    parser.add_argument("--corr-min-periods", type=int, default=40, help="最少共同收益观测数，默认40")
    args = parser.parse_args()

    try:
        # 1. 初始化配置
        settings = get_settings()

        # 2. 初始化日志
        logger = get_logger(__name__)
        logger.info("Sequoia-X V2 启动")

        if args.study:
            from datetime import datetime

            from sequoia_x.signal_study import study_signals

            summary_path = study_signals(
                strategy=args.study,
                db_path=settings.db_path,
                results_dir="results",
                run_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                start=args.start or settings.start_date,
                end=args.end or date.today().isoformat(),
                horizons=args.horizons,
                dedup_days=args.dedup_days,
                corr_threshold=args.corr_threshold,
                corr_window=args.corr_window,
                corr_min_periods=args.corr_min_periods,
            )
            logger.info(f"信号收益研究完成，汇总写入 {summary_path}")
            return

        # 3. 初始化数据引擎
        engine = DataEngine(settings)

        if args.backfill:
            # ── 回填模式：单线程保守拉历史 K 线，自动多轮重跑 ──
            logger.info("进入回填模式...")
            all_symbols = engine.get_all_symbols()
            engine.backfill(all_symbols)
            logger.info("Sequoia-X V2 回填模式运行完成")
            return

        # ── 日常模式：单次 API 补今天 + 策略 + 推送 ──
        logger.info("开始拉取最新快照...")
        count = engine.sync_today_bulk()
        logger.info(f"快照同步完成，写入 {count} 只股票")

        # 4. 策略列表（新增策略在此追加即可）
        strategies: list[BaseStrategy] = [
            MaVolumeStrategy(engine=engine, settings=settings),
            TurtleTradeStrategy(engine=engine, settings=settings),
            HighTightFlagStrategy(engine=engine, settings=settings),
            LimitUpShakeoutStrategy(engine=engine, settings=settings),
            UptrendLimitDownStrategy(engine=engine, settings=settings),
            RpsBreakoutStrategy(engine=engine, settings=settings),
            PrivatePlacementStrategy(engine=engine, settings=settings),
        ]

        notifier = BarkNotifier(settings)

        # 5. 遍历策略，有结果则按策略分组推送
        for strategy in strategies:
            strategy_name = type(strategy).__name__
            logger.info(f"执行策略：{strategy_name}")

            selected = strategy.run()
            symbols = (
                [symbol for symbol, _ranking in selected]
                if isinstance(strategy, MaVolumeStrategy)
                else selected
            )
            logger.info(f"{strategy_name} 选出 {len(symbols)} 只股票")

            if symbols:
                notifier.send(
                    symbols=symbols,
                    strategy_name=strategy_name,
                    webhook_key=strategy.webhook_key,
                )
            else:
                logger.info(f"{strategy_name} 无选股结果，跳过推送")

    except Exception:
        try:
            _logger = get_logger(__name__)
            _logger.exception("主流程发生未捕获异常，程序终止")
        except Exception:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    logger.info("Sequoia-X V2 运行完成")


if __name__ == "__main__":
    main()
