# Sequoia-X: 王者回归 | The King Returns

> A 股量化选股系统 V2 | A-Share Quantitative Stock Selection System V2

---
当前优先复现 [Issue #128](https://github.com/sngyai/Sequoia-X/issues/128) 的 MA 信号统计。
研究纳入全部金叉放量信号，按固定交易日统计后续收益，已移除原来的买卖规则组合和资金账户模拟。
---

## 简介 | Introduction

Sequoia-X V2 是面向 A 股市场的量化选股系统，基于现代 Python 工程化标准从零重构。
系统以 OOP 架构、向量化计算和增量数据更新为核心设计原则，每日收盘后自动选股并推送至 Bark。

数据层使用 [baostock](http://baostock.com)（免费、无需注册、无限流）拉取历史及增量日 K 数据。
后复权日线用于选股和信号收益研究，不复权日线保留供价格核对，两种日线分开保存。
存储于本地 SQLite，彻底规避东方财富反爬问题。

---

## 运行模式

```bash
python main.py               # 日常模式：增量补数据 + 跑策略 + Bark 推送（2~3分钟）
python main.py --backfill     # 回填模式：全市场历史K线一次性灌入（约12分钟）
python main.py --study MaVolume --start 2025-06-01 --end 2026-09-30
                              # 信号研究：默认逐一统计 20 至 60 个交易日
python main.py --study MaVolume --start 2025-06-01 --end 2026-09-30 --horizons 20 40 60
                              # 也可以只观察指定周期；--compare 是 --study 的兼容别名
python main.py --study MaVolume --start 2025-06-01 --end 2026-09-30 --horizons 5 20 60 --dedup-days 120
                              # 固定120日去重，收益仍观察5/20/60日；文件名附加_dedup120
python main.py --study MaVolume --start 2025-06-01 --end 2026-09-30 --horizons 5 20 60 --dedup-days 120 --corr-threshold 0.8
                              # 再加相关性去重：默认60日对数收益率、至少40个共同观测
```

研究只读本地 `stock_daily`，不联网补数据或推送通知，不要求存在不复权日线。
信号日区间之外的数据仍用于计算历史均线和完整的后续观察窗口。

每个信号的 n 日收益为 `close(D+n) / close(D) - 1`，使用同一份后复权收盘价。
MA 条件保持为：昨日 MA5 < MA20，今日 MA5 > MA20，今日量 > 含当日的 20 日均量 × 1.5。
研究不设止盈止损、均线退出、费用、资金或仓位限制，也不模拟涨跌停能否成交。
全部信号与仅时间去重口径不按排名筛选；可选相关性口径会使用排名确定接受顺序。
研究同时输出“全部信号”基准和“持有期去重”：各观察周期独立处理，区间开始时无占用，
接受 D 日信号后，同股在 D 至 D+n 日内不再接受新信号，D+n+1 起恢复，不另加卖出后冷却期。
被拒绝的信号不延长占用。去重先按历史顺序决定，不用未来收益或目标价格是否缺失来选择信号。
可用 `--dedup-days` 将去重窗口独立设为更长的固定长度，不能短于最长观察周期。
例如固定120日去重时，接受D日信号后到D+120日均拒绝同股新信号，D+121日起恢复；收益目标日仍为D+5、D+20、D+60。

`--corr-threshold` 显式启用相关性去重，并新增“时间去重+相关性”对照。
按同一天的MA排序值降序、同分代码升序，依次检查新股票与当天已接受、以及仍在各自观察期内股票的相关性。
只比较截至信号日的对数收益率 `ln(close_t / close_t-1)`，用共同有效观测计算Pearson相关系数；只过滤高正相关。
停牌、缺失日不前向填充。共同观测不足或零方差记为未知，不填成0，也不单凭未知拒绝；明细保存未知对数。
可用 `--corr-window` 和 `--corr-min-periods` 修改历史窗口与最少共同观测数，默认分别为60和40。
组合规则独立逐日重放，只有实际接受的信号触发时间锁定；观察期结束即移出相关性比较，不等待更长的时间冷却窗口结束。
此时排名会实际影响入选，明细记录MA排序值、比较股票、最大已知相关系数、观测数和拒绝原因。

`results` 中输出汇总 `.csv`、逐信号逐周期的 `_samples.csv` 和记录运行口径的 `_meta.json`。
每种去重口径均包含各周期有效样本和所有请求周期的共同有效样本，分别给出接受与过滤数量、胜率、平均收益和标准中位数。
相关性启用时同时保留全部信号、仅时间去重、时间加相关性三种独立口径，便于比较筛选带来的变化。
明细中的“纳入去重样本”和“去重说明”区分被保留与被拒绝的信号，并保存实际去重窗口。
各周期分别去重会产生不同的信号集合，比较周期效果时应同时查看共同有效样本。
这里按观察期模拟持仓占用；实际执行时，是否禁止重复买入应以真实成交与持仓记录为准。
收益率以小数保存，例如 `0.0352` 表示 `3.52%`。
缺少后续交易日或目标日价格的样本在明细中标注原因，不补值、不顺延、不提前平仓。

作者评论确认 short/mid/long 为 5/20/60 日，但未公开完整的股票池、去重窗口和入场价格口径。
因此当前实现提供可重复核对的研究定义，不保证复算出相同数字，也不能由原帖认定 60 日是最优周期。
MA 排序公式和研究细节见 [研究说明](docs/research/ma-volume-entry.md)。

---

## 内置策略 | Strategies

| 策略 | 说明 |
|---|---|
| **TurtleTrade** | 海龟突破：20日新高 + 成交额过亿 + 阳线防诱多，按涨幅排序 |
| **MaVolume** | 均线+放量突破 |
| **HighTightFlag** | 高而窄的旗形整理突破 |
| **LimitUpShakeout** | 涨停洗盘回踩确认 |
| **UptrendLimitDown** | 上升趋势中的跌停反包 |
| **RpsBreakout** | 欧奈尔 RPS 相对强度突破 |

---

## 快速开始 | Quick Start

### 环境要求

- Python >= 3.10

### 1. 安装依赖

```bash
# 推荐使用 uv（快速包管理器）
uv sync

# 或者 pip
pip install .
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填写 Bark 设备 Key
```

### 3. 首次回填历史数据

```bash
python main.py --backfill
```

一次回填同时保存同一批股票的后复权日线和不复权日线。

### 4. 日常运行

```bash
python main.py
```

建议配合 crontab 每个交易日收盘后自动执行：

```cron
15 19 * * 1-5 cd /root/Sequoia-X && .venv/bin/python main.py >> log.txt 2>&1
```

---

## 目录结构 | Project Structure

```
Sequoia-X/
├── main.py                      # 入口：日常/回填/信号研究模式
├── pyproject.toml               # 依赖声明 + ruff/pytest 配置
├── .env.example                 # 环境变量模板
├── data/                        # SQLite 数据库（运行时生成，不入 git）
├── sequoia_x/
│   ├── signal_study.py          # MA 信号在固定交易日后的收益统计
│   ├── core/
│   │   ├── config.py            # Pydantic-settings 配置管理
│   │   └── logger.py            # rich 结构化日志
│   ├── data/
│   │   └── engine.py            # 数据引擎（baostock 回填 + 增量同步 + SQLite）
│   ├── strategy/
│   │   ├── base.py              # 策略抽象基类
│   │   ├── turtle_trade.py      # 海龟交易策略
│   │   ├── ma_volume.py         # 均线放量策略
│   │   ├── high_tight_flag.py   # 高窄旗形策略
│   │   ├── limit_up_shakeout.py # 涨停洗盘策略
│   │   ├── uptrend_limit_down.py # 上升跌停策略
│   │   └── rps_breakout.py      # RPS 突破策略
│   └── notify/
│       ├── bark.py              # Bark 推送
│       └── feishu.py            # 飞书 Webhook 推送（保留，主流程不再使用）
└── tests/                       # 属性测试（hypothesis）
```

---

## 数据说明

- **数据源**：[baostock](http://baostock.com)（免费、无需注册、无限流）
- **复权方式**：后复权日线（`stock_daily`）和不复权日线（`stock_daily_raw`）分开保存，重复回填不互相覆盖
- **存储**：本地 SQLite（`data/sequoia_v2.db`），可直接拷贝到其他机器使用
- **日常增量**：8 进程并行通过 baostock 拉取，2~3 分钟完成全市场更新

---

## 许可证 | License

MIT
