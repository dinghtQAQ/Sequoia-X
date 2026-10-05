# MA+成交量策略：名单与买入点位研究笔记

> 研究范围：仅核查本地 MA 入口及其直接的数据同步、调用、通知输出路径；不修改程序、不实际交易。源码事实均附文件与行号。Issue 内容按主代理通过 Chrome 核对的正文记录。

## 结论先行

1. **当日名单本质上只是日线选股信号，不是买入指令。** `MaVolumeStrategy.run()` 的返回类型是 `list[str]`，只返回股票代码；没有信号日期、成交价、委托方式或仓位信息（`sequoia_x/strategy/ma_volume.py:23-29,54-63`）。
2. **当前实现没有定义实际成交价，也没有定义止损/出场规则。** 主流程把代码列表原样交给通知器，通知内容只有日期、策略、数量、股票代码和雪球链接，没有价格或订单规则（`main.py:78-93`；`sequoia_x/notify/bark.py:68-83`）。
3. **名单的信号数据可能不是今天。** 策略取每只股票本地数据库中按 `date` 排序后的最后两行，而不是检查最后一行日期是否等于今天（`sequoia_x/data/engine.py:156-163`；`sequoia_x/strategy/ma_volume.py:33-46`）。同步在非交易日或无新数据时会返回 0，但主流程仍继续执行策略；单只股票同步失败时，其旧数据也可能继续被读取（`sequoia_x/data/engine.py:173-218`；`main.py:60-63,78-84`）。
4. **当前信号使用后复权日 K 数据。** 数据同步和回填都调用 `adjustflag="1"` 并注释为后复权（`sequoia_x/data/engine.py:60-67,293-300`）；README 也明确写为后复权（`README.md:114-119`）。因此信号中的 `close` 是项目存储的后复权收盘价，不应直接当作可下单的现实成交价。Issue #128 还明确提醒其回测存在“事后复权轻微前视”等限制（[Issue #128](https://github.com/sngyai/Sequoia-X/issues/128)）。

## 1. 本地 MA 信号究竟如何计算

对每个本地数据库中的股票代码，策略读取完整历史数据（按日期升序），少于 20 行则跳过（`sequoia_x/strategy/ma_volume.py:30-38`；`sequoia_x/data/engine.py:156-163`）。随后计算：

\[
MA5_t = \frac{1}{5}\sum_{i=0}^{4} Close_{t-i}
\]

\[
MA20_t = \frac{1}{20}\sum_{i=0}^{19} Close_{t-i}
\]

\[
VMA20_t = \frac{1}{20}\sum_{i=0}^{19} Volume_{t-i}
\]

源码对应的滚动计算为 `close.rolling(5)`, `close.rolling(20)` 和 `volume.rolling(20)`（`sequoia_x/strategy/ma_volume.py:39-42`）。取最后一行 `t` 与前一行 `t-1`，信号条件是：

\[
S_t = (MA5_{t-1} < MA20_{t-1})
\land (MA5_t > MA20_t)
\land (Volume_t > 1.5 \times VMA20_t)
\]

即前一交易日严格低于、最新一行严格高于，并且最新一行成交量严格大于 20 日均量的 1.5 倍（`sequoia_x/strategy/ma_volume.py:44-55`）。等号不满足条件。

### 与 Issue #128 的差异

- Issue 正文描述的“5 日均线上穿 20 日均线 + 当日量 > 20 日均量 1.5 倍”与本地实现的核心布尔条件一致（Issue #128；`sequoia_x/strategy/ma_volume.py:13-16,39-55`）。
- Issue 中的 short/mid/long 样本和收益统计属于 Issue 正文给出的研究结果；本地 `MaVolumeStrategy` 没有 short/mid/long 参数或周期定义，周期在代码中直接写死为 5、20 和 1.5（`sequoia_x/strategy/ma_volume.py:13-16,39-55`）。因此不能根据该 Issue 或本地代码自行推断 short/mid/long 各自代表多长持有期。
- Issue 正文提醒：样本约 15 个月（2025-06 至 2026-09）、使用事后复权并有轻微前视、未计交易成本、未考虑涨停买不进、样本可能相关；结论只能用于筛选策略，不能直接实盘。这些回测限制不等于本地程序已经实现了成交和止损规则；Issue 也没有给出明确的成交价、入场时点、出场价或止损代码。

## 2. 信号时序：什么时候才知道信号

### 代码实际使用的“最新”

- `get_ohlcv()` 查询某代码的全部 `stock_daily` 记录，并 `ORDER BY date`（`sequoia_x/data/engine.py:156-163`）。
- `run()` 使用 DataFrame 的最后两行作为 `t-1` 和 `t`（`sequoia_x/strategy/ma_volume.py:44-46`）。这里的“最新”是**本地数据库中最后一条记录**，不是由策略自行确认的“今天收盘”。
- 数据表虽保存 `date/open/high/low/close/volume/turnover`，MA 策略实际只用 `close` 和 `volume`（`sequoia_x/data/engine.py:21-33`；`sequoia_x/strategy/ma_volume.py:39-42`）。

### 日常主流程

日常模式先调用 `sync_today_bulk()`，再执行策略（`main.py:60-67`）。同步函数以系统日历日期 `today_str` 为请求截止日，并从每只股票的本地最大日期之后开始拉取（`sequoia_x/data/engine.py:179-199`）。但：

- 没有新数据时（例如非交易日或接口没有返回当天记录），同步函数记录“无新数据”并返回 0（`sequoia_x/data/engine.py:214-218`），主流程没有因返回 0 而跳过策略，仍会继续 `strategy.run()`（`main.py:60-63,78-84`）。
- 单只股票同步失败时，失败股票仍可能保留旧的最大日期；同步函数只记录失败并继续处理其他股票（`sequoia_x/data/engine.py:208-218`）。
- Bark 消息使用 `date.today()` 写入通知日期，但消息不包含每只股票实际参与计算的最后一条 K 线日期（`sequoia_x/notify/bark.py:68-83`）。所以通知上显示的日期不能证明信号来自当天日线。

**判断：** 如果运行前已成功写入交易日 `t` 的完整日线，名单可理解为“截至 `t` 收盘的日线信号”；如果数据同步未带来新记录，名单可能只是旧交易日 `t-1`、`t-2` 等的重复计算。程序没有把这个数据日期随名单输出。

## 3. 是否存在实际买入点、成交价和止损

### 已实现内容

- 基类规定策略 `run()` 返回 `list[str]`（`sequoia_x/strategy/base.py:32-40`）。
- MA 策略只 `selected.append(symbol)` 并返回代码列表（`sequoia_x/strategy/ma_volume.py:54-63`）。
- 主流程仅把 `selected`、策略名和分组标识传给通知器（`main.py:83-91`）。
- Bark 载荷只组装当前系统日期、策略名、数量、股票名称/代码和雪球链接（`sequoia_x/notify/bark.py:68-83`）。

### 未实现内容

在上述 MA 直接路径中没有定义：

- 以信号日收盘价、次日开盘价、盘口价还是其他价格成交；
- 是否等待次日确认；
- 滑点、佣金、印花税、涨跌停无法成交等交易约束；
- 止损价、止盈价、持有期、卖出信号或仓位管理。

测试也只验证 `run()` 返回 `list[str]` 以及字符串元素，没有价格或交易行为断言（`tests/test_strategy.py:16-40`）。因此，从 MA 名单本身**无法确认唯一买入点位**；它最多告诉你某个日线数据截面满足了布尔条件。

## 4. 如何核对一只名单股票（不把额外规则误认为现有策略）

建议先做“信号核对”，再由使用者另行决定交易规则：

1. **确认数据日期：** 对该股票查看本地 `stock_daily` 的最大 `date`，确认它是否是你认为的信号交易日；不要只看 Bark 的通知日期。`get_ohlcv()` 的排序和策略取最后两行逻辑见 `sequoia_x/data/engine.py:156-163`、`sequoia_x/strategy/ma_volume.py:44-46`。
2. **按同一数据口径复算：** 使用本地同一份后复权 `close`、`volume`，复算 `MA5`、`MA20`、`VMA20`，检查 `t-1` 的 `<`、`t` 的 `>` 和 `Volume_t > 1.5×VMA20_t` 是否同时成立（`sequoia_x/strategy/ma_volume.py:39-55`）。
3. **区分信号价与交易价：** 信号计算使用后复权日线；若要评估真实下单，应另取交易系统认可的实际行情/成交数据，并预先写明复权口径、价格类型和成交可行性。这些都不是当前 MA 策略已实现的规则。

> **待验证的额外规则（不是本项目已实现内容）：** “信号日收盘买入”“次日开盘买入”“次日突破某价买入”以及任意固定百分比止损，都只是可能的回测/交易约定，不能从当前 `MaVolumeStrategy` 或 Issue #128 推导为正式规则。采用哪一种必须单独定义并验证。

## 来源与证据索引

- 本地策略：`sequoia_x/strategy/ma_volume.py:10-63`
- 策略接口：`sequoia_x/strategy/base.py:9-40`
- 本地行情读取与日期：`sequoia_x/data/engine.py:21-33,148-163,173-218,240-300,377-382`
- 主流程调用：`main.py:60-93`
- 通知输出：`sequoia_x/notify/bark.py:68-123`
- 数据说明与运行时序：`README.md:9-13,62-80,114-119`
- 测试覆盖：`tests/test_strategy.py:16-40`
- 外部研究/限制：GitHub [Issue #128](https://github.com/sngyai/Sequoia-X/issues/128)（Issue 正文；主代理已核对，含 2025-06 至 2026-09 样本、回测统计及限制说明）
