# flag 60 日组合基线

`sequoia_x.research.flag` 将恢复版 flag 日常选股条件用于历史逐日截断重放，并把信号交给 MA 使用的同一个组合账户。该工具测量历史模拟结果；工程测试通过不代表策略收益通过，也不产生真实下单授权。

## 不变的形态定义

- 至少 40 根有效行情；40 日最高价 / 最低价严格大于 1.6。
- 最近 10 日最高价 / 最低价严格小于 1.15。
- 最近 10 日最低价不低于 40 日最高价的 80%。
- 当日成交量严格低于此前 20 日平均成交量的 60%，此前均量不包含当日。

高低价比不约束低点先于高点，也不是突破条件。这里不补充突破、回踩、排序或其他选股规则。

## 观察、重复与订单

`replay_flag_signals(snapshot, through=..., window=...)` 返回每日每股观察、形态特征和公共 `Signal`。同一证券连续符合形态的日期共享一个 `episode_id`，首次日期和持续日期分别标注。每个符合日常选股定义的日期仍保留信号，账户依冻结协议的 `repeat_signal_policy` 处理持仓重复；重放层不暗中改变日常选股定义。允许独立 lot 的协议也不能把同一 episode 的多个日期当成独立统计机会。

缺失、不可见、异常或无法形成有效截断输入的行情不会证明形态结束，因而不重置已有 episode。只有一个可核对的有效不成立观察才结束 episode。数据恢复后先满足共同的连续预热要求；缺口不会制造新的独立机会。

形态成立不是已经下单或成交。观察中的 `signal_id` 与组合订单、成交、拒绝理由和未平仓交易关联；订单触发、未成交及成交结果保存在公共组合账本。

## 共同的账户与证据边界

`run_flag_portfolio(snapshot, frozen_protocol, execution_policy, selection_policy, ...)` 返回观察重放与 `PortfolioResult`，直接使用 `run_portfolio(..., signals=...)`，不另建资金账户。现金、名额、中性选择、重复持仓、手续费、滑点、容量、T+1、涨跌停和实际入场日起 60 交易日的规则都与 MA 相同。协议中必须精确引用两份执行/选择政策标识。

组合结果保留完整日历净值、未成交订单、完整与未完成交易、费用及未平仓暴露。观察覆盖另列每日候选数、全部形态日期、episode 首次日期和重复日期，以区分候选数量与实际市场暴露。MA 与 flag 分别模拟独立的相同初始账户；分别通过不代表共享资金的联合策略已经通过。

指标行情使用数据契约允许的当时可见价格，成交使用原始市场价格。合成样例只验证功能，不获得真实数据资格；历史覆盖、公司行动或价格口径有缺口时，结果保留不合格/证据不足原因。样本外净胜率与收益是否通过由独立统计检验完成。

## 调用与覆盖口径

```python
from sequoia_x.research.flag import replay_flag_signals, run_flag_portfolio

replay = replay_flag_signals(snapshot, window=window)
result = run_flag_portfolio(
    snapshot, frozen_flag_protocol, execution_policy, selection_policy,
    window=window, benchmark=benchmark_levels,
)
observations = result.replay.observations
account = result.portfolio
```

`observe_flag(frame, on=...)` 是单日公开观察接口，自行截断日期；正式快照重放先调用公共 `signal_frame`，选择当时可见的指标口径和有效预热输入。其 `FlagCondition` 保留形态状态、四条件特征及不足预热/无有效行情原因。`FlagReplay.implementation_version` 记录形态与 episode 实现版本。

快照日历可以包含研究开始前的预热行情。观察保留这些历史行，并以 `in_research_window=False` 标记；它们维护真实 episode 起源，却没有交易 `signal_id`，不会产生区间外订单。`window` 同时作用于重放与公共账户，`through` 只限制可见截止。

`coverage.total_observations` 包含预热观察，`pre_window_observations` 单列区间前数量；形成/重复/缺失/预热状态计数和每日候选数仅统计研究窗口。`episode_start_dates` 是研究窗口所见 episode 的真实起源，可能早于窗口开始；它不把跨边界持续形态重新计为新机会。`signal_dates` 是窗口内全部成立日，两者分别输出。

flag 应使用明确登记的自身策略版本与冻结协议；不要将既有 MA 账户改名充作 flag 结果。两者可用同一数据、数值执行政策、中性选择政策、初始资金和窗口分别重放，并比较同一 `PortfolioResult` 口径。窗口末仍持有的交易不参与净胜率，成交量受限、持续重复与无法退出原因仍保留。
