# 收益测量数据样本

`sequoia_x.research.dataset` 是只读审计和样本交换边界，不产生策略通过结论。

## 输入与时点

- `DatasetSnapshot(manifest, calendar, bars, states, actions)` 全部使用冻结模型；记录顺序规范化，`snapshot_id` 是全部输入的 SHA-256，不含导出路径或读取时间。
- `DatasetManifest` 固定来源/来源版本、历史股票池标识/证券、研究开始/结束、数据截至和 `synthetic` / `observed`。`universe_complete`、`states_complete`、`actions_complete`、`calendar_complete` 默认均为 false；它们是来源覆盖声明，不能由“拉到了数据”推导为 true。历史全集声明必须涵盖后来退市证券和当时成员，不可用今天的名册代替。
- `calendar` 是全市场交易 session，不能用单股有行情的日期冒充。可包含研究开始日前预热；审计会检查结束日前全部可消费预热与研究行情。
- `DailyBar` 的 OHLC 是原始市场元/股价格，`volume` 单位为股。`status` 分别记录 `trading`、`suspended`、`missing`、`anomalous`，零量另计。`price_basis` 不是 raw 时禁止获得执行和收益资格。
- 指标价默认 `raw_as_of`，省略指标四价时复用 raw；显式指标四价必须全部提供。`point_in_time_adjusted` 要求 `indicator_open/high/low/close`、`indicator_available_on` 和 `indicator_source_version`，不接受未来公司行动回填。无法证明历史可知的复权数据标记 `unverified_adjusted`。raw 与恢复版后复权日常输入不同，不能声称真实日常信号已经一致；一致性比较必须使用相同输入口径。
- `available_on` 是该信息最早可供当日收盘决策使用的日期，只有日期粒度，不能证明某个盘中时刻已经可知。`bars_as_of(on)` 截断原始/指标可知日，并封顶于结束日和数据截至；不能用当日完整成交量、最高/最低价决定当日开盘订单。日线仅供事后保守成交模拟，不能证明实际排队成交。延迟行情不允许反填历史收益资格。
- `SecurityState(symbol, effective_date, available_on, listed, suspended, is_st, board, lot_size, limit_rule, limit_up, limit_down)` 是精确每日状态；`state_at(symbol, on)` 不延续昨日涨跌停价格。`limit_rule` 默认 unknown，bounded 必须给双限，unbounded 也必须有已核对的适用依据；不按今天规则猜历史限制。当日状态用于开盘执行还需来源证明其事前已知，仅日期粒度不足时只能保守阻断。
- `CorporateAction` 分开公告可知日 `available_on`、生效日 `ex_date` 和现金实际入账日 `pay_date`。`cash_per_share` 是每原股税后实收现金，`withholding_tax_per_share` 单列已扣税，毛额为两者之和；`share_multiplier` 是生效后/生效前股数比例。除权日开盘前持仓登记权益，除权日买入没有该次权益，现金到 pay_date 才可用。仅支持该明确事件模型；配股、延后上市送股、无法确定持有期扣税或缺事件数据时不得声明完整。`action_id` 是不含公告可知日的经济事件指纹，重复事件阻断，避免重复收益。同证券同生效日多条记录缺少独立事件/修订身份时一律阻断；不能把公告修订值当两笔分红，也不支持猜测同日独立事件。

## 公开调用

```python
from datetime import date
from sequoia_x.research.dataset import audit_dataset, load_sample, save_sample, sample_sqlite_cache

sample = sample_sqlite_cache(
    "data/sequoia_v2.db", ("000001",), date(2024, 1, 2), date(2024, 1, 5),
    as_of=date(2026, 11, 7),
)
report = audit_dataset(sample)
print(report.model_dump_json(indent=2))
save_sample(sample, "/tmp/research-sample.json")
assert load_sample("/tmp/research-sample.json").snapshot_id == sample.snapshot_id
```

`sample_sqlite_cache` 以 SQLite `mode=ro` 打开，仅读指定证券和区间。现有 `stock_daily` 来源固定为 baostock `adjustflag=1` 后复权，缺完整历史成员、证券状态和公司行动；导出的 observed 样本保留这些缺口并返回 `data_unqualified`。不改数据库、不拉全市场、不新增行情表。`save_sample` 保存完整输入及指纹，`load_sample` 校验篡改；输出路径由调用者管理，保存不代表资格通过。

`audit_dataset` 返回状态、原因、行情状态计数、缺状态/行情数量及零量数量。`replay_eligible` 仅表示声明和结构检查允许功能重放，不证明真实成交。`synthetic` 可 replay_eligible，但始终 data_unqualified，不能伪装真实研究结果。`qualified` 仅表示真实来源的声明与本地检查通过，仍不证明全市场覆盖、撮合可行或策略有优势，后续统计门槛独立。

## 本票真实采样与尚未验证项

2026-11-07 只读调用现有 baostock，为 `sh.600000` 的 2024-01-02 至 2024-01-05 取得 4 行 `adjustflag=3` 原始行情（字段含 preclose、tradestatus、isST）和 4 个交易日。stock_basic 的上市/退市字段是查询时取得，不能倒填历史可知状态。2024 年 operate 分红返回税后金额字符串“0.2889或0.321”，不是唯一可入账金额；没有替用户选择其中一个。

进一步按固定 `sh.600000` / `sh.600112` 和 2024-01-02 至 2024-06-28 窗口取得每股 117 行、共 234 行 raw 行情及 117 个交易 session，长度已覆盖预热和 60 日功能重放。2024-01-02 历史成员查询返回 6529 条证券成员记录，包含这两只证券；stock_basic 另证实 `sh.600112` 后于 2024-08-15 退市，该事后信息未写成研究期内已知退市状态。2024-06-28 历史成员查询最终因连接重置失败；普通客户端此前还记录了超时与接收错误，改用恢复后的有界会话后成功取得长窗口 raw 和起点成员。2023/2024 operate 分红响应未列出窗口内生效事件，但这不证明全部公司行动或税务完整。

观测样本的源响应、带指纹输入和审计报告保存在本次交付 artifacts，真实样本保持 data_unqualified。原因包括逐日历史全集未核实、适用板块和逐日涨跌停缺来源证据、公司行动/税务覆盖不完整；不能因已有足够长度而推定可信收益资格。测试中手工价格都明确标为 synthetic，不能替代上述真实覆盖。本票交付工程能力与诚实覆盖报告，真实可信收益测量样本的目标如仍缺关键来源应记为 partial，不以代码测试通过关闭该缺口。
