# 单笔基线重放

`sequoia_x.research.replay` 为 #11 提供只读历史模拟。它不下单，不把工程测试通过解释为 MA 有收益优势。

## 公共接口

- `signal_frame(snapshot, symbol, on)`：到当日收盘可知的指标 OHLCV，固定列为 `date/open/high/low/close/volume`。原始口径用 raw，已核实的时点复权口径用独立指标字段；未核实复权口径不产生信号。缺失或异常切断预热，历史停牌和零量按恢复版入库行为跳过，当日不可交易不产生信号。
- `replay_ma_signals(snapshot, through=None)`：复用恢复版的 MA5/MA20 严格金叉和当日量大于含当日 20 日均量的 1.5 倍。指标可消费研究开始日前预热，但只返回声明研究区间内信号。`Signal` 保留证券、信号日、策略、特征和可选 episode；`signal_id` 由规范内容确定。
- `quote_open_fill(bar, state, side, quantity, available_cash, policy, acquired_on=None, used_volume=0)`：供单笔和组合共用的开盘限价 DAY 原语；先用事前限价和预算确定申报，再模拟匹配，无成交也返回原因。买入按整手，卖出完整余股允许不足一手，部分卖出仍按整手。证券的显式当日数量单位及限价来自数据样本，不从今天的规则猜历史限制。
- `apply_corporate_actions(snapshot, symbol, session, quantity, receivables=())`：每日开盘前对前一日持仓调用一次；每个信号 lot 分开持有应收。除权日登记每原股税后现金权益及支持的数量变化，到账日前不能把应收当可用现金；仅有到账日期时保守在该日收盘入账，不能预花到该日开盘集合竞价。持仓已卖清后仍传入该 lot 的应收，到账后返回的应收列表会移除该项。重复应收、非整数股处置和同日多个未合并股数变化明确报错或保留阻断原因。
- `mark_position(snapshot, symbol, session, quantity)`：用有成交量的可交易原始市场收盘价估值，停牌行携带的旧报价不当作新市场价格；缺价可携带最后已知市场价格和日期；跨公司行动缺价返回 `unreliable`、`value=None`，不按买入成本或旧价乘新股数制造价值。
- `replay_trade(snapshot, frozen, signal, quantity, policy, through=None, cancel_entry_on=None)`：一个明确提供的信号生成订单、成交、公司行动、每日现金/持仓/应收/净值和净盈亏。MA 信号应由 `replay_ma_signals` 生成；显式传入的测试信号不自行证明 MA 条件成立。

上述 keyword 参数需按函数签名使用（`quote_open_fill` 的 `acquired_on/used_volume`、重放的 `through/cancel_entry_on` 为 keyword-only）。

## 数值执行政策与协议绑定

费率是成交额的小数比例（`0.001` 为千分之一）；佣金、最低佣金、卖出印花税、双边过户费逐项按历史生效日期配置。滑点是每侧基点，数量和成交量是股，价格及现金单位是人民币元。内部用 `Decimal`，成交价和每项现金费用按分 `ROUND_HALF_UP`，不得因浮点误差透支现金。费用表必须严格按生效日递增，没有适用费率不成交。

先构造数字政策，再把完整政策的指纹写入协议后冻结。例如（费率均需由调用方提供并核对，不是本模块的实盘默认）：

```python
from sequoia_x.research.protocol import ResearchProtocol
from sequoia_x.research.registry import ResearchRegistry
from sequoia_x.research.replay import ExecutionPolicy, FeeSchedule, replay_ma_signals, replay_trade

execution = ExecutionPolicy(
    fees=(FeeSchedule(**verified_fee_schedule),),
    max_order_quantity=verified_max_order_quantity,  # 必填，最多 1_000_000 股
    slippage_bps=verified_slippage_bps,
    max_volume_participation=verified_participation,
    volume_model="daily_volume_upper_bound",
)
protocol_input["data_snapshot_id"] = snapshot.snapshot_id
protocol_input["trading"]["fees"] = [f"execution_policy:{execution.policy_id}"]
protocol_input["trading"]["entry_order_time"] = "next_session_open_limit"
frozen = ResearchRegistry("research.sqlite").freeze(
    ResearchProtocol.model_validate(protocol_input)
)
signals = replay_ma_signals(snapshot)
if signals:
    result = replay_trade(snapshot, frozen, signals[0], requested_shares, execution)
    print(result.model_dump_json(indent=2))
```

政策指纹包含 `daily-open-limit-v1` 算法版本、明确限价订单类型、全部费率表、滑点、容量、申报数量上限、订单有效期、卖款和股息可用时点。旧市价模型或 `signal_next_session_open_day` 协议会明确拒绝，须登记新模型和新协议，不能静默重解释。协议没有精确引用或订单时点相冲突时会清楚报错，不解析自然语言猜数值，也不静默改已冻结协议。数据快照也必须与协议指纹一致。

## 成交和持有时点

D 日收盘信号只在全市场日历的下一交易日开盘尝试一次 DAY 买单；未成交不延期，部分成交的余量当天过期。可显式撤销信号确认后、入场时点前的入场订单，撤销留痕且不产生成交。买入日是持有第 1 个全市场交易日，停牌日也计数；第 60 日收盘后，从下一市场交易日开盘开始尝试退出，停牌、开盘跌停或容量不足就保留全部或部分持仓并逐日重试。卖出原语拒绝买入当日卖出。

本模型仅支持数据中明确核实的普通沪主板 `SecurityState.board="SSE_MAIN"`、100 股单位和有界当日限价。`main` 泛标签、其他板块、无界/未知限价和未支持的风险警示股限制都会明确拒绝；原有持仓仍保留估值和无法退出状态。买入申报限价固定为事前已知的当日涨停价，卖出固定为事前已知的跌停价，不按事后开盘价反推限价。请求数量超过冻结上限直接拒绝，该上限必须显式配置且不得高于 100 万股。

先按限价及该日费用核对可冻结的预算，确定实际申报数量；再用原始开盘价加冻结滑点模型作日线匹配。开盘买涨停或卖跌停保守不成交，完整 raw OHLC 越过已核实边界、滑点价越过观测区间或申报限价也不成交。缺状态、非原始价格、迟到行情或缺费用都不虚构成交；申报与后验无法匹配分别记录。

`FillAttempt` 分开保存 `requested_quantity`（计划）、`submitted_quantity`（实际可申报）、`unsubmitted_quantity`、`quantity`（模拟成交）、`limit_price`、`reserved_cash/released_cash` 和卖单的 `reserved_quantity/released_quantity`。`filled/partial` 相对实际申报量判断，资金不足而从未申报的数量不冒称交易所部分成交。冻结资金按限价加费用，`released_cash` 是 DAY 结束后未使用的冻结金额，不能拿它事后增加同批开盘订单。比如预算 6000 元、计划 600 股、事前限价 12 元、费用为零，只能申报 500 股；若模型开盘价 10 元并成交 500 股，净支出 5000 元、日末释放 1000 元，另 100 股从未申报。

卖出资金在本次开盘集合竞价结束后才可用（`available_after_opening_auction`），不能给同批开盘买单提供申报资金。本模型不假设连续竞价报价，卖款可在收盘决定次日订单。只有 `pay_date` 的股息按 `available_at_session_end` 保守处理。组合调用方必须先冻结同批全部订单，再进行后验匹配，不根据真实开盘价、先成交/先卖清的结果补选名额。

完整当日成交量仅作**事后模拟容量上界**，不是开盘时已知信息，不能证明开盘撮合、排队顺序或可成交量。研究不能据此宣称真实成交已验证。日线模型不模拟盘中触价、逐笔排队或保证全部成交；若研究需要这些结论，须另有相应可用数据与冻结模型。

## 经济结果与资格

完整净盈亏 = 卖出净回款 + 已登记税后分红权益 - 买入含费净现金。税后分红已扣税；另列扣税仅作成本核对，不再从净盈亏重复扣除。分红应收进入净值但到账前不进入可买入现金。股数变化按数据契约的有效日处理；不支持或未核实的事件产生阻断原因，不能当作已解决的数据问题。

只有完全平仓且数据经济核算可用时才报告 `net_pnl`，期末仍持有或无法退出为 `None`，同时保留数量、每日市场价值、应收、现金和状态。公司行动覆盖不全时，即使模拟卖清，也不发布可信完整净盈亏。

`quality_status` 来自数据审计与运行中阻断原因；`synthetic` 永远保留 `data_unqualified`，其可手算数字只验证程序。单笔成功、价格上涨或本模块测试通过都不形成 B0，不替代 #14 的独立统计确认。输出包含协议、样本、执行政策标识，可相同输入完全重放。

历史规则核对参考：[上交所《交易规则（2023年修订）》归档及附件](https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20250612_10824490.shtml)，其中 3.3.6 区分市价委托适用时段、3.3.9 约束普通单笔数量，4.4.9 和 4.4.12 包含风险警示股特殊限制。此版本采用显式限价模型并保守拒绝尚未覆盖完整限制的风险警示股；该归档已废止，不能拿它替代其他历史时段或当前规则的逐项核对。
