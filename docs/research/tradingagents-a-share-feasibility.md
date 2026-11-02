# TradingAgents 用于 A 股实盘的可行性

核对日期：2026-10-27。对象是 TauricResearch/TradingAgents 原版，源码固定到 `e8f5503e9b86d5c2710b0c76b1f686703968c9df`；下列“判断”是本次工程评估，不是作者已验证的收益结论。

## 判断

值得用作收盘后的研究助手和候选股票复核层。技术上可以改造成 A 股低频交易系统的一个组件，但原版不能直接当作完整实盘系统；接通接口也不能证明策略在 A 股有正的净收益。适合先验证是否改善现有选股策略，再建设成交与账户系统。

建议的边界是：确定性条件生成选股名单 → Agent 整理公告、财报和反面证据 → 输出有来源和时间戳的结构化研究结果 → 独立规则检查资金、仓位、可卖数量和订单 → 券商接口 → 用真实回报更新账户。LLM 研究结论与执行风控分开，是基于下述源码与交易约束作出的设计建议。

## 原版实际提供什么

- 多分析师、正反研究、交易员与风险讨论的流程可复用；运行终点是保存研究报告和决策评级。源码中未发现 QMT、xtquant 等 A 股券商交易适配。传入的组合上下文是现金、币种和持仓信息，不等于自动同步账户、成交和可卖余额。[运行流程](https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/graph/trading_graph.py)、[组合上下文](https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/portfolio.py)
- 不能说原版完全不认识 A 股：代码已处理 `600519.SH` → `600519.SS`，也配置了 `.SS`、`.SZ` 的地区比较基准。但默认数据链仍以 Yahoo Finance、SEC EDGAR、FRED 等为主，默认宏观查询也偏美欧。识别代码与拥有完整的 A 股公告、新闻、历史基本面是两回事。[代码映射](https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/dataflows/symbols.py)、[默认配置](https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/default_config.py)
- 作者说明美国公司按 SEC 披露时点提供财务数据；非 SEC 公司转到 Yahoo，没有公告时间的财报在历史分析中会被扣留。因此 A 股历史基本面的完整回放仍需补充有披露时点的数据。[作者数据说明](https://github.com/TauricResearch/TradingAgents#fundamentals-as-filed)、[历史数据约束](https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/dataflows/date_window.py)
- 风险讨论由 LLM 提示词和结构化报告驱动，不等价于阻止超仓、重复委托或卖出当日新买股票的硬校验。[组合经理实现](https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/agents/managers/portfolio_manager.py)

## 收益证据能支持到哪里

当前 `backtest.py` 明确用于决策质量评估，不是组合成交模拟器：没有数量、成交价格与现金流水，各股票和日期单独运行，组合输入不随成交流转。默认结算从分析日之后的首个收盘价起算，观察后续五个交易日；其中的 alpha 是股票收益减比较基准收益，并非扣除费用、滑点和不能成交情形后的账户收益。[评估实现](https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/backtest.py)、[结算实现](https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/memory/settlement.py)

原论文 v7 的实验区间为 2024-01-01 至 2024-03-29，围绕美国科技股，结果表列出 AAPL、GOOGL、AMZN；作者解释只测三个月与调用预算有关。论文配置每次预测约 11 次 LLM 调用和 20 多次工具调用，这不是当前版本的实测成本。论文结论把部署真实交易环境列为后续工作，实验说明没有给出足够的佣金、滑点和订单成交参数来证明 A 股可交易净收益。[原论文 §5–7](https://arxiv.org/pdf/2412.20138v7)

据此判断：短区间、少量美股结果不足以推导跨市场有效性；历史数据过滤也不能独自排除新模型在预训练中记住历史事件的可能。应固定模型、提示词、数据快照和运行配置，开展前瞻模拟，并与不使用 Agent 的同一选股策略比较净收益、回撤、换手和漏掉的机会。

## A 股需要补足的交易约束

- 普通 A 股股票不能当日买入后当日卖出。深交所 2026 年规则第 3.1.4 条规定交收前不得卖出，回转交易例外品种单列；股票执行器必须使用券商返回的可卖数量。[深交所现行交易规则](https://docs.static.szse.cn/www/lawrules/rule/trade/current/W020260424690713155663.pdf)
- 数量与价格约束要按板块、证券状态和生效日期处理。深市一般竞价买入为 100 股整数倍，主板通常 10%、创业板通常 20%；新股等存在例外。科创板限价申报最低为 200 股，不能统一写成所有股票 100 股整数倍。[深交所规则第 3.3 节](https://docs.static.szse.cn/www/lawrules/rule/trade/current/W020260424690713155663.pdf)、[上交所规则第 6.6–6.7 条](https://www.sse.com.cn/lawandrules/sselawsrules2025/fund/trading/c/c_20260424_10817739.shtml)
- 规则会变化：2026-07-06 起主板风险警示股票涨跌幅由 5% 调整为 10%。不能把旧规则永久写死。[上交所官方修订说明](https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20260424_10816474.shtml)
- 工程上还需显式模拟停牌、涨跌停无对手方、部分成交、撤单失败、费用与滑点，并保存委托状态、成交回报和账户对账。日线价格变化或生成 SELL 都不能视为订单已经成交。这是由交易约束推导的实现要求。

## 接口与程序化报告

迅投官方文档确认，XtQuant 通过 MiniQMT 提供行情、报撤单、资产/持仓/委托/成交查询及变动推送；运行前需启动 MiniQMT。券商 QMT 要向对应券商申请，各券商要求不同。应核实本人账户能否开通 MiniQMT/外部 Python 接口、支持的品种和实盘权限，不能套用网上某个统一资金门槛。[迅投 API 说明](https://dict.thinktrader.net/nativeApi/start_now.html)、[迅投开户与权限说明](https://knowledge.thinktrader.net/docs/freshman/rookie.html)

沪深程序化交易细则于 2025-07-07 施行。上交所定义覆盖计算机自动生成或下达交易指令，明确包括个人投资者；客户首次开展前需向券商报告，核查确认后方可开展。低频不是当然豁免。高频另按单账户每秒申报/撤单最高达到 300 笔，或单日最高达到 20000 笔等认定，附加义务不能泛化为所有每日低频系统都适用。研究报告是否已构成自动生成交易指令，应结合实际输出与执行流程由券商确认。[上交所细则第 2–3、8、33–34 条](https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/c_20250612_10781696.shtml)、[深交所细则](https://www.szse.cn/lawrules/rule/trade/t20250403_612770.html)

## 对 Sequoia-X 的落地建议

当前领域定义是收盘后技术形态选股与固定交易日的信号收益研究。先保持这些定义：Agent 可为少量选股名单提供公告风险、财务异常和行业背景的复核，记录数据缺失以及支持和反对证据；不要把研究用后复权日线当成委托价，也不要把信号收益改名为实盘收益。

实施顺序建议为：接入有披露时点的 A 股资料；验证 Agent 是否提供可重复的增量信息；单独建设考虑可成交性和成本的账户模拟；连续记录前瞻结果；满足券商接口与报告要求后，再评估小规模实盘。当前不应仅凭论文收益或 Agent 辩论结论跳过这些验证。
