# 收益研究协议与上线判据（#9）

研究通过 `sequoia_x.research` 创建、校验和冻结协议。每次实验先登记协议标识、
完整参数、研究阶段及证据来源，再提交结果；失败和未采用结果同样追加保存。
本模块定义研究契约和判定规则。交易回放、数据审计、收益计算及统计区间由后续工单提供。

## 创建和冻结

复制 `examples/research_protocol.json`，替换所有占位文本、日期和示例资金仓位。
示例资金不是建议或系统默认值；经济门槛、胜率要求、风险预算及统计设置没有默认数值。
未填写时协议仍可冻结用于探索，但不能登记确认性实验。百分数用小数，胜率变化用绝对百分点：
`0.02` 表示 2 个百分点；净期望、平均盈利和平均亏损使用账户货币。

```sh
python -m sequoia_x.research validate examples/research_protocol.json
python -m sequoia_x.research freeze protocol.json --registry data/research.sqlite
python -m sequoia_x.research history your-study-id --registry data/research.sqlite
```

`validate` 对结构错误以非零状态退出；结构正确则输出 `exploration_only` 或
`ready_for_confirmation` 及缺失决策。后者只表示协议的决策已填写，仍须经过登记簿的
测试集使用和对照版本检查，不表示研究已通过。`freeze` 输出包含 SHA-256 `protocol_id`、
UTC 冻结时间、修订号、父版本标识和完整口径的 JSON。相同 `study_id` 的每次冻结均产生新修订，
不更新原版本。未知字段、非有限数值、空文本、日期倒置、区间首尾重叠及超出截止日均被拒绝。

最终测试状态默认 `unknown`。只有确认为 `untouched` 才能登记确认实验；
`used_for_selection` 和未知状态只允许探索。声明必须包含登记簿之外已经发生的数据使用。

## 固定的测量口径

- **持有基线**：60 个交易日仅用于预先固定的初始对照，不宣称最优。实际首笔入场成交的
  交易日计第 1 日；分批买入不重置计数。第 60 个交易日收盘后提交退出订单，下一可交易
  开盘执行；停牌、跌停或流动性不足时不虚构成交，延迟至可成交时，期间继续报告未平仓风险。
- **交易归属**：每个入场信号分配独立持仓批次 ID。该信号的部分买入和分批卖出聚合为同一笔，
  现金流、股数及成本按批次分配，全部平仓后才计完整交易。同股不同信号不混并。
  `ignore_while_open` 在该股持仓期间忽略新信号；`independent_signal_lots` 允许独立批次，
  退出订单必须指定批次，不能事后按盈亏重新配对。
- **净胜率**：完整交易净盈亏 = 卖出成交现金 + 分红等现金分配 − 买入成交现金 − 全部适用成本。
  成交价中的滑点不再重复扣减。净盈亏严格大于零才算胜利，零盈亏计入分母；未平仓不计输赢。
- **交易指标**：报告净胜率及依赖和多重检验调整后的区间、正净盈亏均值、负净盈亏均值、
  两者绝对值之比和全部完整交易（含零）的平均净盈亏及区间。
  无完整交易时交易指标为 `null`；无盈利或亏损组时对应均值和盈亏比为 `null`，不能填零或无穷。
- **账户指标**：报告扣费、现金流调整后的账户净值收益，按实际经过年数复合年化、净值最大回撤，
  以及未平仓笔数、暴露和估值依据。不把每笔交易收益相加当账户收益。

完整的机器可读定义随协议一起冻结。资金、仓位、入场订单时间、费用、滑点、容量和重复信号规则
也必须在看结果前写清楚；字符串政策须写出模型及数值，不能仅写“适当费用”。

## 实验登记、证据和判定

```python
from pathlib import Path
from sequoia_x.research import EvaluationEvidence, ResearchProtocol, ResearchRegistry

registry = ResearchRegistry("data/research.sqlite")
protocol = ResearchProtocol.model_validate_json(Path("protocol.json").read_text())
frozen = registry.freeze(protocol)
run = registry.start_experiment(
    frozen.protocol_id,
    parameters={"ma_window": 20},  # 完整参数，登记后不可改写
    stage="validation",         # train / validation / final_test
    purpose="exploration",      # confirmation 必须使用 final_test
    evidence_kind="historical_simulation",  # paper_trading / live_fills
)
# 数据不足、执行失败或没有统计报告也必须留痕：
decision = registry.finish_experiment(run.experiment_id, failure_reason="数据覆盖不足")

# 有证据时，以已登记实验的标识生成报告文件：
# evidence = EvaluationEvidence.model_validate_json(Path("evidence.json").read_text())
# decision = registry.finish_experiment(run.experiment_id, evidence=evidence)
# 同一实验只能提交一次结果；复核应登记新实验并遵守测试集使用规则。
```

证据输入结构可通过 `EvaluationEvidence.model_json_schema()` 获取。证据必须包括实验及协议标识、
数据快照、候选和对照版本、来源、实际评估区间、主要指标、两版本完整指标和费用覆盖报告，
以及独立的净收益增量和净胜率变化区间、配对说明、置信水平、达到的功效、相关性处理方法、
多重检验方法及完整家族、报告标识和重采样种子。区间结构为 `estimate`、`lower`、`upper`。
完整交易报告同时给出 `winning_trades`、`losing_trades`、`breakeven_trades`；
三者之和必须等于 `closed_trades`，净胜率不把零盈亏计为胜利。有盈利或亏损组时不能漏报
相应均值，两组均存在时必须报告一致的盈亏比；净期望须与含零盈亏的全部交易均值一致。
数值报告保留计算精度，汇总一致性校验采用 `1e-8` 绝对容差及 `1e-9` 相对容差。
这些区间须由实际统计实现计算并处理依赖和多重尝试；本模块校验报告的设置和结果，
不把填写了方法名当作已验证该方法的计算实现。`tests/research` 中的数字仅用于契约测试。

确认性研究采用共同验收（所有要求均须满足）：

1. `net_return_delta.lower >= minimum_return_delta`，主要指标预先选定账户净收益或年化净收益。
2. `candidate.net_win_rate.lower >= minimum_win_rate`。
3. `net_win_rate_delta.lower >= minimum_win_rate_delta`；若声称胜率和收益同时改善，
   协议必须提前指定正的胜率增量，两项分别用各自区间支持。
4. 达到功效目标，方法、家族、区间、数据和版本均匹配冻结设置，配对使用同日历及资金约束。
5. 回撤和未平仓暴露均在冻结风险预算内。

初次实验按预声明的初始完整对照比较；存在通过版本后，同一 `study_id`、同一证据来源的
增强版本必须以最近通过版本为对照，不能挑更弱的旧版。历史模拟、纸面交易、真实成交分别维护
通过版本；历史模拟通过不产生真实成交通过状态，更不会发出实际交易订单。

结果状态为 `exploratory`（探索）、`insufficient_evidence`（证据不足）、
`not_adopted`（有证据但未达门槛）或 `adopted`（本来源研究通过）。失败时 `active_version`
保留最近通过版本；没有通过版本则只保留初始对照，不将其宣称为已通过。
用 `experiments(study_id)` 和 `decisions(study_id)` 可回读全部参数及成功、失败结果，
用 `latest_approved(study_id, evidence_kind)` 查询本来源最近通过版本。

## 测试集与审计边界

`single_final_use` 是本版支持的保守测试集使用政策。实验开始即登记所用阶段的完整日期区间；
确认实验登记时保留最终测试区间，未完成、失败也不撤销使用记录。与任何已登记使用区间重叠的
最终测试不能再次登记为确认实验；结果提交时也复核登记后发生的新使用。
改门槛、协议版本、快照名、研究名或证据来源均不清除日期使用历史。
规则在同一登记簿内跨研究保守执行，因此同日期的不同股票池也会被拦截。
若预定家族包含多个候选，先在训练/验证期选定完整版本，再对最后候选使用一次最终测试；
本版不支持把同一最终测试集反复用于家族成员选型。

复核频率及统计家族是冻结的研究计划；外部统计程序负责按计划执行并计算调整后的区间和功效，
本模块不提供复核调度器。所有尝试及参数（包括失败）都须登记并纳入家族；协议或参数改变后，
已经参与选型的数据不能再充当独立最终测试。因持有期产生的跨区间标签重叠，
由后续回放/统计程序在生成报告前隔离，不能靠日期形式不重叠就宣称没有泄漏。

登记簿使用 SQLite 事务串行追加和 SHA-256 哈希链，每次读取或写入先检查已有记录；
API 不提供修改或删除接口，外部修改保留记录会触发完整性错误。
本地哈希链不是第三方签名：不能检测整个文件回滚、尾部截断或持有文件写权限者重算整条链。
应保留冻结 JSON、协议标识和登记簿备份；另建空库不会继承历史，外部数据使用也必须如实声明。
