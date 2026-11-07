# 基线统计验收（#14）

`sequoia_x.research.statistics` 消费 #10 的 `DatasetSnapshot` 和 #12/#13 的
`PortfolioResult`，输出完整 JSON 报告与登记簿决策。它不接受人工填写的汇总收益来
替代组合账本，不自动下单。合成样例可以验证统计实现，但永远不能产生真实收益通过结论。

## 初始 B0 与增强版本

`ResearchProtocol.comparison_role` 默认 `approved_strategy`，保留原来的增强版本语义：
候选与最近通过版本均须有成熟完整交易，并独立满足收益增量、绝对胜率和胜率变化门槛。

显式冻结 `initial_reference` 才能使用无交易现金参考建立初始 B0。该模式仅在同一
`study_id`、同一证据来源尚无通过版本时有效；实验开始和结果提交都检查，包括已经
登记但尚未完成的旧初始实验。只允许 `claim="return_improvement"`，
`minimum_win_rate_delta` 必须为 `null`；候选绝对胜率、每笔净期望区间、相对参考的
净收益增量、功效、风险与独立测试门槛不变。现金参考的胜率、期望和胜率变化均为
`null`，不能用零胜率或假交易代替。候选本身零完整交易仍为证据不足。

MA 和 flag 分别使用独立研究与明确版本。`report.b0_version` 只来自登记簿实际通过的
版本；两个单策略通过不证明共享资金的联合策略有效。没有通过版本时保持停止交易升级。

## 冻结与调用顺序

1. 先取得并审计实际快照，确定完整历史覆盖、执行可用性、资金、协议与所有数值政策。
2. 冻结候选、参考、成本压力和参数邻域各自协议，记录准确的策略版本与政策 hash。
3. 在看最终结果前，以 `StatisticsPlan` 构造 `statistics_plan`，连同
   `execution_policies`、`selection_policies` 的完整数值定义传给
   `ResearchRegistry.start_experiment(parameters=...)`。所有尝试都登记。
4. 再运行账户重放，将候选与参考账户及诊断输入交给 `analyze_baseline`。
   它会结束该实验；失败、探索和不采用均保留，不能覆写同一次实验。
5. 用 `record_report(report, directory)` 保存完整报告；文件名为完整内容 SHA-256。
   `load_report(path, sha256)` 校验后读取。同一内容重复保存幂等，旧内容不能覆盖。

```python
from sequoia_x.research.portfolio import IMPLEMENTATION_VERSION, run_portfolio
from sequoia_x.research.statistics import (
    AccountContract, StatisticsPlan, analyze_baseline, implementation_fingerprint, record_report,
)

# 以下对象须在运行/查看最终账户前构造；资金与门槛由研究者提前决定。
candidate_contract = AccountContract(
    protocol_id=candidate_frozen.protocol_id,
    strategy_version=candidate_frozen.protocol.strategy_version,
    implementation_version=IMPLEMENTATION_VERSION,
    execution_policy_id=execution.policy_id, selection_policy_id=selection.policy_id,
)
# comparison_contract、预登记场景、参照及行业/阶段映射使用各自版本与内容指纹。
plan = StatisticsPlan.model_validate(plan_values)
parameters = {
    "statistics_plan": plan.model_dump(mode="json"),
    "execution_policies": {p.policy_id: p.model_dump(mode="json") for p in execution_models},
    "selection_policies": {p.policy_id: p.model_dump(mode="json") for p in selection_models},
}
run = registry.start_experiment(
    candidate_frozen.protocol_id, parameters=parameters, stage="final_test",
    purpose="confirmation", evidence_kind="historical_simulation",
)
candidate = run_portfolio(snapshot, candidate_frozen, execution, selection, benchmark=market_levels, window=run.used_window)
comparison = run_portfolio(
    snapshot, comparison_frozen, execution, selection, signals=(), benchmark=market_levels, window=run.used_window,
)  # 此例参考为现金；必须有自己的冻结 cash-control 版本，不能事后改名。
report = analyze_baseline(
    registry, run.experiment_id, candidate, comparison, snapshot=snapshot,
    market_reference=market, exposure_reference=exposure, sector_map=sectors,
    regime_map=regimes, scenarios=scenario_accounts,
)
saved = record_report(report, "data/research-reports")
```

`StatisticsPlan.model_json_schema()` 给出全部参数契约。源码指纹来自
`implementation_fingerprint()`，必须在实验登记前固定；代码发生变化后旧计划不能继续确认。
所有账户（包括现金参考与诊断场景）须传入完整同日历的正值基准指数 `market_levels`；
缺少该序列不能仅靠有参照描述而通过资格。年化沿用共享账户口径：最后日减首日的
实际日历天数除以 365.25；零经过天数不估计年化。
参考、行业及市场阶段映射也按内容 hash 绑定；行业和阶段标签须在对应信号日已知，
缺失或多义标签进入报告并阻断通过。描述性分组不能事后晋升为获利规则。
登记簿无法观察库外已经看过的数据，外部使用仍须如实声明。

## 可构造的 Bonferroni 协议示例

以下数值只用于可控测试，绝非真实策略的推荐门槛；替换为研究前已决定的值。
`sample` 是包含 2024 年后研究窗口的 #10 快照，`execution`、`selection` 是已经明确
构造的数值政策。真实缺口仍使报告不通过。

```python
import json
from pathlib import Path
from sequoia_x.research.protocol import ResearchProtocol

values = json.loads(Path("examples/research_protocol.json").read_text())
values.update(
    study_id="ma-initial-example", strategy_version="ma-b0",
    comparison_version="cash-control", comparison_role="initial_reference",
    universe=sample.manifest.universe_id, data_snapshot_id=sample.snapshot_id,
    data_cutoff=sample.manifest.end.isoformat(), final_test_status="untouched",
    train={"start": "2020-01-01", "end": "2022-12-31"},
    validation={"start": "2023-01-01", "end": "2023-12-31"},
    final_test={"start": sample.manifest.start.isoformat(), "end": sample.manifest.end.isoformat()},
)
values["trading"].update(
    entry_order_time="next_session_open_limit",
    fees=[f"execution_policy:{execution.policy_id}"],
    capacity=f"selection_policy:{selection.policy_id}",
)
values["statistics"].update(
    confidence_level=0.95, power_target=0.8,
    dependence_method="moving_block_signal_date_v1",
    multiplicity_method="bonferroni_family_metrics_v1",
    family_id="ma-initial-example-family", family_members=["ma-b0", "ma-b0-neighbor"],
    review_schedule="最终测试仅一次", test_set_rule="single_final_use",
)
values["acceptance"].update(
    primary_return="portfolio_net_return", minimum_return_delta=0.02,
    minimum_win_rate=0.5, minimum_win_rate_delta=None, max_drawdown=0.2,
    max_open_exposure=0.3, claim="return_improvement",
)
protocol = ResearchProtocol.model_validate(values)
assert not protocol.exploration_reasons()
```

## 统计方法与边界

- 只实现 `moving_block_signal_date_v1`：候选与参考共用同一批连续交易日块，
  保留空仓日、同日所有交易以及两账户配对关系。交易按信号日期聚集；
  净日收益来自已经执行的完整账户，不在抽样后再假装重新执行另一条策略路径。
  块长不得小于 60 日或实际信号至退出跨度；少于两块、部分抽样缺完整交易、
  尾部分辨率不足都会阻断确认。日历块处理依赖的依据是
  [Künsch 的原始分块 bootstrap 研究](https://doi.org/10.1214/aos/1176347265)。
- 只实现 `bonferroni_family_metrics_v1`：按预登记家族成员数 × 12 个指标分配错误预算。
  12 项为候选/参考/差值的胜率、净期望、组合净收益和年化收益；不可估计项仍占预算，
  不在看结果后缩小家族。Bonferroni 的家族错误预算规则见
  [NIST 方法说明](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm)。
  若协议写 Holm 或其他未支持方法，报告保持实际算法名称并给证据不足。
- 区间是弱平稳与足够块长假设下的百分位近似，不是有限样本精确保证。
  `independent_block_equivalents` 是冻结块长假设下的信息单位估计，不是已证明独立的交易数。
- `partition_mature_trades` 清除跨训练/验证/最终测试边界的标签，并按预设 session
  隔离区间起点；`as_of` 之后才平仓的交易不能进入当时的滚动训练。
- 功效只支持 `normal_block_standard_error_v1`：使用实际块重采样标准误、预先冻结的
  备择与门槛之间的间距、同一多重检验错误预算计算向通过方向的正态规划近似。
  `achieved_power` 是收益与胜率全部统计门槛的联合保守下界
  `max(0, 1-sum(1-p_i))`，不取单项功效最小值冒充联合功效；所需样本把
  全局失败预算按门槛数分配。风险预算仍另行共同验收。
  未估出块方差、备择未严格超过门槛或功效不足都不通过。所需样本输出独立块等价量，
  不能解释为固定笔数、月份或未来功效保证，也不能在看过最终测试后据此反复调整。
- 成本场景须真实提高预登记全部费率/最低佣金与滑点乘数，并保持订单、信号、选择及
  资金政策；相同账户不能冒称压力结果。当前参数邻域支持预先登记的
  `trading.max_position_fraction`、`trading.max_positions`；其他参数明确标为未支持。
  邻域使用独立版本并纳入家族。输出成本、胜率和收益变化，失败结果同样保留。

报告区分每账户原始交易总数、原始完整交易数、清除数与成熟完整交易数，排除项保留
候选/参考账户归属；场景也使用完全相同的成熟筛选。
报告同时保存净值收益、未平仓与不可退出风险、原始信号/独立 episode、日期簇大小、
日收益滞后相关、季度/行业/市场阶段集中、最大与前五笔盈利贡献、参照超额及全部场景。
零完整交易的胜率/期望/相应区间为 NA；缺少盈利或亏损组时条件均值及盈亏比为 NA。
净期望直接使用全部成熟完整交易净盈亏均值，未平仓估值不会变成一次“胜利”。

## 实现验证与真实结论

`tests/research/test_statistics.py` 的完整正向集成案例，固定 900 个 session、
连续块长 62、5000 次重采样与 seed 23，先登记所有门槛、参照和场景，再执行
共享账户。它为了覆盖合格来源路径而使用显式 `controlled-...-NOT-empirical`
来源与 `observed` 测试标签；全部价格是构造值。其 `adopted` 只验证软件判据，
不能成为 MA/flag 的真实收益 B0。真实来源、研究前经济门槛和统计假设仍须独立提供。

可复现命令：

```sh
python -m pytest -q -s tests/research/test_statistics.py::test_controlled_initial_b0_uses_actual_replay_and_keeps_cash_metrics_na
```

该测试保存完整内容寻址报告，确认现金参考的交易胜率、期望和胜率变化均为 `null`，
候选独立满足收益区间、绝对胜率、联合功效与稳健性条件，并确认已有通过版本后
不能再走初始参考路径。其他案例覆盖已失败最终区间不可复用、同日聚集、跨边界标签、
不支持方法、来源/账本篡改、零交易和未平仓风险。
