# 更新记录

## 2026-09-07 修复 real_society 直接运行的期限遗漏与零供给报错

- 根因复现：`real_society` 只设置训练采样长度 `epoch_length: 27`，实际环境继承 300 期、
  `test: false`，进入训练循环。以原 seed=2、先训练采样 27 步再评估的顺序，第 192 期
  SCF 最近邻劳动规则产生全员零工时，总劳动、产出均为 0；计划需求为 15,917,463.14，
  其中家庭消费需求 4,430.90、投资需求 15,913,032.24，导致下一期价格的 D/S 不可定义。
  这同时暴露了长期劳动行为失稳，不能仅归因于价格函数本身。
- 配置补齐 `Environment.env_core.episode_length: 27`、`Trainer.test: true`。
  仍保留 95% 消费、SCF 劳动和原有冲击；没有调整经济参数来压制失败。
- `env/env_core.py`：有限、非负供需下的零供给在当期结算后标记为
  `termination_reason: zero_goods_supply` 并终止；保留零产出和当期交易价格，不计算下一期价格。
  reset 清除原因。负值、NaN、Inf 仍报错，价格异常信息增加实际供需值。
- `runner.py` 与评估报告：逐期 JSON、episode 结果保存终止原因，正式评估在终端明确打印模拟失败。
  运行说明见 `agents/rule_based/README.md`。原错误数值及复现脚本位于
  `diagnostics/zero_supply_fix_20260907/`；27 期可运行不等于 300 期经济稳定。

## 2026-09-07 家庭固定策略迁入 rule-based

- `real_society` 与 `real_society_calibration` 的 `house_alg` 都改为 `rule_based`。
  `household_consumption_share: 0.95` 直接控制含税消费预算占税后收入的比例。
- `agents/rule_based/households.py` 新增独立的 `fixed_consumption`、`initial_labor`、
  `constant_labor`、`scf_labor` 函数；原完整抽样行为保留为 `age_profile`。
  配置及函数旁均有注释，见 [规则切换说明](agents/rule_based/README.md)。
- 校准基准选择 `initial` 劳动；原 real_society 选择 `scf` 劳动，直接读取 SCF LF，
  保留此前最近邻劳动行为但不再创建 BC 网络。两个美国场景的年龄分组先验选项明确为 US。
- 删除家庭实体中的 `fixed_saving_share` / `fixed_labor` 动作覆盖。
  效率冻结单独命名为 `freeze_labor_efficiency`，只控制环境状态演化，消费和劳动动作由策略提供。
- 核验记录保存在 `diagnostics/rule_based_households_20260907/`，不覆盖上一轮实验 JSON。
- 两个场景各 5 个种子、各 27 期，共 270 期的价格、通胀、实际/名义 GDP、劳动、资本、消费收入比、
  家庭存款和企业贷款与迁移前逐期相同（最大差值为 0）；运行时显式禁止 BC 实例化的核验通过。
  81 项测试通过，包含规则切换、自定义比例实际执行、SCF 劳动语义一致及旧规则兼容。

## 2026-09-07 四项建模收尾与 27 期实验

- `real_society` 启用企业—银行共享结算：初始资本对应初始贷款，企业现金为不计息银行账户；
  工资、销售、投资、实际付息、还本、新投资及营运贷款双边对应。贷款存量不再每期被新投资覆盖，
  默认计划归还期初本金的 5%，未还本金继续记账。新增银行权益及独立资产负债核验，资金缺口有明确终止原因。
- 投资改为有限资本缺口调整，并设置净扩张上限；非正使用成本通过有限可行边界处理，
  不再用极小正数放大目标。初始 K/Y 可显式配置，GDP 对应生产率随之重算，融资不足时报告错误。
- 新增固定总消费与固定劳动接口；`real_society` 暂以消费 95% 税后收入为验证基准，保留原 BC 劳动与冲击。
  `fixed_saving_share: null` 恢复原 BC 消费。旧标签仍明确为食物与租金的部分消费代理，未重训网络。
- 新增 `cfg/real_society_calibration.yaml`：同一政策与人口口径、95% 消费、固定初始有效劳动、无技术冲击、
  资本缺口每期调整 5%、净扩张上限 5%，默认仅测试 27 期。这是阶段二的控制基准，不是美国真实家庭策略。
- 本轮详细结果、原劳动动态及 BC 不利对照、27 期价格/通胀/GDP 明细与图：
  [实施与验收报告](diagnostics/model_closure_implemented_20260907/REPORT.md)。
  受控基准五个种子的平均通胀为 2.31%—3.52%，单期为 1.09%—5.29%，期末实际 GDP 指数为 93.71—120.53（初始 100）。
  原劳动动态仍可产生大幅衰退；不能只凭温和通胀宣称全模型已拟合美国。
- 新结算明确采用单银行贷款—存款对应记账；零准备金率没有机械的“先新增储蓄才能新增贷款”限制。
  正准备金率且无外部准备金融资时可能终止。未实现资本监管、内生违约、破产清算；部分种子企业账面权益偏低/为负。
  共享账本目前只支持 Ramsey 无风险分支，旧 OLG/养老金/风险资产场景默认保持旧结算，不宣称其全面闭合。
- 家庭借款下限、消费税与既有合同利率时序保持原规则。reset 回归包含新增账户。
  可开始受控宏观参数的预备校准；家庭策略准确性及完整随机劳动动态留给后续验证。
- 77 项回归测试通过，`git diff --check` 通过；保存 23 组诊断与配置记录。
  正式 `main.py` 入口运行 27 期，与校准基准默认种子的价格、通胀和 GDP 逐期一致。

## 2026-09-06 按用户确认的资产存款化假设统一初始本金

- `cfg/base_config.yaml` 的 `initial_savings_column` 及家庭读取的默认值改为 `ASSET`。
  同一 SCF 抽样家庭的全部初始资产作为银行本金，与无风险投资分支中期末
  `savings = at_next` 的口径一致。此条取代 9 月 5 日默认使用 `SAVING` 的初始化选择。
- 这是模型对资产形态的简化，不是 SCF 实际存款数据；没有新增资产分类或资产出售机制。
  初始本金在第一次 step 结算利息，reset 时仍无利息，后续继续使用原合同计息时间。
- 回归检查覆盖 ASSET 同行抽样、首期存款变化等于税后收入减资产税及含税消费、
  银行与家庭首期利息一致、reset 恢复初始本金。BC 标签、利率及借款下限保持原实现。
- 70 项测试全部通过，`git diff --check` 通过。同 seed 2 的 real_society 两期核验：
  初始存款 1,110,447,224，首期利息 18,688,826.78，首期末存款 1,195,171,478.38；
  第二期通胀 30.0989%。仅确认初始化及结算口径一致，不表示已获得稳定经济。
  可复现脚本和结果保存在 `diagnostics/asset_deposit_init_20260906/`。

## 2026-09-05 BC 消费标签口径修正与配对实验

- 新增 `agents/behavior_cloning/scf_labels.py`：年度食品加 12 倍月租，移除债务偿付 TPAY；
  使用 SCF 收入减模型现有联邦税额作为税后分母。明确标为部分消费代理，
  记录非正收入和动作范围无法表示的样本数；没有补造缺失消费或存款流量。
- `bc_agent.get_real_data()` 使用新标签；政府原联邦税函数提取为共享静态函数，
  保留税收数值不变。未修改存款账户、投资、价格及借款下限。
- 69 项测试通过。real_society 的 5 个种子分别重跑旧标签/新标签，旧标签轨迹与先前结果完全一致。
  默认种子首期消费/GDP 从 8.93% 增至 14.03%，第二期通胀从 29.70% 至 30.06%；
  修改后第 10 期因原借款下限终止。其余种子也未获得平稳通胀。
- 本轮延续 `bc_test=False` 的标签最近邻执行方式，没有重训已有网络权重。
  完整报告和轨迹：`diagnostics/bc_label_fix_20260905/REPORT.md`。

## 2026-09-05 用户确认的五项核算与 reset 修复

本次修改限于消费税、国债利息、SCF 期初存款及其计息、完整环境 reset、
多企业政府购买。家庭 `at_min` 越界后终止的逻辑保持原样，未加入消费前借款约束。

- `entities/households.py`：动作第 0 维仍为储蓄比例，其补数作用于可支配收入，
  给出含消费税预算。按实际成交额分别记录 `consumption_tax` 和
  `consumption_expenditure`，期末资产扣除含税支出；商品配给后未花掉的预算保留。
  Ramsey、OLG 及风险投资变体使用相同口径；OLG 退出家庭当期已支付的消费税
  单独保留给财政结算，避免删除家庭记录时丢失税收。
- `entities/households.py`、`cfg/base_config.yaml`：新增
  `initial_savings_column: SAVING`，按同一 SCF 抽样行取得储蓄账户本金，
  初始化 `savings_init` / `savings`，不再以零或全部 ASSET 代替期初存款。
  SAVING 是存量账户余额，定义见
  [美联储 SCF 官方变量构造程序](https://www.federalreserve.gov/econres/files/bulletin.macro.txt)。
  本次保留原有后续资产分配规则，未实施完整的住房、证券、存款资产分类模型。
- `entities/bank.py`、`entities/government.py`、`env/env_core.py`：增加统一期初结算。
  reset 是时间 0，没有即时利息收入；第一次 step 表示时间 0 到 1，
  期初 SCF 存款按初始合同利率计息；本期末新增存款到下一次 step 才计息。
  存款和家庭借款分开按上期存/贷款合同利率计算，不先相互抵销；
  银行收支和利润引用同一套实际结算金额，不再用新发放本金计算当期已赚利息。
  OLG 出生/死亡前冻结计息本金，保证退出家庭也完成该期结算。
- 国债采用明确的一期合约简化：新发国债利率等于当期政策利率，
  旧债在下一次 step 按发行时的利率结算；政府 `bond_interest_expense`
  与银行 `government_interest_payment` 使用同一金额。现实固定利率国债
  依合约支付票息，并不随政策利率即时重定价；模型没有实现期限结构和拍卖定价。
  参考：[美国财政部 Treasury Bonds](https://www.treasurydirect.gov/marketable-securities/treasury-bonds/)。
- `env/env_core.py`、`entities/government.py`、`entities/households.py`、`main.py`：
  在初始化与动作空间扩展完成后保存环境和实体状态快照，reset 原位恢复实体，
  清除期间新增缓存、增长率、税率、人口、贷款、分配记录等状态，保持智能体的实体引用有效。
  修正 OLG 人口先恢复、财政规模后初始化的顺序；初始动作重建而非重复追加。
  `set_tax_type()` 将显式选择的财政算法纳入初始化状态；Saez 的期内历史仍会清空。
  `reset(seed=...)` 可指定环境 NumPy 冲击种子；不指定时不强制每回合重复冲击。
  新校准参数须通过新配置创建环境，reset 返回该环境自己的初始状态，不保留临时属性改写。
- `entities/government.py`：先按当期名义 GDP 和政府支出比例计算总购买预算，
  再按企业预算份额和各自价格换算数量；全零分配向量解释为平均分配。
  商品不足时财政只记录实际购买支出。

验证：新增 `tests/test_accounting_reset.py` 的 12 项回归测试，含四种家庭分支、
商品配给、SCF 同行抽样、计息时序、国债双方金额、多价格企业预算、OLG 退出、
全状态恢复、财政算法持久配置与同种子重复轨迹。总计 **65 项测试通过**；
28 个场景完成初始化、单步与 reset 检查；`main.py --problem_scene inflation_control
--central_bank_alg rule_based --eval_episodes 2` 执行成功，结果位于
`agents/models/inflation_control/100/run41/evaluation_results.json`。
家庭借款终止函数通过修改前后 AST 一致性检查；`git diff --check` 通过。
这批修复不代表其他资金流、行为标签和投资动态问题已解决，也不构成正式校准完成的证据。

## 2026-09-04 12:15 CST Real Society 2022 政策基线与非完全竞争定价

- `cfg/real_society.yaml`：政府消费与总投资占 GDP 比例改为 $$17.1\%$$；
  基准利率改为 2022 年有效联邦基金利率月度值的年均值 $$1.683\%$$；
  法定准备金率改为 $$0$$；2022 年联邦遗产税基本免征额改为
  $$12.06$$ 百万美元。这些覆盖仅用于 `real_society`，不改动其他政策实验场景。
- `agents/rule_based/market.py`、`env/set_observation.py`、`entities/market.py`与
  `cfg/base_config.yaml`：非完全竞争的 rule-based 企业不再使用
  `price = Z × 随机倍数`和 `wage = Z × 随机倍数`。价格由 Cobb–Douglas
  单位边际成本乘显式 `markup` 决定，工资等于价格乘劳动边际产品。
  `markup` 保留为未来可校准参数。
- `tests/test_monetary_policy.py`：新增 2022 政策基线和成本定价的公式、
  规模不变性回归测试。$$47$$ 项测试全部通过，$$28$$ 个场景全部可初始化；
  三种非完全竞争市场在强制使用新 rule-based 策略时均完成单步运行。
- `real_society` 更新后的 3 种子、8 期快速检验中，期末价格为
  $$1.17\text{–}1.37$$，平均单期通胀约 $$11\%$$。因此政策参数已按 2022 口径更新，
  但家庭储蓄标签与投资动学尚未校准，不能将该试验解读为已复现 2022 美国经济。

## 2026-09-04 11:40 CST Real Society 切换为 SCF 直接检索家庭策略

- `cfg/real_society.yaml`：家庭策略改为 `house_alg: "bc"` 且
  `bc_test: False`，每期按当前家庭特征检索最近的 2022 SCF 样本并直接使用
  其劳动与当前构造的储蓄动作，不读取已训练的 BC 神经网络。
- `tests/test_monetary_policy.py`：新增配置与数据路径回归测试；完整测试
  $$45$$ 项全部通过。
- 实际检索得到的劳动参与率约为 $$71.4\%$$，但储蓄动作均值约为
  $$80.3\%$$。三个种子的 8 期试验仍出现明显价格波动，说明切换到真实数据检索
  已生效，但当前不完整消费项目除以资产构造的“储蓄率”仍不具有正确经济语义，
  不能作为已完成现实校准的家庭基线。

## 2026-09-04 10:32 CST 资本回报、银行利率与教育特征拆分

- `entities/market.py`：将原 `MarketClear_InterestRate` 更名为
  `capital_marginal_revenue_product`，仅表示资本边际收益产品；市场不再用它覆盖银行存贷款利率。
- `entities/bank.py`：`commercial` 银行继续使用自身决策的利率；`non_profit`
  银行作为被动基准，每期使用当期 `base_interest_rate`；税收或养老政府不再覆盖
  银行的货币政策参数。企业目标资本继续由
  银行名义贷款利率的 Fisher 转换值加折旧率决定。
- `runner.py` 和 `diagnostics/diagnose_deflation.py`：记录资本边际收益产品，便于将投资回报与融资成本分开检查。
- `env/set_observation.py`、`entities/households.py` 和 `agents/behavior_cloning/bc_agent.py`：
  BC 匹配使用原始 SCF `EDUC`，生产仍使用校准后的效率 `e`；OLG 出生、死亡和 reset
  过程保持教育数组与人口索引同步。
- `tests/test_monetary_policy.py`：新增资本回报不改写银行利率、非营利银行基准利率传导、
  OLG 新生教育与效率对齐的回归检验。完整测试 $$45$$ 项全部通过，
  $$28$$ 个场景均可初始化。
- `real_society` 的 10 种子、20 期试验表明：利率覆盖错误已消失，但价格仍不稳定。
  通缩期占比平均为 $$61.5\%$$，部分期间又出现较大通胀跳升；因此该修复不等于
  总需求、投资和价格动学已完成校准。

## 2026-09-03 17:08 CST 初始化口径、政策时序与 OLG 修正

### 经济尺度与初始化

- `entities/government.py`、`entities/households.py`、`entities/market.py` 与 `env/env_core.py`：统一当前 agent 的统计口径。由于目前仍使用 SCF 家庭/Primary Economic Unit 数据，Ramsey 和 OLG 场景都按模拟 PEU 数量缩放基年 GDP：

$$
s_N=\frac{N}{H_{SCF}},
\qquad
GDP_0=s_NGDP_{US,2022}.
$$

  这一口径取代了先前 OLG 按真实人口缩放、但资产却来自 SCF 家庭的混合口径。当前 OLG 仍是“以 SCF 参考人代表 PEU 生命周期”的过渡模型，尚未改造为严格的个人 OLG 数据体系。

- `entities/market.py`：删除 YAML 中手工给定的 $$Z_0$$，在 reset 的第 0 期用缩放后 GDP、资本和有效劳动一次性校准：

$$
Z_0
=
\frac{GDP_0}
{\sum_jK_{j,0}^{\alpha}L_{j,0}^{1-\alpha}}.
$$

  校准只在初始化时执行，不在 `step()` 中重复覆盖，因此后续 $$Z_t$$ 仍可以动态变化。

- `entities/households.py`：将 SCF `EDUC` 定义为初始劳动效率的代理变量，并按实际工时将就业者平均效率归一化为 1。工时动作继续表示 $$[0,1]$$ 比例，实际工时为 $$h_{i,0}=LF_i h_{max}$$；修正了初始动作预先乘 $$h_{max}$$、进入环境后又被乘一次的单位错误。

- `entities/market.py`：按 Cobb–Douglas 劳动份额校准初始每有效工时工资：

$$
W_0=\frac{(1-\alpha)GDP_0}{L_0}.
$$

- `env/env_core.py`：扩展 action space 时改为从 `Box.high` 和 `Box.low` 数组取边界，不再依赖可能为 `None` 的 `high_repr/low_repr`，修正了多市场和多政府配置的偶发初始化失败。

### 生产率增长过程

- `entities/market.py`：根据当前建模选择，使用非负随机技术进步：

$$
\log Z_t
=
\log Z_{t-1}+\sigma_zU_t,
\qquad U_t\sim U(0,1).
$$

  因此 $$\sigma_z$$ 的含义是“每期对数生产率增量上界”，不是冲击标准差；平均对数增长为 $$\sigma_z/2$$，标准差为 $$\sigma_z/\sqrt{12}$$。基础配置中的参数注释已同步更新。未来若需要独立校准长期增长和短期波动，应再拆分为趋势参数 $$g_z$$ 与零均值冲击参数 $$\sigma_z$$。

### 货币政策时序

- `runner.py`、`env/env_core.py`、`entities/bank.py`：中央银行先生成并更新本期基准利率和准备金率，商业银行再观测该政策并决定本期存贷款利率。本期贷款利率、存款利率和投资信贷现在受同一期中央银行动作约束，消除了利率渠道使用上期政策、准备金渠道使用本期政策的混合时序。

### OLG 退休、新生与遗产

- `entities/households.py` 与 `env/env_core.py`：封装 `update_retirement_status()`，在企业汇总劳动之前将退休者工时清零；人口更新后重新计算老龄占比和赡养比。`runner.py` 现在直接读取 households 中真正更新的这两个指标。
- `entities/households.py`：新生主体从已校准的 `e_array_0` 和对应 `work_init` 联合抽样，修正了将“零养老金账户”误当成新生工时的问题。
- `entities/households.py` 与 `entities/government.py`：遗产机制改为读取本期真正死亡主体的期末非负净资产 `at_next`，普通资产税先按现有方法扣除，再对遗产按免征额和遗产税率计税。税后遗产池平均分给本期新生主体；若本期没有新生主体，则结转到下一期。
- 同时修正遗产税未按期清零、遗产税总额被广播到每个家庭从而放大政府税收、以及死亡主体已缴直接税在删除数组时丢失的问题。

### 训练日志与验证

- `agents/behavior_cloning/bc_agent.py` 和 `agents/rl/ppo_agent.py`：记录 loss 前使用 `detach()` 并转成 Python 标量，避免 SwanLab 将 `requires_grad=True` 的 tensor 直接转成 `float` 的警告和意外保留计算图。
- `tests/test_monetary_policy.py`：新增了初始 GDP 缩放、$$Z_0$$ 和 $$W_0$$ 校准、效率归一化、正向生产率增长、当期货币政策传导、退休劳动时序、新生效率、真实死亡对象及遗产总量保存等回归测试。
- 2026-09-03 17:08 CST 执行完整测试：$$38$$ 项全部通过；`compileall` 和 `git diff --check` 通过。

### 当前边界

- OLG 继续沿用 SCF 时，资产和收入是 PEU 家庭量，而年龄、教育和就业状态主要来自参考人。因此当前的出生、死亡、退休、养老金和遗产都是 PEU 层面的过渡性近似，不能直接解读为美国个人人口动态。
- 当前 SCF 预处理文件有 $$22{,}975$$ 行，对应 $$4{,}595$$ 个家庭的五组插补记录，但未保留家庭 ID 和 implicate 标识。当前可用于加权点估计和过渡性抽样，但不应把五组插补当成互相独立的真实家庭，也不应据此低估抽样不确定性。
- 本批测试证明代码按上述公式和时序执行，不等于已完成参数校准、个人 OLG 数据改造或现实政策效果验证。

## 2026-08-13 13:36 CST

- `entities/market.py`：将含工资的 CES 指数替换为年度链式 Fisher 消费价格通胀。
- `env/env_core.py`：保存相邻年份的消费价格和实际消费量。
- `tests/test_monetary_policy.py`：增加 Fisher 指数与单一商品通胀测试。

## 2026-08-13 15:34 CST

- `entities/households.py`：保存限额前的家庭计划消费需求 $$D_t^H$$，避免成交量 $$\min(D_t,S_t)$$ 掩盖短缺。
- `entities/market.py` 与 `env/env_core.py`：本期计划总需求 $$D_t=D_t^H+G_t$$ 决定下一年价格：

$$P_{t+1}=P_t\left(\frac{D_t}{S_t}\right)^\lambda,\qquad \lambda=1.$$

  $$\lambda$$ 是价格调整速度；$$\lambda=1$$ 表示价格完全调整至本期供需隐含的目标水平。
  时序上先用 $$P_t$$ 完成本期交易和指标计算，再生成 $$P_{t+1}$$，避免将下一期价格错用于本期通胀和名义 GDP。
  该规则只用于完全竞争市场；其他市场结构仍使用企业 agent 的定价动作。代码中的 $$10^{-8}$$ 仅是零需求时保持价格为正的数值保护，不是经济冲击参数。
- `entities/bank.py`：投资和资本统一为实际量，名义信贷上限用资本品价格转为实际资本上限：

$$I_t=\max(Y_t-G_t-C_t,0),$$

$$K_{t+1}^{d}=(1-\delta)K_t+I_t,$$

$$Credit_t^{available}=\max\{Deposits_t(1-rr_t)-B_{t+1},0\},$$

$$K_{j,t+1}=K_{j,t+1}^{d}\min\left\{1,\frac{Credit_t^{available}}{\sum_jP_{j,t}K_{j,t+1}^{d}}\right\}.$$

  银行资产负债表中的资本贷款同样按名义价值 $$\sum_jP_{j,t}K_{j,t}$$ 记账。

- `entities/government.py`：区分实际 GDP 与名义 GDP：

$$GDP_t^{real}=\sum_jY_{j,t},\qquad GDP_t^{nominal}=\sum_jP_{j,t}Y_{j,t}.$$

  中央银行的增长率继续使用实际 GDP，避免将价格上涨误认为实际增长。
- `runner.py` 与 `utils/evaluation_report.py`：评估结果保留 `GDP`（实际 GDP），新增 `nominal_GDP`，并在指标结构改变时自动重写旧对比 CSV 的表头，避免列错位。
- `entities/market.py`：更新价格时生成新数组，避免历史价格记录被后续的原地修改覆盖。
- `tests/test_monetary_policy.py`：增加 $$\lambda=1$$ 的价格调整测试。

### 验证结果与边界

- 23 个单元测试全部通过；`main.py --problem_scene inflation_control --central_bank_alg rule_based --eval_episodes 1` 完整运行通过。
- 当前 EconGym 没有独立的计划投资需求；投资仍是 $$Y_t-C_t-G_t$$ 的事后剩余。因此，现阶段价格规则中的计划需求只包含 $$D_t^H+G_t$$，在现有参数下会显示持续供过于求和较强通缩。这是当前模型需求结构的结果，不应解读为已经验证的现实通胀动态。

## 2026-09-01 第一批初始化与价格校准

### 本批修改

- `entities/households.py`：在归一化 SCF 权重前保存 2022 年美国家庭/Primary Economic Unit 总数：

$$
H_{US,2022}=\sum_i WGT_i.
$$

  当前数据计算得到 $$H_{US,2022}=131{,}306{,}389.38347022$$。注释已注明当前数据年份与国家；后续应由统一的数据通道上传对应国家—年份的 GDP、人口和家庭单位数。

- `entities/government.py`：按 agent 的经济单位分别缩放初始 GDP。Ramsey 中每个 agent 代表家庭：

$$
GDP_0^{Ramsey}
=
\frac{GDP_{US,2022}}{H_{US,2022}}N_H.
$$

  OLG 中每个 agent 代表个人：

$$
GDP_0^{OLG}
=
\frac{GDP_{US,2022}}{Population_{US,2022}}N_P.
$$

  其中基础配置暂用 2022 年美国现价 GDP $$25.4746$$ 万亿美元、人口 $$333.428$$ 百万人。初始价格归一化为 $$P_0=1$$，因此该现价 GDP 仅作为基年数量尺度，不表示已构造完整的不变价 GDP 序列。

- `entities/market.py`：家庭资产减政府债务得到初始资本的原逻辑保持不变；在观察首期实际劳动后，按比例校准初始生产率，使 Cobb–Douglas 首期产出与上述 GDP 尺度一致。先计算：

$$
\widetilde Y_0
=
\sum_j Z_{j,init}K_{j,0}^{\alpha}L_{j,0}^{1-\alpha},
$$

  再设置：

$$
Z_{j,0}
=
Z_{j,init}\frac{GDP_0}{\widetilde Y_0},
$$

  因而：

$$
\sum_jY_{j,0}=GDP_0.
$$

  随机生产率冲击从第二期开始，避免首期校准值立即被冲击覆盖。

- `cfg/base_config.yaml` 与 `env/env_core.py`：价格调整速度改为可配置参数，并暂设：

$$
\lambda=0.2.
$$

  完全竞争市场沿用已确认的计划需求—可售供给规则：

$$
D_t=C_t^d+G_t^d+I_t^{F,p},
$$

$$
Q_t=Y_t+N_t,
$$

$$
P_{t+1}
=
P_t
\left(
\frac{D_t+10^{-8}}{Q_t+10^{-8}}
\right)^{0.2}.
$$

  $$\lambda$$ 是每期价格向供需隐含价格调整的速度，不是已由美国官方直接给定的参数；后续需校准。少数不合并基础配置的旧场景使用同一 $$0.2$$ 作为兼容默认值。

### 当前代码中已实现、但本批未重新改动的投资与库存闭环

`entities/bank.py`、`entities/government.py`、`entities/households.py`、`entities/market.py` 和 `env/env_core.py` 当前共同执行以下计算：

$$
r_t^L
=
\frac{1+i_t^L}{1+\pi_{t-1}}-1,
$$

$$
u_t^K
=
\max\{r_t^L+\delta,10^{-8}\},
$$

$$
K_t^*
=
\left(
\frac{\alpha Z_tL_t^{1-\alpha}}{u_t^K}
\right)^{\frac{1}{1-\alpha}},
$$

$$
\bar K_t=(1-\delta)K_t,
$$

$$
I_t^{F,d}=\max\{K_t^*-\bar K_t,0\},
$$

$$
Credit_t^{available}
=
\max\{Deposits_t(1-rr_t)-B_{t+1},0\},
$$

$$
I_t^{F,p}
=
I_t^{F,d}
\min\left\{
1,
\frac{Credit_t^{available}}
{\sum_jP_{j,t}I_{j,t}^{F,d}+10^{-8}}
\right\},
$$

$$
I_t^F
=
\min\{I_t^{F,p},Q_t-G_t-C_t\},
$$

$$
K_{t+1}=\bar K_t+I_t^F,
$$

$$
N_{t+1}=Q_t-G_t-C_t-I_t^F.
$$

本批没有修改 BC 训练标签或家庭动作定义、政府购买比例、库存折旧设定、目标资本公式和银行信贷公式。

### 测试与结果

- `tests/test_monetary_policy.py` 新增/更新 Ramsey 家庭口径、OLG 人口口径、首期 Cobb–Douglas 产出校准、投资恒等式、商品核算、价格公式及单商品零消费通胀测试。
- 全部 $$28$$ 个单元测试通过；`git diff --check` 通过。
- Ramsey 场景、$$100$$ 个家庭 agent 的初始 GDP 为：

$$
GDP_0^{Ramsey}=19{,}400{,}883.779999.
$$

- OLG 场景、$$1000$$ 个个人 agent 的初始 GDP 为：

$$
GDP_0^{OLG}=76{,}402{,}101.803088.
$$

- seed $$1$$ 的受控首期检验中，实际产出精确等于 Ramsey 初始 GDP；商品核算残差、通胀计算误差和价格公式误差均为数值零。首期 $$K_t^*/K_t=1.10384$$，计划、融资支持和实际固定投资均为正；计划需求/可售供给为 $$0.77289$$，价格由 $$1$$ 调整到 $$0.94978$$。
- 完整命令 `main.py --problem_scene inflation_control --central_bank_alg rule_based --eval_episodes 1` 成功运行 $$8$$ 期。最终一期价格为 $$0.32447$$、通胀率为 $$-31.046\%$$、库存为 $$54.6071$$ 百万实际商品单位。

### 结论与边界

本批已经修正“家庭 agent 却按人口缩放 GDP”的口径错误，并保证首期 GDP、资本、劳动和生产率处在同一个 Cobb–Douglas 数量体系内；$$\lambda=0.2$$ 也降低了单期价格跳变。测试证明这些公式按代码预期执行，但完整实验仍出现库存累积和显著通缩。因此，当前可以确认的是初始化与核算实现正确，不能确认整个 EconGym 已完成经济动态校准，也不能据此比较 LLM 与 rule-based 政策优劣。

## 2026-09-02 删除库存，恢复无库存基准

### 修改内容

- `entities/market.py`：删除 `inventory` 状态。每期可交易商品只来自当期产出：

$$
Q_t^s=Y_t.
$$

- `env/env_core.py`、`entities/government.py`、`entities/households.py` 与 `entities/bank.py`：政府、家庭和固定投资继续按既有顺序使用当期商品，实际交易满足：

$$
G_t=\min\{G_t^d,Y_t\},
$$

$$
C_t=\min\{C_t^d,Y_t-G_t\},
$$

$$
I_t^F
=
\min\{I_t^{F,p},Y_t-G_t-C_t\}.
$$

未成交产出为：

$$
X_t
=
Y_t-G_t-C_t-I_t^F\geq0.
$$

当前代码不保存 $$X_t$$，也不将其带入下一期；因此它是未交易、未结转的当期剩余量，不是库存投资。资本积累仍为：

$$
K_{t+1}=(1-\delta)K_t+I_t^F.
$$

- 完全竞争市场的计划需求保持：

$$
D_t=C_t^d+G_t^d+I_t^{F,p}.
$$

价格更新的供给分母由 $$Y_t+N_t$$ 改回当期产出 $$Y_t$$：

$$
P_{t+1}
=
P_t
\left(
\frac{D_t+10^{-8}}{Y_t+10^{-8}}
\right)^{0.2}.
$$

- `runner.py`：终端评估表和跨 run 比较表删除 `inventory` 指标。
- `diagnostics/diagnose_deflation.py`：删除库存结转消融和库存指标，改为报告未成交产出比例；默认价格调整速度与基础配置统一为 $$0.2$$。
- `tests/test_monetary_policy.py`：删除库存恒等式，新增“环境不存在库存状态”“当期商品供给等于当期产出”“实际商品使用不超过当期产出”的检查。

### 测试结果

- $$28$$ 个单元测试全部通过。
- `git diff --check` 通过。
- `main.py --problem_scene inflation_control --central_bank_alg rule_based --eval_episodes 1` 完整运行 $$8$$ 期，输出中不再包含库存指标。
- 本次完整运行的最终报告值为：价格 $$0.41204$$、通胀率 $$-20.4061\%$$。上一版带库存运行对应为价格 $$0.32447$$、通胀率 $$-31.046\%$$；删除库存减轻了通缩，但没有消除通缩。
- seed $$1$$ 的逐期核算中，全部 $$8$$ 期均不存在库存状态，实际使用均不超过当期产出。第一期计划需求/产出为 $$0.77289$$，第二期以后约为 $$0.215$$；对应未成交产出比例接近 $$78.4\%$$。

### 解释边界

这次修改只删除库存及其跨期累积，不等于已经求解 Walrasian 静态均衡。当前仍是“给定当期产出和价格后按短边成交，未成交产出不结转，再按计划需求/产出调整下一期价格”。因此，剩余通缩说明当前家庭消费、政府购买和固定投资形成的计划需求仍明显低于产出；不能仅凭通缩改善就认定环境已经完成经济校准。
