# AgentEWM 自动化校准接入（本地开发版）

**最新进展（冻结口径与价格子空间）：**[口径契约](PRICE_STABILITY_CONTRACT_20260908.md)与[重复搜索报告](PRICE_STABILITY_REPORT_20260908.md)。74 次批量调用、1,856 条路径完成。私人非住宅参考条件下，16-seed 搜索在两个合成真值上分别恢复 4/6、5/6 次，排序改善；但相对默认值的改进区间仍含零，不能宣称显著优势或唯一识别。投资速度／预期系数保持固定参考，协作者机制未改。13 项新测试和重新执行的 64 项接口检查通过；真实经济拟合仍关闭。

**最新状态（基础条件／噪声／核算）：**[完整开发报告](FOUNDATION_NOISE_ACCOUNTING_REPORT_20260908.md)。737 条路径均运行有效，但八组基础条件均未支持三参数共同可区分；目标资本封顶可使投资速度代数抵消并屏蔽预期传导。新增 27 类部门核算，5,892 个期间记录通过，64 项单元检查通过。已补九条官方宏观序列；32-target/16-search/16-validation 显示抽样波动下降不保证验证选择改善。真实数据口径和完整任务书验收仍未完成，协作者机制未修改。下方较早数字保留为历史记录，以本段及最新报告为当前状态。

2026-09-08。基于协作者 [AgentEWM / ddc0a418965db3982fd594c678329309c241aeb1](https://github.com/Miracle1207/EconGym/tree/ddc0a418965db3982fd594c678329309c241aeb1)。

**最新测量接口：**[独立 MAE 验收与投资传导诊断](MAE_INVESTMENT_DIAGNOSTICS_20260908.md) 已完成，54 项单元检查通过；最新开发回归 14 条路径、79 个投资诊断均通过，新增日志不改变原轨迹。旧分块候选仅在事后“平均路径 MAE”下达标，原逐期验证失败不变，真实任务书验收仍未完成。

**最新开发运行（用户随后授权测试）：**[运行与效果报告](RUNTIME_EFFECT_REPORT_20260908.md)。69 条原开发路径及 1 条适配边界修正后回归均有效；模拟目标校准综合误差下降但独立逐矩验证仍失败。当前 30 项单元检查通过。未做真实数据优化，未修改协作者源码。以下“仅接口、不启动模拟器”说明记录的是上一阶段边界，不是对本次已授权开发测试的禁止。

**当前优先级：先完成任务书模块与接口适配，再进行实验跑通，效果改进后置。**当前通用求解核心、三参数映射、基础参数数据卡与数据准备接口已连接；完整任务书验收接口尚未接齐，真实经济校准未启动。

请优先阅读 [当前适配清单与后续顺序](ADAPTATION_STATUS.md)。本轮最新检查为 [29 项单元／接口检查](runs/interface_alignment_v3/checks.json)：无真实模拟器调用、无优化运行。下列旧仿真记录保留为此前版本证据，不代表本轮新增接口已经完成真实模拟器回归。

2026-09-08 数据补充：已接入 41 个现有数据文件（约 126 MB）、宏观候选观测和资本候选，以及只读数据接口。详见 [数据接入与剩余缺口](DATA_INTEGRATION.md)。原任务书和初次对齐报告保留历史状态，以该补充说明更新数据是否已经存在的判断。

活动目录：`H:\EconGym\integrations\AgentEWM-calibration-20260908`。
仅新增本 `calibration/` 目录。未修改协作者的 `entities/`、`env/`、`agents/`、`cfg/`、`runner.py`、`main.py` 或 `update.md`，也未覆盖原有本地 EconGym／H196 机制与历史结果。

- [整合、差异与逐项待确认报告](ALIGNMENT_REPORT.md)
- [收到的任务书原文](TASKBOOK_RECEIVED_20260908.md)
- [逐文件对齐结果](runs/audit_v2/alignment.json)
- [此前接口回归与参数响应](runs/audit_v2/checks.json)
- [27 期开发检查](runs/development_27period_v1/result.json)
- [模拟目标校准闭环，验收失败保留](runs/synthetic_calibration_v2/result.json)

## 方法接入与职责

| 文件 | 作用 |
|---|---|
| `core.py` | 从既有拓展逐字节复用的通用分块／随机搜索核心；候选冻结后一次独立种子验证 |
| `bridge.py` | AgentEWM 参数映射、受控配置、上游规则调用、分期测量、源码保护、隔离子进程 |
| `launch.py` | 正常 Python 或显式本地依赖路径启动；无旧 H196 bootstrap、无机制替换 |
| `taskbook_inputs.py` | 外部基础参数卡、数据包校验与准备状态；不自动从数据估参 |
| `data_access.py` | 已收集原始／整理数据的只读接口、完整性检查与未确认目标拒绝 |
| `test_bridge.py`、`test_taskbook_inputs.py` | 共 29 项配置、输入、统计及替身传递检查 |
| `check_interfaces.py` | 本阶段使用的无模拟器接口检查入口 |
| `audit.py` | 此前上游测试、逐文件对齐、复现与参数响应脚本；包含真实仿真，本阶段不运行 |

本轮三个联合估计控制量为 `capital_adjustment_speed`、`price_adjustment_speed`、`inflation_expectation_lambda`。仅它们能作为优化参数传入后端 `evaluate()`。`alpha`、折旧和初始 K/Y 有独立的 `--foundation-card` 接口；模板为 [foundation_card.template.json](foundation_card.template.json)，未填写的模板不能执行。数据卡随配置和进程请求留档，接口检查不等于数据依据获准。当前未选择或应用新基础参数，继续使用未经估计的协作者默认配置。

从 `real_society.yaml` 合并基础配置，运行时覆盖 `initial` 劳动、冻结效率、`sigma_z=0` 与 95% 消费规则。**不加载 `real_society_calibration.yaml`，不改变场景名称**；后者资本速度 .05，而任务书基准继承 .20。

联合搜索先复用现有 `method='random'` 作为可运行基线；分块核心保留，但没有证据将三个宏观参数逐一绑定到单调矩，因此不默认使用坐标二分。BO、Jacobian 和 H235 识别器尚未迁入本新后端。

每一模拟 seed 使用新环境，参数在创建环境之前进入配置。每一配置使用独立子进程和排他创建的目录。请求、全配置、轨迹、异常、标准输出、数据及源码哈希均留档。

## 如何运行

**当前仅使用不启动模拟器的两个入口：**

```powershell
python -B calibration/check_interfaces.py --repo . --output calibration/runs/new_interface_check
python -B calibration/launch.py --repo . prepare --data-bundle calibration/data/collected_v1 --n 1000 --horizon 27 --output calibration/runs/new_prepare
```

`prepare` 可另加 `--request <三参数JSON>` 和 `--foundation-card <基础参数数据卡JSON>`；不提供时保留原暂定参数。数据附件不会自动成为训练目标，也不会替换初始化家庭样本。准备记录会分别保存优化参数、基础参数及数据就绪状态。

### 后续接口齐备后才使用的模拟命令

在包含 numpy、torch、gymnasium、omegaconf 等协作者依赖的普通 Python 环境中，从仓库根目录运行：

```powershell
python -B calibration/launch.py --repo . prepare --n 1000 --horizon 27 --output calibration/runs/my_prepare
python -B calibration/launch.py --repo . evaluate --n 64 --horizon 3 --seeds 908501 --output calibration/runs/my_evaluate
python -B calibration/launch.py --repo . synthetic-calibrate --output calibration/runs/my_synthetic
```

目录必须不存在，不删除旧结果后重跑。`synthetic-calibrate` 只执行固定的模拟目标软件测试；没有读取真实宏观目标的 CLI，也不会创建 `cfg/real_society_calibrated.yaml`。

当前 H 盘嵌入式 Python 不自动搜索 site-packages，完整启动例：

```powershell
$repo = 'H:\EconGym\integrations\AgentEWM-calibration-20260908'
& 'H:\EconGym\.runtime311\python.exe' -B "$repo\calibration\launch.py" --repo $repo --runtime-path 'H:\EconGym\.runtime311pkgs' --runtime-path 'H:\EconGym\.runtime' prepare --n 1000 --horizon 27 --output "$repo\calibration\runs\new_prepare"
```

`evaluate --request <json>` 的文件只接受以下字段，其他参数拒绝：

```json
{"parameters": {"capital_adjustment_speed": 0.2, "price_adjustment_speed": 0.2, "inflation_expectation_lambda": 0.5}}
```

接口参数域只是开发安全范围，**不是已同意的经济先验／正式搜索域**。不同基准参数、拟合期和真实目标需另建经确认的请求协议。

## 不能混淆的验收

- `valid=true` 只表示路径完成且有限、当前部分核算未报错；不等于任务书全项通过。
- 通胀首期零值从评分矩剔除。GDP 首期增长保留作初始化诊断，其拟合不能算独立验证。
- 比率按同期名义交易额计算；未成交产出单列，不伪造库存投资。总体 GDP 定义仍需共同确认。
- 搜索核心使用标准化逐期残差 RMSE，任务书验收要求 MAE，二者不能混写。已提供按种子平均路径及逐种子计算 MAE 的函数；真实验收层仍关闭。
- 当前部分核算涵盖银行资产负债、资本律和商品非超额使用；企业现金／贷款残差记录但全套逐式尺度、家庭与财政预算尚未独立验收。
- 无加权微观 Gini 现实比较；不采用上游特殊归一化 Gini 冒充官方定义。
- 本次只用公开开发 seeds；未执行任务书 0—29 正式稳定性组，未打开旧封存数据。
- 小样本独立种子验收失败正常保留；不能放宽模拟目标容差使它通过。
