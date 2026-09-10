# AgentEWM calibration extension — integration snapshot

> **Development integration, not a completed real-data calibration release.**
> 本分支用于协作审阅与接口整合；跑通、合成目标拟合、参数识别和现实预测是不同层级的证据。

## Scope / 本次提交范围

Based on `AgentEWM` commit `ddc0a418965db3982fd594c678329309c241aeb1`.
The initial snapshot (`33334ec`) only adds `calibration/`. This documentation update
also adds a calibration navigation notice to the root `README.md`.
Upstream economic mechanisms, configuration files, `main.py`, `runner.py`, and `update.md` remain unchanged.

校准算法和经济引擎分开：通过运行时配置与测量适配器调用协作者代码，不覆盖其源码。
这并不代表运行条件不变：适配器明确限定 Ramsey、单商品、单企业、无风险资产配置、固定规则消费与劳动，并冻结部分随机过程。
目前不是适用于所有 AgentEWM 场景的通用插件，也尚未连接网页或主应用入口。

**首次运行请先阅读 [Quick start / 从零跑通自动校准](#quick-start)。**

### Start here / 阅读顺序

1. [当前整合问题与完成状态](INTEGRATION_ISSUES_STATUS_20260908.md)：接口、数据、识别、核算和验收缺口。
2. [价格子空间口径契约](PRICE_STABILITY_CONTRACT_20260908.md)与[重复搜索报告](PRICE_STABILITY_REPORT_20260908.md)。
3. [基础条件、抽样波动与部门核算](FOUNDATION_NOISE_ACCOUNTING_REPORT_20260908.md)。
4. [独立 MAE 与投资传导诊断](MAE_INVESTMENT_DIAGNOSTICS_20260908.md)。
5. [原开发拟合结果及失败边界](RUNTIME_EFFECT_REPORT_20260908.md)。
6. [数据接入说明](DATA_INTEGRATION.md)与[初始差异清单](ALIGNMENT_REPORT.md)。
7. [发布范围与复现边界](PUBLICATION_SCOPE.md)。

以上报告保留原开发记录，含本机路径及未随此分支发布的证据链接；它们不是本分支的安装说明。
报告内较早的计数和状态以较晚实验及当前问题报告为准。原本地 README 另存为
[LOCAL_DEVELOPMENT_README.md](LOCAL_DEVELOPMENT_README.md)，仅供追溯。

## Components / 方法与接口

| Component | Responsibility / 职责 |
|---|---|
| `core.py` | 既有分块搜索与随机搜索核心；不是新实现的 BO 算法 |
| `bridge.py`, `launch.py` | 三参数映射、受控配置、隔离运行、测量与日志 |
| `taskbook_inputs.py` | 外部基础参数卡与准备状态；不会自动估计基础参数 |
| `data_access.py` | v1 数据只读访问、完整性检查与未确认目标拒绝 |
| `mae_acceptance.py` | 独立 MAE 验收；不覆盖旧逐期验收结论 |
| `investment_diagnostics.py` | 投资链条与约束绑定诊断 |
| `sector_accounting.py` | 限定机制下的 27 类部门核算检查 |
| `experiments/price_stability_v1/` | 冻结价格子空间开发实验与统计测试 |

可传入的优化参数为 `capital_adjustment_speed`、`price_adjustment_speed`、
`inflation_expectation_lambda`。`alpha`、折旧和初始 K/Y 属于单独的外部基础参数卡。
通用接口仍允许三参数输入；价格子空间的固定参数策略只在专用实验中执行，尚未统一到所有入口。

## Evidence / 结果边界

| Development experiment | Result | Limitation |
|---|---|---|
| 合成目标分块 / 随机搜索 | 综合误差下降 | 原独立逐期验证仍失败；不能改写成真实预测改善 |
| 基础条件扫描 | 75 次批量调用、737 条路径有效 | 八组条件均不支持三参数联合识别；投资上界可抵消速度效应并屏蔽预期传导 |
| 价格子空间重复搜索 | 74 次批量调用、1,856 条路径；私人非住宅条件下 16-seed 搜索恢复合成真值 4/6、5/6 次 | 相对默认值的改进区间含零；非唯一识别；重复搜索共享目标和验证池 |
| 核算与日志 | 有限定路由的开发检查证据 | `valid=True` 不代表完整任务书验收；不是现实经济准确性证明 |

四组原始汇总 JSON 保存在 `evidence/`，按原字节复制；不是逐路径复现包。
没有发布最终校准配置，也没有完成真实数据优化、正式留出或新模型上的强统计预测基线比较。

<a id="quick-start"></a>

## Quick start / 从零跑通自动校准

### 0. 运行范围与数据

本节运行 **Ramsey 固定规则路由的合成目标自动校准测试**，用于检查配置、模拟器、
优化器和独立验证能否连通。不是历史真实数据拟合，也不是 1,856 条路径的价格稳定性研究实验。

**不需要另发数据压缩包。**当前路由使用的 `agents/data/advanced_scfp2022_1110.csv`
（约 3.2 MB）已经随上游代码保留。无需 `calibration/data/collected_v1` 或
`collected_supplement_v2`，无需模型权重、GPU 或 OpenAI API key。
请完整克隆仓库，不要只下载 `calibration/` 文件夹。

### 1. 获取代码

私有仓库需要访问权限和正常的 GitHub 登录；不要把令牌写入命令、文件或仓库。

```powershell
git clone --branch calibration/agentewm-integration-20260910 https://github.com/SuperiseHua/EconGym-calibration.git
cd EconGym-calibration
```

后续命令均在仓库根目录执行。已有副本请确认处于该分支，不要覆盖未提交改动。

### 2. 准备 Python 环境

本机已验证 Python **3.11.9 / CPU**。Windows PowerShell 示例将环境建在仓库外，
避免误提交依赖，也不需要修改脚本激活策略：

```powershell
py -3.11 -m venv ..\econgym-calibration-venv
$calPython = (Resolve-Path ..\econgym-calibration-venv\Scripts\python.exe).Path
& $calPython -m pip install numpy pandas gymnasium omegaconf torch
& $calPython -c "import numpy, pandas, gymnasium, omegaconf, torch; print('calibration dependencies imported')"
```

这是当前固定规则路由的直接依赖，不是完整平台依赖表。仅跑本节无需安装网页、RL/LLM
训练等全部依赖。若已有可运行环境，可把后续 `& $calPython` 替换为该环境的 `python`。
Linux/macOS 可使用 `python3.11 -m venv ../econgym-calibration-venv`，调用其 `bin/python`。

本机运行版本记录：Python 3.11.9、NumPy 2.4.6、pandas 2.2.3、gymnasium 1.3.0、
OmegaConf 2.3.0、PyTorch 2.13.0+cpu。它们是本地记录，不是跨平台锁文件。
上述联网安装命令尚未在全新环境逐平台验证；不同版本不保证逐位相同结果。

### 3. 静态测试

```powershell
& $calPython -B calibration/run_unit_tests.py
```

预期 **77 tests / OK**、`simulator_module_imported=False`：64 项接口 / 测量 / 核算测试，
加 13 项稳定性测试。不读取额外调查包，不运行模拟器。
不要改用旧 `check_interfaces.py`，它还依赖完整本地数据包及旧拓展目录。

### 4. 准备配置并检查一条模拟路径

```powershell
& $calPython -B calibration/launch.py --repo . prepare --n 64 --horizon 3 --output calibration/runs/quickstart_prepare_001
& $calPython -B calibration/launch.py --repo . evaluate --n 64 --horizon 3 --seeds 910301 --output calibration/runs/quickstart_evaluate_001
```

`prepare` 只生成 `controlled_config.yaml` 和 `readiness.json`，不调用模拟器。
未附加真实数据时，`real_calibration_ready=False`、`data.attached=False` 是正确状态。
`evaluate` 使用默认参数模拟一条路径，不搜索参数；查看其 `result.json` 的 `valid`、
`paths` 和逐期诊断后，再进入完整闭环。

### 5. 自动校准完整闭环

```powershell
& $calPython -B calibration/launch.py --repo . synthetic-calibrate --n 64 --horizon 3 --output calibration/runs/quickstart_calibration_001
```

内部依次执行：

1. 默认参数、seed `908201` 生成合成目标。
2. 随机搜索入口评估初始配置和另一组随机配置，共 **2 组搜索配置**。
3. 在读取验证结果之前选出候选参数。
4. 使用 seed `908202` 独立验证一次并保存结果，不根据验证结果继续调参。

正常完成共 **4 次批量调用、4 条路径**：目标 1 + 搜索 2 + 验证 1，每条路径 64 个家庭、3 期。
四条路径不是四个独立种子。目标与搜索复用 `908201`，代码已披露其重叠，
因此本测试不能作为确认性参数恢复证据。

**这是固定软件测试入口，不是自由配置的研究实验：**

- `synthetic-calibrate` 固定上述种子、搜索预算和方法；`--seeds` 不改变内置种子。
- 没有 `--method` 或搜索预算 CLI 参数，不接受 `--request` 外部目标。
- 三参数测试范围：资本速度 `[0.05, 0.4]`、价格速度 `[0.05, 0.4]`、预期系数 `[0.1, 0.9]`。
  这些范围不是经实证估计的先验。
- 通用范围 `16 <= --n <= 10000`、`2 <= --horizon <= 27`；合成校准要求 horizon 至少 3。
  首次按示例使用 64 / 3，增加规模不会自动补齐真实目标或识别证据。

所有输出目录必须不存在。再次运行使用 `_002` 等新目录，不删除或覆盖旧结果。
`calibration/runs/` 已被 Git 忽略，运行中不要修改源码、切分支或改变 Git 跟踪集合。

### 6. 查看结果与候选参数

```powershell
$result = Get-Content -Raw calibration/runs/quickstart_calibration_001/result.json | ConvertFrom-Json
$result.status
$result.selected_before_validation
$result.search_score
$result.validation.score
Get-Content -Raw calibration/runs/quickstart_calibration_001/verification.json
```

| 输出 | 含义 |
|---|---|
| `result.json` | 搜索轨迹、候选、误差、验证及最终状态 |
| `selected_before_validation` | 验证前选中的候选，不等于已接受的最终参数 |
| `search_score` / `validation.score` | 搜索和独立验证分数，必须分别查看 |
| `verification.json` | 源码在运行期间是否保持不变，不是经济有效性认证 |
| `sources_before.json` | 本次运行的源码哈希清单 |
| `eval_0000` 至 `eval_0003` | 目标、两组搜索、候选验证的调用记录 |
| 各调用内 `request.json`、`config_seed_*.yaml` | 实际参数、种子、完整配置 |
| 各调用内 `path_seed_*.json`、`response.json` | 轨迹、路径有效性、投资与部门核算诊断 |
| 各调用内 `process.json` | 子进程退出码、标准输出和错误日志；报错优先查这里 |

**退出码 0 不等于校准验收 PASS：**

- **程序完成**：调用正常结束、输出齐全、路径有效。合成模式终端摘要中的 `valid: null`
  不表示路径失败：校准报告使用 `status`，路径有效性在嵌套结果中。
- **`validation_failed`**：选出候选并完成验证，但校准验收未通过，不是缺数据或崩溃。
  内置测试逐矩容差为 `1e-8`，独立验证种子的波动可以导致不通过。
- **`no_valid_candidate`**：没有有效候选，应检查日志和路径异常，不能报告校准成功。
- **`accepted_within_contract`**：仅当前合成契约内通过，不证明真实拟合、预测优势或经济有效性。

本入口不会生成或自动应用 `cfg/real_society_calibrated.yaml`。

### 已完成的最小交付检查

2026-09-10，在发布提交 `33334ec` 的干净本地克隆上运行上述 64 / 3 合成校准，
不复制 `calibration/data/`，使用本机既有依赖：4 个子进程退出码均为 0，4 条路径有效，
12 个期间的投资诊断与部门核算全部 PASS，受跟踪文件不变。
校准结果仍为 **`validation_failed`**，`real_data_fit=False`。
这是代码与数据依赖的运行检查，不是新正式实验或另一台电脑从零安装后的独立复现。

### 常见问题

| 问题 | 排查方式 |
|---|---|
| 克隆失败 / 403 / repository not found | 私有仓库需要协作者权限和正确 GitHub 登录 |
| 找不到 `py -3.11` | 安装 Python 3.11，或指定已有 Python 3.11 可执行文件创建环境 |
| `ModuleNotFoundError` / PyTorch 动态库错误 | 确认安装与执行使用同一 Python，先通过依赖导入检查；不是缺数据包 |
| 找不到 YAML 或家庭 CSV | 完整克隆，在仓库根目录运行，检查 `--repo .`；不要只下载校准目录 |
| 缺 `collected_v1` 或旧拓展目录 | 很可能使用了历史审计入口；本节最小流程不依赖这些目录 |
| `FileExistsError` | 改用未使用的输出目录，不覆盖旧实验 |
| `source changed` / `source drift` | 运行时不要编辑源码、切分支或改跟踪集合；保留记录后用新目录做开发检查 |
| `worker failed` / 超时 | 查看 `eval_*/process.json` 与路径异常；单次子进程超时 300 秒，先用小规模定位 |
| `validation_failed` | 阅读逐矩误差；不靠换种子、放宽阈值把流程测试包装成科学成功 |

价格稳定性及基础条件等研究执行器还需要前置数据与实验文件，不在此最小流程内。
真实数据目标入口仍关闭，不要绕过检查解锁。冻结开发执行器和旧结果原样保留，
不能覆盖旧目录重跑来重新解释既有结果。

## Next integration gates / 后续合并条件

- 统一 v1 / v2 数据接口与观测口径，解决微观映射和经济可比性问题。
- 将价格子空间限制、独立 MAE、投资诊断及部门核算连接成明确的统一验收接口。
- 处理弱识别与抽样不稳定性，再决定是否增加优化算法。
- 非校准模块的必要修改逐项讨论，不在此分支偷偷改变协作者经济机制。
- 完成真实数据拟合与独立验证之后，再讨论最终参数交付及合入 AgentEWM。
