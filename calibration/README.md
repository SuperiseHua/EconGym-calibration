# AgentEWM calibration extension — integration snapshot

> **Development integration, not a completed real-data calibration release.**
> 本分支用于协作审阅与接口整合；跑通、合成目标拟合、参数识别和现实预测是不同层级的证据。

## Scope / 本次提交范围

Based on `AgentEWM` commit `ddc0a418965db3982fd594c678329309c241aeb1`.
Only `calibration/` is added. Upstream economic mechanisms, configuration files,
`main.py`, `runner.py`, root `README.md`, and `update.md` are unchanged.

校准算法和经济引擎分开：通过运行时配置与测量适配器调用协作者代码，不覆盖其源码。
这并不代表运行条件不变：适配器明确限定 Ramsey、单商品、单企业、无风险资产配置、固定规则消费与劳动，并冻结部分随机过程。
目前不是适用于所有 AgentEWM 场景的通用插件，也尚未连接网页或主应用入口。

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

## Static tests / 不运行模拟器的检查

Use a Python environment with `numpy` and `omegaconf` installed. From the repository root:

```powershell
python -B calibration/run_unit_tests.py
```

该入口运行 64 项接口 / 测量 / 核算测试和 13 项稳定性测试；不读取原始调查包、
不执行模拟器、不重跑冻结 seeds，也不把测试通过当作科学结论。
原 `check_interfaces.py` 还依赖本地数据包及旧拓展目录，不是干净 checkout 的通用入口。

在安装完整上游依赖后，可执行仅生成准备记录的命令：

```powershell
python -B calibration/launch.py --repo . prepare --n 1000 --horizon 27 --output calibration/runs/my_prepare
```

输出目录必须不存在。无数据包时应报告缺失 / 未就绪，而不是凭空通过验收。
模拟实验另需上游运行数据、依赖及新输出目录；本次发布没有进行跨机器完整模拟复现。
冻结开发执行器原样保留，不应通过覆盖旧目录重跑来重新解释旧结果。

## Next integration gates / 后续合并条件

- 统一 v1 / v2 数据接口与观测口径，解决微观映射和经济可比性问题。
- 将价格子空间限制、独立 MAE、投资诊断及部门核算连接成明确的统一验收接口。
- 处理弱识别与抽样不稳定性，再决定是否增加优化算法。
- 非校准模块的必要修改逐项讨论，不在此分支偷偷改变协作者经济机制。
- 完成真实数据拟合与独立验证之后，再讨论最终参数交付及合入 AgentEWM。
