# Publication scope / 发布范围与复现边界

Date: 2026-09-10. Base: `ddc0a418965db3982fd594c678329309c241aeb1`.

This is a curated additive integration snapshot, not a replacement for the full local research archive.

## Included

- Current calibration Python sources, configuration templates and static tests.
- Current issue report, measurement contracts, data descriptions and development reports.
- Four byte-preserved aggregate `summary.json` files under `evidence/`.
- The byte-preserved price-stability contract, for interpretation of the latest result.
- A new publication README and data-independent static-test entry point.

`PUBLICATION_MANIFEST.json` lists the source-relative path and SHA-256 of every copied file.
Source Python files are copied without algorithm changes. New publication-only files are listed separately.
Local originals and frozen experimental directories are neither edited nor removed.
The calibration-specific `.gitattributes` disables line-ending conversion for this snapshot so copied artifact hashes remain meaningful.

## Excluded

- SCF / CEX / PSID / ATUS microdata, raw macro series, Excel workbooks, archives and derived data bundles.
- Per-path trajectories, full run trees, caches, local Python environments, preserved historical code directories and local backup trees.
- The received collaborator taskbook itself; the adaptation and alignment reports describe implementation scope.
- Credentials, machine environment configuration, and any final calibrated configuration (none has been accepted).

原始数据的再分发条件尚未逐项审阅。本次不新增上传这些数据；上游已经跟踪的文件保持原样，
这不构成对上游数据再分发许可的重新审计。数据存在于本地不等于可在 GitHub 上传，也不等于已用于训练。

## Evidence map

| Original local artifact | Published aggregate |
|---|---|
| `runs/runtime_effect_v1/summary.json` | [Runtime effect](evidence/runtime_effect_v1/summary.json) |
| `runs/mae_investment_v2/summary.json` | [MAE and investment](evidence/mae_investment_v2/summary.json) |
| `runs/foundation_noise_v1/summary.json` | [Foundation and noise](evidence/foundation_noise_v1/summary.json) |
| `runs/price_stability_v1/summary.json` | [Price stability](evidence/price_stability_v1/summary.json) |
| `runs/price_stability_v1/contract.json` | [Frozen interpretation contract](evidence/price_stability_v1/contract.json) |

本地报告及汇总中的路径、协议哈希和文件引用保留为历史来源线索；引用对象可能没有随分支发布。
不能仅用这五个 JSON 重新计算全部实验统计或验证全部执行轨迹。
旧来源快照形成时 `calibration/` 尚未纳入 Git；发布后的 tracked-file 集合会增加，
因此不能把新 checkout 的源码清单与旧冻结清单不同误判为原实验被修改。
更不能在新 checkout 上重建冻结文件后宣称同一次已冻结实验的复现成功。

## What this release does not claim

- No real-data calibration success or statistically significant prediction advantage.
- No joint identification of all three transmission parameters.
- No support for every economic mechanism, agent policy or interface in AgentEWM.
- No complete taskbook acceptance, fresh-environment simulator replication, or regression clearance for all upstream tests.

本次发布仅重跑无模拟器的静态测试。此前整合环境的上游 23 项测试中有两项因货币政策模板文件缺失报错；
没有通过伪造模板消除错误，也没有在此发布步骤修改协作者源码。
