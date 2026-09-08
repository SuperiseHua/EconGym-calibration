# 家庭规则切换

规则实现集中在 `households.py`，由 `rules_core.py` 分发。自定义固定消费属于 `rule_based`，
不需要 BC 网络、训练或模型权重；环境只执行策略动作并完成预算、消费税和商品配给结算。

## 当前校准基准

```yaml
Trainer:
  house_alg: rule_based
  household_rule: fixed_consumption
  household_consumption_share: 0.95  # 含税消费预算＝税后收入的 95%；不是花掉 95% 的财富
  household_labor_rule: initial     # 保留各家庭初始工时
  household_labor_share: 0.5        # 仅 constant 使用
  household_country: US            # 仅 age_profile 使用
```

想把消费比例改为 90%，只需将 `household_consumption_share` 改为 `0.90`。
该参数支持 `[0, 1.5]`，对应现有储蓄动作范围 `[−0.5, 1]`；超过 100% 表示消费超出当期收入，
仍沿用已有资产结算与借款下限规则，不保证家庭能够长期负担。

## 可选方法

| 配置 | 函数 | 含义 |
|---|---|---|
| `household_rule: fixed_consumption` | `fixed_consumption()` | 自定义总消费比例；剩余收入储蓄，不配置风险资产 |
| `household_labor_rule: initial` | `initial_labor()` | 使用每个家庭自己的初始劳动比例；适合固定人口基准 |
| `household_labor_rule: constant` | `constant_labor()` | 全体家庭使用 `household_labor_share`，工时为该比例乘 `h_max` |
| `household_labor_rule: scf` | `scf_labor()` | 按标准化教育、财富（OLG 加年龄）匹配 SCF 最近邻，取其 LF；不创建 BC 网络 |
| `household_rule: age_profile` | `age_profile()` | 保留原有年龄/国家先验抽样：储蓄、劳动和风险投资由该完整规则决定 |

`household_labor_rule`、`household_labor_share`、`household_consumption_share` 仅供 `fixed_consumption` 使用。
`age_profile` 使用原有 `household_country`（US/China）及抽样函数；其历史数值是启发式先验，本次没有重新验证或校准。
新增规则时在 `HouseholdRules.get_action()` 添加分发，并将行为实现写成独立函数即可。

## 两个 real_society 场景

- `real_society_calibration.yaml`：95% 消费＋`initial` 劳动。
- `real_society.yaml`：95% 消费＋`scf` 劳动，保留此前随家庭状态变化的 SCF 最近邻劳动选择。

两个场景均直接评估 27 期（`Trainer.test: true`、`Environment.env_core.episode_length: 27`）。
在项目目录执行 `python main.py --problem_scene real_society`；当前校准基准则选择
`--problem_scene real_society_calibration`。`main.py` 不带参数仍默认选择 `inflation_control`。
`Trainer.epoch_length` 只控制训练采样长度，不能替代环境的 `episode_length`。

SCF 最近邻劳动规则尚未通过长期稳定性验证：原 300 期设置在一次复现中于第 192 期变为全员不工作。
供给为零时环境完成当期结算、保留真实零 GDP，以 `zero_goods_supply` 标记失败并结束该轨迹；
当期价格保留，不生成下一期价格。该状态不能作为正常完成或长期稳定的证据。

校准基准仍在家庭实体参数中设置 `freeze_labor_efficiency: true`，并设置市场 `sigma_z: 0`。
前者只冻结效率状态演化，不覆盖劳动动作；后者关闭技术冲击。初始工时本身由规则函数返回。
原 `fixed_saving_share` / `fixed_labor` 环境动作覆盖已移除。

要使用真正的 BC，再将 `house_alg` 改为 `bc`：`bc_test: false` 使用 SCF 最近邻动作，
`bc_test: true` 使用已有网络推理。此时上述 rule-based 策略参数不会覆盖 BC 输出。
