# 更新记录

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
