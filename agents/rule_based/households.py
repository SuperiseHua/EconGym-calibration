import numpy as np
import torch
from functools import lru_cache
from pathlib import Path

class HouseholdRules:
    """可切换的家庭消费、劳动与旧年龄分组规则。"""

    @staticmethod
    def get_action(type, obs, action_dim, firm_n, country="China", rule="age_profile",
                   consumption_share=0.95, labor_rule="initial", initial_work=None,
                   labor_share=0.5):
        """切换入口：fixed_consumption 为自定义总消费规则；age_profile 保留旧抽样规则。"""
        if rule == "age_profile":
            # 旧规则同时抽样储蓄、劳动和风险投资；不使用下方固定比例参数。
            return HouseholdRules.age_profile(type, obs, action_dim, firm_n, country)
        if rule != "fixed_consumption":
            raise ValueError(f"Unknown household_rule: {rule}")
        labor_rules = {
            "initial": lambda: HouseholdRules.initial_labor(initial_work, len(obs)),
            "constant": lambda: HouseholdRules.constant_labor(labor_share, len(obs)),
            "scf": lambda: HouseholdRules.scf_labor(type, obs),
        }
        if labor_rule not in labor_rules:
            raise ValueError(f"Unknown household_labor_rule: {labor_rule}")
        return HouseholdRules.fixed_consumption(
            consumption_share, labor_rules[labor_rule](), action_dim, firm_n)

    @staticmethod
    def fixed_consumption(consumption_share, labor, action_dim, firm_n):
        """总消费预算＝比例×税后收入（含消费税）；0.95 表示消费 95%、储蓄动作 5%。

        实际成交与扣税由环境结算。此规则不配置风险资产；多企业时均分消费预算。
        """
        if not np.isfinite(consumption_share) or not 0 <= consumption_share <= 1.5:
            raise ValueError('household_consumption_share must be in [0, 1.5].')
        action = np.zeros((len(labor), action_dim), dtype=float)
        action[:, 0] = 1 - consumption_share
        action[:, 1] = labor
        if firm_n > 1:
            action[:, -firm_n-1] = np.random.rand(len(labor))
            action[:, -firm_n:] = 1 / firm_n
        return action

    @staticmethod
    def initial_labor(initial_work, n):
        """劳动规则 initial：保持每个家庭自身的 SCF 初始劳动比例，保留初始异质性。"""
        labor = np.asarray(initial_work, dtype=float).reshape(-1)
        if labor.size != n or not np.isfinite(labor).all() or np.any((labor < 0) | (labor > 1)):
            raise ValueError('initial labor requires one valid initial work share per household.')
        return labor

    @staticmethod
    def constant_labor(labor_share, n):
        """劳动规则 constant：所有家庭使用同一劳动比例；0.5 表示 h_max 的一半。"""
        if not np.isfinite(labor_share) or not 0 <= labor_share <= 1:
            raise ValueError('household_labor_share must be in [0, 1].')
        return np.full(n, labor_share)

    @staticmethod
    @lru_cache(maxsize=2)
    def _scf_labor_data(has_age):
        # 与原 SCF 最近邻相同的清洗、标准化及距离口径；只读取劳动标签，不创建网络。
        import pandas as pd
        path = Path(__file__).resolve().parents[1] / 'data/advanced_scfp2022_1110.csv'
        frame = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).dropna()
        columns = ['EDUC', 'ASSET'] + (['AGE'] if has_age else [])
        return (torch.tensor(np.ascontiguousarray(frame[columns].to_numpy()), dtype=torch.float32),
                torch.tensor(frame['LF'].to_numpy(), dtype=torch.float32))

    @staticmethod
    def scf_labor(type, obs):
        """劳动规则 scf：按教育、财富（OLG 加年龄）寻找 SCF 最近邻，直接取 LF。"""
        samples, labor = HouseholdRules._scf_labor_data('OLG' in type)
        obs = torch.as_tensor(obs).detach().to(device='cpu', dtype=torch.float32)
        mean, std = samples.mean(0), samples.std(0) + 1e-6
        query = (obs[:, -samples.shape[1]:] - mean) / std
        reference = (samples - mean) / std
        nearest = torch.norm(query[:, None, :] - reference[None, :, :], dim=2).argmin(dim=1)
        return labor[nearest].numpy()

    # -------------------------------
    # China: Saving Rate Parameters
    # -------------------------------
    @staticmethod
    def _china_saving_params(age):
        """
        Return (mu, sd) of saving rate for Chinese households by age of household head.
        Based on empirical studies (CFPS/UHS) showing a U-shaped profile.
        """
        if age < 35:
            return 0.30, 0.10
        elif age < 45:
            return 0.27, 0.10
        elif age < 55:
            return 0.28, 0.09
        elif age < 65:
            return 0.32, 0.08
        else:
            return 0.34, 0.08

    # ------------------------------------
    # China: Risky Investment Parameters
    # ------------------------------------
    @staticmethod
    def _china_risky_invest_params(age):
        """
        Return (mu, sd) of risky investment share for Chinese households.
        Younger households invest more aggressively.
        """
        if age < 35:
            return 0.50, 0.12
        elif age < 45:
            return 0.40, 0.12
        elif age < 55:
            return 0.35, 0.10
        elif age < 65:
            return 0.25, 0.08
        else:
            return 0.15, 0.08

    # -----------------------------
    # US: Saving Rate Parameters
    # -----------------------------
    @staticmethod
    def _us_saving_params(age):
        """
        Return (mu, sd) of saving rate for US households by age of household head.
        Based on BLS-CE 2023 data. Note: values are approximate and smoothed.
        """
        if age < 35:
            return 0.16, 0.10  # ~16.2%
        elif age < 45:
            return 0.16, 0.09  # ~16.1%
        elif age < 55:
            return 0.12, 0.09  # ~12.1%
        elif age < 65:
            return 0.12, 0.08  # ~11.8%
        else:
            return 0.00, 0.10  # ~ -9.6%, clipped to 0

    # ----------------------------------
    # US: Risky Investment Parameters
    # ----------------------------------
    @staticmethod
    def _us_risky_invest_params(age):
        """
        Return (mu, sd) of risky investment share for US households.
        Generally higher than China; younger households invest more.
        """
        if age < 35:
            return 0.70, 0.10
        elif age < 45:
            return 0.60, 0.10
        elif age < 55:
            return 0.50, 0.10
        elif age < 65:
            return 0.35, 0.08
        else:
            return 0.25, 0.08

    # ----------------------------------------------------------
    # Advance Consumption: eats into planned savings
    # ----------------------------------------------------------
    @staticmethod
    def _china_consumption_params(age):
        """
        Return (mu, sd) of 'advance consumption' ratio for Chinese households.
        This is α_adv: the fraction of planned savings eaten by advance consumption.
        """
        if age < 35:
            return 0.50, 0.15
        elif age < 55:
            return 0.40, 0.12
        else:
            return 0.20, 0.08

    @staticmethod
    def _us_consumption_params(age):
        """
        Return (mu, sd) of 'advance consumption' ratio for US households.
        This is α_adv: the fraction of planned savings eaten by advance consumption.
        """
        if age < 35:
            return 0.60, 0.15
        elif age < 55:
            return 0.50, 0.12
        else:
            return 0.30, 0.10

    # -------------------------------------------
    # Routing function to select param functions
    # -------------------------------------------
    @staticmethod
    def _get_param_fn(country: str, kind: str):
        """
        Select the appropriate age-to-(mu, sd) mapping function based on country and kind.
        Args:
            country (str): "China" or "US"
            kind (str): "saving" | "risky" | "adv_consume"
        """
        if kind == "saving":
            if country == "US":
                return HouseholdRules._us_saving_params
            elif country == "China":
                return HouseholdRules._china_saving_params
        elif kind == "risky":
            if country == "US":
                return HouseholdRules._us_risky_invest_params
            elif country == "China":
                return HouseholdRules._china_risky_invest_params
        elif kind == "adv_consume":
            if country == "US":
                return HouseholdRules._us_consumption_params
            elif country == "China":
                return HouseholdRules._china_consumption_params
        raise ValueError(f"Unknown country={country} or kind={kind}.")

    # -------------------------------------------------
    # Sampling saving / risky / advance consumption
    # -------------------------------------------------
    @staticmethod
    def get_proportion(age, n, country, action_kind):
        """
        Generate proportion vector for 'saving' | 'risky' | 'adv_consume'.
        - 'saving'       -> baseline saving rate s in [0,1]
        - 'risky'        -> risky investment share in [0,1]
        - 'adv_consume'  -> α_adv in [0,1], i.e., fraction of planned savings eaten by advance consumption
        Args:
            age (array or None): age array for each agent, or None for global average
            n (int): number of agents
            country (str): "China" or "US"
            action_kind (str): "saving" | "risky" | "adv_consume"
        Returns:
            np.ndarray: clipped values in [0, 1]
        """
        if age is not None:
            age = np.asarray(age).reshape(-1)
            param_fn = HouseholdRules._get_param_fn(country, kind=action_kind)
            mus_sds = np.array([param_fn(a) for a in age], dtype=float)  # shape (N, 2)
            mus, sds = mus_sds[:, 0], mus_sds[:, 1]
            sp = np.random.normal(loc=mus, scale=sds)
        else:
            # Global priors if age is not given (e.g., Ramsey-type agent)
            if action_kind == "saving":
                mu, sd = (0.40, 0.08) if country == "US" else (0.50, 0.10)
            elif action_kind == "risky":
                mu, sd = (0.50, 0.10) if country == "US" else (0.30, 0.10)
            elif action_kind == "adv_consume":
                mu, sd = (0.50, 0.12) if country == "US" else (0.5, 0.15)
            else:
                raise ValueError(f"Unknown action_kind={action_kind}")
            sp = np.random.normal(mu, sd, size=n)

        return np.clip(sp, 0.0, 1.0)

    # ------------------------------------------
    # Main household action sampling interface
    # ------------------------------------------
    @staticmethod
    def age_profile(type, obs, action_dim, firm_n, country="China"):
        """
        旧规则 age_profile：按年龄/国家先验抽样储蓄与风险投资，劳动比例另行随机抽样。
        These existing heuristic priors are not calibrated household policies.
        Action layout:
            - Column 0: saving proportion in [0, 1] (after adv_consume adjustment if enabled)
            - Column 1: labor supply proportion in [0, 1]
            - Column 2: risky investment proportion in [0, 1] (if "risk_invest" in type and action_dim>=3)
            - Column -firm_n-1: normalized selected firm index in (0,1) when firm_n>1
            - Last firm_n columns: consumption shares across firms (sum to 1) when firm_n>1

        Enable 'advance consumption' by including the token "adv_consume" in `type`.
        Example: type="OLG_risk_invest_adv_consume"
        """
        N = len(obs)
        action = np.random.randn(N, action_dim)

        if "OLG" in type:
            ages = obs[:, -1]  # age is assumed to be the last column
        else:
            ages = None

        # --- Baseline saving ---
        saving = HouseholdRules.get_proportion(ages, N, country, action_kind='saving')

        # --- Advance consumption adjustment (s' = s * (1 - alpha_adv)) ---
        if "adv_consume" in type:
            alpha_adv = HouseholdRules.get_proportion(ages, N, country, action_kind='adv_consume')
            saving = np.clip(saving * (1.0 - alpha_adv), 0.0, 1.0)

        action[:, 0] = saving.reshape(-1)

        # --- Labor supply (truncated normal) ---
        mean, std_dev, lower_bound, upper_bound = 0.5, 0.2, 0.0, 1.0
        action[:, 1] = np.clip(np.random.normal(loc=mean, scale=std_dev, size=N), lower_bound, upper_bound)

        # --- Risky investment (optional) ---
        if "risk_invest" in type and action_dim >= 3:
            risk = HouseholdRules.get_proportion(ages, N, country, action_kind='risky')
            action[:, 2] = risk.reshape(-1)

        # --- Firm choice & consumption shares across firms ---
        if firm_n > 1:
            # Extract wage rates and prices from observations
            wagerate = obs[:, 4:4 + firm_n]  # shape (N, firm_n)
            price = obs[:, 4 + firm_n: 4 + firm_n * 2]  # shape (N, firm_n)

            # Convert to torch for stable operations
            wage_t = torch.tensor(wagerate, dtype=torch.float32)
            wage_sum = wage_t.sum(dim=1, keepdim=True).clamp_min(1e-8)
            wage_probs = wage_t / wage_sum

            if torch.isnan(wage_probs).any() or torch.isinf(wage_probs).any():
                print("Warning: NaN or Inf in wagerate_probs")

            firm_index = torch.multinomial(wage_probs, 1)  # (N,1)
            action[:, -firm_n - 1] = firm_index.squeeze(1).numpy() / firm_n  # normalized index in (0,1)

            # Price-based choice probabilities (softmax on -price)
            price_t = torch.tensor(price, dtype=torch.float32)
            price_exp = torch.exp(-price_t)
            price_sum = price_exp.sum(dim=1, keepdim=True).clamp_min(1e-8)
            price_probs = price_exp / price_sum

            if torch.isnan(price_probs).any() or torch.isinf(price_probs).any():
                print("Warning: NaN or Inf in price_probs")

            action[:, -firm_n:] = price_probs.numpy()
            # Ensure exact row-wise normalization (numerical safety)
            row_sum = action[:, -firm_n:].sum(axis=1, keepdims=True)
            action[:, -firm_n:] = action[:, -firm_n:] / np.clip(row_sum, 1e-8, None)

        return action
