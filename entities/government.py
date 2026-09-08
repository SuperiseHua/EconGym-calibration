from entities.base import BaseEntity
import numpy as np
from gymnasium.spaces import Box
from agents.saez import SaezGovernment
import copy


class Government(BaseEntity):
    name = 'government'

    def __init__(self, entity_args):
        super().__init__()
        self.entity_args = entity_args
        self.__dict__.update(entity_args['params'])
        self.action_dim = entity_args[self.type]['action_dim']
        self.policy_action_len = copy.copy(self.action_dim)
        self.real_action_max = np.array(entity_args[self.type]['real_action_max'])
        self.real_action_min = np.array(entity_args[self.type]['real_action_min'])

    @staticmethod
    def compute_scale_factor(real_population, real_household_units, agents_n, household_type):
        """Scale SCF-backed agents by the represented family/PEU population."""
        if real_household_units <= 0:
            raise ValueError("The real SCF family-unit count must be positive.")
        return agents_n / real_household_units

    @staticmethod
    def compute_initial_gdp(real_gdp, real_population, real_household_units,
                            agents_n, household_type):
        """Scale aggregate GDP to the number of simulated SCF family/PEU agents."""
        return real_gdp * Government.compute_scale_factor(
            real_population, real_household_units, agents_n, household_type
        )

    def reset(self, **custom_cfg):
        if self.type == 'tax' and self.tax_type == 'saez':
            self.saez_gov = SaezGovernment()
        households_n = custom_cfg['households_n']
        firm_n = custom_cfg['firm_n']
        real_gdp = self.real_gdp
        real_debt_rate = self.real_debt_rate
        real_population = self.real_population
        real_household_units = custom_cfg['real_household_units']
        household_type = custom_cfg['household_type']

        self.action_space = Box(
            low=self.entity_args[self.type]["action_space"]["low"],
            high=self.entity_args[self.type]["action_space"]["high"],
            shape=(self.action_dim,), dtype=np.float32
        )

        initial_actions = self.entity_args[self.type]['initial_action']
        for name in ('tau', 'xi', 'tau_a', 'xi_a', 'Gt_prob'):
            setattr(self, name, self.entity_args.params[name])
        for name, value in initial_actions.items():
            setattr(self, name, value)

        if self.type == "central_bank":
            self.base_interest_rate = initial_actions["base_interest_rate"]
            self.reserve_ratio = initial_actions["reserve_ratio"]

        policy_action = np.array(list(initial_actions.values()))[:self.policy_action_len]
        allocation_dim = self.action_dim - self.policy_action_len
        allocation = np.full(allocation_dim, 1 / allocation_dim) if allocation_dim else np.array([])
        self.initial_action = np.concatenate([policy_action, allocation])

        # Static macro inputs and household microdata currently refer to the United States in 2022.
        # TODO(data-channel): replace them with metadata from an uploaded country-year data source.
        self.scale_factor = self.compute_scale_factor(
            real_population, real_household_units, households_n, household_type
        )
        self.GDP = real_gdp * self.scale_factor
        self.old_GDP = self.GDP
        self.growth_rate = 0.05  # Preserve the declared initial observation, never the last episode.
        self.per_household_gdp = self.GDP / households_n
        self.old_per_gdp = self.per_household_gdp
        self.initial_GDP = self.GDP
        self.real_GDP = self.GDP
        self.nominal_GDP = self.GDP
        self.Bt_next = real_debt_rate * self.GDP
        self.Bt = copy.copy(self.Bt_next)
        self.pension_fund = self.entity_args.get('initial_pension_fund', 1e-8)
        self.contribution_rate = initial_actions.get('contribution_rate', self.entity_args.params.contribution_rate)
        self.retire_age = initial_actions.get('retire_age', self.entity_args.params.retire_age)
        self.old_percent = 0
        self.dependency_ratio = 0
        self.estate_tax_revenue = 0.
        self.deceased_direct_tax_revenue = 0.
        self.deceased_consumption_tax_revenue = 0.
        self.bond_interest_expense = 0.0
        self.gov_spending_d = np.zeros((firm_n, 1))
        self.gov_spending = np.zeros((firm_n, 1))
        self._spending_step = -1

        # Initialize Gt_prob_j as an empty array or with a default value to avoid AttributeError
        self.Gt_prob_j = np.full((firm_n, 1), self.Gt_prob / firm_n)

    def get_action(self, actions, firm_n):

        policy_actions = actions[:self.policy_action_len]
        Gt_prob_ratios = np.ones(firm_n) / firm_n
        if self.type == "pension":
            # self.retire_age, self.contribution_rate, self.pension_growth_rate = policy_actions
            self.retire_age, self.contribution_rate = policy_actions
        elif self.type == "tax":
            if self.tax_type == "ai_agent":
                self.tau, self.xi, self.tau_a, self.xi_a, self.Gt_prob = policy_actions
                Gt_prob_ratios = actions[self.policy_action_len:]
        elif self.type == "central_bank":
            self.base_interest_rate, self.reserve_ratio = policy_actions
        else:
            raise ValueError("Invalid government type specified!")

        if firm_n != 1:
            if np.sum(Gt_prob_ratios) == 0:
                # No allocation preference means equal shares, not an erased budget.
                self.Gt_prob_j = np.full((firm_n, 1), self.Gt_prob / firm_n)
            else:
                self.Gt_prob_j = (Gt_prob_ratios / np.sum(Gt_prob_ratios))[:, np.newaxis] * self.Gt_prob
        else:
            self.Gt_prob_j = self.Gt_prob

    def step(self, society):
        self.old_per_gdp = copy.copy(self.per_household_gdp)
        self.old_GDP = copy.copy(self.GDP)
        self.Bt = copy.copy(self.Bt_next)

        self.tax_step(society)
        if self.type == "pension" and ("OLG" in society.households.type):
            self.pension_step(society)

    def plan_spending(self, society):
        """Set desired and goods-constrained government purchases for this period."""
        nominal_output = float(np.sum(society.market.price * society.market.Yt_j))
        self.gov_spending_d = self.Gt_prob_j * nominal_output / society.market.price
        if hasattr(society.households, 'BUI'):
            self.gov_spending_d += society.households.households_n * society.households.BUI
        self.gov_spending = np.minimum(self.gov_spending_d, society.market.goods_supply)
        self._spending_step = society.step_cnt

    def tax_step(self, society):
        """Calculate government metrics such as taxes, investment, and GDP."""
        households = society.households
        if self._spending_step != society.step_cnt:
            self.plan_spending(society)
        self.tax_array = households.income_tax + households.asset_tax + households.consumption_tax
        self.estate_tax_revenue = households.estate_tax
        self.deceased_direct_tax_revenue = households.deceased_direct_tax
        self.deceased_consumption_tax_revenue = households.deceased_consumption_tax
        self.bond_interest_expense = society.bank.government_interest_payment
        self.Bt_next = (self.Bt + self.bond_interest_expense + np.sum(
            self.gov_spending * society.market.price)
                        - np.sum(self.tax_array) - self.estate_tax_revenue
                        - self.deceased_direct_tax_revenue - self.deceased_consumption_tax_revenue)

        self.real_GDP = np.sum(society.market.Yt_j)
        self.nominal_GDP = np.sum(society.market.price * society.market.Yt_j)
        self.GDP = self.real_GDP  # Backward-compatible name used by growth calculations.
        self.per_household_gdp = self.GDP / max(households.households_n, 1e-8)

    def pension_step(self, society):
        """Update the pension fund for all households."""
        self.current_net_households_pension = society.households.pension.sum()
        self.pension_fund = (1 + self.pension_growth_rate) * self.pension_fund - self.current_net_households_pension

    def calculate_pension(self, households):
        """Calculate the pension benefits for a given household."""
        pension = -households.income * ~households.is_old * self.contribution_rate  # <0 : pension contribute from young individuals

        income_mean = np.nanmean(households.income)

        if households.is_old.any():
            income_old_mean = np.nanmean(households.income[households.is_old])
            avg_wage = (income_old_mean + income_mean) / 2

            if hasattr(self, 'pension_rate') and self.pension_rate is not None:
                basic_pension = avg_wage * households.working_years[households.is_old] * 0.01 * self.pension_rate
            else:
                basic_pension = avg_wage * households.working_years[households.is_old] * 0.01
            personal_pension = households.accumulated_pension_account[households.is_old] / self.get_annuity_factor(
                self.retire_age)

            pension[households.is_old] = basic_pension + personal_pension  # >0 : payoff for old individuals

        return pension

    def get_annuity_factor(self, age):
        """Return the annuity factor for a given age."""
        annuity_factors = {
            40: 233, 41: 230, 42: 226, 43: 223, 44: 220, 45: 216, 46: 212,
            47: 208, 48: 204, 49: 199, 50: 195, 51: 190, 52: 185, 53: 180,
            54: 175, 55: 170, 56: 164, 57: 158, 58: 152, 59: 145, 60: 139,
            61: 132, 62: 125, 63: 117, 64: 109, 65: 101, 66: 93, 67: 84,
            68: 75, 69: 65, 70: 56
        }
        return annuity_factors.get(age, 56)

    def tax_function(self, income, asset):
        """Compute taxes based on government policies."""

        def tax_formula(x, tau, xi):
            x = np.maximum(x, 1e-8)
            if np.isclose(xi, 1.0):
                return x - (1 - tau) * np.log(x)
            else:
                return x - ((1 - tau) / (1 - xi)) * np.power(x, 1 - xi)

        income_tax = tax_formula(income, self.tau, self.xi)
        asset_tax = tax_formula(asset, self.tau_a, self.xi_a)

        return income_tax, asset_tax

    @staticmethod
    def reference_federal_income_tax(income):
        """Shared existing model schedule; not a complete US household tax model."""
        income = np.asarray(income, dtype=float)
        personal_allowance = 12950
        tax_credit = 559.98
        marginal_rates = np.array([10, 12, 22, 24, 32, 35, 37]) / 100
        thresholds = np.array([0, 10275, 41775, 89075, 170050, 215950, 539900, np.inf])
        taxable_income = np.maximum(0, income - personal_allowance)
        taxes_paid = np.zeros_like(income, dtype=float)
        for i in range(1, len(thresholds)):
            income_in_bracket = np.minimum(taxable_income, thresholds[i]) - thresholds[i - 1]
            taxes_paid += np.maximum(0, income_in_bracket) * marginal_rates[i - 1]
        return np.maximum(0, taxes_paid - tax_credit)

    def calculate_progressive_taxes(self, income, asset):
        """Calculate income and asset taxes based on US federal tax brackets."""
        income_tax = self.reference_federal_income_tax(income)

        _, asset_tax = self.tax_function(income, asset)
        return income_tax, asset_tax

    def compute_tax(self, income, asset):
        """Compute income and asset taxes based on the tax policy."""
        if self.tax_type == "us_federal":
            income_tax, asset_tax = self.calculate_progressive_taxes(income, asset)
        elif self.tax_type == "saez":
            self.saez_gov.saez_step()
            self.saez_gov.update_saez_buffer(income)
            income_tax = np.array(self.saez_gov.tax_due(income)).reshape(-1, 1)
            _, asset_tax = self.tax_function(income, asset)
        elif self.tax_type == "ai_agent":
            income_tax, asset_tax = self.tax_function(income, asset)
        else:
            print("Free market policy: no taxes applied.")
            income_tax = np.zeros_like(income)
            asset_tax = np.zeros_like(asset)
        return income_tax, asset_tax

    def softsign(self, x):
        return x / (1.0 + np.abs(x))

    def safe_sigmoid(self, x, threshold=80.0):
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, -threshold, threshold)
        return 1 / (1 + np.exp(-x))

    def get_reward(self, society, gov_goal=None):
        """
        Compute the government's reward based on its goal.

        Notes
        -----
        - For self.type in {"pension", "tax"}:
          They can share the same reward function definitions below (gov_goal routing).
        - For self.type == "central_bank":
          Delegates to `get_reward_central(inflation_rate, growth_rate)`.
        """
        SCALE = dict(
            gdp_growth=0.05,
            gini_scale=0.167,
        )

        self.growth_rate = (self.GDP + 1e-8) / (self.old_GDP + 1e-8) - 1

        # Assign self.gov_task to gov_goal if no valid value is provided for gov_goal
        gov_goal = gov_goal or self.gov_task
        # Central bank uses its own reward
        if self.type == "central_bank":
            reward = self.get_reward_central(society.inflation_rate, self.growth_rate)  # \in (0,1)
            return np.array([reward])

        if gov_goal == "gdp":
            log_gdp_growth = np.log(self.GDP + 1e-8) - np.log(self.old_GDP + 1e-8)
            # reward = self.softsign(log_gdp_growth / SCALE["gdp_growth"])
            reward = self.safe_sigmoid(log_gdp_growth / SCALE["gdp_growth"])
            return np.array([reward])  # \in (0,1)

        elif gov_goal == "gini":
            # Wealth Gini improvement
            before_tax_wealth_gini = society.gini_coef(society.households.at)
            after_tax_wealth_gini = society.gini_coef(society.households.post_asset)
            impr_w = before_tax_wealth_gini - after_tax_wealth_gini

            # Income Gini improvement
            before_tax_income_gini = society.gini_coef(society.households.income)
            after_tax_income_gini = society.gini_coef(society.households.post_income)
            impr_i = before_tax_income_gini - after_tax_income_gini
            # return (delta_income_gini + delta_wealth_gini) * 100
            reward = self.safe_sigmoid((impr_w + impr_i) / (2 * SCALE["gini_scale"]))  # \in (0,1)
            return np.array([reward])

        elif gov_goal == "social_welfare":
            # Sum of household utilities (social welfare)
            social_welfare = np.sum(society.households.get_reward())
            return np.array([social_welfare])

        elif gov_goal == "mean_welfare":
            # When population changes (OLG), mean welfare ≠ social welfare.
            # Mean welfare better reflects policy effects without population-size confounds.
            mean_welfare = np.mean(society.households.get_reward())
            return np.array([mean_welfare])

        elif gov_goal == "gdp_gini":
            # mixed goal
            gdp_rew = self.get_reward(society, gov_goal='gdp')
            gini_rew = self.get_reward(society, gov_goal='gini')
            return (gdp_rew + gini_rew) / 2  # \in (0,1)

        elif gov_goal == "pension_gap":
            # pension_surplus = sum(-self.calculate_pension(society.households))
            return self.get_pension_reward(self.pension_fund, society.households.households_n)

        else:
            raise ValueError("Invalid government goal specified.")

    def get_pension_reward(self, pension_surplus, households_n, k=2, center=1e6):
        """
        Compute the reward for pension fund sustainability.

        This function:
        - Rewards increasing pension surplus.
        - Applies a penalty for extreme surpluses or deficits.
        - Uses a logarithmic transformation to handle large surpluses and ensures diminishing returns.
        - Normalizes the reward using the tanh function to keep it within a bounded range.

        Parameters:
        -----------
        pension_surplus : float
            The surplus or deficit of the pension fund.

        k : float, steepness of the curve
        center : float, center of rapid growth region (default=1e6)
        """

        # Log transformation for diminishing returns and penalty for extreme surpluses
        per_household_pension_surplus = pension_surplus / max(households_n, 1e-8)
        pension_surplus = max(0, per_household_pension_surplus)
        pension_growth = np.log(1 + pension_surplus)
        log_center = np.log(center)

        # Normalize the reward using tanh
        normalized_reward = self.safe_sigmoid(k * (pension_growth - log_center))

        return np.array([normalized_reward])

    def get_reward_central(self,
                           actual_inflation,
                           actual_growth,
                           target_inflation=0.02,
                           target_growth=0.05,
                           weight_inflation=120.0,  # penalty weight for inflation deviations
                           weight_growth_low=80.0,  # penalty weight when growth < target
                           weight_growth_high=20.0,  # penalty weight when growth > target
                           smoothness=0.01  # smoothing parameter for differentiability
                           ):
        """
        Central bank reward function (smooth & asymmetric).

        - Penalizes inflation deviations symmetrically (quadratic).
        - Penalizes growth below target more heavily than growth above target (asymmetric).
        - Uses softplus for smooth approximation of max(0, x), keeping the function differentiable.
        - Final reward is bounded in (0, 1] via exponential decay.

        Parameters
        ----------
        actual_inflation : float
            Observed inflation rate.
        actual_growth : float
            Observed economic growth rate.
        target_inflation : float, default=0.02
            Central bank's target inflation rate (e.g., 2%).
        target_growth : float, default=0.05
            Desired sustainable growth rate (e.g., 5%).
        weight_inflation : float, default=120.0
            Weight applied to inflation deviation penalty.
        weight_growth_low : float, default=80.0
            Weight applied when growth falls below target.
        weight_growth_high : float, default=20.0
            Weight applied when growth exceeds target.
        smoothness : float, default=0.01
            Controls how closely softplus approximates ReLU. Smaller = sharper.

        Returns
        -------
        reward : float
            A smooth, bounded reward in (0, 1]. Higher values indicate conditions closer
            to central bank objectives.
        """

        # Stable softplus: softplus(x) = log(1 + exp(x))
        def softplus(x):
            return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

        # Inflation deviation (symmetric quadratic penalty)
        self.inflation_gap = actual_inflation - target_inflation
        loss_inflation = weight_inflation * (self.inflation_gap ** 2)

        # Growth deviation
        self.growth_gap = actual_growth - target_growth

        # Smooth decomposition into below-target and above-target parts
        below_target = softplus(-self.growth_gap / smoothness) * smoothness  # >0 if growth < target
        above_target = softplus(self.growth_gap / smoothness) * smoothness  # >0 if growth > target

        # Asymmetric growth penalties
        loss_growth = weight_growth_low * (below_target ** 2) + weight_growth_high * (above_target ** 2)

        # Total loss and reward
        total_loss = loss_inflation + loss_growth
        reward = np.exp(-total_loss)

        return reward

    def is_terminal(self):
        '''
        If `self.pension_fund < 0` (pension pool exhausted),  the environment should trigger a terminal state.
        '''
        if self.pension_fund < 0:
            return True
        return False
