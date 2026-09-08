from entities.base import BaseEntity
import copy
import numpy as np
from gymnasium.spaces import Box


class Market(BaseEntity):
    name = "market"
    
    def __init__(self, entity_args):
        super().__init__()
        self.entity_args = entity_args
        self.__dict__.update(entity_args['params'])
        self.firm_n = entity_args[self.type]['firm_n']
        self.action_dim = entity_args[self.type]['action_dim']
        self.markup = float(entity_args[self.type].get('markup', 1.0))
        if (self.type == "perfect" or self.type == "monopoly") and self.firm_n != 1:
            raise ValueError("Invalid market type specified or invalid firm number specified.")
        
        self.action_space = Box(
            low=-0.2, high=0.2, shape=(self.firm_n, self.action_dim), dtype=np.float32
        )
    
    @staticmethod
    def compute_inflation_rate(prices, quantities, old_prices, old_quantities=None, old_index=100.0):
        """Annual PCE-style chain Fisher inflation from consumer prices and consumption."""
        p, q = np.asarray(prices).ravel(), np.asarray(quantities).ravel()
        p_old = np.asarray(old_prices).ravel()
        q_old = q if old_quantities is None else np.asarray(old_quantities).ravel()
        if not (p.size == q.size == p_old.size == q_old.size):
            raise ValueError("Prices and quantities must have the same length.")
        if np.any(p <= 0) or np.any(p_old <= 0) or np.any(q < 0) or np.any(q_old < 0):
            raise ValueError("Prices must be positive and quantities non-negative.")
        if p.size == 1:
            fisher = p.item() / p_old.item()
            return float(fisher - 1), float(old_index * fisher)
        if q.sum() == 0 or q_old.sum() == 0:
            raise ValueError("Fisher inflation requires positive consumption.")

        laspeyres = np.dot(p, q_old) / np.dot(p_old, q_old)
        paasche = np.dot(p, q) / np.dot(p_old, q)
        fisher = np.sqrt(laspeyres * paasche)
        return float(fisher - 1), float(old_index * fisher)

    def update_price(self, planned_demand, supply, adjustment_speed=1.0):
        """Set next year's price by P[t+1] = P[t] * (D[t] / S[t]) ** lambda."""
        # lambda is the price-adjustment speed; lambda=1 means full adjustment.
        demand = np.asarray(planned_demand, dtype=float).reshape(self.price.shape)
        supply = np.asarray(supply, dtype=float).reshape(self.price.shape)
        if (not np.all(np.isfinite(demand)) or not np.all(np.isfinite(supply))
                or np.any(demand < 0) or np.any(supply <= 0)):
            raise ValueError(
                "Price adjustment requires finite non-negative demand and positive supply; "
                f"demand={demand.tolist()}, supply={supply.tolist()}."
            )
        self.price = self.price * np.power((demand + 1e-8) / (supply + 1e-8), adjustment_speed)
    
    def update_firm_productivity(self):
        """Apply a non-negative productivity-growth shock."""
        log_next_z = np.log(self.Zt) + self.sigma_z * np.random.rand(*self.Zt.shape)
        self.Zt = np.exp(log_next_z)

    @staticmethod
    def calibrate_initial_productivity(GDP, Kt, Lt, alpha):
        """Infer Z_0 so Cobb-Douglas output matches the observed base-year GDP."""
        base_output = np.sum(
            np.asarray(Kt, dtype=float) ** alpha
            * np.asarray(Lt, dtype=float) ** (1 - alpha)
        )
        if GDP <= 0 or base_output <= 0:
            raise ValueError("Initial productivity calibration requires positive GDP and inputs.")
        return float(GDP / base_output)

    @staticmethod
    def calibrate_initial_wage(GDP, Lt, alpha):
        """Return the Cobb-Douglas labor income per effective work hour."""
        total_labor = float(np.sum(Lt))
        if GDP <= 0 or total_labor <= 0 or not 0 < alpha < 1:
            raise ValueError("Initial wage calibration requires positive GDP/labor and 0 < alpha < 1.")
        return float((1 - alpha) * GDP / total_labor)
    
    def reset(self, **custom_cfg):
        GDP = custom_cfg['GDP']
        scale_factor = custom_cfg['scale_factor']
        households_asset = custom_cfg['households_at']
        real_debt_rate = custom_cfg['real_debt_rate']
        real_capital_rate = 18.3 * 0.01
        real_total_hours = 265888.875e6  # total hours worked, large L

        self.initial_GDP = float(GDP)
        self.Lt = real_total_hours * scale_factor
        # self.Kt = real_capital_rate * GDP / self.firm_n * np.ones((self.firm_n, 1))
        self.Kt = (np.sum(households_asset) - GDP * real_debt_rate )/ self.firm_n * np.ones((self.firm_n, 1))
        capital_ratio = getattr(self, 'initial_capital_output_ratio', None)
        if capital_ratio is not None:
            self.Kt[:] = float(capital_ratio) * GDP / self.firm_n
        if not np.all(np.isfinite(self.Kt)) or np.any(self.Kt <= 0):
            raise ValueError('Initial physical capital must be finite and positive.')
        self.Kt_next = copy.copy(self.Kt)
        self.initial_labor_j = np.full((self.firm_n, 1), self.Lt / self.firm_n)
        self.Z = self.calibrate_initial_productivity(
            self.initial_GDP, self.Kt, self.initial_labor_j, self.alpha
        )
        self.Zt = np.full((self.firm_n, 1), self.Z)
        self.Yt_j = self.production_output(self.Kt, self.initial_labor_j)
        self.goods_supply = np.zeros((self.firm_n, 1))
        self.planned_demand = np.zeros((self.firm_n, 1))
        self.price = np.ones((self.firm_n, 1))
        self.initial_WageRate = self.calibrate_initial_wage(
            self.initial_GDP, self.initial_labor_j, self.alpha
        )
        self.WageRate = np.full((self.firm_n, 1), self.initial_WageRate)
        self.capital_marginal_revenue_product = (
            self.price * self.Zt * self.alpha
            * np.power(self.Kt / (self.initial_labor_j + 1e-8), self.alpha - 1)
        )
    
    def get_action(self, actions):
        if actions is not None:
            self.price = actions[:, 0][:, np.newaxis]
            self.WageRate = actions[:, 1][:, np.newaxis]

    
    def step(self, society):
        """Calculate firm's labor demand and production output."""
        self.Kt = np.clip(copy.copy(self.Kt_next), 1e-8, None)
        # Compute firm's labor demand
        self.firm_labor_j = (society.households.h_ij_ratio * society.households.ht * society.households.e).sum(axis=0)[:, np.newaxis]
        self.Lt = np.sum(self.firm_labor_j)
        if society.step_cnt > 0:
            self.update_firm_productivity()
        self.Yt_j = self.production_output(self.Kt, self.firm_labor_j)
        self.MarketClear_WageRate = self.price * self.Zt * (1 - self.alpha) * np.power((self.Kt) / (self.firm_labor_j + 1e-8), self.alpha)
        self.capital_marginal_revenue_product = (
            self.price * self.Zt * self.alpha
            * np.power(self.Kt / (self.firm_labor_j + 1e-8), self.alpha - 1)
        )

        if self.type == "perfect":
            self.WageRate = copy.copy(self.MarketClear_WageRate)
            
    def production_output(self, Kt, Lt):
        """Compute the production output."""
        Kt = np.clip(Kt, a_min=0, a_max=None)
        Lt = np.clip(Lt, a_min=0, a_max=None)
        
        Y = self.Zt * (Kt ** self.alpha) * (Lt ** (1 - self.alpha))
        return Y
    
    def get_reward(self, society):
        """Calculate the firm's profit."""
        if self.type == "perfect":
            return np.array([0.])
        else:
            profit = (self.sales - self.payroll - self.interest_paid
                      if getattr(society.bank, 'firm_accounting', False)
                      else self.price * society.real_deals - self.WageRate * self.firm_labor_j - society.bank.lending_rate * self.Kt)
            reward = self.scaled_reward(profit)
            if isinstance(reward, np.ndarray):
                return reward
            else:
                return np.array([reward])
            

    def scaled_reward(self, x, eps=1e-8, k=0.15):  # \in (0,1)
        x = np.asarray(x, dtype=np.float64)
        log_scaled = np.sign(x) * np.log1p(np.abs(x) + eps)
        
        if np.any(np.abs(-k * log_scaled) > 50):
            print(f"[Warning Firm reward] Large input to exp detected: max |x| = {np.max(np.abs(-k * log_scaled)):.2f}")
        return 1 / (1 + np.exp(-k * log_scaled))
    
    def is_terminal(self):
        if np.sum(self.Kt_next) < 0:
            return True
        else:
            return False



