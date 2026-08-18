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
        self.Zt_init = self.Z * (1 + np.random.rand(self.firm_n, 1))
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
        if np.any(demand < 0) or np.any(supply <= 0):
            raise ValueError("Price adjustment requires non-negative demand and positive supply.")
        # The floor is only a numerical safeguard when planned demand is zero.
        self.price = self.price * np.power(np.maximum(demand, 1e-8) / supply, adjustment_speed)
    
    def update_firm_productivity(self):
        """Update the production quality (technology shock)."""
        log_next_z = np.log(self.Zt) + self.sigma_z * np.random.rand(*self.Zt.shape)
        self.Zt = np.exp(log_next_z)
    
    def reset(self, **custom_cfg):
        households_n = custom_cfg['households_n']
        GDP = custom_cfg['GDP']
        households_asset = custom_cfg['households_at']
        real_debt_rate = custom_cfg['real_debt_rate']
        real_capital_rate = 18.3 * 0.01
        real_total_hours = 265888.875e6  # total hours worked, large L
        real_population = 333428e3

        self.Zt = copy.copy(self.Zt_init)
        self.Lt = (real_total_hours / real_population) * households_n
        # self.Kt = real_capital_rate * GDP / self.firm_n * np.ones((self.firm_n, 1))
        self.Kt = (np.sum(households_asset) - GDP * real_debt_rate )/ self.firm_n * np.ones((self.firm_n, 1))
        self.Kt_next = copy.copy(self.Kt)
        self.price = np.ones((self.firm_n, 1))
        self.WageRate = self.price * self.Zt * (1 - self.alpha) * np.power(self.Kt / self.Lt, self.alpha)
    
    def get_action(self, actions):
        if actions is not None:
            self.price = actions[:, 0][:, np.newaxis]
            self.WageRate = actions[:, 1][:, np.newaxis]

    
    def step(self, society):
        """Calculate firm's labor demand and production output."""
        self.Kt = np.clip(copy.copy(self.Kt_next), 1e-8, None)
        self.update_firm_productivity()
        # Compute firm's labor demand
        self.firm_labor_j = (society.households.h_ij_ratio * society.households.ht * society.households.e).sum(axis=0)[:, np.newaxis]
        self.Lt = np.sum(self.firm_labor_j)
        self.Yt_j = self.production_output(self.Kt, self.firm_labor_j)
        self.MarketClear_WageRate = self.price * self.Zt * (1 - self.alpha) * np.power((self.Kt) / (self.firm_labor_j + 1e-8), self.alpha)
        self.MarketClear_InterestRate = self.price * self.Zt * self.alpha * np.power((self.Kt) / (self.firm_labor_j + 1e-8), self.alpha-1)

        if self.type == "perfect":
            self.WageRate = copy.copy(self.MarketClear_WageRate)
            
        if society.bank.type == "non_profit":
            if self.type == "perfect":
                society.bank.lending_rate = np.nanmean(self.MarketClear_InterestRate)
                society.bank.deposit_rate = np.nanmean(self.MarketClear_InterestRate)
            else:
                society.bank.lending_rate = society.bank.base_interest_rate
                society.bank.deposit_rate = society.bank.base_interest_rate
        
        
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
            profit = self.price * society.real_deals - self.WageRate * self.firm_labor_j - society.bank.lending_rate * self.Kt
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



