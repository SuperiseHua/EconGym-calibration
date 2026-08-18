import copy

from entities.base import BaseEntity
import numpy as np
from gymnasium.spaces import Box
from omegaconf import ListConfig


class Bank(BaseEntity):
    name = 'bank'

    def __init__(self, entity_args):
        super().__init__()
        self.entity_args = entity_args
        self.__dict__.update(entity_args['params'])
        self.action_dim = entity_args[self.type]['action_dim']
        self.initial_action = entity_args[self.type]['initial_action']
        self.action_space = Box(
            low=self.action_space['low'], high=self.action_space['high'], shape=(self.action_dim,), dtype=np.float32
        )

    def reset(self, **custom_cfg):
        households_at = custom_cfg['households_at']
        if isinstance(self.initial_action, (list, ListConfig)):
            self.initial_action = np.array(self.initial_action) + np.random.randn(
                self.action_dim) * self.action_space.high

        self.current_account = np.sum(households_at)

        self.deposit_rate = self.entity_args['params'].deposit_rate
        self.lending_rate = self.entity_args['params'].lending_rate
        self.reserve_ratio = self.entity_args['params'].reserve_ratio
        self.base_interest_rate = self.entity_args['params'].base_interest_rate
        self.last_deposit_rate = copy.copy(self.deposit_rate)
        self.last_lending_rate = copy.copy(self.lending_rate)
        self.capital_loan = 0.0
        # self.last_lending_rate_j = copy.copy(self.lending_rate)

    def get_action(self, actions, central_bank_exist=False):
        if self.type == 'commercial':
            # For commercial banks, actions are lending rate and deposit rate
            self.lending_rate, self.deposit_rate = actions
            if central_bank_exist:
                self.lending_rate = np.clip(self.lending_rate, self.base_interest_rate + 0.01, self.base_interest_rate + 0.03)
                self.deposit_rate = np.clip(self.deposit_rate, self.base_interest_rate - 0.01, self.base_interest_rate)


    def step(self, society):
        # Retrieve the first government agent from the society's government dictionary
        self.gov_agent = society.main_gov

        # Check if the society has a 'government' attribute
        if hasattr(society, 'government'):
            # If 'central_bank' exists in the government dictionary, prioritize it
            # The central bank dynamically adjusts reserve_ratio and base_interest_rate, so they must be assigned first
            if "central_bank" in society.government:
                central_bank = society.government["central_bank"]
                self.reserve_ratio = central_bank.reserve_ratio
                self.base_interest_rate = central_bank.base_interest_rate
            else:
                # If no central bank, assign values from the first government agent
                self.reserve_ratio = self.gov_agent.reserve_ratio
                self.base_interest_rate = self.gov_agent.base_interest_rate
                
        if society.step_cnt == 0:
            self.capital_loan = np.sum(society.market.price * society.market.Kt)
            self.current_account -= self.gov_agent.Bt + self.capital_loan

        # Settle the previous period's borrowing interest and deposit rate
        # Government debt rates are usually based on the central bank's benchmark rate.
        previous_settlement = - (1 + self.deposit_rate) * np.sum(society.households.at) \
                              + (self.lending_rate + 1 - self.depreciation_rate) * self.capital_loan \
                              + (1 + self.lending_rate) * self.gov_agent.Bt

        current_deposit = np.sum(society.households.at_next)  # Deposits at this step in the bank

        total_deposit = self.current_account + previous_settlement + current_deposit

        society.market.Kt_next = self.compute_next_kt(society, total_deposit)

        capital_loan = np.sum(society.market.price * society.market.Kt_next)
        current_loan = capital_loan + self.gov_agent.Bt_next  # Nominal loans issued

        self.profit = self.lending_rate * capital_loan + self.lending_rate * self.gov_agent.Bt_next \
                      - self.deposit_rate * current_deposit

        self.current_account += previous_settlement + current_deposit - current_loan  # Current account balance
        self.capital_loan = capital_loan

        self.last_deposit_rate = copy.copy(self.deposit_rate)
        self.last_lending_rate = copy.copy(self.lending_rate)

    def compute_next_kt(self, society, total_deposit):
        consumption_sum = society.households.final_consumption.sum(axis=0)[:, np.newaxis]
        real_investment = np.maximum(
            society.market.Yt_j - self.gov_agent.gov_spending - consumption_sum, 0.0
        )
        Kt_next = real_investment + (1 - self.depreciation_rate) * society.market.Kt

        nominal_credit = max(total_deposit * (1 - self.reserve_ratio) - self.gov_agent.Bt_next, 0.0)
        desired_credit = np.sum(society.market.price * Kt_next)
        if desired_credit > nominal_credit:
            Kt_next *= nominal_credit / (desired_credit + 1e-8)
        return Kt_next

    def get_reward(self):
        """Profit is based on the interest spread between loans and deposits."""
        if self.type == "non_profit":
            return np.array([0.])
        elif self.type == "commercial":
            reward = self.scaled_reward(self.profit)
            if isinstance(reward, np.ndarray):
                return reward
            else:
                return np.array([reward])
        else:
            raise ValueError(f"Invalid bank type: '{self.type}'. Expected 'non_profit' or 'commercial'.")

    def scaled_reward(self, x, eps=1e-8, k=0.15):  # \in (0,1)
        x = np.asarray(x, dtype=np.float64)
        log_scaled = np.sign(x) * np.log1p(np.abs(x) + eps)
        return 1 / (1 + np.exp(-k * log_scaled))

    def is_terminal(self):
        return False
