import copy
import numpy as np
import torch

from .government import GovernmentRules
from .households import HouseholdRules
from .market import MarketRules
from .bank import BankRules


class rule_agent:
    """
    Lightweight rule-based agent wrapper.

    - Keeps the high-level interface (get_action / train / save).
    - Delegates concrete rules to role-specific modules.
    """

    def __init__(self, envs, args, agent_name=None, type=None):
        self.envs = envs
        self.eval_env = copy.copy(envs)
        self.args = args
        self.agent_name = agent_name
        self.agent_type = type
        self.on_policy = True

    def get_action(self, obs_tensor):
        """
        Dispatch to the correct rule set by role.
        """
        name = self.agent_name
        firm_n = self.envs.market.firm_n
        if name == "government":
            action_dim = self.envs.government[self.agent_type].action_dim
            return GovernmentRules.get_action(type=self.agent_type, obs=obs_tensor, action_dim=action_dim)

        if name == "households":
            action_dim = self.envs.households.action_dim
            country = getattr(self.args, "household_country", "China")
            return HouseholdRules.get_action(
                type=self.agent_type, obs=obs_tensor, action_dim=action_dim,
                firm_n=firm_n, country=country,
                rule=getattr(self.args, "household_rule", "age_profile"),
                consumption_share=getattr(self.args, "household_consumption_share", 0.95),
                labor_rule=getattr(self.args, "household_labor_rule", "initial"),
                initial_work=self.envs.households.work_init,
                labor_share=getattr(self.args, "household_labor_share", 0.5),
            )

        if name == "market":
            action_dim = self.envs.market.action_dim
            return MarketRules.get_action(
                type=self.agent_type,
                obs=obs_tensor,
                action_dim=action_dim,
                alpha=self.envs.market.alpha,
                depreciation_rate=self.envs.bank.depreciation_rate,
                markup=self.envs.market.markup,
            )

        if name == "bank":
            action_dim = self.envs.bank.action_dim
            return BankRules.get_action(type=self.agent_type, obs=obs_tensor, action_dim=action_dim)

        raise ValueError(f"Unknown agent_name: {name}")

    # Keep the framework API
    def train(self, transition):
        return 0, 0

    def save(self, dir_path):
        pass
