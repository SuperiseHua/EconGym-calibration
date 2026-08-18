"""Map generic LLM decisions to EconGym roles."""

import numpy as np

from .monetary_policy import (
    DEFAULT_SLOW_POLICY,
    FAST_POLICY_SCHEMA,
    SLOW_POLICY_SCHEMA,
    fast_policy_constraints,
    policy_to_action,
)
from .prompts import (
    build_bank_prompt,
    build_fast_monetary_prompt,
    build_market_prompt,
    build_pension_prompt,
    build_slow_monetary_prompt,
    build_tax_prompt,
)


class EconAdapter:
    """Role-specific behavior kept outside the generic LLM client."""

    def __init__(self, env, agent_name, agent_type):
        self.env, self.name, self.type = env, agent_name, agent_type
        if agent_name == "government":
            if agent_type not in env.government:
                raise ValueError(f"Unknown government type: {agent_type}")
            self.entity = env.government[agent_type]
        elif agent_name in {"bank", "market"}:
            self.entity = getattr(env, agent_name)
        else:
            raise ValueError(f"Unsupported LLM role: {agent_name}")

    @property
    def uses_slow_policy(self):
        return self.name == "government" and self.type == "central_bank"

    def slow_prompt(self, obs):
        return build_slow_monetary_prompt(obs)

    def prompt(self, obs, slow_policy=None, previous_obs=None, previous_decision=None):
        if self.name == "government" and self.type == "central_bank":
            cb = self.entity
            return build_fast_monetary_prompt(
                obs, slow_policy,
                previous_obs=previous_obs,
                previous_decision=previous_decision,
                execution_constraints=fast_policy_constraints(slow_policy, obs),
                policy_rate_bounds=(cb.real_action_min[0], cb.real_action_max[0]),
                reserve_ratio_bounds=(cb.real_action_min[1], cb.real_action_max[1]),
            )
        if self.name == "government" and self.type == "pension":
            return build_pension_prompt(obs, objective=self.entity.gov_task)
        if self.name == "government":
            return build_tax_prompt(obs, objective=self.entity.gov_task)
        return build_bank_prompt(obs) if self.name == "bank" else build_market_prompt(obs)

    def repair_prompt(self, error):
        return (
            "The previous JSON decision violated a binding controller constraint: "
            f"{error} Return a corrected JSON decision only. Do not change the approved slow policy."
        )

    def slow_repair_prompt(self, error):
        return (
            "The proposed slow policy is internally inconsistent: "
            f"{error}. Revise only the reaction framework as needed, preserve the JSON schema, "
            "and return the complete corrected slow policy."
        )

    def action(self, decision, obs):
        if self.name == "government" and self.type == "central_bank":
            cb = self.entity
            return policy_to_action(
                decision,
                obs[2:4],
                cb.real_action_min,
                cb.real_action_max,
            )
        if self.name == "government" and self.type == "pension":
            return np.array([decision["retire_age"], decision["contribution_rate"]], dtype=float)
        if self.name == "government":
            keys = ("tau", "xi", "tau_a", "xi_a", "Gt_prob")
            return np.array([decision[key] for key in keys], dtype=float)
        if self.name == "bank":
            return np.array([decision["lending_rate"], decision["deposit_rate"]], dtype=float)
        return np.array([[item["price"], item["wage"]] for item in decision], dtype=np.float32)

    def fallback(self, obs):
        if self.name == "government" and self.type == "central_bank":
            return np.asarray(obs[2:4], dtype=np.float32)
        if self.name == "government" and self.type == "pension":
            return np.asarray(obs[-2:])
        if self.name == "government":
            return np.array([0, 0, 0, 0, obs[-1]], dtype=float)
        if self.name == "bank":
            rate = float(obs[0]) if len(obs) else 0.03
            return np.array([rate + 0.02, rate - 0.005])
        firms = obs.shape[0] if obs.ndim == 2 else 1
        return np.zeros((firms, 2), dtype=np.float32)

    def direct_action(self, obs):
        if self.name == "bank" and self.type == "non_profit":
            return np.random.randn(self.entity.action_dim)
        if self.name == "market" and self.type == "perfect":
            return np.random.randn(len(obs), self.entity.action_dim)
        return None

    def slow_fallback(self):
        return DEFAULT_SLOW_POLICY.copy()

    def request_options(self, model, stage="fast"):
        options = {"temperature": 0.0 if self.type == "central_bank" else 0.3}
        if self.name == "government" and self.type == "central_bank" and "gpt" in model.lower():
            schema = SLOW_POLICY_SCHEMA if stage == "slow" else FAST_POLICY_SCHEMA
            options["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": f"{stage}_monetary_policy", "strict": True, "schema": schema},
            }
        return options
