"""Convert an LLM monetary-policy JSON response to an EconGym action."""

import json
from pathlib import Path

import numpy as np

from .json_io import parse_json


def load_schema(name):
    schema = json.loads(Path(__file__).with_name(name).read_text(encoding="utf-8"))
    schema.pop("$schema", None)
    return schema


FAST_POLICY_SCHEMA = load_schema("fast_policy_template.json")
SLOW_POLICY_SCHEMA = load_schema("slow_policy_template.json")

DEFAULT_SLOW_POLICY = {
    "policy_name": "Balanced medium-term stabilization",
    "diagnosis": "Use inflation stabilization as the anchor while avoiding prolonged contraction.",
    "objective": {"inflation_target_pct": 2.0, "horizon": "medium_term", "growth_safeguard": "avoid_prolonged_contraction"},
    "instrument_strategy": {"primary_instrument": "policy_rate", "reserve_ratio_role": "rarely"},
    "reaction_rule": {
        "inflation_tolerance_pp": 0.5, "weak_growth_pct": 0.0,
        "high_inflation_positive_growth": "tighten", "high_inflation_weak_growth": "hold",
        "low_inflation_positive_growth": "hold", "low_inflation_weak_growth": "ease",
        "near_target": "hold",
    },
    "risk_management": {"max_step_bps": 25, "consider_policy_lags": True, "avoid_unnecessary_reversal": True},
    "review_interval": 8,
}


def validate_slow_policy(policy):
    """Reject directionally inconsistent reaction rules without prescribing a stance."""
    required = set(SLOW_POLICY_SCHEMA["required"])
    if not isinstance(policy, dict) or set(policy) != required:
        raise ValueError("Slow policy has missing or extra fields.")
    rule, risk = policy["reaction_rule"], policy["risk_management"]
    rank = {"ease": -1, "hold": 0, "tighten": 1}
    comparisons = (
        ("high_inflation_positive_growth", "low_inflation_positive_growth"),
        ("high_inflation_weak_growth", "low_inflation_weak_growth"),
        ("high_inflation_positive_growth", "high_inflation_weak_growth"),
        ("low_inflation_positive_growth", "low_inflation_weak_growth"),
    )
    conflicts = [
        f"{left}={rule[left]} is looser than {right}={rule[right]}"
        for left, right in comparisons if rank[rule[left]] < rank[rule[right]]
    ]
    if conflicts:
        raise ValueError("; ".join(conflicts))
    if risk["max_step_bps"] not in {25, 50, 75}:
        raise ValueError("Invalid maximum policy step.")
    return policy


def fast_policy_constraints(slow_policy, obs, inflation_target=0.02):
    """Compute the binding fast-policy constraints implied by the slow rule."""
    rule = slow_policy["reaction_rule"]
    gap_pp = (float(obs[0]) - inflation_target) * 100
    growth_pct = float(obs[1]) * 100
    if abs(gap_pp) <= rule["inflation_tolerance_pp"]:
        regime = "near_target"
    else:
        inflation = "high_inflation" if gap_pp > 0 else "low_inflation"
        growth = "weak_growth" if growth_pct <= rule["weak_growth_pct"] else "positive_growth"
        regime = f"{inflation}_{growth}"

    stance = rule[regime]
    expected = {"tighten": "increase", "ease": "decrease", "hold": "hold"}[stance]
    max_step = slow_policy["risk_management"]["max_step_bps"]
    allowed_steps = [0] if expected == "hold" else [step for step in (25, 50, 75) if step <= max_step]
    reserve_role = slow_policy["instrument_strategy"]["reserve_ratio_role"]
    reserve_constraint = {
        "rarely": "hold",
        "complementary": f"hold or {expected}",
        "active": "increase, decrease, or hold",
    }[reserve_role]
    return {
        "applicable_regime": regime,
        "slow_policy_stance": stance,
        "required_policy_rate_direction": expected,
        "allowed_policy_rate_change_bps": allowed_steps,
        "maximum_change_bps_for_any_instrument": max_step,
        "reserve_ratio_role": reserve_role,
        "allowed_reserve_ratio_direction": reserve_constraint,
    }


def validate_fast_policy(policy, slow_policy, obs, inflation_target=0.02):
    """Ensure the fast decision executes the approved reaction rule."""
    constraints = fast_policy_constraints(slow_policy, obs, inflation_target)
    expected = constraints["required_policy_rate_direction"]
    rate_action = policy["policy_rate_action"]
    if rate_action["direction"] != expected:
        raise ValueError(
            f"Policy-rate direction must be {expected} in regime "
            f"{constraints['applicable_regime']}."
        )
    if rate_action["change_bps"] not in constraints["allowed_policy_rate_change_bps"]:
        raise ValueError("Policy-rate step is not allowed by the approved slow policy.")
    max_step = constraints["maximum_change_bps_for_any_instrument"]
    if max(block["change_bps"] for block in policy.values()) > max_step:
        raise ValueError("Fast policy exceeds the approved maximum step.")
    reserve_action = policy["reserve_ratio_action"]
    reserve_role = constraints["reserve_ratio_role"]
    if reserve_role == "rarely" and reserve_action["direction"] != "hold":
        raise ValueError("The slow policy assigns no routine role to reserve-ratio changes.")
    if reserve_role == "complementary" and reserve_action["direction"] not in {"hold", expected}:
        raise ValueError("A complementary reserve-ratio action must reinforce the policy-rate direction.")
    return policy

POLICY_FIELDS = ("policy_rate_action", "reserve_ratio_action")
SIGNS = {"increase": 1, "decrease": -1, "hold": 0}
ALLOWED_BPS = {0, 25, 50, 75}

def policy_to_action(output, current, action_min=(-0.02, 0.0), action_max=(0.10, 0.20)):
    """Return absolute EconGym action ``[base_interest_rate, reserve_ratio]``."""
    policy = parse_json(output)
    if not isinstance(policy, dict) or set(policy) != set(POLICY_FIELDS):
        raise ValueError(f"Policy must contain exactly {POLICY_FIELDS}.")

    changes = []
    for field in POLICY_FIELDS:
        block = policy[field]
        if not isinstance(block, dict) or set(block) != {"direction", "change_bps"}:
            raise ValueError(f"Invalid {field}.")
        direction, bps = block["direction"], block["change_bps"]
        if direction not in SIGNS or type(bps) is not int or bps not in ALLOWED_BPS:
            raise ValueError(f"Invalid direction or change_bps in {field}.")
        if (direction == "hold") != (bps == 0):
            raise ValueError("hold requires 0 bps; increase/decrease requires a positive value.")
        changes.append(SIGNS[direction] * bps / 10_000)

    action = np.asarray(current, dtype=np.float32) + np.asarray(changes, dtype=np.float32)
    lower, upper = np.asarray(action_min), np.asarray(action_max)
    if action.shape != (2,) or lower.shape != (2,) or upper.shape != (2,):
        raise ValueError("Current action and bounds must each contain two values.")
    if np.any((action < lower) | (action > upper)):
        raise ValueError(f"Action {action.tolist()} is outside EconGym bounds.")
    return action
