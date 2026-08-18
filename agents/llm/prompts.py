import json

import numpy as np


def _round_pct(value):
    rounded = round(float(value) * 100, 2)
    return 0.0 if rounded == 0 else rounded


def monetary_state(obs, inflation_target=0.02, previous_obs=None):
    """Return the small set of interpretable variables used by monetary policy."""
    current = np.asarray(obs, dtype=float).ravel()
    if len(current) != 5:
        raise ValueError("Central-bank observation does not match the EconGym layout.")

    state = {
        "inflation_pct": _round_pct(current[0]),
        "inflation_gap_pp": _round_pct(current[0] - inflation_target),
        "gdp_growth_pct": _round_pct(current[1]),
        "policy_rate_pct": _round_pct(current[2]),
        "reserve_ratio_pct": _round_pct(current[3]),
        "lending_rate_pct": _round_pct(current[4]),
    }
    if previous_obs is not None:
        previous = np.asarray(previous_obs, dtype=float).ravel()
        state["inflation_change_pp"] = _round_pct(current[0] - previous[0])
        state["growth_change_pp"] = _round_pct(current[1] - previous[1])
    return state


def build_slow_monetary_prompt(obs, *, inflation_target=0.02):
    state = monetary_state(obs, inflation_target)
    return f"""You are the monetary-policy authority in EconGym. Analyze the supplied state and design a general medium-term, state-dependent policy framework. Do not assume in advance that the current state requires tightening, easing, or no change.

Initial macroeconomic state (percent unless marked pp):
{json.dumps(state, indent=2)}

Official objectives and considerations (not predetermined actions):
- The 2022 Federal Reserve framework seeks maximum employment and inflation that averages {inflation_target * 100:.1f}% over time.
- Policy decisions depend on incoming information, the medium-term outlook, the balance of risks, and transmission lags.
- When price-stability and real-activity objectives conflict, assess the size of each deviation and the horizon over which it may close.

EconGym mappings and experimental conventions (not official Federal Reserve rules):
- GDP growth is the available real-activity indicator; it is not an employment gap or a complete measure of economic health.
- The policy rate maps to the main interest-rate instrument. The reserve ratio is an additional simulated instrument whose role you must choose.
- Discrete thresholds, maximum step size, and review interval are experimental controller settings.

First diagnose only what the supplied variables support. Then encode a coherent reaction framework for high or low inflation under both positive and weak growth, plus inflation near target. Apply the same economic reasoning consistently across regimes. Do not claim that the result is an estimated real-world reaction function."""


def build_fast_monetary_prompt(
    obs,
    slow_policy,
    *,
    inflation_target=0.02,
    previous_obs=None,
    previous_decision=None,
    execution_constraints=None,
    policy_rate_bounds=(-0.02, 0.10),
    reserve_ratio_bounds=(0.0, 0.20),
):
    state = monetary_state(obs, inflation_target, previous_obs)
    bounds = {
        "policy_rate_pct": [round(float(x) * 100, 2) for x in policy_rate_bounds],
        "reserve_ratio_pct": [round(float(x) * 100, 2) for x in reserve_ratio_bounds],
    }
    return f"""You are implementing an approved medium-term monetary-policy framework in EconGym. Use macroeconomic reasoning, but do not redesign the framework in this fast decision.

Approved slow policy:
{json.dumps(slow_policy, indent=2)}

Current macroeconomic state (percent unless marked pp):
{json.dumps(state, indent=2)}

Previous fast decision:
{json.dumps(previous_decision) if previous_decision else "None (first decision of the episode)"}

Binding execution constraints computed from the approved slow policy:
{json.dumps(execution_constraints, indent=2)}

Decision procedure:
1. Execute the listed applicable regime and required policy-rate direction; do not reinterpret the slow rule.
2. Use only the listed allowed step sizes and respect the reserve-ratio constraint.
3. Consider observed changes, policy lags, and the previous decision only when selecting an allowed magnitude.
4. Keep the resulting levels within these percent bounds: {json.dumps(bounds)}.

Return only the two instrument actions required by the JSON schema. Do not include state variables, scores, or explanations."""


def build_pension_prompt(pension_obs: np.ndarray, objective: str) -> str:
    objective_text = {"pension_gap": "improve pension-fund sustainability"}.get(objective, objective)
    o = [float(x) for x in pension_obs]
    return f"""You lead the national pension authority. Adjust next-period retirement age and contribution rate to {objective_text} while protecting household welfare.

State: fund={o[0]:.2f}, population={o[1]:.0f}, retired={o[2]:.0f}, retirement_age={o[3]:.1f}, contribution_rate={o[4]:.3f}, debt={o[5]:.2f}, GDP={o[6]:.2f}.
Bounds: retirement_age in [60, 70], contribution_rate in [0.05, 0.20].
Return strict JSON only: {{"retire_age": 67, "contribution_rate": 0.08}}"""


def build_tax_prompt(tax_obs: np.ndarray, objective) -> str:
    objective_text = {
        "gdp": "maximize GDP growth", "gini": "reduce inequality",
        "social_welfare": "maximize social welfare", "mean_welfare": "increase mean welfare",
        "gdp_gini": "balance growth and equality",
    }.get(objective, objective)
    o = [round(float(x), 4) for x in tax_obs]
    return f"""You lead fiscal policy. Use the current EconGym state {o} to {objective_text}.
Choose tau in [0,0.6], xi in [0,2], tau_a in [0,0.05], xi_a in [0,2], and Gt_prob in [0,0.6].
Return strict JSON only with keys tau, xi, tau_a, xi_a, and Gt_prob."""


def build_bank_prompt(bank_obs: np.ndarray) -> str:
    b = [round(float(x), 4) for x in bank_obs]
    return f"""You manage a commercial bank. State [policy rate, reserve ratio, previous lending rate, previous deposit rate, loans, deposits]: {b}.
Choose lending_rate within policy rate + [0.01,0.03] and deposit_rate within policy rate + [-0.01,0].
Return strict JSON only with keys lending_rate and deposit_rate."""


def build_market_prompt(firm_obs: np.ndarray) -> str:
    firms = np.asarray(firm_obs, dtype=float).round(4).tolist()
    return f"""You set firm prices and wages. State rows [capital, productivity, borrowing rate]: {firms}.
Return one strict JSON object per row with price and wage, each within [0.0001,100]."""
