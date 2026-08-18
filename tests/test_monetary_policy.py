import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agents.llm.monetary_policy import (
    DEFAULT_SLOW_POLICY,
    fast_policy_constraints,
    policy_to_action,
    validate_fast_policy,
    validate_slow_policy,
)
from agents.llm.llm_agent import llm_agent
from agents.llm.llm_client import create_llm_client
from agents.llm.econ_adapter import EconAdapter
from agents.llm.prompts import build_fast_monetary_prompt, build_slow_monetary_prompt
from agents.rule_based.government import GovernmentRules
from env.env_core import EconomicSociety
from entities.market import Market
from utils.config import load_config


VALID_DECISION = {
    "policy_rate_action": {"direction": "increase", "change_bps": 75},
    "reserve_ratio_action": {"direction": "decrease", "change_bps": 25},
}


class MonetaryPolicyTests(unittest.TestCase):
    def test_chain_fisher_inflation(self):
        prices_old, prices = np.array([1.0, 2.0]), np.array([1.1, 1.8])
        quantities_old, quantities = np.array([10.0, 5.0]), np.array([8.0, 8.0])
        inflation, index = Market.compute_inflation_rate(
            prices, quantities, prices_old, quantities_old,
        )
        laspeyres = np.dot(prices, quantities_old) / np.dot(prices_old, quantities_old)
        paasche = np.dot(prices, quantities) / np.dot(prices_old, quantities)
        expected = np.sqrt(laspeyres * paasche)
        self.assertAlmostEqual(inflation, expected - 1)
        self.assertAlmostEqual(index, 100 * expected)

    def test_single_good_inflation_is_its_price_change(self):
        inflation, index = Market.compute_inflation_rate([1.05], [10], [1.0])
        self.assertAlmostEqual(inflation, 0.05)
        self.assertAlmostEqual(index, 105.0)

    def test_price_fully_adjusts_to_planned_demand(self):
        market = object.__new__(Market)
        market.price = np.array([[1.0]])
        market.update_price([[120.0]], [[100.0]])
        self.assertAlmostEqual(float(market.price.item()), 1.2)

    def test_valid_decision_converts_to_absolute_econgym_action(self):
        action = policy_to_action(
            VALID_DECISION,
            current=[0.01, 0.08],
        )
        np.testing.assert_allclose(action, np.array([0.0175, 0.0775]), atol=1e-7)

    def test_json_code_block_is_parsed(self):
        text = f"```json\n{json.dumps(VALID_DECISION)}\n```"
        action = policy_to_action(
            text,
            current=[0.01, 0.08],
        )
        np.testing.assert_allclose(action, np.array([0.0175, 0.0775]), atol=1e-7)

    def test_hold_with_nonzero_change_is_rejected(self):
        invalid = {
            **VALID_DECISION,
            "policy_rate_action": {"direction": "hold", "change_bps": 25},
        }
        with self.assertRaises(ValueError):
            policy_to_action(invalid, current=[0.01, 0.08])

    def test_extra_output_field_is_rejected(self):
        invalid = {**VALID_DECISION, "rationale": "not part of the policy action"}
        with self.assertRaises(ValueError):
            policy_to_action(invalid, current=[0.01, 0.08])

    def test_out_of_bounds_action_is_rejected(self):
        with self.assertRaises(ValueError):
            policy_to_action(
                VALID_DECISION,
                current=[0.10, 0.08],
            )

    def test_hold_fallback_preserves_both_current_values(self):
        action = policy_to_action(
            {
                "policy_rate_action": {"direction": "hold", "change_bps": 0},
                "reserve_ratio_action": {"direction": "hold", "change_bps": 0},
            },
            current=[0.01625, 0.0],
        )
        np.testing.assert_allclose(action, np.array([0.01625, 0.0]), atol=1e-7)

    def test_prompt_contains_state_but_forbids_repeating_it(self):
        obs = np.array([0.021234, 0.034567, 0.03, 0.08, 0.045])
        constraints = fast_policy_constraints(DEFAULT_SLOW_POLICY, obs)
        prompt = build_fast_monetary_prompt(
            obs, DEFAULT_SLOW_POLICY, execution_constraints=constraints
        )
        self.assertIn('"inflation_pct": 2.12', prompt)
        self.assertIn('"inflation_gap_pp": 0.12', prompt)
        self.assertIn('"policy_rate_pct": 3.0', prompt)
        self.assertNotIn("wealth", prompt.lower())
        self.assertNotIn("education", prompt.lower())
        self.assertNotIn("np.float64", prompt)
        self.assertIn("Binding execution constraints", prompt)
        self.assertIn('"required_policy_rate_direction": "hold"', prompt)
        self.assertIn('"allowed_policy_rate_change_bps": [\n    0\n  ]', prompt)

    def test_slow_prompt_uses_policy_relevant_state_and_principles(self):
        prompt = build_slow_monetary_prompt(np.array([0.04, 0.01, 0.03, 0.08, 0.045]))
        self.assertIn('"inflation_gap_pp": 2.0', prompt)
        self.assertIn("medium-term", prompt)
        self.assertIn("not predetermined actions", prompt)
        self.assertIn("not official Federal Reserve rules", prompt)
        self.assertIn("Do not assume in advance", prompt)
        self.assertNotIn("top10", prompt)

    def test_runtime_schema_matches_dataset_template(self):
        runtime_schema = json.loads(
            (Path(__file__).parents[1] / "agents/llm/fast_policy_template.json").read_text()
        )
        dataset_schema = json.loads(
            (Path(__file__).parents[3]
             / "dataset/us_fed_monetary_policy_2022/templates/fast_policy_template.json").read_text()
        )
        self.assertEqual(runtime_schema, dataset_schema)

    def test_parsed_policy_runs_one_econgym_step(self):
        np.random.seed(0)
        config = load_config("inflation_control")
        env = EconomicSociety(config.Environment)
        env.reset()
        central_bank = env.government["central_bank"]

        action = policy_to_action(
            json.dumps({
                "policy_rate_action": {"direction": "increase", "change_bps": 25},
                "reserve_ratio_action": {"direction": "decrease", "change_bps": 25},
            }),
            current=[central_bank.base_interest_rate, central_bank.reserve_ratio],
            action_min=central_bank.real_action_min,
            action_max=central_bank.real_action_max,
        )
        actions = {
            "government": {"central_bank": action},
            "households": env.households.initial_action.copy(),
            "market": env.market.action_space.sample(),
            "bank": np.array([0.045, 0.025], dtype=np.float32),
        }
        next_observations, _, _ = env.step(actions, t=0)

        np.testing.assert_allclose(
            [central_bank.base_interest_rate, central_bank.reserve_ratio], action, atol=1e-7
        )
        np.testing.assert_allclose(
            [env.bank.base_interest_rate, env.bank.reserve_ratio], action, atol=1e-7
        )
        self.assertEqual(env.step_cnt, 1)
        self.assertEqual(set(next_observations), set(env.agents))
        self.assertAlmostEqual(
            next_observations["government"]["central_bank"][3],
            central_bank.reserve_ratio,
        )
        self.assertAlmostEqual(
            next_observations["government"]["central_bank"][1],
            central_bank.growth_rate,
        )

    def test_reset_restores_initial_monetary_policy(self):
        config = load_config("inflation_control")
        env = EconomicSociety(config.Environment)
        central_bank = env.government["central_bank"]
        central_bank.base_interest_rate = 0.075
        central_bank.reserve_ratio = 0.15

        observations = env.reset()

        np.testing.assert_allclose(
            [central_bank.base_interest_rate, central_bank.reserve_ratio],
            [0.03, 0.08],
        )
        np.testing.assert_allclose(
            observations["government"]["central_bank"][2:4],
            [0.03, 0.08],
        )
        np.testing.assert_allclose(
            [env.bank.base_interest_rate, env.bank.reserve_ratio],
            [0.03, 0.08],
        )

    def test_llm_agent_uses_prompt_and_strict_schema_without_network(self):
        np.random.seed(0)
        config = load_config("inflation_control")
        env = EconomicSociety(config.Environment)
        observations = env.reset()
        args = SimpleNamespace(
            llm_name="gpt-4.1-mini",
            api_key="test-key-not-used",
            cuda=False,
        )
        agent = llm_agent(env, args, type="central_bank", agent_name="government")

        class FakeCompletions:
            requests = []

            def create(self, **request):
                self.requests.append(request)
                stage = request["response_format"]["json_schema"]["name"]
                decision = DEFAULT_SLOW_POLICY if stage.startswith("slow") else {
                    "policy_rate_action": {"direction": "hold", "change_bps": 0},
                    "reserve_ratio_action": {"direction": "hold", "change_bps": 0},
                }
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(decision)))]
                )

        fake_completions = FakeCompletions()
        agent.client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )
        action = agent.get_action(observations["government"]["central_bank"])

        np.testing.assert_allclose(action, np.array([0.03, 0.08]), atol=1e-7)
        self.assertEqual(len(fake_completions.requests), 2)
        slow_request, fast_request = fake_completions.requests
        self.assertIn("medium-term", slow_request["messages"][0]["content"])
        response_format = fast_request["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertNotIn("$schema", response_format["json_schema"]["schema"])
        self.assertIn("Approved slow policy", fast_request["messages"][0]["content"])

    def test_invalid_fast_policy_is_repaired_once(self):
        config = load_config("inflation_control")
        env = EconomicSociety(config.Environment)
        obs = env.reset()["government"]["central_bank"]
        args = SimpleNamespace(llm_name="gpt-4.1-mini", api_key="test-key", cuda=False)
        agent = llm_agent(env, args, type="central_bank", agent_name="government")

        class RepairCompletions:
            def __init__(self):
                self.requests = []

            def create(self, **request):
                self.requests.append(request)
                stage = request["response_format"]["json_schema"]["name"]
                if stage.startswith("slow"):
                    decision = DEFAULT_SLOW_POLICY
                elif len(request["messages"]) == 1:
                    decision = {
                        "policy_rate_action": {"direction": "increase", "change_bps": 50},
                        "reserve_ratio_action": {"direction": "hold", "change_bps": 0},
                    }
                else:
                    decision = {
                        "policy_rate_action": {"direction": "hold", "change_bps": 0},
                        "reserve_ratio_action": {"direction": "hold", "change_bps": 0},
                    }
                message = SimpleNamespace(content=json.dumps(decision))
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        completions = RepairCompletions()
        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        action = agent.get_action(obs)

        np.testing.assert_allclose(action, [0.03, 0.08], atol=1e-7)
        self.assertEqual(len(completions.requests), 3)
        repair_messages = completions.requests[-1]["messages"]
        self.assertEqual(len(repair_messages), 3)
        self.assertIn("binding controller constraint", repair_messages[-1]["content"])
        self.assertIsNone(agent.last_error)

    def test_inconsistent_slow_policy_is_repaired_once(self):
        config = load_config("inflation_control")
        env = EconomicSociety(config.Environment)
        obs = env.reset()["government"]["central_bank"]
        args = SimpleNamespace(llm_name="gpt-4.1-mini", api_key="test-key", cuda=False)
        agent = llm_agent(env, args, type="central_bank", agent_name="government")

        inconsistent = json.loads(json.dumps(DEFAULT_SLOW_POLICY))
        inconsistent["reaction_rule"]["high_inflation_positive_growth"] = "hold"
        inconsistent["reaction_rule"]["high_inflation_weak_growth"] = "tighten"

        class SlowRepairCompletions:
            def __init__(self):
                self.slow_calls = 0
                self.repair_prompt = ""

            def create(self, **request):
                stage = request["response_format"]["json_schema"]["name"]
                if stage.startswith("slow"):
                    self.slow_calls += 1
                    if len(request["messages"]) > 1:
                        self.repair_prompt = request["messages"][-1]["content"]
                    decision = inconsistent if self.slow_calls == 1 else DEFAULT_SLOW_POLICY
                else:
                    decision = {
                        "policy_rate_action": {"direction": "hold", "change_bps": 0},
                        "reserve_ratio_action": {"direction": "hold", "change_bps": 0},
                    }
                message = SimpleNamespace(content=json.dumps(decision))
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        completions = SlowRepairCompletions()
        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        agent.get_action(obs)

        self.assertEqual(completions.slow_calls, 2)
        self.assertIn("high_inflation_positive_growth", completions.repair_prompt)
        self.assertEqual(agent.slow_policy, DEFAULT_SLOW_POLICY)

    def test_slow_policy_rejects_basic_economic_inconsistency(self):
        invalid = json.loads(json.dumps(DEFAULT_SLOW_POLICY))
        invalid["reaction_rule"]["high_inflation_positive_growth"] = "ease"
        with self.assertRaises(ValueError):
            validate_slow_policy(invalid)

    def test_fast_policy_must_follow_slow_rule_and_step_limit(self):
        high_inflation = np.array([0.04, 0.02, 0.03, 0.08, 0.045])
        valid = {
            "policy_rate_action": {"direction": "increase", "change_bps": 25},
            "reserve_ratio_action": {"direction": "hold", "change_bps": 0},
        }
        self.assertIs(validate_fast_policy(valid, DEFAULT_SLOW_POLICY, high_inflation), valid)

        inconsistent = json.loads(json.dumps(valid))
        inconsistent["policy_rate_action"] = {"direction": "decrease", "change_bps": 25}
        with self.assertRaises(ValueError):
            validate_fast_policy(inconsistent, DEFAULT_SLOW_POLICY, high_inflation)

        too_large = json.loads(json.dumps(valid))
        too_large["policy_rate_action"]["change_bps"] = 50
        with self.assertRaises(ValueError):
            validate_fast_policy(too_large, DEFAULT_SLOW_POLICY, high_inflation)

        reserve_change = json.loads(json.dumps(valid))
        reserve_change["reserve_ratio_action"] = {"direction": "increase", "change_bps": 25}
        with self.assertRaises(ValueError):
            validate_fast_policy(reserve_change, DEFAULT_SLOW_POLICY, high_inflation)

    def test_runtime_slow_schema_matches_dataset_template(self):
        runtime = json.loads(
            (Path(__file__).parents[1] / "agents/llm/slow_policy_template.json").read_text()
        )
        dataset = json.loads(
            (Path(__file__).parents[3]
             / "dataset/us_fed_monetary_policy_2022/templates/slow_policy_template.json").read_text()
        )
        self.assertEqual(runtime, dataset)

    def test_taylor_rule_reads_compact_central_bank_observation(self):
        obs = np.array([0.02, 0.05, 0.03, 0.08, 0.0345])
        action = GovernmentRules.cb_rule_taylor(obs, noise=None)
        np.testing.assert_allclose(action, [0.04, 0.08], atol=1e-8)

    def test_openrouter_client_configuration(self):
        args = SimpleNamespace(
            llm_provider="openrouter",
            api_key="test-key-not-used",
            openrouter_site_url="https://example.edu/research",
            openrouter_app_name="AgentEWM",
        )
        client = create_llm_client(args)

        self.assertEqual(str(client.base_url), "https://openrouter.ai/api/v1/")
        self.assertEqual(client.api_key, "test-key-not-used")
        self.assertEqual(client.default_headers["HTTP-Referer"], args.openrouter_site_url)
        self.assertEqual(client.default_headers["X-Title"], args.openrouter_app_name)

    def test_generic_adapter_keeps_other_economic_roles(self):
        tax_env = EconomicSociety(load_config("optimal_tax").Environment)
        tax_obs = tax_env.reset()["government"]["tax"]
        tax = EconAdapter(tax_env, "government", "tax")
        np.testing.assert_allclose(
            tax.action({"tau": .1, "xi": .2, "tau_a": .01, "xi_a": .3, "Gt_prob": .4}, tax_obs),
            [.1, .2, .01, .3, .4],
        )

        market_env = EconomicSociety(load_config("monopoly").Environment)
        market_obs = market_env.reset()["market"]
        market = EconAdapter(market_env, "market", "monopoly")
        self.assertEqual(market.action([{"price": 1.1, "wage": .9}], market_obs).shape, (1, 2))

        bank = EconAdapter(market_env, "bank", "commercial")
        np.testing.assert_allclose(
            bank.action({"lending_rate": .05, "deposit_rate": .02}, market_env.reset()["bank"]),
            [.05, .02],
        )


if __name__ == "__main__":
    unittest.main()
