import copy
import random

import numpy as np
import torch
import os, sys
# import wandb
import swanlab as wandb
import json
from omegaconf import ListConfig

sys.path.append(os.path.abspath('../..'))
from agents.log_path import make_logpath, save_args
from utils.experience_replay import ReplayBuffer
from utils.evaluation_report import report_evaluation
from datetime import datetime

torch.autograd.set_detect_anomaly(True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)


def clean_metric(value):
    """Convert scalar-shaped NumPy metrics to plain numbers for readable reports."""
    array = np.asarray(value)
    if array.size == 1:
        return array.item()
    return array.tolist()


def summarize_trajectory(trajectory):
    """Summarize one episode using scale-free macro diagnostics."""
    if not trajectory:
        return {}

    gdp = np.asarray([row["real_gdp"] for row in trajectory], dtype=float)
    growth = np.asarray([row["gdp_growth"] for row in trajectory], dtype=float)
    investment = np.asarray([row["investment_gdp"] for row in trajectory], dtype=float)
    zero_investment = investment <= 1e-10
    running_peak = np.maximum.accumulate(gdp)

    return {
        "GDP_end": gdp[-1],
        "GDP_mean": float(np.mean(gdp)),
        "GDP_cumulative": float(np.sum(gdp)),
        "GDP_index_end": trajectory[-1]["real_gdp_index"],
        "GDP_growth_end": growth[-1],
        "GDP_growth_mean": float(np.mean(growth)),
        "GDP_growth_volatility": float(np.std(growth)),
        "GDP_max_drawdown": float(np.min(gdp / np.maximum(running_peak, 1e-8) - 1)),
        "price_index_end": trajectory[-1]["price_index"],
        "inflation_end": trajectory[-1]["inflation_rate"],
        "inflation_mean": float(np.mean([row["inflation_rate"] for row in trajectory])),
        "consumption_gdp_mean": float(np.mean([row["consumption_gdp"] for row in trajectory])),
        "investment_gdp_mean": float(np.mean(investment)),
        "government_gdp_mean": float(np.mean([row["government_gdp"] for row in trajectory])),
        "unused_output_gdp_mean": float(np.mean([row["unused_output_gdp"] for row in trajectory])),
        "zero_investment_share": float(np.mean(zero_investment)),
        "investment_switch_share": float(np.mean(zero_investment[1:] != zero_investment[:-1]))
        if len(zero_investment) > 1 else 0.0,
        "user_cost_floor_share": float(np.mean([row["user_cost_at_floor"] for row in trajectory])),
        "nonpositive_user_cost_share": float(np.mean([row.get("nonpositive_user_cost", False) for row in trajectory])),
        "credit_binding_share": float(np.mean([row["credit_binding"] for row in trajectory])),
        "goods_binding_share": float(np.mean([row["goods_binding"] for row in trajectory])),
    }


class Runner:
    def __init__(self, envs, args, house_agent, government_agent, firm_agent, bank_agent):
        self.envs = copy.deepcopy(envs)
        self.args = args
        self.eval_env = copy.deepcopy(envs)

        self.house_agent = house_agent
        self.government_agent = government_agent
        self.firm_agent = firm_agent
        self.bank_agent = bank_agent
        self.agents_policy = {
            "government": self.government_agent,
            "households": self.house_agent,
            "market": self.firm_agent,
            "bank": self.bank_agent,
        }

        self.households_n = self.envs.households.households_n

        self.eva_year_indicator = 0
        self.eva_reward_indicator = 0

        # define the replay buffer
        self.buffer = ReplayBuffer(self.args.batch_size)
        self.device = 'cuda' if getattr(self.args, "cuda", False) else 'cpu'

        self.model_path, self.file_name = make_logpath(args=self.args, n=self.households_n,
                                                       task=self.envs.problem_scene)
        save_args(path=self.model_path, args=self.args)
        self.wandb = self.args.wandb

        if self.wandb:
            from omegaconf import OmegaConf
            config_dict = OmegaConf.to_container(self.args, resolve=True)

            wandb.init(
                config=config_dict,
                project="EconGym",
                entity="EconGym",     # TODO: Replace with your swanlab account or team name
                name=self.file_name + "_seed=" + str(self.args.seed),
                dir=str(self.model_path),
                job_type="training",
            )

    def _get_tensor_inputs(self, obs_dict):
        def to_tensor(x):
            if isinstance(x, dict):
                return {k: to_tensor(v) for k, v in x.items()}
            else:
                return torch.as_tensor(x, dtype=torch.float32, device=self.device)

        return {k: to_tensor(v) for k, v in obs_dict.items()}

    # ------------------------------
    # RL-specific action postprocessing
    # ------------------------------
    def _process_rl_action(self, policy, action, path):
        """
        Postprocess RL actions:
        - Require network outputs in (-1, 1).
        - Scale them into [action_min, action_max] as defined by the agent.
        """
        rl_agent_list = ['ppo', 'ddpg', 'sac']  # extendable list of RL agents
        if getattr(policy, "name", None) not in rl_agent_list:
            return action  # skip if this is not an RL agent

        # Locate the corresponding agent entity
        if "." in path:
            main, sub = path.split(".", 1)
            agent = self.envs.agents[main][sub]
        else:
            agent = self.envs.agents[path]

        # Ensure real_action_min/max are numpy arrays
        action_min, action_max = agent.real_action_min, agent.real_action_max
        if isinstance(action_min, ListConfig):
            action_min = np.array(action_min, dtype=np.float32)
        if isinstance(action_max, ListConfig):
            action_max = np.array(action_max, dtype=np.float32)

        # Scale to [action_min, action_max]
        action = action_min + (action + 1.0) * (action_max - action_min) / 2.0
        return action

    # ------------------------------
    # Main function
    # ------------------------------
    def agents_get_action(self, obs_dict_tensor):
        """
        Get actions from all agents.
        Returns:
            raw_actions_dict: actions directly from policy.get_action (for replay buffer)
            processed_actions_dict: actions scaled/processed for environment execution
        """

        def act(policy, obs, path=""):
            if isinstance(policy, dict):
                raw, proc = {}, {}
                for k in policy:
                    raw_k, proc_k = act(policy[k], obs[k], f"{path}.{k}" if path else k)
                    raw[k], proc[k] = raw_k, proc_k
                return raw, proc

            try:
                action = policy.get_action(obs)
                # Raw action (saved into replay buffer)
                raw_action = action
                # Processed action (executed in the environment)
                proc_action = self._process_rl_action(policy, action, path)
                return raw_action, proc_action

            except KeyError as e:
                print(f"[Warning] obs missing key at '{path}': {e}")
                return None, None
            except Exception as e:
                print(
                    f"[Warning] get_action failed at '{path}' "
                    f"for policy {getattr(policy, 'name', type(policy).__name__)}: {e}")
                return None, None

        raw_actions_dict, processed_actions_dict = {}, {}
        ordered_agents = ["government"] + [
            name for name in self.agents_policy if name != "government"
        ]
        for agent_name in ordered_agents:
            if agent_name == "bank":
                central_bank_action = processed_actions_dict.get("government", {}).get("central_bank")
                if central_bank_action is not None:
                    bank_obs = obs_dict_tensor[agent_name]
                    if torch.is_tensor(bank_obs):
                        bank_obs = bank_obs.clone()
                        policy_state = torch.as_tensor(
                            central_bank_action, dtype=bank_obs.dtype, device=bank_obs.device
                        ).reshape(-1)
                    else:
                        bank_obs = np.array(bank_obs, copy=True)
                        policy_state = np.asarray(central_bank_action).reshape(-1)
                    bank_obs[..., :2] = policy_state[:2]
                    obs_dict_tensor[agent_name] = bank_obs

            raw_action, processed_action = act(
                self.agents_policy[agent_name], obs_dict_tensor[agent_name], agent_name
            )
            raw_actions_dict[agent_name] = raw_action
            processed_actions_dict[agent_name] = processed_action
        return raw_actions_dict, processed_actions_dict

    def reset_agent_episodes(self):
        def reset(policy):
            if isinstance(policy, dict):
                for item in policy.values():
                    reset(item)
            elif hasattr(policy, "reset_episode"):
                policy.reset_episode()
        reset(self.agents_policy)

    def run(self):
        obs_dict = self.envs.reset()
        self.reset_agent_episodes()

        for epoch in range(self.args.n_epochs):
            transition_dict = {
                "obs_dict": [],
                "action_dict": [],
                "reward_dict": [],
                "next_obs_dict": [],
                "done": []
            }

            sum_loss = {
                "actor_loss": {},
                "critic_loss": {}
            }

            for t in range(self.args.epoch_length):
                obs_dict_tensor = self._get_tensor_inputs(obs_dict)
                action_dict, processed_actions_dict = self.agents_get_action(obs_dict_tensor)
                next_obs_dict, reward_dict, done = self.envs.step(processed_actions_dict, t)

                on_policy_process = all(self.envs.recursive_decompose_dict(self.agents_policy, lambda a: a.on_policy))

                if on_policy_process:
                    # on policy
                    for key in transition_dict:
                        transition_dict[key].append(locals()[key])
                else:
                    # off-policy
                    for key in transition_dict:
                        transition_dict[key] = (locals()[key])
                    self.buffer.add(transition_dict)

                obs_dict = next_obs_dict
                if done:
                    obs_dict = self.envs.reset()
                    self.reset_agent_episodes()

            for agent_name in self.agents_policy:
                sub_agent_policy = self.agents_policy[agent_name]
                batch_size = self.args.epoch_length if on_policy_process else self.args.batch_size
                agent_data = self.buffer.sample(agent_name=agent_name, agent_policy=sub_agent_policy,
                                                batch_size=batch_size, on_policy=on_policy_process,
                                                transition_dict=transition_dict)
                if isinstance(sub_agent_policy, dict):
                    for name in sub_agent_policy:
                        sum_loss = self.sub_agent_training(agent_name=name,
                                                           agent_policy=sub_agent_policy[name],
                                                           transitions=agent_data[name],
                                                           loss=sum_loss)
                else:
                    sum_loss = self.sub_agent_training(agent_name=agent_name,
                                                       agent_policy=sub_agent_policy,
                                                       transitions=agent_data,
                                                       loss=sum_loss)

            # print the log information
            if epoch % self.args.display_interval == 0:
                economic_idicators_dict = self._evaluate_agent()

                if self.wandb:
                    wandb.log(economic_idicators_dict)
                    wandb.log(sum_loss)

                print(
                    "[{}] Epoch: {} / {}, Frames: {} | Real GDP={:.1f} | Growth={:+.1%} | "
                    "Price={:.1f} | Inflation={:+.1%} | C/Y={:.1%} | I/Y={:.1%} | G/Y={:.1%}".format(
                        datetime.now(), epoch, self.args.n_epochs, (epoch + 1) * self.args.epoch_length,
                        economic_idicators_dict.get("GDP_index_end", 0.0),
                        economic_idicators_dict.get("GDP_growth_end", 0.0),
                        economic_idicators_dict.get("price_index_end", 0.0),
                        economic_idicators_dict.get("inflation_end", 0.0),
                        economic_idicators_dict.get("consumption_gdp_mean", 0.0),
                        economic_idicators_dict.get("investment_gdp_mean", 0.0),
                        economic_idicators_dict.get("government_gdp_mean", 0.0),
                    )
                )

            if epoch % self.args.save_interval == 0:  # save_interval=10
                self.envs.recursive_decompose_dict(self.agents_policy, lambda a: a.save(dir_path=self.model_path))

        if self.wandb:
            wandb.finish()

    def sub_agent_training(self, agent_name, agent_policy, transitions, loss):
        if agent_policy.on_policy == True:
            actor_loss, critic_loss = agent_policy.train(transitions)
            loss['actor_loss'][agent_name] = actor_loss
            loss['critic_loss'][agent_name] = critic_loss
        else:
            total_actor_loss = 0.
            total_critic_loss = 0.
            for _ in range(self.args.update_cycles):
                actor_loss, critic_loss = agent_policy.train(transitions)
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
            loss['actor_loss'][agent_name] = total_actor_loss
            loss['critic_loss'][agent_name] = total_critic_loss
        return loss

    def test(self):
        ''' record the actions of gov and households'''
        results = self._evaluate_agent(write_evaluate_data=False, return_episodes=True)
        results["experiment"] = {
            "run": self.model_path.name,
            "problem_scene": self.eval_env.problem_scene,
            "central_bank_alg": self.args.get("central_bank_gov_alg", self.args.get("gov_alg")),
            "seed": self.args.seed,
            "eval_episodes": self.args.eval_episodes,
        }
        report_evaluation(results, self.model_path.parent / "evaluation_comparison.csv")
        result_path = self.model_path / "evaluation_results.json"
        with open(result_path, "w") as file:
            json.dump(results, file, cls=NumpyEncoder, indent=2)
        print(f"Evaluation results saved to: {result_path}")
        return results

    def economic_snapshot(self, period, initial_gdp, previous_gdp, expected_inflation):
        """Return one readable, scale-free snapshot without changing the simulation."""
        gdp = float(self.eval_env.main_gov.GDP)
        consumption = float(np.sum(self.eval_env.households.final_consumption))
        government = float(np.sum(self.eval_env.main_gov.gov_spending))
        desired = float(np.sum(self.eval_env.bank.desired_fixed_investment))
        financed = float(np.sum(self.eval_env.bank.financed_fixed_investment))
        investment = float(np.sum(self.eval_env.bank.actual_fixed_investment))
        capital = float(np.sum(self.eval_env.market.Kt))
        target_capital = float(np.sum(self.eval_env.bank.target_capital))
        tolerance = 1e-8 * max(gdp, 1.0)

        return {
            "period": period,
            "real_gdp": gdp,
            "real_gdp_index": 100.0 * gdp / max(float(initial_gdp), 1e-8),
            "gdp_growth": gdp / max(float(previous_gdp), 1e-8) - 1.0,
            "price_index": float(self.eval_env.price_index),
            "inflation_rate": float(self.eval_env.inflation_rate),
            "consumption_gdp": consumption / max(gdp, 1e-8),
            "investment_gdp": investment / max(gdp, 1e-8),
            "government_gdp": government / max(gdp, 1e-8),
            "unused_output_gdp": max(gdp - consumption - government - investment, 0.0)
            / max(gdp, 1e-8),
            "expected_inflation_used": float(expected_inflation),
            "lending_rate": float(self.eval_env.bank.lending_rate),
            "real_lending_rate": float(self.eval_env.bank.real_lending_rate),
            "capital_user_cost": float(self.eval_env.bank.capital_user_cost),
            "capital_target_ratio": target_capital / max(capital, 1e-8),
            "desired_investment_gdp": desired / max(gdp, 1e-8),
            "financed_investment_gdp": financed / max(gdp, 1e-8),
            "user_cost_at_floor": False,  # Legacy field: the artificial floor was removed.
            "nonpositive_user_cost": self.eval_env.bank.capital_user_cost <= 0,
            "bank_balance_sheet_residual": getattr(self.eval_env.bank, "balance_sheet_residual", None),
            "firm_loan_balance": float(self.eval_env.bank.capital_loan),
            "firm_cash": getattr(self.eval_env.bank, "firm_deposits", None),
            "accounting_failure": self.eval_env.bank.accounting_failure,
            "termination_reason": self.eval_env.termination_reason,
            "credit_binding": financed < desired - tolerance,
            "goods_binding": investment < financed - tolerance,
        }

    def viz_data(self, house_model_path, government_model_path):
        self.house_agent.load(dir_path=house_model_path)
        self.government_agent.load(dir_path=government_model_path)
        # this data is used for visualization
        self._evaluate_agent(write_evaluate_data=True)

    def init_economic_dict(self, reward_dict):
        gov_rewards = reward_dict['government']
        households_reward = reward_dict['households']
        firm_reward = reward_dict['market']
        bank_reward = reward_dict['bank']

        gov_reward = sum([reward_dict['government'][key] for key in reward_dict['government']])

        self.econ_dict = {
            "gov_reward": gov_reward,  # sum
            "tax_gov_reward": gov_rewards.get('tax', 0),
            "central_bank_gov_reward": gov_rewards.get('central_bank', 0),
            "pension_gov_reward": gov_rewards.get('pension', 0),
            "social_welfare": np.sum(households_reward),  # sum
            "house_reward": households_reward,  # sum
            "firm_reward": firm_reward,
            "bank_reward": bank_reward,
            "years": self.eval_env.step_cnt,  # max
            "house_income": self.eval_env.households.post_income,  # post_tax income
            "house_total_tax": self.eval_env.main_gov.tax_array,
            "house_income_tax": self.eval_env.households.income_tax,
            "house_pension": self.eval_env.households.pension,
            "house_wealth": self.eval_env.households.at_next,
            "house_wealth_tax": self.eval_env.households.asset_tax,
            "per_gdp": self.eval_env.main_gov.per_household_gdp,
            "GDP": self.eval_env.main_gov.GDP,  # real GDP, sum
            "nominal_GDP": self.eval_env.main_gov.nominal_GDP,
            "firm_production": self.eval_env.market.Yt_j,  # sum
            "income_gini": self.eval_env.income_gini,
            "wealth_gini": self.eval_env.wealth_gini,
            "WageRate": self.eval_env.market.WageRate,
            "price": self.eval_env.market.price,
            "total_labor": self.eval_env.market.Lt,
            "house_consumption": self.eval_env.households.consumption,
            "house_work_hours": self.eval_env.households.ht,
            "gov_spending": self.eval_env.main_gov.gov_spending,
            "house_age": self.eval_env.households.age,
            "deposit_rate": self.eval_env.bank.deposit_rate,
            "lending_rate": self.eval_env.bank.lending_rate,
            "real_lending_rate": self.eval_env.bank.real_lending_rate,
            "capital_user_cost": self.eval_env.bank.capital_user_cost,
            "capital_marginal_revenue_product": self.eval_env.market.capital_marginal_revenue_product,
            "target_capital": self.eval_env.bank.target_capital,
            "desired_fixed_investment": self.eval_env.bank.desired_fixed_investment,
            "financed_fixed_investment": self.eval_env.bank.financed_fixed_investment,
            "actual_fixed_investment": self.eval_env.bank.actual_fixed_investment,
            "available_investment_credit": self.eval_env.bank.available_investment_credit,
        }

        if 'central_bank' in self.eval_env.government:
            cb = self.eval_env.government['central_bank']
            self.econ_dict['inflation_rate'] = self.eval_env.inflation_rate
            self.econ_dict['inflation_gap'] = cb.inflation_gap
            self.econ_dict['growth_gap'] = cb.growth_gap
            self.econ_dict['base_interest_rate'] = cb.base_interest_rate
            self.econ_dict['reserve_ratio'] = cb.reserve_ratio
        if 'pension' in self.eval_env.government:
            pension = self.eval_env.government['pension']
            self.econ_dict['retire_age'] = pension.retire_age
            self.econ_dict['contribution_rate'] = pension.contribution_rate
            self.econ_dict['pension_fund'] = pension.pension_fund
            self.econ_dict['old_percent'] = self.eval_env.households.old_percent
            self.econ_dict['dependency_ratio'] = self.eval_env.households.dependency_ratio

    def sum_non_uniform_dict(self, sequences):
        total_sum = 0
        for sublist in sequences:
            if isinstance(sublist, list) or isinstance(sublist, np.ndarray):
                sublist_sum = np.sum(sublist)
                total_sum += sublist_sum
            else:
                raise ValueError("Unsupported data type within the sequence")
        return total_sum

    def mean_non_uniform_dict(self, sequences):
        flat_list = [item for sublist in sequences for item in
                     (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])]
        return np.mean(flat_list)

    def _evaluate_agent(self, write_evaluate_data=False, return_episodes=False):
        eval_econ = ["gov_reward", "tax_gov_reward", "central_bank_gov_reward", "pension_gov_reward",
                     "house_reward", "social_welfare", "per_gdp", "income_gini", "firm_production",
                     "wealth_gini", "years", "GDP", "nominal_GDP", "gov_spending", "house_total_tax", "house_income_tax",
                     "house_wealth_tax", "house_wealth", "house_income", "house_consumption", "house_pension",
                     "house_work_hours", "total_labor", "WageRate", "price", "house_age", "firm_reward", "bank_reward",
                     "deposit_rate", "lending_rate", "real_lending_rate", "capital_user_cost",
                     "capital_marginal_revenue_product", "target_capital",
                     "desired_fixed_investment", "financed_fixed_investment", "actual_fixed_investment",
                     "available_investment_credit"]

        if 'pension' in self.eval_env.government:
            eval_econ += [
                "retire_age",
                "contribution_rate",
                "pension_fund",
                "old_percent",
                "dependency_ratio"
            ]
        if 'central_bank' in self.eval_env.government:
            eval_econ += [
                "inflation_rate", "inflation_gap", "growth_gap", "base_interest_rate", "reserve_ratio"
            ]
        episode_econ_dict = dict(zip(eval_econ, [[] for i in range(len(eval_econ))]))
        # final_econ_dict = dict(zip(eval_econ, [None for i in range(len(eval_econ))]))
        final_econ_dict, episode_results, trajectories, trajectory_summaries = {}, [], [], []

        for epoch_i in range(self.args.eval_episodes):
            obs_dict = self.eval_env.reset()
            self.reset_agent_episodes()
            eval_econ_dict = dict(zip(eval_econ, [[] for i in range(len(eval_econ))]))
            episode_trajectory = []
            initial_gdp = float(self.eval_env.main_gov.initial_GDP)
            previous_gdp = initial_gdp
            t = 0
            while True:
                with torch.no_grad():
                    obs_dict_tensor = self._get_tensor_inputs(obs_dict)
                    action_dict, processed_actions_dict = self.agents_get_action(obs_dict_tensor)
                    expected_inflation = float(self.eval_env.expected_inflation)
                    next_obs_dict, rewards_dict, done = self.eval_env.step(processed_actions_dict, t)
                t += 1
                self.init_economic_dict(rewards_dict)
                snapshot = self.economic_snapshot(
                    period=t,
                    initial_gdp=initial_gdp,
                    previous_gdp=previous_gdp,
                    expected_inflation=expected_inflation,
                )
                episode_trajectory.append(snapshot)
                previous_gdp = snapshot["real_gdp"]

                for each in eval_econ:
                    if "house_" in each or each == "WageRate":
                        eval_econ_dict[each].append(self.econ_dict[each].tolist())
                    else:
                        eval_econ_dict[each].append(self.econ_dict[each])

                obs_dict = next_obs_dict
                if done:
                    break

            for key, value in eval_econ_dict.items():
                if key in {"gov_reward", "bank_reward", "firm_reward"}:
                    episode_econ_dict[key].append(np.sum(value, axis=0))
                elif key in {"GDP", "nominal_GDP"}:
                    # GDP is a per-period flow. Keep the final-period level here;
                    # cumulative and mean GDP are reported under explicit names below.
                    episode_econ_dict[key].append(value[-1])
                elif key == "price" or key == "WageRate" or key == "firm_production":
                    episode_econ_dict[key].append(np.mean(value, axis=0))
                elif key == "years":
                    episode_econ_dict[key].append(np.max(value))
                elif key == "house_reward":
                    episode_econ_dict[key].append(self.sum_non_uniform_dict(value))
                elif "house_" in key and key != "house_reward":
                    episode_econ_dict[key].append(self.mean_non_uniform_dict(value))
                elif key == "age":
                    episode_econ_dict[key].append(value)
                else:
                    episode_econ_dict[key].append(np.mean(value))

            trajectory_summary = summarize_trajectory(episode_trajectory)
            trajectory_summaries.append(trajectory_summary)
            trajectories.append(episode_trajectory)
            episode_results.append({
                "episode": epoch_i + 1,
                "termination_reason": self.eval_env.termination_reason,
                **{key: clean_metric(episode_econ_dict[key][-1]) for key in episode_econ_dict},
                **trajectory_summary,
            })

        for key, value in episode_econ_dict.items():
            value = np.array(value)
            if value.ndim == 3 and value.shape[1] > 1:
                for i in range(value.shape[1]):
                    final_econ_dict[f"{key}_{i}"] = np.mean(value[:, i])
            else:
                final_econ_dict[key] = np.mean(value)

        for key in trajectory_summaries[0]:
            final_econ_dict[key] = float(np.mean([summary[key] for summary in trajectory_summaries]))

        if int(self.econ_dict['years']) > int(self.eva_year_indicator):
            write_evaluate_data = True
            self.eva_year_indicator = self.econ_dict['years']
        elif self.econ_dict['years'] == self.eva_year_indicator:
            gov_return = np.sum(eval_econ_dict['gov_reward'])
            if gov_return > self.eva_reward_indicator:
                write_evaluate_data = True
                self.eva_reward_indicator = copy.deepcopy(gov_return)

        # write_evaluate_data=False
        if write_evaluate_data:
            store_path = "viz/data/"
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            file_name = f"{self.file_name}_data.json"

            file_path = os.path.join(store_path, file_name)
            with open(file_path, "w") as file:
                json.dump(eval_econ_dict, file, cls=NumpyEncoder)

            print("============= Finish Writing================")

        if return_episodes:
            return {
                "episodes": episode_results,
                "aggregate": final_econ_dict,
                "trajectories": trajectories,
            }
        return final_econ_dict
