import copy
import numpy as np
from gymnasium.spaces import Box
from pathlib import Path
ROOT_PATH = str(Path(__file__).resolve().parent.parent)
from entities.households import Household
from entities.government import Government
from entities.market import Market
from entities.bank import Bank
from .set_observation import EconObservations


class EconomicSociety:
    def __init__(self, cfg):
        super().__init__()
        self.__dict__.update(cfg['env_core'])  # update cfg to self
        self.agents = {'households': None, 'government': {}, 'market': None, 'bank': None}

        for entity_arg in cfg['Entities']:
            entity_name = entity_arg['entity_name']
            entity_args = entity_arg['entity_args']
    
            # Add new entity instance to the appropriate list in agents
            if entity_name == 'households':
                self.agents[entity_name] = Household(entity_args)
            elif entity_name == 'government':
                type = entity_args.params.type
                self.agents[entity_name][type] = Government(entity_args)
            elif entity_name == 'market':
                self.agents[entity_name] = Market(entity_args)
            elif entity_name == 'bank':
                self.agents[entity_name] = Bank(entity_args)

        self.households = self.agents['households']
        self.government = self.agents['government']
        self.market = self.agents['market']
        self.bank = self.agents['bank']
        
        self.fiscal_gov = (
            self.government.get('tax')
            or self.government.get('pension')
            or self.government.get('central_bank')
        )
        self.main_gov = self.fiscal_gov  # Backward-compatible alias.
        
        observations_dict = self.reset()

        for name, entity in self.agents.items():
            obs = observations_dict[name]  # Get the corresponding observations for the entity
            # Check if the entity is a dict (like for government agents)
            if isinstance(entity, dict):
                for sub_type, sub_entity in entity.items():
                    sub_entity.observation_space = Box(
                        low=-np.inf, high=np.inf, shape=(obs[sub_type].shape[-1],), dtype=np.float32
                    )
            else:
                # If the entity is not a dict, directly assign the observation space
                entity.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs.shape[-1],), dtype=np.float32 )
        
        # Expand the action_space based on the number of firms.
        if self.market.firm_n > 1:
            self.expand_action_space(firm_n=self.market.firm_n)
        
        self.display_mode = False
        # Capture only after spaces and initial actions have their final dimensions.
        self.reset()
        self._initial_state = self._capture_initial_state()

    def _entity_records(self):
        return {
            'households': self.households, 'market': self.market, 'bank': self.bank,
            **{f'government.{name}': gov for name, gov in self.government.items()},
        }

    _entity_reference_names = (
        'agents', 'households', 'market', 'bank', 'government', 'fiscal_gov', 'main_gov',
    )

    def _capture_initial_state(self):
        return copy.deepcopy({
            'environment': {key: value for key, value in self.__dict__.items()
                            if key not in self._entity_reference_names and key != '_initial_state'},
            'entities': {name: entity.__dict__ for name, entity in self._entity_records().items()},
        })

    def _restore_initial_state(self):
        # Preserve entity identities: policies may hold references to these objects.
        initial_state = self._initial_state
        references = {name: getattr(self, name) for name in self._entity_reference_names}
        restored = copy.deepcopy(initial_state)
        for name, entity in self._entity_records().items():
            entity.__dict__.clear()
            entity.__dict__.update(restored['entities'][name])
        self.__dict__.clear()
        self.__dict__.update(restored['environment'])
        self.__dict__.update(references)
        self._initial_state = initial_state

    def set_tax_type(self, tax_type):
        """Persist an explicitly selected fiscal algorithm across episode resets."""
        from agents.saez import SaezGovernment
        gov = self.government['tax']
        gov.tax_type = tax_type
        initial_gov = self._initial_state['entities']['government.tax']
        initial_gov['tax_type'] = tax_type
        if tax_type == 'saez':
            gov.saez_gov = SaezGovernment()
            initial_gov['saez_gov'] = copy.deepcopy(gov.saez_gov)
        else:
            gov.__dict__.pop('saez_gov', None)
            initial_gov.pop('saez_gov', None)

    def expand_action_space(self, firm_n):
        '''
        Expand action space for different economic roles:
        - For households:
            1) Select the index of the firm they work for. (array(1) in {0,..., firm_n-1})
            2) Choose the consumption amount for each of the firm_n firms' goods. (array([firm_n,]))

        - For fiscal government (type="tax"):
            Add government spending action for each firm:
            1) Gt_prob_j represents the proportion of government spending allocated to firm j.

        - Other economic roles' actions remain unaffected.
        '''
        # Expand action dimension for households:
        # 1: Firm selection (scalar), firm_n: Consumption distribution over firms.
        self.households.action_dim += 1 + firm_n
    
        N = self.households.households_n
        new_shape = (N, self.households.action_dim)
        
        # Update action space for households
        action_max = float(self.households.action_space.high.max())
        action_min = float(self.households.action_space.low.min())
        self.households.action_space = Box(low=action_min, high=action_max, shape=new_shape, dtype=np.float32)
    
        # For fiscal government (type="tax"), expand the action space to include spending proportions for each firm
        if "tax" in self.government:
            action_max = float(self.government['tax'].action_space.high.max())
            action_min = float(self.government['tax'].action_space.low.min())
            self.government['tax'].action_dim += firm_n
            self.government['tax'].action_space = Box(low=action_min, high=action_max, shape=(self.government['tax'].action_dim,),
                                                      dtype=np.float32)

    @property
    def action_spaces(self):
        """Return a dictionary of action spaces for each agent."""
        action_spaces = {
            self.households.name: self.households.action_space,
            self.market.name: self.market.action_space,
            self.bank.name: self.bank.action_space
        }
    
        # Add action spaces for all government agents, using type as a unique identifier
        for gov_type, gov_agent in self.government.items():
            action_spaces[f"{gov_agent.name}_{gov_type}"] = gov_agent.action_space
    
        return action_spaces

    @property
    def observation_spaces(self):
        """Return a dictionary of observation spaces for each agent."""
        observation_spaces = {
            self.households.name: self.households.observation_space,
            self.market.name: self.market.observation_space,
            self.bank.name: self.bank.observation_space
        }
    
        # Add observation spaces for all government agents, using type as a unique identifier
        for gov_type, gov_agent in self.government.items():
            observation_spaces[f"{gov_agent.name}_{gov_type}"] = gov_agent.observation_space
    
        return observation_spaces

    def action_wrapper(self, action_dict):
        processed_action_dict = {}
    
        for agent_name, agent_action in action_dict.items():
            agent = self.agents[agent_name]

            if isinstance(agent, dict):
                processed_sub_agents = {}
                for sub_agent_name, sub_agent in agent.items():
                    processed_sub_agents[sub_agent_name] = self.check_agent_action(sub_agent, agent_action[sub_agent_name], sub_agent_name)
                processed_action_dict[agent_name] = processed_sub_agents
        
            else:
                processed_action_dict[agent_name] = self.check_agent_action(agent, agent_action, agent_name)
    
        return processed_action_dict

    @staticmethod
    def check_agent_action(agent, agent_action, agent_name):
        """
        Check the agent's actions.
        1. Verify if the dimensions of the provided actions match the dimensions set by the environment.
        2. Ensure the values of the provided actions are within the range defined by the environment.
        """
        
        expected_dim = agent.action_dim
        current_shape = getattr(agent_action, 'shape', (0,))
        current_action_dim = current_shape[-1]
        
        if current_action_dim == 0 and expected_dim == 0:
            return None
        elif current_action_dim != expected_dim:
            raise ValueError(
                f"Invalid actions for {agent_name}. Expected shape: {expected_dim}, Found: {current_action_dim}"
            )

        expected_shape = getattr(agent.action_space, 'shape', None)
        if expected_shape is not None and tuple(current_shape) != tuple(expected_shape):
            raise ValueError(
                f"Invalid actions for {agent_name}. Expected shape: {expected_shape}, Found: {current_shape}"
            )
    
        expected_action_min = agent.real_action_min
        expected_action_max = agent.real_action_max
        
        # When the number of firms (firm_n > 1) increases, causing the action dimension of households and tax_gov to expand, we supplement expected_action_min/max.
        fill_in_len = expected_dim - len(expected_action_min)

        if fill_in_len > 0:
            expected_action_min = np.pad(expected_action_min, (0, fill_in_len), constant_values=0)
            expected_action_max = np.pad(expected_action_max, (0, fill_in_len), constant_values=1)

        return np.clip(agent_action, expected_action_min, expected_action_max)

    def get_actions(self, action_dict):
        """Get and process actions for all agents."""
        valid_action_dict = self.is_valid(action_dict)
        processed_action_dict = self.action_wrapper(valid_action_dict)
        if isinstance(self.government, dict):
            for gov_type, gov_agent in self.government.items():
                gov_agent.get_action(processed_action_dict[self.government[gov_type].name][gov_type], firm_n=self.market.firm_n)

        if "central_bank" in self.government:
            central_bank = self.government["central_bank"]
            self.bank.base_interest_rate = central_bank.base_interest_rate
            self.bank.reserve_ratio = central_bank.reserve_ratio
        self.bank.get_action(processed_action_dict[self.bank.name], central_bank_exist=("central_bank" in self.government))
        self.market.get_action(processed_action_dict[self.market.name])
        self.households.get_action(
            processed_action_dict[self.households.name],
            firm_n=self.market.firm_n,
            update_efficiency=self.step_cnt > 0,
        )

    def step(self, action_dict, t=None):
        """Perform a simulation step given the actions."""
    
        # === Phase 1: Agents Take Action ===
        self.get_actions(action_dict)
        self.bank.prepare_settlement(self)
        if "OLG" in self.households.type:
            self.households.update_retirement_status(self.main_gov.retire_age)
    
        # === Phase 2: Entities Step Forward ===
        self.market.step(self)
        # No inventory carry-over: only current production can be traded this period.
        self.market.goods_supply = self.market.Yt_j
        self.main_gov.plan_spending(self)
        self.households.step(self, t)
        self.update_governments()
        self.bank.step(self)
    
        # === Phase 3: Update Environment State ===
        self.update_metrics()

        if self.market.type == "perfect":
            # Current planned demand determines the next year's competitive-market price.
            self.market.planned_demand = (
                self.households.planned_consumption_demand
                + self.main_gov.gov_spending_d
                + self.bank.financed_fixed_investment
            )
            demand, supply = self.market.planned_demand, self.market.goods_supply
            if (np.any(supply == 0) and np.all(np.isfinite(supply))
                    and np.all(supply >= 0) and np.all(np.isfinite(demand))
                    and np.all(demand >= 0)):
                # Settlement is complete, but D/S cannot define a next-period price.
                # Keep actual zero output and this period's transaction price.
                self.termination_reason = 'zero_goods_supply'
            else:
                self.market.update_price(
                    demand, supply,
                    adjustment_speed=getattr(self, "price_adjustment_speed", 0.2),
                )
        self.step_cnt += 1
        self.done = self.is_terminal()
    
        # === Phase 4: Observation & NaN Check ===
        next_obs = EconObservations(self).get_obs()

        # === Phase 5: Final Updates ===
        self.last_prices = self.consumer_prices.copy()
        self.last_consumption = self.consumption_quantities.copy()

        return (
            next_obs,
            self.rewards,
            self.done,
        )

    def update_governments(self):
        """Update the shared fiscal ledger once and synchronize other policy authorities."""
        self.fiscal_gov.step(self)
        for gov_agent in self.government.values():
            if gov_agent is self.fiscal_gov:
                continue
            gov_agent.old_GDP = copy.copy(gov_agent.GDP)
            gov_agent.GDP = copy.copy(self.fiscal_gov.GDP)
            gov_agent.real_GDP = copy.copy(self.fiscal_gov.real_GDP)
            gov_agent.nominal_GDP = copy.copy(self.fiscal_gov.nominal_GDP)
            gov_agent.per_household_gdp = copy.copy(self.fiscal_gov.per_household_gdp)
            gov_agent.Bt = copy.copy(self.fiscal_gov.Bt)
            gov_agent.Bt_next = copy.copy(self.fiscal_gov.Bt_next)
            if gov_agent.type == 'pension' and 'OLG' in self.households.type:
                gov_agent.pension_step(self)

    def update_metrics(self):
        """Update evaluation metrics such as Gini coefficients, price index, and rewards."""
        # Compute Gini coefficients
        self.wealth_gini = self.gini_coef(self.households.post_asset)
        self.income_gini = self.gini_coef(self.households.post_income)

        # Annual PCE-style chain Fisher index based only on consumer transactions.
        self.consumer_prices = self.market.price * (1 + self.consumption_tax_rate)
        self.consumption_quantities = self.households.final_consumption.sum(axis=0)
        self.inflation_rate, self.price_index = self.market.compute_inflation_rate(
            self.consumer_prices, self.consumption_quantities,
            self.last_prices, self.last_consumption, self.price_index,
        )
        self.expected_inflation = self.update_expected_inflation(
            self.inflation_rate,
            self.expected_inflation,
            self.inflation_expectation_lambda,
        )
        
        self.real_deals = (
            self.households.final_consumption.sum(axis=0)[:, np.newaxis]
            + self.main_gov.gov_spending
            + self.bank.actual_fixed_investment
        )
        
        households_reward = self.households.get_reward()
        
        if isinstance(self.government, dict):
            government_reward = {}
            for gov_type, gov_agent in self.government.items():
                government_reward[gov_type] = gov_agent.get_reward(self)
        else:
            government_reward = self.government.get_reward(self)

        firm_reward = self.market.get_reward(self)
        bank_reward = self.bank.get_reward()

        self.rewards = {
            gov_agent.name: government_reward,
            self.households.name: households_reward,
            self.market.name: firm_reward,
            self.bank.name: bank_reward,
        }

    @staticmethod
    def update_expected_inflation(realized_inflation, previous_expectation, weight):
        """Form next-period expectations from last inflation and prior expectations."""
        weight = float(weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("inflation_expectation_lambda must be in [0, 1].")
        return weight * float(realized_inflation) + (1.0 - weight) * float(previous_expectation)
        
    def reset(self, **custom_cfg):
        """Restore this configuration's initial sample; optional seed controls shocks.

        Create a new environment for a different calibration configuration. Reset
        clears all episode state, including attributes first created during step.
        """
        if 'seed' in custom_cfg:
            np.random.seed(custom_cfg['seed'])
        if hasattr(self, '_initial_state'):
            self._restore_initial_state()
            return EconObservations(self).get_obs()
        self.step_cnt = 0
        # Restore the population before scaling fiscal aggregates (OLG changes N).
        self.households.reset()
        for _, gov_agent in self.government.items():
            gov_agent.reset(
                households_n=self.households.households_n,
                firm_n=self.market.firm_n,
                household_type=self.households.type,
                real_household_units=self.households.real_unit_count,
            )
        
        if "OLG" in self.households.type:
            self.households.update_retirement_status(self.main_gov.retire_age)
        self.bank.reset(
            household_savings=self.households.savings,
            government_bonds=self.main_gov.Bt,
            government_bond_rate=(self.government['central_bank'].base_interest_rate
                                  if 'central_bank' in self.government
                                  else self.bank.entity_args.params.base_interest_rate),
        )
        if "central_bank" in self.government:
            central_bank = self.government["central_bank"]
            self.bank.base_interest_rate = central_bank.base_interest_rate
            self.bank.reserve_ratio = central_bank.reserve_ratio

        self.market.reset(
            GDP=self.main_gov.GDP,
            scale_factor=self.main_gov.scale_factor,
            households_at=self.households.at,
            real_debt_rate=self.main_gov.real_debt_rate,
        )
        self.households.align_initial_effective_labor(self.market.Lt)
        self.bank.initialize_firm_accounts(self)

        self.price_index = 100.0
        self.inflation_rate = 0.02
        self.expected_inflation = self.inflation_rate
        self.inflation_expectation_lambda = float(
            getattr(self, "inflation_expectation_lambda", 0.5)
        )
        if not 0.0 <= self.inflation_expectation_lambda <= 1.0:
            raise ValueError("inflation_expectation_lambda must be in [0, 1].")
        self.last_prices = self.market.price * (1 + self.consumption_tax_rate)
        self.last_consumption = None
        self.ini_income_gini = self.gini_coef(self.households.income)
        self.ini_wealth_gini = self.gini_coef(self.households.at)
        self.done = False
        self.termination_reason = None
        self.display_mode = False

        return EconObservations(self).get_obs()

    def is_terminal(self):
        """
        Check if the simulation has reached a terminal state.
        """

        gini_invalid = self.wealth_gini >= 1 or self.income_gini >= 1 or np.isnan(self.wealth_gini) or np.isnan(self.income_gini)
        data_nan = any(self.recursive_decompose_dict(self.rewards, lambda a: np.isnan(a)))
        
        episode_completed = self.step_cnt >= self.episode_length
        agent_terminal = any(self.recursive_decompose_dict(self.agents, lambda a: a.is_terminal()))
        # if (gini_invalid or data_nan or episode_completed or agent_terminal) and self.step_cnt == 2:
        #     print(self.recursive_decompose_dict(self.agents, lambda a: a.is_terminal()))
        #     print(1)
        
        return (self.termination_reason is not None or gini_invalid or data_nan
                or episode_completed or agent_terminal)

    def recursive_decompose_dict(self, input_dict, func):
        
        results = []
        if isinstance(input_dict, dict):
            for key_i in input_dict:
                current_item = input_dict[key_i]
                if isinstance(current_item, dict):
                    for key_j in current_item:
                        results.append(func(current_item[key_j]))
                elif isinstance(current_item, np.ndarray):
                    current_result = func(current_item).any()
                    results.append(current_result)
                else:
                    results.append(func(current_item))
        return results

    def is_valid(self, action_dict):
        """Validate the actions provided by the agents."""
        expected_agents = set(self.agents.keys())
        received_agents = set(action_dict.keys())
        if expected_agents == received_agents:
            return action_dict
        else:
            raise ValueError(
                "Invalid actions. Expected agents: {}, Received agents: {}".format(expected_agents, received_agents))

    def gini_coef(self, values):
        """Finite-sample normalized Gini supporting negative income or wealth."""
        values = np.asarray(values, dtype=float).reshape(-1)
        n = values.size
        if n <= 1:
            return 0.
        if not np.all(np.isfinite(values)):
            return np.nan

        absolute_total = np.sum(np.abs(values))
        if np.isclose(absolute_total, 0.0):
            return 0.

        values = np.sort(values)
        ranks = np.arange(1, n + 1)
        pairwise_difference_sum = np.dot(2 * ranks - n - 1, values)
        gini = pairwise_difference_sum / ((n - 1) * absolute_total)
        return float(np.clip(gini, 0.0, 1.0))


    def render(self):
        pass

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
