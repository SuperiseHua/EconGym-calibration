import numpy as np


class MarketRules:
    """
    Rule set for market/firm actions.
    """
    
    @staticmethod
    def get_action(type, obs, action_dim, alpha=0.36,
                   depreciation_rate=0.06, markup=1.0):
        """
        Generate actions for the firms based on the market type and observations.
        Each market type will have different rules for setting prices and wage rates.
        
        Many economic methods require full knowledge of the environment's transition,
        including the consumer demand function, which must be learned from trajectory data in the field of AI.
        No explicit economic rules are provided; please design your own for testing.
        """
        # Retrieve the number of firms from the observation
        firm_n = len(obs)
        
        # Handle different market types
        if type == "perfect":
            return np.random.randn(firm_n, action_dim)   # Firms in perfect competition have no pricing power.
        
        elif type in {"monopoly", "oligopoly", "monopolistic_competition"}:
            return MarketRules._cost_based_action(
                obs,
                alpha=alpha,
                depreciation_rate=depreciation_rate,
                markup=markup,
            )
        else:
            raise ValueError("Unsupported market type.")
    

    
    @staticmethod
    def _cost_based_action(obs, alpha, depreciation_rate, markup):
        """Use Cobb-Douglas unit cost for price and labor's marginal product for wage.

        Observation columns are [K, Z, loan rate, L, previous price,
        previous wage]. Markup is explicit and can be calibrated by market type.
        """
        obs = np.asarray(obs, dtype=float)
        capital = np.clip(obs[:, 0:1], 1e-8, None)
        productivity = np.clip(obs[:, 1:2], 1e-8, None)
        rental_rate = np.clip(obs[:, 2:3] + depreciation_rate, 1e-8, None)
        labor = np.clip(obs[:, 3:4], 1e-8, None)
        previous_wage = np.clip(obs[:, 5:6], 1e-8, None)

        marginal_cost = (
            np.power(rental_rate / alpha, alpha)
            * np.power(previous_wage / (1 - alpha), 1 - alpha)
            / productivity
        )
        price = float(markup) * marginal_cost
        marginal_product_labor = (
            (1 - alpha) * productivity * np.power(capital / labor, alpha)
        )
        wage_rate = price * marginal_product_labor
        return np.hstack([price, wage_rate])
    
