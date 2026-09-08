import copy

from entities.base import BaseEntity
import numpy as np
from gymnasium.spaces import Box


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
        household_savings = np.asarray(custom_cfg['household_savings'], dtype=float)
        government_bonds = float(custom_cfg.get('government_bonds', 0.0))
        self.initial_action = copy.deepcopy(self.entity_args[self.type]['initial_action'])

        self.current_account = float(np.sum(household_savings))
        self.current_loans = 0.0
        self.government_bonds = government_bonds
        self.total_deposits = float(np.maximum(household_savings, 0).sum())
        self.household_loans = float(np.maximum(-household_savings, 0).sum())
        self.total_account = self.total_deposits  # Backward-compatible observation name.

        self.deposit_rate = self.entity_args['params'].deposit_rate
        self.lending_rate = self.entity_args['params'].lending_rate
        self.reserve_ratio = self.entity_args['params'].reserve_ratio
        self.base_interest_rate = self.entity_args['params'].base_interest_rate
        self.last_deposit_rate = copy.copy(self.deposit_rate)
        self.last_lending_rate = copy.copy(self.lending_rate)
        self.government_bond_rate = float(custom_cfg['government_bond_rate'])
        self.government_interest_payment = 0.0
        self.deposit_interest_expense = 0.0
        self.household_loan_interest_income = 0.0
        self.capital_interest_income = 0.0
        self.household_interest = np.zeros_like(household_savings)
        self.profit = 0.0
        self.capital_loan = 0.0
        self.real_lending_rate = 0.0
        self.capital_user_cost = 0.0
        self.available_investment_credit = 0.0
        self.target_capital = 0.0
        self.desired_fixed_investment = 0.0
        self.financed_fixed_investment = 0.0
        self.actual_fixed_investment = 0.0
        self.equity = 0.0
        self.firm_deposits = 0.0
        self.accounting_failure = None
        # self.last_lending_rate_j = copy.copy(self.lending_rate)

    def get_action(self, actions, central_bank_exist=False):
        if self.type == 'non_profit':
            # A non-profit bank is a passive policy-rate benchmark. Capital returns
            # are market outcomes and must not overwrite these financing rates.
            self.lending_rate = self.base_interest_rate
            self.deposit_rate = self.base_interest_rate
        elif self.type == 'commercial':
            # For commercial banks, actions are lending rate and deposit rate
            self.lending_rate, self.deposit_rate = actions
            if central_bank_exist:
                self.lending_rate = np.clip(self.lending_rate, self.base_interest_rate + 0.01, self.base_interest_rate + 0.03)
                self.deposit_rate = np.clip(self.deposit_rate, self.base_interest_rate - 0.01, self.base_interest_rate)


    def prepare_settlement(self, society):
        """Accrue opening balances at previously contracted rates for this interval.

        reset represents time 0; the first step settles time 0 -> 1. Funds saved
        at the end of this step earn interest only during the following step.
        Snapshot before OLG exits/entries so every opening account is settled.
        """
        balances = np.asarray(society.households.savings, dtype=float)
        deposits = np.maximum(balances, 0.0)
        loans = np.maximum(-balances, 0.0)
        self.opening_household_balance = float(balances.sum())
        self.household_interest = (
            self.last_deposit_rate * deposits - self.last_lending_rate * loans
        )
        self.deposit_interest_expense = float((self.last_deposit_rate * deposits).sum())
        self.household_loan_interest_income = float((self.last_lending_rate * loans).sum())
        self.capital_interest_income = self.last_lending_rate * self.capital_loan
        self.government_interest_payment = (
            self.government_bond_rate * float(society.main_gov.Bt_next)
        )

    def step(self, society):
        # Retrieve the first government agent from the society's government dictionary
        self.gov_agent = society.main_gov

        # Only a central-bank authority can update monetary-policy settings.
        # Tax or pension authorities must not overwrite the bank's configured base rate.
        if "central_bank" in society.government:
            central_bank = society.government["central_bank"]
            self.reserve_ratio = central_bank.reserve_ratio
            self.base_interest_rate = central_bank.base_interest_rate

        if getattr(self, 'firm_accounting', False):
            self.settle_firms(society)
            return
                
        if society.step_cnt == 0:
            self.current_account -= self.gov_agent.Bt

        # Same principals/interest as the household and government ledgers.
        previous_settlement = (
            -self.opening_household_balance - self.deposit_interest_expense
            + self.household_loan_interest_income
            + self.capital_loan + self.capital_interest_income
            + self.gov_agent.Bt + self.government_interest_payment
        )

        current_balances = society.households.savings
        current_deposit = float(np.sum(current_balances))  # Net end-of-period bank balances.

        total_deposit = self.current_account + previous_settlement + current_deposit

        society.market.Kt_next = self.compute_next_kt(society, total_deposit)

        capital_loan = np.sum(society.market.price * self.actual_fixed_investment)
        current_loan = capital_loan + self.gov_agent.Bt_next  # Nominal loans issued

        self.profit = (self.capital_interest_income + self.government_interest_payment
                       + self.household_loan_interest_income - self.deposit_interest_expense)

        self.current_account += previous_settlement + current_deposit - current_loan  # Current account balance
        self.capital_loan = capital_loan
        self.current_loans = float(capital_loan)
        self.government_bonds = float(self.gov_agent.Bt_next)
        self.total_deposits = float(np.maximum(current_balances, 0.0).sum())
        self.household_loans = float(np.maximum(-current_balances, 0.0).sum())
        self.total_account = self.total_deposits

        self.last_deposit_rate = copy.copy(self.deposit_rate)
        self.last_lending_rate = copy.copy(self.lending_rate)
        # One-period bonds: today's policy rate prices the new period's debt.
        self.government_bond_rate = float(self.base_interest_rate)

    @staticmethod
    def compute_real_lending_rate(nominal_rate, expected_inflation):
        expected_inflation = max(float(expected_inflation), -1 + 1e-8)
        return (1 + float(nominal_rate)) / (1 + expected_inflation) - 1

    @staticmethod
    def compute_target_capital(alpha, productivity, labor, user_cost):
        labor = np.maximum(np.asarray(labor, dtype=float), 1e-8)
        return np.power(alpha * productivity * np.power(labor, 1 - alpha) / user_cost,
                        1 / (1 - alpha))

    def compute_next_kt(self, society, total_deposit):
        market = society.market
        self.real_lending_rate = self.compute_real_lending_rate(
            self.lending_rate, society.expected_inflation
        )
        self.capital_user_cost = self.real_lending_rate + self.depreciation_rate
        speed = float(getattr(self, 'capital_adjustment_speed', 0.2))
        growth = float(getattr(self, 'max_net_capital_growth', 0.05))
        if not (0 < speed <= 1 and np.isfinite(growth) and growth >= 0
                and 0 <= self.depreciation_rate < 1):
            raise ValueError('Invalid capital adjustment, growth bound, or depreciation.')
        # A bounded decision needs no infinite unconstrained optimum. First test
        # the marginal product at the largest target that can affect this step.
        upper_target = market.Kt * (1 + growth / speed)
        upper_mpk = (market.alpha * market.Zt
                     * np.power(market.firm_labor_j, 1 - market.alpha)
                     * np.power(upper_target, market.alpha - 1))
        self.target_capital = self.compute_target_capital(
            market.alpha, market.Zt, market.firm_labor_j,
            np.maximum(self.capital_user_cost, upper_mpk + 1e-12))
        self.target_capital = np.minimum(self.target_capital, upper_target)
        self.target_at_capacity = self.capital_user_cost <= upper_mpk
        self.target_capital = np.where(self.target_at_capacity, upper_target, self.target_capital)

        surviving_capital = (1 - self.depreciation_rate) * market.Kt
        self.desired_fixed_investment = np.clip(
            self.depreciation_rate * market.Kt + speed * (self.target_capital - market.Kt),
            0.0, (self.depreciation_rate + growth) * market.Kt)
        self.available_investment_credit = max(
            float(total_deposit) * (1 - self.reserve_ratio) - float(self.gov_agent.Bt_next), 0.0
        )
        desired_credit = float(np.sum(market.price * self.desired_fixed_investment))
        if getattr(self, 'firm_accounting', False):
            # One consolidated bank: a new loan creates a matching firm deposit.
            # A zero reserve requirement does not imply a deposits-based loan cap.
            # Production capacity bounds the request; no external reserve funding.
            self.available_investment_credit = self.credit_capacity(desired_credit, total_deposit)
        credit_scale = min(1.0, self.available_investment_credit / (desired_credit + 1e-8))
        self.financed_fixed_investment = self.desired_fixed_investment * credit_scale

        consumption = society.households.final_consumption.sum(axis=0)[:, np.newaxis]
        goods_left = np.maximum(market.goods_supply - self.gov_agent.gov_spending - consumption, 0.0)
        self.actual_fixed_investment = np.minimum(self.financed_fixed_investment, goods_left)
        return surviving_capital + self.actual_fixed_investment

    def initialize_firm_accounts(self, society):
        """Finance opening physical capital through the same bank as household wealth."""
        if not getattr(self, 'firm_accounting', False):
            return
        if society.households.type != 'ramsey':
            raise ValueError('Shared firm ledger currently supports the Ramsey no-risk baseline.')
        m = society.market
        m.loan_balance = m.price * m.Kt
        m.cash = np.zeros_like(m.Kt)  # Non-interest-bearing deposits at this bank.
        m.book_capital = m.loan_balance.copy()
        m.book_equity = np.zeros_like(m.Kt)
        self.capital_loan = float(m.loan_balance.sum())
        self.current_loans = self.capital_loan
        self.current_account -= self.government_bonds + self.capital_loan
        if self.current_account < -1e-5:
            raise ValueError('Initial capital and bonds exceed net household funding; reduce initial K/Y.')
        self.current_account = max(self.current_account, 0.0)
        self.balance_sheet_residual = 0.0
        self.reserve_shortfall = 0.0

    def credit_capacity(self, requested, deposits):
        """Finite commitment under a fixed-reserve, single-bank simplification."""
        if not 0 <= self.reserve_ratio <= 1:
            raise ValueError('reserve_ratio must be in [0, 1].')
        if self.reserve_ratio == 0:
            return max(float(requested), 0.0)
        room = self.current_account / self.reserve_ratio - max(float(deposits), 0.0)
        return min(max(float(requested), 0.0), max(room, 0.0))

    def settle_firms(self, society):
        """One cash settlement shared by firms and bank; loans are persistent stocks.

        Investment purchases use each firm's modeled good as capital. Temporary
        operating deficits require explicit new loans. Interest is paid only from
        cash, and unpaid principal rolls over under the amortizing credit line.
        """
        m, h, g = society.market, society.households, society.main_gov
        old_cash, old_loans = m.cash.copy(), m.loan_balance.copy()
        deposits = float(np.maximum(h.savings, 0).sum())
        m.Kt_next = self.compute_next_kt(society, deposits + self.firm_deposits)
        m.investment_payment = m.price * self.actual_fixed_investment
        m.sales = m.price * (h.final_consumption.sum(axis=0)[:, None]
                            + g.gov_spending + self.actual_fixed_investment)
        m.payroll = m.WageRate * m.firm_labor_j
        m.new_investment_loan = m.investment_payment.copy()
        cash = old_cash + m.sales - m.payroll  # Investment loan and purchase cancel.
        interest_due = self.last_lending_rate * old_loans
        operating_need = np.maximum(interest_due - cash, 0.0)
        granted = self.credit_capacity(float(operating_need.sum()),
                                       deposits + float(np.maximum(cash, 0).sum()))
        m.new_operating_loan = operating_need * min(1.0, granted / (float(operating_need.sum()) + 1e-8))
        cash += m.new_operating_loan
        # Negative loan rates are signed payments from bank to firm.
        m.interest_paid = np.minimum(interest_due, np.maximum(cash, 0.0))
        m.unpaid_interest = interest_due - m.interest_paid
        cash -= m.interest_paid
        repayment_rate = float(getattr(self, 'principal_repayment_rate', 0.05))
        if not 0 <= repayment_rate <= 1:
            raise ValueError('principal_repayment_rate must be in [0, 1].')
        m.principal_due = repayment_rate * old_loans
        m.principal_repaid = np.minimum(m.principal_due, np.maximum(cash, 0.0))
        m.principal_rolled_over = m.principal_due - m.principal_repaid
        m.cash = cash - m.principal_repaid
        m.loan_balance = old_loans - m.principal_repaid + m.new_investment_loan + m.new_operating_loan
        m.cash_residual = (m.cash - old_cash - m.sales + m.payroll + m.investment_payment
                           + m.interest_paid + m.principal_repaid
                           - m.new_investment_loan - m.new_operating_loan)
        m.loan_residual = m.loan_balance - old_loans + m.principal_repaid - m.new_investment_loan - m.new_operating_loan
        m.book_capital = (1 - self.depreciation_rate) * m.book_capital + m.investment_payment
        m.book_equity = m.book_capital + m.cash - m.loan_balance
        self.capital_interest_income = float(m.interest_paid.sum())
        self.profit = (self.capital_interest_income + self.government_interest_payment
                       + self.household_loan_interest_income - self.deposit_interest_expense)
        self.equity += self.profit
        self.capital_loan = self.current_loans = float(m.loan_balance.sum())
        self.firm_deposits = float(m.cash.sum())
        self.government_bonds = float(g.Bt_next)
        self.total_deposits = self.total_account = deposits  # Existing household observation.
        self.household_loans = float(np.maximum(-h.savings, 0.0).sum())
        self.balance_sheet_residual = (self.current_account + self.capital_loan + self.household_loans
                                      + self.government_bonds - deposits - self.firm_deposits - self.equity)
        self.reserve_shortfall = max(self.reserve_ratio * (deposits + self.firm_deposits)
                                     - self.current_account, 0.0)
        self.accounting_failure = None
        if np.any(m.cash < -1e-5) or np.any(m.unpaid_interest > 1e-5):
            self.accounting_failure = 'unfunded_firm_payment'
        elif self.reserve_shortfall > 1e-5:
            self.accounting_failure = 'reserve_shortfall_no_external_funding'
        elif abs(self.balance_sheet_residual) > 1e-5 * max(1, deposits / 1e8):
            self.accounting_failure = 'bank_balance_sheet_mismatch'
        self.last_deposit_rate = copy.copy(self.deposit_rate)
        self.last_lending_rate = copy.copy(self.lending_rate)
        self.government_bond_rate = float(self.base_interest_rate)

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
        return self.accounting_failure is not None
