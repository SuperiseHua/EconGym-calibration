"""Independent stock-flow checks for the existing Ramsey single-firm no-risk route."""


def opening_snapshot(env):
    import numpy as np
    def cp(x): return np.asarray(x, dtype=float).copy()
    h, m, b, g = env.households, env.market, env.bank, env.main_gov
    return {'household_assets': cp(h.at_next), 'savings': cp(h.savings), 'risky_income': cp(h.risky_income),
            'firm_cash': cp(m.cash), 'firm_loans': cp(m.loan_balance), 'book_capital': cp(m.book_capital),
            'bank_equity': float(b.equity), 'government_bonds': float(g.Bt_next),
            'deposit_rate': float(b.last_deposit_rate), 'lending_rate': float(b.last_lending_rate),
            'bond_rate': float(b.government_bond_rate)}


def equation(terms, inequality=False):
    import numpy as np
    values = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in terms])
    if not all(np.all(np.isfinite(v)) for v in values):
        return {'passed': False, 'status': 'NONFINITE'}
    residual = sum(values)
    scale = sum(np.abs(v) for v in values)
    tol = 1e-5+1e-10*scale
    violation = np.maximum(-residual, 0) if inequality else np.abs(residual)
    ratio = violation/tol
    ix = int(np.argmax(ratio))
    return {'passed': bool(np.all(ratio <= 1)), 'relation': 'sum_terms>=0' if inequality else 'sum_terms=0',
            'equations_checked': int(residual.size), 'violations': int(np.sum(ratio > 1)),
            'max_abs_residual': float(np.max(np.abs(residual))), 'max_violation_to_tolerance': float(np.max(ratio)),
            'worst_flat_index': ix, 'worst_signed_terms': [float(v.flat[ix]) for v in values],
            'worst_scale': float(scale.flat[ix]), 'worst_tolerance': float(tol.flat[ix])}


def collect_sectors(env, before, transaction_price):
    import numpy as np
    h, m, b, g = env.households, env.market, env.bank, env.main_gov
    if h.type != 'ramsey' or m.firm_n != 1 or np.any(h.investment_p != 0):
        return {'status': 'UNAVAILABLE', 'reason': 'requires Ramsey single firm zero risky allocation'}
    checks = {}
    def eq(name, *terms, inequality=False): checks[name] = equation(terms, inequality)
    labor = h.e * np.dot(h.ht*h.h_ij_ratio, m.WageRate)
    deposits, hh_loans = np.maximum(h.savings, 0.), np.maximum(-h.savings, 0.)
    household_interest = before['deposit_rate']*np.maximum(before['savings'], 0.) - before['lending_rate']*np.maximum(-before['savings'], 0.)
    gov_purchase = float(np.sum(g.gov_spending*transaction_price))
    eq('household_budget', h.at_next, -before['household_assets'], -h.income,
       h.income_tax, h.asset_tax, h.consumption_expenditure)
    eq('household_income', h.income, -labor, -h.saving_interest, -before['risky_income'])
    eq('household_interest_contract', h.saving_interest, -household_interest)
    eq('household_no_risk_allocation', h.savings, -h.at_next)
    eq('household_consumption_payment', h.consumption_expenditure,
       -np.sum(h.final_consumption*transaction_price, axis=1).reshape(-1, 1), -h.consumption_tax)
    eq('consumption_tax_mapping', h.consumption_tax,
       -np.sum(h.final_consumption*transaction_price, axis=1).reshape(-1, 1)*env.consumption_tax_rate)
    eq('government_budget', g.Bt_next, -before['government_bonds'], -g.bond_interest_expense, -gov_purchase,
       np.sum(h.income_tax), np.sum(h.asset_tax), np.sum(h.consumption_tax), g.estate_tax_revenue,
       g.deceased_direct_tax_revenue, g.deceased_consumption_tax_revenue)
    eq('government_interest_contract', g.bond_interest_expense, -before['bond_rate']*before['government_bonds'])
    eq('firm_cash', m.cash, -before['firm_cash'], -m.sales, m.payroll, m.investment_payment,
       m.interest_paid, m.principal_repaid, -m.new_investment_loan, -m.new_operating_loan)
    eq('firm_loans', m.loan_balance, -before['firm_loans'], m.principal_repaid, -m.new_investment_loan, -m.new_operating_loan)
    eq('firm_book_capital', m.book_capital, -(1-b.depreciation_rate)*before['book_capital'], -m.investment_payment)
    eq('firm_book_balance_sheet', m.book_capital, m.cash, -m.loan_balance, -m.book_equity)
    eq('firm_interest_due', m.interest_paid, m.unpaid_interest, -before['lending_rate']*before['firm_loans'])
    eq('bank_balance_sheet', b.current_account, b.capital_loan, b.household_loans, b.government_bonds,
       -b.total_deposits, -b.firm_deposits, -b.equity)
    eq('bank_profit', b.profit, -b.capital_interest_income, -b.government_interest_payment,
       -b.household_loan_interest_income, b.deposit_interest_expense)
    eq('bank_equity', b.equity, -before['bank_equity'], -b.profit)
    eq('wage_cross_ledger', np.sum(labor), -np.sum(m.payroll))
    eq('firm_interest_cross_ledger', b.capital_interest_income, -np.sum(m.interest_paid))
    eq('household_interest_cross_ledger', np.sum(h.saving_interest), -b.deposit_interest_expense, b.household_loan_interest_income)
    eq('household_deposits_cross_ledger', b.total_deposits, -np.sum(deposits))
    eq('household_loans_cross_ledger', b.household_loans, -np.sum(hh_loans))
    eq('firm_deposits_cross_ledger', b.firm_deposits, -np.sum(m.cash))
    eq('firm_loans_cross_ledger', b.capital_loan, -np.sum(m.loan_balance))
    eq('government_bonds_cross_ledger', b.government_bonds, -g.Bt_next)
    eq('goods_resource_feasibility', m.goods_supply, -h.final_consumption.sum(axis=0)[:, None],
       -g.gov_spending, -b.actual_fixed_investment, inequality=True)
    eq('capital_transition', m.Kt_next, -(1-b.depreciation_rate)*m.Kt, -b.actual_fixed_investment)
    eq('reserve_feasibility', b.current_account, -b.reserve_ratio*(b.total_deposits+b.firm_deposits), inequality=True)
    return {'schema_version': 1, 'status': 'PASS' if all(c['passed'] for c in checks.values()) else 'FAIL',
            'scope': '27_declared_Ramsey_no_risk_stock_flow_equation_families', 'checks': checks,
            'failures': [k for k, v in checks.items() if not v['passed']],
            'full_taskbook_acceptance': False,
            'not_established': ['national_accounts_GDP_mapping', 'micro_distribution_validity',
                                'all_other_scenarios', 'economic_correctness_of_behavior_rules']}
