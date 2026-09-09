"""Read-only, single-firm investment transmission diagnostics, no model mutation."""
import math


def diagnose_investment(s):
    required = {'capital_open', 'capital_next', 'alpha', 'productivity', 'labor', 'depreciation_rate',
                'adjustment_speed', 'max_net_growth', 'nominal_lending_rate', 'expected_inflation_used',
                'real_lending_rate', 'capital_user_cost', 'target_capital', 'desired_investment',
                'financed_investment', 'actual_investment', 'available_credit_nominal',
                'transaction_price', 'goods_supply', 'household_consumption_real', 'government_purchases_real'}
    if set(s) != required or any(type(v) not in (int, float) or not math.isfinite(v) for v in s.values()):
        raise ValueError('complete finite scalar investment snapshot required')
    k, delta, speed, growth = s['capital_open'], s['depreciation_rate'], s['adjustment_speed'], s['max_net_growth']
    if not (k > 0 and 0 <= delta < 1 and 0 < speed <= 1 and growth >= 0 and
            0 < s['alpha'] < 1 and s['productivity'] > 0 and s['labor'] >= 0 and s['transaction_price'] > 0):
        raise ValueError('invalid single-firm structural domain')
    upper = k*(1+growth/speed)
    upper_mpk = s['alpha']*s['productivity']*s['labor']**(1-s['alpha'])*upper**(s['alpha']-1)
    rate = (1+s['nominal_lending_rate'])/(1+max(s['expected_inflation_used'], -1+1e-8))-1
    cost = rate+delta
    target = upper if cost <= upper_mpk else (s['alpha']*s['productivity']*
        max(s['labor'], 1e-8)**(1-s['alpha'])/max(cost, upper_mpk+1e-12))**(1/(1-s['alpha']))
    target = min(target, upper)
    raw_desired = delta*k+speed*(s['target_capital']-k)
    desired = min(max(raw_desired, 0.), (delta+growth)*k)
    desired_credit = s['transaction_price']*s['desired_investment']
    credit_scale = min(1., s['available_credit_nominal']/(desired_credit+1e-8))
    financed = s['desired_investment']*credit_scale
    goods_left = max(s['goods_supply']-s['government_purchases_real']-s['household_consumption_real'], 0.)
    actual = min(s['financed_investment'], goods_left)
    residuals = {}
    def residual(name, signed_terms):
        value = sum(signed_terms)
        scale = sum(abs(t) for t in signed_terms)
        tolerance = 1e-5+1e-10*scale
        residuals[name] = {'residual': value, 'scale': scale, 'tolerance': tolerance, 'passed': abs(value) <= tolerance}
    residual('real_rate_mapping', [s['real_lending_rate'], -rate])
    residual('user_cost_mapping', [s['capital_user_cost'], -s['real_lending_rate'], -delta])
    residual('bounded_target_mapping', [s['target_capital'], -target])
    residual('desired_investment_mapping', [s['desired_investment'], -desired])
    residual('credit_rationing_mapping', [s['financed_investment'], -financed])
    residual('goods_rationing_mapping', [s['actual_investment'], -actual])
    residual('capital_transition', [s['capital_next'], -(1-delta)*k, -s['actual_investment']])
    real_tolerance = 1e-5+1e-10*sum(abs(s[name]) for name in
        ('desired_investment', 'financed_investment', 'actual_investment', 'goods_supply'))
    # Tag numerical boundary contact robustly; this does not change upstream clipping
    # or any acceptance tolerance. Algebraically equal cap expressions can differ by ulps.
    boundary_tolerance = 1e-10+1e-12*max(k, abs(raw_desired), abs((delta+growth)*k))
    flags = {'target_cap_active': cost <= upper_mpk,
             'investment_zero_floor_active': raw_desired <= boundary_tolerance,
             'investment_growth_cap_active': raw_desired >= (delta+growth)*k-boundary_tolerance,
             'credit_rationing_material': s['desired_investment']-s['financed_investment'] > real_tolerance,
             'goods_rationing_material': s['financed_investment']-s['actual_investment'] > real_tolerance}
    ordering = (-real_tolerance <= s['actual_investment'] <= s['financed_investment']+real_tolerance and
                -real_tolerance <= s['financed_investment'] <= s['desired_investment']+real_tolerance and
                s['available_credit_nominal'] >= 0)
    return {'schema_version': 1, 'status': 'PASS' if ordering and all(r['passed'] for r in residuals.values()) else 'FAIL',
            'scope': 'single_firm_read_only_implementation_consistency_not_full_economic_accounting',
            'inputs': dict(s), 'derived': {'target_capital_upper': upper, 'marginal_product_at_upper': upper_mpk,
                'target_minus_open_capital': s['target_capital']-k, 'desired_before_clipping': raw_desired,
                'replacement_investment': delta*k, 'max_gross_investment': (delta+growth)*k,
                'desired_credit_nominal': desired_credit, 'credit_scale_reconstructed': credit_scale,
                'goods_left_for_investment': goods_left,
                'credit_shortfall_real': s['desired_investment']-s['financed_investment'],
                'goods_shortfall_real': s['financed_investment']-s['actual_investment'],
                'actual_net_capital_growth': (s['capital_next']-k)/k,
                'boundary_contact_tolerance_real': boundary_tolerance},
            'binding_flags': flags, 'ordering_passed': ordering, 'residuals': residuals,
            'causal_identification_claim': False}


def collect_investment(env, expected_used, transaction_price):
    """Read after settlement, before losing the pre-update expectation/transaction price.

    market.price is already next-period; caller supplies recorded transaction price.
    market.Kt is this step's production capital, not the pre-step stale Kt.
    """
    m, b = env.market, env.bank
    def scalar(value):
        return float(value.item()) if hasattr(value, 'item') else float(value)
    return diagnose_investment({
        'capital_open': scalar(m.Kt), 'capital_next': scalar(m.Kt_next), 'alpha': scalar(m.alpha),
        'productivity': scalar(m.Zt), 'labor': scalar(m.firm_labor_j),
        'depreciation_rate': scalar(b.depreciation_rate), 'adjustment_speed': scalar(b.capital_adjustment_speed),
        'max_net_growth': scalar(b.max_net_capital_growth), 'nominal_lending_rate': scalar(b.lending_rate),
        'expected_inflation_used': float(expected_used), 'real_lending_rate': scalar(b.real_lending_rate),
        'capital_user_cost': scalar(b.capital_user_cost), 'target_capital': scalar(b.target_capital),
        'desired_investment': scalar(b.desired_fixed_investment),
        'financed_investment': scalar(b.financed_fixed_investment), 'actual_investment': scalar(b.actual_fixed_investment),
        'available_credit_nominal': scalar(b.available_investment_credit), 'transaction_price': scalar(transaction_price),
        'goods_supply': scalar(m.goods_supply), 'household_consumption_real': float(env.households.final_consumption.sum()),
        'government_purchases_real': float(env.main_gov.gov_spending.sum())})
