import copy
import unittest
from mae_acceptance import AXES, assess_mae
from investment_diagnostics import diagnose_investment


def contract(seeds=(1,), horizon=4):
    return {'schema_version': 1, 'target_kind': 'synthetic', 'evidence_level': 'development_predeclared',
            'horizon': horizon, 'expected_seeds': list(seeds), 'periods': list(range(2, horizon+1)),
            'thresholds': {a: .01 for a in AXES}, 'targets': {a: {str(t): 0. for t in range(2, horizon+1)} for a in AXES},
            'units': 'fraction'}


def response(seeds=(1,), horizon=4):
    return {'valid': True, 'paths': [{'valid': True, 'seed': s,
        'trajectory': [{'period': t, **{a: 0. for a in AXES}} for t in range(1, horizon+1)]} for s in seeds]}


class MAETests(unittest.TestCase):
    def test_zero_error_pass_not_economic_pass(self):
        r = assess_mae(response(), contract())
        self.assertEqual(r['macro_mae_status'], 'PASS')
        self.assertEqual(r['taskbook_acceptance'], 'NOT_EVALUATED')
        self.assertFalse(r['economic_validity_certified'])

    def test_mae_can_pass_when_one_period_fails(self):
        r = response()
        r['paths'][0]['trajectory'][2]['inflation'] = .024
        report = assess_mae(r, contract())
        self.assertAlmostEqual(report['axes']['inflation']['mean_path_mae'], .008)
        self.assertEqual(report['macro_mae_status'], 'PASS')
        self.assertFalse(report['axes']['inflation']['pointwise_within_same_threshold_diagnostic_only'])

    def test_temporal_signed_cancellation_not_allowed(self):
        r = response()
        r['paths'][0]['trajectory'][1]['inflation'] = .03
        r['paths'][0]['trajectory'][2]['inflation'] = -.03
        self.assertEqual(assess_mae(r, contract())['macro_mae_status'], 'FAIL')

    def test_mean_path_distinct_from_mean_seed_mae(self):
        r = response((1, 2))
        for t in range(1, 4):
            r['paths'][0]['trajectory'][t]['inflation'] = .02
            r['paths'][1]['trajectory'][t]['inflation'] = -.02
        a = assess_mae(r, contract((1, 2)))['axes']['inflation']
        self.assertEqual(a['mean_path_mae'], 0)
        self.assertAlmostEqual(a['mean_of_seed_maes'], .02)
        self.assertEqual(a['seed_mae_pass_fraction'], 0)

    def test_failed_seed_not_dropped(self):
        r = response((1, 2))
        r['paths'][1]['valid'] = False
        report = assess_mae(r, contract((1, 2)))
        self.assertEqual(report['macro_mae_status'], 'INVALID')
        self.assertEqual(report['axes'], {})

    def test_seed_coverage_fail_closed(self):
        for seeds in ((1,), (1, 1), (1, 3)):
            with self.subTest(seeds=seeds):
                self.assertEqual(assess_mae(response(seeds), contract((1, 2)))['macro_mae_status'], 'INVALID')

    def test_truncation_and_order_rejected(self):
        for operation in ('truncate', 'reorder'):
            r = response()
            if operation == 'truncate': r['paths'][0]['trajectory'].pop()
            else: r['paths'][0]['trajectory'].reverse()
            self.assertEqual(assess_mae(r, contract())['macro_mae_status'], 'INVALID')

    def test_nonfinite_rejected(self):
        r = response()
        r['paths'][0]['trajectory'][2]['inflation'] = float('nan')
        self.assertEqual(assess_mae(r, contract())['macro_mae_status'], 'INVALID')

    def test_missing_axis_not_zero_filled(self):
        c = contract()
        c['targets']['inflation'] = None
        report = assess_mae(response(), c)
        self.assertEqual(report['macro_mae_status'], 'NOT_EVALUATED')
        self.assertEqual(report['axes']['inflation']['status'], 'NOT_EVALUATED')

    def test_unknown_units_and_partial_periods_rejected(self):
        c = contract()
        c['units'] = 'percent'
        with self.assertRaises(ValueError): assess_mae(response(), c)
        c = contract()
        del c['targets']['inflation']['3']
        with self.assertRaises(ValueError): assess_mae(response(), c)

    def test_initialization_excluded(self):
        r = response()
        r['paths'][0]['trajectory'][0]['inflation'] = 100
        self.assertEqual(assess_mae(r, contract())['macro_mae_status'], 'PASS')
        c = contract()
        c['periods'] = [1, 2]
        with self.assertRaises(ValueError): assess_mae(r, c)

    def test_one_seed_no_fabricated_variance(self):
        self.assertIsNone(assess_mae(response(), contract())['axes']['inflation']['seed_mae_sd'])

    def test_aggregated_moments_are_ignored(self):
        r = response()
        r['moments'] = {'inflation:t2': 999.}
        self.assertEqual(assess_mae(r, contract())['macro_mae_status'], 'PASS')

    def test_inputs_unchanged(self):
        r, c = response(), contract()
        before = copy.deepcopy((r, c))
        assess_mae(r, c)
        self.assertEqual((r, c), before)


def investment_fixture(kind='goods'):
    # Hand-solvable fixture: alpha=.5, Z=2, L=100, K=100 -> MPK=1, cap=125.
    k, target, nominal = 100., 125., 0.
    if kind == 'floor':
        nominal = 2.
        target = (10/2.06)**2
    desired = min(max(6+.2*(target-k), 0.), 11.)
    credit = 5. if kind == 'credit' else 100.
    financed = desired*min(1., credit/(desired+1e-8))
    goods = 20. if kind != 'goods' else 7.
    actual = min(financed, goods)
    return {'capital_open': k, 'capital_next': 94+actual, 'alpha': .5, 'productivity': 2., 'labor': 100.,
            'depreciation_rate': .06, 'adjustment_speed': .2, 'max_net_growth': .05,
            'nominal_lending_rate': nominal, 'expected_inflation_used': 0., 'real_lending_rate': nominal,
            'capital_user_cost': nominal+.06, 'target_capital': target, 'desired_investment': desired,
            'financed_investment': financed, 'actual_investment': actual, 'available_credit_nominal': credit,
            'transaction_price': 1., 'goods_supply': 50., 'household_consumption_real': 25.,
            'government_purchases_real': 25.-goods}


class InvestmentTests(unittest.TestCase):
    def test_goods_constraint(self):
        r = diagnose_investment(investment_fixture())
        self.assertEqual(r['status'], 'PASS')
        self.assertTrue(r['binding_flags']['goods_rationing_material'])
        self.assertFalse(r['binding_flags']['credit_rationing_material'])
        self.assertEqual(r['derived']['goods_shortfall_real'], 4.)

    def test_credit_constraint(self):
        r = diagnose_investment(investment_fixture('credit'))
        self.assertEqual(r['status'], 'PASS')
        self.assertTrue(r['binding_flags']['credit_rationing_material'])
        self.assertFalse(r['binding_flags']['goods_rationing_material'])

    def test_zero_floor(self):
        r = diagnose_investment(investment_fixture('floor'))
        self.assertEqual(r['status'], 'PASS')
        self.assertTrue(r['binding_flags']['investment_zero_floor_active'])
        self.assertEqual(r['inputs']['actual_investment'], 0)

    def test_growth_cap(self):
        r = diagnose_investment(investment_fixture('unconstrained_goods'))
        self.assertTrue(r['binding_flags']['target_cap_active'])
        self.assertTrue(r['binding_flags']['investment_growth_cap_active'])
        self.assertAlmostEqual(r['derived']['actual_net_capital_growth'], .05)

    def test_roundoff_does_not_hide_cap_contact(self):
        s = investment_fixture('unconstrained_goods')
        s['target_capital'] -= 1e-12
        r = diagnose_investment(s)
        self.assertEqual(r['status'], 'PASS')
        self.assertTrue(r['binding_flags']['investment_growth_cap_active'])

    def test_wrong_price_detected(self):
        s = investment_fixture('credit')
        s['transaction_price'] = 2.
        r = diagnose_investment(s)
        self.assertEqual(r['status'], 'FAIL')
        self.assertFalse(r['residuals']['credit_rationing_mapping']['passed'])

    def test_wrong_expectation_timing_detected(self):
        s = investment_fixture()
        s['expected_inflation_used'] = .2
        self.assertFalse(diagnose_investment(s)['residuals']['real_rate_mapping']['passed'])

    def test_capital_equation_tamper(self):
        s = investment_fixture()
        s['capital_next'] += 1
        self.assertFalse(diagnose_investment(s)['residuals']['capital_transition']['passed'])

    def test_nonfinite_and_invalid_domain_rejected(self):
        for key, value in [('labor', float('nan')), ('capital_open', 0), ('depreciation_rate', 1)]:
            s = investment_fixture()
            s[key] = value
            with self.assertRaises(ValueError): diagnose_investment(s)

    def test_inputs_not_mutated(self):
        s = investment_fixture()
        before = copy.deepcopy(s)
        diagnose_investment(s)
        self.assertEqual(s, before)


if __name__ == '__main__':
    unittest.main()
