import copy
import unittest
from bridge import check_parameters, check_run, controlled_config, trajectory_moments, temporal_mae
from core import Budget, CalibrationError


class ContractTests(unittest.TestCase):
    def test_unknown_parameter(self):
        with self.assertRaises(ValueError): check_parameters({'g_z_delta': .01})

    def test_nonfinite(self):
        with self.assertRaises(ValueError): check_parameters({'price_adjustment_speed': float('nan')})

    def test_negative(self):
        with self.assertRaises(ValueError): check_parameters({'inflation_expectation_lambda': -.1})

    def test_zero_capital_speed(self):
        with self.assertRaises(ValueError): check_parameters({'capital_adjustment_speed': 0})

    def test_boolean(self):
        with self.assertRaises(ValueError): check_parameters({'price_adjustment_speed': True})

    def test_duplicate_seeds(self):
        with self.assertRaises(ValueError): check_run(64, 3, [1, 1])

    def test_bad_horizon(self):
        with self.assertRaises(ValueError): check_run(64, 28, [1])

    def test_seed_overlap(self):
        with self.assertRaises(CalibrationError): Budget(2, (1,), (1,)).validate()

    def test_taskbook_config(self):
        c = controlled_config({}, 64, 3, 908101)
        e = {x.entity_name: x.entity_args.params for x in c.Environment.Entities}
        self.assertEqual(c.Environment.env_core.problem_scene, 'real_society')
        self.assertEqual(e['bank'].capital_adjustment_speed, .2)  # alternate scene has .05
        self.assertEqual(c.Trainer.household_labor_rule, 'initial')
        self.assertEqual(c.Trainer.household_consumption_share, .95)
        self.assertTrue(e['households'].freeze_labor_efficiency)
        self.assertEqual(e['market'].sigma_z, 0)

    def test_all_three_routed_before_construction(self):
        c = controlled_config(dict(capital_adjustment_speed=.3, price_adjustment_speed=.4,
                                   inflation_expectation_lambda=.6), 64, 3, 5)
        b = next(x.entity_args.params for x in c.Environment.Entities if x.entity_name == 'bank')
        self.assertEqual(b.capital_adjustment_speed, .3)
        self.assertEqual(c.Environment.env_core.price_adjustment_speed, .4)
        self.assertEqual(c.Environment.env_core.inflation_expectation_lambda, .6)

    def test_no_mutable_config_leak(self):
        controlled_config({'price_adjustment_speed': .8}, 64, 3, 1)
        self.assertEqual(controlled_config({}, 64, 3, 1).Environment.env_core.price_adjustment_speed, .2)

    def test_period_means_first_inflation_excluded(self):
        rows = [{'inflation': 0, 'real_gdp_growth': 0, 'consumption_gdp': .6, 'investment_gdp': .2},
                {'inflation': .02, 'real_gdp_growth': .03, 'consumption_gdp': .6, 'investment_gdp': .2}]
        second = copy.deepcopy(rows)
        second[1]['inflation'] = .04
        moments = trajectory_moments([{'trajectory': rows}, {'trajectory': second}], 2)
        self.assertNotIn('inflation:t1', moments)
        self.assertAlmostEqual(moments['inflation:t2'], .03)

    def test_mae_not_absolute_mean_error(self):
        self.assertAlmostEqual(temporal_mae({'inflation:t2': .03, 'inflation:t3': .01},
                                           {'inflation:t2': .02, 'inflation:t3': .02})['inflation'], .01)


if __name__ == '__main__':
    unittest.main()
