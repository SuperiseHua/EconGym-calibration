import unittest
from types import SimpleNamespace
from sector_accounting import equation, opening_snapshot, collect_sectors


class SectorAccountingTests(unittest.TestCase):
    def test_exact_identity(self):
        self.assertTrue(equation([10, -3, -7])['passed'])

    def test_offsetting_household_errors_not_cancelled(self):
        result = equation([[10, 10], [-9, -11]])
        self.assertFalse(result['passed'])
        self.assertEqual(result['violations'], 2)

    def test_positive_slack(self):
        self.assertTrue(equation([10, -9], inequality=True)['passed'])

    def test_negative_slack(self):
        self.assertFalse(equation([9, -10], inequality=True)['passed'])

    def test_nonfinite(self):
        self.assertFalse(equation([float('nan'), 0])['passed'])
        self.assertFalse(equation([float('inf'), 0])['passed'])

    def test_fixed_absolute_tolerance(self):
        self.assertTrue(equation([0, 1e-6])['passed'])
        self.assertFalse(equation([0, 1e-3])['passed'])

    def test_scale_tolerance(self):
        self.assertTrue(equation([1e10, -1e10+1])['passed'])
        self.assertFalse(equation([1e10, -1e10+100])['passed'])

    def test_broadcast_count(self):
        r=equation([[1,2,3], -1])
        self.assertEqual(r['equations_checked'],3)
        self.assertEqual(r['violations'],2)

    def test_snapshot_not_alias(self):
        import numpy as np
        h=SimpleNamespace(at_next=np.array([10.]),savings=np.array([10.]),risky_income=np.array([0.]))
        m=SimpleNamespace(cash=[2.],loan_balance=[3.],book_capital=[4.])
        b=SimpleNamespace(equity=1,last_deposit_rate=.01,last_lending_rate=.02,government_bond_rate=.03)
        env=SimpleNamespace(households=h,market=m,bank=b,main_gov=SimpleNamespace(Bt_next=5))
        before=opening_snapshot(env);h.at_next[0]=20
        self.assertEqual(before['household_assets'][0],10)

    def test_unsupported_route_abstains(self):
        env=SimpleNamespace(households=SimpleNamespace(type='olg'),market=SimpleNamespace(),bank=None,main_gov=None)
        self.assertEqual(collect_sectors(env,{},1)['status'],'UNAVAILABLE')


if __name__=='__main__': unittest.main()
