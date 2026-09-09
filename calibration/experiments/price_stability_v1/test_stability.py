import copy
import unittest
import numpy as np
from stability_stats import losses,choose,bootstrap_losses,ambiguity_set,rank_agreement,check_seed_roles,valid_rows

class Tests(unittest.TestCase):
    def test_protocol_budget(self):
        from contract import make_protocol
        c={'foundation_policy':{'secondary_private_nonresidential':{'alpha':.36}}}
        p=make_protocol(c)
        self.assertEqual(p['unique_seed_count'],576)
        calls=paths=0
        for profile in p['profiles']:
            calls+=len(profile['target_seeds'])+len(p['grid'])*len(profile['search_seed_pools'])+len(p['grid'])
            paths+=sum(map(len,profile['target_seeds'].values()))+len(p['grid'])*sum(map(len,profile['search_seed_pools']))+len(p['grid'])*len(profile['validation_seeds'])
        self.assertEqual(calls,p['expected_simulator_calls']);self.assertEqual(paths,p['expected_physical_paths'])
    def test_loss(self):
        np.testing.assert_allclose(losses([[[0],[2]],[[3],[3]]],[[1],[1]]),[0,2])
    def test_nonfinite(self):
        with self.assertRaises(ValueError):losses([[[float('nan')]]],[[0]])
    def test_empty(self):
        with self.assertRaises(ValueError):losses(np.zeros((2,0,1)),[[0]])
    def test_tie(self):
        self.assertEqual(choose([1,1,1],[.15,.2,.25]),1)
    def test_best(self):
        self.assertEqual(choose([0,1,1],[.15,.2,.25]),0)
    def test_pairing(self):
        x=np.array([[[1],[2],[3]],[[1],[2],[3]]]);b=bootstrap_losses(x,[[0],[2]],30,42)
        np.testing.assert_array_equal(b[:,0],b[:,1]);self.assertEqual(ambiguity_set(b,0),[0,1])
    def test_reproducible(self):
        x=np.array([[[1],[2]],[[3],[4]]]);np.testing.assert_array_equal(bootstrap_losses(x,[[0],[1]],20,3),bootstrap_losses(x,[[0],[1]],20,3))
    def test_ranks(self):
        self.assertEqual(rank_agreement([1,2,3],[1,2,3]),1);self.assertEqual(rank_agreement([1,2,3],[3,2,1]),-1)
    def test_roles(self):
        p={'profiles':[{'target_seeds':{'a':[100,101]},'search_seed_pools':[[102,103],[104,105]],'validation_seeds':[106,107]}]}
        self.assertEqual(check_seed_roles(p),8);p['profiles'][0]['validation_seeds']=[101,102]
        with self.assertRaises(ValueError):check_seed_roles(p)
    def test_reserved(self):
        p={'profiles':[{'target_seeds':{'a':[2]},'search_seed_pools':[[100]],'validation_seeds':[101]}]}
        with self.assertRaises(ValueError):check_seed_roles(p)
    def test_rows(self):
        row=lambda i:{'period':i,'sector_accounting':{'status':'PASS'},'investment_diagnostic':{'status':'PASS'}}
        r={'valid':True,'paths':[{'seed':101,'valid':True,'trajectory':[row(1),row(2)]}]}
        valid_rows(r,[101],2);x=copy.deepcopy(r);x['paths'][0]['trajectory'][0]['sector_accounting']['status']='FAIL'
        with self.assertRaises(ValueError):valid_rows(x,[101],2)
        with self.assertRaises(ValueError):valid_rows(r,[101,102],2)
        with self.assertRaises(ValueError):valid_rows(r,[101],3)
    def test_fake_pipeline(self):
        grid=[.15,.175,.2,.225,.25];noise=np.array([-.01,.01,-.02,.02])
        x=np.array([(p+noise)[:,None] for p in grid]);y=np.array([[.225],[.225]])
        best=choose(losses(x,y),grid);self.assertEqual(best,3)
        self.assertIn(best,ambiguity_set(bootstrap_losses(x,y,50,77),best))

if __name__=='__main__':unittest.main()
