import unittest
from pathlib import Path
from decimal import Decimal as D
import numpy as np
import research as r

class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=Path(__file__).parent/'data'
        cls.b,cls.tr,_=r.load(cls.path)
        cls.stages,cls.search,cls.selected=r.calibrate(cls.b)
    def test_original_reference_mae(self):
        for stage,expected in zip(self.stages[1:4],[.046540,.041631,.037261]):
            self.assertAlmostEqual(stage['mae'],expected,places=6)
    def test_direct_formula_independently(self):
        for i,row in self.b.iterrows():
            numerator=sum(D(str(row[f'ask{j}_qty']))*D(str(row[f'bid{j}_px']))+D(str(row[f'bid{j}_qty']))*D(str(row[f'ask{j}_px'])) for j in range(1,4))
            denominator=sum(D(str(row[f'ask{j}_qty']))+D(str(row[f'bid{j}_qty'])) for j in range(1,4))
            self.assertAlmostEqual(r.predict(self.b,{'kind':'direct'})[i],float(numerator/denominator),places=12)
    def test_flow_alignment(self):
        for i,row in self.b.iterrows():
            ts=self.tr[self.tr.printed_before==row.timestamp]
            expected=sum((1 if x.aggressor=='BUY' else -1)*x.qty for x in ts.itertuples())/ts.qty.sum() if len(ts) else 0
            self.assertEqual(row.flow,expected)
    def test_depth_example(self):
        legs,left=r.walk(self.b.iloc[10],'bid',50)
        self.assertEqual(left,0)
        self.assertAlmostEqual(sum(q*p for _,q,p in legs)/50,8.485)
        self.assertAlmostEqual(sum(q*p for _,q,p in legs)-50*8.50-.40,-1.15)
        self.assertGreater(r.walk(self.b.iloc[0],'bid',10000)[1],0)
    def test_selection_and_both_cash_ledgers(self):
        self.assertEqual(self.selected['model'],dict(kind='weighted',w2=1.,w3=.0625,decay='none',gamma=-.5))
        for test,expected in [(False,6.25),(True,3.)]:
            b,_,_=r.load(self.path,test)
            for stage in self.stages:
                rep=r.simulate(b,r.predict(b,stage['model']),**r.EXECUTION)
                r.audit(b,*rep)
            rep=r.simulate(b,r.predict(b,self.selected['model']),**r.EXECUTION)
            self.assertEqual(rep[2]['net_pnl'],expected)
    def test_future_features_do_not_change_past(self):
        changed=self.b.copy();changed.loc[19,'bid1_qty']*=20;changed.loc[19,'flow']=-1
        np.testing.assert_allclose(r.predict(self.b,self.selected['model'])[:19],r.predict(changed,self.selected['model'])[:19])
    def test_midpoint_and_incomplete_liquidation(self):
        self.assertEqual(r.simulate(self.b,self.b.mid.to_numpy())[2]['orders'],0)
        b=self.b.copy()
        for j in range(1,4):b.loc[19,f'bid{j}_qty']=1
        result=r.simulate(b,np.full(20,100.))[2]
        self.assertGreater(result['terminal_position'],0)
        self.assertIsNone(result['net_pnl'])
    def test_decay_and_zero_flow_invariance(self):
        m=dict(kind='weighted',w2=.25,w3=.0625,decay='none')
        np.testing.assert_allclose(r.predict(self.b,m),r.predict(self.b,{**m,'decay':'exp','tau':1e12}),atol=1e-9)
        b=self.b.copy();b['flow']=0
        np.testing.assert_allclose(r.predict(b,m),r.predict(b,{**m,'gamma':.5}))

if __name__=='__main__':unittest.main()
