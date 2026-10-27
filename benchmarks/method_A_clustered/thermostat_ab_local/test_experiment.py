"""Regression tests for derived scientific diagnostics, independent of data files."""
import unittest
import numpy as np
from ou_core import EdgeOU, ou_a
from stopping_audit import evaluate, counts, GaussianSource, NegRNG, ideal_a_path


class TestStoppingAudit(unittest.TestCase):
    def row(self, rank, probe, change, error):
        return dict(config='synthetic', operator='full', alpha=1., dt=.2,
                    kind='noise', rank=rank, probe=probe,
                    step_change=change, error=error, tolerance=1e-3)

    def test_false_positive_detected_despite_small_rank_change(self):
        rows=[self.row(2,'a',None,.1), self.row(4,'a',1e-5,.02)]
        per,all_probes=evaluate(rows)
        self.assertEqual(per[0]['decision'],'no_previous_rank')
        self.assertEqual(per[1]['decision'],'false_positive')
        self.assertEqual(all_probes[1]['decision'],'false_positive')

    def test_all_probes_rule_requires_every_probe(self):
        rows=[self.row(4,'a',1e-5,1e-4),self.row(4,'b',.01,1e-4),
              self.row(6,'a',1e-5,1e-4),self.row(6,'b',1e-5,.02)]
        per,all_probes=evaluate(rows)
        self.assertEqual(counts(per),dict(no_previous_rank=0,true_positive=2,true_negative=0,
                                         false_negative=1,false_positive=1))
        self.assertEqual(all_probes[0]['decision'],'false_negative')
        self.assertEqual(all_probes[1]['decision'],'false_positive')

    def test_csv_empty_change_is_unavailable(self):
        _,aggregate=evaluate([self.row(2,'a','',.01),self.row(2,'b','',.02)])
        self.assertEqual(aggregate[0]['decision'],'no_previous_rank')
        self.assertIsNone(aggregate[0]['max_step_change'])

    def test_negative_source_preserves_every_draw_and_order(self):
        z=np.arange(6.)
        plus=GaussianSource(z,17)
        minus=NegRNG(GaussianSource(z,17))
        for shape in (6,(2,3),6,(2,3)):
            np.testing.assert_array_equal(plus.normal(shape),-minus.normal(shape))

    def test_independent_ideal_a_path_matches_full_rank_production(self):
        rng=np.random.default_rng(31)
        K=np.array([[2.,.2,0],[.2,1.,0],[0,0,1.]])
        edge=EdgeOU([(0,1,K)])
        long=np.block([[K,-K],[-K,K]])/3
        p,z=rng.normal(size=(2,6));dt=.2;substeps=2
        exact=ideal_a_path(edge,long,p,dt,substeps,GaussianSource(z,71))
        source=GaussianSource(z,71)
        production=p.copy()
        for _ in range(substeps):
            zz=source.normal(6);ef=source.normal((1,3));er=source.normal((1,3))
            production,_=ou_a(lambda v:long@v,production,zz,ef,er,edge,dt/substeps,3,3)
        np.testing.assert_allclose(production,exact,atol=2e-15)


if __name__ == '__main__':
    unittest.main()
