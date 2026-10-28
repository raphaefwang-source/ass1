"""Small direct tests of the Stage 0 probe and error-validation protocol."""
import unittest
import numpy as np

from rank_validation import (DenseOU, PSDViolation, directions, project,
                             relative_error, validate_probe_set, _candidate_ranks)


class RankValidationTests(unittest.TestCase):
    def setUp(self):
        self.q = np.array([[.1,.2,.3],[1.4,.3,.2],[.4,1.8,.6],[1.6,1.4,1.2]])
        rng = np.random.default_rng(710)
        b = rng.normal(size=(12,12))
        pi = np.kron(np.eye(4)-np.ones((4,4))/4,np.eye(3))
        self.g = pi @ (b@b.T+np.eye(12)) @ pi
        self.ref = DenseOU(self.g)
        self.p, self.z = rng.normal(size=(2,12))

    def test_registered_directions_include_physical_modes(self):
        probes = directions(self.q,self.ref,self.p,self.z)
        for name in ('slow','fast','random','actual_momentum','actual_gaussian',
                     'lowk_axis0_sin_L0','lowk_axis0_sin_T1',
                     'lowk_axis2_cos_L2','lowk_axis2_cos_T0'):
            self.assertIn(name,probes)
        for v in probes.values():
            self.assertLess(np.linalg.norm(v.reshape(-1,3).sum(0)),1e-13)
            self.assertAlmostEqual(np.linalg.norm(v),1.)

    def test_actual_action_validation_with_full_com_free_rank(self):
        probes = directions(self.q,self.ref,self.p,self.z)
        for kind in ('damping','noise'):
            rows, arrays = validate_probe_set(lambda v:self.g@v,self.ref,probes,
                                              9,.005,kind)
            self.assertTrue(all(row['passed'] for row in rows))
            self.assertLess(max(row['relative_error'] for row in rows),1e-12)
            self.assertEqual(len(arrays),3*len(probes))

    def test_low_rank_is_not_automatically_accepted(self):
        probes = {'actual':project(self.p)}
        rows,_ = validate_probe_set(lambda v:self.g@v,self.ref,probes,1,.005,'noise')
        self.assertFalse(rows[0]['passed'])

    def test_material_negative_ritz_aborts(self):
        with self.assertRaises(PSDViolation):
            validate_probe_set(lambda v:-v,self.ref,{'p':project(self.p)},3,.005,'noise')

    def test_candidates_are_bounded_and_include_dimension(self):
        self.assertEqual(_candidate_ranks(5,9),[5,8,9])
        self.assertEqual(_candidate_ranks(14,9),[9])
        self.assertEqual(relative_error(np.ones(3),np.ones(3)),0.)


if __name__ == '__main__':
    unittest.main()
