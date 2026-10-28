import unittest
from pathlib import Path
import numpy as np
import joint_spatial as J

class JointSpatialTests(unittest.TestCase):
    def test_conservative_reference_allowance(self):
        self.assertAlmostEqual(J.spatial_bound(.02,10.,.001),.021/9.999)
        with self.assertRaises(ValueError):J.spatial_bound(0.,.1,.1)
    def test_probe_bound_is_safe_even_when_direction_soft(self):
        G=np.diag([10.,.01]);H=G+np.diag([0.,.001]);v=np.array([0.,1.])
        bound=J.probe_rejection_lower_bound(np.linalg.norm((H-G)@v),10.,0.)
        self.assertAlmostEqual(bound,.0001)
        self.assertLessEqual(bound,np.linalg.norm(H-G,2)/np.linalg.norm(G,2))
    def test_joint_candidates_cover_splitting_and_asymmetry(self):
        first=J.candidate_round_one()
        self.assertEqual(set(p['xi'] for p in first),{.45,.55,.65,.75,.85,1.})
        self.assertTrue(any(p['sR']!=p['sF'] for p in first))
        self.assertTrue(any(p==J.BASE_PARAMS for p in first))
        self.assertLessEqual(len(first)+len(J.adaptive_refinements([])),60)
        self.assertEqual(len({J.setting_id(p) for p in first}),len(first))
    def test_readonly_direct_reference_new_physics(self):
        state=J.load_states()[0];G,meta=J.physical_reference(state)
        self.assertEqual(meta['kappa'],.7)
        self.assertEqual(meta['c'],J.BASE_PARAMS['c'])
        self.assertTrue(meta['reused_readonly_reference'])
        self.assertGreater(meta['tail_absolute'],0.)
        self.assertEqual(J.array_sha(G),meta['matrix_sha256'])
        self.assertLess(meta['tail_relative'],1e-9)
    def test_illegal_mesh_parameters_rejected(self):
        with self.assertRaises(ValueError):J.build_mesh(J.make_params(1.,4.7,5.2,8,6),J.BASE_PARAMS['L'])
        bad=J.make_params(.65,4.7,4.7,32,6);bad['kappa']=.2
        with self.assertRaises(ValueError):J.build_mesh(bad,J.BASE_PARAMS['L'])

if __name__=='__main__':unittest.main()
