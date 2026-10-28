"""Small audit regressions; preserved legacy arrays are readonly fixtures."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
import independent_audit as A


class AuditReconstructionTests(unittest.TestCase):
    def test_independent_actions_project_COM_and_resolve_damping_and_noise(self):
        N=4;Q=A.qbasis(N);lam=np.linspace(.1,2.,3*N-3)
        G=(Q*lam)@Q.T;v=np.arange(3*N,dtype=float).reshape(N,3)
        x=(v-v.mean(0)).ravel();h=.017
        for kind,f in [('damping',np.exp(-h*lam)),('noise',np.sqrt(-np.expm1(-2*h*lam)))]:
            actual=A.exact_action_from_matrix(G,v,h,kind)
            self.assertTrue(np.allclose(actual.ravel(),Q@(f*(Q.T@x)),rtol=1e-12,atol=1e-12))
            self.assertLess(np.linalg.norm(actual.sum(0)),1e-12)
        R,C,_=A.exact_reduced(G,Q)
        self.assertTrue(np.allclose(C,np.eye(len(R))-R@R.T,rtol=1e-12,atol=1e-12))

    def test_full_norm_retains_nonsymmetric_error(self):
        matrix=np.array([[0.,3.],[0.,0.]])
        self.assertEqual(A.norm2(matrix),3.)
        self.assertNotEqual(A.norm2(matrix),np.max(np.abs(np.linalg.eigvalsh((matrix+matrix.T)/2))))

    def test_actual_readonly_direct_reference_packed_annuli_reassemble_graph(self):
        path=A.PILOT/'results/initial_0_spatial_direct.npz'
        metadata=A.js(path.with_suffix('.json'))
        with np.load(path,allow_pickle=False) as data:
            self.assertEqual(data['annulus_tensors'].shape[-1],6)
            self.assertEqual(A.array_sha(data['G']),metadata['matrix_sha256'])
            actual=A.assemble_annulus_graph(data['annulus_tensors'],data['pi'],data['pj'],len(data['q']))
            np.testing.assert_allclose(actual,data['G'],rtol=2e-13,atol=2e-13)

    def test_legacy_actual_A_and_B_arrays_exercise_raw_RNG_and_BAOAB_audit(self):
        # Repackage old metadata in a temporary fixture, preserving all raw bytes.
        # This checks the new audit's schema and reconstruction, not a new timing.
        for method in ('A','B'):
            label=f'curve_k2_start0_{method}';legacy=A.MATCHED/'results';meta=A.js(legacy/f'{label}.json')
            rows=A.read_csv(legacy/f'{label}_steps.csv');inner=A.read_csv(legacy/f'{label}_substeps.csv')
            meta.update(params=dict(A.BASE_PARAMS),r_d=meta['ranks']['r_d'],r_n=meta['ranks']['r_n'],
                wall_per_simulated_time=meta['compute_wall_s']/(meta['steps']*A.DT),
                loop_wall_including_evidence_s=meta['loop_wall_s'])
            for row in rows:
                row.update({k:A.BASE_PARAMS[k] for k in ('xi','sR','sF','M','p')})
                row['wall_per_simulated_time']=row['t_baoab_s']/A.DT
            with tempfile.TemporaryDirectory() as td:
                here=Path(td);results=here/'results';results.mkdir();(here/'inputs').mkdir()
                (here/meta['file']).symlink_to(A.MATCHED/meta['file'])
                A.save(results/f'{label}.json',meta)
                A.save(results/f'{label}_rng.json',A.js(legacy/f'{label}_rng.json'))
                A.write_csv(results/f'{label}_steps.csv',rows);A.write_csv(results/f'{label}_substeps.csv',inner)
                with mock.patch.multiple(A,HERE=here,RESULTS=results):
                    _,_,_,detail=A.audit_run(label)
                    self.assertEqual(detail['issues'],[],detail)


if __name__=='__main__':unittest.main()
