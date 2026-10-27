"""Selection policy tests; no expensive reference/operator experiments."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import search as Q


class MemoryProtocol:
    RESULTS=Path('/virtual/final_exp2_selection/results')
    def __init__(self):self.files={}
    def read(self,path):return copy.deepcopy(self.files.get(str(path),[]))
    def write(self,path,rows):self.files[str(path)]=copy.deepcopy(rows)
    def candidate(self,method,par,**meta):
        p=Q.parfrom(par)
        return dict(meta,method=method,id=Q.S.ident(method,p),**p)


class SelectionTests(unittest.TestCase):
    def point(self,**changes):
        return Q.S.parameters(xi=1.,sR=3.,sF=3.,M=20,p=5,**changes)

    def test_fft_friendly_sizes_are_minimal(self):
        self.assertEqual(Q.next_smooth(14),15)
        self.assertEqual(Q.previous_smooth(15),12)
        self.assertEqual(Q.next_smooth(128),128)
        self.assertIsNone(Q.next_smooth(129))

    def test_existing_tail_prefactor_lambert_initialization(self):
        t=Q.cutoff_initialization(1e-4,1.05)
        for name,pref in (('theory_sR','theory_CS_relative'),('theory_sF','theory_CF_relative')):
            s,C=t[name],t[pref]
            self.assertAlmostEqual(C*s*Q.np.exp(-s*s),1e-4/3,places=16)
        self.assertIn('not certificate',t['cutoff_model'])

    def test_minimum_legal_M_and_influence_match_existing_mesh(self):
        par=self.point()
        M=Q.minimum_legal_M(par)
        self.assertTrue(Q.legal_metadata('PPPM-grid-PSD',dict(par,M=M))['search_mesh_legal'])
        for smaller in Q.smooth_sizes():
            if smaller>=M:break
            self.assertFalse(Q.legal_metadata('PPPM-grid-PSD',dict(par,M=smaller))['search_mesh_legal'])
        info=Q.legal_metadata('PPPM-grid-PSD',par)
        K=Q.S.E.kernel(.2,1.,0.)
        _,kc,_=Q.S.E.cutoffs(par,Q.L0)
        mesh,old=Q.S.E.check_mesh(Q.L0,K,kc,par['M'],par['p'])
        self.assertIsNotNone(mesh)
        self.assertAlmostEqual(info['W2min_prediction'],mesh.W2min,places=14)
        self.assertEqual(info['n_excluded_prediction'],0)

    def test_explicit_eta_safety_and_B2_distinction(self):
        par=self.point();par.update(M=15,xi=1.1)
        info=Q.legal_metadata('PPPM-grid-PSD',par)
        self.assertFalse(info['search_mesh_legal'])
        self.assertTrue(Q.legal_metadata('B2-positive',par)['search_mesh_legal'])
        self.assertFalse(Q.legal_metadata('PPPM-grid-PSD',dict(par,M=129))['search_mesh_legal'])
        self.assertFalse(Q.legal_metadata('PPPM-grid-PSD',dict(par,xi=float('nan')))['search_mesh_legal'])

    def test_theory_main_minimal_reasonable_p_and_low_order_probes(self):
        pts=Q.theory_candidates(1e-4,1e-3,xis=(1.05,))
        mains=[p for p in pts if p['seed_role']=='smallest_predicted_reasonable_p']
        self.assertEqual(len(mains),1)
        main=mains[0]
        self.assertLessEqual(main['alias_prediction'],1e-4/3)
        for p in range(2,main['p']):
            self.assertGreater(main['alias_C']*main['alias_q']**p,1e-4/3)
        probes=[p for p in pts if p['seed_role']=='independent_low_order_probe']
        self.assertEqual({p['p'] for p in probes},{3,5,7})
        self.assertTrue(all(p['M']==main['initial_minimum_legal_M'] for p in probes))
        self.assertTrue(all(not p['bound_used_as_constraint'] and not p['PSD_eigenvalue_used_as_constraint'] for p in pts))

    def test_alias_calibration_uses_same_physics_and_is_not_certificate(self):
        r=dict(self.point(),method='PPPM-original',status='ok',E_G=2e-5)
        fit=Q.alias_calibration([r,dict(r,kappa=.7),dict(r,method='PPPM-grid-PSD')])
        self.assertEqual(fit['n_unique_calibration_points'],1)
        self.assertGreater(fit['alias_C'],0)
        self.assertFalse(fit['certified'])
        self.assertEqual(Q.alias_calibration([])['alias_C'],1e-3)

    def test_joint_neighbors_cover_all_five_variables(self):
        center=dict(self.point(),method='PPPM-grid-PSD',id='parent')
        points=Q.joint_neighbors(center,1e-4)
        varied={k for p in points for k in ('xi','sR','sF','M','p') if p[k]!=center[k]}
        self.assertEqual(varied,{'xi','sR','sF','M','p'})
        joint=next(p for p in points if p['neighbor_reason']=='joint_relax')
        self.assertLess(joint['M'],center['M'])
        self.assertLess(joint['p'],center['p'])
        self.assertLess(joint['sR'],center['sR'])
        self.assertLess(joint['sF'],center['sF'])
        self.assertTrue(all(p['parent_id']=='parent' for p in points))

    def test_shortlist_does_not_use_projection_bounds_or_PSD_eigenvalues(self):
        a=dict(id='fast',status='ok',E_G=5e-5,t_screen_s=.001,E_proj=99.,B_off=99.,lam_min_h=-99.)
        b=dict(id='accurate',status='ok',E_G=1e-8,t_screen_s=.1)
        fail=dict(id='failed',status='ok',E_G=.01,t_screen_s=.0001)
        selected=Q.shortlist_rows([a,b,fail],1e-4)
        self.assertEqual({r['id'] for r in selected},{'fast','accurate'})

    def test_nine_config_precision_condition_is_actual_full_error(self):
        summary=dict(status='ok',n_configs=9,max_E_G=5e-5,E_proj=1.,B_off=100.,min_eigenvalue=-3.)
        self.assertTrue(Q.nine_pass(summary,1e-4))
        self.assertFalse(Q.nine_pass(dict(summary,n_configs=8),1e-4))
        self.assertFalse(Q.nine_pass(dict(summary,max_E_G=1.001e-4),1e-4))
        self.assertFalse(Q.nine_pass(dict(summary,max_E_G=float('nan')),1e-4))

    def test_fresh_physical_candidate_reused_across_accuracy_memberships(self):
        memory=MemoryProtocol();p=dict(self.point(),epsilon=1e-3,stage='seeds',origin='theory')
        with patch.object(Q,'_protocol',return_value=memory):
            first=Q._record_candidates([p],'PPPM-grid-PSD')
            second=Q._record_candidates([dict(p,epsilon=1e-4)],'PPPM-grid-PSD')
        self.assertEqual(len(first),1)
        self.assertEqual(len(second),0)
        self.assertEqual(len(memory.read(memory.RESULTS/'candidates.csv')),1)
        self.assertEqual(len(memory.read(memory.RESULTS/'candidate_memberships.csv')),2)

    def test_resume_keeps_recorded_neighborhood_centers(self):
        memory=MemoryProtocol();par=self.point()
        existing=dict(par,method='PPPM-grid-PSD',id='existing-local',epsilon=1e-4,stage='local')
        memory.write(memory.RESULTS/'candidates.csv',[existing])
        memory.write(memory.RESULTS/'candidate_memberships.csv',[
            dict(id='existing-local',method='PPPM-grid-PSD',epsilon=1e-4,stage='local')])
        with patch.object(Q,'_protocol',return_value=memory),patch.object(Q,'evaluate_candidates',return_value=[]) as evaluate:
            Q.neighborhoods('local',(1e-4,),2)
        self.assertEqual([r['id'] for r in evaluate.call_args.args[0]],['existing-local'])

    def test_historical_candidate_is_seed_only(self):
        points=Q.comparator_candidates('PPPM-original',1e-4)
        self.assertGreater(len(points),0)
        self.assertTrue(all(p['historical_error_reused_as_acceptance'] is False for p in points))
        self.assertTrue(all(p['origin']=='historical_initialization_fresh_recheck' for p in points))


if __name__=='__main__':
    started=time.perf_counter()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):program=unittest.main(exit=False)
    result=program.result;here=Path(__file__).resolve().parent
    (here/'results').mkdir(exist_ok=True)
    (here/'results'/'tests_summary.json').write_text(json.dumps(dict(tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),ok=result.wasSuccessful(),
        elapsed_s=time.perf_counter()-started,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (here/'search.py',Path(__file__))}),indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
