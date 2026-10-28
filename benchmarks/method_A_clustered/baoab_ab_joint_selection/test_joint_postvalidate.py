import unittest
from copy import deepcopy
from joint_postvalidate import select_endpoints,reconstruct_draws
from joint_common import array_sha
from model import rng_sha
import numpy as np

class PostValidationTests(unittest.TestCase):
    def heads(self,rounds=6):
        group=dict(group_id='g',comparison_type='independent_optimum',epsilon_frozen=1e-6)
        runs=[dict(group_id='g',method=m,initial_index=r%3,round=r,run=f'g_{r}_{m}') for r in range(rounds) for m in ('A','B')]
        return dict(groups=[group],runs=runs)
    def test_samples_last_actual_round_per_start(self):
        rows=select_endpoints(self.heads())
        self.assertEqual({r['round'] for r in rows},{3,4,5})
        self.assertEqual(len(rows),6)
        self.assertEqual({(r['method'],r['initial_index']) for r in rows},{(m,i) for m in ('A','B') for i in range(3)})
    def test_three_round_and_explicit_group_reuse(self):
        heads=self.heads(3)
        heads['groups'].append(dict(group_id='same',reuse_group_id='g'))
        rows=select_endpoints(heads)
        self.assertEqual(len(rows),6)
        self.assertEqual({r['round'] for r in rows},{0,1,2})
    def test_missing_control_cannot_be_invented(self):
        heads=self.heads(3);heads['runs'].pop()
        with self.assertRaises(ValueError):select_endpoints(heads)
    def test_gaussian_substep_indexing_and_independent_reverse(self):
        rng=np.random.default_rng(44);before=deepcopy(rng.bit_generator.state);rows=[];expected=[]
        for i in range(2):
            s=rng_sha(rng);z=rng.standard_normal((4,3));ef=rng.standard_normal((3,3));er=rng.standard_normal((3,3))
            rows.append(dict(substep=i+1,z_sha256=array_sha(z),eta_forward_sha256=array_sha(ef),
                eta_reverse_sha256=array_sha(er),rng_state_before_sha256=s,rng_state_after_sha256=rng_sha(rng)))
            expected.append((z,ef,er))
        record=dict(before=before,after=deepcopy(rng.bit_generator.state))
        draws,_=reconstruct_draws(dict(method='A',k=2),(4,3),record,rows,3)
        for draw,triple in zip(draws,expected):
            for name,a in zip(('z','ef','er'),triple):np.testing.assert_array_equal(draw[name],a)
        rows[1]['eta_reverse_sha256']=rows[1]['eta_forward_sha256']
        with self.assertRaises(RuntimeError):reconstruct_draws(dict(method='A',k=2),(4,3),record,rows,3)

if __name__=='__main__':unittest.main()
