"""Guard scientific plot accounting and fresh B-control grouping."""
import unittest
import numpy as np
import importlib.util
from pathlib import Path
_spec=importlib.util.spec_from_file_location('matched_plot_results',Path(__file__).resolve().parent/'plot_results.py')
P=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

class PlotEvidenceTests(unittest.TestCase):
    def row(self,**extra):
        return dict(method='A',k=4,run='matched_t1_round0_A',epsilon_frozen=1e-6,
            round=0,initial_index=0,t_baoab_s=1.,t_dynamic_setup_s=.2,t_LJ_forces_s=.1,
            t_rng_s=.1,t_ou_wall_s=.5,t_baoab_other_s=.1,
            t_fft_s=.3,t_edge_eigh_s=.08,t_reorth_s=.1,**extra)

    def test_disjoint_phases_do_not_add_nested_diagnostics(self):
        values=P.phase_vector(self.row())
        self.assertAlmostEqual(values.sum(),1000.)
        self.assertEqual(len(values),5)

    def test_invalid_total_or_negative_phase_rejects_plot(self):
        row=self.row();row['t_baoab_s']=.8
        with self.assertRaises(ValueError):P.phase_vector(row)
        row=self.row();row['t_rng_s']=-.1
        with self.assertRaises(ValueError):P.phase_vector(row)

    def test_B_control_x_coordinate_does_not_repeat_B_OU(self):
        row=self.row();row.update(method='B',k=1)
        self.assertEqual(P.curve_k(row,{1e-6:4}),4)
        row.update(run='curve_k32_start2_B',epsilon_frozen=None)
        self.assertEqual(P.curve_k(row,{1e-6:4}),32)

    def test_range_uses_round_blocks_not_independent_step_CI(self):
        rows=[]
        for block,mean in [(0,1.),(1,2.)]:
            for step in range(3):
                row=self.row();row.update(round=block,t_baoab_s=mean)
                rows.append(row)
        summary=P.block_summary(rows)
        self.assertEqual(summary['n_steps'],6)
        self.assertEqual(summary['n_blocks'],2)
        self.assertEqual(summary['block_mean_min_ms'],1000.)
        self.assertEqual(summary['block_mean_max_ms'],2000.)

if __name__=='__main__':unittest.main()
