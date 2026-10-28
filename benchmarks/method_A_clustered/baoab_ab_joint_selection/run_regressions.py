"""Run all106 inherited relevant regressions with new fixture output routing."""
from joint_common import *
import unittest
for path in (THERMOSTAT,THERMOSTAT/'optimized_pairou',HERE.parent/'validation_exp1/deconv_psd_comparison',PILOT,MATCHED):
    sys.path.insert(0,str(path))
import protocol
protocol.INPUTS=INPUTS/'regression_fixtures'
names=['test_ou_core','test_spatial','test_experiment','test_compiled_edge','test_validation','test_grid_psd',
    'test_dynamics','test_spatial_pilot','test_rank_validation','test_matched_dynamics',
    'test_frozen_accuracy','test_stochastic_validation','test_plot_results']
suite=unittest.defaultTestLoader.loadTestsFromNames(names)
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
