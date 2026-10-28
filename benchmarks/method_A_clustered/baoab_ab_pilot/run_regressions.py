"""Run reused implementation regressions, redirecting generated inputs here."""
from evidence import HERE, INPUTS
import sys
import unittest

parent=HERE.parent/'thermostat_ab_local'
for path in (parent,parent/'optimized_pairou',HERE.parent/'validation_exp1/deconv_psd_comparison'):
    sys.path.insert(0,str(path))
import protocol
protocol.INPUTS=INPUTS/'regression_fixtures'
suite=unittest.defaultTestLoader.loadTestsFromNames([
    'test_ou_core','test_spatial','test_experiment','test_compiled_edge',
    'test_validation','test_grid_psd'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
