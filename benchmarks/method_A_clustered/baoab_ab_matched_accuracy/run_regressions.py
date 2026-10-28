"""Re-run all relevant PR8--10 tests with generated fixtures redirected here."""
from matched_common import HERE,LEGACY,THERMOSTAT,INPUTS
import sys
import unittest
for path in (THERMOSTAT,THERMOSTAT/'optimized_pairou',HERE.parent/'validation_exp1/deconv_psd_comparison',LEGACY):
    sys.path.insert(0,str(path))
import protocol
protocol.INPUTS=INPUTS/'regression_fixtures'
suite=unittest.defaultTestLoader.loadTestsFromNames(['test_ou_core','test_spatial','test_experiment',
    'test_compiled_edge','test_validation','test_grid_psd','test_dynamics','test_spatial_pilot','test_rank_validation'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
