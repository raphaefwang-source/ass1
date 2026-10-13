#!/bin/bash
# Stage 5, run from ewald_longitudinal/. Run alone on the machine, single thread
# (sec6_timing.py sets OMP/OPENBLAS/MKL threads to 1).
set -e
python3 sec6_timing.py direct_accuracy      # choose R_d for the direct-sum baseline
python3 sec6_timing.py spatial_check        # T7 vs T9 action difference at every timed N
python3 sec6_timing.py time --method pppm direct --fresh
python3 sec6_timing.py dense
python3 sec6_timing.py report
