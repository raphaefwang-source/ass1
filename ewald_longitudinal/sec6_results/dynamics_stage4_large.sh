#!/bin/bash
# Stage 4 (large): short coupled production vs high-rank Lanczos runs, run from ewald_longitudinal/.
set -e
python3 sec6_dynamics.py large --N 16000 4000 --trajs 0 1 --nsteps 25 --hi 160 40 --workers 4
