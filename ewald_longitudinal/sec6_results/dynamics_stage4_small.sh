#!/bin/bash
# Stage 4 (small) and stage 3 offline checks, run in this order from ewald_longitudinal/.
set -e
python3 sec6_dynamics.py offline --workers 4
python3 sec6_dynamics.py compare --N 64 --T 6 --ntraj 16 --hi 120 40 --workers 4
python3 sec6_dynamics.py compare --N 256 --T 6 --ntraj 4 --hi 160 40 --workers 4
python3 sec6_dynamics.py spatial --N 64 --T 6 --ntraj 16 --workers 4
python3 sec6_dynamics.py dtstudy --N 64 --T 6 --ntraj 16 --levels 1 2 --workers 4
