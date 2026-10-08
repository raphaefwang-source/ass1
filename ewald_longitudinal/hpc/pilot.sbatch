#!/bin/bash
# Short pilot: N = 512, laws A and B, 200 steps each (costopt_hiacc); runner unit tests; restart-consistency checks
# (N = 64, step-limit split and SIGTERM); acceptance checks; resource estimate for the default production task list.
# One core, no GPU; about 5-10 min of run time.
# Submit with hpc/submit.sh pilot (it adds --account/--partition/--output from hpc/cluster.env).
#SBATCH --job-name=toy_pilot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
set -euo pipefail
: "${TOY_ENV_FILE:?TOY_ENV_FILE not set (use hpc/submit.sh)}"
source "${TOY_HPC_DIR:?TOY_HPC_DIR not set (use hpc/submit.sh)}/common.sh"
toy_load_env "$TOY_ENV_FILE"
toy_threads
eval "$PY_SETUP"
cd "$REPO_DIR/ewald_longitudinal"
PILOT_DIR="$RUN_ROOT/pilot/${SLURM_JOB_ID:-local-$(date +%Y%m%dT%H%M%S)}"
mkdir -p "$PILOT_DIR"
echo "pilot directory: $PILOT_DIR"
"$PYTHON" -u hpc/env_report.py | tee "$PILOT_DIR/env_report.json"
TASKS="$PILOT_DIR/production_tasks.tsv"
"$PYTHON" -u hpc/make_tasklist.py --out "$TASKS"               # default production plan, for the estimate only
for K in A B; do
    "$PYTHON" -u toy_run.py --config costopt_hiacc --potential double_well --kernel "$K" --N 512 --seed 101 \
        --burn-in-steps 0 --production-steps 200 --save-every-steps 20 --checkpoint-every-steps 100 \
        --monitor-every-steps 50 --out "$PILOT_DIR/double_well_${K}_N512_s101"
done
fail=0
"$PYTHON" -u test_toy_run.py > "$PILOT_DIR/test_toy_run.log" 2>&1 || { fail=1; echo "test_toy_run.py FAILED (see log)"; }
tail -3 "$PILOT_DIR/test_toy_run.log"
"$PYTHON" -u hpc/restart_check.py --work "$PILOT_DIR/restart_check" --N 64 --kernel A --steps 60 --segments 25 20 || fail=1
"$PYTHON" -u hpc/restart_check.py --work "$PILOT_DIR/restart_check_signal" --N 64 --kernel B --steps 400 --signal || fail=1
"$PYTHON" -u hpc/check_run.py "$PILOT_DIR"/double_well_?_N512_s101 --expect-config costopt_hiacc \
    --json "$PILOT_DIR/pilot_check.json" || fail=1
"$PYTHON" -u hpc/estimate_resources.py --pilot "$PILOT_DIR"/double_well_?_N512_s101 --tasklist "$TASKS" \
    --margin 1.5 --json "$PILOT_DIR/resource_estimate.json" || fail=1
echo "PILOT $([ "$fail" = 0 ] && echo PASSED || echo FAILED): $PILOT_DIR"
exit "$fail"
