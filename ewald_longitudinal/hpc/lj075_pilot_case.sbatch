#!/bin/bash
# LJ law A HPC short pilot, one case per job: dense (full periodic Gamma, dense matrix functions) or fast (PPPM +
# Lanczos 14/5) O-step, dt 0.005 or 0.01, N = 256, t = 1 from the shared canonical state. One core, one thread.
# Submit the four case jobs and the analysis job with:  hpc/submit.sh lj075-pilot   (never as an array of replicas)
#
# The job runs lj075_hpc_pilot.py case (see its docstring), writing $PILOT_ROOT/$CASE/:
#   run/   three segments: stopped by SIGUSR1 after the first checkpoint (exit 75), resumed with a 30-step limit
#          (exit 75) while a duplicate resume on the same directory must exit 3 (lock), resumed to t = 1 (exit 0);
#   twin/  a continuous run to 10 steps past the second restart, compared bitwise with run/;
#   env_report.json, lock_check.json, case_summary.json.
# Slurm's own USR1 (--signal, 120 s before the limit) goes to this shell and is forwarded: the running segment
# checkpoints and exits 75, no further phase starts. Nothing is resubmitted automatically. The job exits 0 only if the
# run completed, every exit code was as expected, the duplicate was refused, lock_check passed and the restart check
# passed.
#
# Local emulation without Slurm (not an HPC validation): hpc/lj075_pilot_local.sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --signal=B:USR1@120
#SBATCH --no-requeue
# --no-requeue: a requeued job would find its case directory in use and stop; resubmit a failed pilot by hand with a
# new PILOT_ROOT (hpc/submit.sh lj075-pilot) after looking at the Slurm output.
set -uo pipefail
: "${TOY_ENV_FILE:?TOY_ENV_FILE not set (use hpc/submit.sh)}" "${CASE:?CASE not set}" "${PILOT_ROOT:?PILOT_ROOT not set}"
source "${TOY_HPC_DIR:?TOY_HPC_DIR not set (use hpc/submit.sh)}/common.sh"
toy_load_env "$TOY_ENV_FILE" || exit 1
toy_threads
eval "$PY_SETUP"
cd "$REPO_DIR/ewald_longitudinal" || exit 1
mkdir -p "$PILOT_ROOT"
echo "$(date +%T) lj075 pilot case $CASE -> $PILOT_ROOT/$CASE (job ${SLURM_JOB_ID:-none}, host $(hostname), threads" \
     "OPENBLAS=$OPENBLAS_NUM_THREADS OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS)"
pid=""
trap 'echo "$(date +%T) batch shell got a signal; forwarding USR1 to ${pid:-none}"; [ -n "$pid" ] && kill -USR1 "$pid" 2>/dev/null' USR1 TERM
"$PYTHON" -u lj075_hpc_pilot.py case --case "$CASE" --root "$PILOT_ROOT" &
pid=$!
rc=0
while true; do                                   # wait returns early when a trapped signal arrives
    wait "$pid"; rc=$?
    kill -0 "$pid" 2>/dev/null || break
done
echo "$(date +%T) case $CASE exit $rc"
exit "$rc"
