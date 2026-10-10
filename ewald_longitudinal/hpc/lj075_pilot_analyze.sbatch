#!/bin/bash
# LJ law A HPC short pilot: analysis job, submitted by hpc/submit.sh lj075-pilot with --dependency=afterany on the
# four case jobs (it also runs when a case failed, and reports the failure). PILOT_JOB_IDS: case job ids joined by ':'.
# Writes $PILOT_ROOT/report/: pilot_report.md, pilot_table.md, pilot_results.json, pilot_differences.npz, figures,
# and sacct.txt (Slurm's own accounting of the case jobs: elapsed, CPU, MaxRSS, node), when sacct is available.
# Copy $PILOT_ROOT/report back into the repository (lj075_results/hpc_pilot/) for review.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
set -uo pipefail
: "${TOY_ENV_FILE:?TOY_ENV_FILE not set (use hpc/submit.sh)}" "${PILOT_ROOT:?PILOT_ROOT not set}"
source "${TOY_HPC_DIR:?TOY_HPC_DIR not set (use hpc/submit.sh)}/common.sh"
toy_load_env "$TOY_ENV_FILE" || exit 1
toy_threads
eval "$PY_SETUP"
cd "$REPO_DIR/ewald_longitudinal" || exit 1
mkdir -p "$PILOT_ROOT/report"
if [ -n "${PILOT_JOB_IDS:-}" ] && command -v sacct > /dev/null; then
    sacct -j "${PILOT_JOB_IDS//:/,}" -P --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,CPUTime,MaxRSS,ReqMem,NodeList \
        > "$PILOT_ROOT/report/sacct.txt" 2>&1
fi
"$PYTHON" -u lj075_hpc_pilot.py analyze --root "$PILOT_ROOT" --out "$PILOT_ROOT/report"
rc=$?
echo "$(date +%T) analysis exit $rc; report: $PILOT_ROOT/report/pilot_report.md"
exit "$rc"
