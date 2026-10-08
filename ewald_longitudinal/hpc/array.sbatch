#!/bin/bash
# One production trajectory per array task (one core, no GPU). Every task runs toy_run.py --resume, so a task that
# is already complete exits at once, an interrupted one continues from its last checkpoint, and a new one starts.
# Submit with hpc/submit.sh array ... (it sets --array=IDS%MAX_PARALLEL, --time, --mem, --account, --partition,
# --output and exports TOY_ENV_FILE, TASKLIST, MAX_WALL_HOURS).
#SBATCH --job-name=toy_array
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --signal=B:USR1@300
set -uo pipefail
: "${TOY_ENV_FILE:?}" "${TASKLIST:?}" "${SLURM_ARRAY_TASK_ID:?not an array task}"
source "${TOY_HPC_DIR:?TOY_HPC_DIR not set (use hpc/submit.sh)}/common.sh"
toy_load_env "$TOY_ENV_FILE" || exit 1
toy_threads
eval "$PY_SETUP"
line=$(awk -F'\t' -v id="$SLURM_ARRAY_TASK_ID" 'NR > 1 && $1 == id' "$TASKLIST")
[ -n "$line" ] || { echo "task $SLURM_ARRAY_TASK_ID not in $TASKLIST" >&2; exit 1; }
IFS=$'\t' read -r tid pot ker N seed cfg burn prod save rel <<< "$line"
out="$RUN_ROOT/$rel"
mkdir -p "$(dirname "$out")"
cd "$REPO_DIR/ewald_longitudinal"
echo "task $tid: $pot $ker N=$N seed=$seed config=$cfg burn-in=$burn production=$prod -> $out"
"$PYTHON" -u toy_run.py --config "$cfg" --potential "$pot" --kernel "$ker" --N "$N" --seed "$seed" \
    --burn-in "$burn" --production "$prod" --save-every "$save" --out "$out" --resume \
    ${MAX_WALL_HOURS:+--max-wall-hours "$MAX_WALL_HOURS"} &
pid=$!
# Slurm sends USR1 to this shell 300 s before the time limit (--signal=B:USR1@300); forward it so that the run
# checkpoints and exits 75. TERM (scancel, preemption) is forwarded as well.
trap 'echo "forwarding signal to $pid"; kill -USR1 "$pid" 2>/dev/null' USR1 TERM
rc=0
while true; do
    wait "$pid"; rc=$?
    kill -0 "$pid" 2>/dev/null || break          # still alive: wait was interrupted by a trapped signal
done
echo "toy_run exit code $rc (0 complete, 75 stopped early and resumable, 3 run directory locked by another live process, 2 failed, 1 configuration error)"
exit "$rc"
