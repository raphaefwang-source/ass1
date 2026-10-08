#!/bin/bash
# Submission wrapper. Reads hpc/cluster.env (copy of cluster.env.example with every FILL_ME replaced).
#
#   hpc/submit.sh pilot [--dry-run]        (two pilot jobs: kernels A and B, N = 512)
#   hpc/submit.sh array TASKLIST --time HH:MM:SS --mem 2G [--ids 1-40] [--max-parallel 20] [--submit]
#
# "array" only prints the sbatch command unless --submit is given (long runs are never submitted by default).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${TOY_ENV_FILE:-$HERE/cluster.env}"
source "$HERE/common.sh"
toy_load_env "$ENV_FILE"
mkdir -p "$LOG_ROOT"
common=(--account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION")
[ -n "${SLURM_QOS:-}" ] && common+=(--qos="$SLURM_QOS")
mode="${1:-}"; shift || true
case "$mode" in
pilot)
    # two short jobs, kernels A and B (N = 512, one core each); see hpc/pilot_task.sbatch
    for K in A B; do
        cmd=(sbatch "${common[@]}" --job-name="toy_pilot_$K" --output="$LOG_ROOT/%x-%j.out"
             --export="ALL,TOY_ENV_FILE=$ENV_FILE,TOY_HPC_DIR=$HERE,KERNEL=$K" "$HERE/pilot_task.sbatch")
        if [ "${1:-}" = "--dry-run" ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
    done
    ;;
array)
    tasks="${1:?task list}"; shift
    tasks="$(cd "$(dirname "$tasks")" && pwd)/$(basename "$tasks")"
    time="" mem="" ids="" maxpar=20 submit=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --time) time="$2"; shift 2 ;;
            --mem) mem="$2"; shift 2 ;;
            --ids) ids="$2"; shift 2 ;;
            --max-parallel) maxpar="$2"; shift 2 ;;
            --submit) submit=1; shift ;;
            *) echo "unknown option $1" >&2; exit 1 ;;
        esac
    done
    [ -n "$time" ] && [ -n "$mem" ] || { echo "--time and --mem are required (see hpc/estimate_resources.py)" >&2; exit 1; }
    n=$(($(wc -l < "$tasks") - 1))
    ids="${ids:-1-$n}"
    # stop cleanly 10 minutes before the limit even if the USR1 signal is not delivered
    IFS=: read -r hh mm ss <<< "$time"
    maxwall=$(awk -v h="$hh" -v m="$mm" -v s="$ss" 'BEGIN { printf "%.3f", h + m / 60 + s / 3600 - 10 / 60 }')
    # job name toy_<tasklist stem>: hpc/status.py uses it to find this array's queued tasks
    cmd=(sbatch "${common[@]}" --job-name="toy_$(basename "$tasks" .tsv)" --array="${ids}%${maxpar}" --time="$time" --mem="$mem"
         --output="$LOG_ROOT/%x-%A_%a.out"
         --export="ALL,TOY_ENV_FILE=$ENV_FILE,TOY_HPC_DIR=$HERE,TASKLIST=$tasks,MAX_WALL_HOURS=$maxwall" "$HERE/array.sbatch")
    echo "${cmd[*]}"
    if [ "$submit" = 1 ]; then "${cmd[@]}"; else echo "(not submitted; add --submit to submit)"; fi
    ;;
*)
    sed -n '2,8p' "$0"; exit 1 ;;
esac
