#!/bin/bash
# Submission wrapper. Reads hpc/cluster.env (copy of cluster.env.example with every FILL_ME replaced).
#
#   hpc/submit.sh pilot [--dry-run]        (two pilot jobs: kernels A and B, N = 512)
#   hpc/submit.sh lj075-pilot [--dry-run]  (LJ law A short pilot: 4 single-core case jobs + 1 analysis job)
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
lj075-pilot)
    # LJ law A, N = 256: dense / fast O-step x dt 0.005 / 0.01 to t = 1 (lj075_hpc_pilot.py), then the analysis job.
    # Short jobs only; no array, no replicas. Time limits: about 4x the local case wall time, at least 10 min
    # (local: dense_dt0.005 7.3 min, dense_dt0.01 4.7 min, fast 1.2 / 0.8 min; hpc/README.md).
    root="$RUN_ROOT/lj075_pilot/$(date +%Y%m%dT%H%M%S)"
    dry=0; [ "${1:-}" = "--dry-run" ] && dry=1
    ids=()
    for spec in dense_dt0.005:00:30:00 fast_dt0.005:00:10:00 dense_dt0.01:00:20:00 fast_dt0.01:00:10:00; do
        C="${spec%%:*}"; T="${spec#*:}"
        cmd=(sbatch --parsable "${common[@]}" --job-name="lj075_pilot_$C" --time="$T" --output="$LOG_ROOT/%x-%j.out"
             --export="ALL,TOY_ENV_FILE=$ENV_FILE,TOY_HPC_DIR=$HERE,CASE=$C,PILOT_ROOT=$root" "$HERE/lj075_pilot_case.sbatch")
        if [ "$dry" = 1 ]; then echo "${cmd[*]}"; ids+=("JOBID_$C"); else id="$("${cmd[@]}")"; id="${id%%;*}"; ids+=("$id"); echo "$C: job $id"; fi
    done
    dep="$(IFS=:; echo "${ids[*]}")"                  # colons: a comma would split sbatch --export
    cmd=(sbatch --parsable "${common[@]}" --job-name=lj075_pilot_analyze --dependency="afterany:$dep"
         --output="$LOG_ROOT/%x-%j.out"
         --export="ALL,TOY_ENV_FILE=$ENV_FILE,TOY_HPC_DIR=$HERE,PILOT_ROOT=$root,PILOT_JOB_IDS=$dep" "$HERE/lj075_pilot_analyze.sbatch")
    if [ "$dry" = 1 ]; then echo "${cmd[*]}"; else id="$("${cmd[@]}")"; echo "analysis: job ${id%%;*} (after $dep)"; fi
    echo "pilot root: $root (report: $root/report/pilot_report.md)"
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
    sed -n '2,9p' "$0"; exit 1 ;;
esac
