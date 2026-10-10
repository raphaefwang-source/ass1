#!/bin/bash
# Local emulation of "hpc/submit.sh lj075-pilot" on a machine WITHOUT Slurm: runs the same job scripts under bash, the
# four cases one after another (so their step times do not compete for cores), then the analysis job.
# This is a smoke test of the scripts and the comparison; it is NOT an HPC validation, and the report it writes says
# so (no Slurm job id on any segment).
#
#   TOY_ENV_FILE=/path/to/local.env hpc/lj075_pilot_local.sh [PILOT_ROOT]
#
# local.env needs the same keys as cluster.env (SLURM_ACCOUNT / SLURM_PARTITION may be any non-placeholder text here,
# they are not used).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TOY_ENV_FILE:?TOY_ENV_FILE not set}"
source "$HERE/common.sh"
toy_load_env "$TOY_ENV_FILE" || exit 1
root="${1:-$RUN_ROOT/lj075_pilot/local-$(date +%Y%m%dT%H%M%S)}"
mkdir -p "$root" "$LOG_ROOT"
rcs=()
for C in dense_dt0.005 fast_dt0.005 dense_dt0.01 fast_dt0.01; do
    TOY_ENV_FILE="$TOY_ENV_FILE" TOY_HPC_DIR="$HERE" CASE="$C" PILOT_ROOT="$root" \
        bash "$HERE/lj075_pilot_case.sbatch" > "$LOG_ROOT/lj075_pilot_$C-local.out" 2>&1
    rcs+=("$C=$?")
done
TOY_ENV_FILE="$TOY_ENV_FILE" TOY_HPC_DIR="$HERE" PILOT_ROOT="$root" \
    bash "$HERE/lj075_pilot_analyze.sbatch" > "$LOG_ROOT/lj075_pilot_analyze-local.out" 2>&1
arc=$?
echo "case exit codes: ${rcs[*]}; analysis exit $arc"
echo "pilot root: $root (report in $root/report)"
