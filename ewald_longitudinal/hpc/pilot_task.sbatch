#!/bin/bash
# HPC pilot, one job per kernel: N = 512, costopt_hiacc, double well, seed 101, PILOT_STEPS steps (default 1500).
# Purpose: check speed, memory, the accuracy monitor, the directory lock and signal / resume on the real cluster.
# One core, one thread, no GPU.
# Submit with:  hpc/submit.sh pilot   (two jobs, KERNEL=A and KERNEL=B; adds --account/--partition/--output).
#
# Sequence inside the job (all on $RUN_ROOT):
#  0. env_report.json; lock_check on $RUN_ROOT/pilot (the cluster file system must lock coherently).
#  1. Segment 1 runs in the background behind the same signal trap as array.sbatch. Slurm sends USR1 about 4 min
#     after the start (--signal=B:USR1@2160 with a 40 min limit; Slurm may send it up to 60 s early). The run
#     checkpoints and exits 75. If it completes first, "signal not exercised" is recorded.
#  2. While segment 2 runs, a duplicate "toy_run.py --resume" on the same directory must exit 3 (lock) and change
#     nothing. Segment 2 stops after 200 steps (exit 75).
#  3. Resume to completion (exit 0), repeating if a stray signal stops it.
#  4. check_run.py (acceptance), restart_check.py (N = 64, bitwise), pilot_summary.json.
#
# Local emulation only (no Slurm): PILOT_LOCAL_SIGNAL_AFTER=<s> makes the script send itself USR1 after <s> seconds.
# A local run of this script is not an HPC validation.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:40:00
#SBATCH --signal=B:USR1@2160
set -uo pipefail
: "${TOY_ENV_FILE:?TOY_ENV_FILE not set (use hpc/submit.sh)}" "${KERNEL:?KERNEL (A or B) not set}"
source "${TOY_HPC_DIR:?TOY_HPC_DIR not set (use hpc/submit.sh)}/common.sh"
toy_load_env "$TOY_ENV_FILE" || exit 1
toy_threads
eval "$PY_SETUP"
cd "$REPO_DIR/ewald_longitudinal" || exit 1
STEPS="${PILOT_STEPS:-1500}"
SAVE="${PILOT_SAVE_EVERY_STEPS:-1}"                       # production interval: every step (vacf_sampling_results)
PILOT_DIR="$RUN_ROOT/pilot/${SLURM_JOB_ID:-local-$(date +%Y%m%dT%H%M%S)}_${KERNEL}"
RUN="$PILOT_DIR/double_well_${KERNEL}_N512_s101"
mkdir -p "$PILOT_DIR"
T0=$(date +%s)
echo "pilot $KERNEL: $RUN ($STEPS steps, frames every $SAVE steps)"
"$PYTHON" -u hpc/env_report.py > "$PILOT_DIR/env_report.json"
"$PYTHON" -u hpc/lock_check.py --dir "$RUN_ROOT/pilot" --json "$PILOT_DIR/lock_check.json" > /dev/null
lock_rc=$?
ARGS=(--config costopt_hiacc --potential double_well --kernel "$KERNEL" --N 512 --seed 101 --burn-in-steps 0
      --production-steps "$STEPS" --save-every-steps "$SAVE" --checkpoint-every-steps 2000 --monitor-every-steps 50
      --out "$RUN")

pid=""
trap 'echo "$(date +%T) batch shell got a signal; forwarding USR1 to ${pid:-none}"; [ -n "$pid" ] && kill -USR1 "$pid" 2>/dev/null' USR1 TERM
if [ -n "${PILOT_LOCAL_SIGNAL_AFTER:-}" ]; then
    ( sleep "$PILOT_LOCAL_SIGNAL_AFTER"; kill -USR1 $$ ) &            # local emulation of Slurm's --signal
fi
run_bg() {                                                           # run toy_run.py behind the trap, return its code
    "$PYTHON" -u toy_run.py "$@" &
    pid=$!
    local rc=0
    while true; do
        wait "$pid"; rc=$?
        kill -0 "$pid" 2>/dev/null || break
    done
    pid=""
    return "$rc"
}

run_bg "${ARGS[@]}"; rc1=$?
echo "segment 1 exit $rc1 (75 expected: stopped by the scheduler's USR1)"

dup_rc=""
rc2=""
if [ "$rc1" = 75 ]; then
    "$PYTHON" -u toy_run.py --resume --out "$RUN" --max-segment-steps 200 &
    pid=$!
    for i in $(seq 120); do                                          # until segment 2 holds the lock
        [ "$("$PYTHON" -c 'import sys; sys.path.insert(0, "."); import run_lock; print(run_lock.probe(sys.argv[1]))' "$RUN")" = held ] && break
        sleep 0.5
    done
    "$PYTHON" -u toy_run.py --resume --out "$RUN" --lock-wait 2 > "$PILOT_DIR/duplicate.log" 2>&1; dup_rc=$?
    echo "duplicate on the same directory exit $dup_rc (3 expected: refused by the lock)"
    while true; do wait "$pid"; rc2=$?; kill -0 "$pid" 2>/dev/null || break; done
    pid=""
    echo "segment 2 exit $rc2 (75 expected: 200-step segment limit)"
fi
rc3=75
for attempt in 1 2 3 4; do
    [ "$rc3" = 75 ] || break
    st=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['state'])" "$RUN/status.json" 2>/dev/null)
    [ "$st" = complete ] && { rc3=0; break; }
    run_bg --resume --out "$RUN"; rc3=$?
    echo "resume attempt $attempt exit $rc3"
done
"$PYTHON" -u hpc/check_run.py "$RUN" --expect-config costopt_hiacc --json "$PILOT_DIR/pilot_check.json"
check_rc=$?
"$PYTHON" -u hpc/restart_check.py --work "$PILOT_DIR/restart_check" --N 64 --kernel "$KERNEL" --steps 60 \
    --segments 25 20 > "$PILOT_DIR/restart_check.log" 2>&1
restart_rc=$?
T1=$(date +%s)
"$PYTHON" - "$RUN" "$PILOT_DIR" "$rc1" "${rc2:-}" "${dup_rc:-}" "$rc3" "$check_rc" "$restart_rc" "$lock_rc" "$((T1 - T0))" <<'EOF'
import json, sys
from pathlib import Path
run, pdir, rc1, rc2, dup, rc3, chk, rst, lck, wall = sys.argv[1:]
st = json.loads((Path(run) / "status.json").read_text())
chkj = json.loads((Path(pdir) / "pilot_check.json").read_text())[0]
seg = st.get("segments", [])
summ = dict(
    kernel=chkj["kernel"], N=chkj["N"], steps=st.get("step"), total=st.get("total_steps"),
    segment_exit_codes=dict(segment1=int(rc1), segment2=(int(rc2) if rc2 else None), final=int(rc3)),
    duplicate_exit_code=(int(dup) if dup else None),
    signal_exercised=int(rc1) == 75 and "SIGUSR1" in (seg[0].get("stop_reason") or ""),
    stop_reasons=[s_.get("stop_reason") for s_ in seg],
    median_step_s_per_segment=[s_.get("median_step_wall_s") for s_ in seg],
    step_ms_median=chkj["step_ms_median"], step_ms_p90=chkj["step_ms_p90"], peak_rss_mb=st.get("peak_rss_mb"),
    run_wall_s=sum(s_.get("wall_s") or 0 for s_ in seg), job_wall_s=int(wall),
    accuracy_warnings=len(st.get("accuracy_warnings") or []), monitor_err_max=chkj["monitor_err_max"],
    S_kmin_max=chkj["S_kmin_max"], check_run_passed=chkj["passed"], restart_check_passed=int(rst) == 0,
    lock_check_passed=int(lck) == 0, mesh_M=chkj["mesh_M"], ranks=chkj["ranks"], host=chkj["host"])
summ["accepted"] = (summ["signal_exercised"] and summ["duplicate_exit_code"] == 3 and summ["segment_exit_codes"]["final"] == 0
                    and summ["check_run_passed"] and summ["restart_check_passed"] and summ["lock_check_passed"]
                    and summ["accuracy_warnings"] == 0)
(Path(pdir) / "pilot_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
print(json.dumps(summ, indent=1))
print("PILOT", summ["kernel"], "ACCEPTED" if summ["accepted"] else "NOT ACCEPTED (see pilot_summary.json)")
sys.exit(0 if summ["accepted"] else 1)
EOF
