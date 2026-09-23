#!/usr/bin/env bash
# Launch paper/run_all.sh detached via nohup, logging to a master file.
#
# The sweep keeps running after you log out / close the terminal. Per-combo logs
# are still written by run_all.sh under $LOG_DIR; this master log captures the
# wrapper's own stdout/stderr (including the run_all summary).
#
# By default run_all.sh sweeps the multimed dataset. Any env overrides are
# forwarded, e.g.:
#   DATASETS="multimed"           bash paper/run_all_nohup.sh
#   SETTINGS="rnnt_beam aed_beam" bash paper/run_all_nohup.sh

# Run from the repo root so run_all.sh's relative paths resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

LOG_DIR="${LOG_DIR:-paper/logs}"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/run_all_$(date +%Y%m%d_%H%M%S).log"

# Launch in a NEW session (own process group) so a single signal to the group
# tears down run_all.sh AND every child (run_all_steps.sh, run_inference.sh,
# the python eval jobs). With setsid the leader's PID == its PGID, so the
# process-group id to signal is just ${PID}.
setsid nohup bash paper/run_all.sh > "${MASTER_LOG}" 2>&1 &
PID=$!

echo "[run_all_nohup] started run_all.sh (PID/PGID ${PID})"
echo "[run_all_nohup] master log: ${MASTER_LOG}"
echo "[run_all_nohup] follow with: tail -f ${MASTER_LOG}"
echo "[run_all_nohup] stop with:   kill -TERM -- -${PID}   # whole process group"
echo "[run_all_nohup] force-stop:  kill -KILL -- -${PID}   # if it ignores TERM"
