#!/usr/bin/env bash
# Run the full pipeline (run_all_steps.sh) for every (dataset, setting) combo.
#
# The steps scripts honour externally-provided DATASET / SETTING, so
# we just loop over datasets x settings and invoke it for each. A failing combo
# does not abort the rest; failures are summarised at the end.
#
# Override the lists via env vars, e.g.:
#   DATASETS="multimed"           bash paper/run_all.sh
#   SETTINGS="rnnt_beam aed_beam" bash paper/run_all.sh
#
# Multi-GPU: set GPUS to the GPUs you want to use and each (dataset, setting)
# combo is scheduled onto its own GPU, up to ${#GPUS} combos running at once
# (a free GPU is reused as soon as a combo finishes). Examples:
#   GPUS="0 1 2 3"                       bash paper/run_all.sh   # 4-way parallel
#   GPUS="0,1,2,3,4,5,6,7"               bash paper/run_all.sh   # 8-way parallel
#   GPUS=2                               bash paper/run_all.sh   # serial on GPU 2
# Each combo gets CUDA_VISIBLE_DEVICES pinned to a single GPU; run_all_steps.sh
# honours that pin (see its CUDA_VISIBLE_DEVICES fallback).

# Run from the repo root so run_all_steps.sh's relative paths resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

# MultiMed: before any decoder combo, run_multimed_parakeet_wer_filter.sh builds a
# shared kept subset (Parakeet-TDT WER > 0.30 dropped). Combos then decode that
# subset. Skip the prelude with SKIP_PARAKEET_WER_FILTER=1. STOP is unchanged.
#
# Which pipeline variant each combo runs is chosen PER DATASET (see
# steps_script_for below): stop_music/stop_places have a released context list, so they run
# run_all_steps_full_list.sh, which boosts the eval+test entities that the model
# fails on in train (zero-recall of the eval+test list); multimed has no such
# list, so it runs the mining variant (discover the bias list from the model's
# own errors). Setting STEPS_SCRIPT explicitly overrides this per-dataset choice.
STEPS_SCRIPT="${STEPS_SCRIPT:-}"

# Map a dataset to the steps script it should run, unless STEPS_SCRIPT forces one.
steps_script_for () {
    local dataset="$1"
    if [ -n "${STEPS_SCRIPT}" ]; then
        echo "${STEPS_SCRIPT}"
        return
    fi
    case "${dataset}" in
        stop_music*|stop_places*) echo "paper/run_all_steps_full_list.sh" ;;
        *)             echo "paper/run_all_steps_zero_recall_medical.sh" ;;
    esac
}

DATASETS="${DATASETS:-multimed stop_music stop_places}"
SETTINGS="${SETTINGS:-aed_beam aed_greedy rnnt_beam rnnt_greedy ctc_beam ctc_greedy}"

# MultiMed Parakeet-TDT WER prelude (run_multimed_parakeet_wer_filter.sh).
#   0 — run it (skip automatically if parakeet_tdt_wer30/.done exists)
#   1 — skip the prelude
SKIP_PARAKEET_WER_FILTER="${SKIP_PARAKEET_WER_FILTER:-0}"

# GPUs to schedule combos onto. Accepts space- or comma-separated ids; the
# number of ids is the max number of combos run concurrently. Defaults to GPU 2.
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
read -ra GPU_POOL <<< "${GPUS//,/ }"
NUM_GPUS="${#GPU_POOL[@]}"

# Per-run logs: one file per (dataset, setting) under $LOG_DIR, tagged with a
# run-wide timestamp so re-runs don't clobber earlier logs.
LOG_DIR="${LOG_DIR:-paper/logs}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# Failures from background combos are collected in a temp file (one combo per
# line) since a backgrounded subshell can't append to a parent shell variable.
FAIL_FILE="$(mktemp "${TMPDIR:-/tmp}/run_all_failed.XXXXXX")"
trap 'rm -f "${FAIL_FILE}"' EXIT

echo "[run_all] scheduling combos across ${NUM_GPUS} GPU(s): ${GPU_POOL[*]}"

need_multimed_filter=0
for _ds in ${DATASETS}; do
    case "${_ds}" in
        multimed*) need_multimed_filter=1 ;;
    esac
done
PARAKEET_FILTERED_DIR="paper/data/multimed_nemo/parakeet_tdt_wer30"
if [ "${need_multimed_filter}" -eq 1 ] && [ "${SKIP_PARAKEET_WER_FILTER}" != 1 ]; then
    if [ -f "${PARAKEET_FILTERED_DIR}/.done" ]; then
        echo "[run_all] MultiMed Parakeet-TDT WER filter already done (${PARAKEET_FILTERED_DIR})"
    else
        gpu="${GPU_POOL[0]}"
        filter_log="${LOG_DIR}/multimed_parakeet_wer_filter_${RUN_TS}.log"
        echo "[run_all] MultiMed Parakeet-TDT WER filter on GPU ${gpu}  log: ${filter_log}"
        echo "[run_all] follow it with: tail -f ${filter_log}"
        CUDA_VISIBLE_DEVICES="${gpu}" NOHUP_INNER=1 \
            bash paper/run_multimed_parakeet_wer_filter.sh 2>&1 | tee "${filter_log}"
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            echo "[run_all] MultiMed Parakeet-TDT WER filter FAILED (log: ${filter_log})" >&2
            exit 1
        fi
        echo "[run_all] MultiMed Parakeet-TDT WER filter OK"
    fi
fi

# Run a single combo (pinned to one GPU) and record OK/FAILED.
run_combo () {
    local gpu="$1" dataset="$2" setting="$3"
    local steps_script; steps_script="$(steps_script_for "${dataset}")"
    local log="${LOG_DIR}/${dataset}_${setting}_${RUN_TS}.log"
    echo "================================================================"
    echo "=== ${steps_script}: DATASET=${dataset} SETTING=${setting} GPU=${gpu} ==="
    echo "=== full log: ${log}"
    echo "=== follow it with: tail -f ${log}"
    echo "================================================================"
    local t0="${SECONDS}"
    # Each combo's FULL output goes to its own per-combo log file ($log).
    # On stdout (the run_all_nohup master log) we keep only a compact progress
    # feed: the step/test markers the steps script prints, prefixed
    # with a wall-clock timestamp + the combo + GPU, so you can see exactly which
    # step each combo is on and how long each step took, without the combos'
    # output getting mixed together.
    #
    # The prefixing is a `while read` + `date` loop rather than sed/awk on
    # purpose: a timestamp baked into a sed expression is expanded once at
    # pipeline start (identical on every line), and the system awk here is mawk,
    # which buffers its *input* -- so strftime() would stamp a whole block of
    # lines with the moment the buffer flushed, not when each line was produced.
    # This feed is only a handful of marker lines per combo, so a `date` per line
    # is cheap.
    CUDA_VISIBLE_DEVICES="${gpu}" DATASET="${dataset}" SETTING="${setting}" \
        bash "${steps_script}" 2>&1 \
        | tee "${log}" \
        | grep --line-buffered -E '^(Step |=== |\[test|\[best-sl|\[summary)' \
        | while IFS= read -r line; do
              printf '%s [%s/%s|gpu%s] %s\n' \
                  "$(date '+[%Y-%m-%d %H:%M:%S]')" "${dataset}" "${setting}" "${gpu}" "${line}"
          done
    local status="${PIPESTATUS[0]}"
    local mins=$(( (SECONDS - t0) / 60 )) secs=$(( (SECONDS - t0) % 60 ))
    local stamp="[$(date '+%Y-%m-%d %H:%M:%S')]"
    if [ "${status}" -eq 0 ]; then
        echo "${stamp} [run_all] ${dataset}/${setting}: OK in ${mins}m${secs}s (log: ${log})"
    else
        echo "${stamp} [run_all] ${dataset}/${setting}: FAILED (exit ${status}, after ${mins}m${secs}s, log: ${log})" >&2
        echo "${dataset}/${setting}" >> "${FAIL_FILE}"
    fi
}

# GPU scheduler: keep a stack of free GPUs; when none are free, block until a
# running combo finishes and return its GPU to the pool.
declare -A PID_GPU       # pid -> gpu it is using
FREE_GPUS=("${GPU_POOL[@]}")

reap_one () {
    # Block until any background combo finishes, then free every GPU whose
    # combo has exited (wait -n may coincide with several finishing at once).
    wait -n
    local pid
    for pid in "${!PID_GPU[@]}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            FREE_GPUS+=("${PID_GPU[$pid]}")
            unset 'PID_GPU[$pid]'
        fi
    done
}

for DATASET in ${DATASETS}; do
    for SETTING in ${SETTINGS}; do
        # Wait for a free GPU.
        while [ "${#FREE_GPUS[@]}" -eq 0 ]; do
            reap_one
        done
        # Pop a free GPU and launch this combo on it.
        gpu="${FREE_GPUS[-1]}"
        unset 'FREE_GPUS[-1]'
        FREE_GPUS=("${FREE_GPUS[@]}")
        run_combo "${gpu}" "${DATASET}" "${SETTING}" &
        PID_GPU[$!]="${gpu}"
    done
done

# Drain the remaining running combos.
while [ "${#PID_GPU[@]}" -gt 0 ]; do
    reap_one
done

echo "================================================================"
if [ -s "${FAIL_FILE}" ]; then
    echo "[run_all] FAILED combos: $(paste -sd' ' "${FAIL_FILE}")" >&2
    exit 1
fi
echo "[run_all] all (dataset, setting) combos completed successfully"
