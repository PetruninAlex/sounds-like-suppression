#!/usr/bin/env bash
# MultiMed phase 1: Parakeet-TDT reference-quality filter (WER > 0.30).
#
# Strips speaker labels and [...] caption tags, decodes train/eval/test with
# nvidia/parakeet-tdt-0.6b-v2 (not in the experiment table), drops utterances
# whose WER exceeds 0.30, and writes a shared kept manifest set. run_all.sh
# runs this in-process (NOHUP_INNER=1) before MultiMed decoder combos.
#
# From repo root (detaches immediately; log under paper/logs/):
#   CUDA_VISIBLE_DEVICES=0 bash paper/run_multimed_parakeet_wer_filter.sh
#   tail -f paper/logs/multimed_parakeet_wer_filter_*.log
#
# Outputs:
#   paper/data/predictions/${DATASET}/parakeet_tdt/baseline{,_train,_eval}.json
#   paper/data/predictions/${DATASET}/parakeet_tdt/*.wer_filtered.json
#   ${FILTERED_DIR}/{train,eval,test}.json   (shared kept subset)

export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

LOG_DIR="${LOG_DIR:-paper/logs}"
mkdir -p "${LOG_DIR}"
if [ "${NOHUP_INNER:-0}" != 1 ]; then
    LOG="${LOG:-${LOG_DIR}/multimed_parakeet_wer_filter_$(date +%Y%m%d_%H%M%S).log}"
    echo "[wer-filter] nohup log: ${LOG}"
    echo "[wer-filter] follow with: tail -f ${LOG}"
    NOHUP_INNER=1 nohup bash "${SCRIPT_DIR}/run_multimed_parakeet_wer_filter.sh" \
        >> "${LOG}" 2>&1 &
    echo "[wer-filter] pid $!  (echo $! > ${LOG_DIR}/multimed_parakeet_wer_filter.pid)"
    echo $! > "${LOG_DIR}/multimed_parakeet_wer_filter.pid"
    exit 0
fi

set -euo pipefail

export DATASET="${DATASET:-multimed}"
RUN_TAG="${RUN_TAG:-_zr_medical}"
case "${DATASET}" in
    *"${RUN_TAG}") SRC_DATASET="${DATASET%${RUN_TAG}}" ;;
    *)             SRC_DATASET="${DATASET}"
                   export DATASET="${DATASET}${RUN_TAG}" ;;
esac
export SETTING="${SETTING:-parakeet_tdt}"
if [ "${SETTING}" != "parakeet_tdt" ]; then
    echo "[fatal] this script must run SETTING=parakeet_tdt (got ${SETTING})" >&2
    exit 1
fi

WER_FILTER_MAX="${WER_FILTER_MAX:-0.30}"
SRC_MANIFEST_DIR="paper/data/${SRC_DATASET}_nemo"
FILTERED_DIR="${FILTERED_DIR:-${SRC_MANIFEST_DIR}/parakeet_tdt_wer30}"
CLEAN_DIR="paper/data/${SRC_DATASET}_nemo/clean/${DATASET}/${SETTING}"
PRED_DIR="paper/data/predictions/${DATASET}/${SETTING}"

echo "[wer-filter] source=${SRC_DATASET}  outputs=${DATASET}  "\
"setting=${SETTING}  max-wer=${WER_FILTER_MAX}  filtered=${FILTERED_DIR}"

for _split in train eval test; do
    if [ ! -f "${SRC_MANIFEST_DIR}/${_split}.json" ]; then
        echo "[fatal] missing ${SRC_MANIFEST_DIR}/${_split}.json" >&2
        exit 1
    fi
done

echo "Step 2b: Stripping speaker labels"
python paper/analysis/strip_speaker_labels.py \
    --input  "${SRC_MANIFEST_DIR}/train.json" \
             "${SRC_MANIFEST_DIR}/eval.json" \
             "${SRC_MANIFEST_DIR}/test.json" \
    --out-dir "${CLEAN_DIR}"

echo "Step 2c: Stripping [...] caption tags"
python paper/analysis/strip_bracket_tags.py \
    --input "${CLEAN_DIR}/train.json" \
            "${CLEAN_DIR}/eval.json" \
            "${CLEAN_DIR}/test.json"

echo "Step 3: Parakeet-TDT train inference"
DATASET="${DATASET}" SETTING="${SETTING}" \
MANIFEST="${CLEAN_DIR}/train.json" \
    bash paper/inference/run_inference.sh

echo "Step 3b: Parakeet-TDT validation inference"
DATASET="${DATASET}" SETTING="${SETTING}" \
MANIFEST="${CLEAN_DIR}/eval.json" \
    bash paper/inference/run_inference.sh

echo "Step 3c: Parakeet-TDT test inference"
DATASET="${DATASET}" SETTING="${SETTING}" \
MANIFEST="${CLEAN_DIR}/test.json" \
    bash paper/inference/run_inference.sh

echo "Step 4: WER filter train (max-wer=${WER_FILTER_MAX})"
python paper/analysis/filter_predictions_by_wer.py \
    --predictions "${PRED_DIR}/baseline_train.json" \
    --out         "${PRED_DIR}/baseline_train.wer_filtered.json" \
    --max-wer "${WER_FILTER_MAX}"
echo "Step 4b: WER filter validation"
python paper/analysis/filter_predictions_by_wer.py \
    --predictions "${PRED_DIR}/baseline_eval.json" \
    --out         "${PRED_DIR}/baseline_eval.wer_filtered.json" \
    --max-wer "${WER_FILTER_MAX}"
echo "Step 4c: WER filter test"
python paper/analysis/filter_predictions_by_wer.py \
    --predictions "${PRED_DIR}/baseline.json" \
    --out         "${PRED_DIR}/baseline.wer_filtered.json" \
    --max-wer "${WER_FILTER_MAX}"

echo "Writing shared kept manifests to ${FILTERED_DIR}"
mkdir -p "${FILTERED_DIR}"
python paper/analysis/subset_manifest_by_ids.py \
    --manifest "${CLEAN_DIR}/train.json" \
    --keep-from "${PRED_DIR}/baseline_train.wer_filtered.json" \
    --out "${FILTERED_DIR}/train.json"
python paper/analysis/subset_manifest_by_ids.py \
    --manifest "${CLEAN_DIR}/eval.json" \
    --keep-from "${PRED_DIR}/baseline_eval.wer_filtered.json" \
    --out "${FILTERED_DIR}/eval.json"
python paper/analysis/subset_manifest_by_ids.py \
    --manifest "${CLEAN_DIR}/test.json" \
    --keep-from "${PRED_DIR}/baseline.wer_filtered.json" \
    --out "${FILTERED_DIR}/test.json"

date -Iseconds > "${FILTERED_DIR}/.done"
echo "[wer-filter] done. Shared manifests: ${FILTERED_DIR}/{train,eval,test}.json"
echo "[wer-filter] Next: run_all.sh MultiMed combos use ${FILTERED_DIR}"
