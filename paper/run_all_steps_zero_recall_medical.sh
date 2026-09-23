#!/usr/bin/env bash
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Force protobuf's pure-Python backend. The installed protobuf (5.x) is
# incompatible with onnx's generated *_pb2.py descriptors, so importing NeMo ASR
# dies with "Descriptors cannot be created directly" under the default C++
# backend. The pure-Python parser skips that check; the perf hit is negligible
# here (protobuf is only touched at model/config load, not during decoding).
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# GPU pin: honour an externally-provided CUDA_VISIBLE_DEVICES (e.g. set per-combo
# by run_all.sh so each (dataset, setting) lands on its own GPU). Falls back to
# GPU 0 when run standalone.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Checkpoint location. Defaults to a `models` directory next to the repo, i.e.
# <repo>/../models, matching paper/inference/*.sh. Override MODELS_DIR, or point
# HYBRID_MODEL / CANARY_MODEL straight at a .nemo file.
MODELS_DIR="${MODELS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/models}"

export DATASET="${DATASET:-multimed}"
# Variant output isolation. Every artifact this script writes lands under
# paper/data/predictions/${DATASET}${RUN_TAG}/ so the phrase-mining strategies can
# be compared side by side without overwriting each other's results.
# SRC_DATASET keeps pointing at the real dataset directory
# (paper/data/<name>_nemo) that holds the source manifests, since only the
# OUTPUT location is tagged -- not the input data.
RUN_TAG="${RUN_TAG:-}"
case "${DATASET}" in
    *"${RUN_TAG}") SRC_DATASET="${DATASET%${RUN_TAG}}" ;;   # already tagged; don't double-tag
    *)             SRC_DATASET="${DATASET}"
                   export DATASET="${DATASET}${RUN_TAG}" ;;
esac
echo "[variant] strategy=zero-recall-of-LLM-medical-terms  source data=${SRC_DATASET}  outputs=${DATASET}"
export SETTING="${SETTING:-rnnt_beam}"

# This variant differs from run_all_steps_zero_recall.sh ONLY in how the
# candidate boosting phrases are found (Steps 5-7 below). Instead of mining
# candidates from the model's own errors and asking the LLM which are
# specialised terms, we:
#   (5)  ask the LLM (GPT-5.5) to tag MEDICAL terms in each VALIDATION + TEST
#        reference line (extract_boost_phrases.py, medical tagger only), giving
#        the candidate universe;
#   (6)  keep only those tagged terms the baseline never gets right on TRAIN
#        (recall == 0) with train count >= ZR_MIN_COUNT, using the same
#        find_zero_recall_terms_from_list.py as the STOP pipeline. So the
#        candidate SET comes from eval+test, but which of them are actually
#        boosted is decided by the model's TRAIN errors.
# Everything after that (surface forms, tokenizability, boost lists, the
# validation boost sweep, sounds-like discovery, and the test
# evaluation) is identical to the zero-recall variant.
#
# NOTE ON LEAKAGE: because the candidate terms are tagged from the test (and
# eval) references, a term only survives if it ALSO appears in train and the
# baseline fails it on train (recall==0, count>=ZR_MIN_COUNT). Terms that occur
# only in test/eval and never in train are dropped by Step 6 (they are absent
# from the train alignment). Reviewers may still flag that the candidate net is
# drawn from the scored split; if strict held-out construction is required, tag
# from TRAIN instead (--manifest "${TRAIN_MANIFEST}" in Step 5).

# Max WER deterioration (percentage points over the NO-BOOST baseline) a config
# may incur and still be eligible; measured against that same baseline for both
# the boost and sounds-like stages so the cap bounds the TOTAL cost of biasing.
# Set to a negative value to disable and pick purely on the metric.
export MAX_WER_DETERIORATION="${MAX_WER_DETERIORATION:-1.0}"
# Sweep ranges for the boost value and the sounds-like suppression weight.
export BOOST_MIN="${BOOST_MIN:-1}"
export BOOST_MAX="${BOOST_MAX:-10}"
export SL_MIN_WEIGHT="${SL_MIN_WEIGHT:-1}"
export SL_MAX_WEIGHT="${SL_MAX_WEIGHT:-10}"
# Boosting-tree depth scaling (gamma). Hybrid = CTC/RNN-T; AED = Canary.
# Inference wrappers pick these up as boosting_tree.depth_scaling.
export BT_DEPTH_SCALING_HYBRID="${BT_DEPTH_SCALING_HYBRID:-1.0}"
export BT_DEPTH_SCALING_AED="${BT_DEPTH_SCALING_AED:-1.0}"

# Decoding-throughput (RTFx) measurement for the three test methods (Step 13).
export MEASURE_RTFX="${MEASURE_RTFX:-1}"
export RTFX_WARMUP_STEPS="${RTFX_WARMUP_STEPS:-1}"

# Azure OpenAI secrets for the LLM steps (medical tagging in Step 5, sounds-like
# during sounds-like discovery). Sourced from a gitignored env file (repo-root/.env by
# default; override with ENV_FILE=/path/to/file). See .env.example for the
# template. `set -a` exports every var the file defines.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${_SCRIPT_DIR}/../.env}"
if [ -f "${ENV_FILE}" ]; then
    set -a
    # shellcheck disable=SC1090
    . "${ENV_FILE}"
    set +a
else
    echo "[warn] env file not found: ${ENV_FILE} -- the LLM steps (5 and the" \
         "sounds-like discovery) will fail without the AZURE_OPENAI_*_gpt5_5 vars." \
         "Copy .env.example to .env" >&2
fi

# Step 1: Dataset preparation is expected to have happened already: MultiMed
# ships its manifests out of band. The shared Parakeet-TDT WER subset is built
# by run_multimed_parakeet_wer_filter.sh (run_all.sh runs it before any MultiMed combo).
FILTERED_DIR="paper/data/${SRC_DATASET}_nemo/parakeet_tdt_wer30"
for _split in train eval test; do
    if [ ! -f "${FILTERED_DIR}/${_split}.json" ]; then
        echo "[fatal] missing ${FILTERED_DIR}/${_split}.json --" \
             "run paper/run_multimed_parakeet_wer_filter.sh first (or run_all.sh)." >&2
        exit 1
    fi
done
if [ ! -f "${FILTERED_DIR}/.done" ]; then
    echo "[fatal] ${FILTERED_DIR}/.done missing -- Parakeet-TDT WER filter did not finish." >&2
    exit 1
fi
echo "[variant] using shared Parakeet-TDT WER subset ${FILTERED_DIR}"

# Step 2b: Stage a per-setting copy of the already-filtered manifests.
# script. The copy is only so parallel settings do not race on inference
# split files next to the shared manifests (same reason as STOP).
CLEAN_DIR="paper/data/${SRC_DATASET}_nemo/clean/${DATASET}/${SETTING}"
echo "Step 2b: Staging filtered manifests for ${DATASET} ${SETTING}"
mkdir -p "${CLEAN_DIR}"
cp "${FILTERED_DIR}/train.json" \
   "${FILTERED_DIR}/eval.json" \
   "${FILTERED_DIR}/test.json" \
   "${CLEAN_DIR}/"

# Step 3: Train baseline inference on the filtered train manifest.
PRED_DIR="paper/data/predictions/${DATASET}/${SETTING}"
echo "Step 3: Running inference for ${DATASET} ${SETTING}"
DATASET="${DATASET}" SETTING="${SETTING}" \
MANIFEST="${CLEAN_DIR}/train.json" \
    bash paper/inference/run_inference.sh

echo "Step 3b: Running validation (eval) inference for ${DATASET} ${SETTING}"
DATASET="${DATASET}" SETTING="${SETTING}" \
MANIFEST="${CLEAN_DIR}/eval.json" \
    bash paper/inference/run_inference.sh

echo "Step 3c: Running test inference (no biasing) for ${DATASET} ${SETTING}"
DATASET="${DATASET}" SETTING="${SETTING}" \
MANIFEST="${CLEAN_DIR}/test.json" \
    bash paper/inference/run_inference.sh

# Subset is already the Parakeet-TDT WER-kept set; keep the *.wer_filtered.json
# names the rest of the pipeline expects.
echo "Step 4: copying baselines to *.wer_filtered.json (shared Parakeet-TDT subset)"
cp "${PRED_DIR}/baseline_train.json" "${PRED_DIR}/baseline_train.wer_filtered.json"
cp "${PRED_DIR}/baseline_eval.json"  "${PRED_DIR}/baseline_eval.wer_filtered.json"
cp "${PRED_DIR}/baseline.json"       "${PRED_DIR}/baseline.wer_filtered.json"

TRAIN_MANIFEST="paper/data/predictions/${DATASET}/${SETTING}/baseline_train.wer_filtered.json"
VAL_MANIFEST="paper/data/predictions/${DATASET}/${SETTING}/baseline_eval.wer_filtered.json"

# Tokenizer model (needed by Step 6 recall/n-gram matching's downstream
# tokenizability filter in Step 7b, and by the sounds-like token recovery).
case "${SETTING}" in
    aed_*) TOKENIZER_MODEL="${CANARY_MODEL:-${MODELS_DIR}/canary-1b/canary-1b.nemo}" ;;
    *)     TOKENIZER_MODEL="${HYBRID_MODEL:-${MODELS_DIR}/stt_en_fastconformer_hybrid_large_pc/stt_en_fastconformer_hybrid_large_pc.nemo}" ;;
esac

# Candidate-selection knob.
#   ZR_MIN_COUNT - kept at 1: a candidate is boosted if the baseline fails it
#                  even once on TRAIN (recall == 0). The val+test presence filter
#                  (Step 5b) already removes thin / single-split terms, so the
#                  train side stays permissive.
# Candidate terms must occur at least once in BOTH val and test (Step 5b), so
# every boosted term is selectable on val and scored on test.
ZR_MIN_COUNT="${ZR_MIN_COUNT:-1}"

PHRASE_DIR="paper/data/predictions/${DATASET}/${SETTING}"
# Candidate medical terms are tagged from the VALIDATION + TEST references
# (Step 5); the WER-filtered test predictions double as the tagging source and,
# later, as the Step 13 test manifest.
TEST_MANIFEST_FOR_TAGGING="${PHRASE_DIR}/baseline.wer_filtered.json"
MEDICAL_TERMS="${PHRASE_DIR}/boost_phrases.medical_valtest.tsv"
MEDICAL_TERMS_LIST="${PHRASE_DIR}/boost_phrases.medical_valtest.txt"
# The zero-recall + min-count filter (Step 6) is measured on TRAIN: a candidate
# term is boosted only if the baseline fails it on the train split it was
# decoded on.
ZERO_RECALL_MEDICAL="${PHRASE_DIR}/baseline_train.wer_filtered_zero_recall_medical.tsv"

# Step 5: LLM-tag medical terms in the VALIDATION + TEST REFERENCE sentences.
# This is the candidate universe; the zero-recall filter that decides which of
# them to actually boost is applied on TRAIN in Step 6. extract_boost_phrases.py
# reads only the `text` (reference) field of each manifest line and, with
# --no-spacy, runs only the GPT-5.5 medical tagger (diseases, procedures,
# medications, clinical conditions). --min-count 1 keeps every tagged term
# (Step 5b enforces the val+test presence rule).
echo "Step 5: LLM-tagging medical terms in VALIDATION + TEST reference sentences for ${DATASET} ${SETTING}"
python paper/analysis/extract_boost_phrases.py \
    --manifest "${VAL_MANIFEST}" "${TEST_MANIFEST_FOR_TAGGING}" \
    --output   "${MEDICAL_TERMS}" \
    --backend azure --azure-suffix gpt5_5 \
    --no-spacy \
    --max-concurrent 8 \
    --min-count 1

# Step 5b: keep only terms occurring at least once in BOTH the val and test
# references, so every boosted term is selectable on val and scored on test
# (drops single-split terms that only add WER risk without measurable upside).
MEDICAL_TERMS_PRESENT="${PHRASE_DIR}/boost_phrases.medical_valtest.present.tsv"
echo "Step 5b: keeping medical terms present in BOTH val and test for ${DATASET} ${SETTING}"
python paper/analysis/filter_terms_present_in_each.py \
    --input    "${MEDICAL_TERMS}" \
    --manifest "${VAL_MANIFEST}" "${TEST_MANIFEST_FOR_TAGGING}" \
    --output   "${MEDICAL_TERMS_PRESENT}"
# Flatten the surviving terms to a one-per-line list for the zero-recall filter.
awk -F'\t' 'NR>1{print $1}' "${MEDICAL_TERMS_PRESENT}" > "${MEDICAL_TERMS_LIST}"

# Step 6: Of the val+test-tagged medical terms, keep only those the baseline
# never gets right on TRAIN (recall == 0) with train count >= ZR_MIN_COUNT.
# So the candidate SET comes from the eval+test references, but which of them
# are actually boosted is decided by the model's TRAIN errors. This is the same
# filter the STOP full-list pipeline uses (find_zero_recall_terms_from_list.py),
# which searches each term directly in the train predictions -- no n-gram window
# limit, so multi-word terms of any length are matched.
echo "Step 6: Selecting zero-recall medical terms (min-count=${ZR_MIN_COUNT}) for ${DATASET} ${SETTING}"
python paper/analysis/find_zero_recall_terms_from_list.py \
    --list        "${MEDICAL_TERMS_LIST}" \
    --predictions "${TRAIN_MANIFEST}" \
    --output      "${ZERO_RECALL_MEDICAL}" \
    --min-count   "${ZR_MIN_COUNT}" \
    --max-recall  "${ZR_MAX_RECALL:-0.0}"

# Step 7: Recover the original surface spelling of each surviving term. The
# survivors are zero-recall on TRAIN, so they occur in train -- recover casing
# from the train references.
echo "Step 7: Finding correct surface forms for candidate phrases for ${DATASET} ${SETTING}"
python paper/analysis/find_correct_word_forms.py \
    --input    "${ZERO_RECALL_MEDICAL}" \
    --manifest "${TRAIN_MANIFEST}" \
    --output   paper/data/predictions/${DATASET}/${SETTING}/boost_phrases.correct_forms.tsv

# Step 7b: Drop correct surface forms the model tokenizer cannot represent.
echo "Step 7b: Dropping untokenizable surface forms for ${DATASET} ${SETTING}"
python paper/analysis/filter_untokenizable_phrases.py \
    --input  paper/data/predictions/${DATASET}/${SETTING}/boost_phrases.correct_forms.tsv \
    --output paper/data/predictions/${DATASET}/${SETTING}/boost_phrases.correct_forms.tokenizable.tsv \
    --model  "${TOKENIZER_MODEL}"

# Step 8: Build boosting lists from the (tokenizable) correct word forms.
echo "Step 8: Building boosting lists from correct word forms for ${DATASET} ${SETTING}"
python paper/analysis/build_boost_lists.py \
    --input   paper/data/predictions/${DATASET}/${SETTING}/boost_phrases.correct_forms.tokenizable.tsv \
    --out-dir paper/data/predictions/${DATASET}/${SETTING} \
    --max-boost "${BOOST_MAX}"

# Step 9: Boost-value sweep on VALIDATION (dev picks the weight, per TurboBias).
BOOST_VAL_SUBDIR="boost_only_transcriptions_validation"
echo "Step 9: Running biased inference sweep (original boost) on validation for ${DATASET} ${SETTING}"
for BOOST_VALUE in $(seq "${BOOST_MIN}" "${BOOST_MAX}"); do
        DATASET="${DATASET}" \
    SETTING="${SETTING}" \
    BOOST_SUFFIX="_b${BOOST_VALUE}" \
    KEY_PHRASES="paper/data/predictions/${DATASET}/${SETTING}/boost_only_files/boost${BOOST_VALUE}.txt" \
    MANIFEST="${VAL_MANIFEST}" \
    OUT_SUBDIR="${BOOST_VAL_SUBDIR}" \
        bash paper/inference/run_inference_with_boosting.sh
done

# Step 10: Per-word F1 sweep for the validation boost-only transcriptions.
echo "Step 10: Summarising per-word F1 sweep (original boost, validation) for ${DATASET} ${SETTING}"
python paper/analysis/summarize_boost_f1.py \
    --dataset "${DATASET}" \
    --setting "${SETTING}" \
    --max-boost "${BOOST_MAX}" \
    --subdir "${BOOST_VAL_SUBDIR}" \
    --baseline-manifest "${VAL_MANIFEST}"

BOOST_VAL_TSV="paper/data/predictions/${DATASET}/${SETTING}/${BOOST_VAL_SUBDIR}/f1_all.tsv"
VAL_BOOST_BASELINE_WER=$(awk -F'\t' '$1=="WER"{print $2; exit}' "${BOOST_VAL_TSV}")
echo "No-boost validation baseline WER: ${VAL_BOOST_BASELINE_WER}"

# Step 11: Select the boost value on validation, under the WER budget.
echo "Step 11: Finding best boost value (validation) for ${DATASET} ${SETTING}"
BEST_BOOST=$(python paper/analysis/find_best_boost.py \
    --dataset "${DATASET}" \
    --setting "${SETTING}" \
    --subdir "${BOOST_VAL_SUBDIR}" \
    --max-wer-deterioration "${MAX_WER_DETERIORATION}" \
    --baseline-wer "${VAL_BOOST_BASELINE_WER}")
BEST_BOOST_F1=$(python paper/analysis/find_best_boost.py \
    --dataset "${DATASET}" \
    --setting "${SETTING}" \
    --subdir "${BOOST_VAL_SUBDIR}" \
    --max-wer-deterioration "${MAX_WER_DETERIORATION}" \
    --baseline-wer "${VAL_BOOST_BASELINE_WER}" \
    --print f1)

BOOST_ONLY_VAL_WER=$(awk -F'\t' -v c="$((2 + BEST_BOOST))" '$1=="WER"{print $c; exit}' "${BOOST_VAL_TSV}")
echo "Boost-only (B=${BEST_BOOST}) validation baseline WER: ${BOOST_ONLY_VAL_WER}"

# Step 11b: Decode TRAIN once at the selected value (the mining input).
echo "Step 11b: Train inference at selected boost value B=${BEST_BOOST} for ${DATASET} ${SETTING}"
DATASET="${DATASET}" \
SETTING="${SETTING}" \
BOOST_SUFFIX="_b${BEST_BOOST}" \
KEY_PHRASES="paper/data/predictions/${DATASET}/${SETTING}/boost_only_files/boost${BEST_BOOST}.txt" \
MANIFEST="${TRAIN_MANIFEST}" \
OUT_SUBDIR="boost_only_transcriptions" \
    bash paper/inference/run_inference_with_boosting.sh
echo "Step 11c: Summarising train F1 at B=${BEST_BOOST} for ${DATASET} ${SETTING}"
python paper/analysis/summarize_boost_f1.py \
    --dataset "${DATASET}" \
    --setting "${SETTING}" \
    --max-boost "${BOOST_MAX}" \
    --subdir "boost_only_transcriptions" \
    --baseline-manifest "${TRAIN_MANIFEST}"
BASELINE_F1_TSV="paper/data/predictions/${DATASET}/${SETTING}/boost_only_transcriptions/f1_all.tsv"
BASELINE_WER=$(awk -F'\t' '$1=="WER"{print $2; exit}' "${BASELINE_F1_TSV}")
echo "No-boost train baseline WER: ${BASELINE_WER}"
echo "Best boost value: ${BEST_BOOST} (validation micro F1: ${BEST_BOOST_F1})"

# Step 12+: Sounds-like discovery and re-biasing (identical to the zero-recall
# variant): mining reads the boosted TRAIN transcription for boost-word
# confusions, the LLM names the sounds-alike span, the suppression tokens are
# collected, and the SL weight is swept on VALIDATION. Train mines, validation
# selects, test reports.
PRED_DIR="paper/data/predictions/${DATASET}/${SETTING}"
BOOST_FILE="${PRED_DIR}/boost_only_files/boost1.txt"
MANIFEST_FILE="${TRAIN_MANIFEST}"

case "${SETTING}" in
    aed_*) SL_TOKENS_MODEL="${CANARY_MODEL:-${MODELS_DIR}/canary-1b/canary-1b.nemo}" ;;
    *)     SL_TOKENS_MODEL="${HYBRID_MODEL:-${MODELS_DIR}/stt_en_fastconformer_hybrid_large_pc/stt_en_fastconformer_hybrid_large_pc.nemo}" ;;
esac

# The sounds-like track: discovery, weight selection and the Step 13 test
# evaluation.
#
# Suppression is plain minus-paths: a negative-weight phrase lowers the score of
# the tokens the recogniser emitted, and nothing else intervenes. Identical for
# greedy and beam decoding.
run_sounds_like_track () {
  local TRACK_TAG=""
  echo "=== sounds-like track (plain minus-paths suppression) ==="

  # The mining seed is the boost-only train decode from Step 11b, so
  # both tracks start from the same seed.
  local PREV_TRANSCRIPTION="${PRED_DIR}/boost_only_transcriptions/boost_b${BEST_BOOST}.json"
  local TAG SENTENCES CLASSIFIED COMBINED SL_FILES_DIR
  local SL_TRANS_SUBDIR_VALIDATION SL_TOKENS WORD_TOKENS_FILE
  local SL_WEIGHT SL_VAL_TSV SL_VAL_BASELINE_WER BEST_SL BEST_SL_F1_VAL f
echo "=== Sounds-like discovery for ${DATASET} ${SETTING} ==="
TAG="${TRACK_TAG}sl"
SENTENCES="${PRED_DIR}/${TRACK_TAG}boost_word_confusion_sentences.tsv"
CLASSIFIED="${PRED_DIR}/${TRACK_TAG}boost_word_confusions.classified.tsv"
COMBINED="${PRED_DIR}/${TRACK_TAG}boost_word_confusions.tsv"
SL_FILES_DIR="${PRED_DIR}/${TAG}_boost_with_sounds_like_files"
SL_TRANS_SUBDIR_VALIDATION="${TAG}_boost_with_sounds_like_transcriptions_validation"
# 1) Collect misrecognized boost-word sentences from the previous best transcription.
echo "Collecting misrecognized boost-word sentences from ${PREV_TRANSCRIPTION}"
python paper/analysis/find_boost_word_confusions.py \
    --transcription "${PREV_TRANSCRIPTION}" \
    --boost-file "${BOOST_FILE}" \
    --out "${SENTENCES}"
# 2) LLM names the sounds-alike confusion from the full ref + hyp sentences.
echo "LLM-finding sounds-like pairs from sentences"
python paper/analysis/llm_find_sounds_like.py \
    --input  "${SENTENCES}" \
    --output "${CLASSIFIED}" \
    --azure-suffix gpt5_5
# 2b) Recover the observed token-id sequences for each confusion.
SL_TOKENS="${PRED_DIR}/${TRACK_TAG}boost_word_confusions.word_tokens.raw.tsv"
echo "Extracting observed tokens for sounds-like confusions -> ${SL_TOKENS}"
python paper/analysis/find_sounds_like_tokens.py \
    --classified "${CLASSIFIED}" \
    --model      "${SL_TOKENS_MODEL}" \
    --out        "${SL_TOKENS}"
WORD_TOKENS_FILE="${PRED_DIR}/${TRACK_TAG}boost_word_confusions.word_tokens.tsv"
tail -n +2 "${SL_TOKENS}" | sort -u \
    | { printf 'word	token_ids
'; cat; } > "${WORD_TOKENS_FILE}"
echo "confusion-token suppression map -> ${WORD_TOKENS_FILE}"
# 3) Build the boost+SL files from the classified rows.
echo "Building boost with sounds-like files"
awk -F'\t' 'FNR==1 && NR==1 {print; next} FNR==1 {next} !seen[$2 FS $6]++' \
    "${CLASSIFIED}" > "${COMBINED}"
python paper/analysis/build_boost_with_sounds_like_files.py \
    --input   "${COMBINED}" \
    --boost-file "${BOOST_FILE}" \
    --out-dir "${SL_FILES_DIR}" \
    --best-boost ${BEST_BOOST} \
    --min-sl-weight "${SL_MIN_WEIGHT}" \
    --max-sl-weight "${SL_MAX_WEIGHT}" \
    --manifest "${MANIFEST_FILE}"
# 4) Sweep SL weight on VALIDATION.
echo "Running biased inference with sounds-like on VALIDATION (sweep ${SL_MIN_WEIGHT}..${SL_MAX_WEIGHT})"
for SL_WEIGHT in $(seq "${SL_MIN_WEIGHT}" "${SL_MAX_WEIGHT}"); do
            DATASET="${DATASET}" \
    SETTING="${SETTING}" \
    BOOST_SUFFIX="_b${SL_WEIGHT}" \
    OUT_SUBDIR="${SL_TRANS_SUBDIR_VALIDATION}" \
    KEY_PHRASES="${PRED_DIR}/boost_only_files/boost${BEST_BOOST}.txt" \
    SOUNDS_LIKE="${SL_FILES_DIR}/boost_with_sounds_like_${SL_WEIGHT}.txt" \
    WORD_TOKENS_FILE="${WORD_TOKENS_FILE}" \
    MANIFEST="${VAL_MANIFEST}" \
        bash paper/inference/run_inference_with_boosting_and_sounds_like.sh
done
# 5) Per-word F1 sweep for the VALIDATION sounds-like transcriptions.
echo "Summarising per-word F1 sweep (validation)"
python paper/analysis/summarize_boost_f1.py \
    --dataset "${DATASET}" \
    --setting "${SETTING}" \
    --max-boost "${SL_MAX_WEIGHT}" \
    --subdir "${SL_TRANS_SUBDIR_VALIDATION}" \
    --file-fmt "boost_sl_b{B}.json" \
    --col-prefix "sounds_like" \
    --kw-file "${SL_FILES_DIR}/boost_with_sounds_like_${SL_MIN_WEIGHT}.txt" \
    --baseline-manifest "${VAL_MANIFEST}"
# 6) Pick the best SL weight on VALIDATION under the WER cap.
SL_VAL_TSV="${PRED_DIR}/${SL_TRANS_SUBDIR_VALIDATION}/f1_all.tsv"
SL_VAL_BASELINE_WER=$(awk -F'\t' '$1=="WER"{print $2; exit}' "${SL_VAL_TSV}")
BEST_SL=$(python paper/analysis/find_best_boost.py \
    --dataset "${DATASET}" \
    --setting "${SETTING}" \
    --subdir "${SL_TRANS_SUBDIR_VALIDATION}" \
    --col-prefix "sounds_like" \
    --max-wer-deterioration "${MAX_WER_DETERIORATION}" \
    --baseline-wer "${SL_VAL_BASELINE_WER}")
BEST_SL_F1_VAL=$(python paper/analysis/find_best_boost.py \
    --dataset "${DATASET}" \
    --setting "${SETTING}" \
    --subdir "${SL_TRANS_SUBDIR_VALIDATION}" \
    --col-prefix "sounds_like" \
    --max-wer-deterioration "${MAX_WER_DETERIORATION}" \
    --baseline-wer "${SL_VAL_BASELINE_WER}" \
    --print f1)
echo "Best sounds-like weight (validation): ${BEST_SL} (validation micro F1: ${BEST_SL_F1_VAL})"

# The validation-selected weight is the configuration to report.
local BEST_SL_FILE="${SL_FILES_DIR}/boost_with_sounds_like_${BEST_SL}.txt"

# Step 13: Final test-set evaluation (no biasing / boost-only / boost+sounds-like).
echo "=== Step 13: Test-set evaluation for ${DATASET} ${SETTING} ==="
local TEST_MANIFEST="${PRED_DIR}/baseline.wer_filtered.json"
local TEST_SUBDIR="${TRACK_TAG}test_eval"
local BEST_BOOST_FILE="${PRED_DIR}/boost_only_files/boost${BEST_BOOST}.txt"
local RTFX_LOG_DIR
echo "[test] best boost-only file:  ${BEST_BOOST_FILE} (B=${BEST_BOOST})"
echo "[test] best sounds-like file: ${BEST_SL_FILE} (SL=${BEST_SL}, validation micro F1: ${BEST_SL_F1_VAL})"
echo "[test] no-boost baseline (WER-filtered, from Steps 3c/4c): ${TEST_MANIFEST}"

RTFX_LOG_DIR="${PRED_DIR}/${TEST_SUBDIR}/rtfx_logs"
mkdir -p "${RTFX_LOG_DIR}"

echo "[test] run 1/2: boost-only (B=${BEST_BOOST})"
DATASET="${DATASET}" \
SETTING="${SETTING}" \
MANIFEST="${TEST_MANIFEST}" \
OUT_SUBDIR="${TEST_SUBDIR}" \
BOOST_SUFFIX="_boost_only" \
KEY_PHRASES="${BEST_BOOST_FILE}" \
CALCULATE_RTFX="${MEASURE_RTFX}" \
    bash paper/inference/run_inference_with_boosting.sh 2>&1 \
    | tee "${RTFX_LOG_DIR}/boost_only.log"

echo "[test] run 2/2: boost + sounds-like (B=${BEST_BOOST}, SL=${BEST_SL})"
DATASET="${DATASET}" \
SETTING="${SETTING}" \
MANIFEST="${TEST_MANIFEST}" \
OUT_SUBDIR="${TEST_SUBDIR}" \
BOOST_SUFFIX="_sounds_like" \
KEY_PHRASES="${BEST_BOOST_FILE}" \
SOUNDS_LIKE="${BEST_SL_FILE}" \
CALCULATE_RTFX="${MEASURE_RTFX}" \
    bash paper/inference/run_inference_with_boosting_and_sounds_like.sh 2>&1 \
    | tee "${RTFX_LOG_DIR}/sounds_like.log"

if [ "${MEASURE_RTFX}" = "1" ]; then
    echo "[test] rtfx: timing-only no-biasing pass over ${TEST_MANIFEST}"
    RTFX_SCRATCH="paper/data/predictions_rtfx"
    DATASET="${DATASET}" \
    SETTING="${SETTING}" \
    MANIFEST="${TEST_MANIFEST}" \
    OUT_DIR="${RTFX_SCRATCH}" \
    CALCULATE_RTFX=1 \
        bash paper/inference/run_inference.sh 2>&1 \
        | tee "${RTFX_LOG_DIR}/no_boost.log"
    rm -rf "${RTFX_SCRATCH:?}/${DATASET}/${SETTING}"
fi

echo "[test] reporting avg F1 (boosted words) and total WER per run"
python paper/analysis/report_test_methods.py \
    --no-boost    "${PRED_DIR}/baseline.wer_filtered.json" \
    --boost       "${PRED_DIR}/${TEST_SUBDIR}/boost_boost_only.json" \
    --sounds-like "${PRED_DIR}/${TEST_SUBDIR}/boost_sl_sounds_like.json" \
    --kw-file     "${BOOST_FILE}" \
    --out         "${PRED_DIR}/${TEST_SUBDIR}/test_method_comparison.tsv"

if [ "${MEASURE_RTFX}" = "1" ]; then
    echo "[test] reporting RTFx (decoding throughput) per run"
    python paper/analysis/collect_rtfx.py \
        --manifest "${TEST_MANIFEST}" \
        --log "no_boost=${RTFX_LOG_DIR}/no_boost.log" \
        --log "boost_only=${RTFX_LOG_DIR}/boost_only.log" \
        --log "sounds_like=${RTFX_LOG_DIR}/sounds_like.log" \
        --out "${PRED_DIR}/${TEST_SUBDIR}/test_rtfx.tsv"
fi

# Record the selected weight for the Step 14 summary.
echo "${BEST_SL}"       > "${PRED_DIR}/${TRACK_TAG}best_sl.weight"

echo "=== Step 14: Summary for ${DATASET} ${SETTING} ==="
python paper/analysis/summarize_setting.py \
    --dataset "${DATASET}" \
    --setting "${SETTING}" \
    --best-boost "${BEST_BOOST}" \
    --best-sl-weight "${BEST_SL}" \
    --track-tag "${TRACK_TAG}"
}   # end run_sounds_like_track

run_sounds_like_track
