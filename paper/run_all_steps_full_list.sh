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
# Force pure-Python protobuf parsing. The installed protobuf (5.x) is too new for
# onnx's generated *_pb2.py, which NeMo imports on the inference path; without
# this the import raises "TypeError: Descriptors cannot be created directly" and
# every inference step (3/3b/3c, boost/sounds-like sweeps) crashes. This is the
# documented workaround (no protobuf downgrade, negligible cost since protobuf is
# not in the ASR hot path). See: protobuf 2022-05-06 python-updates note.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# GPU pin: honour an externally-provided CUDA_VISIBLE_DEVICES (e.g. set per-combo
# by run_all.sh so each (dataset, setting) lands on its own GPU). Falls back to
# GPU 0 when run standalone.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Checkpoint location. Defaults to a `models` directory next to the repo, i.e.
# <repo>/../models, matching paper/inference/*.sh. Override MODELS_DIR, or point
# HYBRID_MODEL / CANARY_MODEL straight at a .nemo file.
MODELS_DIR="${MODELS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/models}"

export DATASET="${DATASET:-stop_music}"
# Variant output isolation. Every artifact this script writes lands under
# paper/data/predictions/${DATASET}${RUN_TAG}/ so the phrase-mining strategies can
# be compared side by side without overwriting each other's results.
# SRC_DATASET keeps pointing at the real dataset directory
# (paper/data/<name>_nemo) that holds the source manifests / oracle list, since
# only the OUTPUT location is tagged -- not the input data.
RUN_TAG="${RUN_TAG:-}"
case "${DATASET}" in
    *"${RUN_TAG}") SRC_DATASET="${DATASET%${RUN_TAG}}" ;;   # already tagged; don't double-tag
    *)             SRC_DATASET="${DATASET}"
                   export DATASET="${DATASET}${RUN_TAG}" ;;
esac
echo "[variant] strategy=zero-recall-of-eval-test-list  source data=${SRC_DATASET}  outputs=${DATASET}"
export SETTING="${SETTING:-rnnt_beam}"

# No ZR_DOMAIN here, unlike the mining variant: candidates come from the released
# term list, so there is no LLM term classifier to tell what counts as a
# specialised term for this corpus.
export SELECT_BY="${SELECT_BY:-f1}"
echo "[variant] strategy=zero-recall-of-eval-test-list  selection=${SELECT_BY}"
# Max WER deterioration (in percentage points over the NO-BOOST baseline) that a
# config may incur and still be eligible. Both stages are measured against that
# same baseline, so the cap is a bound on the TOTAL cost of biasing, not a
# per-stage allowance: whatever boosting spends is unavailable to sounds-like.
#
# Two other schemes were tried and rejected. Budgeting sounds-like against
# boost-only lets each stage spend the full cap, so the total reached ~1.3 pts.
# Splitting the cap evenly (half each) bounds the total correctly but rejects a
# stage for exceeding a half the other stage never needed -- on stop_places ctc_greedy
# it threw out B=6 (F1 0.1667 -> 0.2222) for costing +0.56 of a +0.5 half, while
# the total would have been fine. Measuring both against no-boost has neither
# problem: the constraint is exactly the quantity being promised.
#
# Set to a negative value to disable the filter and pick purely on the metric.
export MAX_WER_DETERIORATION="${MAX_WER_DETERIORATION:-1.0}"
# Sweep ranges. The boost sweep runs BOOST_MIN..BOOST_MAX and the sounds-like
# suppression sweep runs SL_MIN_WEIGHT..SL_MAX_WEIGHT. Both go to 10 rather than
# a narrower window: with the selector capped on WER it is the budget, not the
# range, that stops the sweep, and at a ceiling of 8 the chosen value sat ON that
# ceiling in several cells (SL=8 on stop_places aed_greedy / aed_beam and stop_music
# rnnt_beam) -- which means the optimum was outside the range and the bound, not
# the data, was doing the choosing.
export BOOST_MIN="${BOOST_MIN:-1}"
export BOOST_MAX="${BOOST_MAX:-10}"
export SL_MIN_WEIGHT="${SL_MIN_WEIGHT:-1}"
export SL_MAX_WEIGHT="${SL_MAX_WEIGHT:-10}"
# Boosting-tree depth scaling (gamma). Hybrid = CTC/RNN-T; AED = Canary.
# Inference wrappers pick these up as boosting_tree.depth_scaling.
export BT_DEPTH_SCALING_HYBRID="${BT_DEPTH_SCALING_HYBRID:-1.0}"
export BT_DEPTH_SCALING_AED="${BT_DEPTH_SCALING_AED:-1.0}"

# Measure decoding throughput (RTFx = seconds of test audio decoded per second
# of compute) for the three test methods in Step 13. Set MEASURE_RTFX=0 to skip.
# Cost: RTFX_WARMUP_STEPS untimed warm-up decodes per timed pass (the first pass
# through a decoder pays lazy CUDA init and graph capture, which would otherwise
# be charged to it), plus one extra no-biasing decode of the test set -- the
# Step 3c baseline ran over the *unfiltered* test.json, so its decode time covers
# different audio than the boost / sounds-like passes.
# Caveat: run_all.sh gives each combo its own GPU but they share host CPU and
# dataloader workers, so timings taken during a parallel run are noisier than a
# dedicated measurement on an idle machine.
export MEASURE_RTFX="${MEASURE_RTFX:-1}"
export RTFX_WARMUP_STEPS="${RTFX_WARMUP_STEPS:-1}"

# Azure OpenAI secrets for the LLM steps (term classification in Step 6,
# sounds-like discovery). Kept OUT of version control: sourced from a gitignored env
# file (repo-root/.env by default; override with ENV_FILE=/path/to/file). The
# file holds plain KEY=value lines (no `export`), e.g.:
#   AZURE_OPENAI_API_KEY_gpt5_5=...
#   AZURE_OPENAI_ENDPOINT_gpt5_5=https://<resource>.cognitiveservices.azure.com/
#   deploymentid_gpt5_5=gpt-5.5
#   OPENAI_API_VERSION_gpt5_5=2024-12-01-preview
# See .env.example for the template. `set -a` exports every var the file defines.
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

# Step 1: Dataset preparation is expected to have happened already -- the
# STOP-Music / STOP-Places manifests are built out of band by
# paper/data/prepare_stop_slots.py, which
# extracts the biasing entities from STOP's seqlogical slot annotations and
# emits train/eval/test.json following STOP's own native split (train = mining,
# eval = selection, test = reporting). No custom re-splitting is applied.
# Nothing below can work without the source manifests, and no later step aborts
# the run on its own (each just fails and the next one starts), so a missing
# dataset used to produce a full log of FileNotFoundError tracebacks and an "OK"
# verdict from run_all.sh. Check once, here.
for _split in train eval test; do
    if [ ! -f "paper/data/${SRC_DATASET}_nemo/${_split}.json" ]; then
        echo "[fatal] missing paper/data/${SRC_DATASET}_nemo/${_split}.json --" \
             "prepare the dataset before running the pipeline." >&2
        exit 1
    fi
done

# Step 2b: Stage a per-setting copy of the source manifests.
#
# No speaker-label stripping here, unlike the MultiMed pipeline: STOP prompts are
# single-speaker voice-assistant queries with no "Name:" turn labels for that
# pass to remove.
#
# The copy is still needed, and not for cleanliness: run_inference.sh splits a
# manifest into <stem>_short.json / <stem>_long.json NEXT TO the manifest itself,
# so pointing every setting at the shared paper/data/<corpus>_nemo/ directory
# would have the six settings run_all.sh runs concurrently overwrite each other's
# split files. A per-setting directory keeps them isolated.
SRC_MANIFEST_DIR="paper/data/${SRC_DATASET}_nemo"
CLEAN_DIR="paper/data/${SRC_DATASET}_nemo/clean/${DATASET}/${SETTING}"
echo "Step 2b: Staging source manifests for ${DATASET} ${SETTING}"
mkdir -p "${CLEAN_DIR}"
cp "${SRC_MANIFEST_DIR}/train.json" \
   "${SRC_MANIFEST_DIR}/eval.json" \
   "${SRC_MANIFEST_DIR}/test.json" \
   "${CLEAN_DIR}/"

# Step 3: Run inference (train baseline) on the staged train manifest.
echo "Step 3: Running inference for ${DATASET} ${SETTING}"
DATASET="${DATASET}" SETTING="${SETTING}" \
MANIFEST="${CLEAN_DIR}/train.json" \
    bash paper/inference/run_inference.sh

# Step 3b: Validation-set inference.
# Discovery/mining (Steps 5-8) stays on train, but model selection (best boost /
# best sounds-like) is done on a held-out VALIDATION set so the chosen configs
# are not tuned on the same utterances we mined them from. Run the no-boost
# baseline on eval.json here (right after the train baseline); it is WER-filtered
# alongside the train filtering in Step 4b, and the boost/SL sweeps later
# transcribe the filtered eval set so selection reads its per-word F1 + WER.
echo "Step 3b: Running validation (eval) inference for ${DATASET} ${SETTING}"
DATASET="${DATASET}" SETTING="${SETTING}" \
MANIFEST="${CLEAN_DIR}/eval.json" \
    bash paper/inference/run_inference.sh

# Step 3c: Test-set no-biasing inference.
# Run the no-boost baseline on test.json here (alongside train + validation) so
# it is WER-filtered in Step 4c. The final test evaluation reuses this filtered
# baseline as the no-boost result AND as the manifest for its boost / sounds-like
# passes, so all three test methods are scored on the same clean set.
# test.json -> baseline.json (test stem yields no suffix).
echo "Step 3c: Running test inference (no biasing) for ${DATASET} ${SETTING}"
DATASET="${DATASET}" SETTING="${SETTING}" \
MANIFEST="${CLEAN_DIR}/test.json" \
    bash paper/inference/run_inference.sh

# No Step 4 reference-quality filter on this variant.
#
# The filter exists to drop utterances whose baseline hypothesis is far from the
# reference, on the assumption that the REFERENCE is bad. STOP's references are
# annotator-checked TOPv2 prompts, so there is nothing for it to catch. Worse, it
# would be actively harmful here: STOP utterances are voice-assistant queries of a
# median 6-8 words, so one wrong word is already 12-17% WER, and the entities we
# are here to recover are frequently multi-word ("martin luther king boulevard").
# Filtering on WER would therefore discard exactly the utterances the zero-recall
# step needs, in proportion to how badly the model fails on them.
#
# Downstream stages consume the raw baseline predictions directly.
TRAIN_MANIFEST="paper/data/predictions/${DATASET}/${SETTING}/baseline_train.json"
VAL_MANIFEST="paper/data/predictions/${DATASET}/${SETTING}/baseline_eval.json"

# Step 5-7: Build the boost list = the eval+test entities that the model FAILS
# on in train (zero recall).
#
# Two stages:
#   1. Build the candidate entity list directly from the VALIDATION + TEST
#      manifests' slot 'hotwords' (the entities we will be scored on).
#   2. Of those, keep only the ones that occur in the TRAIN references and are
#      never transcribed correctly there (recall == 0 on train).
#
# So the candidate set is exactly the val+test entities, and we boost only its
# hard, model-failed subset -- never train-only terms that would add WER without
# ever being scored.
#
# Leakage note: test is used only to define the LIST scope (stage 1, entity
# annotations only). The zero-recall filter (stage 2), the boost-value sweep
# (Step 11) and the sounds-like mining all use train/eval only -- no test
# transcription output enters filtering or selection.
case "${SRC_DATASET}" in
    stop_music*|stop_places*) ;;
    *) echo "[error] ${0##*/} needs a released context list (stop_music/stop_places)" >&2; exit 1 ;;
esac
case "${SETTING}" in
    aed_*) TOKENIZER_MODEL="${CANARY_MODEL:-${MODELS_DIR}/canary-1b/canary-1b.nemo}" ;;
    *)     TOKENIZER_MODEL="${HYBRID_MODEL:-${MODELS_DIR}/stt_en_fastconformer_hybrid_large_pc/stt_en_fastconformer_hybrid_large_pc.nemo}" ;;
esac
PHRASE_DIR="paper/data/predictions/${DATASET}/${SETTING}"
# Stage 1: build the candidate entity list directly from the VALIDATION + TEST
# source manifests, using their per-record 'hotwords' (the entities extracted
# from STOP's slot annotations by prepare_stop_slots.py). This is exactly "the
# entities present in val+test", so no separate oracle-list + presence
# intersection is needed. We require each entity to occur at least once in BOTH
# the val and test splits (--require-each): a term must be present in val (so
# selection can see it) and in test (so it is actually scored). Terms confined
# to a single split are dropped. The zero-recall + min-count filter (stage 2)
# runs on TRAIN.
SRC_EVAL_MANIFEST="paper/data/${SRC_DATASET}_nemo/eval.json"
SRC_TEST_MANIFEST="paper/data/${SRC_DATASET}_nemo/test.json"
echo "Step 5 (stage 1): building entity list present in BOTH val and test from slot hotwords for ${DATASET} ${SETTING}"
python paper/analysis/build_attested_list_from_manifests.py \
    --manifest "${SRC_EVAL_MANIFEST}" "${SRC_TEST_MANIFEST}" \
    --require-each \
    --out      "${PHRASE_DIR}/eval_test_entities.txt"
# Stage 2: of those, keep the ones with zero recall on TRAIN. min-count 1: a
# term is kept if it occurs at least once in train and the baseline never gets
# it right there. The entity is anchored by the given eval+test list either way;
# this only governs how much train evidence is needed to call it hard, and at 1
# a single zero-recall train occurrence is enough (maximises the kept-term set).
ZR_MIN_COUNT="${ZR_MIN_COUNT:-1}"
echo "Step 6-7 (stage 2): zero-recall-on-train filter (min-count ${ZR_MIN_COUNT}) for ${DATASET} ${SETTING}"
python paper/analysis/find_zero_recall_terms_from_list.py \
    --list        "${PHRASE_DIR}/eval_test_entities.txt" \
    --predictions "${TRAIN_MANIFEST}" \
    --output      "${PHRASE_DIR}/boost_phrases.correct_forms.tsv" \
    --min-count   "${ZR_MIN_COUNT}" \
    --max-recall  "${ZR_MAX_RECALL:-0.0}"

# Step 7a: LLM misspelling / mis-segmentation filter on the zero-recall list.
# STOP slot annotations carry orthographic noise -- misspelled names (dehli,
# san fransisco, pittsburg), mis-segmentations (near by, disney land), and
# spelled-out road numbers (a hundred one, inine five). These read as
# zero-recall on train because the model spells the entity correctly, so they
# survive stage 2, get boosted, and then over-trigger on test by flipping the
# model's correct output into the malformed form. Drop them with an LLM judging
# each candidate in isolation. The entity type is per corpus but the rule is
# identical across corpora, so this is a blind normalization pass, not a
# hand-removal of the terms that happened to hurt test. Leakage-safe: it reads
# only the entity string, never any test output. Set LLM_FILTER_ENTITIES=0 to
# skip (e.g. to measure its contribution).
if [ "${LLM_FILTER_ENTITIES:-1}" = "1" ]; then
    case "${SRC_DATASET}" in
        stop_music*)  _ETYPE="music artist names and event names" ;;
        stop_places*) _ETYPE="US and world place / location names" ;;
        *)      _ETYPE="proper entity names" ;;
    esac
    echo "Step 7a: LLM misspelling filter (type: ${ENTITY_TYPE:-${_ETYPE}}) for ${DATASET} ${SETTING}"
    cp "${PHRASE_DIR}/boost_phrases.correct_forms.tsv" \
       "${PHRASE_DIR}/boost_phrases.correct_forms.prefilter.tsv"
    python paper/analysis/llm_filter_entities.py \
        --input       "${PHRASE_DIR}/boost_phrases.correct_forms.prefilter.tsv" \
        --output      "${PHRASE_DIR}/boost_phrases.correct_forms.tsv" \
        --entity-type "${ENTITY_TYPE:-${_ETYPE}}" \
        --azure-suffix gpt5_5
fi

# Step 7c: Truecase the entity forms. STOP references are lowercased, so the
# recovered surface forms are lowercase ("st louis", "rowlett") -- but the PC
# models emit cased, punctuated text ("St. Louis", "Rowlett"). Boosting the
# lowercase token path forces the decoder off its natural cased path, which
# needs a large boost weight to win and then over-triggers (precision collapse,
# worst on AED greedy). An LLM rewrites each term into the form the model would
# actually produce, changing ONLY casing/punctuation (a normalization guard
# rejects any reply that alters the words), so we boost what the model emits.
# Leakage-safe: reads only the entity string, with few-shot examples from
# outside our corpora. Set TRUECASE_ENTITIES=0 to skip (e.g. for an ablation).
if [ "${TRUECASE_ENTITIES:-1}" = "1" ]; then
    echo "Step 7c: LLM truecasing of entity forms for ${DATASET} ${SETTING}"
    cp "${PHRASE_DIR}/boost_phrases.correct_forms.tsv" \
       "${PHRASE_DIR}/boost_phrases.correct_forms.pretruecase.tsv"
    python paper/analysis/truecase_boost_terms.py \
        --input       "${PHRASE_DIR}/boost_phrases.correct_forms.pretruecase.tsv" \
        --output      "${PHRASE_DIR}/boost_phrases.correct_forms.tsv" \
        --azure-suffix gpt5_5
fi

# Step 7b: Drop phrases the model tokenizer cannot represent (they tokenize to
# <unk>, which the model never emits, so they are dead weight in the graph).
echo "Step 7b: Dropping untokenizable phrases for ${DATASET} ${SETTING}"
python paper/analysis/filter_untokenizable_phrases.py \
    --input  "${PHRASE_DIR}/boost_phrases.correct_forms.tsv" \
    --output "${PHRASE_DIR}/boost_phrases.correct_forms.tokenizable.tsv" \
    --model  "${TOKENIZER_MODEL}"

# Step 8: Build boosting lists from the (tokenizable) correct word forms
echo "Step 8: Building boosting lists from correct word forms for ${DATASET} ${SETTING}"
python paper/analysis/build_boost_lists.py \
    --input   paper/data/predictions/${DATASET}/${SETTING}/boost_phrases.correct_forms.tokenizable.tsv \
    --out-dir paper/data/predictions/${DATASET}/${SETTING} \
    --max-boost "${BOOST_MAX}"

# Step 9: Boost-value sweep on VALIDATION.
#
# The sweep used to run on train, and the best value was selected there, but a
# value chosen on the split its phrases were mined from does not transfer: on
# stop_music rnnt_greedy the train sweep took B=7 because it fit the +1.0 pt WER
# budget on train, and the same config cost +1.42 pts on test. Train F1 is also
# monotone in B on that corpus (0.13 -> 0.70 across B=1..8, argmax at the sweep
# edge), because boosting harder keeps paying off on the very utterances the
# phrases came from -- whereas on a corpus with more data the curve turns over
# well before the edge (MultiMed peaks at B=4 and collapses to 0.23 by B=8). A
# train-selected B therefore reliably overshoots.
#
# Sweeping on validation is also what the reference implementation does
# (arXiv:2508.07014 Sec. III: "the dev sets were used for boosting parameters'
# search, and the test for obtaining the final results"). It leaves train with
# the job it is actually needed for here and that TurboBias has no equivalent
# of: MINING the bias list. Train discovers what to boost, validation decides
# how hard, test reports.
#
# The boosting tree's unk_score (the per-token reward for staying outside the
# graph) is left at the library default of 0. Tying it to B, as TurboBias
# suggests, was measured here and is much worse: because the reward is PER
# TOKEN, beam search farms it by emitting longer hypotheses (hyp/ref word ratio
# 0.98 -> 1.01 on stop_music ctc_beam), roughly doubling validation WER (0.207 ->
# 0.410 at B=8) while F1 collapses 0.36 -> 0.03. Greedy is immune to the length
# effect -- a constant per-token offset cannot change a per-step argmax -- so
# its WER improves slightly, but only because the boost stops firing at all
# (F1 0.30 -> 0.17).
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

# Step 11: Select the boost value on validation, under the WER budget measured
# against the validation baseline.
echo "Step 11: Finding best boost value (validation) for ${DATASET} ${SETTING}"
BEST_BOOST=$(python paper/analysis/find_best_boost.py \
    --dataset "${DATASET}" \
    --setting "${SETTING}" \
    --subdir "${BOOST_VAL_SUBDIR}" \
    --max-wer-deterioration "${MAX_WER_DETERIORATION}" \
    --select "${SELECT_BY}" \
    --baseline-wer "${VAL_BOOST_BASELINE_WER}")
BEST_BOOST_F1=$(python paper/analysis/find_best_boost.py \
    --dataset "${DATASET}" \
    --setting "${SETTING}" \
    --subdir "${BOOST_VAL_SUBDIR}" \
    --max-wer-deterioration "${MAX_WER_DETERIORATION}" \
    --select "${SELECT_BY}" \
    --baseline-wer "${VAL_BOOST_BASELINE_WER}" \
    --print f1)

# What boosting alone cost on validation, logged for accounting: the
# sounds-like cap below is measured against the NO-BOOST baseline, so this is
# the part of the total budget already spent by the time suppression is chosen.
# Column layout of the WER row: $2 is no_boost, $3 is boost_1, so B is at 2+B.
BOOST_ONLY_VAL_WER=$(awk -F'\t' -v c="$((2 + BEST_BOOST))" '$1=="WER"{print $c; exit}' "${BOOST_VAL_TSV}")
echo "Boost-only (B=${BEST_BOOST}) validation baseline WER: ${BOOST_ONLY_VAL_WER}"

# Step 11b: Decode TRAIN once at the selected value. Sounds-like discovery mines
# its confusions from the boosted TRAIN transcription, so that one file
# is still needed -- but only at B*, which is why the train side is a single
# pass now instead of the full sweep it used to be. Since train is ~3x the audio
# of eval here, moving the sweep to validation made the step cheaper, not dearer.
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

# Step 12+: Sounds-like discovery and re-biasing.
#
# Mine the boost-only TRAIN transcription (written by Step 11b at the
# validation-selected BEST_BOOST) for boost-word confusions -- a boost word
# present in the reference but replaced by something else -- ask the LLM to name
# the sounds-alike span from the full sentences, build the sounds-like files, then
# sweep the suppression weight ON VALIDATION and take the winner. So train mines,
# validation selects, test reports.
PRED_DIR="paper/data/predictions/${DATASET}/${SETTING}"
BOOST_FILE="${PRED_DIR}/boost_only_files/boost1.txt"
MANIFEST_FILE="${TRAIN_MANIFEST}"

# Model whose tokenizer segments pred_token_ids when recovering the observed
# token sequences for sounds-like confusions (CTC/RNN-T -> hybrid; AED -> Canary).
case "${SETTING}" in
    aed_*) SL_TOKENS_MODEL="${CANARY_MODEL:-${MODELS_DIR}/canary-1b/canary-1b.nemo}" ;;
    *)     SL_TOKENS_MODEL="${HYBRID_MODEL:-${MODELS_DIR}/stt_en_fastconformer_hybrid_large_pc/stt_en_fastconformer_hybrid_large_pc.nemo}" ;;
esac

# The sounds-like track: discovery, weight selection, the Step 13 test evaluation
# and the Step 14 summary.
#
# Suppression is plain minus-paths: a negative-weight phrase lowers the score of
# the tokens the recogniser emitted, and nothing else intervenes. Identical for
# greedy and beam decoding.
run_sounds_like_track () {
  local TRACK_TAG=""
  echo "=== sounds-like track (plain minus-paths suppression) ==="

# Mining reads the boost-only train transcription Step 11b already wrote at the
# validation-selected BEST_BOOST, so no extra inference is needed here.
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
# The weight sweep lands in the *_validation subdir, which the summary reads.
SL_TRANS_SUBDIR_VALIDATION="${TAG}_boost_with_sounds_like_transcriptions_validation"
# 1) Collect misrecognized boost-word sentences from the previous best transcription
echo "Collecting misrecognized boost-word sentences from ${PREV_TRANSCRIPTION}"
python paper/analysis/find_boost_word_confusions.py \
    --transcription "${PREV_TRANSCRIPTION}" \
    --boost-file "${BOOST_FILE}" \
    --out "${SENTENCES}"
# 2) Show the LLM the full ref + hyp sentences and let it find the
#    sounds-alike confusion produced in place of each boost word.
echo "LLM-finding sounds-like pairs from sentences"
python paper/analysis/llm_find_sounds_like.py \
    --input  "${SENTENCES}" \
    --output "${CLASSIFIED}" \
    --azure-suffix gpt5_5
# 2b) Recover the observed token-id sequences the model emitted for each
#     sounds-like confusion (from pred_token_ids), saved word_tokens.tsv-style
#     so the boosting graph can suppress exactly those tokens.
SL_TOKENS="${PRED_DIR}/${TRACK_TAG}boost_word_confusions.word_tokens.raw.tsv"
echo "Extracting observed tokens for sounds-like confusions -> ${SL_TOKENS}"
python paper/analysis/find_sounds_like_tokens.py \
    --classified "${CLASSIFIED}" \
    --model      "${SL_TOKENS_MODEL}" \
    --out        "${SL_TOKENS}"
# Dedupe identical word+token_ids rows into the map the boosting graph reads;
# any confusion missing from it falls back to canonical text_to_ids.
WORD_TOKENS_FILE="${PRED_DIR}/${TRACK_TAG}boost_word_confusions.word_tokens.tsv"
tail -n +2 "${SL_TOKENS}" | sort -u \
    | { printf 'word	token_ids
'; cat; } > "${WORD_TOKENS_FILE}"
echo "confusion-token suppression map -> ${WORD_TOKENS_FILE}"
# 3) Build the boost-with-sounds-like files from the classified rows
#    (header once, de-duped on the
#    (word, sounds_like) pair -- columns $2 and $6 of the classified TSV).
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
# 4) Sweep SL_WEIGHT over SL_MIN_WEIGHT..SL_MAX_WEIGHT on VALIDATION, for the
#    same reason the boost sweep moved there: the sounds-like pairs are mined
#    from TRAIN errors, so a weight fit on train is scored on the very
#    utterances that produced the confusions, and the suppression strength
#    that wins there overshoots on held-out audio. Validation picks the weight.
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
#    The selection below reads this same table, so the choice comes off
#    held-out data.
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
# 6) Pick the best sounds-like weight ON VALIDATION, with the WER cap
#    measured against the validation NO-BOOST baseline, so the cap
#    bounds the total cost of boosting plus suppression.
SL_VAL_TSV="${PRED_DIR}/${SL_TRANS_SUBDIR_VALIDATION}/f1_all.tsv"
SL_VAL_BASELINE_WER=$(awk -F'\t' '$1=="WER"{print $2; exit}' "${SL_VAL_TSV}")
BEST_SL=$(python paper/analysis/find_best_boost.py \
    --dataset "${DATASET}" \
    --setting "${SETTING}" \
    --subdir "${SL_TRANS_SUBDIR_VALIDATION}" \
    --col-prefix "sounds_like" \
    --max-wer-deterioration "${MAX_WER_DETERIORATION}" \
--select "${SELECT_BY}" \
    --baseline-wer "${SL_VAL_BASELINE_WER}")
BEST_SL_F1_VAL=$(python paper/analysis/find_best_boost.py \
    --dataset "${DATASET}" \
    --setting "${SETTING}" \
    --subdir "${SL_TRANS_SUBDIR_VALIDATION}" \
    --col-prefix "sounds_like" \
    --max-wer-deterioration "${MAX_WER_DETERIORATION}" \
--select "${SELECT_BY}" \
    --baseline-wer "${SL_VAL_BASELINE_WER}" \
    --print f1)
echo "Best sounds-like weight (validation): ${BEST_SL} (validation micro F1: ${BEST_SL_F1_VAL})"

# The validation-selected weight is the configuration to report.
local BEST_SL_FILE="${SL_FILES_DIR}/boost_with_sounds_like_${BEST_SL}.txt"

# Step 13: Final test-set evaluation.
# Take the best boost-only file (B*) and the best sounds-like file (the weight
# that won on validation), then
# transcribe the TEST split three ways -- no biasing, boost-only, and
# boost+sounds-like -- and report the average F1 over the boosted words and the
# total WER for each.
#
# The no-biasing pass already happened in Step 3c, so here we reuse its
# predictions (baseline.json) both as the no-boost result and as the manifest for
# the boost / sounds-like passes -- all three test methods are thus scored on the
# same utterance set.
echo "=== Step 13: Test-set evaluation for ${DATASET} ${SETTING} ==="
local TEST_MANIFEST="${PRED_DIR}/baseline.json"
local TEST_SUBDIR="${TRACK_TAG}test_eval"
local BEST_BOOST_FILE="${PRED_DIR}/boost_only_files/boost${BEST_BOOST}.txt"
local RTFX_LOG_DIR
echo "[test] best boost-only file:  ${BEST_BOOST_FILE} (B=${BEST_BOOST})"
echo "[test] best sounds-like file: ${BEST_SL_FILE} (SL=${BEST_SL}, validation micro F1: ${BEST_SL_F1_VAL})"
echo "[test] no-boost baseline (WER-filtered, from Steps 3c/4c): ${TEST_MANIFEST}"

# Decoding-throughput measurement (see MEASURE_RTFX at the top of this script):
# the two runs below are timed in-process by speech_to_text_eval.py, and their
# output is tee'd here so collect_rtfx.py can read the per-pass
# "Model time avg" lines back out.
RTFX_LOG_DIR="${PRED_DIR}/${TEST_SUBDIR}/rtfx_logs"
mkdir -p "${RTFX_LOG_DIR}"

# 1) Best boost-only -> ${PRED_DIR}/${TEST_SUBDIR}/boost_boost_only.json
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

# 2) Best sounds-like -> ${PRED_DIR}/${TEST_SUBDIR}/boost_sl_sounds_like.json
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

# 2b) No-biasing pass over the SAME manifest, for timing only. The no-boost
# transcription already exists (Steps 3c/4c) but was decoded over the unfiltered
# test.json, so its decode time covers different audio than the two runs above
# and cannot be compared with them. Predictions land in a scratch tree nothing
# else reads and are deleted again; only the log matters here.
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

# 3) Report avg F1 over boosted words + total WER for each of the three runs.
echo "[test] reporting avg F1 (boosted words) and total WER per run"
python paper/analysis/report_test_methods.py \
    --no-boost    "${PRED_DIR}/baseline.json" \
    --boost       "${PRED_DIR}/${TEST_SUBDIR}/boost_boost_only.json" \
    --sounds-like "${PRED_DIR}/${TEST_SUBDIR}/boost_sl_sounds_like.json" \
    --kw-file     "${BOOST_FILE}" \
    --out         "${PRED_DIR}/${TEST_SUBDIR}/test_method_comparison.tsv"

# 4) Decoding throughput (RTFx) per test method, from the decode times the runs
# above logged. Note this is the pipeline's own decoding configuration (two
# passes: short clips batched, long clips at batch=1 with local attention), not
# a leaderboard-style single-batch benchmark.
if [ "${MEASURE_RTFX}" = "1" ]; then
    echo "[test] reporting RTFx (decoding throughput) per run"
    python paper/analysis/collect_rtfx.py \
        --manifest "${TEST_MANIFEST}" \
        --log "no_boost=${RTFX_LOG_DIR}/no_boost.log" \
        --log "boost_only=${RTFX_LOG_DIR}/boost_only.log" \
        --log "sounds_like=${RTFX_LOG_DIR}/sounds_like.log" \
        --out "${PRED_DIR}/${TEST_SUBDIR}/test_rtfx.tsv"
fi

# Step 14: Consolidated per-(dataset, setting) summary of avg F1 + WER for the
# validation and test sides -- no-boost, best boost-only, and best
# boost+sounds-like -- read from the tables written above. Written to
# ${PRED_DIR}/summary.tsv (+ summary.txt) and echoed to the logs.
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
