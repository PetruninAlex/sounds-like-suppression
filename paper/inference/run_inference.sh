#!/usr/bin/env bash
# ASR inference runner (no biasing / baseline).
#
# Runs plain (non-biased) inference for a single dataset × decoder setting,
# using NeMo's official entrypoint `examples/asr/speech_to_text_eval.py`.
#
# Decoder strategies (per docs + paper III-A):
#   CTC   greedy: strategy="greedy_batch"
#   CTC   beam=8: strategy="beam_batch"  beam_size=8
#   RNN-T greedy: strategy="greedy_batch"
#   RNN-T beam=8: strategy="malsd_batch" beam_size=8
#   AED   greedy: strategy="beam"        beam_size=1
#   AED   beam=3: strategy="beam"        beam_size=3
#
# Phrase boosting (GPU-PB) is intentionally disabled here -- this is
# the no-biasing baseline pass.
#
# Long-audio handling (CTC + RNN-T on the FastConformer hybrid model):
#   To fit clips longer than LONG_DUR_S into a 48 GB GPU at full attention,
#   the manifest is split into short / long sub-manifests. Short clips are
#   decoded at BATCH_SIZE with full attention; long clips are decoded with
#   batch=1 + local attention. Predictions are concatenated back into a
#   single output file.
#
#   AED rows (Canary) are run single-pass (no split) -- Canary's encoder
#   uses a different attention path and the local-attn override is not
#   guaranteed to apply.
#
# Required env vars:
#   DATASET  — dataset name (e.g. multimed, earnings21)
#   SETTING  — one of: ctc_greedy ctc_beam rnnt_greedy rnnt_beam aed_greedy aed_beam parakeet_tdt
#
# Output: predictions/${DATASET}/${SETTING}/baseline${MANIFEST_SUFFIX}.json

set -euo pipefail

# ----------------------------------------------------------- paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="${PAPER_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
NEMO_DIR="${NEMO_DIR:-$(cd "${PAPER_DIR}/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${PAPER_DIR}/data}"
MODELS_DIR="${MODELS_DIR:-${NEMO_DIR}/../models}"

# Force Python to import NeMo from the workspace (NEMO_DIR) instead of any
# globally-installed copy under site-packages. Keeps inference consistent with
# the workspace code that other paper scripts depend on.
export PYTHONPATH="${NEMO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Force protobuf's pure-Python backend so importing NeMo ASR doesn't die with
# "Descriptors cannot be created directly" (installed protobuf 5.x vs onnx's
# generated pb2 descriptors). Negligible perf impact for inference.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

HYBRID_MODEL="${HYBRID_MODEL:-${MODELS_DIR}/stt_en_fastconformer_hybrid_large_pc/stt_en_fastconformer_hybrid_large_pc.nemo}"
CANARY_MODEL="${CANARY_MODEL:-${MODELS_DIR}/canary-1b/canary-1b.nemo}"

DATASET="${DATASET:?DATASET must be set (e.g. multimed)}"
SETTING="${SETTING:?SETTING must be set (e.g. rnnt_beam)}"

MANIFEST="${MANIFEST:-${DATA_DIR}/${DATASET}_nemo/train.json}"

# Derive an output-filename suffix from each manifest's stem so non-test splits
# (train, eval, ...) don't overwrite each other in the same setting folder.
#   test.json   -> ""        e.g. .../rnnt_beam/baseline.json
#   train.json  -> "_train"  e.g. .../rnnt_beam/baseline_train.json
manifest_suffix () {
    local stem
    stem="$(basename "$1")"; stem="${stem%.*}"
    [ "${stem}" = "test" ] && echo "" || echo "_${stem}"
}
MANIFEST_SUFFIX="$(manifest_suffix "${MANIFEST}")"

OUT_DIR="${OUT_DIR:-${DATA_DIR}/predictions}"
DATASET_OUT="${OUT_DIR}/${DATASET}"
mkdir -p "${DATASET_OUT}/${SETTING}"

EVAL_PY="${NEMO_DIR}/examples/asr/speech_to_text_eval.py"

# ----------------------------------------------------------- knobs
BATCH_SIZE="${BATCH_SIZE:-32}"
LONG_DUR_S="${LONG_DUR_S:-30}"
LONG_ATT_CTX="${LONG_ATT_CTX:-[128,128]}"

HYBRID_BEAM_SIZE="${HYBRID_BEAM_SIZE:-8}"
AED_BEAM_SIZE="${AED_BEAM_SIZE:-3}"
# Canary uses global attention, so the attention matrix scales with batch x T^2.
# Short clips (<=~30s) are fine at this default; for manifests with multi-minute
# clips, lower it (e.g. AED_BATCH_SIZE=1) to avoid OOM on the long attention matrix.
AED_BATCH_SIZE="${AED_BATCH_SIZE:-16}"

# Optional RTFx (decoding throughput) measurement. speech_to_text_eval.py
# inherits calculate_rtfx / warmup_steps / run_steps from transcribe_speech.py,
# which times only the asr_model.transcribe() call on a CUDA-synchronised timer,
# so model load, the short/long manifest split and writing the predictions are
# all excluded. Every pass logs "Model time avg: <seconds>" (and, with
# calculate_rtfx, its own RTFx); collect_rtfx.py sums those across the two passes.
# Off by default because each warm-up step re-decodes the whole manifest.
RTFX_ARGS=()
case "${CALCULATE_RTFX:-0}" in
    1|true|True) RTFX_ARGS=(calculate_rtfx=True warmup_steps="${RTFX_WARMUP_STEPS:-1}") ;;
esac

# ----------------------------------------------------------- helpers
split_manifest_by_duration () {
    python - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
src, max_s = sys.argv[1], float(sys.argv[2])
p = Path(src)
short = str(p.with_name(f"{p.stem}_short{p.suffix}"))
long_ = str(p.with_name(f"{p.stem}_long{p.suffix}"))
ns = nl = 0
with open(src) as f, open(short, "w") as fs, open(long_, "w") as fl:
    for ln in f:
        d = json.loads(ln)
        if d.get("duration", 0.0) <= max_s:
            fs.write(ln); ns += 1
        else:
            fl.write(ln); nl += 1
print(f"[split] {src} -> short={ns} ({short})  long={nl} ({long_})")
PY
}

# ===========================================================================
# Dispatch on SETTING
# ===========================================================================
OUT_BASE="${DATASET_OUT}/${SETTING}/baseline${MANIFEST_SUFFIX}"

case "${SETTING}" in
    ctc_greedy|ctc_beam|rnnt_greedy|rnnt_beam)
        # --- Determine decoder_type and Hydra overrides ---
        case "${SETTING}" in
            ctc_greedy)
                DEC_TYPE=ctc
                DECODING_ARGS=(ctc_decoding.strategy="greedy_batch") ;;
            ctc_beam)
                DEC_TYPE=ctc
                DECODING_ARGS=(
                    ctc_decoding.strategy="beam_batch"
                    ctc_decoding.beam.beam_size="${HYBRID_BEAM_SIZE}"
                ) ;;
            rnnt_greedy)
                DEC_TYPE=rnnt
                DECODING_ARGS=(rnnt_decoding.strategy="greedy_batch") ;;
            rnnt_beam)
                DEC_TYPE=rnnt
                DECODING_ARGS=(
                    rnnt_decoding.strategy="malsd_batch"
                    rnnt_decoding.beam.beam_size="${HYBRID_BEAM_SIZE}"
                ) ;;
        esac

        # --- Two-pass decode (short + long) ---
        echo "[baseline] ${DATASET}/${SETTING} (two-pass, decoder=${DEC_TYPE})"
        split_manifest_by_duration "${MANIFEST}" "${LONG_DUR_S}"

        SHORT_M="${MANIFEST%.*}_short.${MANIFEST##*.}"
        LONG_M="${MANIFEST%.*}_long.${MANIFEST##*.}"

        # Pass A: short clips
        if [ -s "${SHORT_M}" ]; then
            echo "[baseline]   pass A (short, batch=${BATCH_SIZE}) -> ${OUT_BASE}_short.json"
            python "${EVAL_PY}" \
                model_path="${HYBRID_MODEL}" \
                dataset_manifest="${SHORT_M}" \
                batch_size="${BATCH_SIZE}" \
                output_filename="${OUT_BASE}_short.json" \
                decoder_type="${DEC_TYPE}" \
                save_token_ids=True \
                "${DECODING_ARGS[@]}" \
                "${RTFX_ARGS[@]}"
        else
            echo "[baseline]   pass A skipped (no short clips)"
            : > "${OUT_BASE}_short.json"
        fi

        # Pass B: long clips (batch=1, local attention)
        if [ -s "${LONG_M}" ]; then
            echo "[baseline]   pass B (long, batch=1, local-attn) -> ${OUT_BASE}_long.json"
            python "${EVAL_PY}" \
                model_path="${HYBRID_MODEL}" \
                dataset_manifest="${LONG_M}" \
                batch_size=1 \
                output_filename="${OUT_BASE}_long.json" \
                decoder_type="${DEC_TYPE}" \
                save_token_ids=True \
                model_change.conformer.self_attention_model=rel_pos_local_attn \
                model_change.conformer.att_context_size="${LONG_ATT_CTX}" \
                "${DECODING_ARGS[@]}" \
                "${RTFX_ARGS[@]}"
        else
            echo "[baseline]   pass B skipped (no long clips)"
            : > "${OUT_BASE}_long.json"
        fi

        # Concat (and drop the intermediate per-pass files)
        cat "${OUT_BASE}_short.json" "${OUT_BASE}_long.json" > "${OUT_BASE}.json"
        rm -f "${OUT_BASE}_short.json" "${OUT_BASE}_long.json"

        echo "[baseline] -> ${OUT_BASE}.json"
        ;;

    aed_greedy|aed_beam)
        # --- Determine AED beam size ---
        case "${SETTING}" in
            aed_greedy) BEAM_SIZE=1 ;;
            aed_beam)   BEAM_SIZE="${AED_BEAM_SIZE}" ;;
        esac

        # --- Two-pass AED decode (short + long) ---
        # Canary uses global attention, so the attention matrix scales with
        # batch x T^2 and long clips OOM at large batches. Short clips run at
        # AED_BATCH_SIZE with global attention; long clips run at batch=1 with
        # local attention to keep the attention matrix bounded.
        echo "[baseline] ${DATASET}/${SETTING} (two-pass AED, beam=${BEAM_SIZE})"
        split_manifest_by_duration "${MANIFEST}" "${LONG_DUR_S}"

        SHORT_M="${MANIFEST%.*}_short.${MANIFEST##*.}"
        LONG_M="${MANIFEST%.*}_long.${MANIFEST##*.}"

        # Pass A: short clips
        if [ -s "${SHORT_M}" ]; then
            echo "[baseline]   pass A (short, batch=${AED_BATCH_SIZE}) -> ${OUT_BASE}_short.json"
            python "${EVAL_PY}" \
                model_path="${CANARY_MODEL}" \
                dataset_manifest="${SHORT_M}" \
                batch_size="${AED_BATCH_SIZE}" \
                output_filename="${OUT_BASE}_short.json" \
                save_token_ids=True \
                gt_lang_attr_name="target_lang" \
                gt_text_attr_name="text" \
                multitask_decoding.strategy="beam" \
                multitask_decoding.beam.beam_size="${BEAM_SIZE}" \
                "${RTFX_ARGS[@]}"
        else
            echo "[baseline]   pass A skipped (no short clips)"
            : > "${OUT_BASE}_short.json"
        fi

        # Pass B: long clips (batch=1, local attention)
        if [ -s "${LONG_M}" ]; then
            echo "[baseline]   pass B (long, batch=1, local-attn) -> ${OUT_BASE}_long.json"
            python "${EVAL_PY}" \
                model_path="${CANARY_MODEL}" \
                dataset_manifest="${LONG_M}" \
                batch_size=1 \
                output_filename="${OUT_BASE}_long.json" \
                save_token_ids=True \
                gt_lang_attr_name="target_lang" \
                gt_text_attr_name="text" \
                multitask_decoding.strategy="beam" \
                multitask_decoding.beam.beam_size="${BEAM_SIZE}" \
                model_change.conformer.self_attention_model=rel_pos_local_attn \
                model_change.conformer.att_context_size="${LONG_ATT_CTX}" \
                "${RTFX_ARGS[@]}"
        else
            echo "[baseline]   pass B skipped (no long clips)"
            : > "${OUT_BASE}_long.json"
        fi

        # Concat (and drop the intermediate per-pass files)
        cat "${OUT_BASE}_short.json" "${OUT_BASE}_long.json" > "${OUT_BASE}.json"
        rm -f "${OUT_BASE}_short.json" "${OUT_BASE}_long.json"

        echo "[baseline] -> ${OUT_BASE}.json"
        ;;

    parakeet_tdt)
        # Independent reference-quality model (not in the experiment table).
        # nvidia/parakeet-tdt-0.6b-v2: FastConformer-TDT, greedy, HuggingFace.
        PARAKEET_MODEL="${PARAKEET_MODEL:-nvidia/parakeet-tdt-0.6b-v2}"
        echo "[baseline] ${DATASET}/${SETTING} (two-pass TDT greedy, ${PARAKEET_MODEL})"
        split_manifest_by_duration "${MANIFEST}" "${LONG_DUR_S}"

        SHORT_M="${MANIFEST%.*}_short.${MANIFEST##*.}"
        LONG_M="${MANIFEST%.*}_long.${MANIFEST##*.}"

        if [ -s "${SHORT_M}" ]; then
            echo "[baseline]   pass A (short, batch=${BATCH_SIZE}) -> ${OUT_BASE}_short.json"
            python "${EVAL_PY}" \
                pretrained_name="${PARAKEET_MODEL}" \
                dataset_manifest="${SHORT_M}" \
                batch_size="${BATCH_SIZE}" \
                output_filename="${OUT_BASE}_short.json" \
                "${RTFX_ARGS[@]}"
        else
            echo "[baseline]   pass A skipped (no short clips)"
            : > "${OUT_BASE}_short.json"
        fi

        if [ -s "${LONG_M}" ]; then
            echo "[baseline]   pass B (long, batch=1, local-attn) -> ${OUT_BASE}_long.json"
            python "${EVAL_PY}" \
                pretrained_name="${PARAKEET_MODEL}" \
                dataset_manifest="${LONG_M}" \
                batch_size=1 \
                output_filename="${OUT_BASE}_long.json" \
                model_change.conformer.self_attention_model=rel_pos_local_attn \
                model_change.conformer.att_context_size="${LONG_ATT_CTX}" \
                "${RTFX_ARGS[@]}"
        else
            echo "[baseline]   pass B skipped (no long clips)"
            : > "${OUT_BASE}_long.json"
        fi

        cat "${OUT_BASE}_short.json" "${OUT_BASE}_long.json" > "${OUT_BASE}.json"
        rm -f "${OUT_BASE}_short.json" "${OUT_BASE}_long.json"
        echo "[baseline] -> ${OUT_BASE}.json"
        ;;

    *)
        echo "[baseline] ERROR: unknown SETTING='${SETTING}'" >&2
        echo "[baseline]        must be one of: ctc_greedy ctc_beam rnnt_greedy rnnt_beam aed_greedy aed_beam parakeet_tdt" >&2
        exit 1
        ;;
esac
