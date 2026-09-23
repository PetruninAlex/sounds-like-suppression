#!/usr/bin/env bash
# Biased ASR inference (GPU-PB / TurboBias) for a single decoder setting.
#
# Self-contained script: calls `python speech_to_text_eval.py` directly
# with all necessary Hydra overrides for boosting.
#
# For CTC/RNN-T decoders, runs two passes:
#   Pass A: short clips (<=LONG_DUR_S) at BATCH_SIZE with full attention
#   Pass B: long clips (>LONG_DUR_S) at batch=1 with local attention
# For AED (Canary): single-pass (no split).
#
# Required env vars:
#   SETTING  — one of: ctc_greedy ctc_beam rnnt_greedy rnnt_beam aed_greedy aed_beam
#
# Optional env vars:
#   DATASET, KEY_PHRASES, BOOST_SUFFIX, HYBRID_BEAM_SIZE, AED_BEAM_SIZE,
#   BT_ALPHA, BT_CONTEXT_SCORE, BATCH_SIZE, LONG_DUR_S, LONG_ATT_CTX,
#   MANIFEST, HYBRID_MODEL, CANARY_MODEL, PAPER_DIR, NEMO_DIR,
#   CALCULATE_RTFX, RTFX_WARMUP_STEPS, ...
#
# Output: predictions/${DATASET}/${SETTING}/boost_only_transcriptions/boost${BOOST_SUFFIX}.json

set -euo pipefail

# ----------------------------------------------------------- paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="${PAPER_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
NEMO_DIR="${NEMO_DIR:-$(cd "${PAPER_DIR}/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${PAPER_DIR}/data}"
MODELS_DIR="${MODELS_DIR:-${NEMO_DIR}/../models}"

# Force Python to import NeMo from the workspace (NEMO_DIR) instead of any
# globally-installed copy under site-packages. Without this, edits to
# nemo/collections/asr/parts/context_biasing/* never take effect at runtime.
export PYTHONPATH="${NEMO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Force protobuf's pure-Python backend so importing NeMo ASR doesn't die with
# "Descriptors cannot be created directly" (installed protobuf 5.x vs onnx's
# generated pb2 descriptors). Negligible perf impact for inference.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

HYBRID_MODEL="${HYBRID_MODEL:-${MODELS_DIR}/stt_en_fastconformer_hybrid_large_pc/stt_en_fastconformer_hybrid_large_pc.nemo}"
CANARY_MODEL="${CANARY_MODEL:-${MODELS_DIR}/canary-1b/canary-1b.nemo}"

DATASET="${DATASET:-multimed}"
MANIFEST="${MANIFEST:-${DATA_DIR}/${DATASET}_nemo/train.json}"
OUT_DIR="${OUT_DIR:-${DATA_DIR}/predictions}"
DATASET_OUT="${OUT_DIR}/${DATASET}"

EVAL_PY="${NEMO_DIR}/examples/asr/speech_to_text_eval.py"

# ----------------------------------------------------------- knobs
SETTING="${SETTING:?SETTING must be set (e.g. rnnt_beam)}"

BOOST_SUFFIX="${BOOST_SUFFIX:-}"

if [ -n "${BOOST_SUFFIX}" ] && [ -z "${KEY_PHRASES:-}" ]; then
    echo "[turbobias] ERROR: BOOST_SUFFIX='${BOOST_SUFFIX}' is set but KEY_PHRASES is not." >&2
    echo "[turbobias]        Set KEY_PHRASES to the correct boost file for this sweep iteration." >&2
    exit 1
fi
KEY_PHRASES="${KEY_PHRASES:-${DATASET_OUT}/${SETTING}/boost_only_files/boost1.txt}"

BATCH_SIZE="${BATCH_SIZE:-32}"
LONG_DUR_S="${LONG_DUR_S:-30}"
LONG_ATT_CTX="${LONG_ATT_CTX:-[128,128]}"

HYBRID_BEAM_SIZE="${HYBRID_BEAM_SIZE:-8}"
AED_BEAM_SIZE="${AED_BEAM_SIZE:-3}"
# Canary uses global attention, so the attention matrix scales with batch x T^2.
# Short clips (<=~30s) are fine at this default; for manifests with multi-minute
# clips, lower it (e.g. AED_BATCH_SIZE=1) to avoid OOM on the long attention matrix.
AED_BATCH_SIZE="${AED_BATCH_SIZE:-16}"

BT_ALPHA="${BT_ALPHA:-0.5}"
BT_CONTEXT_SCORE="${BT_CONTEXT_SCORE:-1.0}"
BT_DEPTH_SCALING_HYBRID="${BT_DEPTH_SCALING_HYBRID:-1.0}"
BT_DEPTH_SCALING_AED="${BT_DEPTH_SCALING_AED:-1.0}"

BOOST_TOKENS_FILE="${BOOST_TOKENS_FILE:-}"  # optional; omit to skip

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

# ----------------------------------------------------------- validate
if [ ! -f "${KEY_PHRASES}" ]; then
    echo "[turbobias] ERROR: KEY_PHRASES file not found: ${KEY_PHRASES}" >&2
    echo "[turbobias]        run §8 for SETTING=${SETTING} first, or override KEY_PHRASES=..." >&2
    exit 1
fi

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
OUT_SUBDIR="${OUT_SUBDIR:-boost_only_transcriptions}"
OUT_BASE="${DATASET_OUT}/${SETTING}/${OUT_SUBDIR}/boost${BOOST_SUFFIX}"
mkdir -p "$(dirname "${OUT_BASE}")"

case "${SETTING}" in
    ctc_greedy|ctc_beam|rnnt_greedy|rnnt_beam)
        # --- Determine decoder_type and Hydra overrides ---
        case "${SETTING}" in
            ctc_greedy)
                DEC_TYPE=ctc
                DECODING_ARGS=(
                    ctc_decoding.strategy="greedy_batch"
                    ctc_decoding.greedy.boosting_tree.key_phrases_file="${KEY_PHRASES}"
                    ctc_decoding.greedy.boosting_tree.context_score="${BT_CONTEXT_SCORE}"
                    ctc_decoding.greedy.boosting_tree.depth_scaling="${BT_DEPTH_SCALING_HYBRID}"
                    ctc_decoding.greedy.boosting_tree_alpha="${BT_ALPHA}"
                )
                [ -n "${BOOST_TOKENS_FILE:-}" ] && DECODING_ARGS+=(ctc_decoding.greedy.boosting_tree.boost_tokens_file="${BOOST_TOKENS_FILE}")
                ;;
            ctc_beam)
                DEC_TYPE=ctc
                DECODING_ARGS=(
                    ctc_decoding.strategy="beam_batch"
                    ctc_decoding.beam.beam_size="${HYBRID_BEAM_SIZE}"
                    ctc_decoding.beam.boosting_tree.key_phrases_file="${KEY_PHRASES}"
                    ctc_decoding.beam.boosting_tree.context_score="${BT_CONTEXT_SCORE}"
                    ctc_decoding.beam.boosting_tree.depth_scaling="${BT_DEPTH_SCALING_HYBRID}"
                    ctc_decoding.beam.boosting_tree_alpha="${BT_ALPHA}"
                )
                [ -n "${BOOST_TOKENS_FILE:-}" ] && DECODING_ARGS+=(ctc_decoding.beam.boosting_tree.boost_tokens_file="${BOOST_TOKENS_FILE}")
                ;;
            rnnt_greedy)
                DEC_TYPE=rnnt
                DECODING_ARGS=(
                    rnnt_decoding.strategy="greedy_batch"
                    rnnt_decoding.greedy.boosting_tree.key_phrases_file="${KEY_PHRASES}"
                    rnnt_decoding.greedy.boosting_tree.context_score="${BT_CONTEXT_SCORE}"
                    rnnt_decoding.greedy.boosting_tree.depth_scaling="${BT_DEPTH_SCALING_HYBRID}"
                    rnnt_decoding.greedy.boosting_tree_alpha="${BT_ALPHA}"
                )
                [ -n "${BOOST_TOKENS_FILE:-}" ] && DECODING_ARGS+=(rnnt_decoding.greedy.boosting_tree.boost_tokens_file="${BOOST_TOKENS_FILE}")
                ;;
            rnnt_beam)
                DEC_TYPE=rnnt
                DECODING_ARGS=(
                    rnnt_decoding.strategy="malsd_batch"
                    rnnt_decoding.beam.beam_size="${HYBRID_BEAM_SIZE}"
                    rnnt_decoding.beam.boosting_tree.key_phrases_file="${KEY_PHRASES}"
                    rnnt_decoding.beam.boosting_tree.context_score="${BT_CONTEXT_SCORE}"
                    rnnt_decoding.beam.boosting_tree.depth_scaling="${BT_DEPTH_SCALING_HYBRID}"
                    rnnt_decoding.beam.boosting_tree_alpha="${BT_ALPHA}"
                )
                [ -n "${BOOST_TOKENS_FILE:-}" ] && DECODING_ARGS+=(rnnt_decoding.beam.boosting_tree.boost_tokens_file="${BOOST_TOKENS_FILE}")
                ;;
        esac

        # --- Two-pass decode (short + long) ---
        echo "[turbobias] ${OUT_BASE} (two-pass, decoder=${DEC_TYPE})"
        split_manifest_by_duration "${MANIFEST}" "${LONG_DUR_S}"

        SHORT_M="${MANIFEST%.*}_short.${MANIFEST##*.}"
        LONG_M="${MANIFEST%.*}_long.${MANIFEST##*.}"

        # Pass A: short clips
        if [ -s "${SHORT_M}" ]; then
            echo "[turbobias]   pass A (short, batch=${BATCH_SIZE}) -> ${OUT_BASE}_short.json"
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
            echo "[turbobias]   pass A skipped (no short clips)"
            : > "${OUT_BASE}_short.json"
        fi

        # Pass B: long clips (batch=1, local attention)
        if [ -s "${LONG_M}" ]; then
            echo "[turbobias]   pass B (long, batch=1, local-attn) -> ${OUT_BASE}_long.json"
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
            echo "[turbobias]   pass B skipped (no long clips)"
            : > "${OUT_BASE}_long.json"
        fi

        # Concat (and drop the intermediate per-pass files)
        cat "${OUT_BASE}_short.json" "${OUT_BASE}_long.json" > "${OUT_BASE}.json"
        rm -f "${OUT_BASE}_short.json" "${OUT_BASE}_long.json"

        echo "[turbobias] -> ${OUT_BASE}.json"
        ;;

    aed_greedy|aed_beam)
        # --- Determine AED beam size ---
        case "${SETTING}" in
            aed_greedy) BEAM_SIZE=1 ;;
            aed_beam)   BEAM_SIZE="${AED_BEAM_SIZE}" ;;
        esac

        # --- Common AED args (model/decoding/boosting), shared by both passes ---
        AED_BASE=(
            model_path="${CANARY_MODEL}"
            save_token_ids=True
            gt_lang_attr_name="target_lang"
            gt_text_attr_name="text"
            multitask_decoding.strategy="beam"
            multitask_decoding.beam.beam_size="${BEAM_SIZE}"
            multitask_decoding.beam.boosting_tree.key_phrases_file="${KEY_PHRASES}"
            multitask_decoding.beam.boosting_tree.context_score="${BT_CONTEXT_SCORE}"
            multitask_decoding.beam.boosting_tree.depth_scaling="${BT_DEPTH_SCALING_AED}"
            multitask_decoding.beam.boosting_tree_alpha="${BT_ALPHA}"
        )
        [ -n "${BOOST_TOKENS_FILE:-}" ] && AED_BASE+=(multitask_decoding.beam.boosting_tree.boost_tokens_file="${BOOST_TOKENS_FILE}")

        # --- Two-pass AED decode (short + long) ---
        # Canary uses global attention (batch x T^2), so long clips OOM at large
        # batches. Short clips run at AED_BATCH_SIZE; long clips run at batch=1 with
        # local attention to keep the attention matrix bounded.
        echo "[turbobias] ${OUT_BASE} (two-pass AED, beam=${BEAM_SIZE})"
        split_manifest_by_duration "${MANIFEST}" "${LONG_DUR_S}"

        SHORT_M="${MANIFEST%.*}_short.${MANIFEST##*.}"
        LONG_M="${MANIFEST%.*}_long.${MANIFEST##*.}"

        # Pass A: short clips
        if [ -s "${SHORT_M}" ]; then
            echo "[turbobias]   pass A (short, batch=${AED_BATCH_SIZE}) -> ${OUT_BASE}_short.json"
            python "${EVAL_PY}" \
                dataset_manifest="${SHORT_M}" \
                batch_size="${AED_BATCH_SIZE}" \
                output_filename="${OUT_BASE}_short.json" \
                "${AED_BASE[@]}" \
                "${RTFX_ARGS[@]}"
        else
            echo "[turbobias]   pass A skipped (no short clips)"
            : > "${OUT_BASE}_short.json"
        fi

        # Pass B: long clips (batch=1, local attention)
        if [ -s "${LONG_M}" ]; then
            echo "[turbobias]   pass B (long, batch=1, local-attn) -> ${OUT_BASE}_long.json"
            python "${EVAL_PY}" \
                dataset_manifest="${LONG_M}" \
                batch_size=1 \
                output_filename="${OUT_BASE}_long.json" \
                model_change.conformer.self_attention_model=rel_pos_local_attn \
                model_change.conformer.att_context_size="${LONG_ATT_CTX}" \
                "${AED_BASE[@]}" \
                "${RTFX_ARGS[@]}"
        else
            echo "[turbobias]   pass B skipped (no long clips)"
            : > "${OUT_BASE}_long.json"
        fi

        # Concat (and drop the intermediate per-pass files)
        cat "${OUT_BASE}_short.json" "${OUT_BASE}_long.json" > "${OUT_BASE}.json"
        rm -f "${OUT_BASE}_short.json" "${OUT_BASE}_long.json"

        echo "[turbobias] -> ${OUT_BASE}.json"
        ;;

    *)
        echo "[turbobias] ERROR: unknown SETTING='${SETTING}'" >&2
        echo "[turbobias]        must be one of: ctc_greedy ctc_beam rnnt_greedy rnnt_beam aed_greedy aed_beam" >&2
        exit 1
        ;;
esac
