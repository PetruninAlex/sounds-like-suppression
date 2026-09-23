#!/usr/bin/env python3
"""Step 4: Filter ASR predictions by per-utterance WER (paper §IV).

Reads a NeMo predictions JSONL produced in step 3, recomputes WER per
utterance from the normalized ``text`` / ``pred_text`` fields, drops rows
with ``wer > --max-wer`` (default 0.30), and writes a filtered predictions
JSONL that downstream steps (5. align, 6. error pairs, ...) consume as a
drop-in replacement. Drops utterances whose ground-truth transcript
disagrees badly with what the baseline ASR model produces -- those are
almost always mis-aligned / mis-labeled references and they pollute the
WER and per-word F1 lift signals computed by the later analysis steps.

WER is computed on text normalized the same way ``align_predictions.py``
does it (lowercase + punctuation stripped except apostrophes + collapsed
whitespace) so the filter is consistent with the downstream alignment step.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from jiwer import wer as jiwer_wer

from align_predictions import normalize_text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, required=True,
                    help="NeMo predictions JSONL produced in step 3.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output filtered predictions JSONL.")
    ap.add_argument("--max-wer", type=float, default=0.30,
                    help="Drop utterances with WER above this (default: %(default)s).")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = drop = empty_ref = 0
    with args.predictions.open() as fin, args.out.open("w") as fo:
        for line in fin:
            if not line.strip(): continue
            d = json.loads(line)
            ref_line = normalize_text(d.get("text") or "")
            pred_line = normalize_text(d.get("pred_text") or "")

            if not ref_line:
                empty_ref += 1
                drop += 1
                continue
            wer = jiwer_wer(ref_line, pred_line)
            d["wer"] = wer
            if wer <= args.max_wer:
                fo.write(json.dumps(d) + "\n"); kept += 1
            else:
                drop += 1
    total = kept + drop
    pct = 100.0 * kept / total if total else 0.0
    print(f"[filter] {args.predictions.name} -> {args.out.name}: "
          f"kept {kept}/{total} ({pct:.1f}%), dropped {drop} "
          f"(empty-ref: {empty_ref})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
