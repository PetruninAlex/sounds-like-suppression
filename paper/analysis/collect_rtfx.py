#!/usr/bin/env python3
"""Turn the decode timings in the test-set inference logs into an RTFx table.

RTFx (inverse real-time factor) is the audio duration a decoder gets through per
second of wall clock: ``RTFx = total audio seconds / decoding seconds``. Higher
is faster; RTFx = 1 is exactly real time.

The timings come from ``speech_to_text_eval.py`` (via ``transcribe_speech.py``),
which wraps ``asr_model.transcribe()`` in a CUDA-synchronised timer and logs

    Model time avg: <seconds>

once per process. Because model load, the manifest short/long split and writing
the predictions all sit outside that timer, this is decoding time only -- unlike
wall-clocking the shell script, where a small test set is dominated by the ~30 s
of NeMo import + checkpoint load.

CTC/RNN-T (and Canary) test runs decode in TWO processes -- short clips at the
full batch size, long clips at batch=1 with local attention -- so a log holds two
``Model time avg`` lines covering disjoint halves of the manifest. Summing them
and dividing the FULL manifest duration by that sum gives the one RTFx for the
run, which is what this script reports.

Every method must be timed on the same manifest for the numbers to be
comparable, hence the single ``--manifest``.

Usage:
    python paper/analysis/collect_rtfx.py \
        --manifest .../baseline.wer_filtered.json \
        --log no_boost=.../rtfx_logs/no_boost.log \
        --log boost_only=.../rtfx_logs/boost_only.log \
        --log sounds_like=.../rtfx_logs/sounds_like.log \
        --out .../test_eval/test_rtfx.tsv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from summarize_boost_f1 import write_aligned_txt  # noqa: E402

# "[NeMo I 2026-08-02 12:00:00 transcribe_speech:438] Model time avg: 41.703"
MODEL_TIME_RE = re.compile(r"Model time avg:\s*([0-9]*\.?[0-9]+)")


def manifest_duration(manifest: Path) -> float:
    total = 0.0
    with manifest.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                total += float(json.loads(line).get("duration") or 0.0)
    return total


def decode_seconds(log: Path) -> list[float]:
    """Per-process decode times found in one method's log (short pass, long pass)."""
    if not log.is_file():
        return []
    text = log.read_text(encoding="utf-8", errors="replace")
    return [float(m) for m in MODEL_TIME_RE.findall(text)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Manifest every timed method decoded (needs a 'duration' field).")
    ap.add_argument("--log", action="append", default=[], metavar="METHOD=PATH",
                    help="Inference log for one method; repeat per method.")
    ap.add_argument("--out", type=Path, required=True, help="Output TSV.")
    args = ap.parse_args()

    if not args.manifest.is_file():
        print(f"[rtfx] ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    audio_s = manifest_duration(args.manifest)
    if audio_s <= 0:
        print(f"[rtfx] ERROR: no audio duration in {args.manifest}", file=sys.stderr)
        return 1

    cols = ["method", "audio_s", "decode_s", "rtfx", "passes"]
    rows: list[list[str]] = []
    for spec in args.log:
        method, _, path = spec.partition("=")
        times = decode_seconds(Path(path))
        if not times:
            print(f"[rtfx] WARN: no 'Model time avg' line in {path}; skipping {method}",
                  file=sys.stderr)
            continue
        total = sum(times)
        rows.append([method, f"{audio_s:.1f}", f"{total:.1f}",
                     f"{audio_s / total:.1f}", str(len(times))])

    if not rows:
        print("[rtfx] ERROR: no usable logs, nothing written", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for row in rows:
            fh.write("\t".join(row) + "\n")
    write_aligned_txt(
        args.out.with_suffix(".txt"), cols, rows, label_cols=1,
        title=f"Decoding throughput on {args.manifest.name} ({audio_s / 3600:.2f} h audio)",
    )

    widths = [max(len(cols[i]), *(len(r[i]) for r in rows)) for i in range(len(cols))]

    def line(cells: list[str]) -> str:
        return "  ".join(
            cells[i].ljust(widths[i]) if i == 0 else cells[i].rjust(widths[i])
            for i in range(len(cells))
        ).rstrip()

    print(f"[rtfx] {args.manifest} ({audio_s / 3600:.2f} h audio)")
    print("[rtfx] " + line(cols))
    for row in rows:
        print("[rtfx] " + line(row))
    print(f"[rtfx] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
