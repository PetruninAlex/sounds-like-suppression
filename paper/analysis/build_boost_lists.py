#!/usr/bin/env python3
"""Build boosting lists from a list of candidate words.

Reads a TSV of words to boost and emits canonical boosting lists in NeMo's
`phrase_boostvalue` per-line format
(see nemo/collections/asr/parts/context_biasing/boosting_graph_batched.py
and docs/source/asr/asr_customization/word_boosting.rst).

The input TSV may either be:
  * a correct-forms / word list with a `word` column (preferred), or
  * an LLM-filtered error-pairs TSV with a `ref` column (legacy);
in both cases the boosted surface form is taken from that column.

Boost values must be in [1, 10] (`DEFAULT_BOOST_VALUE = 1`). One file per
boost value is emitted, so the biased-inference ablation can sweep the boost
weight directly:

  <out-dir>/boost_only_files/boost{B}.txt              for B in 1..max-boost
      one line per unique word: `word_B`  (canonical only)

Pure-Python, no API calls -- safe and cheap to re-run.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load_words(path: Path) -> list[str]:
    """Read the input TSV and return the boost words in input order.

    Prefers a `word` column (correct-forms list); falls back to `ref`
    (legacy error-pairs TSV)."""
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = reader.fieldnames or []
        if "word" in fields:
            col = "word"
        elif "ref" in fields:
            col = "ref"
        else:
            sys.exit(f"{path} has neither a 'word' nor a 'ref' column "
                     f"(found: {fields})")
        return [r[col] for r in reader]


def write_boost_only(words, out_dir: Path, boost_values: list[int]) -> None:
    """One file per boost value: canonical entries only (`word_B`),
    one per unique word in input order (TSV order, count desc)."""
    uniq = list(dict.fromkeys(words))
    sub = out_dir / "boost_only_files"
    sub.mkdir(parents=True, exist_ok=True)
    for b in boost_values:
        (sub / f"boost{b}.txt").write_text(
            "".join(f"{w}_{b}\n" for w in uniq))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True, type=Path,
                    help="TSV of words to boost (a `word` column, or legacy `ref` column).")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("paper/data/predictions"),
                    help="Where to write boost_only_files/ subdir (default: %(default)s).")
    ap.add_argument("--max-boost", type=int, default=10,
                    help="Emit boost1.txt..boost{max-boost}.txt per folder; "
                         "boost values must be in [1, 10] (default: %(default)s).")
    args = ap.parse_args()

    if not 1 <= args.max_boost <= 10:
        sys.exit(f"--max-boost must be in [1, 10], got {args.max_boost}")

    words = load_words(args.input)
    if not words:
        sys.exit(f"no rows in {args.input}")
    print(f"[info] {len(words)} rows ({len(set(words))} unique words) "
          f"from {args.input.name}", file=sys.stderr)

    # Drop words that are too short (<=2 chars). Single-char or two-char terms
    # (like "o", "al") are too generic to boost reliably. Multi-word phrases
    # are measured by total length including spaces.
    MIN_CHARS = 3
    before = len(words)
    words = [w for w in words if len(w) >= MIN_CHARS]
    if before - len(words):
        print(f"[info] {before - len(words)} words dropped (< {MIN_CHARS} chars); "
              f"{len(words)} remain", file=sys.stderr)

    boost_values = list(range(1, args.max_boost + 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_boost_only(words, args.out_dir, boost_values)
    print(f"[done] wrote {args.out_dir}/boost_only_files/boost{{1..{args.max_boost}}}.txt",
          file=sys.stderr)


if __name__ == "__main__":
    main()
