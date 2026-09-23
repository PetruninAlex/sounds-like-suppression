#!/usr/bin/env python3
"""Per-word F1 table for the test-set method comparison.

Given transcriptions of the SAME test manifest produced up to four ways --

  1) no biasing            (no_boost),
  2) best boost-only file  (boost_only),
  3) best boost + sounds-like file (sounds_like),
  4) best coupled boost + sounds-like file (sounds_like_coupled, optional),

build a per-word F1 table over the boosted key words: one row per boosted word
with a ``ref`` column (reference occurrence count, shared across methods) and,
per method, an F1 column plus a ``<method>_pred`` column (times the method
produced the word = tp + fp). An ``average_F1 (macro)`` row holds the mean F1
per method together with the total reference / predicted counts (sums); an
``average_F1 (ref>0, macro)`` row holds the mean F1 per method over only the
words that occur in the reference (so it is not diluted by mined words absent
from the test references), with its ref cell holding the count of such words;
``micro_P`` / ``micro_R`` / ``micro_F1`` hold the pooled metric and
``over_trigger`` the false positives on ref=0 words; and a ``WER`` row holds the
total corpus WER per method. The F1/WER helpers are reused from
``summarize_boost_f1.py`` so normalization and alignment match the pipeline.

Prints the aligned table to stdout and (optionally) writes a TSV + .txt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from summarize_boost_f1 import (  # noqa: E402
    load_keys,
    manifest_wer,
    per_word_stats,
    prf,
    write_aligned_txt,
)


def word_f1(stats: list[int]) -> float:
    """F1 for one word/method given [tp, ref_count, fp].

    Standard convention: when precision and recall are both 0 (incl. the empty
    [0, 0, 0] case), F1 is 0.
    """
    return prf(stats[0], stats[1], stats[2])[2]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--no-boost", required=True, type=Path,
                    help="Transcription JSONL with no biasing.")
    ap.add_argument("--boost", required=True, type=Path,
                    help="Transcription JSONL with the best boost-only file.")
    ap.add_argument("--sounds-like", required=True, type=Path,
                    help="Transcription JSONL with the best sounds-like file.")
    ap.add_argument("--sounds-like-coupled", type=Path, default=None,
                    help="Optional transcription JSONL with the best COUPLED "
                         "boost+sounds-like file (boost W + suppress -W). Added as "
                         "an extra method column when given.")
    ap.add_argument("--kw-file", required=True, type=Path,
                    help="Boost key-words file (e.g. boost_only_files/boost1.txt).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional TSV to write (a .txt sibling is written too).")
    args = ap.parse_args()

    key_list = load_keys(args.kw_file)
    if not key_list:
        print(f"[error] empty key-words file {args.kw_file}", file=sys.stderr)
        return 1
    keys = set(key_list)
    print(f"[info] {len(key_list)} boosted key words from {args.kw_file}", file=sys.stderr)

    methods = [
        ("no_boost", args.no_boost),
        ("boost_only", args.boost),
        ("sounds_like", args.sounds_like),
    ]
    if args.sounds_like_coupled is not None:
        methods.append(("sounds_like_coupled", args.sounds_like_coupled))
    present = []
    for name, manifest in methods:
        if manifest.is_file():
            present.append((name, manifest))
        else:
            print(f"[warn] missing {manifest}; skipping '{name}'", file=sys.stderr)
    if not present:
        print("[error] no transcriptions found to report", file=sys.stderr)
        return 1

    # Per-word tp/ref_count/fp for each method, once.
    stats_by_method = {name: per_word_stats(m, keys) for name, m in present}

    # Drop only words with stats [0, 0, 0] in EVERY method -- i.e. never in the
    # reference, never correct, and never produced by any method. Words a method
    # emitted (fp > 0) are kept even if absent from the reference.
    kept = [w for w in key_list
            if any(any(stats_by_method[name][w]) for name, _ in present)]
    dropped = len(key_list) - len(kept)
    if dropped:
        print(f"[info] dropped {dropped} word(s) with no presence in any method "
              f"(stats [0,0,0]); {len(kept)} kept", file=sys.stderr)
    if not kept:
        print("[error] no boosted words have any presence in the transcriptions",
              file=sys.stderr)
        return 1
    key_list = kept

    # Reference count is a property of the (shared) ground-truth text, so it is
    # the same across methods; predicted count = tp + fp is per method.
    def ref_count(word: str) -> int:
        return max(stats_by_method[name][word][1] for name, _ in present)

    def pred_count(name: str, word: str) -> int:
        st = stats_by_method[name][word]
        return st[0] + st[2]

    cols = ["word", "ref"]
    for name, _ in present:
        cols += [name, f"{name}_pred"]
    rows: list[list[str]] = []
    for word in key_list:
        row = [word, str(ref_count(word))]
        for name, _ in present:
            row.append(f"{word_f1(stats_by_method[name][word]):.4f}")
            row.append(str(pred_count(name, word)))
        rows.append(row)

    n = len(key_list)
    # "average"/total row: mean F1 per method, total reference and total
    # predicted counts (sums) so the column totals are meaningful.
    avg_row = ["average_F1 (macro)", str(sum(ref_count(w) for w in key_list))]
    for name, _ in present:
        f1s = [word_f1(stats_by_method[name][w]) for w in key_list]
        avg_row.append(f"{sum(f1s) / n:.4f}")
        avg_row.append(str(sum(pred_count(name, w) for w in key_list)))
    rows.append(avg_row)

    # "average (ref>0)" row: mean F1 per method over ONLY the words that occur in
    # the reference (ref_count > 0). The plain "average" above is diluted by mined
    # words that never appear in the test references, so this reflects the real
    # lift on words the model can actually be scored on. The ref cell holds the
    # number of such words (the averaging denominator); pred cells are left blank.
    ref_words = [w for w in key_list if ref_count(w) > 0]
    n_ref = len(ref_words)
    avg_ref_row = ["average_F1 (ref>0, macro)", str(n_ref)]
    for name, _ in present:
        if n_ref:
            f1s = [word_f1(stats_by_method[name][w]) for w in ref_words]
            avg_ref_row.append(f"{sum(f1s) / n_ref:.4f}")
        else:
            avg_ref_row.append("0.0000")
        avg_ref_row.append("")
    rows.append(avg_ref_row)

    # Micro-averaged precision / recall / F1: pool tp / fp / ref across all kept
    # words, then compute the score once. Unlike the macro "average" rows (which
    # mean per-word F1s and force ref=0 words to 0 for everyone), this charges
    # every false positive -- including spurious hits on words absent from the
    # reference -- against precision, so a method that over-triggers boosted
    # words it should not is penalized rather than scored identically to a
    # method that stayed silent. ``over_trigger`` counts exactly those false
    # positives on ref=0 words.
    micro_p_row = ["micro_P", ""]
    micro_r_row = ["micro_R", ""]
    micro_f1_row = ["micro_F1", ""]
    over_row = ["over_trigger (fp@ref=0)", ""]
    for name, _ in present:
        tp = sum(stats_by_method[name][w][0] for w in key_list)
        fp = sum(stats_by_method[name][w][2] for w in key_list)
        ref = sum(stats_by_method[name][w][1] for w in key_list)
        p, r, f = prf(tp, ref, fp)
        micro_p_row += [f"{p:.4f}", ""]
        micro_r_row += [f"{r:.4f}", ""]
        micro_f1_row += [f"{f:.4f}", ""]
        over = sum(stats_by_method[name][w][2] for w in key_list if ref_count(w) == 0)
        over_row += [str(over), ""]
    rows.append(micro_p_row)
    rows.append(micro_r_row)
    rows.append(micro_f1_row)
    rows.append(over_row)

    wer_row = ["WER", ""]
    for _, manifest in present:
        wer_row.append(f"{manifest_wer(manifest):.8f}")
        wer_row.append("")
    rows.append(wer_row)

    widths = [max(len(cols[i]), *(len(r[i]) for r in rows)) for i in range(len(cols))]

    def fmt(cells: list[str]) -> str:
        return "  ".join(
            cells[i].ljust(widths[i]) if i == 0 else cells[i].rjust(widths[i])
            for i in range(len(cells))
        )

    print(fmt(cols))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print(fmt(row))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            fh.write("\t".join(cols) + "\n")
            for row in rows:
                fh.write("\t".join(row) + "\n")
        write_aligned_txt(
            args.out.with_suffix(".txt"), cols, rows, label_cols=1,
            title="Test-set per-word F1 (boosted words) with reference + predicted counts + total WER",
        )
        print(f"[done] wrote {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
