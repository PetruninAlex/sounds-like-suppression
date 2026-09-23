#!/usr/bin/env python3
"""Step 16: Build boost with sounds-like files from classified pairs after best boosting.

Reads the LLM-classified TSV produced in step 15 and emits one boost-with-
sounds-like file per suppression weight. Each file uses the best boost value
(from step 12) as the canonical boost and sweeps the sounds-like weight from
min-sl-weight to max-sl-weight:

  <out-dir>/boost_with_sounds_like_{W}.txt   for W in min-sl-weight..max-sl-weight
      `ref_{BEST_BOOST}`   (boost the target word)
      `hyp_-{W}`           (suppress the sounds-like confusion)

A boost word can have several confusions and a confusion can be shared by
several boost words, so each phrase is emitted at most once: one boost line per
unique target and one suppression line per unique confusion. A confusion that
is also a boost word is never suppressed.

When --manifest is given, confusions that appear frequently in the training
references (>= --min-ref-count occurrences) are also excluded from suppression:
they are legitimate vocabulary words and suppressing them would hurt ASR on
utterances where they are correct.

Pure-Python, no API calls.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path


def load_pairs(path: Path) -> list[tuple[str, str]]:
    """Read step-15 LLM-classified TSV; return [(word, sounds_like)] in input
    order. `word` is the target boost word, `sounds_like` is the confusion the
    ASR produced in its place."""
    with path.open() as f:
        return [(r["word"], r["sounds_like"])
                for r in csv.DictReader(f, delimiter="\t")]


def count_words_in_references(manifest: Path) -> Counter:
    """Count how many utterances each lowercased word appears in, and also
    return the full normalized reference texts for phrase matching."""
    counts: Counter = Counter()
    texts: list[str] = []
    with manifest.open() as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text", "") or ""
            # Normalize: lowercase, strip punctuation except apostrophes
            text = text.lower()
            text = re.sub(r"[^\w\s']", " ", text)
            tokens = text.split()
            # Count unique words per utterance (presence, not frequency)
            counts.update(set(tokens))
            texts.append(" ".join(tokens))
    return counts, texts


def phrase_in_references(phrase: str, ref_counts: Counter, ref_texts: list[str], min_count: int) -> bool:
    """Check if the full phrase (as a contiguous sequence) appears in >= min_count reference utterances."""
    needle = phrase.lower().strip()
    if not needle:
        return False
    # For single words, use the fast counter lookup
    if " " not in needle:
        return ref_counts.get(needle, 0) >= min_count
    # For multi-word phrases, check how many utterances contain the full phrase
    occurrences = sum(1 for text in ref_texts if needle in text)
    return occurrences >= min_count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True, type=Path,
                    help="Step-15 LLM-classified TSV (columns: audio_filepath word "
                         "ref_text hyp_text pred_token_ids sounds_like).")
    ap.add_argument("--boost-file", type=Path, default=None,
                    help="Boost file (e.g. boost1.txt); words without a sounds-like pair "
                         "get a canonical word_{BEST_BOOST} entry.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory to write boost_with_sounds_like_{1..N}.txt files.")
    ap.add_argument("--best-boost", type=int, required=True,
                    help="Best boost value from step 12. Used as the fixed target "
                         "boost weight unless --coupled is set.")
    ap.add_argument("--min-sl-weight", type=int, default=1,
                    help="Lowest sounds-like weight to emit (default: %(default)s).")
    ap.add_argument("--max-sl-weight", type=int, default=10,
                    help="Emit sounds_like_{min-sl-weight}.txt..sounds_like_{max-sl-weight}.txt "
                         "(default: %(default)s).")
    ap.add_argument("--coupled", action="store_true",
                    help="Couple the target boost weight to the suppression weight: "
                         "file W boosts targets at W and suppresses confusions at -W "
                         "(so a sweep varies both together). Default: boost targets at "
                         "the fixed --best-boost while sweeping only the suppression.")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Training manifest (JSONL). When given, confusion words that "
                         "appear in >= --min-ref-count utterances in the references are "
                         "excluded from suppression (they are legitimate vocabulary).")
    ap.add_argument("--min-ref-count", type=int, default=1,
                    help="Minimum utterance-level occurrence count in the manifest "
                         "references for a confusion word to be considered legitimate "
                         "vocabulary and excluded from suppression (default: %(default)s).")
    args = ap.parse_args()

    if not 1 <= args.best_boost <= 10:
        sys.exit(f"--best-boost must be in [1, 10], got {args.best_boost}")
    if not 1 <= args.max_sl_weight <= 10:
        sys.exit(f"--max-sl-weight must be in [1, 10], got {args.max_sl_weight}")
    if not 1 <= args.min_sl_weight <= args.max_sl_weight:
        sys.exit(f"--min-sl-weight must be in [1, {args.max_sl_weight}], "
                 f"got {args.min_sl_weight}")

    pairs = load_pairs(args.input)
    if not pairs:
        sys.exit(f"no rows in {args.input}")
    print(f"[info] {len(pairs)} pairs from {args.input.name}", file=sys.stderr)

    # Load boost words and find those without sounds-like pairs. Compared upper-
    # cased because the TSV's `word` column is copied from STOP's upper-cased
    # reference while --boost-file is truecased by Step 7c; without it nothing was
    # ever excluded and every mined target got a second, upper-cased boost line.
    refs_with_pairs = {ref.upper() for ref, _ in pairs}
    words_without_a_pair: list[str] = []
    if args.boost_file:
        boost_words: list[str] = []
        for line in args.boost_file.read_text(encoding="utf-8").splitlines():
            w = line.strip().split("_")[0]
            if w:
                boost_words.append(w)
        words_without_a_pair = [w for w in boost_words
                                if w.upper() not in refs_with_pairs]
        print(f"[info] {len(words_without_a_pair)} boost words without sounds-like pairs "
              f"(will get word_{args.best_boost} entries)", file=sys.stderr)

    # A boost word can have several sounds-like confusions, and one confusion can
    # be shared by several boost words, so emit each phrase at most once: one
    # `word_B` boost line per unique target, one `hyp_-W` suppression line per
    # unique confusion. A confusion that is itself a boosted word is never
    # suppressed (boosting wins over suppressing the same surface form).
    # Targets are taken from --boost-file so they keep its casing.
    words_with_a_pair = [w for w in boost_words if w.upper() in refs_with_pairs]
    boost_set = set(words_with_a_pair) | set(words_without_a_pair)
    suppress_words = list(dict.fromkeys(
        hyp for _, hyp in pairs if hyp not in boost_set))
    n_confusions_also_boosted = len({hyp for _, hyp in pairs} & boost_set)
    if n_confusions_also_boosted:
        print(f"[info] {n_confusions_also_boosted} sounds-like confusions skipped (also boost words)",
              file=sys.stderr)

    # Drop boost words and suppression words that are too short (<=2 chars).
    # Single-char or two-char terms (like "o", "al") are too generic to boost
    # or suppress reliably; they cause noise. Multi-word phrases are measured
    # by total length including spaces (e.g. "a b" = 3 chars, OK).
    MIN_CHARS = 3
    before_boost = len(words_with_a_pair) + len(words_without_a_pair)
    words_with_a_pair = [w for w in words_with_a_pair if len(w) >= MIN_CHARS]
    words_without_a_pair = [w for w in words_without_a_pair if len(w) >= MIN_CHARS]
    boost_set = set(words_with_a_pair) | set(words_without_a_pair)
    after_boost = len(words_with_a_pair) + len(words_without_a_pair)
    if before_boost - after_boost:
        print(f"[info] {before_boost - after_boost} boost words dropped (< {MIN_CHARS} chars)",
              file=sys.stderr)

    before_suppress = len(suppress_words)
    suppress_words = [w for w in suppress_words if len(w) >= MIN_CHARS]
    if before_suppress - len(suppress_words):
        print(f"[info] {before_suppress - len(suppress_words)} suppression words dropped "
              f"(< {MIN_CHARS} chars); {len(suppress_words)} remain", file=sys.stderr)

    # Filter out suppression words that are legitimate vocabulary (appear
    # frequently in the training references). These are real words the model
    # should sometimes produce; suppressing them causes over-triggering of
    # boost words on utterances where the suppressed word is correct.
    if args.manifest is not None:
        ref_counts, ref_texts = count_words_in_references(args.manifest)
        before = len(suppress_words)
        suppress_words = [
            w for w in suppress_words
            if not phrase_in_references(w, ref_counts, ref_texts, args.min_ref_count)
        ]
        n_filtered = before - len(suppress_words)
        if n_filtered:
            print(f"[info] {n_filtered} suppression words filtered (appear in "
                  f">={args.min_ref_count} reference utterances); "
                  f"{len(suppress_words)} remain", file=sys.stderr)
    

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for w in range(args.min_sl_weight, args.max_sl_weight + 1):
        # In --coupled mode the target boost weight tracks the suppression weight
        # (file W boosts targets at W and suppresses confusions at -W); otherwise
        # targets stay at the fixed --best-boost and only suppression sweeps.
        b = w if args.coupled else args.best_boost
        out_file = args.out_dir / f"boost_with_sounds_like_{w}.txt"
        lines = []
        for word in words_with_a_pair:
            lines.append(f"{word}_{b}\n")
        for hyp in suppress_words:
            lines.append(f"{hyp}_-{w}\n")
        for word in words_without_a_pair:
            lines.append(f"{word}_{b}\n")
        out_file.write_text("".join(lines))

    print(f"[done] wrote {args.out_dir}/boost_with_sounds_like_"
          f"{{{args.min_sl_weight}..{args.max_sl_weight}}}.txt", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
