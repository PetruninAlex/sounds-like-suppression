#!/usr/bin/env python3
"""Recover the original surface spelling of each candidate word / term.

The alignment pipeline normalizes reference text (lowercase + punctuation
stripped), so the candidate units (e.g. from `llm_classify_words.py`) are
lowercased and depunctuated -- 'snps', "sjogren's", 'ehdi', or multi-word
terms like "sjogren's syndrome". This step looks those units back up in the
*original* reference transcripts and reports the surface form they were
actually written as -- 'SNPs', "Sjogren's", 'EHDI', "Sjogren's syndrome".

For every unit in the input TSV (must have a `word` column) we scan the
`text` field of the manifest, tokenize it the same way `align_predictions.py`
normalizes (split on punctuation except apostrophes), and match by lowercase.
Multi-word terms are matched as consecutive token windows of the same length.

The output is a two-column TSV (`word`, `count`): one row per observed
surface form with its reference occurrence count, sorted by count (desc)
then alphabetically.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Mirror align_predictions.py: strip only commas, periods, and question marks;
# keep all other punctuation (apostrophes, hyphens, etc.).
_PUNCT_RE = re.compile(f"[{re.escape(',.?')}]")


def surface_tokens(text: str):
    """Tokenize like normalize_text but keep the original casing."""
    return _PUNCT_RE.sub("", text).split()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True, type=Path,
                    help="TSV of candidate words (must have a `word` column).")
    ap.add_argument("--manifest", "-m", required=True, type=Path, nargs="+",
                    help="One or more NeMo predictions JSONLs whose `text` field "
                         "holds the original (un-normalized) reference "
                         "transcripts. Pass several (e.g. train + dev) to recover "
                         "surface forms over the same pooled corpus.")
    ap.add_argument("--output", "-o", required=True, type=Path,
                    help="Output TSV (input columns plus `correct_form`).")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.input.open(), delimiter="\t"))
    targets = {r["word"] for r in rows}
    # Group targets by token length so multi-word terms are matched as windows.
    targets_by_len: dict = {}
    for t in targets:
        n = len(t.split())
        if n:
            targets_by_len.setdefault(n, set()).add(t)

    forms: dict = defaultdict(Counter)  # normalized unit -> Counter(surface form)
    for manifest in args.manifest:
        with manifest.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get("text", "") or ""
                toks = surface_tokens(text)
                for n, tset in targets_by_len.items():
                    for i in range(len(toks) - n + 1):
                        window = toks[i:i + n]
                        low = " ".join(w.lower() for w in window)
                        if low in tset:
                            forms[low][" ".join(window)] += 1

    # Flatten every observed surface form into its own (word, count) row.
    out_counts: Counter = Counter()
    for r in rows:
        counter = forms.get(r["word"])
        if counter:
            out_counts.update(counter)
    ordered = sorted(out_counts.items(), key=lambda kv: (-kv[1], kv[0]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["word", "count"])
        for form, n in ordered:
            w.writerow([form, n])

    print(f"[done] {len(ordered)} surface forms -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
