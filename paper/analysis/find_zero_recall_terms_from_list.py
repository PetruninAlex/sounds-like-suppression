#!/usr/bin/env python3
"""Keep the released context-list terms the model never gets right.

Two setups already exist: boost the whole released list (dense, mostly terms the
baseline already handles), or mine candidates from the model's errors as free
n-grams. The second targets the right vocabulary but discovers strings that are
not entities -- on stop_music rnnt_beam it produced `beyonce concert` alongside
`beyonce`, plus `jackson 's`, `the beatles '`, and halves of names like `aiko`
and `bone thugs n` -- because every n-gram length is counted independently and
nothing enforces entity boundaries.

This combines them: candidates are exactly the released terms, and a term is kept
only when the baseline never once recognises it (recall == 0) in the split it is
measured on. The result is the hard subset of a known list, so entity boundaries
come from the annotation while the selection still comes from the model's own
errors.

Recall is per occurrence: for each utterance whose reference contains the term,
the hypothesis is checked for it. A term with no hits anywhere has recall 0.
"""

import argparse
import csv
import json
import re
from pathlib import Path


def normalize(text: str) -> str:
    # Mirror align_predictions.py (the F1 scorer): strip only commas, periods
    # and question marks by DELETING them (not replacing with a space), keep all
    # other punctuation, then collapse whitespace. Deleting rather than spacing
    # means the model's punctuated output ("Portland, Maine") normalizes to the
    # same "portland maine" as the reference, so a term the model actually gets
    # right is not wrongly counted as zero-recall.
    return " ".join(re.sub(r"[,.?]", "", (text or "").lower()).split())


def count_in(term: str, text: str) -> int:
    return len(re.findall(rf"(?<![a-z]){re.escape(term)}(?![a-z])", text))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", type=Path, required=True,
                    help="Released context list, one term per line.")
    ap.add_argument("--predictions", type=Path, required=True,
                    help="Baseline predictions JSONL with 'text' and 'pred_text'.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--min-count", type=int, default=3,
                    help="Minimum reference occurrences for a term to be "
                         "considered at all (default: %(default)s).")
    ap.add_argument("--max-recall", type=float, default=0.0,
                    help="Keep terms whose recall is at most this. 0.0 is the "
                         "zero-recall rule; raise it to also catch terms the "
                         "model gets right only occasionally (default: %(default)s).")
    args = ap.parse_args()

    terms = [t.strip() for t in args.list.read_text().splitlines() if t.strip()]
    recs = [json.loads(l) for l in args.predictions.read_text().splitlines() if l.strip()]
    pairs = [(normalize(r.get("text")), normalize(r.get("pred_text"))) for r in recs]

    rows = []
    for term in terms:
        t = term.lower()
        n_ref = n_hit = 0
        for ref, hyp in pairs:
            c = count_in(t, ref)
            if not c:
                continue
            n_ref += c
            # Credit at most the number of reference occurrences, so a term the
            # model over-emits cannot look better recalled than it is.
            n_hit += min(c, count_in(t, hyp))
        if n_ref < args.min_count:
            continue
        recall = n_hit / n_ref
        if recall <= args.max_recall:
            rows.append((term, n_ref, recall))

    rows.sort(key=lambda r: -r[1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["word", "count", "recall"])
        w.writerows((t, n, f"{r:.4f}") for t, n, r in rows)

    print(f"[zr-from-list] {len(terms)} listed terms -> {len(rows)} with recall "
          f"<= {args.max_recall} and >= {args.min_count} occurrences -> {args.output}")


if __name__ == "__main__":
    main()
