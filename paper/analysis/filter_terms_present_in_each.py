#!/usr/bin/env python3
"""Keep only candidate terms that occur in EVERY given manifest's references.

Used to enforce "present in both val and test": a boosted term should appear at
least once in the validation references (so the boost/sounds-like selection can
see it) and at least once in the test references (so it is actually scored).
Terms confined to a single split add WER risk with no measurable upside on the
other, so they are dropped.

Occurrence is a contiguous whole-word window match in the normalized reference
`text` field -- the same normalization the rest of the pipeline uses -- so it is
consistent with find_correct_word_forms.py / the F1 scorer.

Input is a TSV with a `word` column (e.g. extract_boost_phrases.py output).
Output is the same TSV filtered to surviving rows (columns preserved).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def normalize(text: str) -> str:
    # Mirror align_predictions.py (the F1 scorer): strip only commas, periods
    # and question marks by DELETING them (not replacing with a space), keep all
    # other punctuation, then collapse whitespace, so presence matching agrees
    # with how the term is scored.
    return " ".join(re.sub(r"[,.?]", "", (text or "").lower()).split())


def present_set(manifest: Path, terms: set[str]) -> set[str]:
    """Return the subset of `terms` occurring at least once in this manifest's
    normalized references (whole-word contiguous window match)."""
    if not manifest.is_file():
        sys.exit(f"[fatal] manifest not found: {manifest}")
    refs = []
    with manifest.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                refs.append(normalize(json.loads(line).get("text")))
    blob = "\n".join(refs)  # scan the whole corpus once per term
    found = set()
    for t in terms:
        if re.search(rf"(?<![a-z]){re.escape(t.lower())}(?![a-z])", blob):
            found.add(t)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", "-i", required=True, type=Path,
                    help="TSV with a `word` column (e.g. extract_boost_phrases output).")
    ap.add_argument("--manifest", "-m", required=True, type=Path, nargs="+",
                    help="Manifests whose `text` references a term must appear in "
                         "EACH of (e.g. val.json test.json).")
    ap.add_argument("--output", "-o", required=True, type=Path,
                    help="Filtered TSV (same columns).")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.input.open(), delimiter="\t"))
    if not rows:
        args.output.write_text("", encoding="utf-8")
        print(f"[present-each] empty input {args.input}", file=sys.stderr)
        return 0
    fieldnames = list(rows[0].keys())
    if "word" not in fieldnames:
        sys.exit(f"{args.input}: expected a 'word' column, got {fieldnames}")

    terms = {r["word"] for r in rows}
    # A term survives iff it is present in every manifest -> intersection.
    survivors = None
    for m in args.manifest:
        found = present_set(m, terms)
        survivors = found if survivors is None else (survivors & found)

    kept = [r for r in rows if r["word"] in survivors]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)

    print(f"[present-each] {len(rows)} terms -> kept {len(kept)} present in all "
          f"{len(args.manifest)} manifest(s) (dropped {len(rows) - len(kept)}) "
          f"-> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    main()
