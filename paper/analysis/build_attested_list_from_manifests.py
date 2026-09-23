#!/usr/bin/env python3
"""Build the attested entity list from one or more manifests' ``hotwords``.

STOP manifests (produced by paper/data/prepare_stop_slots.py) carry a per-record
``hotwords`` list -- the entity strings extracted from that utterance's slot
annotations. The context-biasing candidate universe for the "full list" variant
is simply the set of those entities that occur in the splits we are willing to
draw candidates from.

Given the VALIDATION + TEST manifests, this emits the sorted unique set of their
hotwords, one term per line. That replaces the old two-step "all-splits oracle
list -> intersect with eval+test" dance: the list is built directly from val+test,
so no separate presence-intersection stage is needed. The downstream zero-recall
+ min-count filter (on TRAIN) is applied separately by the pipeline.

Falls back to no entities gracefully if a manifest lacks ``hotwords`` (prints a
warning), so a mis-pointed manifest fails loudly rather than silently boosting
nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", "-m", required=True, type=Path, nargs="+",
                    help="One or more manifests (JSONL) with a per-record "
                         "'hotwords' list (e.g. eval.json test.json).")
    ap.add_argument("--out", "-o", required=True, type=Path,
                    help="Output: one attested entity term per line, sorted unique.")
    ap.add_argument("--min-count", type=int, default=1,
                    help="Keep only entities occurring at least this many times "
                         "pooled across all given manifests' hotwords "
                         "(default: %(default)s).")
    ap.add_argument("--require-each", action="store_true",
                    help="Keep only entities that occur in EVERY given manifest "
                         "at least once (e.g. present in both val and test), not "
                         "just pooled. Guarantees each term is both selectable on "
                         "val and scored on test.")
    args = ap.parse_args()

    from collections import Counter
    pooled: Counter = Counter()          # term -> total occurrences (all manifests)
    per_manifest: list[Counter] = []     # one Counter per manifest
    n_records = 0
    n_with_hotwords = 0
    for m in args.manifest:
        if not m.is_file():
            sys.exit(f"[fatal] manifest not found: {m}")
        c: Counter = Counter()
        with m.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                n_records += 1
                rec = json.loads(line)
                hw = rec.get("hotwords")
                if not hw:
                    continue
                n_with_hotwords += 1
                for term in hw:
                    t = (term or "").strip()
                    if t:
                        c[t] += 1
                        pooled[t] += 1
        per_manifest.append(c)

    if not pooled:
        print(f"[warn] no 'hotwords' found across {len(args.manifest)} manifest(s); "
              f"the attested list will be empty. Check the manifests carry a "
              f"'hotwords' field.", file=sys.stderr)

    def keep(term: str) -> bool:
        if pooled[term] < args.min_count:
            return False
        if args.require_each and not all(c.get(term, 0) >= 1 for c in per_manifest):
            return False
        return True

    kept = sorted(t for t in pooled if keep(t))
    n_dropped = len(pooled) - len(kept)
    rule = (f"present in each of {len(args.manifest)} manifest(s)"
            if args.require_each else f"pooled occurrence >= {args.min_count}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    print(f"[attested] {n_records} records ({n_with_hotwords} with hotwords) across "
          f"{len(args.manifest)} manifest(s) -> {len(pooled)} unique terms, "
          f"kept {len(kept)} ({rule}; dropped {n_dropped}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
