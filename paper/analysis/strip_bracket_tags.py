#!/usr/bin/env python3
"""Strip non-speech ``[...]`` spans from NeMo manifest reference text.

Caption-style tags such as ``[MUSIC PLAYING]``, ``[INAUDIBLE]``, ``[APPLAUSE]``,
``[LAUGHTER]``, and bracketed speaker IDs (``[ANNA]``, ``[AUDIENCE MEMBER]``)
are not spoken. They are not the utterance: only the bracketed span is removed.
The rest of the reference is kept.

Rewrites the ``text`` field in place (or to ``--out-dir``) and copies every
other field verbatim.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

TAG_RE = re.compile(r"\[[^\[\]]*\]")


def strip_tags(text: str) -> tuple[str, list[str]]:
    removed = TAG_RE.findall(text)
    if not removed:
        return text, []
    cleaned = TAG_RE.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned).strip()
    return cleaned, removed


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True,
                    help="Manifest(s) (jsonl with a 'text' field).")
    ap.add_argument("--out-dir",
                    help="Write cleaned copies here (same basenames). "
                         "Default: overwrite --input in place.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report removals without writing.")
    ap.add_argument("--show", type=int, default=40,
                    help="How many distinct removed tags to print (default 40).")
    args = ap.parse_args()

    total: Counter = Counter()
    for path in args.input:
        objs = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    objs.append(json.loads(line))
        spans: Counter = Counter()
        n_mod = 0
        for o in objs:
            cleaned, removed = strip_tags(o.get("text") or "")
            if removed:
                n_mod += 1
                for r in removed:
                    spans[re.sub(r"\s+", " ", r).strip()] += 1
                o["text"] = cleaned
        total += spans
        print(f"[{os.path.basename(path)}] {len(objs)} utts, {n_mod} modified, "
              f"{sum(spans.values())} tags removed ({len(spans)} distinct)",
              file=sys.stderr)
        if args.dry_run:
            continue
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            out_path = os.path.join(args.out_dir, os.path.basename(path))
        else:
            out_path = path
        with open(out_path, "w", encoding="utf-8") as fh:
            for o in objs:
                fh.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(f"    -> {out_path}", file=sys.stderr)

    print(f"\n[all] {sum(total.values())} tags, {len(total)} distinct. "
          f"Top {args.show}:", file=sys.stderr)
    for span, cnt in total.most_common(args.show):
        print(f"  {cnt:4d}  {span}", file=sys.stderr)


if __name__ == "__main__":
    main()
