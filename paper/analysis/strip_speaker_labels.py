#!/usr/bin/env python3
"""Strip diarization / speaker-label spans from NeMo manifest reference text.

MultiMed transcripts carry speaker turn labels inside the reference ``text``
field -- e.g. ``William Collins, M.D. : ...``, ``Dr. Justin Marchegiani: ...``,
``Evan Brand: ...``, ``Male Speaker: ...``, ``Marsha: ...``. These labels are
metadata, not spoken audio, so the ASR model never emits them. Left in the
reference they become guaranteed false negatives: they wreck WER and per-word
F1 (``collins`` / ``william collins`` alone account for ~39% of the still-failing
recall mass on the test set) and, because the phrase list is mined from these
references, they inject un-boostable speaker names (collins, andrew, chris,
evan, mack, sarah, ...) into the boost set.

This script removes only the ``Name:`` label span; ordinary in-sentence mentions
of the same name (e.g. "thanks Jarred") are left untouched. It rewrites the
``text`` field of each manifest line and copies every other field verbatim.

A candidate label is an optional title (Dr./Mr./Mrs./Ms./Prof.), 1-5 capitalized
name/role tokens (ALL-CAPS and internal hyphens/apostrophes allowed, but NOT a
trailing or double hyphen -- so ``East Coast-- CHRIS PALMER:`` only drops
``CHRIS PALMER:``), an optional ``, M.D.`` suffix, then a colon. Name tokens may
not span a period, so ``... University. Marsha:`` only drops ``Marsha:``.

To avoid deleting genuine spoken words that happen to precede a colon
(``Erlotinib:``, ``Surgery:``, ``TB:``), a candidate is removed only if it is
clearly a speaker label:
  * it has a title (``Dr. Teixido:``) or a ``, M.D.`` suffix, or
  * it has >= 2 name tokens (``Andrew Huberman:``, ``Male Speaker:``), or
  * it is a single token that is a known role word, or a name harvested from the
    multi-token / titled labels elsewhere in the corpus (so bare ``Chris:`` is
    removed because ``Chris Palmer:`` exists, but ``Erlotinib:`` is kept).

Usage (batch, one clean copy per input, same basename):
    python strip_speaker_labels.py --input a.json b.json --out-dir DIR
Inspect first without writing:
    python strip_speaker_labels.py --input a.json --out-dir DIR --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

# A name/role token: capital letter, then letters/’'/ with only *internal* single
# hyphens (Shu-Fen, Nolen-Hoeksema). No trailing hyphen -> "Coast--" won't match.
TOKEN = r"[A-Z][A-Za-z\u2019']*(?:-[A-Za-z\u2019']+)*"
CAND_RE = re.compile(
    r"(?P<title>(?:Dr|Mr|Mrs|Ms|Prof)\.\s+)?"
    rf"(?P<body>(?:{TOKEN}\s+){{0,4}}{TOKEN})"
    r"(?P<md>,\s*M\.D\.)?"
    r"\s*:"
)

# Single-token labels are only stripped if the token is one of these roles or a
# harvested speaker name (see module docstring).
ROLE_WORDS = {
    "speaker", "narrator", "reporter", "audience", "announcer", "host",
    "student", "man", "woman", "moderator", "interviewer", "panelist",
    "producer", "newsreader", "note", "intro", "outro", "voice-over",
}


def _tokens(body: str) -> list[str]:
    return [t.strip(",") for t in body.split()]


def harvest_names(texts) -> set[str]:
    """Collect speaker-name tokens from title/multi-token labels across the corpus."""
    names: set[str] = set()
    for text in texts:
        for m in CAND_RE.finditer(text):
            toks = _tokens(m.group("body"))
            titled = bool(m.group("title") or m.group("md"))
            if titled or len(toks) >= 2:
                for t in toks:
                    tl = t.lower()
                    if len(tl) > 1 and tl not in ROLE_WORDS:
                        names.add(tl)
    return names


def make_stripper(known_names: set[str]):
    def should_remove(m: re.Match) -> bool:
        toks = _tokens(m.group("body"))
        if m.group("title") or m.group("md") or len(toks) >= 2:
            return True
        return toks[0].lower() in ROLE_WORDS or toks[0].lower() in known_names

    def strip(text: str) -> tuple[str, list[str]]:
        removed: list[str] = []

        def _sub(m: re.Match) -> str:
            if should_remove(m):
                removed.append(m.group(0))
                return " "
            return m.group(0)

        cleaned = CAND_RE.sub(_sub, text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned, removed

    return strip


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True,
                    help="Source manifest(s) (jsonl with a 'text' field).")
    ap.add_argument("--out-dir",
                    help="Directory for cleaned copies (same basenames). "
                         "Required unless --dry-run.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report removals without writing any output.")
    ap.add_argument("--show", type=int, default=40,
                    help="How many distinct removed spans to print (default 40).")
    args = ap.parse_args()

    if not args.dry_run and not args.out_dir:
        ap.error("--out-dir is required unless --dry-run")

    # Load every manifest once (keep objects so we rewrite in a second pass).
    manifests: list[tuple[str, list[dict]]] = []
    for path in args.input:
        objs = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    objs.append(json.loads(line))
        manifests.append((path, objs))

    # Pass 1: harvest speaker names corpus-wide so single-token labels resolve.
    known = harvest_names(o.get("text", "") for _, objs in manifests for o in objs)
    strip = make_stripper(known)

    total: Counter = Counter()
    for path, objs in manifests:
        spans: Counter = Counter()
        n_mod = 0
        for o in objs:
            cleaned, removed = strip(o.get("text", ""))
            if removed:
                n_mod += 1
                for r in removed:
                    spans[re.sub(r"\s*:$", "", r).strip()] += 1
                o["text"] = cleaned
        total += spans
        print(f"[{os.path.basename(path)}] {len(objs)} utts, {n_mod} modified, "
              f"{sum(spans.values())} spans removed ({len(spans)} distinct)",
              file=sys.stderr)
        if not args.dry_run:
            out_path = os.path.join(args.out_dir, os.path.basename(path))
            os.makedirs(args.out_dir, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                for o in objs:
                    fh.write(json.dumps(o, ensure_ascii=False) + "\n")
            print(f"    -> {out_path}", file=sys.stderr)

    print(f"\n[all] {sum(total.values())} spans, {len(total)} distinct. "
          f"Top {args.show}:", file=sys.stderr)
    for span, cnt in total.most_common(args.show):
        print(f"  {cnt:4d}  {span}", file=sys.stderr)


if __name__ == "__main__":
    main()
