#!/usr/bin/env python3
"""Drop misspelled / mis-segmented / malformed entries from a candidate entity list.

STOP's slot annotations contain orthographic noise: real names spelled wrongly,
mis-segmentations (a word wrongly split or joined), and spelled-out number/road
fragments. Because the ASR emits the *correct* orthography, these look
zero-recall on train, get boosted, and then over-trigger on test -- flipping the
model's correct spelling into the malformed one and destroying precision.

This step asks an LLM to judge each candidate entity in isolation: is it a
well-formed, correctly-spelled instance of the expected entity type, or a
misspelling / mis-segmentation / garbled fragment? Only well-formed entities are
kept. It is applied identically to every corpus so it is a blind normalization
pass, not a hand-removal of terms that happened to hurt.

Input is the zero-recall TSV (columns: word count recall). Output is the same
TSV filtered to kept rows; a sidecar ``*.decisions.tsv`` logs every entity with
keep/drop and the reason.

Auth (env vars; suffix via ``--azure-suffix``, default ``gpt5_5``):
    AZURE_OPENAI_API_KEY_<suffix>
    AZURE_OPENAI_ENDPOINT_<suffix>
    OPENAI_API_VERSION_<suffix>
    deploymentid_<suffix>
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

from diskcache import Cache
from openai import AzureOpenAI, RateLimitError

CACHE_DIR = Path(__file__).resolve().with_suffix(".cache")
cache = Cache(str(CACHE_DIR))

SYSTEM_PROMPT = (
    "You are cleaning a contextual-biasing entity list for an English speech "
    "recognizer. You are given ONE candidate entity string and the TYPE of "
    "entity the list is supposed to contain (e.g. place names, person names, "
    "media/artist or event names).\n"
    "Decide whether the candidate is a WELL-FORMED, CORRECTLY-SPELLED instance "
    "of that type as it would be written in standard English, or whether it is "
    "MALFORMED -- specifically any of:\n"
    "  - a misspelling of a real entity (a proper name with one or more letters "
    "wrong or missing from its standard spelling);\n"
    "  - a mis-segmentation, i.e. one word wrongly split into two or two wrongly "
    "joined into one, relative to the standard written form;\n"
    "  - a spelled-out number or road/route fragment written as words rather "
    "than as the standard name;\n"
    "  - a garbled fragment or something that is not a real entity of the type.\n"
    "Be CONSERVATIVE: keep genuine but rare or foreign proper names -- rarity is "
    "not a reason to drop. Only drop a candidate when you are confident it is "
    "misspelled, mis-segmented, a spelled-out number, or not a real entity. When "
    "unsure, KEEP it.\n"
    "Judge spelling/segmentation only; ignore capitalization and possessive 's.\n"
    'Reply with one JSON object: {"keep": true|false, "reason": "<short '
    'reason>"}.'
)


def get_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"missing env var: {name}")
    return v


_PROMPT_HASH = hashlib.sha1(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]


def cache_key(entity: str, entity_type: str) -> str:
    return f"{_PROMPT_HASH}\x00{entity_type}\x00{entity.lower().strip()}"


def classify(client: AzureOpenAI, deployment: str, entity: str,
             entity_type: str, max_retries: int = 6) -> dict:
    attempt = 0
    other_errors = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"TYPE: {entity_type}\nCANDIDATE: {entity}"},
                ],
                response_format={"type": "json_object"},
            )
            obj = json.loads(resp.choices[0].message.content or "{}")
            keep = obj.get("keep", True)
            if not isinstance(keep, bool):
                keep = str(keep).strip().lower() in ("true", "1", "yes")
            reason = obj.get("reason", "")
            if not isinstance(reason, str):
                reason = ""
            return {"keep": keep, "reason": reason.strip()}
        except RateLimitError:
            time.sleep(min(2 ** (attempt + 1), 60) + random.random())
            attempt += 1
        except Exception:
            other_errors += 1
            if other_errors > max_retries:
                raise
            time.sleep(min(2 ** other_errors, 60) + random.random())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True, type=Path,
                    help="Zero-recall TSV (columns: word count recall).")
    ap.add_argument("--output", "-o", required=True, type=Path,
                    help="Filtered TSV (same columns, malformed entities dropped).")
    ap.add_argument("--entity-type", required=True,
                    help="Description of the entity type the list should hold "
                         "(e.g. 'US and world place / location names').")
    ap.add_argument("--decisions", type=Path, default=None,
                    help="Optional sidecar TSV logging every entity with "
                         "keep/drop + reason (default: <output>.decisions.tsv).")
    ap.add_argument("--azure-suffix", default="gpt5_5",
                    help="Suffix on the AZURE_OPENAI_*_<suffix> env vars "
                         "(default: %(default)s).")
    ap.add_argument("--max-concurrent", type=int, default=4,
                    help="Concurrent requests per process (default: %(default)s).")
    args = ap.parse_args()

    sx = args.azure_suffix
    client = AzureOpenAI(
        api_key=get_env(f"AZURE_OPENAI_API_KEY_{sx}"),
        azure_endpoint=get_env(f"AZURE_OPENAI_ENDPOINT_{sx}"),
        api_version=get_env(f"OPENAI_API_VERSION_{sx}"),
    )
    deployment = get_env(f"deploymentid_{sx}")

    rows = list(csv.DictReader(args.input.open(), delimiter="\t"))
    fieldnames = list(rows[0].keys()) if rows else ["word", "count", "recall"]

    def needs_call(word: str) -> bool:
        v = cache.get(cache_key(word, args.entity_type))
        return not (isinstance(v, dict) and "keep" in v)

    todo = [r["word"] for r in rows if needs_call(r["word"])]
    print(f"[info] {len(rows)} candidate entities, {len(todo)} new, "
          f"{len(rows) - len(todo)} cached (type={args.entity_type!r}, "
          f"deployment={deployment}, cache={CACHE_DIR})", file=sys.stderr)

    errors = 0
    with cf.ThreadPoolExecutor(max_workers=args.max_concurrent) as ex:
        futs = {ex.submit(classify, client, deployment, w, args.entity_type): w
                for w in todo}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            w = futs[fut]
            try:
                cache[cache_key(w, args.entity_type)] = fut.result()
            except Exception as e:
                errors += 1
                print(f"[warn] {w!r}: {type(e).__name__}: {e}", file=sys.stderr)
            if i % 50 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} done ({errors} failed)", file=sys.stderr)
    if errors:
        # A persistent failure means we have no verdict for that entity. Keep it
        # (conservative: don't drop an entity we couldn't check) and note it.
        print(f"[warn] {errors}/{len(todo)} calls still failed after retries; "
              f"those entities are KEPT (unchecked).", file=sys.stderr)

    kept, dropped, decisions = [], 0, []
    for r in rows:
        v = cache.get(cache_key(r["word"], args.entity_type))
        if isinstance(v, dict) and "keep" in v:
            keep, reason = bool(v["keep"]), v.get("reason", "")
        else:
            keep, reason = True, "unchecked (LLM call failed)"
        decisions.append((r["word"], "keep" if keep else "drop", reason))
        if keep:
            kept.append(r)
        else:
            dropped += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)

    dec_path = args.decisions or args.output.with_suffix(".decisions.tsv")
    with dec_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["entity", "decision", "reason"])
        w.writerows(decisions)

    print(f"[done] kept {len(kept)}/{len(rows)} entities, dropped {dropped} "
          f"(malformed) -> {args.output}; decisions -> {dec_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
