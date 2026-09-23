#!/usr/bin/env python3
"""Find sounds-alike confusions from full (REF, HYP) sentence pairs via an LLM.

Reads the output of find_boost_word_confusions.py (columns: audio_filepath word
ref_text hyp_text pred_token_ids). For each row the LLM is shown the TARGET
boost word, the FULL reference sentence, and the FULL ASR hypothesis sentence,
and is asked to find the word/phrase the ASR produced *in place of* the target
word -- i.e. its sounds-alike confusion -- using both sentences as context.

The output keeps only the *classified* rows (those for which the LLM found a
plausible sounds-alike confusion) and appends the discovered confusion as a
`sounds_like` column. All provenance columns from the input ride along, so the
output is the input TSV filtered to classified rows plus one extra column:

    audio_filepath word ref_text hyp_text pred_token_ids sounds_like

ready for the downstream sounds-like boosting steps.

Decisions are cached in a `diskcache` store at `CACHE_DIR` so resumed runs skip
already-decided sentence pairs; each successful call is persisted immediately.

Auth (env vars; suffix configurable via `--azure-suffix`, default `gpt5_5`):
    AZURE_OPENAI_API_KEY_<suffix>
    AZURE_OPENAI_ENDPOINT_<suffix>
    OPENAI_API_VERSION_<suffix>
    deploymentid_<suffix>           # used as the chat-completions `model`
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from diskcache import Cache
from openai import AzureOpenAI, RateLimitError

CACHE_DIR = Path(__file__).resolve().with_suffix(".cache")
cache = Cache(str(CACHE_DIR))

SYSTEM_PROMPT = (
    "You are given a TARGET word from an English speech corpus, the reference "
    "transcript (REF) of an utterance, and an ASR system's hypothesis "
    "transcript (HYP) for the same audio. The TARGET word appears in REF but "
    "the ASR did not transcribe it correctly. Your job is to find the word or "
    "short phrase in HYP that the ASR produced *in place of* the TARGET word "
    "-- its sounds-alike confusion.\n"
    "Use BOTH full sentences to align them and locate what was produced where "
    "the TARGET word should be.\n"
    "A valid sounds-alike confusion is phonetically close to the TARGET at the "
    "SOUND level. Ignore differences in possessive 's, plural -s, apostrophes, "
    "capitalization, and minor inflectional endings. The confusion may be a "
    "non-word, fragment, or misspelling -- that is fine; judge only whether it "
    "is a believable acoustic rendering of the TARGET word.\n"
    "Do NOT return common, everyday English words (high-frequency function words "
    "like 'the', 'and', 'were', or very common content words). Suppressing "
    "such words to boost the TARGET would hurt the ASR elsewhere, since they occur "
    "legitimately all over the corpus. Only return a confusion that is itself "
    "uncommon/rare -- a non-word, fragment, misspelling, or otherwise unusual token. "
    "If the only candidate produced in place of the TARGET is a common word, return "
    "an empty string instead.\n"
    "SPAN EXCEPTION: sometimes the ASR mis-segments the TARGET, splitting an "
    "unstressed leading or trailing syllable into its own tiny neighbouring word "
    "(often a function word). When that happens, return the FULL contiguous HYP span "
    "aligned to the TARGET, INCLUDING that adjacent short word, as long as the whole "
    "span still reads as one acoustic rendering of the TARGET. This overrides the "
    "no-common-words rule, but only for the function word GLUED to the confusion -- "
    "never return a bare common word on its own.\n"
    "Return the confusion EXACTLY as it appears in HYP, preserving its "
    "original capitalization and spelling -- do not lowercase or otherwise "
    "alter it. If there is no plausible sounds-alike confusion for the TARGET "
    "in HYP (e.g. the word was simply deleted, or what appears is not "
    "phonetically close), return an empty string.\n"
    'Reply with one JSON object: {"sounds_like": "<confusion word/phrase from '
    'HYP, or empty string>"}.'
)


def get_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"missing env var: {name}")
    return v


_PROMPT_HASH = hashlib.sha1(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]


def surface_form(word: str, ref_text: str) -> str:
    """Return the boost word as it actually appears (casing/spelling) in
    `ref_text`, matched case-insensitively as a whole word. Falls back to the
    given `word` if no match is found."""
    m = re.search(rf"(?<!\w){re.escape(word)}(?!\w)", ref_text, flags=re.IGNORECASE)
    return m.group(0) if m else word


def cache_key(word: str, ref_text: str, hyp_text: str) -> str:
    # Include a hash of the prompt so edits to SYSTEM_PROMPT invalidate stale
    # cached answers automatically.
    return f"{_PROMPT_HASH}\x00{word}\x00{ref_text}\x00{hyp_text}"


def find_sounds_like(client: AzureOpenAI, deployment: str,
                     term: str, ref_text: str, hyp_text: str,
                     max_retries: int = 6) -> dict:
    # Rate-limit errors (HTTP 429) are common when several (dataset, setting)
    # combos hit the same deployment at once; they are transient throttling, so we
    # keep retrying them indefinitely with capped backoff rather than dropping the
    # row. Other errors retry only up to `max_retries` times before propagating,
    # so a genuine failure can't loop forever. Backoff is exponential + jitter,
    # capped at 60s.
    attempt = 0
    other_errors = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"TARGET: {term}\nREF: {ref_text}\nHYP: {hyp_text}"},
                ],
                response_format={"type": "json_object"},
            )
            obj = json.loads(resp.choices[0].message.content or "{}")
            sl = obj.get("sounds_like", "")
            if not isinstance(sl, str):
                sl = ""
            return {"sounds_like": sl.strip()}
        except RateLimitError:
            # Don't count 429s against the cap -- wait out the throttle.
            time.sleep(min(2 ** (attempt + 1), 60) + random.random())
            attempt += 1
        except Exception:
            other_errors += 1
            if other_errors > max_retries:
                raise
            time.sleep(min(2 ** other_errors, 60) + random.random())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True, type=Path,
                    help="Sentence pairs TSV (columns: audio_filepath word ref_text hyp_text).")
    ap.add_argument("--output", "-o", required=True, type=Path,
                    help="Output TSV: the input rows the LLM classified, plus a "
                         "`sounds_like` column (columns: audio_filepath word "
                         "ref_text hyp_text pred_token_ids sounds_like).")
    ap.add_argument("--azure-suffix", default="gpt5_5",
                    help="Suffix on the AZURE_OPENAI_*_<suffix> / "
                         "deploymentid_<suffix> env vars (default: %(default)s).")
    ap.add_argument("--max-concurrent", type=int, default=4,
                    help="Concurrent requests PER PROCESS (default: %(default)s). "
                         "run_all.sh runs several (dataset, setting) combos at "
                         "once, all hitting the same LLM deployment, so the real "
                         "load is this value times the number of concurrent "
                         "combos -- keep it low to avoid 429 rate-limit errors.")
    args = ap.parse_args()

    sx = args.azure_suffix
    client = AzureOpenAI(
        api_key=get_env(f"AZURE_OPENAI_API_KEY_{sx}"),
        azure_endpoint=get_env(f"AZURE_OPENAI_ENDPOINT_{sx}"),
        api_version=get_env(f"OPENAI_API_VERSION_{sx}"),
    )
    deployment = get_env(f"deploymentid_{sx}")

    rows = list(csv.DictReader(args.input.open(), delimiter="\t"))

    # Present the TARGET to the LLM exactly as it appears in its ref sentence
    # (casing/spelling), rather than the canonical boost-list form.
    for r in rows:
        r["word"] = surface_form(r["word"], r["ref_text"])

    def needs_call(r: dict) -> bool:
        v = cache.get(cache_key(r["word"], r["ref_text"], r["hyp_text"]))
        return not (isinstance(v, dict) and "sounds_like" in v)

    todo = [r for r in rows if needs_call(r)]
    print(f"[info] {len(rows)} sentence pairs, {len(todo)} new, "
          f"{len(rows) - len(todo)} cached "
          f"(deployment={deployment}, cache={CACHE_DIR})", file=sys.stderr)

    errors = 0
    with cf.ThreadPoolExecutor(max_workers=args.max_concurrent) as ex:
        futs = {
            ex.submit(find_sounds_like, client, deployment,
                      r["word"], r["ref_text"], r["hyp_text"]):
            (r["word"], r["ref_text"], r["hyp_text"])
            for r in todo
        }
        for i, fut in enumerate(cf.as_completed(futs), 1):
            word, ref_text, hyp_text = futs[fut]
            try:
                cache[cache_key(word, ref_text, hyp_text)] = fut.result()
            except Exception as e:
                errors += 1
                print(f"[warn] {word!r}: {type(e).__name__}: {e}",
                      file=sys.stderr)
            if i % 50 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} done ({errors} failed)",
                      file=sys.stderr)
    if errors:
        # Don't abort: each call already retried up to its backoff limit, so the
        # remaining failures are persistent. Write what we have (successful +
        # cached rows) so downstream steps can proceed; the failed rows are simply
        # absent from the output and will be retried on the next run (they are not
        # cached). This keeps one throttled combo from taking the whole run down.
        print(f"[warn] {errors}/{len(todo)} calls still failed after retries; "
              f"writing partial output (failed rows omitted, not cached -- "
              f"re-run to retry them).", file=sys.stderr)

    # Keep only the rows the LLM classified as having a sounds-like confusion
    # (non-empty, and not identical to the target word), appending the discovered
    # confusion as a `sounds_like` column. Every provenance column from the input
    # rides along, so the output is the input TSV filtered to classified rows.
    fieldnames = (list(rows[0].keys()) if rows
                  else ["audio_filepath", "word", "ref_text", "hyp_text",
                        "pred_token_ids"])
    if "sounds_like" not in fieldnames:
        fieldnames.append("sounds_like")

    out_rows: list[dict] = []
    for r in rows:
        v = cache.get(cache_key(r["word"], r["ref_text"], r["hyp_text"])) or {}
        sl = (v.get("sounds_like") or "").strip()
        if sl and sl.lower() != r["word"].strip().lower():
            out_rows.append({**r, "sounds_like": sl})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    print(f"[done] {len(out_rows)}/{len(rows)} sentences yielded a sounds-like "
          f"confusion -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
