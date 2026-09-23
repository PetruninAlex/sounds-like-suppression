#!/usr/bin/env python3
"""Rewrite boost-list terms into the surface form a punctuation+capitalization
(PC) ASR model actually emits, using an LLM.

Motivation. Canary and the PC FastConformer emit cased, punctuated text
("St. Louis", "Beyonce"), but the mined boost list stores the lowercased
reference form ("st louis", "beyonce"). Boosting the lowercase token path forces
the decoder off its natural cased path, which needs a large boost weight to win
and then over-triggers (precision collapse, esp. AED greedy). Boosting the form
the model would actually produce aligns the boost with the model's own tokens.

What it does. For each unique term in the input TSV (columns: word, count,
recall) it asks the LLM for the correct written form: proper nouns
capitalized (names, places, brands, acronyms), ordinary words left lowercase,
standard punctuation added where a PC model would ("St." not "st"). It changes
ONLY casing and punctuation -- a guard rejects any reply whose normalized form
(lowercase, no ,.?, collapsed spaces) differs from the input, so the term
identity we boost never changes; on rejection it keeps the original.

Accents/OOV are left to the downstream tokenizability filter: if the cased form
tokenizes to <unk> (e.g. "Beyonce" is fine but an accented form would not), that
filter drops it exactly as before.

Output. The same TSV with the `word` column replaced by the truecased form
(count/recall preserved), so the existing build_boost_lists step is unchanged.

LLM plumbing (Azure OpenAI client, diskcache, retry, concurrency) mirrors
extract_boost_phrases.py. Env vars, per --azure-suffix:
    AZURE_OPENAI_API_KEY_<suffix>, AZURE_OPENAI_ENDPOINT_<suffix>,
    OPENAI_API_VERSION_<suffix>, deploymentid_<suffix>
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
from openai import AzureOpenAI
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

CACHE_DIR = Path(__file__).resolve().with_suffix(".cache")
cache = Cache(str(CACHE_DIR))


def case_key(s: str) -> str:
    """Whitespace-collapsed, lowercased view. Two strings with the same
    case_key differ only in capitalization (and incidental whitespace) -- the
    truecaser is allowed to move only within one such class, so it never changes
    letters or punctuation, which keeps the term tokenizable by construction."""
    return " ".join((s or "").split()).lower()


# The few-shot examples are deliberately drawn from domains OUTSIDE our corpora
# (no STOP artist/place names, no MultiMed medical terms): tech, sports,
# history, general vocabulary. This shows the capitalization rules without
# leaking anything about the evaluation data.
SYSTEM_PROMPT = (
    "You recapitalize a short term into the surface form a modern speech "
    "recognizer with capitalization would output. Apply English orthography:\n"
    "  - Capitalize proper nouns: people, places, organizations, brands, works.\n"
    "  - Capitalize acronyms fully; use standard mixed case for known stylings.\n"
    "  - Leave common/generic words lowercase.\n"
    "CHANGE ONLY capitalization. Do NOT add or remove any character: no "
    "punctuation, no periods, no accents or non-ASCII characters, no spelling "
    "changes. Keep the exact same letters and punctuation in the same order --- "
    "only their upper/lower case may differ.\n"
    "Examples (not related to the input domain):\n"
    "  'barack obama' -> 'Barack Obama'\n"
    "  'san jose' -> 'San Jose'\n"
    "  'mount kilimanjaro' -> 'Mount Kilimanjaro'\n"
    "  'iphone' -> 'iPhone'\n"
    "  'playstation' -> 'PlayStation'\n"
    "  'nasa' -> 'NASA'\n"
    "  'the rolling stones' -> 'The Rolling Stones'\n"
    "  'mount fuji' -> 'Mount Fuji'\n"
    "  'running shoes' -> 'running shoes'\n"
    "  'coffee shop' -> 'coffee shop'\n"
    'Reply with one JSON object: {"form": "..."}.'
)
PROMPT_HASH = hashlib.sha1(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]

RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
MAX_RETRIES = 6


def get_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"missing env var: {name}")
    return v


def _create_with_retry(client: AzureOpenAI, **kwargs):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except RETRYABLE as e:
            if attempt == MAX_RETRIES:
                raise
            headers = getattr(getattr(e, "response", None), "headers", {}) or {}
            try:
                delay = float(headers.get("retry-after"))
            except (TypeError, ValueError):
                delay = min(2.0 * (2 ** attempt), 60.0)
            delay += random.uniform(0, delay * 0.25)
            time.sleep(delay)


def _parse_form(text: str) -> str:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return ""
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return ""
    return str(obj.get("form", "")).strip() if isinstance(obj, dict) else ""


def _ask(client: AzureOpenAI, deployment: str, term: str, prior_bad: str | None):
    """One LLM turn. When prior_bad is given, include it and a corrective
    instruction so the model can fix a reply that changed characters."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"TERM: {term}"},
    ]
    if prior_bad is not None:
        msgs.append({"role": "assistant", "content": json.dumps({"form": prior_bad})})
        msgs.append({"role": "user", "content": (
            f'That reply changed the characters. Recapitalize "{term}" using the '
            "EXACT same letters and punctuation in the same order --- only "
            "upper/lower case may differ. Do not add periods, accents, or any "
            "other character.")})
    resp = _create_with_retry(
        client, model=deployment, messages=msgs,
        response_format={"type": "json_object"},
    )
    return _parse_form(resp.choices[0].message.content)


def truecase_term(client: AzureOpenAI, deployment: str, term: str,
                  max_tries: int = 2) -> str:
    """Recapitalize term via the LLM, verifying case-only equality.

    A reply is accepted only if lowercasing it reproduces the original term
    (same letters and punctuation, case aside). If it does not --- the model
    added a period, an accent, or respelled something --- we re-ask once more
    with corrective feedback, and after ``max_tries`` failed attempts fall back
    to the original term. Case-only acceptance means the recased form is
    tokenizable whenever the original is, so no tokenizer check is needed.
    """
    prior = None
    for _ in range(max_tries):
        form = _ask(client, deployment, term, prior)
        if form and case_key(form) == case_key(term):
            return form
        prior = form or ""
    return term


def cache_key(model_id: str, term: str) -> str:
    return f"truecase:{PROMPT_HASH}:{model_id}:{term}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True, type=Path,
                    help="TSV with a `word` column (e.g. boost_phrases.correct_forms.tsv).")
    ap.add_argument("--output", "-o", required=True, type=Path,
                    help="Output TSV, same columns, `word` truecased.")
    ap.add_argument("--azure-suffix", default="5", help="Env-var suffix for the Azure deployment.")
    ap.add_argument("--max-tries", type=int, default=2,
                    help="LLM attempts per term: first ask plus corrective "
                         "retries when the reply changes characters (default 2).")
    ap.add_argument("--max-concurrent", type=int, default=8)
    args = ap.parse_args()

    with args.input.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fields = reader.fieldnames or []
        rows = list(reader)
    if "word" not in fields:
        sys.exit(f"[error] no `word` column in {args.input}")

    sx = args.azure_suffix
    client = AzureOpenAI(
        api_key=get_env(f"AZURE_OPENAI_API_KEY_{sx}"),
        azure_endpoint=get_env(f"AZURE_OPENAI_ENDPOINT_{sx}"),
        api_version=get_env(f"OPENAI_API_VERSION_{sx}"),
    )
    model_id = get_env(f"deploymentid_{sx}")

    terms = list(dict.fromkeys(r["word"] for r in rows if r.get("word")))
    todo = [t for t in terms if not isinstance(cache.get(cache_key(model_id, t)), str)]
    print(f"[truecase] model={model_id}: {len(terms)} terms, {len(todo)} new, "
          f"{len(terms) - len(todo)} cached (cache={CACHE_DIR})", file=sys.stderr)

    def work(t: str) -> str:
        return truecase_term(client, model_id, t, args.max_tries)

    with cf.ThreadPoolExecutor(max_workers=args.max_concurrent) as ex:
        futs = {ex.submit(work, t): t for t in todo}
        for fut in cf.as_completed(futs):
            t = futs[fut]
            try:
                cache.set(cache_key(model_id, t), fut.result())
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {t!r}: {e}; keeping original", file=sys.stderr)
                cache.set(cache_key(model_id, t), t)

    # truecase_term already applied the case-only guard, so the cached value is
    # the final form (recased, or the original on failure).
    mapping = {t: cache.get(cache_key(model_id, t), t) for t in terms}
    changed = sum(1 for t in terms if mapping[t] != t)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for r in rows:
            r = dict(r)
            r["word"] = mapping.get(r["word"], r["word"])
            writer.writerow(r)
    print(f"[truecase] {changed}/{len(terms)} terms recased -> {args.output}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
