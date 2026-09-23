#!/usr/bin/env python3
"""Extract candidate boosting phrases from reference transcripts.

This replaces the alignment/zero-recall discovery (align_predictions.py +
find_zero_recall_words.py + llm_classify_words.py) with the phrase-selection
approach from the turbo-bias paper: candidate phrases are drawn from the
*reference* sentences of a (WER-filtered) manifest, independently of the ASR
errors, via two taggers:

  1) an Azure OpenAI chat model prompted to tag MEDICAL terms -- diseases,
     procedures, medications, and clinical conditions; and
  2) the spaCy NER pipeline for NAMED ENTITIES -- personal names (PERSON),
     brand / product names (PRODUCT), and organizations (ORG).

Every tagged span is normalized the same way the alignment pipeline normalizes
reference text (lowercase, strip commas/periods/question-marks, collapse
whitespace) and then VERIFIED to occur verbatim as a contiguous word window in
at least one reference sentence -- this drops any paraphrased / hallucinated
span and guarantees the downstream find_correct_word_forms.py can recover the
original surface spelling. Each kept phrase's `count` is its total number of
contiguous-window occurrences across all reference sentences (so a later step
can drop low-frequency phrases, e.g. < 10 occurrences, as in the paper).

Output is a TSV with columns:
    word    count   source
where `word` is the normalized phrase (consumed by find_correct_word_forms.py /
build_boost_lists.py), `count` its corpus occurrence count, and `source` a
comma-joined set of the taggers that proposed it (`llm_medical`,
`spacy_PERSON`, `spacy_ORG`, `spacy_PRODUCT`). Rows are sorted by count (desc)
then alphabetically.

LLM decisions are cached in a `diskcache` store at CACHE_DIR keyed on the prompt
hash + sentence, so resumed runs skip already-tagged sentences.

Backends (medical tagging), selected via `--backend`:
  * bedrock (default): AWS Bedrock, default model Claude Opus 4.8
    (us.anthropic.claude-opus-4-8). Uses standard AWS credential env vars
    (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN) and
    --aws-region (or $AWS_DEFAULT_REGION, else us-east-1).
  * azure: Azure OpenAI; auth via env vars (suffix from `--azure-suffix`):
        AZURE_OPENAI_API_KEY_<suffix>
        AZURE_OPENAI_ENDPOINT_<suffix>
        OPENAI_API_VERSION_<suffix>
        deploymentid_<suffix>       # used as the chat-completions `model`
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
from collections import Counter, defaultdict
from pathlib import Path

from diskcache import Cache
from openai import AzureOpenAI
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from botocore.exceptions import ClientError

CACHE_DIR = Path(__file__).resolve().with_suffix(".cache")
cache = Cache(str(CACHE_DIR))

# Mirror align_predictions.normalize_text: strip only commas, periods, and
# question marks; keep apostrophes, hyphens, etc.; lowercase; collapse spaces.
_PUNCT_RE = re.compile(f"[{re.escape(',.?')}]")

# spaCy entity labels for the paper's named-entity categories: personal names,
# brand / product names, and organizations.
DEFAULT_SPACY_LABELS = ("PERSON", "ORG", "PRODUCT")

SYSTEM_PROMPT = (
    "You tag MEDICAL terms in a single English sentence taken from a medical "
    "speech corpus. Extract every span that is a medical term, specifically:\n"
    "  - diseases and disorders,\n"
    "  - medical / surgical procedures,\n"
    "  - medications and drugs,\n"
    "  - clinical conditions, findings, and symptoms.\n"
    "Return each term EXACTLY as it appears in the sentence (same words, same "
    "order, verbatim substring -- do not correct, expand, lemmatize, or "
    "translate it). Do NOT include generic, non-medical words, whole clauses, "
    "or the surrounding articles/determiners. If the sentence contains no "
    "medical term, return an empty list.\n"
    'Reply with one JSON object: {"terms": ["...", "..."]}.'
)

PROMPT_HASH = hashlib.sha1(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]

RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
# Transient Bedrock error codes worth retrying with backoff.
BEDROCK_RETRYABLE = {"ThrottlingException", "TooManyRequestsException",
                     "ServiceUnavailableException", "InternalServerException",
                     "ModelTimeoutException", "ModelNotReadyException"}
MAX_RETRIES = 6
# Default Bedrock inference profile for Claude Opus 4.8 (on-demand throughput
# requires the `us.`-prefixed cross-region inference profile, not the bare id).
DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-opus-4-8"


def get_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"missing env var: {name}")
    return v


def normalize_text(s: str) -> str:
    """Lowercase, strip ,.?, collapse whitespace -- as align_predictions does."""
    s = s.lower()
    s = _PUNCT_RE.sub("", s)
    return " ".join(s.split())


def read_references(manifests: list[Path]) -> list[str]:
    """Return the `text` (reference) field of every non-empty JSONL line across
    all given manifests (so occurrences can be counted over train + dev)."""
    refs = []
    for manifest in manifests:
        with manifest.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = (json.loads(line).get("text", "") or "").strip()
                if t:
                    refs.append(t)
    return refs


def _create_with_retry(client: AzureOpenAI, **kwargs):
    """chat.completions.create with exponential backoff + jitter on transient
    errors, honouring a server Retry-After header when present."""
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


def _json_terms(text: str) -> list[str]:
    """Parse a model reply into a list of term strings, tolerant of stray prose
    around the JSON object."""
    text = (text or "").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    terms = obj.get("terms", []) if isinstance(obj, dict) else []
    if not isinstance(terms, list):
        return []
    return [str(t) for t in terms if isinstance(t, (str, int, float)) and str(t).strip()]


def tag_medical_azure(client: AzureOpenAI, deployment: str, sentence: str) -> list[str]:
    """Ask an Azure OpenAI deployment for the medical-term spans in one sentence."""
    resp = _create_with_retry(
        client,
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"SENTENCE: {sentence}"},
        ],
        response_format={"type": "json_object"},
    )
    return _json_terms(resp.choices[0].message.content)


def _bedrock_text(resp: dict) -> str:
    """Join every text block of a Bedrock converse response (tolerating an
    empty content list or non-text blocks such as reasoning)."""
    blocks = (resp.get("output", {}).get("message", {}).get("content") or [])
    return "".join(b["text"] for b in blocks
                   if isinstance(b, dict) and isinstance(b.get("text"), str))


def tag_medical_bedrock(client, model_id: str, sentence: str) -> list[str]:
    """Ask an AWS Bedrock model (e.g. Claude Opus 4.8) for the medical-term
    spans in one sentence, retrying transient throttling/5xx AND empty-body
    responses (converse can return HTTP 200 with an empty content list under
    load, which would otherwise IndexError) with backoff."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.converse(
                modelId=model_id,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[{"role": "user",
                           "content": [{"text": f"SENTENCE: {sentence}"}]}],
                inferenceConfig={"maxTokens": 1024},
            )
            text = _bedrock_text(resp)
            if text.strip():
                return _json_terms(text)
            # Bedrock/Anthropic safety filter blocked this sentence: deterministic,
            # so don't retry -- treat as "no medical terms" (spaCy NER still runs).
            if resp.get("stopReason") in ("content_filtered", "guardrail_intervened"):
                return []
            # Otherwise empty/odd response (no text block): transient -> retry.
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"empty Bedrock response (stopReason="
                    f"{resp.get('stopReason')!r}) after {MAX_RETRIES} retries")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if attempt == MAX_RETRIES or code not in BEDROCK_RETRYABLE:
                raise
        delay = min(2.0 * (2 ** attempt), 60.0)
        time.sleep(delay + random.uniform(0, delay * 0.25))


def cache_key(model_id: str, sentence: str) -> str:
    """Cache key includes the prompt hash and model id so switching model (or
    editing the prompt) re-tags instead of reusing a stale decision."""
    return PROMPT_HASH + "\x00" + model_id + "\x00" + sentence


def run_llm(sentences: list[str], args) -> dict[str, list[str]]:
    """Tag every unique sentence with the medical LLM (cached). Dispatches to the
    Azure OpenAI or AWS Bedrock backend. Returns {sentence: [terms]}."""
    if args.backend == "azure":
        sx = args.azure_suffix
        client = AzureOpenAI(
            api_key=get_env(f"AZURE_OPENAI_API_KEY_{sx}"),
            azure_endpoint=get_env(f"AZURE_OPENAI_ENDPOINT_{sx}"),
            api_version=get_env(f"OPENAI_API_VERSION_{sx}"),
        )
        model_id = get_env(f"deploymentid_{sx}")
        tag = lambda s: tag_medical_azure(client, model_id, s)
    else:  # bedrock
        import boto3
        region = args.aws_region or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        client = boto3.client("bedrock-runtime", region_name=region)
        model_id = args.bedrock_model
        tag = lambda s: tag_medical_bedrock(client, model_id, s)

    uniq = list(dict.fromkeys(sentences))
    key = lambda s: cache_key(model_id, s)
    todo = [s for s in uniq if not isinstance(cache.get(key(s)), list)]
    print(f"[llm] backend={args.backend} model={model_id}: "
          f"{len(uniq)} unique sentences, {len(todo)} new, "
          f"{len(uniq) - len(todo)} cached (cache={CACHE_DIR})", file=sys.stderr)

    errors = 0
    with cf.ThreadPoolExecutor(max_workers=args.max_concurrent) as ex:
        futs = {ex.submit(tag, s): s for s in todo}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            s = futs[fut]
            try:
                cache[key(s)] = fut.result()
            except Exception as e:
                errors += 1
                print(f"[warn] sentence tag failed: {type(e).__name__}: {e}",
                      file=sys.stderr)
            if i % 100 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} done ({errors} failed)", file=sys.stderr)
    if errors:
        sys.exit(f"[fail] {errors}/{len(todo)} sentences failed; "
                 f"failures NOT cached, re-run to retry.")

    return {s: (cache.get(key(s)) or []) for s in uniq}


def run_spacy(sentences: list[str], model: str, labels: set[str]) -> dict[str, list[tuple[str, str]]]:
    """NER over unique sentences. Returns {sentence: [(entity_text, label)]}."""
    import spacy

    try:
        nlp = spacy.load(model, disable=["lemmatizer", "tagger", "parser", "attribute_ruler"])
    except OSError:
        sys.exit(f"spaCy model {model!r} not found. Install it, e.g.:\n"
                 f"  pip install "
                 f"https://github.com/explosion/spacy-models/releases/download/"
                 f"{model}-3.7.1/{model}-3.7.1-py3-none-any.whl")

    uniq = list(dict.fromkeys(sentences))
    print(f"[spacy] NER over {len(uniq)} unique sentences with {model} "
          f"(labels: {sorted(labels)})", file=sys.stderr)
    out: dict[str, list[tuple[str, str]]] = {}
    for sent, doc in zip(uniq, nlp.pipe(uniq, batch_size=256)):
        out[sent] = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in labels]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", "-m", required=True, type=Path, nargs="+",
                    help="One or more manifest JSONLs whose `text` field holds "
                         "the reference transcripts to tag. Pass several (e.g. "
                         "train + dev) so occurrence counts are pooled across "
                         "them and more phrases clear the --min-count floor.")
    ap.add_argument("--output", "-o", required=True, type=Path,
                    help="Output TSV (columns: word, count, source).")
    ap.add_argument("--backend", choices=("bedrock", "azure"), default="bedrock",
                    help="LLM backend for medical tagging (default: %(default)s).")
    ap.add_argument("--bedrock-model", default=DEFAULT_BEDROCK_MODEL,
                    help="Bedrock model id / inference profile for --backend "
                         "bedrock (default: %(default)s). Uses standard AWS "
                         "credential env vars (AWS_ACCESS_KEY_ID, etc.).")
    ap.add_argument("--aws-region", default=None,
                    help="AWS region for Bedrock (default: $AWS_DEFAULT_REGION "
                         "or us-east-1).")
    ap.add_argument("--azure-suffix", default="gpt5_5",
                    help="Suffix on the AZURE_OPENAI_*_<suffix> / "
                         "deploymentid_<suffix> env vars for --backend azure "
                         "(default: %(default)s).")
    ap.add_argument("--max-concurrent", type=int, default=8,
                    help="Concurrent LLM requests (default: %(default)s).")
    ap.add_argument("--spacy-model", default="en_core_web_sm",
                    help="spaCy model for NER (default: %(default)s).")
    ap.add_argument("--spacy-labels", default=",".join(DEFAULT_SPACY_LABELS),
                    help="Comma-separated spaCy entity labels to keep "
                         "(default: %(default)s).")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip the medical-term LLM tagging (spaCy NER only).")
    ap.add_argument("--no-spacy", action="store_true",
                    help="Skip spaCy NER (medical-term LLM tagging only).")
    ap.add_argument("--min-count", type=int, default=1,
                    help="Drop phrases occurring fewer than this many times in "
                         "the reference set (default: %(default)s).")
    args = ap.parse_args()

    if args.no_llm and args.no_spacy:
        sys.exit("nothing to do: both --no-llm and --no-spacy given")

    refs = read_references(args.manifest)
    if not refs:
        sys.exit(f"no reference sentences in {args.manifest}")
    print(f"[info] {len(refs)} reference sentences from "
          f"{', '.join(m.name for m in args.manifest)}", file=sys.stderr)

    # Collect (normalized_phrase -> set(sources)) from both taggers.
    sources: dict[str, set[str]] = defaultdict(set)

    def add(raw_phrase: str, source: str) -> None:
        norm = normalize_text(raw_phrase)
        if norm:
            sources[norm].add(source)

    if not args.no_llm:
        tagged = run_llm(refs, args)
        for sent in refs:
            for term in tagged.get(sent, []):
                add(term, "llm_medical")

    if not args.no_spacy:
        labels = {s.strip() for s in args.spacy_labels.split(",") if s.strip()}
        ner = run_spacy(refs, args.spacy_model, labels)
        for sent in refs:
            for ent_text, label in ner.get(sent, []):
                add(ent_text, f"spacy_{label}")

    if not sources:
        sys.exit("no candidate phrases tagged")

    # Verify each candidate occurs verbatim as a contiguous word window in the
    # references and count its total occurrences. Grouping by token length lets
    # us scan each sentence once per distinct phrase length.
    ref_tokens = [normalize_text(r).split() for r in refs]
    by_len: dict[int, set[str]] = defaultdict(set)
    for phrase in sources:
        n = len(phrase.split())
        if n:
            by_len[n].add(phrase)

    counts: Counter = Counter()
    for toks in ref_tokens:
        for n, phrases in by_len.items():
            if len(toks) < n:
                continue
            for i in range(len(toks) - n + 1):
                window = " ".join(toks[i:i + n])
                if window in phrases:
                    counts[window] += 1

    rows = []
    n_dropped_absent = n_dropped_lowfreq = 0
    for phrase, srcs in sources.items():
        c = counts.get(phrase, 0)
        if c == 0:
            n_dropped_absent += 1
            continue
        if c < args.min_count:
            n_dropped_lowfreq += 1
            continue
        rows.append((phrase, c, ",".join(sorted(srcs))))
    rows.sort(key=lambda r: (-r[1], r[0]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["word", "count", "source"])
        w.writerows(rows)

    print(f"[done] {len(rows)} phrases -> {args.output} "
          f"(dropped {n_dropped_absent} not-verbatim, "
          f"{n_dropped_lowfreq} below min-count={args.min_count})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
