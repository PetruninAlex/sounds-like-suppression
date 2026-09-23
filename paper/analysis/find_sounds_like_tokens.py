#!/usr/bin/env python3
"""Extract the observed token-id sequence the model emitted for each sounds-like
confusion.

Each classified row (from ``llm_find_sounds_like.py``) carries the utterance's
raw decoded ``pred_token_ids`` and the exact ``sounds_like`` confusion the ASR
produced in place of the boost word. So the job is simply: segment that row's
own token ids into words (by the tokenizer's "▁" word-boundary marker) and find
the contiguous run of segments that spells the ``sounds_like`` word -- those are
exactly the tokens the model emitted for the confusion. No alignment needed.

Output is a word→token-ids TSV in the SAME format as the inference-time
``*.word_tokens.tsv`` files:

    word<TAB>token_ids       # token_ids space-separated; one row per observed
                             # tokenization (a word may appear on several rows)

keyed by the ``sounds_like`` confusion exactly as it appears in the classified
TSV, so the boosting graph can suppress exactly those tokens.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# Strip only commas, periods, and question marks (and lowercase) when comparing
# surfaces, so the confusion matches regardless of trailing punctuation or the
# PC model's casing. The emitted token ids are always the raw decoded ones.
_PUNCT_RE = re.compile(f"[{re.escape(',.?')}]")


def _norm(s: str) -> str:
    return " ".join(_PUNCT_RE.sub("", s.lower()).split())


def segment_token_ids(tokenizer, token_ids: list[int]) -> list[tuple[str, tuple[int, ...]]]:
    """Segment a flat token-id list into [(word, (id, ...)), ...] by "▁" markers.

    Mirrors ``generate_boost_tokenizations.extract_word_tokenizations`` but
    operates on a raw id list (as stored in ``pred_token_ids``) rather than a
    Hypothesis. Drops ids the tokenizer cannot render (e.g. the CTC blank id).
    """
    ovs = getattr(tokenizer, "original_vocab_size", None)
    specials = getattr(tokenizer, "id_to_special_token", {}) or {}
    if ovs is not None:
        token_ids = [t for t in token_ids if t < ovs or t in specials]
    if not token_ids:
        return []
    tokens = tokenizer.ids_to_tokens(token_ids)
    groups: list[list[int]] = []
    for i, tok in enumerate(tokens):
        if tok.startswith("▁") or not groups:
            groups.append([i])
        else:
            groups[-1].append(i)
    out: list[tuple[str, tuple[int, ...]]] = []
    for grp in groups:
        word = "".join(tokens[i] for i in grp).replace("▁", "")
        if word:
            out.append((word, tuple(token_ids[i] for i in grp)))
    return out


def find_confusion_tokens(segs, sounds_like) -> tuple[int, ...] | None:
    """Find the contiguous run of segments whose surface spells `sounds_like`,
    and return its token ids. Returns None if the confusion isn't found in the
    segmented tokens."""
    sl_norm = _norm(sounds_like)
    if not sl_norm:
        return None
    n = len(segs)
    for i in range(n):
        parts: list[str] = []
        ids: list[int] = []
        for j in range(i, n):
            w, w_ids = segs[j]
            parts.append(_norm(w))
            ids.extend(w_ids)
            running = " ".join(p for p in parts if p)
            if running == sl_norm:
                return tuple(ids)
            if not sl_norm.startswith(running + " "):
                break
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classified", "-c", required=True, type=Path,
                    help="LLM sounds-like TSV from llm_find_sounds_like.py "
                         "(columns: audio_filepath word ref_text hyp_text "
                         "pred_token_ids sounds_like).")
    ap.add_argument("--model", "-m", required=True,
                    help="Path to .nemo model whose tokenizer segmented the audio.")
    ap.add_argument("--out", "-o", required=True, type=Path,
                    help="Output word→token-ids TSV (word_tokens.tsv format).")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.classified.open(), delimiter="\t"))

    print(f"[info] loading tokenizer from {args.model}", file=sys.stderr)
    import nemo.collections.asr as nemo_asr  # noqa: E402

    model = nemo_asr.models.ASRModel.restore_from(args.model, map_location="cpu")
    tokenizer = model.tokenizer
    print(f"[info] tokenizer={type(tokenizer).__name__}", file=sys.stderr)

    # confusion word -> set of observed token-id tuples (deduped across rows)
    word_tokens: dict[str, set[tuple[int, ...]]] = {}
    n_rows = n_found = 0
    for r in rows:
        sl = (r.get("sounds_like", "") or "").strip()
        ids_str = (r.get("pred_token_ids", "") or "").strip()
        if not sl or not ids_str:
            continue
        n_rows += 1
        token_ids = [int(x) for x in ids_str.split()]
        segs = segment_token_ids(tokenizer, token_ids)
        ids = find_confusion_tokens(segs, sl)
        if ids:
            n_found += 1
            word_tokens.setdefault(sl, set()).add(ids)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_seqs = 0
    with args.out.open("w", encoding="utf-8", newline="") as f:
        f.write("word\ttoken_ids\n")
        for word in sorted(word_tokens):
            for ids_tuple in sorted(word_tokens[word]):
                f.write(f"{word}\t{' '.join(str(t) for t in ids_tuple)}\n")
                n_seqs += 1

    print(f"[done] {n_rows} classified rows, {n_found} with the confusion found "
          f"in their tokens -> {len(word_tokens)} confusion words, {n_seqs} token "
          f"sequences in {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
