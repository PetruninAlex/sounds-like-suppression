#!/usr/bin/env python3
"""Drop candidate boost phrases the model's tokenizer cannot represent.

The boosting graph tokenizes every key phrase with the ASR model's tokenizer
(``tokenizer.text_to_ids(phrase)``; see
``nemo/collections/asr/parts/context_biasing/boosting_graph_batched.py``). If a
phrase contains characters the tokenizer has no piece for -- e.g. digits like
``"type 2"`` or symbols like ``"me/cfs"`` on a tokenizer trained on spoken-form
text -- ``text_to_ids`` yields the ``<unk>`` id. The model never emits ``<unk>``,
so such a phrase can never actually match: it is dead weight in the graph.

This step reuses the exact tokenizability test from
``find_zero_recall_words.py`` (load ``model.tokenizer`` once, then keep a phrase
only when ``tokenizer.unk_id`` is not in ``tokenizer.text_to_ids(phrase)``) and
applies it to the candidate-phrase TSV, preserving all columns and row order.

For Canary's ``AggregateTokenizer`` (AED) ``text_to_ids`` needs a language id and
offsets the sub-tokenizer's ids, so the per-language ``<unk>`` id is computed as
``token_id_offset[lang] + sub_tokenizer.unk_id`` and the phrase is tokenized with
``--source-lang`` (default ``en``).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def make_tokenizability_check(tokenizer, source_lang: str):
    """Return a ``is_tokenizable(phrase) -> bool`` closure for `tokenizer`.

    Mirrors ``find_zero_recall_words.word_tokenizable``: a phrase is tokenizable
    iff its ``text_to_ids`` is non-empty and contains no ``<unk>`` id. Handles
    both the plain SentencePiece tokenizer (CTC/RNN-T) and Canary's
    ``AggregateTokenizer`` (AED), which needs a language id.
    """
    from nemo.collections.common.tokenizers import AggregateTokenizer

    if isinstance(tokenizer, AggregateTokenizer):
        if source_lang not in tokenizer.tokenizers_dict:
            sys.exit(f"--source-lang {source_lang!r} not in aggregate tokenizer "
                     f"langs: {sorted(tokenizer.tokenizers_dict)}")
        sub = tokenizer.tokenizers_dict[source_lang]
        unk_id = tokenizer.token_id_offset[source_lang] + sub.unk_id

        def to_ids(phrase: str):
            return tokenizer.text_to_ids(phrase, source_lang)
    else:
        unk_id = tokenizer.unk_id

        def to_ids(phrase: str):
            return tokenizer.text_to_ids(phrase)

    cache: dict[str, bool] = {}

    def is_tokenizable(phrase: str) -> bool:
        if phrase in cache:
            return cache[phrase]
        try:
            ids = to_ids(phrase)
            ok = bool(ids) and unk_id not in ids
        except Exception:
            ok = False
        cache[phrase] = ok
        return ok

    return is_tokenizable


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True, type=Path,
                    help="Candidate-phrase TSV with a `word` column "
                         "(e.g. boost_phrases.tsv from extract_boost_phrases.py).")
    ap.add_argument("--output", "-o", required=True, type=Path,
                    help="Output TSV (same columns as --input, minus dropped rows).")
    ap.add_argument("--model", "-m", required=True,
                    help="Path to the .nemo model whose tokenizer the boosting "
                         "graph uses (hybrid FastConformer for CTC/RNN-T, Canary "
                         "for AED).")
    ap.add_argument("--source-lang", default="en",
                    help="Language id for Canary's aggregate tokenizer "
                         "(ignored for plain SentencePiece; default: %(default)s).")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.input.open(), delimiter="\t"))
    fields = rows[0].keys() if rows else ["word", "count", "source"]
    if "word" not in fields:
        sys.exit(f"{args.input} has no 'word' column (found: {list(fields)})")

    print(f"[info] loading tokenizer from {args.model}", file=sys.stderr)
    import nemo.collections.asr as nemo_asr  # noqa: E402

    model = nemo_asr.models.ASRModel.restore_from(args.model, map_location="cpu")
    tokenizer = model.tokenizer
    print(f"[info] tokenizer={type(tokenizer).__name__}", file=sys.stderr)

    is_tokenizable = make_tokenizability_check(tokenizer, args.source_lang)

    kept, dropped = [], []
    for r in rows:
        (kept if is_tokenizable(r["word"]) else dropped).append(r)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), delimiter="\t")
        w.writeheader()
        w.writerows(kept)

    if dropped:
        preview = ", ".join(repr(r["word"]) for r in dropped[:20])
        more = "" if len(dropped) <= 20 else f", ... (+{len(dropped) - 20} more)"
        print(f"[info] dropped {len(dropped)} untokenizable phrases: {preview}{more}",
              file=sys.stderr)
    print(f"[done] {len(rows)} -> {len(kept)} tokenizable phrases -> {args.output}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
