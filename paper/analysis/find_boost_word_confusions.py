#!/usr/bin/env python3
"""Collect the utterances where a boost word was misrecognized.

For every utterance in a best-boost transcription, align the reference against
the prediction. Whenever a word we want to boost appears in the reference but
the hypothesis has a *different, non-empty* word at that aligned position (i.e.
the boost word was substituted -- "the hyp is wrong in the boost word's
place"), emit one row with the boost word and the FULL reference and
hypothesis sentences.

This is alignment-only (no re-inference). The detection is done on normalized
text, but the emitted sentences are the original (un-normalized) ones so a
downstream LLM gets the full, readable context.

Output TSV columns: audio_filepath word ref_text hyp_text pred_token_ids. The
`audio_filepath` column is the source utterance's audio path (copied verbatim
from the transcription) so a row can be traced back to the audio it came from.
The
`pred_token_ids` column carries the hypothesis' raw decoded token-id sequence
(space-separated, as written by inference with `save_token_ids=True`) so a
downstream step can suppress the exact tokens the model emitted. It is meant to
be fed to llm_find_sounds_like.py, which reads the two sentences and identifies
the sounds-alike confusion produced in place of each boost word.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from align_predictions import (  # noqa: E402
    align_pair,
    group_keys_by_length,
    normalize_text,
)


def load_boost_words(path: Path) -> dict[str, str]:
    """Map normalized (lowercased) boost word/term -> original-cased form.

    The first `_`-separated token per line is the boost word or multi-word term
    (terms contain spaces, never `_`, so splitting on `_` keeps them intact);
    its original casing (e.g. `SNPs`, `Sjogren's`, `EHDI`, `Sjogren's syndrome`)
    is preserved as the value so we can emit it verbatim, while the lowercased
    key is used for matching against the normalized alignment.
    """
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        w = line.strip().split("_")[0]
        if w:
            out.setdefault(w.lower(), w)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcription", "-t", required=True, type=Path, nargs="+",
                    help="One or more best-boost predictions JSONL files "
                         "(text + pred_text per line). When several are given "
                         "(e.g. train + validation) their utterances are pooled "
                         "and (word, ref, hyp) triples are deduped across them.")
    ap.add_argument("--boost-file", required=True, type=Path,
                    help="Canonical boost list (e.g. boost1.txt) -- the words we boost.")
    ap.add_argument("--out", "-o", required=True, type=Path,
                    help="Output TSV (columns: audio_filepath word ref_text hyp_text pred_token_ids).")
    args = ap.parse_args()

    boost_words = load_boost_words(args.boost_file)
    boost_by_len = group_keys_by_length(boost_words.keys())
    print(f"[info] {len(boost_words)} boost words/terms from {args.boost_file.name}",
          file=sys.stderr)

    # Dedupe identical (word, ref_text, hyp_text) triples so we don't ask the
    # LLM the same question twice. pred_token_ids rides along (it is a function
    # of hyp_text, so it is constant within a dedup group).
    seen: set[tuple[str, str, str]] = set()
    rows: list[tuple[str, str, str, str, str]] = []
    n_utts = 0
    for transcription in args.transcription:
        with transcription.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                audio_filepath = (obj.get("audio_filepath", "") or "").strip()
                ref_text = (obj.get("text", "") or "").strip()
                hyp_text = (obj.get("pred_text", "") or "").strip()
                # Raw decoded token-id sequence for this hypothesis (space-separated,
                # matching the word_tokens.tsv convention). Empty if not saved.
                pred_token_ids = " ".join(str(int(t)) for t in obj.get("pred_token_ids", []))
                ref = normalize_text(ref_text).split()
                hyp = normalize_text(hyp_text).split()
                ra, ha = align_pair(ref, hyp)
                # Boost words/terms present in the reference that the hypothesis did
                # not reproduce exactly, yet produced *something* (not a pure
                # deletion) -- i.e. the term was substituted. Slide a window of the
                # term's length over the non-gap reference tokens (keeping each
                # token's aligned hyp counterpart) and keep the original casing.
                ref_pairs = [(r, h) for r, h in zip(ra, ha) if r]
                misrecognized = set()
                for n, kset in boost_by_len.items():
                    for i in range(len(ref_pairs) - n + 1):
                        win = ref_pairs[i:i + n]
                        term = " ".join(r for r, _ in win)
                        if term in kset and not all(r == h for r, h in win) \
                                and any(h for _, h in win):
                            misrecognized.add(boost_words[term])
                for word in sorted(misrecognized):
                    key = (word, ref_text, hyp_text)
                    if key not in seen:
                        seen.add(key)
                        rows.append((audio_filepath, word, ref_text, hyp_text, pred_token_ids))
                n_utts += 1

    rows.sort(key=lambda x: (x[1], x[2]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["audio_filepath", "word", "ref_text", "hyp_text", "pred_token_ids"])
        writer.writerows(rows)

    print(f"[done] {n_utts} utterances -> {len(rows)} misrecognized-boost-word "
          f"rows in {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
