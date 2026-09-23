#!/usr/bin/env python3
"""Per-row F1 for every boost value in a sweep.

Walks predictions on disk (produced by steps 3, 9, 10), normalises
ref + hyp via ``align_predictions.normalize_text``, aligns word-by-word
with ``align_predictions.align_pair`` (jiwer), and counts per-key-word
tp / ref_count / fp. From those counts F1 is reported in two TSVs:

  <setting_dir>/boost_only_transcriptions/f1_all.tsv
      Schema: word \\t no_boost \\t boost_1 \\t … \\t boost_{max-boost}
      Rows: one per canonical key word (the *boosted* word).
      `no_boost`  = baseline F1 (from baseline_train.wer_filtered.json,
                    i.e. inference with no biasing).
      `boost_{b}` = F1 of that boosted word in the B-biased predictions.

  Baseline columns are emitted only when
  <setting_dir>/baseline_train.wer_filtered.json (the §4 manifest used
  as input by the §10 sweep) is present.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from align_predictions import (  # noqa: E402
    align_pair,
    group_keys_by_length,
    normalize_text,
)


def load_keys(path: Path) -> list[str]:
    """Read canonical (lowercased) key words from a `phrase_value` file.

    Each line is `phrase_value`: a positive value boosts the phrase, a negative
    value suppresses a sounds-like confusion. Only the boosted (non-negative)
    phrases are scored, so suppression lines are skipped. The trailing `_value`
    is stripped; phrases never contain `_` themselves (multi-word terms use
    spaces). Words are lowercased so casing variants of the same word
    (e.g. `CAS` and `Cas`) collapse to a single key, and so they match the
    normalized (lowercased) reference/hypothesis tokens in ``per_word_stats``.
    """
    seen, out = set(), []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        word, sep, value_str = s.rpartition("_")
        if sep:
            try:
                if float(value_str) < 0:
                    continue  # suppression line -- not a boosted key word
            except ValueError:
                word = s  # no numeric suffix -> the whole line is the phrase
        else:
            word = s
        w = word.lower()
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def load_pairs(path: Path) -> list[tuple[str, str]]:
    """Read (ref, alt) sounds-like pairs from a `phrase_value` file.

    The flat format stores each pair as two consecutive lines: the boosted
    target (`ref_B`, positive value) immediately followed by the suppressed
    confusion (`alt_-B`, negative value). Pairs are recovered by matching each
    positive line with the negative line that follows it; unpaired (canonical)
    positive lines are ignored. Pairs are lowercased and deduplicated.
    """
    def parse(line: str) -> tuple[str, float] | None:
        s = line.strip()
        if not s:
            return None
        word, sep, value_str = s.rpartition("_")
        if not sep:
            return None
        try:
            return word, float(value_str)
        except ValueError:
            return None

    entries = [e for e in (parse(l)
                           for l in path.read_text(encoding="utf-8").splitlines())
               if e is not None]
    seen, out = set(), []
    i = 0
    while i < len(entries) - 1:
        (ref, v_ref), (alt, v_alt) = entries[i], entries[i + 1]
        if v_ref >= 0 and v_alt < 0:
            key = (ref.lower(), alt.lower())
            if key not in seen:
                seen.add(key)
                out.append(key)
            i += 2
        else:
            i += 1
    return out


def per_word_stats(manifest: Path, keys: set[str]) -> dict[str, list[int]]:
    """Return {key: [tp, ref_count, fp]} for one manifest, normalized + aligned.

    Keys may be single words *or* multi-word terms (e.g. "sjogren's syndrome").
    For an n-word key we slide a window of n consecutive non-gap tokens:

      ref_count = times the term appears (as consecutive tokens) in the reference
      tp        = those occurrences where every aligned hyp token matches
                  (the whole term was transcribed correctly)
      fp        = times the term appears in the hypothesis without a matching
                  reference occurrence at that position

    For n == 1 this reduces exactly to the original per-word counting.
    """
    stats: dict[str, list[int]] = {k: [0, 0, 0] for k in keys}
    by_len = group_keys_by_length(keys)
    with manifest.open() as f:
        for line in f:
            obj = json.loads(line)
            ref = normalize_text(obj.get("text", "") or "").split()
            hyp = normalize_text(obj.get("pred_text", "") or "").split()
            ra, ha = align_pair(ref, hyp)
            # Drop gaps on each side so windows are over real tokens; each kept
            # ref/hyp token keeps its aligned counterpart for correctness checks.
            ref_pairs = [(r, h) for r, h in zip(ra, ha) if r]
            hyp_pairs = [(r, h) for r, h in zip(ra, ha) if h]
            for n, kset in by_len.items():
                for i in range(len(ref_pairs) - n + 1):
                    win = ref_pairs[i:i + n]
                    term = " ".join(r for r, _ in win)
                    if term in kset:
                        stats[term][1] += 1
                        if all(r == h for r, h in win):
                            stats[term][0] += 1
                for i in range(len(hyp_pairs) - n + 1):
                    win = hyp_pairs[i:i + n]
                    term = " ".join(h for _, h in win)
                    if term in kset and not all(r == h for r, h in win):
                        stats[term][2] += 1
    return stats


def prf(tp: int, ref_count: int, fp: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / ref_count if ref_count else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def write_aligned_txt(path: Path,
                      header: list[str],
                      rows: list[list[str]],
                      *,
                      label_cols: int = 1,
                      gap: int = 2,
                      title: str | None = None) -> None:
    """Write ``rows`` as a fixed-width aligned text table at ``path``.

    The first ``label_cols`` columns are left-aligned (treated as text
    labels); the remaining columns are right-aligned (treated as numeric
    values). Each column is padded to fit the widest cell or its header.
    Used to emit an easy-to-read ``.txt`` sibling next to each ``.tsv``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    widths = [
        max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
        for i in range(len(header))
    ]
    sep = " " * gap

    def fmt(cells: list[str]) -> str:
        return sep.join(
            cell.ljust(widths[i]) if i < label_cols else cell.rjust(widths[i])
            for i, cell in enumerate(cells)
        ).rstrip()

    total_w = sum(widths) + gap * (len(widths) - 1)
    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("=" * max(len(title), total_w))
        lines.append("")
    lines.append(fmt(header))
    lines.append("-" * total_w)
    for r in rows:
        lines.append(fmt(r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def full_sweep(base_dir: Path, subdir: str, file_fmt: str,
               max_b: int, keys: set[str]
               ) -> dict[int, dict[str, list[int]]]:
    """Return {B: per_word_stats} for every B=1..max_b whose manifest exists."""
    all_stats: dict[int, dict[str, list[int]]] = {}
    for b in range(1, max_b + 1):
        m = base_dir / subdir / file_fmt.format(B=b)
        if m.is_file():
            all_stats[b] = per_word_stats(m, keys)
    return all_stats



def manifest_wer(manifest: Path) -> float:
    """Compute total WER over all utterances in a manifest using jiwer.wer."""
    from jiwer import wer
    refs, hyps = [], []
    with manifest.open() as f:
        for line in f:
            obj = json.loads(line)
            refs.append(normalize_text(obj.get("text", "") or ""))
            hyps.append(normalize_text(obj.get("pred_text", "") or ""))
    if not refs:
        return 0.0
    return wer(refs, hyps)


def write_f1_all(path: Path,
                 all_stats: dict[int, dict[str, list[int]]],
                 boosted_words: list[str],
                 baseline_stats: dict[str, list[int]] | None = None,
                 baseline_manifest: Path | None = None,
                 boost_manifests: dict[int, Path] | None = None,
                 col_prefix: str = "boost") -> None:
    """Write f1_all.tsv: rows = words, columns = boost values, values = F1.

    When ``baseline_stats`` is provided, a leading ``no_boost`` column is
    inserted between ``word`` and ``boost_1`` — the per-word F1 measured
    on the unboosted baseline predictions.

    Summary rows are appended per column:
      * ``average_F1 (macro)`` — macro mean of per-word F1 across all words.
      * ``micro_P`` / ``micro_R`` / ``micro_F1`` — pooled precision / recall /
        F1 from summed tp / fp / ref_count across all words. Unlike the macro
        ``average_F1 (macro)``, micro charges every false positive (incl. spurious hits on
        words absent from the reference) against precision, so it is the metric
        used for model selection (see ``find_best_boost.py``).
      * ``over_trigger`` — count of false positives on ref=0 words (boosted
        words emitted where they never occur in the reference).
      * ``WER``          — total word error rate, computed over all words in the
        full transcriptions (not just keywords).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    boost_values = sorted(all_stats)
    cols = ["word"]
    if baseline_stats is not None:
        cols.append("no_boost")
    cols += [f"{col_prefix}_{b}" for b in boost_values]

    rows: list[list[str]] = []
    for boosted_word in boosted_words:
        row = [boosted_word]
        if baseline_stats is not None:
            c = baseline_stats[boosted_word]
            _, _, f1 = prf(c[0], c[1], c[2])
            row.append(f"{f1:.4f}")
        for b in boost_values:
            c = all_stats[b][boosted_word]
            _, _, f1 = prf(c[0], c[1], c[2])
            row.append(f"{f1:.4f}")
        rows.append(row)

    n_words = len(boosted_words)

    # Per-column stats dicts, in the same column order as the table body
    # (optional no_boost first, then each boost value).
    col_stats: list[dict[str, list[int]]] = []
    if baseline_stats is not None:
        col_stats.append(baseline_stats)
    col_stats += [all_stats[b] for b in boost_values]

    def pooled(stats: dict[str, list[int]]) -> tuple[float, float, float, int]:
        tp = sum(stats[w][0] for w in boosted_words)
        ref = sum(stats[w][1] for w in boosted_words)
        fp = sum(stats[w][2] for w in boosted_words)
        p, r, f = prf(tp, ref, fp)
        over = sum(stats[w][2] for w in boosted_words if stats[w][1] == 0)
        return p, r, f, over

    avg_row = ["average_F1 (macro)"]
    micro_p_row = ["micro_P"]
    micro_r_row = ["micro_R"]
    micro_f1_row = ["micro_F1"]
    over_row = ["over_trigger"]
    wer_row = ["WER"]

    # Body data columns start at index 1 (col 0 is the word) in both layouts.
    for k, stats in enumerate(col_stats):
        body_col = 1 + k
        col_f1s = [float(rows[i][body_col]) for i in range(n_words)]
        avg_row.append(f"{sum(col_f1s) / n_words:.4f}")
        p, r, f, over = pooled(stats)
        micro_p_row.append(f"{p:.4f}")
        micro_r_row.append(f"{r:.4f}")
        micro_f1_row.append(f"{f:.4f}")
        over_row.append(str(over))

    # WER row, in column order.
    if baseline_stats is not None:
        wer_row.append(
            f"{manifest_wer(baseline_manifest):.8f}" if baseline_manifest is not None else ""
        )
    for b in boost_values:
        if boost_manifests is not None and b in boost_manifests:
            wer_row.append(f"{manifest_wer(boost_manifests[b]):.8f}")
        else:
            wer_row.append("")

    rows.append(avg_row)
    rows.append(micro_p_row)
    rows.append(micro_r_row)
    rows.append(micro_f1_row)
    rows.append(over_row)
    rows.append(wer_row)

    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for row in rows:
            fh.write("\t".join(row) + "\n")

    write_aligned_txt(
        path.with_suffix(".txt"), cols, rows,
        label_cols=1,
        title=f"F1 by word and boost level  ({path.name})",
    )


# ── main ─────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions-dir", type=Path, default=Path("paper/data/predictions"))
    p.add_argument("--dataset", required=True, help="Dataset name (e.g. multimed).")
    p.add_argument("--setting", required=True,
                   help="Decoder setting (e.g. rnnt_beam).")
    p.add_argument("--max-boost", type=int, default=10)
    p.add_argument("--subdir", default="boost_only_transcriptions",
                   help="Subdirectory containing transcription JSONs (default: %(default)s).")
    p.add_argument("--file-fmt", default="boost_b{B}.json",
                   help="Filename format with {B} placeholder (default: %(default)s).")
    p.add_argument("--col-prefix", default="boost",
                   help="Column name prefix for sweep values (default: %(default)s).")
    p.add_argument("--kw-file", type=Path, default=None,
                   help="Key-words file to use (default: <setting>/boost_only_files/boost1.txt).")
    p.add_argument("--baseline-manifest", type=Path, default=None,
                   help="No-boost baseline predictions used for the `no_boost` F1 "
                        "column and the no_boost WER cell (default: "
                        "<setting>/baseline_train.wer_filtered.json). Point this at "
                        "the validation baseline when selecting on a held-out set.")
    args = p.parse_args()

    setting_dir = args.predictions_dir / args.dataset / args.setting
    kw_file = args.kw_file if args.kw_file else setting_dir / "boost_only_files" / "boost1.txt"
    if not kw_file.is_file():
        print(f"[error] no key-words file at {kw_file}", file=sys.stderr)
        return 1
    key_list = load_keys(kw_file)
    if not key_list:
        print(f"[error] empty key-words file {kw_file}", file=sys.stderr)
        return 1
    keys = set(key_list)
    print(f"[info] {args.dataset}/{args.setting}: {len(key_list)} keys from {kw_file}",
          file=sys.stderr)

    baseline_manifest = (args.baseline_manifest if args.baseline_manifest
                         else setting_dir / "baseline_train.wer_filtered.json")
    if not baseline_manifest.is_file():
        print(f"[warn] missing baseline {baseline_manifest}; "
              f"`no_boost` columns will be omitted", file=sys.stderr)
        baseline_manifest = None

    bo_subdir = args.subdir
    bo_stats = full_sweep(setting_dir, bo_subdir, args.file_fmt,
                          args.max_boost, keys)
    if bo_stats:
        bo_baseline = (per_word_stats(baseline_manifest, keys)
                       if baseline_manifest is not None else None)
        bo_manifests = {b: setting_dir / bo_subdir / args.file_fmt.format(B=b)
                        for b in bo_stats}
        bo_out = setting_dir / bo_subdir / "f1_all.tsv"
        write_f1_all(bo_out, bo_stats, key_list,
                     baseline_stats=bo_baseline,
                     baseline_manifest=baseline_manifest,
                     boost_manifests=bo_manifests,
                     col_prefix=args.col_prefix)
        print(f"[info] wrote {bo_out}"
              f"{' (+ no_boost)' if bo_baseline is not None else ''}",
              file=sys.stderr)
    else:
        print(f"[warn] no transcriptions found under {bo_subdir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
