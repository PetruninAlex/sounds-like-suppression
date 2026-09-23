#!/usr/bin/env python3
"""Step 12: Find the best boost value from the F1 sweep table.

Reads the f1_all.tsv produced by step 11, finds the boost value (column)
with the highest micro F1 across all key words (subject to the WER filter),
and prints it to stdout. This value is used by subsequent steps to select
which boosted transcription to mine further. Pass
``--metric-row "average_F1 (macro)"`` for the legacy macro-F1 selection.

Pure-Python, no API calls.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def find_best_boost(
    f1_path: Path,
    col_prefix: str = "boost",
    max_wer_deterioration: float | None = 1.0,
    baseline_col: str = "no_boost",
    baseline_wer: float | None = None,
    metric_row: str = "micro_F1",
    select: str = "f1",
) -> tuple[int, float]:
    """Read f1_all.tsv and return (best sweep value, its score on ``metric_row``).

    `col_prefix` selects the sweep columns (e.g. `boost` -> `boost_1`, or
    `sounds_like` -> `sounds_like_1`); the trailing integer is the value.

    `metric_row` is the summary row used to rank configs. Defaults to
    ``micro_F1`` (pooled tp/fp/ref), which -- unlike the macro
    ``average_F1 (macro)`` -- penalizes over-triggering on words absent from the
    reference. Falls back to ``average_F1 (macro)`` with a warning if the
    requested row is missing (e.g. an old table generated before micro rows were
    added).

    `max_wer_deterioration` limits how much WER (in percentage points) a config
    is allowed to degrade relative to the baseline (no-boost) WER. Only configs
    whose WER stays within `baseline_wer + max_wer_deterioration/100` are
    considered when picking the best average F1. Pass ``None`` to disable the
    filter. When no config meets the budget the lowest-WER config is returned
    rather than the highest-F1 one -- see the comment at that branch.

    The baseline WER is taken from `baseline_wer` if given (a fraction, e.g.
    0.1257), otherwise read from the `baseline_col` cell of the `WER` row. The
    filter is skipped (with a warning) if no baseline can be determined or the
    file has no `WER` row.

    `select` chooses the objective. ``f1`` maximises `metric_row` under the WER
    budget. ``wer`` instead minimises WER outright, breaking ties on
    `metric_row`, which is what TurboBias does (arXiv:2508.07014 Sec. IV: the
    boosting weights are "selected ... in such a way as to obtain the minimum
    possible WER value on development sets"). That rule only makes sense when
    the context list is dense enough for term errors to move WER; where it is
    not, WER is blind to the terms and the F1 objective with a WER guard is the
    right one. The budget is not applied in ``wer`` mode, since minimising WER
    cannot exceed it.
    """
    with f1_path.open() as f:
        header = f.readline().strip().split("\t")
        rows = [line.strip().split("\t") for line in f]

    score_row = next((r for r in rows if r[0] == metric_row), None)
    if score_row is None:
        print(
            f"[warn] no '{metric_row}' row in {f1_path}; "
            "falling back to 'average_F1 (macro)'",
            file=sys.stderr,
        )
        score_row = next(r for r in rows if r[0] == "average_F1 (macro)")
    wer_row = next((r for r in rows if r[0] == "WER"), None)

    pfx = f"{col_prefix}_"
    cols = [c for c in header if c.startswith(pfx)]
    if not cols:
        raise ValueError(f"no '{pfx}*' columns in {f1_path}")

    if select == "wer":
        if wer_row is None:
            raise ValueError(f"--select wer needs a WER row in {f1_path}")
        best_col = min(
            cols,
            key=lambda c: (float(wer_row[header.index(c)]),
                           -float(score_row[header.index(c)])),
        )
        return int(best_col[len(pfx):]), float(score_row[header.index(best_col)])

    eligible = cols
    if max_wer_deterioration is not None:
        if baseline_wer is None and wer_row is not None and baseline_col in header:
            baseline_wer = float(wer_row[header.index(baseline_col)])
        if wer_row is None or baseline_wer is None:
            print(
                f"[warn] no baseline WER (passed or '{baseline_col}' column) / "
                f"WER row in {f1_path}; skipping WER-deterioration filter",
                file=sys.stderr,
            )
        else:
            max_wer = baseline_wer + max_wer_deterioration / 100.0
            eligible = [
                c for c in cols
                if float(wer_row[header.index(c)]) <= max_wer
            ]
            if not eligible:
                # Nothing meets the budget, so the choice is which way to fail.
                # Ranking all configs by F1 (the old behaviour) picks the single
                # most WER-damaging one, because F1 and WER cost rise together
                # under suppression -- on stop_music rnnt_greedy that took SL=7 at
                # +3.25 pts over a +1.0 budget. Taking the cheapest config
                # instead still returns a result and still overshoots, but by
                # the smallest possible margin (SL=1, +1.00 pts there).
                best_col = min(cols, key=lambda c: float(wer_row[header.index(c)]))
                over = float(wer_row[header.index(best_col)]) - baseline_wer
                print(
                    f"[warn] no '{pfx}*' config within +{max_wer_deterioration} "
                    f"WER pts of baseline ({baseline_wer * 100:.4f}%); "
                    f"selecting the lowest-WER config '{best_col}' "
                    f"(+{over * 100:.2f} pts) instead of the highest-F1 one",
                    file=sys.stderr,
                )
                return int(best_col[len(pfx):]), float(score_row[header.index(best_col)])

    # Ties on F1 go to the lowest-WER config. They are common because validation
    # scores few term occurrences, so several sweep values recover exactly the
    # same words -- on stop_music rnnt_greedy B=1, 2 and 3 all scored 0.2909. Taking
    # the first maximum, as plain max() does, resolved that by column order,
    # i.e. arbitrarily. When validation cannot separate two configs on the metric
    # we are maximising, the one that costs less WER is the better bet.
    def rank(c):
        f1 = float(score_row[header.index(c)])
        return (f1, -float(wer_row[header.index(c)])) if wer_row is not None else (f1,)

    best_col = max(eligible, key=rank)
    best_f1 = float(score_row[header.index(best_col)])
    return int(best_col[len(pfx):]), best_f1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions-dir", type=Path,
                    default=Path("paper/data/predictions"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--subdir", default="boost_only_transcriptions",
                    help="Transcriptions subdir holding f1_all.tsv (default: %(default)s).")
    ap.add_argument("--col-prefix", default="boost",
                    help="Sweep column prefix in f1_all.tsv (e.g. boost, sounds_like; "
                         "default: %(default)s).")
    ap.add_argument("--print", dest="print_what", choices=["value", "f1"],
                    default="value",
                    help="What to print to stdout: the best sweep value or its "
                         "average F1 (default: %(default)s).")
    ap.add_argument("--max-wer-deterioration", type=float, default=1.0,
                    help="Max allowed WER increase (in percentage points) over "
                         "the no-boost baseline; only configs within this margin "
                         "are eligible for selection. Use a negative value to "
                         "disable the filter (default: %(default)s).")
    ap.add_argument("--baseline-col", default="no_boost",
                    help="Header column holding the baseline (no-boost) WER, used "
                         "only when --baseline-wer is not given (default: %(default)s).")
    ap.add_argument("--baseline-wer", type=float, default=None,
                    help="Baseline (no-boost) WER as a fraction (e.g. 0.1257). "
                         "Overrides the value read from the f1_all.tsv WER row.")
    ap.add_argument("--metric-row", default="micro_F1",
                    help="Summary row used to rank configs (default: %(default)s). "
                         "Use 'average_F1 (macro)' for the legacy macro-F1 selection.")
    ap.add_argument("--select", choices=["f1", "wer"], default="f1",
                    help="Objective: 'f1' maximises --metric-row under the WER "
                         "budget; 'wer' minimises WER outright (ties on the "
                         "metric), the TurboBias rule, which suits dense context "
                         "lists where term errors actually move WER "
                         "(default: %(default)s).")
    args = ap.parse_args()

    f1_path = (args.predictions_dir / args.dataset / args.setting
               / args.subdir / "f1_all.tsv")
    if not f1_path.is_file():
        print(f"[error] missing {f1_path}", file=sys.stderr)
        return 1

    max_wer = None if args.max_wer_deterioration < 0 else args.max_wer_deterioration
    best_b, best_f1 = find_best_boost(f1_path, args.col_prefix, max_wer,
                                      args.baseline_col, args.baseline_wer,
                                      args.metric_row, args.select)
    print(f"[info] best value: {best_b} ({args.metric_row}: {best_f1:.4f})",
          file=sys.stderr)
    print(best_f1 if args.print_what == "f1" else best_b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
