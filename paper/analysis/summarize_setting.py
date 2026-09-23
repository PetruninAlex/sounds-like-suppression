#!/usr/bin/env python3
"""Per-(dataset, setting) summary of avg F1 + WER (+ test RTFx) for val and test.

Consolidates the numbers the pipeline already wrote to disk into a single
``summary.tsv`` (+ aligned ``summary.txt``) under the setting directory, so
every time ``run_all_steps.sh`` finishes one (dataset, setting) combo you get a
compact table of:

  train_no_boost     no-biasing baseline on the (wer-filtered) train set
  train_best_boost   best boost-only config, on the split it was selected on
  val_no_boost       no-biasing baseline on the held-out (wer-filtered) eval set
  val_best_boost     the selected boost value, decoded on the eval set
  val_best_sl        best boost + sounds-like config on the eval set
  test_no_boost      no-biasing baseline on the TEST set
  test_boost_only    best boost-only config on the TEST set
  test_sounds_like   best boost + sounds-like config on the TEST set

"Best" is the value the pipeline already selected (highest micro F1 subject
to the WER-deterioration filter); this script does not re-select, it just reads
the matching cells, so the summary always agrees with what the pipeline ran.

The train and validation rows are kept apart because the boost sweep runs on
TRAIN -- that is the split the phrases were mined from -- so its table reports
train-fitted numbers. They used to be labelled "val", which made the boost rows
look held out when they were not. The genuinely held-out boost figure comes from
the separate single-config eval decode (Step 11b), and the sounds-like one from
the sounds-like validation sweep.

All F1 cells use the pooled ``micro_F1`` row (the metric the pipeline selects
on); test numbers come from ``test_eval/test_method_comparison.tsv`` (written
by ``report_test_methods.py``) and, for the RTFx column, from
``test_eval/test_rtfx.tsv`` (written by ``collect_rtfx.py``). Missing inputs are
reported as ``-``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from summarize_boost_f1 import write_aligned_txt  # noqa: E402

MISSING = "-"


def read_cell(tsv: Path, row_label: str, col_name: str) -> str:
    """Return the cell at (first-column == row_label, header == col_name).

    Returns ``MISSING`` if the file/row/column is absent or the cell is empty.
    """
    if not tsv.is_file():
        return MISSING
    with tsv.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if col_name not in header:
            return MISSING
        col = header.index(col_name)
        for line in fh:
            cells = line.rstrip("\n").split("\t")
            if cells and cells[0] == row_label:
                return cells[col] if col < len(cells) and cells[col] != "" else MISSING
    return MISSING


def fmt_f1(v: str) -> str:
    return f"{float(v):.4f}" if v != MISSING else MISSING


def fmt_wer(v: str) -> str:
    return f"{float(v):.4f}" if v != MISSING else MISSING


def fmt_rtfx(v: str) -> str:
    return f"{float(v):.1f}" if v != MISSING else MISSING


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--predictions-dir", type=Path,
                    default=Path("paper/data/predictions"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--best-boost", required=True,
                    help="Best boost-only value B (selected by the pipeline).")
    ap.add_argument("--best-sl-weight", required=True,
                    help="SL weight of the best sounds-like config.")
    ap.add_argument("--track-tag", default="",
                    help="Optional output-namespace prefix for the sounds-like / "
                         "test dirs, for running a labelled variant side by side. "
                         "Empty in normal use. The boost-only rows always read the "
                         "untagged dirs.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output TSV (default: <setting_dir>/<track-tag>summary.tsv).")
    args = ap.parse_args()

    tag = args.track_tag
    setting_dir = args.predictions_dir / args.dataset / args.setting
    # Boost-only decodes never involve suppression, so they are shared across
    # tracks and always read from the untagged dirs.
    boost_tsv = setting_dir / "boost_only_transcriptions" / "f1_all.tsv"
    boost_val_tsv = setting_dir / "boost_only_transcriptions_validation" / "f1_all.tsv"
    # The sounds-like sweep and the test evaluation carry the track tag.
    sl_tsv = (setting_dir
              / f"{tag}sl_boost_with_sounds_like_transcriptions_validation"
              / "f1_all.tsv")
    test_tsv = setting_dir / f"{tag}test_eval" / "test_method_comparison.tsv"
    rtfx_tsv = setting_dir / f"{tag}test_eval" / "test_rtfx.tsv"

    # label -> (avg F1, WER) pulled from the already-written tables.
    # The boost/SL sweep tables are produced on the held-out VALIDATION (eval)
    # set, where best boost / best SL are selected, so these rows are val numbers.
    # Every F1 cell is the pooled ``micro_F1`` row -- the metric the pipeline
    # selects on -- since micro-averaging charges every false positive (incl.
    # spurious hits on words absent from the reference) against precision,
    # unlike the macro per-word mean. Validation numbers come from the boost/SL
    # sweep tables; test numbers from ``test_method_comparison.tsv``.
    #
    # The RTFx column (decoding throughput, higher is faster) is only defined for
    # the test rows: it comes from ``test_eval/test_rtfx.tsv``, which times the
    # three test methods on the same manifest (written by collect_rtfx.py when
    # the pipeline runs with MEASURE_RTFX=1). Validation rows have no timing.
    specs = [
        ("train_no_boost", boost_tsv, "no_boost", "micro_F1", None,
         "train, no biasing"),
        ("train_best_boost", boost_tsv, f"boost_{args.best_boost}", "micro_F1", None,
         f"train, B={args.best_boost}"),
        ("val_no_boost", boost_val_tsv, "no_boost", "micro_F1", None,
         "val, no biasing"),
        ("val_best_boost", boost_val_tsv, f"boost_{args.best_boost}", "micro_F1", None,
         f"val, B={args.best_boost}"),
        ("val_best_sl", sl_tsv, f"sounds_like_{args.best_sl_weight}", "micro_F1", None,
         f"val, SL={args.best_sl_weight}"),
        ("test_no_boost", test_tsv, "no_boost", "micro_F1", "no_boost",
         "test, no biasing"),
        ("test_boost_only", test_tsv, "boost_only", "micro_F1", "boost_only",
         f"test, B={args.best_boost}"),
        ("test_sounds_like", test_tsv, "sounds_like", "micro_F1", "sounds_like",
         f"test, SL={args.best_sl_weight}"),
    ]

    cols = ["row", "micro_F1", "WER", "RTFx", "note"]
    rows: list[list[str]] = []
    for label, tsv, col, src_row, rtfx_row, note in specs:
        f1 = fmt_f1(read_cell(tsv, src_row, col))
        wer = fmt_wer(read_cell(tsv, "WER", col))
        rtfx = fmt_rtfx(read_cell(rtfx_tsv, rtfx_row, "rtfx")) if rtfx_row else MISSING
        rows.append([label, f1, wer, rtfx, note])


    out = args.out if args.out is not None else setting_dir / f"{tag}summary.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for row in rows:
            fh.write("\t".join(row) + "\n")
    write_aligned_txt(
        out.with_suffix(".txt"), cols, rows, label_cols=1,
        title=f"Summary: {args.dataset} / {args.setting}  (micro F1 + WER + RTFx)",
    )

    # Echo the table to stdout so it lands in the per-combo + master logs.
    widths = [max(len(cols[i]), *(len(r[i]) for r in rows)) for i in range(len(cols))]

    def line(cells: list[str]) -> str:
        return "  ".join(
            cells[i].ljust(widths[i]) if i in (0, len(cols) - 1) else cells[i].rjust(widths[i])
            for i in range(len(cells))
        ).rstrip()

    print(f"[summary] {args.dataset}/{args.setting}")
    print("[summary] " + line(cols))
    for row in rows:
        print("[summary] " + line(row))
    print(f"[summary] wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
