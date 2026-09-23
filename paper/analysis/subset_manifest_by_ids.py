#!/usr/bin/env python3
"""Keep manifest rows whose audio_filepath appears in a filtered predictions JSONL.

Used after the Parakeet-TDT WER filter to write the shared train/eval/test
manifests every decoder will decode.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            fp = json.loads(line).get("audio_filepath")
            if not fp:
                raise SystemExit(f"missing audio_filepath in {path}")
            ids.add(fp)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Full cleaned source manifest (jsonl).")
    ap.add_argument("--keep-from", type=Path, required=True,
                    help="WER-filtered predictions jsonl; IDs to keep.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    keep = _ids(args.keep_from)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    missing = set(keep)
    with args.manifest.open(encoding="utf-8") as fin, args.out.open("w", encoding="utf-8") as fo:
        for line in fin:
            if not line.strip():
                continue
            n_in += 1
            d = json.loads(line)
            fp = d.get("audio_filepath")
            if fp in keep:
                fo.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_out += 1
                missing.discard(fp)
    if missing:
        print(f"[subset] {args.manifest.name}: {len(missing)} keep-from IDs "
              f"not in manifest (e.g. {next(iter(missing))})", file=sys.stderr)
        return 1
    print(f"[subset] {args.manifest.name} -> {args.out}: "
          f"kept {n_out}/{n_in} (want {len(keep)})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
