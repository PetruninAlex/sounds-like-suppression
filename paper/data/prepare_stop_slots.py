#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import soundfile as sf

_SLOT_RE = re.compile(r"\[SL:([A-Z_]+)\s+([^\[\]]+?)\s*\]")
_SPK0_DIR_RE = re.compile(r"_0$")


SUBSET_SLOTS = {
    "stop_music": {"MUSIC_ARTIST_NAME", "NAME_EVENT"},
    "stop_places": {"LOCATION", "DESTINATION", "SOURCE", "LOCATION_MODIFIER", "POINT_ON_MAP"},
}


def to_disk_path(file_id: str) -> str:
    parts = file_id.strip().split("/")
    if len(parts) != 3:
        return file_id.strip()
    split_dir, domain_dir, name = parts
    return f"{split_dir}/{_SPK0_DIR_RE.sub('', domain_dir)}/{name}"


def extract_entities(seqlogical: str, slot_types: set) -> list:
    """Return the entity strings whose slot tag is in ``slot_types`` (train only)."""
    return [value for slot, value in _SLOT_RE.findall(seqlogical)
            if slot in slot_types]


def make_record(rel: str, ref: str, hotwords: list, stop_root: Path):
    audio = stop_root / rel
    if not audio.exists():
        return None
    info = sf.info(str(audio))
    return {
        "utt_id": rel,
        "audio_filepath": str(audio.resolve()),
        "duration": round(info.frames / float(info.samplerate), 3),
        "text": ref,
        "hotwords": hotwords,
        "stop_split": rel.split("/")[0],
        "domain": rel.split("/")[1],
    }


def build_from_slots(tsv_path: Path, stop_root: Path, slot_types: set) -> list:
    """Keep the utterances carrying a target slot, tagged with its values."""
    records, missing_audio = [], 0
    with tsv_path.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            fid = row.get("file_id")
            seq = row.get("normalized_seqlogical")
            ref = row.get("normalized_utterance")
            if not fid or not ref:
                continue
            hotwords = extract_entities(seq, slot_types)
            if not hotwords:
                continue
            rec = make_record(to_disk_path(fid), ref, hotwords,
                              stop_root)
            if rec is None:
                missing_audio += 1
                continue
            records.append(rec)
    if missing_audio:
        print(f"[warn] {tsv_path.name}: skipped {missing_audio} rows without audio",
              file=sys.stderr)
    return records


def write_manifest(records, path: Path) -> None:
    records.sort(key=lambda r: r["utt_id"])
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=sorted(SUBSET_SLOTS), default="stop_music")
    ap.add_argument("--stop-root", type=Path, default=here / "stop_raw" / "stop",
                    help="extracted STOP root (contains manifests/ and train/eval_*/test_*)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <data>/<subset>_nemo")
    args = ap.parse_args()

    slot_types = SUBSET_SLOTS[args.subset]
    out_dir = args.out_dir or here / f"{args.subset}_nemo"
    out_dir.mkdir(parents=True, exist_ok=True)

    def tsv(split):
        p = args.stop_root / "manifests" / f"{split}.tsv"
        return p if p.exists() else sys.exit(f"[fatal] missing {p}")

    by_split = {}
    for split in ("train", "eval", "test"):
        by_split[split] = build_from_slots(tsv(split), args.stop_root, slot_types)
        write_manifest(by_split[split], out_dir / f"{split}.json")

    # Bias list: slot values attested in eval+test, the splits that get scored.
    uniq = sorted({h for s in ("eval", "test")
                   for r in by_split[s] for h in r["hotwords"]})
    (out_dir / "oracle_list.attested.txt").write_text("\n".join(uniq) + "\n")


if __name__ == "__main__":
    main()
