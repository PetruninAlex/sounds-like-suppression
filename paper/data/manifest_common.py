"""Shared helpers for turning a Hugging Face audio dataset into NeMo manifests.

Each `prepare_*_manifest.py` script is a thin config wrapper around
`export_dataset`: it only declares the dataset name / cache dir / output dir,
which field holds the transcript, and (optionally) any extra per-row fields to
carry into the manifest.
"""

from datasets import load_dataset, Audio
from pathlib import Path
import soundfile as sf
import io
import json


def export_dataset(
    dataset_name,
    cache_dir,
    out_dir,
    *,
    config=None,
    text_key="text",
    extra_fields=None,
    split_map=None,
    transform=None,
    trust_remote_code=False,
    ext="jsonl",
    log_every=100,
):
    """Export every split of a HF dataset to `<out_dir>/<stem>.<ext>` + wavs.

    Args:
        dataset_name: HF dataset id passed to `load_dataset`.
        cache_dir: local HF cache directory.
        out_dir: output directory for manifests and extracted wavs.
        config: optional HF dataset config name.
        text_key: sample field holding the transcript.
        extra_fields: optional `sample -> dict` callable returning extra
            manifest fields (e.g. keyword contexts) to merge into each record.
        split_map: optional `{hf_split: output_stem}` mapping to rename splits
            on export (e.g. use a dataset's `validation` split as `train`).
        transform: optional `DatasetDict -> DatasetDict` callable applied after
            loading (and after disabling audio decoding) but before export. Use
            it to drop junk rows or re-partition splits (e.g. carve eval/test
            out of `validation` when a dataset's official test refs are
            withheld). Runs before `split_map` renaming.
        trust_remote_code: forward `trust_remote_code=True` to `load_dataset`
            for script-based datasets (e.g. TED-LIUM). Only meaningful on
            `datasets<4` (4.x removed loading scripts); prepare such datasets in
            an isolated `datasets==3.6.0` venv. Left off for Parquet datasets so
            it stays a no-op on `datasets>=4`.
        ext: manifest file extension (no dot).
        log_every: progress print interval.
    """
    out_dir = Path(out_dir)
    split_map = split_map or {}
    load_kwargs = {"cache_dir": cache_dir}
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    ds = load_dataset(dataset_name, config, **load_kwargs) if config else \
        load_dataset(dataset_name, **load_kwargs)

    # Disable HF's audio auto-decoding (it now requires `torchcodec`); we decode
    # the raw bytes ourselves with soundfile in `_read_audio`.
    ds = ds.cast_column("audio", Audio(decode=False))

    if transform is not None:
        ds = transform(ds)

    for split_name in ds.keys():
        _export_split(
            ds[split_name],
            split_map.get(split_name, split_name),
            out_dir=out_dir,
            text_key=text_key,
            extra_fields=extra_fields,
            ext=ext,
            log_every=log_every,
        )


def _read_audio(audio):
    """Decode an undecoded HF audio cell ({'bytes': ..., 'path': ...}) to (array, sr)."""
    if audio.get("bytes") is not None:
        return sf.read(io.BytesIO(audio["bytes"]))
    return sf.read(audio["path"])


def _export_split(ds, split_name, *, out_dir, text_key, extra_fields, ext, log_every):
    wav_dir = out_dir / split_name / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / f"{split_name}.{ext}"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for i, sample in enumerate(ds):
            array, sampling_rate = _read_audio(sample["audio"])

            wav_path = (wav_dir / f"{i:08d}.wav").resolve()
            sf.write(str(wav_path), array, sampling_rate)
            duration = len(array) / float(sampling_rate)

            record = {
                "audio_filepath": str(wav_path),
                "duration": round(duration, 3),
                "text": sample[text_key],
            }
            if extra_fields is not None:
                record.update(extra_fields(sample))

            mf.write(json.dumps(record, ensure_ascii=False) + "\n")

            if i % log_every == 0:
                print(f"{split_name}: exported {i}")

    print(f"Wrote {manifest_path}")
