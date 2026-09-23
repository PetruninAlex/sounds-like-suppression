# Sounds-Like Suppression: Bidirectional Context Biasing for ASR

Code for the paper. Build the three benchmarks, then one command sweeps all
18 (benchmark, decoder) pairs and writes the reported numbers.

This is NeMo **2.7.3** with four changed files (apart from our own pipeline code):

- `parts/context_biasing/context_graph_universal.py` — the signed depth term
  $w_d = c_0\beta + \mathrm{sign}(c_0)\ln(d)$, so a suppression path grows more
  negative with depth instead of having its penalty shrink away
- `parts/context_biasing/boosting_graph_batched.py` — negative weights in
  $[-10,10]$, and `word_tokens_file` so a suppressed phrase is keyed on the token
  ids the model actually emitted
- `parts/utils/transcribe_utils.py` — `save_token_ids`, which is what lets mining
  recover those emitted sequences
- `examples/asr/transcribe_speech.py` — declares `save_token_ids` on the transcription
  config

Everything else is [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) at tag `v2.7.3`, whose
own README is kept as [README_NeMo.md](README_NeMo.md).

## 1. Setup

**Environment.** Build the image these experiments used:

```bash
docker build -f docker/Dockerfile.speech -t nemo-speech:latest .
docker run --gpus all -it --shm-size=16g -v "$PWD/..:$PWD/.." nemo-speech:latest bash
```

Run everything from the repo root. Mount the repo's
*parent*: the checkpoints live in `../models`.

**Models.** Fetch both into the paths the pipeline resolves (`MODELS_DIR`, default
`../models`):

```bash
MODELS_DIR="${MODELS_DIR:-$PWD/../models}"
huggingface-cli download nvidia/stt_en_fastconformer_hybrid_large_pc \
    stt_en_fastconformer_hybrid_large_pc.nemo \
    --local-dir "$MODELS_DIR/stt_en_fastconformer_hybrid_large_pc"
huggingface-cli download nvidia/canary-1b canary-1b.nemo \
    --local-dir "$MODELS_DIR/canary-1b"
```

The hybrid serves CTC and RNN-T, Canary-1B the AED configurations.
`nvidia/parakeet-tdt-0.6b-v2` is pulled at run time for MultiMed's filter.

**LLM Credentials.** `cp .env.example .env` and fill in the four `*_gpt5_5` Azure OpenAI values.
Responses cache in `paper/analysis/*.cache/`.

## 2. Build the benchmarks

Each produces `paper/data/<name>_nemo/{train,eval,test}.json`. The pipeline
resolves that path from the dataset name, so the names matter.

### STOP-Music and STOP-Places

Derived from [STOP](https://facebookresearch.github.io/spoken_task_oriented_parsing/):

```bash
mkdir -p paper/data/stop_raw && cd paper/data/stop_raw
wget https://dl.fbaipublicfiles.com/stop/stop.tar.gz
tar -xzf stop.tar.gz ./LICENSE.txt ./stop
cd ../../..
```

`prepare_stop_slots.py` expects exactly that layout under
`paper/data/stop_raw/stop/`.
Then:

```bash
python paper/data/prepare_stop_slots.py --subset stop_music
python paper/data/prepare_stop_slots.py --subset stop_places
```

| subset | slots | train/val/test |
| --- | --- | --- |
| `stop_music` | `MUSIC_ARTIST_NAME`, `NAME_EVENT` | 3,744 / 1,109 / 2,421 |
| `stop_places` | `LOCATION`, `DESTINATION`, `SOURCE`, `LOCATION_MODIFIER`, `POINT_ON_MAP` | 23,137 / 6,287 / 12,804 |

An utterance is kept only if it carries one of those slots, which is why these are
far smaller than full STOP. STOP's own split is preserved, so mining and evaluation
stay disjoint.

### MultiMed

These two use paths relative to the working directory:

```bash
cd paper/data
python download_multimed_en.py
python prepare_multimed_en_manifest.py
cd ../..
```

MultiMed also needs a reference-quality pass: its transcripts carry speaker labels
and caption tags that are never spoken, and some references are badly misaligned.
Following TurboBias, utterances with WER > 0.30 are dropped, measured once with
Parakeet-TDT so all six decoders share one subset. **`run_all.sh` does this
automatically** and skips it if `parakeet_tdt_wer30/.done` exists. It leaves
22,010 / 2,372 / 4,133 utterances (85.7 h).

## 3. Run

```bash
bash paper/run_all_nohup.sh
```

Detaches via `setsid nohup`, prints the PID, the master log, and how to stop it.
18 combos, one per GPU.

| variable | default | meaning |
| --- | --- | --- |
| `DATASETS` | `multimed stop_music stop_places` | benchmarks |
| `SETTINGS` | the six decoders | `{ctc,rnnt,aed}_{greedy,beam}` |
| `GPUS` | `0,1,2,3,4,5,6,7` | pool; its size caps concurrency |
| `RUN_TAG` | empty | suffix on the output dir, to keep variants apart |
| `MAX_WER_DETERIORATION` | `1.0` | WER budget, in points, for weight selection |
| `SL_MIN_WEIGHT` / `SL_MAX_WEIGHT` | `1` / `10` | suppression sweep range |
| `SKIP_PARAKEET_WER_FILTER` | `0` | skip MultiMed's filter |
| `MODELS_DIR` / `ENV_FILE` / `LOG_DIR` | see above | |

With `RUN_TAG` empty a re-run overwrites `paper/data/predictions/<dataset>/`. Set
it to keep variants apart, and pass `DATASET` untagged — the tag is appended.

Each combo ends with a `[summary]` table in its log and in
`paper/data/predictions/<dataset>/<setting>/summary.{tsv,txt}`:

```
row                micro_F1     WER    RTFx  note
test_no_boost        0.1320  0.1508  1622.7  test, no biasing
test_boost_only      0.5461  0.1463  1554.5  test, B=6
test_sounds_like     0.5944  0.1456  1750.6  test, SL=6
```

`test_sounds_like` vs `test_boost_only` is the paper's headline comparison. Train
mines the bias list and sounds-like alternatives, validation selects $B^\star$ and
$W^\star$, test only reports.

## 4. The paper

**Sounds-Like Suppression: Bidirectional Context Biasing for Automatic Speech
Recognition**

Alexander Petrunin<sup>1,2</sup>, Avraham Faizakof<sup>2</sup>, Bracha
Laufer-Goldshtein<sup>1</sup> &nbsp;·&nbsp; <sup>1</sup>Tel Aviv University,
<sup>2</sup>Genesys
