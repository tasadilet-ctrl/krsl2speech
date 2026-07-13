# KRSL2Speech — Context for Claude Code

## What this project is
KRSL2Speech translates **Kazakh Sign Language video → spoken Kazakh** in three phases:
- **Phase 1 (Sign→Text):** Uni-Sign ST-GCN pose encoder (5.3M, pretrained on CSL) → mT5-base decoder (582M). `train/train_encoder_mt5.py`.
- **Phase 2 (Prosody):** ProsodyGAN (~15M) predicts F0 + energy @100 Hz from encoder features. `train/train_prosody.py`.
- **Phase 3 (TTS):** FastSpeech2 (text + prosody → mel) + vocoder (HiFi-GAN). `train/train_tts.py`.
- End-to-end: `inference/sign2speech.py`. Trained on the **asan dataset** (~360 h) on HPC.

Encoder input is **enriched pose, dim 1410** = 5×282 blocks: `[offset, absolute, velocity, acceleration, validity]`. See `data/utils.py` (`enrich_keypoints`, `ENRICHED_DIM`) and `models/unisign_encoder.py`.

## Infrastructure (two HPC boxes, one repo synced from Mac)
- **Box A (training, has all data):** `adilet_tasbolat@100.122.214.73`, home `/data/home/adilet_tasbolat`, asan at `/data/shared/asan-dataset`. Holds the trained **v3** checkpoint.
- **Box B (this box):** `adilet_tasbolat@10.201.24.68`, home `/home/adilet_tasbolat`, datasets at `/raid/shared/dataset` (sources `informburo`, `khabar`, `qazaqstantv`, each with `annotations/kz/{train,dev,test}.json` and `pose/kz/processed/{split}/...pkl`). `/raid/shared/dataset` is **read-only**.
- Single GPU per box — pick the live index from `nvidia-smi`, then `CUDA_VISIBLE_DEVICES=<n>`.
- Sync from Mac: `rsync -av --exclude '__pycache__' --exclude 'output' ~/Desktop/ISSAI-code/krsl2speech/ adilet_tasbolat@<host>:~/krsl2speech/`

### Per-box paths via env vars (do NOT hardcode in config)
`utils/paths.py::apply_env_overrides` overrides config paths from env, so one synced `config.yaml` works on both boxes. Wired into `train_encoder_mt5.py`, `train_prosody.py`, `scripts/evaluate_phase1.py`. Recognized:
- `ASAN_ROOT` → `paths.asan.root`
- `ASAN_PROSODY_ROOT` → `paths.asan.prosody_root`
- `KRSL_OUTPUT` → `paths.output`

On **Box B**, `/raid/shared/dataset` annotations are read-only, so we built a writable root `~/asan_local` with real annotation files + **symlinked pose** back to `/raid`:
```
mkdir -p ~/asan_local
for s in informburo khabar qazaqstantv; do
  mkdir -p ~/asan_local/$s/annotations/kz
  ln -sfn /raid/shared/dataset/$s/pose ~/asan_local/$s/pose
  rsync -av adilet_tasbolat@100.122.214.73:/data/shared/asan-dataset/$s/annotations/kz/ ~/asan_local/$s/annotations/kz/
done
export ASAN_ROOT=/home/adilet_tasbolat/asan_local   # put in ~/.bashrc
```
Deps to install per box: `pip install peft editdistance sacrebleu sentencepiece`.

## Phase 1 status / findings
- **Fixed trainer** (already in repo): encoder attention mask excludes label tokens; `pose_norm` LayerNorm on pose embeds (stops prior-collapse loops); `generate()` uses `no_repeat_ngram_size=3, repetition_penalty=1.3`. Masked-pose aux loss (w=0.1) reconstructs masked joints from masked-input embeddings.
- **OOM note:** full mT5 finetune at batch 16 OOMs on 94 GB. Use `batch_size: 8`, `grad_accum: 4` (already in config) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **v3** (enriched + masked-pose, full mT5 finetune, 25 ep): CE fell well (train ~2.10 / val ~2.77) but **WER ~1.0, BLEU 0.04, chrF2 19.1, content-word recall 0.010 ≈ chance 0.008**. Diagnosis: **no visual grounding** — decoder collapsed to a Kazakh-news LM prior, emitting the same 2–3 generic sentences regardless of input video.
- **v4 (char-CTC, w=0.3):** CTC loss stayed **flat ~1.04, never dropped**. Root cause found: char-level CTC needs frames ≥ chars, but median clip is **223 frames < 258 chars** → **68.9% of clips zeroed** by `ctc_loss(zero_infinity=True)`, 95.5% too tight. CTC got almost no gradient.

## Change made this session (needs a fresh v4 run)
Switched CTC targets from **characters → subword-BPE** (SentencePiece), which cuts median target length 258 → ~100 so frames > tokens. In `train/train_encoder_mt5.py`:
- New `_build_bpe_tokenizer(paths, vocab_size)` — trains/caches `ctc_bpe_<vocab>.model` in cwd from training transcripts.
- `self.ctc_tokenizer` / `self.ctc_vocab_size` replace the old `char2id`; targets built as `[i+1 for i in sp.encode(text.lower())]` (+1 keeps CTC blank=0 free).
- New CLI arg `--ctc-vocab-size` (default 2000, ~2.5 chars/piece). Checkpoint saves `ctc_bpe_vocab_size`.
`scripts/evaluate_phase1.py` also got `apply_env_overrides` + a guard that reports the real cause when 0 clips are collected (missing split file vs. failed pose loads).

## IMMEDIATE NEXT TASK
1. `rsync` repo to the box; verify: `grep -c "ctc-vocab-size\|_build_bpe_tokenizer" train/train_encoder_mt5.py` → ≥2.
2. Confirm the length fix with the trained BPE (want `T<L` down from 68.9% to single digits):
   tokenize dev transcripts with `ctc_bpe_2000.model`, compare `ceil(T/2)` (frames, downsample=2, cap 1000) vs `len(sp.encode(text))`.
3. Launch **v4 (subword-CTC):**
```
export ASAN_ROOT=/home/adilet_tasbolat/asan_local
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python train/train_encoder_mt5.py \
  --config configs/config.yaml \
  --pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth \
  --use-enriched --masked-pose-ratio 0.05 --ctc-weight 0.3 --ctc-vocab-size 2000 \
  --epochs 25 --save-dir output/phase1_v4
```
   Success signal: the `CTC:` batch-log column now **falls** epoch over epoch (not flat ~1.0).
4. Evaluate: `scripts/evaluate_phase1.py --ckpt output/phase1_v4/phase1_mt5_best.pth --use-enriched --split val --max-clips 500`. The metric that matters is **content-word recall vs its chance baseline** (>> chance = real grounding). BLEU/chrF are the citable paper baselines.

If CTC still won't drop by ~epoch 8: raise `--ctc-vocab-size` to 4000 (shorter targets) or `--ctc-weight` to ~1.0; last resort is `downsample_every=1` (doubles frames, ~2× memory/time).

## Key files
`train/train_encoder_mt5.py` (Phase 1), `models/unisign_encoder.py` (ST-GCN), `data/asan_dataset.py`, `data/utils.py`, `data/collators.py` (`SimpleCollator`), `scripts/evaluate_phase1.py`, `utils/paths.py`, `configs/config.yaml`, `train/train_prosody.py`, `train/train_tts.py`, `inference/sign2speech.py`.

## Parked decision
Phase 3 trains TTS on noisy TV-broadcast audio → muddy mel is expected. Fallback if quality is poor: fine-tune the **KazakhTTS2** ParallelWaveGAN vocoder (mel-config must match byte-for-byte + `stats.h5` normalization).
