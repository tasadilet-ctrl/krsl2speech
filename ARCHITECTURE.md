# KRSL → Kazakh text — Architecture

## Pipeline

```
Sign Video → Keypoints (COCO-WholeBody) → ST-GCN Encoder → mT5 → Kazakh Text
```

Two further stages once existed — a ProsodyGAN producing F0/energy from the
frozen encoder, and a FastSpeech2 + HiFi-GAN vocoder turning text and prosody
into audio. Both were dropped, along with an RGB-fusion direction, and their
code was removed from the working tree; it remains in git history. What is
left is a single-stage sign-to-text model. See
`EXPERIMENT_prosody_supervision.md` for the evidence that closed the prosody
work.

## Phase 1: Sign → Text (current bottleneck)

### Encoder: Uni-Sign ST-GCN (5.35M params)
- **Input:** (B, T, 282) offset keypoints, or (B, T, 1410) dual-coord enriched
- **Groups:** body (9 nodes, remapped from 11 COCO points), left/right hand (21 each, share all weights), face (18 nodes, remapped from 68 of 88 face+lip points — the 20 lip points are carried in the raw 282-dim vector but dropped at this mapping step)
- **Projection:** `Linear(3→64)` per group (`Linear(5→64)` for enriched dual-coord: x_off, y_off, x_abs, y_abs, score)
- **Spatial ST-GCN:** 3 blocks, 64→128→256 channels, adaptive graph adjacency, no temporal mixing (`t_kernel_size=1`)
- **Body-anchoring fusion:** detached body wrist/neck features added into hand/face branches
- **Temporal ST-GCN:** 3 blocks, 256 channels, kernel_size=5 — the only stage that mixes across frames
- **Output:** mean-pool over graph nodes per group → concat 4×256=1024 → `+part_para` (trainable offset) → `Linear(1024→768)`
- **Padding:** batches are zero-padded to the longest clip; padded timesteps are re-zeroed immediately before/after the temporal ST-GCN so Conv2d bias / BatchNorm affine drift can't leak into real boundary frames (fixed 2026-07-13 — see `input_lengths` param on `KeypointEncoder.forward`)
- **Pretrained:** Uni-Sign CSL stage1 weights (HuggingFace) — hands load exactly, body partially, face reinitialized

### Bridge to the decoder
- `pose_norm`: `LayerNorm(768)` on the encoder output before mT5 sees it — raw pose embeddings sit at ~6× mT5's embedding scale with near-identical direction across clips without this, saturating cross-attention.
- Fixed prefix `"Translate sign language video to Kazakh: "`, re-embedded through mT5's own table every forward, concatenated with pose embeddings as `inputs_embeds`.

### Decoder: mT5-base (~580M params, LoRA optional)
- **Input:** `concat(prefix_embeds, pose_emb)` → (B, P+T, 768), attention mask marks real vs. padded frames using each sample's true frame count (not zero-detection)
- **Output:** Kazakh text via beam search (width 4, `no_repeat_ngram_size=3`, `repetition_penalty=1.3` to suppress degenerate loops)
- **Training:** full fine-tune, or freeze mT5 and train LoRA adapters on Q/V attention projections only

### Auxiliary training signals (grounding the encoder to the video)
Cross-entropy alone let the decoder become a fluent Kazakh-news language-model prior that ignored the input video (v3 result: CE fell to ~2.1/2.77 train/val, but WER≈1.0, BLEU 0.04, content-word recall ≈ chance). Two opt-in auxiliary losses target this:
- **Masked-pose reconstruction** (`--masked-pose-ratio`, weight 0.1): corrupt frames/spans/joints (50/25/25 split of the masking budget), reconstruct original keypoints from the corrupted-input embeddings via a small MLP — self-supervised temporal-context signal.
- **Subword-BPE CTC** (`--ctc-weight`, `--ctc-vocab-size`): a linear head over per-frame embeddings predicts a SentencePiece BPE vocabulary (trained on training transcripts) with a CTC alignment loss, forcing frame-level features to align to the transcript. Character-level CTC failed here (median clip 223 frames < median transcript 258 chars → >2/3 of batches zeroed by `ctc_loss(zero_infinity=True)`); BPE pieces (~2.5 chars/piece) cut target length to ~100, restoring frames > targets.

### Training
- **Optimizer:** AdamW, differential LR — encoder at `base_lr/10` (5e-5), mT5/LoRA at `base_lr` (5e-4)
- **Scheduler:** cosine decay, warmup capped at 10% of total steps
- **Effective batch:** 8 × grad_accum 4 = 32 (single 94GB GPU, full mT5 fine-tune)
- **Checkpoint contents:** encoder, `pose_norm`, mT5 (or LoRA adapters), CTC head + BPE vocab size (if used), masked-pose decoder (if used)

## Datasets

- **asan-dataset** (primary, ~360 h): three Kazakh broadcast sources — informburo, khabar, qazaqstantv — each with **predefined, video-disjoint** train/dev/test splits (`annotations/kz/{train,dev,test}.json`), pre-extracted COCO-WholeBody pose per clip. Videos are 50 fps, downsampled 2× for the encoder. Respected as-is, not re-split.
- **khabar_kz / informburo_kz** (legacy per-source loaders): signer-disjoint 90/10 split by video_id, still supported as a fallback path if `asan.root` isn't reachable.

## Current status (2026-07-13)

- **Phase 1 v3** (full mT5 finetune + masked-pose aux, enriched features, 25 epochs): CE fell well but produced **no visual grounding** — decoder recited a Kazakh-news prior regardless of input video.
- **Phase 1 v4** (subword-BPE CTC added): about to launch, targeting the grounding failure directly.
- **Encoder padding-mask fix**: the ST-GCN encoder never masked zero-padded timesteps before its temporal convolution, letting batch padding leak a bias-driven artifact into real boundary frames and skewing BatchNorm running stats — a second, independent candidate contributor to the same symptom. Fixed and verified (see `models/unisign_encoder.py`, `input_lengths` parameter).
- **Phase 2/3 removed (2026-09-14).** They were code-complete but never
  evaluated, blocked on Phase 1 producing grounded text — which it still does
  not. With sign-to-speech dropped as a goal, carrying two untrained stages
  and their dependencies was pure maintenance cost. Recoverable from git
  history if the goal ever returns.

## File Map

| File | Role |
|------|------|
| `data/utils.py` | Keypoint assembly, offset/velocity/acceleration/validity, dual-coord enrichment |
| `data/asan_dataset.py` | Primary asan-dataset loader (video-disjoint predefined splits, optional prosody) |
| `data/khabar_dataset.py`, `data/informburo_dataset.py`, `data/kazsign_dataset.py` | Legacy per-source loaders (signer-disjoint split) |
| `data/collators.py` | Shared `PoseTextCollator` (pads variable-length keypoints/text/prosody) |
| `models/unisign_encoder.py` | ST-GCN encoder (enriched input, validity-as-score, padding-masked temporal conv) |
| `train/train_encoder_mt5.py` | Phase 1 training (LoRA, masked-pose aux, subword-CTC aux, enriched features) |
| `scripts/evaluate_phase1.py` | WER/BLEU/chrF2 + content-word recall (vs. chance) on a Phase-1 checkpoint |
| `utils/paths.py` | Per-box path overrides via env vars (`ASAN_ROOT`, `ASAN_PROSODY_ROOT`, `KRSL_OUTPUT`) |

Deprecated, kept for reference only (superseded by the files above): `train/train_encoder.py`, `train/train_encoder_ce.py`, `train/train_encoder_finetune.py`, `models/ctr_gcn_encoder.py`, `models/str_gcn.py`, `models/gloss_decoder.py`.

## Usage

### Basic Phase 1 training:
```bash
PYTHONPATH=. python train/train_encoder_mt5.py \
    --config configs/config.yaml \
    --pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth
```

### Current recommended run (v4, subword-CTC):
```bash
export ASAN_ROOT=/home/<user>/asan_local
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python train/train_encoder_mt5.py \
    --config configs/config.yaml \
    --pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth \
    --use-enriched --masked-pose-ratio 0.05 --ctc-weight 0.3 --ctc-vocab-size 2000 \
    --epochs 25 --save-dir output/phase1_v4
```

### Multi-GPU (if a box has more than one GPU available):
```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. torchrun --nproc_per_node=2 train/train_encoder_mt5.py \
    --config configs/config.yaml \
    --pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth \
    --use-enriched --use-lora --masked-pose-ratio 0.15 --ctc-weight 0.3
```

### Dependencies:
- Core: `torch`, `transformers`, `librosa`, `numpy`, `pyyaml`
- LoRA: `pip install peft`
- Metrics: `pip install editdistance sacrebleu sentencepiece`
