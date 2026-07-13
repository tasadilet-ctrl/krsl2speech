# KRSL → Kazakh Speech — Architecture

## Pipeline

```
Sign Video → Keypoints (COCO-WholeBody) → Encoder → Pose Embeddings → MT5 → Kazakh Text → TTS → Audio
```

## Phase 1: Sign → Text (Current)

### Encoder: Uni-Sign ST-GCN (5.3M params)
- **Input:** (B, T, 282) or (B, T, 1128) offset/enriched keypoints → mapped to Uni-Sign format
- **Projection:** Linear(3→64) per group (body-9, face-18, hands-21×2)
- **Spatial STGCN:** [64→128→256] per group
- **Temporal STGCN:** [256×3, kernel=5] per group
- **Output:** (B, T, 768) pose embeddings
- **Pretrained:** Uni-Sign CSL stage1 weights from HuggingFace

### Decoder: MT5-Base (388M params, LoRA optional)
- **Prefix token:** "Translate sign language video to Kazakh: "
- **Input:** concat(prefix_embeds, pose_emb) → (B, P+T, 768)
- **Output:** Kazakh text via beam search
- **Training:** LoRA on MT5 (frozen base, trainable adapters only) or full fine-tune

### Training
- **Optimizer:** AdamW with differential LR
  - Encoder: 5e-5 (gentle adaptation of pretrained features)
  - MT5 LoRA: 5e-4 (or full MT5: 5e-4)
- **Scheduler:** Warmup (4000 steps) + cosine decay
- **Loss:** Cross-entropy + optional masked-pose reconstruction aux loss

### Data Features
- **Offset keypoints:** (B, T, 282) — skeleton-relative coordinates
- **Velocity:** (B, T, 282) — Δt between consecutive frames
- **Acceleration:** (B, T, 282) — Δt² between consecutive frames
- **Validity:** (B, T, 282) — binary mask (1 = detected, 0 = NaN/imputed)
- **Enriched input:** (B, T, 1128) = 4 × 282 (offset + velocity + acceleration + validity)
- **Encoder adaptation:** Mapper extracts offset for skeleton mapping, validity as score channel. Velocity/acceleration implicitly captured by temporal STGCN.

### Dataset Split
- **Signer-disjoint:** Train/val split by video_id (not random clips)
- **Datasets:** Khabar KZ (30,886 clips), Informburo KZ (2,439 clips)
- **Split ratio:** 90/10 by unique video_id

## Phase 2: Sign → Prosody (Not yet trained)
- Frozen encoder + Prosody GAN generator/discriminator
- Maps pose embeddings → [F0, energy] per frame

## Phase 3: Text + Prosody → Audio (Not yet implemented)
- FastSpeech2 + HiFi-GAN vocoder

## Key Design Decisions (from KZ-RU SignFormer audit)

### Adopted and Implemented
1. **Signer-disjoint split** — ✅ Done. All datasets support `split='train'/'val'` parameter. Splits by video_id so same signer doesn't appear in both train and val.
2. **Richer pose features** — ✅ Done. `enrich_keypoints()` in `data/utils.py` computes velocity, acceleration, validity. Datasets return (T, 1128) when `use_enriched=True`. Encoder mapper uses validity as joint confidence score.
3. **Masked-pose reconstruction** — ✅ Done. `--masked-pose-ratio` flag randomly masks frames during training, adds MSE reconstruction loss via MLP decoder.
4. **LoRA mT5** — ✅ Done. `--use-lora` flag applies peft LoRA adapters to MT5 attention layers. Falls back to full fine-tuning if peft not installed.

### Not adopted (ablations for later)
- Conformer encoder (ST-GCN already works, would need full retrain)
- New GCN layout (Uni-Sign graph is battle-tested)
- RGB branch (requires source video, expensive)

## File Map

| File | Role |
|------|------|
| `data/utils.py` | Keypoint assembly, offset/velocity/acceleration features, enrichment |
| `data/khabar_dataset.py` | Khabar KZ DataLoader (signer-disjoint split, enriched features) |
| `data/informburo_dataset.py` | Informburo DataLoader (signer-disjoint split, enriched features) |
| `data/kazsign_dataset.py` | KazSign DataLoader (signer-disjoint split, enriched features) |
| `data/tts_dataset.py` | TTS DataLoader (audio → mel) |
| `models/unisign_encoder.py` | ST-GCN encoder (handles enriched input, validity-as-score) |
| `models/prosody_gan.py` | Prosody GAN (mean-pooled global query, no CLS token) |
| `models/fastspeech2.py` | FastSpeech2 text+prosody → mel |
| `train/train_encoder_mt5.py` | Phase 1 training (LoRA, masked-pose aux loss, enriched features, signer split) |
| `train/train_prosody.py` | Phase 2 training |
| `train/train_tts.py` | Standalone FastSpeech2 trainer |

## Usage

### Basic training (standard features):
```bash
PYTHONPATH=. python train/train_encoder_mt5.py \
    --config configs/config.yaml \
    --pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth
```

### With all improvements:
```bash
PYTHONPATH=. python train/train_encoder_mt5.py \
    --config configs/config.yaml \
    --pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth \
    --use-enriched \
    --use-lora \
    --masked-pose-ratio 0.15
```

### Multi-GPU:
```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. torchrun --nproc_per_node=2 train/train_encoder_mt5.py \
    --config configs/config.yaml \
    --pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth \
    --use-enriched --use-lora --masked-pose-ratio 0.15
```

### Dependencies:
- Core: `torch`, `transformers`, `librosa`, `numpy`, `pyyaml`
- LoRA: `pip install peft`
- WER: `pip install editdistance`
