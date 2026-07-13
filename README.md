# KRSL → Kazakh Speech (S2PFormer-adapted)

Adapted from: **Sign-to-Speech Prosody Transfer via Sign Reconstruction-based GAN** (S2PFormer, arXiv:2604.10413)

## Architecture

```
Sign Keypoints (.npz)
    ↓
Keypoint Encoder (Temporal Transformer)
    ↓
┌─────────────────────┬──────────────────────┐
│ Gloss Decoder (CTC) │  Prosody GAN         │
│ → Kazakh text       │  → F0, energy, dur   │
└─────────┬───────────┴──────────┬───────────┘
          │                      │
          └──────┬───────────────┘
                 ↓
         FastSpeech2 (Kazakh TTS)
                 ↓
         HiFi-GAN Vocoder
                 ↓
         Kazakh Speech Audio
```

## Datasets Used

| Dataset | Clips | Role |
|---|---|---|
| khabar_kz | 30,891 clips | Main training (keypoints + text + audio) |
| kazsign-dataset | 10,037 entries | Perfectly paired sign+audio for prosody GAN |
| informburo | ~1,371 clips | Additional training data |
| Slovo (RSL) | 1,001 classes | Transfer learning (MViTv2-S → KRSL) |

## Quick Start

### 0. Setup

```bash
# On SSH server
cd /data/home/adilet_tasbolat
git clone <repo> krsl2speech  # or SCP from local
cd krsl2speech

uv venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt
```

### 1. Build Tokenizer

```bash
# Extract all Kazakh text from manifest
python3 -c "
import json
with open('/data/shared/srp-manifest/khabar_kz/khabar_kz.jsonl') as f, \
     open('data/kazakh_corpus.txt', 'w') as out:
    for line in f:
        text = json.loads(line).get('norm_text', '')
        if text:
            out.write(text + '\n')
"

# Build SentencePiece BPE tokenizer
python -c "
from sentencepiece import SentencePieceTrainer
SentencePieceTrainer.Train(
    input='data/kazakh_corpus.txt',
    model_prefix='configs/kazakh_sp',
    vocab_size=8000,
    character_coverage=0.999,
    model_type='bpe',
)
"
```

### 2. Phase 1: Encoder + Gloss Decoder

```bash
python train/train_encoder.py \
  --config configs/config.yaml \
  --tokenizer configs/kazakh_sp.model \
  --epochs 50 \
  --save-dir output/phase1
```

### 3. Phase 2: Prosody GAN

```bash
# First extract prosody from audio (optional, done on-the-fly)
python inference/extract_prosody.py \
  --audio-dir /data/shared/khabar/audio/kz \
  --output-dir data/prosody_khabar

python train/train_prosody_gan.py \
  --config configs/config.yaml \
  --encoder-checkpoint output/phase1/phase1_best.pth \
  --epochs 100 \
  --save-dir output/phase2
```

### 4. Phase 3: End-to-End Fine-tuning

```bash
python train/train_tts.py \
  --config configs/config.yaml \
  --phase1-checkpoint output/phase1/phase1_best.pth \
  --phase2-checkpoint output/phase2/phase2_best.pth \
  --epochs 30 \
  --save-dir output/phase3
```

### 5. Inference

```bash
# Single clip
python inference/sign2speech.py \
  --config configs/config.yaml \
  --checkpoint output/phase3/phase3_epoch30.pth \
  --keypoints /data/shared/srp-manifest/khabar_kz/keypoints/160745/khabar__160745__seg00000.npz \
  --output output_audio

# Batch (entire manifest)
python inference/sign2speech.py \
  --config configs/config.yaml \
  --checkpoint output/phase3/phase3_epoch30.pth \
  --keypoints /data/shared/srp-manifest/khabar_kz/khabar_kz.jsonl \
  --output output_audio_batch
```

## Config

Edit `configs/config.yaml` to adjust:
- Dataset paths (for your server)
- Model dimensions (d_model, nhead, layers)
- Training hyperparameters (batch_size, learning_rate)
- Vocabulary size

## Project Structure

```
krsl2speech/
├── configs/
│   └── config.yaml              # main config
├── data/
│   ├── khabar_dataset.py        # khabar_kz DataLoader
│   ├── kazsign_dataset.py       # kazsign-dataset DataLoader
│   └── utils.py                 # keypoint loading, prosody extraction
├── models/
│   ├── keypoint_encoder.py      # Temporal Transformer
│   ├── gloss_decoder.py         # CTC decoder
│   ├── prosody_gan.py           # SignRecGAN
│   └── fastspeech2.py           # FastSpeech2 TTS
├── train/
│   ├── train_encoder.py         # Phase 1
│   ├── train_prosody_gan.py     # Phase 2
│   └── train_tts.py             # Phase 3
├── inference/
│   ├── sign2speech.py           # end-to-end inference
│   └── extract_prosody.py       # bulk prosody extraction
├── utils/
│   ├── losses.py                # loss functions
│   └── metrics.py               # WER/CER
├── requirements.txt
└── README.md
```

## Key Papers

- **S2PFormer**: Manabe et al., "Sign-to-Speech Prosody Transfer via Sign Reconstruction-based GAN" (2024)
- **FastSpeech2**: Ren et al., "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" (2021)
- **HiFi-GAN**: Jiang et al., "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" (2021)
- **MViTv2**: Tu et al., "MaxViT: Multi-Axis Vision Transformer" (2022)
- **COCO-WholeBody**: Jiang et al., "COCO-WholeBody: Keypoint Dataset for Whole-Body Human Pose Estimation" (CVPR 2022)
