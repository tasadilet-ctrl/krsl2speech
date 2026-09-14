# KRSL → Kazakh text

Kazakh Sign Language translation: pose keypoints in, Kazakh text out.

> **Scope note.** This began as a sign-to-*speech* pipeline adapted from
> S2PFormer — a pose encoder feeding a prosody GAN and a FastSpeech2
> vocoder. Both later stages were dropped, and so was an RGB-fusion
> direction. The project is now keypoints → Kazakh text only. The prosody
> work was closed on evidence, not preference: see
> [EXPERIMENT_prosody_supervision.md](EXPERIMENT_prosody_supervision.md) for
> the ablation and why it was a negative result. The code for the removed
> stages is in git history, not in the working tree.

## Architecture

```
Sign keypoints (.npz, COCO-WholeBody)
    ↓
Pose encoder  (temporal transformer; optional Uni-Sign init, LoRA, masked-pose aux)
    ↓
mT5 decoder   (optionally a colleague's Kazakh-fine-tuned checkpoint)
    ↓
Kazakh text
```

An optional CTC head (`--ctc-weight`) can be trained alongside the
sequence-to-sequence objective.

## Where this stands

| metric | value |
|---|---|
| WER | 0.919 |
| BLEU | 3.90 |
| ROUGE-1 | 0.183 |
| BERTScore | 0.716 |

Enriched pose encoder plus a fine-tuned mT5. **These are weak in absolute
terms** and the model's characteristic failure is fluent but content-free
output — `scripts/diagnose_phase1.py` traced this to encoder embeddings
collapsing to ~0.98 pairwise cosine across genuinely different clips.

Two things to know before trusting any comparison on this setup:

- **Noise floor.** Run-to-run nondeterminism alone moves WER by ~0.002 and
  BLEU by ~0.17 (measured from accidentally duplicated epoch-3 runs). A
  single-seed gap smaller than that means nothing.
- **Select on generation quality, not val CE.** Validation cross-entropy is
  *anti-correlated* with WER/BLEU here — it rises while generation improves.
  `--select-metric` defaults to `wer` for this reason (`899cf39`).

## Datasets

| Dataset | Role |
|---|---|
| asan-dataset | umbrella corpus with predefined train/dev/test splits |
| khabar_kz | broadcaster sign footage: keypoints + text |
| informburo | additional broadcaster clips |
| kazsign-dataset | paired sign + audio |
| Slovo (RSL) | transfer-learning source |

**Data availability:** none of these are redistributed here. `khabar_kz` and
`informburo` are built from Kazakhstani broadcaster footage held under
restricted-access agreements at ISSAI; `kazsign-dataset` and `Slovo` are
third-party datasets under their own licenses. Only code is included. Point
`configs/config.yaml` — or the `ASAN_ROOT` / `KRSL_OUTPUT` env overrides in
`utils/paths.py` — at your own copies.

A known data problem is open: roughly a third of the training clips come from
a source reported to be dirty, and a cleaned re-collection exists but is not
yet accessible. See [NOTES_clean_qazaqstantv.md](NOTES_clean_qazaqstantv.md).
Data quality plausibly dominates method tweaks at the current numbers.

## Quick start

```bash
pip install -r requirements.txt
export ASAN_ROOT=/path/to/asan-dataset      # config default will not exist on your box
```

### Train

```bash
python train/train_encoder_mt5.py \
  --config configs/config.yaml \
  --use-enriched \
  --select-metric wer \
  --epochs 10 \
  --save-dir output/run1
```

Useful flags: `--pretrained-unisign` (Uni-Sign init), `--use-lora`,
`--freeze-spatial`, `--masked-pose-ratio` (masked-pose auxiliary loss),
`--ctc-weight`, `--resume`, `--overfit-n` (sanity-check on N clips).

### Evaluate

```bash
python scripts/evaluate_phase1.py \
  --ckpt output/run1/best.pth --use-enriched --split test --num-beams 4
```

### Diagnose

```bash
python scripts/diagnose_phase1.py --ckpt output/run1/best.pth
```

Reports pairwise embedding cosine per keypoint group — the collapse
measurement above.

## Project structure

```
krsl2speech/
├── configs/config.yaml          # paths, model dims, hyperparameters
├── data/
│   ├── asan_dataset.py          # umbrella dataset (current)
│   ├── khabar_dataset.py        # broadcaster corpus
│   ├── kazsign_dataset.py       # paired sign + audio
│   ├── informburo_dataset.py
│   ├── collators.py
│   └── utils.py                 # keypoint loading, audio feature helpers
├── models/
│   ├── keypoint_encoder.py      # temporal transformer
│   ├── unisign_encoder.py       # Uni-Sign-initialised encoder (enriched)
│   ├── gloss_decoder.py         # CTC decoder
│   ├── pgf_fusion.py            # pose-guided fusion (RGB direction, closed)
│   ├── ctr_gcn_encoder.py
│   └── str_gcn.py
├── train/
│   ├── train_encoder_mt5.py     # main trainer
│   ├── train_encoder_ce.py
│   ├── train_encoder_finetune.py
│   ├── train_encoder.py
│   └── train_pose_pretrain.py
├── scripts/                     # evaluation, diagnosis, extraction utilities
├── utils/
│   ├── losses.py
│   ├── metrics.py               # WER/BLEU/ROUGE
│   └── paths.py                 # env path overrides
└── requirements.txt
```

## Papers

- **Uni-Sign** — unified sign-language pretraining; source of the encoder init.
- **mT5** — Xue et al., "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer" (2021)
- **COCO-WholeBody** — Jiang et al., CVPR 2022; the keypoint format.
- **S2PFormer** — Manabe et al. (2024). The original basis, retained for
  provenance; the prosody/speech stages it motivated are no longer part of
  this project.
