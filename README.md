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

**First valid numbers (E1, 2026-09-17).** Everything reported before this date
was measured on 200 informburo clips with a mixed metric path and should be
ignored — the trainer generated on the first 25 unshuffled batches, and
because sources concatenate in list order, generation never reached khabar or
qazaqstantv at all. The README previously headlined WER 0.919 / BLEU 3.90 on
that basis.

Scored on the canonical 453-clip common-clean subset, beam 4:

| checkpoint | WER | BLEU (norm) | BLEU (raw) | chrF | content recall |
|---|---|---|---|---|---|
| init (`ours_enriched_friend_mt5`) | 1.042 | 0.04 | 0.03 | 13.44 | 0.008 |
| clean_treat epoch 5 | 0.920 | 4.87 | 4.65 | 35.77 | 0.196 |
| clean_treat best (~ep 7) | **0.898** | **5.37** | **5.13** | **36.25** | **0.207** |
| clean_treat epoch 10 | 0.900 | 5.24 | 5.08 | 35.99 | 0.201 |

Read these carefully:

- The three trained rows differ by ≤0.5 chrF on 453 clips with no bootstrap
  intervals computed. Treat them as **indistinguishable, not as a ranking**.
- `init` scoring near-random is expected — it carries no trained `pose_norm`,
  so it is not a usable standalone baseline.
- These are **still exploratory**: the initializer is a colleague's mT5 that
  predates the data recollection, so its own train/test provenance is an open
  question.
- `clean_treat` trained on a split sharing 523 of 746 canonical qazaqstantv
  dev videos, which is why 147 of the 600 selection clips are excluded from
  every row above — the comparison is kept on identical data.

Absolute numbers remain weak and the characteristic failure is fluent but
content-free output; `scripts/diagnose_phase1.py` traced this to encoder
embeddings collapsing to ~0.98 pairwise cosine across genuinely different
clips.

**Two traps that make results on this setup easy to misread:**

- **Splits leak across dataset versions.** Archive qazaqstantv train shares
  563 video IDs with clean dev (64.2% of dev clips). Checking
  split-disjointness *within* a single manifest does not catch this. Use
  `ASAN_ROOT=~/asan_canonical` with its frozen selection manifest.
- **Select on generation quality, not val CE.** Validation cross-entropy is
  *anti-correlated* with WER/BLEU here — it rises while generation improves.
  `--select-metric` defaults to `wer` for this reason (`899cf39`).

Decoding is not a lever: beam 4 beats beam 1 by ~1.3 chrF consistently, but
repetition penalty and trigram blocking move nothing (all beam-4 variants
within 0.27 chrF).

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

**Data cleaning (resolved 2026-09-17).** A chunking bug in the transcription
pass had attached wildly mismatched transcripts to short clips — in one case a
128-word transcript on a 0.74s clip. It affected 12.9% of qazaqstantv clips,
about 4.3% of the training corpus. (An earlier note put this at "a third of
the training set"; that was wrong.) A re-collection is now integrated and
verified: training clips 40,240 -> 52,835 (+31.3%), with feature distributions
checked identical to the archive so only the intended variable changed. See
[NOTES_clean_qazaqstantv.md](NOTES_clean_qazaqstantv.md) and
[EXPERIMENT_clean_data.md](EXPERIMENT_clean_data.md).

## Quick start

```bash
pip install -r requirements.txt
export ASAN_ROOT=$HOME/asan_canonical   # config default will not exist on your box
```

All new runs should use the canonical root together with its frozen selection
manifest (`--selection-manifest $ASAN_ROOT/selection_manifest.json`): a
leak-free cross-version split with a 600-clip selection set, 200 per source.
Earlier roots leak across dataset versions.

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
