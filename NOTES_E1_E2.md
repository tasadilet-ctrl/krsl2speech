# E1 re-scoring + E2 pose repairs — 2026-09-17

## E1 — existing checkpoints on the canonical selection set

`scripts/rescore_checkpoints.py`, run via `~/krsl2speech/run_e1.sh` (tmux `krsl`).
Outputs, including every prediction: `~/krsl2speech/output/e1/`.

**Common clean subset: 453/600 clips** (all informburo + khabar, only 53
qazaqstantv). `clean_treat` trained on the recollected train split, which shares
523 of 746 canonical qazaqstantv dev videos, so 147 selection clips are
excluded for every checkpoint to keep the comparison on identical data.

| checkpoint | WER | BLEU (norm) | BLEU (raw) | chrF | content recall |
|---|---|---|---|---|---|
| init (`ours_enriched_friend_mt5`) | 1.042 | 0.04 | 0.03 | 13.44 | 0.008 |
| clean_treat epoch 5 | 0.920 | 4.87 | 4.65 | 35.77 | 0.196 |
| clean_treat best (≈ep 7) | **0.898** | **5.37** | **5.13** | **36.25** | **0.207** |
| clean_treat epoch 10 | 0.900 | 5.24 | 5.08 | 35.99 | 0.201 |

Reading:
- `init` near-random is expected: it has no trained `pose_norm`, so it is not a
  usable standalone baseline — every run that resumed from it trained that
  layer from scratch.
- Epoch 5 / best / 10 differ by ≤0.5 chrF on 453 clips. No bootstrap intervals
  yet; treat these as indistinguishable, not as a ranking.
- Earlier logs said BLEU 3.24–3.56 for these epochs. That was 200 informburo
  clips with the old metric path; these are the first source-representative
  numbers. Still exploratory (initializer provenance, below).

### Decoding sweep (treat_best, dev selection only)

| | chrF | BLEU | WER |
|---|---|---|---|
| beam 1, all 6 settings | 34.69–34.96 | 4.78–5.06 | 0.913–0.921 |
| beam 4, all 6 settings | 36.00–36.27 | 5.37–5.54 | 0.898–0.902 |

- **Beam 4 over beam 1 is consistent** (~+1.3 chrF, +0.5 BLEU) across all settings.
- **Repetition penalty (1.0/1.1/1.3) and trigram blocking don't matter**: all
  beam-4 variants fall within 0.27 chrF of each other. The audit's concern that
  they suppress legitimate output is not supported here. Keep the current
  default (beam 4, rp 1.3, trigram block). Decoding is not a lever.

## E2 — pose representation repairs

All three are **off by default**; the default path was regression-checked as
bit-identical, so existing checkpoints and the E1 numbers reproduce. Each
changes encoder input, so **none can be combined with `--resume`** on a
checkpoint trained without it.

### `--block-padding-mask` — padding leaked into real frames
Every temporal ST-GCN block runs a kernel-5 conv; conv bias + BatchNorm shift
made padded frames nonzero, and the next block mixed them into the last real
frames. Masking only around the whole chain (the old code) didn't stop it.
- Reproduced with realistic BN statistics: **max diff 0.65** on real frames for
  the same clip at different padding (the audit's 0.024 used a fresh init).
- Fix: re-zero padding after the GCN unit and after each block
  (`STGCN_block.forward`, `STGCNChain.forward`). **Result: exactly 0.0** on the
  synthetic probe; 6e-8 (float noise) on a real clip with all flags on.
- Not fixed: in TRAINING mode BatchNorm statistics still count padded frames.

### `--real-wrists` — wrist nodes were elbows
`_map_coco_to_unisign_body` repeated elbow values into wrist slots in the
offset, absolute AND confidence channels; hand-anchor fusion read a pseudo-wrist.
- The true wrist-elbow bone can't be rebuilt inside the model (the absolute
  channel is standardized per clip on a different scale), so the dataset writes
  `wrist − elbow` (from GLOBAL-frame coords, valid under SignSpace too) into the
  hand-root offset slots, which were always zero. The mapper reads them for the
  body wrists, uses hand-root absolute position, sets wrist confidence to
  min(hand-root, elbow), then zeroes the root again so **the hand graph's input
  is unchanged** (verified tensor-equal).
- One flag sets dataset and encoder together — a mismatch would be silent.
- 14/14 checks pass (`scratchpad/test_wrists.py`).

### `--score-quantiles` — confidence channel was constant
Measured on train, all three sources: **99.6–100% of body/face/hand scores ≥1,
none ≤0.** `clip(score,0,1)` produced a channel with std **0.0000** — one unique
value. It encoded neither graded confidence nor detection, though raw scores vary
(hands p10 3.1, p90 7.8), and on a scale consistent across sources.
- Fix: `scripts/fit_score_quantiles.py` fits a per-group empirical CDF on
  **canonical train only** (450 clips) → `~/asan_canonical/score_quantiles.json`.
  Detected joints map to [0.05, 1], undetected stay 0. Real-data std 0.27.
- This is a **rank-normalized detector response, not a calibrated probability**
  — nothing is fitted against labelled keypoint correctness.
- `person_found` was never False in 17,462 sampled recollected frames; that part
  of the audit concern doesn't show up in the data.

### Not done in E2 (recorded, deliberately deferred)
- **Velocity/acceleration (564:1128) still unread by the mapper.** Using them is
  a new motion branch (E6), not a repair; dropping them changes input dims.
  `compute_velocity` also assumes 50 fps (qazaqstantv is 30).
- **Neck** is the average of the two shoulder *bone vectors*, not a coordinate.
- Face confidence is one mean over 88 points broadcast to 18 nodes.
- Upstream Uni-Sign preprocessing parity (arm B of the SignSpace experiment).

## Provenance

The colleague's mT5 checkpoint dates from **2026-07-16**; the recollected
manifests from **2026-08-31** — so it trained on archive-era data, not the
recollected splits. Sources confirmed: qazaqstantv, khabar, informburo.
**Open:** did they use the archive's own train split or re-split videos? If the
archive split, canonical dev/test are clean for it; until then results from this
initializer stay exploratory.

## Colleague's design (for E5/E6)

ST-GCN → Conv1D stride-2 temporal compression → (B,128,768) → mT5; loss CE +
VAP-lite visual–text contrastive; encoder frozen, only compression layer + mT5
trained. Maps directly onto audit E6 (temporal compression) and E5
(alignment). Note the fixed 128-token output and the frozen encoder — the latter
matters given E2: a frozen encoder can't adapt to repaired inputs.

## Next: E3

One run from `--pretrained-unisign`, three arms differing only in input
handling, all with the E2 repairs on, on `~/asan_canonical` with
`--selection-manifest`:
legacy normalization / upstream-Uni-Sign preprocessing / SignSpace.
