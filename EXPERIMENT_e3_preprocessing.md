# E3: input-handling ablation (legacy / upstream Uni-Sign / SignSpace)

Launched 2026-09-17, tmux `krsl` window `e3`, scheduler `run_e3.sh`.
Logs: `output/e3_arm{A,C,B}_*/train.log`, scheduler `output/e3_scheduler.log`.

## Arms

All three share: `ASAN_ROOT=~/asan_canonical`, the frozen 600-clip selection
manifest, `--pretrained-unisign csl_stage1_weight.pth`, `--block-padding-mask`,
`--epochs 10`, `--seed 0`, identical batch/schedule/decoding. They differ
**only in spatial preprocessing**.

| arm | flags | input |
|---|---|---|
| **A legacy** | `--use-enriched --real-wrists --score-quantiles` | 1410-dim: bone offsets + per-clip standardized absolutes + velocity/accel (unread) + confidence |
| **C signspace** | A + `--signspace` | as A, but body in a 3×shoulder box and hands/face normalized locally |
| **B upstream** | `--unisign-preprocess` | 207-dim: exact upstream Uni-Sign port — 9-joint body with real wrists, clip-level body box, wrist-relative hands, nose-relative face, (x, y, score), score ≤ 0.3 masked |

**A vs C isolates normalization.** B is a package, not a controlled contrast:
it changes normalization, node selection, channel layout and score handling at
once. Its job is to answer a different question — whether our preprocessing was
mismatched to the pretrained weights all along — so a B win must not be read as
"SignSpace loses".

## Why this initializer

Uni-Sign CSL weights + **base** mT5. Neither has seen KRSL, so provenance is
fully documented and disjoint from canonical dev/test — unlike
`ours_enriched_friend_mt5.pth`, whose mT5 history is unrecorded. Absolute
scores will likely start below `clean_treat` (which began from a KRSL-tuned
mT5); that is the price of a defensible comparison. E1 scores are not
comparable to these.

## Arm B fidelity

`data/utils.py::unisign_part_features` is a port of upstream `load_part_kp` /
`crop_scale`. Verified **identical to upstream's own function on 80 randomized
cases** (realistic scores, mixed, mostly-below-threshold, single frame) by
executing the functions straight out of the official `datasets.py`.

Upstream oddities reproduced deliberately, not fixed:
- `np.clip(result, -1, 1)` spans the whole array, so **upstream's score channel
  is capped at 1 too**. Since raw RTMPose scores are ≥1 for ~100% of joints,
  upstream's confidence channel is as saturated as ours was before E2. Arm B
  therefore does NOT get the E2 score fix — that is what upstream is.
- Body box is clip-level (one box for all frames), not per-frame.
- Hand coordinates are taken relative to the hand root even when that root is
  undetected.

Arm B also loads the pretrained projection **natively** (3 channels, no
5-channel adaptation), which A and C cannot do.

## Pre-committed readings

| Outcome | Reading |
|---|---|
| C > A, C > B | SignSpace helps on its own terms. Make it default; report the ablation. |
| B > A and B ≈/> C | Our preprocessing was mismatched to the pretrained weights. The headline is compatibility, not normalization. |
| C > A, B lowest | Normalization is the lever and our richer input beats upstream's. Strongest case for the current architecture. |
| A ≈ C | Nuisance variation was not a limiter here. Keep SignSpace behind a flag; do not report a win. |
| All ≈ equal | Input handling is not where the headroom is. Move to E5 (alignment). |

Margins must be read per source and against seeds. Single-seed gaps of a few
tenths of chrF on 600 clips mean nothing; finalists get 3 seeds (E8).

## Scheduler

Starts each arm when a GPU has ≥36 GB free, one arm per GPU, waiting for the
first training batch before starting the next (simultaneous CUDA inits have
stalled on this box). Both GPUs currently carry ~60–72 GB from other users, so
the arms will most likely run sequentially on GPU 1 at roughly 5 h each. If
GPU 0 frees, the queue uses it automatically. Three attempts per arm before it
is dropped.

## Next

Re-score all three arms with `scripts/rescore_checkpoints.py` on the canonical
selection set (arm B needs `--unisign-preprocess`), compare on the common clean
subset, then decide E4/E5.
