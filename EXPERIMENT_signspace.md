# Experiment: SignSpace pose normalization

**Status: UNBLOCKED and launched 2026-09-17 as E3 arm C — see `EXPERIMENT_e3_preprocessing.md`. E0 (canonical splits, frozen selection, metric parity) and E1 (re-scoring, decoding sweep) are complete; `NOTES_E0_evaluation.md`, `NOTES_E1_E2.md`.**

The code is on both machines behind `--signspace` (default off, verified
byte-identical to the old path across 200 randomized trials). What is missing
is a measurement setup capable of detecting the effect.

---

## Why this is blocked

The audit of 2026-09-17 (`REVIEW_architecture_research_2026-09-17.md`) found
two defects that would make this experiment unreadable, both since verified:

**1. Checkpoint selection never sees most of the data.**
`validate(..., max_gen_batches=25)` generates on the first 25 batches only.
`val_loader` is not shuffled and `AsanDataset` concatenates sources in list
order, so those 200 clips are **100% informburo** — confirmed directly. Of the
11,496-clip dev set, informburo is 1,241 clips (10.8%); qazaqstantv is 5,584
(48.6%) and is **never generated on**.

Every BLEU/WER we have recorded — including WER 0.919 / BLEU 3.90, and the
in-flight run's BLEU 3.33 — is a 200-clip informburo number.

A normalization change acts on hands and face across all sources. Measuring it
on one source's first 200 clips would be measuring almost nothing.

**2. Splits leak across dataset versions.**
Archive qazaqstantv TRAIN shares **563 video IDs with clean DEV (3,583 clips,
64.2% of dev)** and 533 with clean TEST (3,802 clips, 68.0%). Each manifest is
internally split-disjoint, which is what I checked earlier and why I wrongly
called it clean; the recollection re-split the same source videos.

Any arm initialized from a checkpoint that saw archive train is therefore
evaluating partly on videos it has already seen.

**Prerequisites (audit E0/E1):** one canonical source-video split across
dataset versions; a frozen selection manifest stratified by source and
duration; a single metric implementation (the trainer and
`scripts/evaluate_phase1.py` currently compute BLEU differently — normalized vs
raw); initializer provenance audited.

---

## Hypothesis

Normalizing hands and face **locally** removes signer hand-size and
camera-distance nuisance variation, giving the encoder a handshape
representation that is comparable across signers.

The current `normalize_signer_scale` divides every group by one global median
shoulder width. A synthetic check: a hand 2.25× larger produces hand features
2.25× larger under the current scheme, and identical features (diff 2.1e-06)
under SignSpace.

Prior evidence, How2Sign BLEU-4: **none 0.73 → frame-wise 1.13 → SignSpace
2.17** (arXiv:2507.01532), the largest single effect in their ablation. That
is a different language, corpus, and reference set — it justifies testing,
and forecasts nothing about KRSL.

**Not a dynamic-range fix.** The STGCN applies BatchNorm after its first
convolution, which largely absorbs a pure scale difference between groups. The
mechanism claimed here is nuisance-invariance only.

---

## What is implemented

Two functions, doing deliberately different jobs:

| function | applies to | effect |
|---|---|---|
| `signspace_global()` | **absolute** channel | every joint into the body-centred box (side = 3× shoulder distance). Hand **position** in signing space is preserved. |
| `signspace_normalize()` | **offset/bone** channel | body global, but face/hands each to their own per-frame box, aspect ratio preserved. Handshape becomes size- and distance-invariant. |

Splitting the channels is load-bearing. Local normalization discards a hand's
position, and the body graph cannot give it back: `_map_coco_to_unisign_body`
fills the wrist slots with repeated **elbow** values, so the encoder's
hand-anchor fusion reads a pseudo-wrist. Normalizing both channels locally
would erase signing location outright.

Ordering also matters: `remove_keypoint_spikes` uses a threshold in
shoulder-width units, so SignSpace runs *after* despiking, leaving
`despike_thresh=1.5` meaning what it always meant.

### Test coverage (`scratchpad/test_signspace.py`, 8 checks, all passing)

- body within [-1,1]; face/hands fill their own box
- invariant to signer scale + translation (2.3e-06)
- **invariant to hand size** (2.1e-06) vs 2.25× drift under the old scheme
- imputed joints stay exactly (0,0)
- degenerate input (all-zero, single frame, no shoulders) stays finite
- aspect ratio preserved — handshape not squashed
- `signspace_global` **does** move with hand position; `signspace_normalize`
  does **not** — the two channels verified complementary

---

## Design

Three arms, identical in everything but the normalization applied.

| arm | normalization |
|---|---|
| **A — legacy** | `normalize_signer_scale` (one global divisor), current behaviour |
| **B — upstream** | Uni-Sign's own part-wise crop/scale + confidence masks |
| **C — SignSpace** | `signspace_global` (absolute) + `signspace_normalize` (offset) |

Arm B is not optional garnish. Our weight loading copies pretrained x/y
projection weights into bone-offset columns, which preserves a linear function
only for equivalent inputs; upstream uses different coordinate meanings and
face node order. Without B we cannot tell "SignSpace helps" from "our
preprocessing was mismatched to the pretrained weights all along."

### Controls

- **`--resume` is forbidden.** All three arms change the input distribution,
  so every arm starts from `--pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth`
  (uploaded and md5-verified). This is also why arm B matters.
- Matched **update budget**, not matched epochs — clip counts differ.
- Identical data root, seed, effective batch, scheduler, decoding settings.
- Per-source metrics reported separately. A hands-and-face change should show
  up broadly; a gain in one source only is a red flag, not a win.
- Evaluation on the **full** corrected dev set for finalists, via the single
  shared metric implementation.

### Acceptance

Arm C beats arm A on full dev, **broadly across sources**, by a margin that
survives three seeds with paired bootstrap intervals clustered by source
video.

The often-quoted ~0.17 BLEU figure came from two accidentally duplicated runs.
That is a single paired observation, not an established noise floor — it is a
reason to run seeds, not a threshold to test against.

---

## Pre-committed readings

| Outcome | Reading |
|---|---|
| C > A and C > B | SignSpace helps on its own terms. Strongest result; make it the default and report the ablation. |
| C ≈ B > A | The win is "fixing preprocessing", not SignSpace specifically. Report honestly as a preprocessing-compatibility result. |
| B > C > A | Upstream-compatible preprocessing is the better target; SignSpace as implemented is a partial fix. |
| C ≈ A | Nuisance variation was not a limiter here. Cheap to keep behind a flag; do not report as a win. |
| C < A | Suspect the anchor split — check whether signing location degraded, using per-source and short/fast-sign breakdowns. |

---

## Follow-ups this surfaced (separate experiments, do not bundle)

- **Velocity/acceleration are computed and discarded.** `map_keypoints_to_unisign_format`
  reads `0:282`, `282:564`, `1128:1410` and never touches `564:1128` — 40% of
  the 1410-dim vector is padded, transferred to GPU, and dropped. Either drop
  the bandwidth or add a gated motion branch. `compute_velocity` also assumes
  50 fps; qazaqstantv is 30 fps before 2× downsampling.
- **Real wrists.** Wholebody indices 9/10 carry actual wrists; the body graph
  uses repeated elbows. Fixing this changes what hand-anchor fusion means, and
  interacts directly with the anchor split above.
- **Confidence saturates.** 99.85% of sampled hand scores are ≥1 before
  clipping to [0,1], so the channel is near-constant. Carry a separate
  detection-validity mask; the NPZ loader also ignores `person_found`.
- **Padding leaks into real frames.** Masking only around the temporal chain
  leaves internal blocks free to mix padding into boundary frames — measured
  max difference 0.024 for the same clip at different padding. Mask inside
  every time-mixing block.

## Command (once E0/E1 land — not before)

```bash
ASAN_ROOT=$HOME/asan_canonical PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. PYTHONUNBUFFERED=1 .venv/bin/python \
  train/train_encoder_mt5.py \
  --pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth \
  --use-enriched --signspace --epochs 10 --save-dir output/ss_armC
```
