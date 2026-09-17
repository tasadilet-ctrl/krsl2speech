# Experiment: does the clean qazaqstantv data improve translation quality?

> **INVALIDATED 2026-09-17 — see "Why this cannot answer the question" below.**
> The control arm was cancelled before it started. The treatment arm was left
> running because its weights are still useful, but its reported metrics are
> not what this document originally claimed.

Launched 2026-09-17 on the HPC box. Two arms, identical in every respect
except **which qazaqstantv training clips they see**.

## Design

| | control | treatment |
|---|---|---|
| `ASAN_ROOT` | `~/asan_archive_cleaneval` | `~/asan_clean` |
| qazaqstantv train | archive (corrupted) | recollection (clean) |
| khabar / informburo train | archive | archive (identical symlinks) |
| **train clips** | 40,240 | **52,835** |
| **val clips** | 11,496 | 11,496 |
| save dir | `output/clean_control` | `output/clean_treat` |

Both resume from `output/ours_enriched_friend_mt5.pth` (`epoch: -1`, no
optimizer state, so both start fresh at epoch 1 with identical init),
`--use-enriched --epochs 10`, selection metric `wer` (the default).

**The validation set is byte-identical across arms** — verified by hashing
the sorted clip_id list: `95d1ac5d8165` for both. This is the whole point of
the `asan_archive_cleaneval` root: the archive's own dev/test carry the same
12.9% corruption, so measuring on them would be measuring with a broken ruler.
Here the ruler is the same clean instrument in both arms.

## Why this is needed at all

The recorded WER 0.919 / BLEU 3.90 were measured on the archive's 8,795-clip
test set. The clean test set has 11,448 clips. Those numbers are **not
comparable**, so "clean data" cannot be evaluated by simply re-running and
comparing against them — hence a matched control trained on the old data and
scored on the new instrument.

## Execution note

The two arms run **sequentially on GPU 1**, not in parallel. GPU 0 had
~34.8 GB free (another user holds ~60 GB at 0% utilization) and the run needs
~33.3 GB plus CUDA context; the control OOM'd there on a 24 MB allocation.

Shrinking the control's batch size to fit was rejected: the STGCN uses
`BatchNorm2d`, so a different per-step batch changes BN statistics and would
confound the comparison with exactly the sort of implementation detail that
Mercanoglu Sincan et al. (arXiv:2603.13240) show dominates apparent gains.
The control is queued in tmux session `control`, waiting on the treatment
PID, and starts automatically.

Throughput: ~4.5 batches/s, 6,605 batches/epoch (treatment) → ~25 min/epoch.
Estimated ~6–9 h per arm including generation-based validation.

## Reading the result

The measured noise floor on this setup is **~0.002 WER / ~0.17 BLEU** from
run-to-run nondeterminism alone. A gap smaller than that is not a result.
If the gap is close to the floor, run a second seed per arm before claiming
anything.

| Outcome | Reading |
|---|---|
| treatment clearly better | data quality was a real limiter; report the delta and re-baseline everything on clean data |
| flat (within noise) | the 12.9% corruption was not what held the model back — worth reporting, and it makes the pose-normalization work the more promising lever |
| treatment worse | investigate before believing it; most likely a segmentation difference (clip lengths/boundaries changed) rather than label quality |

## Monitoring

```bash
tmux attach -t treat      # or: tail -f ~/krsl2speech/output/clean_treat.log
tmux attach -t control
```


---

## Why this cannot answer the question

Two defects, both verified directly after the audit
(`REVIEW_architecture_research_2026-09-17.md`):

**1. The selection metric never sees qazaqstantv.**
`validate(..., max_gen_batches=25)` generates on the first 25 batches only;
`val_loader` is unshuffled and sources are concatenated in list order, so those
200 clips are 100% informburo. qazaqstantv is 5,584 of the 11,496 dev clips and
is never generated on. Every `WER:`/`BLEU:` line in both logs is a 200-clip
informburo number — it cannot see the variable this experiment changes.

**2. The control arm was contaminated.**
Archive qazaqstantv TRAIN shares 563 video IDs with clean DEV (3,583 clips,
64.2% of dev) and 533 with clean TEST (3,802 clips, 68.0%). I verified each
manifest was internally split-disjoint and wrongly concluded the comparison was
clean; the recollection re-split the same source videos, so the control would
have been evaluated largely on videos it trained on — biasing the result
*against* the treatment arm.

**Also corrected:** the "~33% of training data was dirty" framing was wrong.
qazaqstantv is ~33% of training clips, and 12.9% *of those* were corrupted —
roughly 4.3% of the training corpus. And the byte-identical claim for the val
sets was overstated: hashing sorted clip IDs establishes ID membership, not
identical references, poses, or preprocessing.

## What was kept

The treatment arm (`output/clean_treat/`) finishes on clean data and saves
periodic checkpoints. Those weights are still a useful starting point and can
be **re-scored** on a corrected dev set without retraining. Nothing about the
training is wrong; only the measurement is.

## What replaces this

Audit stages E0/E1 first: one canonical source-video split across dataset
versions, a frozen selection manifest stratified by source and duration, one
metric implementation, and audited initializer provenance. Then re-score
existing checkpoints before spending further GPU time. See
`EXPERIMENT_signspace.md` for the same prerequisites.
