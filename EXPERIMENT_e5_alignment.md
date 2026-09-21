# E5: pose–text contrastive alignment

Launched 2026-09-20, tmux `krsl` window `e5`, `run_e5.sh`. Finished 2026-09-21;
**result: 0/3 takeoff** (see Results — read it with the caveat there).
Logs: `output/e5_align_seed{0,1,2}/train.log`, scheduler `output/e5_scheduler.log`.

## Why

E3 established that training here is bimodal: a run either starts using the pose
input ("takeoff", chrF ~30, 598/600 distinct outputs) or regresses to the corpus's
most frequent sentence (chrF 17–23). **Takeoff happened in 1 of 8 runs**, and three
seeds of one config spanned 19.8–29.8 chrF. Cross-entropy alone never requires the
encoder to carry information, so nothing makes takeoff reliable — a longer schedule
(B20) just overfit the text prior.

This adds a term that is only minimised by pose representations that identify their
own transcript.

## Design

Arm-B config (upstream Uni-Sign preprocessing, `--block-padding-mask`, 10 epochs,
canonical root, frozen 600-clip selection manifest) **plus** `--align-weight 0.5`.
Baseline is exactly that config without alignment: the B seeds, takeoff 1/3.

- **Loss:** InfoNCE from masked-mean-pooled pose embedding → 256-d shared space,
  against the clip's frozen text vector. Pose→text only; the text side is frozen,
  so the symmetric direction would train the text head against itself.
- **Text teacher:** base mT5 encoder, mean over non-pad tokens, computed once for
  all 67,465 train+dev clips (`scripts/cache_text_embeddings.py`). Frozen and
  cached, so there is no second mT5 on the GPU (~36 GB free) and no drifting target.
  Base mT5 has never seen KRSL, so this adds no leakage.
- **Negatives:** 256 sampled from the cached table plus the 7 in-batch ones. With
  batch size 8 an in-batch loss gives only 7, and gradient accumulation adds none.
  Sampled negatives whose normalized text equals the positive's are masked out
  (1.3% of cached texts repeat).
- **Seeds 0, 1, 2**, otherwise identical. Sequential on GPU 1 (~5.7 h each); other
  users hold GPU 0.

**Judge by takeoff rate across seeds, not one run's chrF.** A single good run is
exactly what E3 showed to be uninformative.

## Teacher sanity check

Cached vectors are distinguishable per clip: mean pairwise cosine 0.72 (not ~1.0),
effective rank 90.5 of 768 dims. Note the "100% self-retrieval among 256
candidates" check I ran first was trivially satisfied — it compared each vector
with itself and proves nothing.

## Two bugs found before this ran (both would have produced a plausible null)

1. **Alignment heads were in no optimizer group.** Parameter groups here are
   explicit, so the new heads never trained; the loss then pushed the encoder
   toward *random frozen* projections of the text and sat at chance (5.675 vs
   ln(264)=5.576) for 1,000 batches.
2. **The guard I added for bug 1 caused a worse one.** Several groups are appended
   as generators (`core.mt5.parameters()`); iterating them to check coverage
   *consumed* them, so AdamW got empty lists and only the encoder trained. Caught
   by comparing the loss curve against B's: CE stuck at ~23 and rising where B fell
   to 5.9 by batch 4000. Fixed by materialising groups to lists first; the startup
   check now also asserts optimizer params == model trainable params and logs both.

Healthy restart, CE with the alignment term subtracted vs B at the same batches:
16.7/11.3/9.0 vs 16.4/11.1/8.9 — alignment does not disturb translation training —
while align falls 5.580 → 5.445 → 5.294, below chance.

## Readings (pre-committed)

| Outcome | Reading |
|---|---|
| 3/3 seeds take off | Alignment fixes the bimodality. Adopt it, then redo the preprocessing arms with seeds. |
| 2/3 | Promising but not solved; try a higher weight or local alignment before building on it. |
| 1/3 (= baseline) | No effect on the failure mode; the align loss can still fall while the decoder ignores the encoder. Look at weight, pooling, or a local (token-level) objective. |
| 0/3 | Alignment at this weight actively interferes; check whether CE degraded versus the B seeds. |

Note the alignment heads are train-only and deliberately not saved in checkpoints;
inference is unchanged.

## Results (2026-09-22)

All six checkpoints at **epoch 10**, re-scored together on the canonical
600-clip selection set (200 per source) through one metric path, beam 4.
Epoch 10 rather than `best`: `best` is chosen by WER, which is ~1.00 for every
run that doesn't take off, so it would add noise to the selection.
All six were trained with exactly `--unisign-preprocess --block-padding-mask`.
Contamination was 0 for every checkpoint, measured against the validated
canonical train split (10,398 training videos).

| checkpoint | chrF | BLEU | WER | distinct outputs | top-1 share | content recall | takeoff |
|---|---|---|---|---|---|---|---|
| B seed 0 | **29.79** | 2.63 | 0.962 | **598/600** | 0.3% | **0.128** | **yes** |
| B seed 1 | 21.02 | 0.42 | 1.018 | 503/600 | 5.3% | 0.031 | no |
| B seed 2 | 19.83 | 0.27 | 1.013 | 523/600 | 2.3% | 0.025 | no |
| E5 seed 0 | 21.70 | 0.26 | 1.037 | 566/600 | 1.0% | 0.029 | no |
| E5 seed 1 | 22.10 | 0.37 | 1.021 | 586/600 | 1.3% | 0.037 | no |
| E5 seed 2 | 20.95 | 0.27 | 1.032 | 521/600 | 1.8% | 0.025 | no |

Training-log values at epoch 10:

| seed | Val CE | align (chance 5.58) |
|---|---|---|
| E5 0 | 2.688 | 1.96 |
| E5 1 | 2.655 | 1.84 |
| E5 2 | 2.713 | 1.99 |
| B 1 / B 2 (no takeoff) | 2.691 / 2.718 | — |
| B 0 (takeoff) | 2.287 | — |

**Count: 0/3 takeoff, against a 1/3 baseline.** At three seeds per arm, 0/3
and 1/3 can't be told apart statistically.

**The table's 0/3 reading does not hold up against its own check.** It says
"actively interferes; check whether CE degraded versus the B seeds." On every
measure available, E5 is at least as good as B's two runs that didn't take off:
CE (mean 2.685 vs 2.704), chrF (mean 21.6 vs 20.4), and distinct outputs
(521–586 vs 503–523). Nothing shows interference.

The 1/3 row's wording describes what happened: **the align loss fell while the
decoder ignored the encoder.** Alignment fell far below chance on all three
seeds, so the pose embeddings did come to identify their own transcripts.
Translation still never took off. That separates "the encoder carries
information about the clip" from "the decoder uses it", and suggests the
bottleneck is the second.

**Content recall is the cleanest separator**: 0.128 for the one run that took
off, and 0.025–0.037 for every other run with or without alignment. E5 made
outputs somewhat more varied without making them more correct.

**The failure mode is not collapse to a single sentence.** No non-takeoff run
has one output above 5.3%. They cycle through a few stock news-broadcast
phrases ("now to the main world news", "my colleague reports the details"),
with the top five covering 3–12% of outputs. The "Why" section above describes
this as regression to the corpus's most frequent sentence, which is close but
not exact.

### Next, per the 1/3 row

Weight, pooling, or a local (token-level) objective. The result argues for
putting pressure on the decoder rather than on the encoder: the encoder is
already informative under this objective and the decoder still doesn't use it.

### Reproducing

Takes about 40 s per checkpoint:

```bash
C=$HOME/asan_canonical
PYTHONPATH=. python scripts/rescore_checkpoints.py \
  --root $C --unisign-preprocess --block-padding-mask --out output/e5_rescore \
  --ckpt output/e3_armB_upstream/phase1_mt5_epoch10.pth=B_s0:$C \
  --ckpt output/e3_armS1_upstream_seed1/phase1_mt5_epoch10.pth=B_s1:$C \
  --ckpt output/e3_armS2_upstream_seed2/phase1_mt5_epoch10.pth=B_s2:$C \
  --ckpt output/e5_align_seed0/phase1_mt5_epoch10.pth=E5_s0:$C \
  --ckpt output/e5_align_seed1/phase1_mt5_epoch10.pth=E5_s1:$C \
  --ckpt output/e5_align_seed2/phase1_mt5_epoch10.pth=E5_s2:$C
```

Distinct outputs and top-1/top-5 share are counted from the `hyp` field of each
`output/e5_rescore/<label>.predictions.jsonl`. Every per-clip prediction is kept
there.

