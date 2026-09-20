# E5: pose–text contrastive alignment

Launched 2026-09-20, tmux `krsl` window `e5`, `run_e5.sh`.
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
