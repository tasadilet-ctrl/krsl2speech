# E6: token-level pose–text alignment

Status: **complete (2026-09-23) -- 1/3 takeoff, same as baseline.** Results go in a section at the end;
nothing above it is edited after launch.

## Why

E5 aligned a mean-pooled pose embedding with a mean-pooled frozen text vector.
On all three seeds the alignment loss fell far below chance (5.58 → ~1.9), so
the encoder did learn to identify its transcript — yet none of the runs took
off (0/3, against 1/3 for the same config without alignment). E5's own
readings table named the untried levers: *weight, pooling, or a local
(token-level) objective*. E6 tests pooling.

## The one variable

Everything is E5's arm-B config — `--unisign-preprocess --block-padding-mask`,
10 epochs, canonical root, frozen 600-clip selection manifest,
`--align-weight 0.5`, 256 sampled negatives plus in-batch ones, the same two
256-d heads, the same learnable temperature, the same frozen `google/mt5-base`
teacher — except that **pooling is removed on both sides**
(`--align-mode token`).

For clip *b* and text *n* the score is the mean, over *n*'s content tokens, of
each token's best cosine against any valid frame of *b*. InfoNCE over the
batch's own texts plus the sampled negatives, pose→text only, duplicate texts
masked, all as in E5.

It is deliberately **not CTC** (already available as `--ctc-weight`): CTC
forces a monotonic alignment, and sign order does not follow Kazakh word
order. Here every word must be found *somewhere* in the clip, in any order.

Teacher states come from `scripts/cache_text_embeddings.py --per-token`: the
same encoder pass as E5's cache with the mean left out (EOS and padding
dropped). `--check-against` recomputes E5's pooled vectors from the same
states and requires them to match the existing cache.

## Checks done before launch

- 13 unit tests on the committed source: matches a brute-force loop
  implementation of the definition (max error 6e-8); invariant to chunk size,
  frame padding, token padding and frame order; a clip with no valid frame stays
  finite; gradient reaches both heads, the temperature and the pose embedding,
  and none reaches padded frames; a masked duplicate negative equals a
  removed one exactly; learns a separable toy task. Each of six deliberate
  breakages is caught.
- At real dimensions the untrained loss sits at chance: 5.577 ± 0.023 against
  ln(264) = 5.576. That is the line the smoke run must fall below.
- The per-token cache reproduces E5's teacher: recomputing E5's pooled
  vectors from the stored states gives max |difference| 4.9e-4 against the E5
  cache, i.e. fp16 rounding. 3,940,466 token rows, 58.4 per clip, 6.05 GB.
- Smoke run (this exact config, seed 0, stopped at batch 2,500): the align loss
  goes 5.559 / 5.503 / 5.446 / 5.397 / 5.354 at batches 500–2,500, below
  chance and falling, somewhat slower than E5's 5.580 / 5.444 / 5.294 at
  500–1,500. Step time 0.242 s/batch against E5's 0.241; memory 34.4 GB.
  The optimizer guard passes with the heads included.
- One unexpected smoke observation, recorded here so it can't be picked up
  after the fact: training CE (total loss minus 0.5·align) was **below every B
  and E5 run** at matched batches — 14.85 / 10.11 / 8.23 at 500 / 1,000 /
  1,500, against 16.18–16.86 / 11.15–12.30 / 8.94–10.40. That is not evidence
  of takeoff: B seed 0, which took off, and B seed 1, which didn't, were at
  8.94 and 8.99 at batch 1,500. Early training CE did not separate the modes
  before, so it is not a prediction now.
- E5's code path — `_alignment_loss`, `_align_batch` — and checkpointing,
  validation and generation are unchanged (AST-identical to the previous
  commit). The alignment heads are still train-only and not saved.

## Takeoff, defined numerically before any E6 result

Scored at epoch 10 by `scripts/rescore_checkpoints.py` on the canonical
selection set. Across the six runs scored so far (E3 B seeds, E5 seeds) the two
modes are far apart: chrF 29.8 vs ≤ 22.1, content recall 0.128 vs ≤ 0.037.
A run **takes off** if **chrF ≥ 26 and content recall ≥ 0.08**. The thresholds
sit in that gap and were set from those six runs, not from E6.

Baselines, same scorer: B 1/3 (chrF 29.79 / 21.02 / 19.83), E5 0/3
(21.70 / 22.10 / 20.95).

## Readings (pre-committed)

Each reading states the evidence it needs, because E5's table mapped 0/3 to
"interferes" and the check attached to it then showed no interference.

| Outcome | Requires | Reading |
|---|---|---|
| ≥ 2/3 take off | — | Pooling was hiding the signal. Adopt token-level alignment; confirm with three more seeds before building on it. |
| ≤ 1/3, align loss well below chance, mean Val CE ≤ 2.75 | CE no worse than the worst non-takeoff baseline (2.718) by more than noise | Pooling was not the problem. Two encoder-side objectives have now made the encoder informative without the decoder using it. Next: pressure on what the decoder reads — the mT5 encoder output it cross-attends to — rather than the pose embedding. |
| any count, mean Val CE > 2.75 | CE clearly worse than every non-takeoff baseline | Token-level alignment interferes with translation at weight 0.5; retry at a lower weight before drawing a conclusion. |
| align loss never clearly below chance | — | The objective did not engage. An implementation or temperature problem, not a result. |

Power, stated up front: at three seeds, 0/3, 1/3 and E5's 0/3 cannot be
told apart. Only ≥ 2/3, or chrF and recall shifting as a whole, counts as
evidence of an effect.

## Running

`run_e6.sh`: three seeds, one per free GPU (36 GB each), then rescoring all
nine checkpoints (B, E5, E6) through one metric path and writing
`output/e6_rescore/summary.txt`. The whole thing runs on the box; nothing
depends on a session staying open.

## Results (2026-09-23)

All three seeds finished (10 epochs each; seeds 0/1 in parallel, seed 2 after).
Re-scored on the canonical 600-clip selection set, beam 4, epoch 10, one metric
path. Contamination 0/600 for every checkpoint.

| run | Val CE | chrF | BLEU | content recall | final align (chance 5.577) | takeoff |
|---|---|---|---|---|---|---|
| E6 seed 0 | 2.671 | 21.39 | 0.47 | 0.034 | 1.52 | no |
| **E6 seed 1** | **2.300** | **29.31** | **2.60** | **0.123** | 1.18 | **yes** |
| E6 seed 2 | 2.701 | 20.59 | 0.21 | 0.033 | 1.56 | no |

**Takeoff 1/3.** Against baselines: B (no alignment) 1/3, E5 (sequence
alignment) 0/3. Pooled, the two alignment arms are 1/6 against the baseline's
1/3 — no evidence that either changes the takeoff rate, and at three seeds per
arm these counts cannot be told apart anyway.

**Which pre-committed row fires:** "≤ 1/3, align loss well below chance, mean
Val CE ≤ 2.75", with its condition met. Alignment fell to 1.18–1.56 against a
chance level of 5.577, so the objective engaged hard. Mean Val CE 2.557, and
both non-takeoff seeds (2.671, 2.701) are better than the worst non-takeoff
baseline (2.718), so there is no interference.

Reading, as written before the run: **pooling was not the problem.** Two
encoder-side objectives have now driven the alignment loss far below chance
while leaving the takeoff rate unchanged. The encoder is made informative; the
decoder still does not use it.

**Takeoff lands in the same place regardless of objective.** The three runs that
took off across all of E3/E5/E6 sit at chrF 29.79 (B s0), 29.31 (E6 s1) — with
content recall 0.128 and 0.123. Non-takeoff runs sit at 19.8–22.1 with recall
0.025–0.040. Nothing tried so far moves either mode; only which mode a run lands
in, and that has looked like chance throughout.

### Next

Per the fired row: pressure on what the decoder reads, not on the pose
embedding. One untested lever is cheaper and more basic — **E4, the
learning-rate balance**. mT5 trains at 5e-4 while the encoder trains at 5e-5, a
10x gap, and the audit flagged it on 2026-09-17 as a hypothesis never screened.
A decoder that adapts ten times faster than its encoder is a plausible
mechanism for "decoder fits the language prior before the encoder is worth
reading", which is exactly the failure both alignment arms left untouched.
Screen it by takeoff rate over three seeds, same as here.
