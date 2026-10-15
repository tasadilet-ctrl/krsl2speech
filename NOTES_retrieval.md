# Sign -> text retrieval already works (2026-10-08)

**Held-out result (E12, 2026-10-13, test split, B_s0, run once):** cross-broadcast
200-way top-1 0.807 (informburo 0.835, khabar 0.885, qazaqstantv 0.700); within-
broadcast alignment top-1 0.825 vs 0.161 chance, MRR 0.882. Tool:
`scripts/sign_retrieve.py`. Details: EXPERIMENT_e12_retrieval_task.md.

`scripts/eval_retrieval.py`, outputs in `output/retrieval/`. No training: an
existing translation checkpoint is used as a scorer.

## Method

For every clip in the frozen selection set, score all 200 references of the
**same source** by the checkpoint's own teacher-forced likelihood
log p(text | pose), using the training prefix and tokenisation exactly.
Same-source only, because qazaqstantv text is cased and punctuated and the
others are not; mixing sources would let a model "retrieve" by spotting style.

Two scores:

- **raw** `log p(t | x)` -- dominated by how likely the text is in general.
- **PMI** `log p(t | x) - log p(t | null)`, where null is the mean over 3 other
  clips' poses from the same source. This cancels the text prior, which is the
  thing that has dominated generation throughout this project.

## Result (B_s0, a takeoff checkpoint, epoch 10)

| source | score | top-1 | top-5 | top-10 | MRR | median rank |
|---|---|---|---|---|---|---|
| informburo | raw | 0.010 | 0.100 | 0.180 | 0.064 | 54 |
| informburo | **PMI** | **0.855** | **0.935** | 0.955 | 0.894 | **1** |
| khabar | raw | 0.015 | 0.215 | 0.320 | 0.102 | 31 |
| khabar | **PMI** | **0.865** | **0.965** | 0.970 | 0.906 | **1** |
| chance (200-way) | | 0.005 | 0.025 | 0.050 | 0.029 | 100 |

**The same model that writes the right sentence almost never (WER ~0.96)
picks it out of 200 candidates 86% of the time.** All of the signal appears
only after the text prior is subtracted: raw likelihood is near chance.

## Controls run

1. **Duration.** Clip length predicts transcript length (r = 0.63). Ranking
   candidates by length match alone, with the length model fitted on the test
   clips themselves (optimistic), gives top-1 0.010, median rank 47. Duration
   does not explain 0.855.
2. **Re-aired sentences.** Retrieval could be recognition of a memorised
   training example if dev sentences re-air training sentences. Nearest
   training sentence by content-word Jaccard: median 0.09 (informburo) / 0.12
   (khabar); 1 of 400 queries at >= 0.5. No memorisation path. (Video-level
   contamination was already 0.)
3. **Non-takeoff checkpoint.** If retrieval reflects reading the signs, a
   checkpoint that barely uses its encoder should fail at it:

   | checkpoint | coupling delta | PMI top-1 | top-5 | median rank |
   |---|---|---|---|---|
   | B_s0 (took off) | 9.52 | **0.855** | 0.935 | **1** |
   | B_s1 (did not) | 1.27 | 0.015 | 0.155 | 25 |
   | chance | -- | 0.005 | 0.025 | 100 |

   It does. B_s1 is above chance (top-5 0.155 against 0.025), matching its
   small but significant coupling, and nowhere near B_s0. **Retrieval tracks
   coupling, and it needs a run that took off.** Informburo only (200-way).

## Why this matters

Generation has to get every token right from weak, partial evidence and
loses to the text prior. Retrieval only has to rank, and a handful of
correctly read signs -- a name, a number, a place -- is enough to single out
one sentence. The positional analysis (NOTES_positional_recall_and_lag.md)
and the 31-51 px hands (NOTES_input_resolution.md) both say the pose carries
partial information; retrieval is the task shape that partial information
can support.

Practical forms: matching a clip to the right line of a known script or
broadcast transcript, sign-video search, keyword/topic spotting, or scoring
an n-best list from a language model.

## Caveats

- **Replicated on a second takeoff checkpoint from a different arm:** E4b_s1,
  informburo, PMI top-1 0.850, top-5 0.935, MRR 0.890, median rank 1 (raw
  top-1 0.010). Against B_s0's 0.855 / 0.935 / 0.894 / 1.
- Retrieval needs a run that took off (control 3), so it inherits the ~1/3
  lottery -- another reason to start future work from a coupled checkpoint
  (`--init-from`) rather than from scratch.
- 200-way within one source. Larger pools are harder; the scores in
  `*.scores.npy` allow re-ranking at any pool size without re-running.
- PMI's null uses 3 poses; more would lower its variance.
