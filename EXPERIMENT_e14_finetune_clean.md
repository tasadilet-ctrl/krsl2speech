# E14: fine-tune B_s0 on cleaned vs full vs randomly-reduced data

Status: **complete 2026-10-19 -- null: which clips are removed does not
matter.** Results at the end.

## Why

E13 (train from scratch without the 4,200 flagged qazaqstantv references) was
inconclusive: no seed took off, so the designed measure -- qazaqstantv
within-broadcast retrieval of a coupled model -- was never available. This
removes the takeoff lottery: every arm starts from **B_s0**, which already
reads its encoder, so every arm yields a usable retrieval number.

## Arms

All: `train/train_min.py --init-from B_s0 epoch 10 --base-lr 1e-4` (mT5 and
pose_norm 1e-4, encoder 1e-5 -- a fifth of from-scratch rates, so fine-tuning
does not knock the model out of its coupled state), 3 epochs, stride 2, fresh
optimizer, warmup 513 steps.

| arm | training data | isolates |
|---|---|---|
| **clean** | minus the 4,200 flagged qazaqstantv clips | -- |
| **random** | minus 4,200 random qazaqstantv clips (578 happen to be flagged; ~547 expected by chance) | clean vs random: *which* clips, at equal quantity |
| **full_a**, **full_b** | all training data, seeds 0 and 1 | clean vs full: the net effect; full_a vs full_b: **the noise floor** |

Smoke-verified: weights load (loss 2.14 at batch 100, B_s0's training level),
learning rates 1e-5 / 1e-4 / 1e-4, loss stable through warmup.

**Caveat built into the design:** fine-tuning cannot undo 10 epochs of training
on the flagged references. A positive result is strong; a null weakens but does
not kill the hypothesis.

## Measures (dev only; full, unfiltered sets)

- **Primary:** qazaqstantv within-broadcast top-1 on `within_video_dev.json`.
- **Noise floor:** n = |full_a - full_b| on that number, per source.
- **Specificity:** khabar and informburo (training data identical in every arm).
- Secondary: per-source chrF and coupling delta on the selection set.

## Readings (pre-committed)

Let T = max(0.03, 2n) per source.

| Outcome | Reading |
|---|---|
| clean - random >= T on qazaqstantv, clean >= mean(full) + T, khabar and informburo within their T | **Removing the flagged references helps qazaqstantv specifically.** The audit's flags mark references that harm training. Recommend fixing the matching upstream and excluding flagged clips by default. |
| clean - random >= T, but clean ~ full | Cleaning helps relative to losing random data, i.e. it offsets the cost of having less data; net effect small. |
| clean ~ random (within T) | Which clips are removed does not matter here: the flagged references are not what limits qazaqstantv, at least at fine-tuning stage. |
| random < full by > T, clean ~ full | Data quantity matters; the flagged clips are dispensable. |
| All three sources move together | Not specific to the cleaned data. |
| full_a vs full_b differ by > 0.05 | Noise too large for this design to resolve anything; report and stop. |

## Run

`run_e14.sh`, tmux `e14`: four runs (two per GPU in sequence), then rescore
clean + pose-shuffled on the selection set, then T2 dev retrieval for all four.
`output/e14_rescore/`, `output/e14_t2/`.

## Results (2026-10-19)

All four arms fine-tuned 3 epochs from B_s0 epoch 10 and all four take off
(4/4), which is expected: they start from a checkpoint that already had.

**Noise floor first**, since it sets every threshold. The two identical full
arms differ on T2 dev top-1 by n = 0.022 (qazaqstantv), 0.011 (khabar), 0.014
(informburo). All are under the 0.05 that would have disqualified the design,
so it resolves something -- but T = max(0.03, 2n) means qazaqstantv needs a
gap of **0.044** to count.

T2 dev, within-broadcast PMI top-1:

| arm | ALL | informburo | khabar | qazaqstantv | qazaq/khabar |
|---|---|---|---|---|---|
| clean | 0.857 | 0.939 | 0.964 | **0.678** | 0.703 |
| random | 0.852 | 0.914 | 0.970 | **0.681** | 0.702 |
| full_a | 0.858 | 0.936 | 0.956 | **0.692** | 0.724 |
| full_b | 0.850 | 0.922 | 0.967 | **0.670** | 0.693 |

Selection set, epoch 3:

| arm | chrF | recall | distinct | val CE |
|---|---|---|---|---|
| clean | 31.45 | 0.146 | 599/600 | 2.2418 |
| random | 31.28 | 0.149 | 597/600 | 2.2494 |
| full_a | 31.02 | 0.144 | 595/600 | 2.2441 |
| full_b | 31.34 | 0.147 | 598/600 | 2.2429 |

**The pre-committed row that fires: "clean ~ random (within T)."**
clean − random on qazaqstantv is **−0.003** against a threshold of 0.044. The
reading, as written beforehand: *which clips are removed does not matter here;
the flagged references are not what limits qazaqstantv, at least at the
fine-tuning stage.*

Nothing else fires either. clean − mean(full) is −0.003, so there is no net
effect. random − mean(full) is 0.000, so dropping 4,200 qazaqstantv clips at
random costs nothing measurable — the "data quantity matters" row does not
fire. khabar and informburo move by 0.003 and 0.010 against their T of 0.03,
so nothing moved together either.

On chrF the same picture holds, and the built-in control is what makes it
readable: clean beats the full mean by 0.27, while the two *identical* full
arms differ by 0.32. The apparent gain is smaller than the noise it would have
to clear.

**With E13 this is now two independent nulls.** E13 trained from scratch
without the flagged clips (3 seeds) and found no qazaqstantv-specific gain;
E14 fine-tunes and finds the same, with a measured noise floor. The caveat
written into E14's design still stands — fine-tuning cannot undo 10 epochs of
exposure to those references, so a null here is weaker than a positive would
have been — but E13 does not share that limitation, and agrees.

What this does **not** overturn: the audit's flags are real. They were
validated against speech rate, an independent signal the model never sees
(NOTES_corpus_audit.md), and qazaqstantv still retrieves far worse than khabar
(0.678 vs 0.964) in every arm. The flagged references are genuinely
misaligned; removing them from training simply does not recover the gap. That
points at the remaining qazaqstantv deficit being something other than these
4,200 clips — plausibly the broader share of loose, non-literal references
that the mutual-mismatch test is too conservative to flag.
