# E13: train without the flagged qazaqstantv references

Status: **pre-registered 2026-10-16, launching.** Results go at the end.

## Why

The corpus audit (NOTES_corpus_audit.md) flags 13.1% of qazaqstantv training
clips (4,200) as references that fit another clip of the same broadcast better
than their own, against 0.4-1.0% for khabar/informburo. The flags are validated
(speech-rate outliers flagged x9.8) and trace to the article-text matching step
(human-text clips flagged x2.8 vs speech-recognition text, robust to
stratification). So far that is observation. This tests it: **does removing
those references improve qazaqstantv?**

## Design

`train/train_min.py`, B config (stride 2, upstream Uni-Sign preprocessing,
padding mask, base mT5, baseline LRs, 10 epochs), three seeds, with
`--exclude-clips ~/asan_canonical/clean_exclude_qz_v1.json` (the 4,200 flagged
qazaqstantv training clips; nothing else changes). Baseline: B's three seeds
(B_s0 took off).

Smoke-verified: exactly 4,200 dropped of 59,008; 6,851 batches/epoch vs 7,376;
17,130 optimizer steps vs 18,440 (-7%); early loss tracks B.

**Judged on DEV only.** Test was used once for E12; reusing it to decide E13
would erode it.

- **Primary:** qazaqstantv within-broadcast retrieval (T2 dev,
  `within_video_dev.json`) for any seed that takes off, against B_s0's 0.670.
  Noise reference: three takeoff checkpoints from different arms scored
  0.670 / 0.684 / 0.684 on this set (spread ~0.015).
- **Specificity control:** khabar and informburo -- their training data is
  untouched, so a qazaqstantv-specific effect should leave them flat.
- **Secondary:** per-source chrF and coupling delta on the frozen selection
  set; takeoff rate (reported, not decisive at 3 seeds).
- **Full** dev sets only -- never flag-filtered. The flags came from B_s0, so a
  filtered evaluation would favour the flagging model.

## The confound, and the control held in reserve

Cleaning also removes 7% of the training data. Less data should not help, so a
**gain** is conservative and stands on its own. A **null or a loss** is
ambiguous (cleaning may help while less data hurts), and would be resolved by a
control arm that removes 4,200 RANDOM qazaqstantv training clips. Pre-
registered here; run only if the result needs it.

## Readings (pre-committed)

| Outcome | Reading |
|---|---|
| A takeoff seed reaches qazaqstantv T2 top-1 >= 0.73 (+0.06 over B_s0, 4x the checkpoint spread) with khabar and informburo within +/-0.03 | **Cleaning helps qazaqstantv specifically.** Misaligned references were costing it. Recommend fixing the matching step upstream and excluding flagged clips by default. |
| A takeoff seed, qazaqstantv within +/-0.03 of 0.670 | No measurable gain. Run the random-removal control before concluding. |
| All three sources move together | Not specific to the cleaned data; likely seed or global variance. |
| qazaqstantv worse | Losing the data cost more than cleaning gained; run the control. |
| 0/3 takeoff | Retrieval is unusable without takeoff; compare qazaqstantv per-source chrF and coupling against B's non-takeoff seeds only, and report the comparison as weak. |

A single takeoff seed compared with B_s0 is one run against one run; the
threshold is set at 4x the observed checkpoint spread for that reason.

## Run

`run_e13.sh`, tmux `e13`: three seeds (one per free GPU), then rescore clean +
pose-shuffled on the selection set, then T2 dev retrieval for all seeds.
Summary: `output/e13_rescore/summary.txt`, `output/e13_t2/`. Epoch-5 screen
(report-only): `output/screen_e13.log`.

## Results (2026-10-16)

All three seeds completed 10 epochs (exactly 4,200 clips excluded each);
contamination 0. Selection set, epoch 10:

| run | chrF | recall | distinct | coupling delta (95% CI) | val CE | takeoff |
|---|---|---|---|---|---|---|
| E13_s0 | 24.96 | 0.068 | 595/600 | 4.83 [4.25, 5.42] | 2.522 | no (partial) |
| E13_s1 | 24.42 | 0.061 | 588/600 | 3.85 [3.28, 4.42] | 2.579 | no (partial) |
| E13_s2 | 19.36 | 0.019 | 347/600 | 0.26 [-0.27, 0.79] | 2.741 | no |
| B_s0 / B_s1 / B_s2 | 29.79 / 21.02 / 19.83 | | | 9.52 / 1.27 / 1.11 | 2.287 / 2.691 / 2.718 | 1/3 |

**Pre-registered row: "0/3 takeoff".** The primary measure (qazaqstantv T2 of a
takeoff seed vs B_s0's 0.670) cannot be applied. s0 and s1 are partial takeoffs
-- coupling 3.9-4.8, between B's failures and its takeoff, like E4b_s5 (3.74).

T2 dev retrieval (within-broadcast top-1):

| run | all | informburo | khabar | qazaqstantv | qazaq / khabar |
|---|---|---|---|---|---|
| E13_s0 | 0.752 | 0.765 | 0.882 | 0.622 | 0.705 |
| E13_s1 | 0.694 | 0.706 | 0.814 | 0.574 | 0.705 |
| E13_s2 | 0.352 | 0.369 | 0.385 | 0.306 | 0.795 |
| B_s0 / E4b_s1 / E9b_s2 (takeoffs) | 0.851-0.862 | | | 0.670-0.684 | 0.693-0.713 |

**Specificity, compared like with like (post-hoc, so weak):** the question is
whether qazaqstantv sits RELATIVELY better against the untouched sources.

| run | level | qazaq/khabar chrF | qazaq/informburo chrF |
|---|---|---|---|
| E4b_s5 (uncleaned) | partial | 0.873 | 1.048 |
| **E13_s0** | partial | 0.875 | 1.081 |
| **E13_s1** | partial | 0.885 | 1.052 |
| B_s0 / E4b_s1 | takeoff | 0.836 / 0.842 | 1.045 / 1.014 |
| B_s1 / B_s2 | failed | 0.825 / 0.900 | 1.080 / 1.041 |
| E13_s2 | failed | 0.924 | 1.298 |

The cleaned partial runs match the uncleaned partial E4b_s5 within the spread
of B's own runs; retrieval's qazaq/khabar ratio (0.705) matches the uncleaned
takeoffs (0.693-0.713). **No qazaqstantv-specific gain is visible.**

### Reading

**Inconclusive, leaning null.** What the data does and does not say:

- No evidence that removing the flagged references helps qazaqstantv
  specifically, on either chrF or retrieval.
- Not a demonstration that it doesn't: no seed took off, so the designed
  measure was never available; the dev references are themselves ~11.5%
  misaligned, which caps any measurable gain; and the 7%-less-data confound is
  unresolved (the random-removal control was reserved for a null WITH a takeoff).
- Two of three seeds reached partial takeoff (B: one full takeoff, two
  failures). At three seeds that is not interpretable.

### What would settle it

The takeoff lottery made this design fragile. The next test should remove the
lottery instead of buying more seeds: **fine-tune from B_s0** (already coupled,
`--init-from`) for a few epochs on (a) cleaned data, (b) full data, (c) data
with 4,200 random qazaqstantv clips removed. All three arms start from the same
coupled model, so each yields a usable qazaqstantv T2 number, and (c) resolves
the data-quantity confound directly. Cost: three short runs, ~3 GPU-hours.
Not run.
