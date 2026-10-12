# E11: native frame rate, capped Uni-Sign-style at 256 frames

Status: **pre-registered, launched 2026-10-10.** Results go at the end; nothing
above them is edited after launch.

## Why

E10 (stride 1, uncapped) was worse than baseline on every measure: 0/3 takeoff,
coupling 0.13-0.40 with every CI including zero, chrF 11-18. Uni-Sign's own
pretraining (EXPERIMENT_e10_native_fps.md, "What Uni-Sign was pretrained on")
kept every frame but **never fed the encoder more than 256 frames**, keeping a
random sorted subset of longer clips. E10 fed 435-770 frames, two to three times
anything the pretrained encoder saw. The arms that worked (B, E9b) fed ~220-390.

## Design

Three seeds, `train/train_min.py` (exact deterministic parity with the old
trainer, scripts/parity_train_min.sh), B config, base mT5, baseline LRs, with
`--downsample-every 1 --cap-frames 256`.

- **Training:** clips longer than 256 frames keep a random sorted subset of 256,
  redrawn every time the clip is loaded -- as upstream.
- **Evaluation:** 256 evenly spaced frames, so scores are reproducible
  (upstream also randomises at eval; we deliberately do not).
- The cap is recorded in `run_args`; the rescorer and retrieval script read it
  from each checkpoint, like the stride.

Verified before launch: capped lengths are exactly min(T, 256); training draws
differ between loads and eval draws are identical; sampled frames are real
frames in temporal order; an old uncapped checkpoint still rescores to its
recorded chrF 29.79 / recall 0.128 / WER 0.9618; smoke run peaks at 22 GB and
tracks B's early loss (18.54 vs 18.47 at batch 300).

### What this does and does not isolate

Capping stride-1 clips at 256 lowers their mean density to ~14-17 fps, close to
E9b's 15 fps. So E11 is **not** a test of "native temporal detail". It isolates:

- **E11 vs E10** -- identical stride-1 input, only the cap differs. This is the
  test of whether E10 failed because its sequences were too long.
- **E11 vs E9b** -- similar length, random jittered subsampling vs a uniform
  15 fps grid.
- **E11 vs B** -- whether the cap is a better default than stride 2.

## Readings (pre-committed)

Primary measure: per-clip coupling delta, against B (non-takeoff 1.11-1.27,
takeoff 9.52) and E10 (0.13-0.40). Takeoff rate reported, not used to decide
-- three seeds cannot resolve it.

| Outcome | Reading |
|---|---|
| Non-takeoff coupling within or above B's 1.11-1.27, or any seed takes off | **Length was E10's problem.** The pretrained encoder needs sequences near its 256-frame pretraining cap; E10's failure was not about temporal detail. |
| ... and on >= 2 seeds non-takeoff coupling > 1.5 or a takeoff with delta > 10 | The cap beats stride 2: adopt `--cap-frames 256` as the new baseline. |
| Coupling at E10's level (<= 0.5) on all seeds, 0/3 | Length is **not** the explanation. Since E9b (uniform 15 fps, similar length) worked, suspect the random subsampling itself, or stride-1 frames themselves, before anything else. |
| Between 0.5 and 1.1 | Partial recovery: length explains some of E10, not all. Report as such; no default change. |

A single seed above E9b_s2's 34.57 is not an outcome in this table.

## Run

`run_e11.sh`, tmux `e11`. One seed per free GPU, then rescores clean and
pose-shuffled and writes `output/e11_rescore/summary.txt`. The epoch-5 screen
(`scripts/screen_epoch5.py`, report-only) is attached separately and logs to
`output/screen_e11.log`.

## Results (2026-10-11)

All three seeds completed 10 epochs (train_min, ~22 min/epoch, 22 GB);
contamination 0.

| run | chrF | recall | coupling delta (95% CI) | info / khab / qaza chrF | ep5 distinct | takeoff |
|---|---|---|---|---|---|---|
| E11_s0 | 17.05 | 0.015 | 0.65 [0.19, 1.11] | 9.75 / 21.38 / 19.49 | 168 | no |
| E11_s1 | 16.55 | 0.016 | 0.62 [0.06, 1.17] | 11.24 / 19.33 / 18.88 | 350 | no |
| **E11_s2** | **26.59** | **0.091** | **6.27 [5.61, 6.93]** | 24.88 / 29.51 / 25.10 | **419** | **YES** |
| E10 (uncapped) | 11.23-17.68 | | 0.13-0.40 | | 33-193 | 0/3 |
| B | 29.79 / 19.83-21.02 | | 9.52 / 1.11-1.27 | | | 1/3 |

**The pre-registered rows overlap, which I did not catch when writing them.**
Row 1 ("any seed takes off") and row 4 ("non-takeoff coupling between 0.5 and
1.1") are both met. Read together: **partial recovery.**

- The cap rescues part of E10's failure: takeoff is possible again (E10 0/3),
  and non-takeoff coupling doubles (0.62-0.65 vs 0.13-0.40). Sequence length
  explains some of what went wrong at stride 1.
- It does not reach B: non-takeoff coupling is half of B's 1.11-1.27, the two
  failures score below B's failures (16.6-17.1 vs 19.8-21.0), and informburo --
  the source whose clips are cut hardest (~770 -> 256) -- collapses in both
  (9.8, 11.2).
- E11_s2 is the **weakest takeoff on record**: coupling 6.27 against a previous
  minimum of 7.66, chrF 26.59 and recall 0.091 just over the 26 / 0.08 bar.

**Decision: no default change.** Stride 2 stays the baseline; `--cap-frames`
remains available.

### What the frame-rate series now says (E9, E9b, E10, E11)

Nothing in it beats stride 2. Lowering the rate (E9) hurt; raising it (E10)
hurt more; capping the raised rate (E11) recovers about half. The best-
supported reading is that the pretrained encoder works best close to the input
statistics it already fits -- which stride 2 happens to approximate -- and that
moving away in either direction costs coupling. That is a post-hoc summary, not
a tested claim.

## The epoch-5 screen failed again

E11_s2 scored **419** distinct at epoch 5 -- below the 500 kill line -- and took
off. Had `--kill` been on, the only takeoff in this arm would have been killed.
This is the second out-of-sample failure (the first was E9b_s3 at 547 against
the original 550 line). The screen is a probability, not a rule; see
NOTES_ep5_takeoff_predictor.md.
