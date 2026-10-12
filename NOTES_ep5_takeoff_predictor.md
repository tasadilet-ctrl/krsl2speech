# An epoch-5 screening signal for takeoff

Built 2026-10-01 from `output/ep5_predictor/` and `output/e7_ep5_rescore/`
(`run_ep5_all.sh`, `run_e7_ep5_rescore.sh`). Every epoch-5 checkpoint whose
epoch-10 verdict is known was scored on the frozen 600-clip selection set
through one metric path. Contamination 0 everywhere.

## Result

**Distinct-output count at epoch 5 separates takeoff from no-takeoff perfectly
on the 18 runs available (B, E4, E4b, E5, E6).**

```
takeoff    (n=4):  577  583  586  591
no-takeoff (n=14):   4   37   54  234  336  355  374  395  396  407  480  494  512  547
```

Any cut in 548-576 gives 18/18. Nothing sits in the gap.

**Top-1 share does NOT work** -- 16/18 at best (cut 1.9%). The four takeoff runs
span 1.0-3.5% and collide with non-takeoff runs at 1.2, 2.2, 2.3, 3.0 and 3.3%.
It detects only catastrophic collapse (E4b s2/s3/s4 at 14.3/48.3/80.2%). The
original hypothesis named top-1 share; the data picked distinct count instead.

Epoch-5 chrF and content recall also trend the right way but overlap: E4b_s5
reaches ep5 chrF 20.28 without taking off, above several runs that likewise
didn't, and below no takeoff run -- usable as a weak signal, not a clean cut.

## Why it is not yet a law

- The threshold is fitted on the same 18 points it is scored on. No held-out
  set, and only 4 positives. One future takeoff at 560 breaks it.
- The gap is wide (547 -> 577, 30 clips) and the four positives cluster tightly,
  which is what makes it worth using despite the above.
- **It is fitted entirely on base-mT5 runs.** It should not be assumed to
  transfer to a different decoder initialisation (see E7 below).

The honest test is to call the next run's epoch-10 outcome from its epoch-5
distinct count *before* scoring epoch 10.

## Adversarial case it survives

E4b_s5 ends at ep10 chrF 24.24 with 597/600 distinct -- diverse output, but
recall 0.059, below the 0.08 takeoff bar. At epoch 5 it is at 480, correctly on
the "no" side. So the signal is not merely tracking "does the model emit varied
text at the end".

## What it says about E7, and why that is not conclusive

| run | ep5 distinct | ep5 top-1 |
|---|---|---|
| E7 s0 | 210/600 | 5.7% |
| E7 s1 | 231/600 | 25.8% |

Both are below every labelled run except the three E4b collapses, and far below
the 577-591 takeoff band. Taken at face value that predicts no takeoff.

**The confound:** E7's decoder was domain-adapted on corpus transcripts, and
this predictor was fitted only on base-mT5 runs. A decoder already fluent in
Kazakh broadcast phrasing may emit less varied text at equal health, so the cut
may not transfer. Distinguishing "E7 was failing" from "the predictor does not
apply to E7" needs at least one E7 seed carried to epoch 10.

## Use

Score epoch 5 on the selection set (~40 s) and count distinct hypotheses. Below
~550, the run has not been recovering in this project's history; consider
killing it and saving ~2.5 h. Do not apply the cut across a change of decoder
initialisation without re-validating it.

## Held-out prediction, recorded before the result exists (2026-10-01)

E7 seed 0 is being resumed from its epoch-5 checkpoint to epoch 10. Its epoch-5
distinct count is **210/600**, far below the 548-576 decision band and below
every labelled run except the three E4b collapses.

**Prediction: NO TAKEOFF** (epoch-10 chrF < 26 or content recall < 0.08).

This is the first out-of-sample call this signal has made, and it is a weak test
in one direction: a correct "no" is consistent both with the predictor working
and with E7 simply being a bad arm. The informative outcome is the opposite one
-- if E7 seed 0 *does* take off from 210 distinct, the 562 cut does not survive
a change of decoder initialisation and must be refitted per-init.

Either way the epoch-10 result also settles the separate question E7 was built
to answer, which the kill at epoch 5 left open.

### Outcome: the prediction was correct (2026-10-01)

E7 seed 0 resumed from epoch 5 and finished epoch 10 at **chrF 18.85, content
recall 0.018, 283/600 distinct, top-1 11.0% -- no takeoff**, as predicted from
its 210/600 at epoch 5.

This is the signal's first out-of-sample call, and it held **across a change of
decoder initialisation**, which was the specific weakness flagged above. The
record is now 18/18 in-sample plus 1/1 held-out.

Still only 5 takeoff events in 19 runs, and still one threshold fitted on the
in-sample 18, so this remains a screening heuristic. But it is now reasonable to
use it as a kill criterion: below ~550 distinct at epoch 5, stop the run and
save ~2.5 h. Note the signal is NOT monotone in time -- E7 seed 0 went from 210
distinct at epoch 5 to 283 at epoch 10 while chrF *fell* 20.91 -> 18.85, so
"distinct count rose" is not evidence a run is recovering.

## Out-of-sample test on six 30 fps seeds, and a correction (2026-10-08)

`scripts/screen_epoch5.py` (report-only) scored the epoch-5 checkpoints of all
six E9b seeds, whose epoch-10 outcomes were already known but never used to fit
anything here:

| seed | ep5 distinct | old rule (cut ~550) | outcome at ep10 |
|---|---|---|---|
| s0 | 72 | no | no |
| s1 | 385 | no | no |
| s2 | 592 | takeoff | **took off** |
| **s3** | **547** | **no** | **took off** (chrF 27.74) |
| s4 | 321 | no | no |
| s5 | 299 | no | no |

**The claim above that "no run below ~550 has ever recovered" is false.** E9b_s3
took off from 547 -- the same value at which E5_s1 did not. The two groups now
touch, so the empty gap the 18/18 result rested on is gone. Had `--kill` been on
with the old line, a successful run would have been killed; that is why killing
was left opt-in until this test.

Record over 27 labelled runs:

    took off (6):   547  577  583  586  591  592
    did not  (21):  4 37 54 72 153 210 234 252 299 321 336 355 374 385 395 396
                    407 480 494 512 547

**Revised rule: kill below 500.** Zero false negatives so far (47 below the lowest
takeoff) while still catching 19 of 21 failures. 500-576 is an ambiguous zone:
let those runs finish. This is the screener's default now. It is still a
heuristic fitted on 27 runs, with a single takeoff (s3) defining its margin.

## E10 (stride 1, a new input pipeline): 3/3 correct (2026-10-10)

Report-only screen: 33, 76, 193 distinct at epoch 5 -> all three below the 500
line; all three failed to take off at epoch 10. Record now 30 labelled runs,
zero false negatives at the 500 line.

## Second out-of-sample failure: stop treating this as a rule (2026-10-11)

E11_s2 scored 419 distinct at epoch 5 and took off (chrF 26.59). E11_s0 (168)
and E11_s1 (350) did not. A takeoff now sits 81 below the 500 kill line.

Record over 33 labelled runs (epoch-5 distinct -> outcome at epoch 10):

| epoch-5 distinct | runs | took off | rate |
|---|---|---|---|
| < 500 | 27 | 1 (419) | ~4% |
| 500-576 | 3 | 1 (547) | 1 of 3 |
| >= 577 | 5 | 5 | 5 of 5 |

That is useful as a **probability**: a run under 500 at epoch 5 has taken off
about 1 time in 27. It is not a safe kill criterion -- each of the two "rules"
fitted so far was broken by the next experiment. The screener now reports the
historical rate for the band a run falls in, and `--kill` prints this record
before acting.
