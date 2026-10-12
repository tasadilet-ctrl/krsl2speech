# E9: normalise the frame rate across sources

Status: **pre-registered, launching 2026-10-01.** Results go in a section at the
end; nothing above it is edited after launch. E8 is on hold.

## Why

The three sources were captured at different frame rates and **nothing in the
pipeline normalised them.** Measured with ffprobe (`r_frame_rate` ==
`avg_frame_rate`, keypoint counts matching video `nb_frames` to within one frame):

| source | fps | frames for ~15 s | % of train | chrF (E4b_s1) | chrF (E1 clean_treat) |
|---|---|---|---|---|---|
| khabar | 25 | ~400 | 36% | **35.19** | **38.38** |
| qazaqstantv | 30 | ~450 | 55% | 29.64 | 38.72 (n=53) |
| informburo | 50 | ~830 | 9% | 29.24 | 33.44 |

All three average ~15 s per clip at ~1.9 words/s -- the same real-world pacing --
so informburo hands the encoder **twice as many frames for the same content** as
khabar. The ST-GCN's temporal kernels are a fixed number of FRAMES wide, so a
sign spanning 10 frames in khabar spans 20 in informburo. The encoder has had to
learn every sign at three temporal scales simultaneously.

Two observations make this more than a theory:

- **qazaqstantv holds 55% of the training clips and scores worst.** If the
  problem were data volume or cleanliness (it is the cleaned recollection), it
  should be winning.
- **khabar, at the slowest rate, scores best** for every checkpoint ever
  measured, under both initialisations.

This was already open in the audit (`compute_velocity` assumes 50 fps while
qazaqstantv is 30; per-clip resampling was never implemented). The per-source
numbers move it from loose end to prime suspect.

## What was built

`data.utils.resample_pose(wb, sc, src_fps, tgt_fps)` plus
`SOURCE_FPS = {informburo: 50, khabar: 25, qazaqstantv: 30}` (measured, not
inferred -- the dataset raises rather than guessing for an unknown source).
Linear interpolation in time, applied in `AsanDataset.__getitem__` immediately
after the pose load, before every other transform, so `min_frames`,
`max_frames` and the downsample stride all act on a common time base.

**Target 25 fps**: it is the minimum of the three, so nothing is upsampled and
no frames are invented. khabar becomes a bit-identical no-op (verified: the
function returns the same array object), informburo halves, qazaqstantv x5/6.

One guard worth recording: an undetected keypoint is stored as exactly (0, 0),
and blending that with a real coordinate would invent a joint position halfway
to the origin. Where either neighbour is (0, 0) the nearer neighbour is copied.
Without the guard, a joint at (10, 10) next to an undetected one produced
(5, 5) -- a plausible-looking, entirely fictional position.

Verified before launch: resampled lengths exactly 0.500 / 1.000 / 0.833 of raw
on real clips; a linear trajectory stays linear to 6e-8 with endpoints exact;
the (0, 0) guard emits only real samples; 25->25 is a true no-op;
`target_fps` is recorded in `run_args` and the rescorer compares it **exactly**,
not as a bool, so a 25 fps checkpoint cannot be scored at 30.

Truncation confound checked and dismissed: only 24 clips of 59,041 (0.04%)
cross `max_frames=1000` in either condition, so this is a clean single-variable
change.

## Design

Three seeds of the **baseline B config** -- upstream Uni-Sign preprocessing,
`--block-padding-mask`, 10 epochs, canonical root, frozen 600-clip selection
manifest, baseline LRs, base mT5 -- changing one thing: `--target-fps 25`.
Baseline is the three B seeds already on record. **E9 vs B isolates frame-rate
normalisation.**

Judged on three things, in order:

1. **Per-clip paired coupling delta** (E8 step 1 instrument): mean over three
   seeds, against B's 3.97 overall / 1.19 for its non-takeoff seeds. This is the
   continuous measure, and it is why E9 can say something from three seeds.
2. **Takeoff rate** against B's 1/3.
3. **Per-source chrF**, which carries the sharpest test below.

## The internal control

**khabar's inputs are bit-identical between B and E9** (25 -> 25). informburo's
change the most (2x), qazaqstantv's by 1.2x.

So if the hypothesis is right, the gain should be concentrated in informburo and
qazaqstantv. khabar can still move -- the encoder no longer has to span three
scales, which may help it indirectly -- but a result where **khabar improves
while informburo does not** falsifies the stated mechanism, whatever it does to
the headline number.

## Readings (pre-committed)

| Outcome | Reading |
|---|---|
| Coupling delta up and informburo/qazaqstantv chrF up more than khabar | The temporal-scale mismatch was a real bottleneck. Adopt 25 fps as the default, re-baseline B, and re-screen E5/E6/E4b on top -- every arm so far was measured against an unnormalised baseline. |
| Takeoff rate >= 2/3 | Same adoption, but note three seeds cannot separate 1/3 from 2/3 (E4b's 2/3 became 2/6). Treat as "run three more", not "solved". |
| Coupling delta unchanged, chrF unchanged | Frame rate is not the bottleneck at this target. Before dropping it, try 30 fps (qazaqstantv's rate, the majority source) -- 25 downsamples 91% of the corpus and may lose detail that offsets the gain. |
| Any source gets WORSE | Interpolation is damaging the signal. Check whether the (0,0) guard fires often on that source; if so, resample scores by nearest-neighbour throughout rather than interpolating. |
| khabar improves but informburo does not | The mechanism as stated is wrong even if numbers improve. Something other than temporal scale changed -- find it before claiming this result. |

Epoch time should also fall (informburo sequences halve); that is a side effect,
not evidence, and must not be reported as a result.

## Run

`run_e9.sh`, tmux `krsl` window `e9`. Trains three seeds (one per free GPU),
then rescores B and E9 together at epoch 10 with one scorer, computes the
per-clip coupling delta for both, applies the takeoff rule, and writes
`output/e9_rescore/summary.txt`. Nothing depends on a session staying open.

## Results (2026-10-02)

Two targets were run. 25 fps was the pre-registered design; 30 fps is the
fallback its null row specified, launched after the epoch-5 read on 25 fps.

All checkpoints at epoch 10, scored on the frozen 600-clip selection set through
one metric path, contamination **0** for all. `mt5_path: google/mt5-base` and
`target_fps` confirmed from each checkpoint's `run_args`.

| run | target | chrF | recall | WER | distinct | coupling delta (95% CI) | takeoff |
|---|---|---|---|---|---|---|---|
| **E9b_s2** | 30 | **34.57** | **0.181** | **0.9247** | 599/600 | **13.59 [12.73, 14.45]** | **YES** |
| E9b_s0 | 30 | 18.96 | 0.025 | 1.0076 | 427/600 | 1.00 [0.42, 1.58] | no |
| E9b_s1 | 30 | 19.39 | 0.025 | 1.0119 | 490/600 | 1.18 [0.65, 1.72] | no |
| E9_s0 | 25 | 17.29 | 0.015 | 1.0081 | 334/600 | 0.50 [-0.05, 1.05] | no |
| E9_s1 | 25 | 20.18 | 0.021 | 1.0119 | 437/600 | 0.60 [0.14, 1.05] | no |

Reference: previous takeoff runs (native fps) reached chrF 29.31-31.47 with
coupling 8.91-10.68; B's non-takeoff seeds sit at coupling 1.11-1.27.

### 25 fps was the wrong target, and the reason was a design error

Every source got worse and coupling fell BELOW baseline (0.50/0.60 vs 1.19).
The justification in "What was built" above -- 25 is the minimum so nothing is
upsampled and no frames are invented -- optimised the wrong risk.
**Downsampling destroys information**: at 25 fps informburo loses half its
frames and qazaqstantv a sixth, degrading 91% of the corpus. The invented-frame
risk guarded against was the smaller one.

The pre-registered "any source gets worse" check was run: the (0,0)
undetected-keypoint guard **never fires** (zero-joint rate 0.000% on all three
sources), so interpolation artefacts are not the cause. Plain information loss
is.

### 30 fps breaks a ceiling that held for 19 runs

E9b_s2 is the best clean-provenance result this project has produced:
chrF 34.57 against a previous best of 31.47, and WER 0.9247 against 0.9495.
E1's chrF 36.25 remains higher but came from a checkpoint whose training history
is unrecorded.

**The coupling delta is the stronger claim.** At 13.59 [12.73, 14.45] it lies
entirely outside the previous takeoff distribution (max 10.68, whose own CI
topped out at 11.46) -- non-overlapping intervals. The run is not merely scoring
higher; it is using its encoder more than any run on record. This is what the E8
step 1 instrument was built for, and it is the first time it has separated two
runs that the takeoff rule would have lumped together as "yes".

### The mechanism in the Why section is wrong

The internal control falsifies it. Comparing takeoff run to takeoff run:

| source | change at 30 fps | E4b_s1 | E9b_s2 | gain |
|---|---|---|---|---|
| informburo | 50->30 (x0.60) | 29.24 | 31.75 | +2.51 |
| khabar | 25->30 (x1.20) | 35.19 | 37.84 | +2.65 |
| qazaqstantv | **30->30 unchanged** | 29.64 | **33.99** | **+4.35** |

**The source whose inputs did not change gained the most.** So it is not "each
source's own temporal scale was wrong". The surviving explanation is that the
encoder no longer has to span three scales at once, which helps every source
including the untouched one -- and that 30 fps succeeds where 25 fails because
it removes the same spread while degrading far less data: only informburo (9% of
the corpus) is downsampled, qazaqstantv is untouched, khabar is upsampled.

That explanation is consistent with both results but was not pre-registered, so
it is a hypothesis, not a finding.

### Status: ceiling, not rate

**Takeoff is 1/3 at 30 fps -- identical to baseline.** Frame-rate normalisation
appears to raise the ceiling without changing reliability, which matches one of
the reading patterns pre-registered in E7.

The ceiling claim rests on **one seed**. The delta CI is over clips, not over
seeds: E9b_s2's coupling is known precisely, but whether a second 30 fps takeoff
lands at 13.6 or at 10 is unknown. Seeds 3-5 are running to settle it
(`run_e9b_more.sh`, `output/e9b_rescore_345/`).

Everything measured before this -- B, E4, E4b, E5, E6, E7, and the E8 step 1
coupling scale -- was measured against an unnormalised baseline. If 30 fps holds
up across seeds, those arms need re-screening on top of it.

## CORRECTION after seeds 3-5 (2026-10-04): the ceiling claim does not hold

Three more 30 fps seeds were run (`run_e9b_more.sh`,
`output/e9b_rescore_345/`). All reached epoch 10; contamination 0.

| run | chrF | recall | WER | coupling delta (95% CI) | takeoff |
|---|---|---|---|---|---|
| E9b_s3 | 27.74 | 0.106 | 0.9739 | **7.66 [6.93, 8.39]** | YES |
| E9b_s4 | 20.78 | 0.025 | 1.0119 | 0.92 [0.40, 1.44] | no |
| E9b_s5 | 19.65 | 0.018 | 1.0119 | 0.49 [-0.02, 1.00] | no |

**The section above claims E9b_s2's coupling "lies entirely outside the previous
takeoff distribution -- non-overlapping intervals". That was a statement about
one seed and it is false for the arm.** E9b_s3 took off at delta 7.66, BELOW
every native-fps takeoff (lowest 8.91), with a CI that does not reach them.

Six seeds at 30 fps, against the native-fps record:

| | takeoff rate | takeoff chrF | takeoff delta |
|---|---|---|---|
| native fps (B, E4b, E6) | 4/18 | 29.31-31.47, mean 30.08 | 8.91-10.68, mean 9.53 |
| 30 fps | **2/6** | 27.74-34.57, mean 31.16 | 7.66-13.59, mean 10.6 |

**30 fps raised neither the ceiling nor the rate.** It WIDENED the spread,
producing both the best run on record and a takeoff below the entire previous
range. A 1.1 chrF difference in means on n=2 vs n=4 is noise. The rate, 2/6, is
identical to baseline's 1/3 and to E4b's 2/6. Non-takeoff coupling averages 0.90
against B's 1.19 -- if anything slightly lower.

**This is the E4b failure mode repeated, after naming it.** The message
reporting E9b_s2 carried "E4b's 2/3 became 2/6" as a caveat while leading with
"breaks a ceiling that held for 19 runs" off a single seed. The caveat should
have been the conclusion. One takeoff seed is not a ceiling.

### What survives

- **E9b_s2 is a real artifact**: chrF 34.57, recall 0.181, WER 0.9247, clean
  provenance, contamination 0 -- the best checkpoint this project has produced.
  It is the lucky tail of a distribution whose mean did not move, not evidence
  that 30 fps is better.
- **25 fps is genuinely harmful** (every source worse, coupling below baseline),
  consistent with the information-loss mechanism. That holds at two seeds.
- **The coupling instrument works.** It separated two runs the takeoff rule
  scored identically as YES (13.59 vs 7.66, non-overlapping CIs) -- the first
  measure here to distinguish takeoff *quality*, and it caught this error in one
  extra scoring pass instead of another six seeds.
- The falsified mechanism (qazaqstantv unchanged yet gaining most) is now moot:
  there is no arm-level gain to explain.

### Status

**E9 is closed. Frame-rate normalisation is not the lever.** 25 fps hurts,
30 fps does nothing to the rate or the mean. Do not adopt a target_fps default;
the flag stays available and is recorded in run_args.

Seven arms, 25 runs, 6 takeoffs (~24%). Nothing has moved the takeoff rate off
~1/3: not encoder objectives (E5, E6), not learning rates (E4, E4b), not decoder
initialisation (E7), not frame rate (E9). The rate increasingly looks like a
property of the training setup rather than something these interventions reach,
which is the argument for E8 -- the one intervention aimed at the objective
itself rather than at either tower.

## CORRECTION 2 (2026-10-04): every rate above is off by 2x

`configs/config.yaml` sets `paths.asan.downsample_every: 2` with the comment
"videos are 50 fps -> 25 fps for the encoder". That premise is true only for
informburo. The trainer passes it to `AsanDataset`, which applies `wb[::2]`
AFTER `resample_pose`; the rescorer reads the same key, so train and eval have
always matched -- at half the stated rate. Both B and E9b logs print
`downsample=2x`.

Effective frame rates actually seen by the encoder:

| | informburo | khabar | qazaqstantv |
|---|---|---|---|
| native (ffprobe) | 50 | 25 | 30 |
| every run B..E7 | 25 | **12.5** | **15** |
| E9 ("25 fps") | 12.5 | 12.5 | 12.5 |
| E9b ("30 fps") | 15 | 15 | 15 |

The relative statements in this document (which source was up- or downsampled
versus baseline, and by how much) still hold, because the stride applies to both
arms. The absolute labels do not: "25 fps" was 12.5 fps, and "25 fps destroys
information" was 12.5 fps destroying information -- if anything stronger
support for the information-loss reading.

**No run in this project has ever trained khabar or qazaqstantv (91% of clips)
at their native temporal resolution.** The E9 smoke test missed this because it
constructed `AsanDataset` directly, with the default `downsample_every=1`,
instead of going through the config path the trainer uses.
