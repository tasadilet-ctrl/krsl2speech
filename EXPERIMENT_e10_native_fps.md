# E10: train at native frame rate (stride 1)

Status: **pre-registered 2026-10-08, queued** -- starts when a GPU has 80 GB
free (the box is currently shared; another user's job holds ~73 GB per card).

## Why

`paths.asan.downsample_every` was 2 under the comment "videos are 50 fps ->
25 fps". Only informburo is 50 fps (ffprobe). Every one of the 34 runs in this
project therefore trained khabar at 12.5 fps and qazaqstantv at 15 fps -- 91% of
the corpus at half its native temporal resolution. Signs are distinguished by
handshape transitions that can last a few frames; at 12.5 fps a transition of
~100 ms is one frame or none.

This is not a hypothesis-driven arm like E4-E9. It is a bug fix whose size is
unknown, and it changes the baseline every other arm was measured against.

## What changed in the code (verified before queuing)

- Config default is now `downsample_every: 1`. The trainer takes
  `--downsample-every` and records the stride it actually used in `run_args`.
- The rescorer reads the stride from each **checkpoint**, not the config.
  Checkpoints without the key resolve to 2, which all 34 train logs confirm.
  Regression test: B_s0 rescored with no flags under the new config resolved to
  stride 2 and reproduced chrF 29.79 / recall 0.128 / WER 0.9618 exactly.
- `--resume` resolves the stride before comparing run_args and treats a missing
  key as 2, so a pre-fix checkpoint cannot silently continue at stride 1. Tested:
  resuming E7_s1 under the new default is refused with
  `{'downsample_every': (1, 2)}` as the only mismatch.

## Design

Three seeds of the B config (upstream Uni-Sign preprocessing,
`--block-padding-mask`, 10 epochs, base mT5, baseline LRs, batch 8 x accum 4)
with `--downsample-every 1`. No `--target-fps`. Baseline: B's three seeds.

Batch size is deliberately NOT reduced to save memory: the ST-GCN may carry
batch-dependent statistics, and changing the micro-batch would add a second
variable.

## Judged on (pre-committed)

1. **Per-clip coupling delta** (clean minus pose-shuffled sentence chrF), mean
   over seeds, against B: takeoff seeds 9.52, non-takeoff 1.11 / 1.27.
2. **Content recall and chrF of takeoff runs** against native-stride takeoffs
   (chrF 29.3-31.5, recall 0.12-0.15).
3. Takeoff rate, reported, **not** used for the decision -- three seeds cannot
   resolve it (E4b, E9b).

| Outcome | Reading |
|---|---|
| Non-takeoff coupling > 2.0 or takeoff recall > 0.17 on >= 2 seeds | Temporal resolution was a real ceiling. Make stride 1 the baseline; everything after builds on it. |
| Takeoff recall 0.15-0.17, coupling within B's range | Small gain. Adopt (it is the correct setting regardless) but it is not the bottleneck. |
| No change on any measure | Temporal resolution is not limiting at 12.5-15 fps. Adopt anyway for correctness; the cost is ~2x training time. |
| Worse | Unexpected. Check max_seq_len truncation and memory-driven behaviour before believing it. |

A single seed above E9b_s2's 34.57 is **not** an outcome in this table. One
seed is not a ceiling; that lesson cost two retractions.

## Results (2026-10-10): native frame rate is WORSE

All three seeds completed 10 epochs; contamination 0; one launch per seed, no
OOM or errors, stride 1 confirmed in every log.

| run | chrF | recall | coupling delta (95% CI) | val CE ep10 | info / khab / qaza chrF | takeoff |
|---|---|---|---|---|---|---|
| E10_s0 | 14.42 | 0.012 | 0.13 [-0.35, 0.61] | 2.811 | 16.22 / 11.93 / 15.12 | no |
| E10_s1 | 11.23 | 0.010 | 0.40 [-0.04, 0.85] | 2.829 | 8.79 / 11.65 / 13.41 | no |
| E10_s2 | 17.68 | 0.014 | 0.35 [-0.17, 0.88] | 2.749 | 13.85 / 19.22 / 19.98 | no |
| B non-takeoff | 19.83-21.02 | 0.025-0.031 | 1.11-1.27 | 2.69-2.72 | | |

**Pre-registered row: "Worse".** Its required checks were run and are clean:
truncation affects 24/59,041 clips (0.04%); no memory-driven change (batch 8 x
accum 4, same optimizer, no OOM, no relaunches). So the result stands.

Every coupling CI includes zero: at native rate the decoder effectively stops
reading the encoder -- lower than B's failures (1.11-1.27) and lower than E9 at
12.5 fps (0.50-0.60). Val CE is above B's non-takeoff band on all three seeds.
E10_s1 emits one sentence for 60% of clips.

**Decision: `downsample_every: 2` restored as the config default**, now on
evidence. The run_args mechanism stays, so scoring is unaffected.

### What this says about the frame-rate story

| effective rate (info / khab / qaza) | arm | takeoff | non-takeoff coupling |
|---|---|---|---|
| 12.5 / 12.5 / 12.5 | E9 "25 fps" | 0/2 | 0.50-0.60 |
| 25 / 12.5 / 15 | B (baseline) | 1/3 | 1.11-1.27 |
| 15 / 15 / 15 | E9b "30 fps" | 2/6 | 0.49-1.18 |
| 50 / 25 / 30 | **E10 native** | **0/3** | **0.13-0.40** |

Both the lowest and the highest rates hurt. That rules out "more temporal
resolution is better" and "normalising the rate is what matters". Two
candidates remain and neither is tested: (a) the pretrained Uni-Sign ST-GCN's
temporal kernels are tuned to a particular frame rate, and stride 2 happens to
sit near it; (b) sequence length -- stride 1 doubles the pose tokens the
decoder cross-attends over, which may dilute the pose signal and make the
prior basin easier. The Uni-Sign pretraining frame rate would discriminate (a).

The epoch-5 screen called all three correctly on this new pipeline (33, 76,
193 distinct -> no takeoff).

## What Uni-Sign was pretrained on (looked up 2026-10-10)

From the upstream code and paper (github.com/ZechengLi19/Uni-Sign, arXiv:2501.15187):

- **Pose extraction keeps every frame**, no resizing (`demo/pose_extraction.py`
  reads every frame with `cap.read()` and runs RTMPose on each).
- **No frame-rate conversion anywhere.** Instead every dataset class caps the
  sequence at `max_length`: if a clip is longer, it keeps a **random sorted
  subset** of `max_length` frames (`random.sample(range(duration), k=max_length)`);
  otherwise all frames.
- **`max_length` = 256** (default in `utils.py`; `script/train_stage1.sh`, the
  CSL-News pose-only pretraining that produced `csl_stage1_weight.pth`, does
  not override it).
- **CSL-News native fps is undocumented** (a dataset survey lists it as
  "varies"). The paper's own numbers bound it: mean clip 9.5 s, and 96.2% of
  clips are under 512 frames. At 50 fps 512 frames is 10.2 s, which roughly
  half the clips would exceed, so 50 fps is ruled out; 25-30 fps fits (512
  frames = 17-20 s). Chinese broadcast TV is PAL (25 fps). Unconfirmed until
  one clip is probed.

So the pretrained encoder saw **~25-30 fps at clip level, but never more than
256 frames per sequence**, with random temporal subsampling on long clips.

### Against our arms

| arm | effective fps (info / khab / qaza) | median frames per clip (info / khab / qaza) |
|---|---|---|
| E9 "25 fps" | 12.5 / 12.5 / 12.5 | ~193 / ~218 / ~189 |
| **B (stride 2)** | 25 / 12.5 / 15 | **~385 / ~218 / ~227** |
| **E9b "30 fps"** | 15 / 15 / 15 | **~231 / ~261 / ~227** |
| E10 native | 50 / 25 / 30 | ~770 / ~435 / ~453 |

- **"Match the pretraining frame rate" (hypothesis a) is contradicted.** E10
  put khabar at 25 fps -- the likely pretraining rate -- and khabar got much
  worse (11.7-19.2 chrF vs ~22).
- **Sequence length relative to the 256-frame cap fits better.** The two arms
  that worked (B, E9b) feed ~220-390 frames; E10 feeds 435-770, two to three
  times anything the encoder saw in pretraining. E9 (~190-220 frames) does not
  fit this story cleanly, though its loss is also explained by information
  destroyed in downsampling.
- A second difference: Uni-Sign never fed long clips whole; it randomly
  subsampled them, a temporal augmentation our pipeline never had.

### The test this suggests (not run)

Stride 1 (full temporal detail) but capped Uni-Sign-style at 256 frames by
random sorted subsampling. If length is the problem, this recovers or beats B;
if native-rate detail itself hurts, it does not.
