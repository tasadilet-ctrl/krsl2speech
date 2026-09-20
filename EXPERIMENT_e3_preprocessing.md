# E3: input-handling ablation (legacy / upstream Uni-Sign / SignSpace)

Launched 2026-09-17, tmux `krsl` window `e3`, scheduler `run_e3.sh`.
Logs: `output/e3_arm{A,C,B}_*/train.log`, scheduler `output/e3_scheduler.log`.

## Arms

All three share: `ASAN_ROOT=~/asan_canonical`, the frozen 600-clip selection
manifest, `--pretrained-unisign csl_stage1_weight.pth`, `--block-padding-mask`,
`--epochs 10`, `--seed 0`, identical batch/schedule/decoding. They differ
**only in spatial preprocessing**.

| arm | flags | input |
|---|---|---|
| **A legacy** | `--use-enriched --real-wrists --score-quantiles` | 1410-dim: bone offsets + per-clip standardized absolutes + velocity/accel (unread) + confidence |
| **C signspace** | A + `--signspace` | as A, but body in a 3×shoulder box and hands/face normalized locally |
| **B upstream** | `--unisign-preprocess` | 207-dim: exact upstream Uni-Sign port — 9-joint body with real wrists, clip-level body box, wrist-relative hands, nose-relative face, (x, y, score), score ≤ 0.3 masked |

**A vs C isolates normalization.** B is a package, not a controlled contrast:
it changes normalization, node selection, channel layout and score handling at
once. Its job is to answer a different question — whether our preprocessing was
mismatched to the pretrained weights all along — so a B win must not be read as
"SignSpace loses".

## Why this initializer

Uni-Sign CSL weights + **base** mT5. Neither has seen KRSL, so provenance is
fully documented and disjoint from canonical dev/test — unlike
`ours_enriched_friend_mt5.pth`, whose mT5 history is unrecorded. Absolute
scores will likely start below `clean_treat` (which began from a KRSL-tuned
mT5); that is the price of a defensible comparison. E1 scores are not
comparable to these.

## Arm B fidelity

`data/utils.py::unisign_part_features` is a port of upstream `load_part_kp` /
`crop_scale`. Verified **identical to upstream's own function on 80 randomized
cases** (realistic scores, mixed, mostly-below-threshold, single frame) by
executing the functions straight out of the official `datasets.py`.

Upstream oddities reproduced deliberately, not fixed:
- `np.clip(result, -1, 1)` spans the whole array, so **upstream's score channel
  is capped at 1 too**. Since raw RTMPose scores are ≥1 for ~100% of joints,
  upstream's confidence channel is as saturated as ours was before E2. Arm B
  therefore does NOT get the E2 score fix — that is what upstream is.
- Body box is clip-level (one box for all frames), not per-frame.
- Hand coordinates are taken relative to the hand root even when that root is
  undetected.

Arm B also loads the pretrained projection **natively** (3 channels, no
5-channel adaptation), which A and C cannot do.

## Pre-committed readings

| Outcome | Reading |
|---|---|
| C > A, C > B | SignSpace helps on its own terms. Make it default; report the ablation. |
| B > A and B ≈/> C | Our preprocessing was mismatched to the pretrained weights. The headline is compatibility, not normalization. |
| C > A, B lowest | Normalization is the lever and our richer input beats upstream's. Strongest case for the current architecture. |
| A ≈ C | Nuisance variation was not a limiter here. Keep SignSpace behind a flag; do not report a win. |
| All ≈ equal | Input handling is not where the headroom is. Move to E5 (alignment). |

Margins must be read per source and against seeds. Single-seed gaps of a few
tenths of chrF on 600 clips mean nothing; finalists get 3 seeds (E8).

## Scheduler

Starts each arm when a GPU has ≥36 GB free, one arm per GPU, waiting for the
first training batch before starting the next (simultaneous CUDA inits have
stalled on this box). Both GPUs currently carry ~60–72 GB from other users, so
the arms will most likely run sequentially on GPU 1 at roughly 5 h each. If
GPU 0 frees, the queue uses it automatically. Three attempts per arm before it
is dropped.

## Next

Re-score all three arms with `scripts/rescore_checkpoints.py` on the canonical
selection set (arm B needs `--unisign-preprocess`), compare on the common clean
subset, then decide E4/E5.

---

## Interim results — 2026-09-18 11:36 (A, C complete; B at epoch 6/10)

Re-scored with `scripts/rescore_checkpoints.py` on the 600-clip selection set.
**Contamination 0/600 for every checkpoint** (clean-provenance init confirmed).
The re-scorer reproduces the trainer's logged BLEU exactly, so the two metric
paths agree.

chrF / content-word recall, overall and per source:

| ckpt | all | informburo | khabar | qazaqstantv | len ratio | full-dev CE |
|---|---|---|---|---|---|---|
| A ep5 | 15.84 / 0.010 | 15.44 / 0.007 | 15.41 / 0.008 | 16.75 / 0.014 | 0.51 | 2.801 |
| C ep5 | 18.36 / 0.021 | 18.06 / 0.016 | 18.85 / 0.023 | 18.13 / 0.024 | 0.64 | 2.768 |
| **B ep5** | **23.16 / 0.059** | **20.61 / 0.032** | **25.71 / 0.077** | **23.02 / 0.068** | 0.74 | **2.602** |
| A ep10 | 19.90 / 0.019 | 18.93 / 0.012 | 20.57 / 0.023 | 20.21 / 0.024 | 0.73 | 2.729 |
| C ep10 | 21.37 / 0.039 | 19.19 / 0.022 | 24.03 / 0.051 | 20.67 / 0.044 | 0.72 | 2.665 |

- **B > C > A at matched epoch 5, in every source, on chrF, recall and
  full-dev CE** (8,411 clips, the least noisy signal here). B at epoch 5
  already beats both A and C at epoch 10.
- **C > A at both matched epochs** (+2.5 / +1.5 chrF; recall ~2×).
- Reading per the pre-committed table: **B > A and B > C → the headline is
  compatibility with the pretrained weights**, with normalization (C > A) a
  secondary, real effect within our own pipeline. Tentative until B finishes
  and seeds are run.

Caveats:
- **Single seed.** Per-source consistency is supporting evidence, not a
  substitute for seeds.
- **B is a package.** It differs from A/C in coordinate type (absolute vs bone
  offsets), channels (3 native vs 5 adapted), face nodes, box scope
  (clip-level) and score handling. Its win cannot be attributed to one of these
  yet. Note A/C already have real wrists via E2, so wrists alone don't explain it.
- **All arms are still near the floor.** Samples show template news sentences
  repeated across different clips (prior collapse); topic-level signal appears
  (war/Middle East clips get war sentences) but content recall is ≤0.08. The
  E1 KRSL-tuned-mT5 init reached chrF 36: the mT5 initialization matters more
  than any input variant here.
- **WER-based checkpoint selection is noise at this level** (WER ≈ 1.00 for
  every epoch): A's "best" is epoch 5, below its epoch 10. Select on chrF or
  full-dev CE instead.

---

## Arm D — upstream + per-frame hand-size scaling (queued 2026-09-18)

`--unisign-preprocess --unisign-hand-scale ~/asan_canonical/hand_ratio.json`,
otherwise identical to B (same `COMMON` line, seed, init, root). **D vs B
isolates hand scaling.** Queued in tmux `krsl` window `e3d` (`run_e3d.sh`); starts
automatically when B frees GPU 1.

Design: B expresses hands relative to the wrist in body-box units. D keeps that
and rescales each hand per frame so its extent equals the TRAIN-median hand size
(`scripts/fit_hand_ratio.py`: **0.205** body units, p10–p90 0.135–0.268;
informburo 0.195, khabar 0.214, qazaqstantv 0.217). Plain SignSpace (stretch each
hand to [-1, 1]) was rejected: it would move hand values far outside the range
the pretrained weights saw, which is the mismatch arm B showed to be costly.

Deliberate deviations from SignSpace: hands only (face stays upstream), and the
target is a corpus constant rather than a unit box. Per-frame bbox scaling still
erases fist-vs-open-hand size differences, as SignSpace does; a palm-length
(wrist → middle MCP) scale would keep them and is the natural follow-up.

Verified: default path still identical to upstream (80 cases); a 1.8× larger
hand gives identical features (upstream drifts 0.114); body, face and scores
unchanged; on 22 real dev clips B's hand extents span 0.118–0.269 (p10–p90)
while D's sit at exactly 0.2050. The GPU smoke test OOM'd on GPU 0 (25 GB free)
after data loading succeeded; beyond the dataset, D's code path is B's.

Reading: D > B → hand-size nuisance matters even with compatible inputs; make
it the default. D ≈ B → B's win was compatibility alone. D < B → rescaling
hurts (suspect lost fist/open-hand size information; try palm-length scaling).

---

## Arm E — arm B + qazaqstantv fragment references → `text_asr` (queued 2026-09-18)

Identical to B (same `COMMON`, flags, seed, init) except
`ASAN_ROOT=~/asan_canonical_asrfrag`, built by
`scripts/build_asr_fragment_root.py`. **E vs B isolates the transcript change.**
Queued in tmux `krsl` window `e3e` (`run_e3e.sh`). Its scheduler waits until arm D
is confirmed training before polling for a GPU, so the two queued arms cannot
launch simultaneously.

Rule, fixed before any result: a qazaqstantv clip whose `text_human` has fewer
than half the words of its `text_asr` gets `text_asr` as its reference. Applied
to all splits so training and evaluation use one convention:
train 3,283/32,268 (10.2%), dev 175/2,500 (7.0%), test 200/2,500 (8.0%).
Verified: identical clip IDs and order to the canonical root; texts changed only
on listed clips (every change recorded with old/new text in
`switched_clips.json`); khabar/informburo and all pose tables shared by symlink.

**Scoring E vs B — the references differ on 7 of the 600 selection clips.**
- Primary: both checkpoints on the **593 selection clips whose reference is
  identical in both roots**.
- Secondary: each root's own references (E on switched refs, B on original),
  reported but not used to rank.
- Trainer dev CE is not comparable between E and B on qazaqstantv: 175 dev
  references differ.

Readings: E > B on the unchanged clips → the fragments were actively teaching
the model to under-generate (look for a rising hypothesis/reference length
ratio on qazaqstantv). E ≈ B → 10% fragments were not a limiter at this stage.
E < B → mixing ASR-style text into qazaqstantv hurts more than fragments did;
consider dropping those clips instead.

### B epoch 8 (scored 2026-09-18 12:59; B still training)

| ckpt | all | informburo | khabar | qazaqstantv | len ratio |
|---|---|---|---|---|---|
| B ep8 | **29.64 / 0.127** | 27.22 / 0.091 | 33.44 / 0.163 | 27.95 / 0.127 | 0.84 |

chrF / content recall on the 600-clip selection set, 0 contaminated. +6.5 chrF and
2.2× recall over B ep5, gains in every source. 597/600 hypotheses are distinct
(ep5 reused template sentences across clips), and samples now carry clip-specific
content, e.g. a disaster clip's hypothesis reproduces the reference's "336 адам".
Trainer metrics still rising steeply at epoch 8 while the cosine schedule reaches 0
at epoch 10, so the 10-epoch budget likely undertrains this init; a longer schedule
for the winning arm is a natural follow-up (D and E stay at 10 epochs to match B).

---

## Final results — 2026-09-19. The interim "B wins → compatibility" reading is NOT supported.

All five arms at epoch 10, 600-clip selection set, 0 contaminated:

| arm | chrF | recall | infor | khabar | qazaq | len | distinct hyps | most-reused hyp |
|---|---|---|---|---|---|---|---|---|
| A legacy | 19.90 | 0.019 | 18.93 | 20.57 | 20.21 | 0.73 | 413/600 | 15× |
| C signspace | 21.37 | 0.039 | 19.19 | 24.03 | 20.67 | 0.72 | 587/600 | 3× |
| **B upstream** | **29.79** | **0.128** | 27.01 | 33.79 | 28.24 | 0.83 | 598/600 | 2× |
| D B + hand scale | 22.85 | 0.046 | 20.76 | 25.67 | 21.86 | 0.80 | 577/600 | 10× |
| E B + ASR fix | 18.27 | 0.016 | 16.21 | 19.20 | 19.47 | 0.69 | **161/600** | **140×** |

E vs B on the 593 clips with identical references: B chrF 29.84 / recall 0.128,
E 18.26 / 0.016.

Full-dev CE by epoch (8,411 clips):

| | 1 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| B | 3.22 | 2.73 | 2.60 | 2.45 | 2.30 | 2.29 |
| D | 3.13 | 2.82 | 2.78 | 2.73 | 2.64 | 2.64 |
| E | 3.27 | 2.81 | 2.79 | 2.75 | 2.74 | 2.74 |

**What this shows.** All arms sit in one band through epoch 4. B then "takes off"
at epoch 5 — the point where the decoder starts using the pose input — and the
others never do within 10 epochs. **E is B's exact configuration with 5.6% of
training targets changed (3,283 of 59,008), same seed, same data order**, and it
ended in prior collapse (one sentence emitted for 140 of 600 clips). A 5.6%
target change producing that outcome is far less plausible than the takeoff
itself being fragile: a small perturbation decides whether it happens inside the
budget. Configurations were checked (inputs, native weight loading, roots) —
not a bug.

**Consequence.** Single-seed E3 differences are dominated by whether takeoff
happened, not by preprocessing. B's lead over A/C/D and E's loss to B are **not
established**; neither "compatibility is the lever" nor "the ASR fix hurts" can
be claimed. What survives: this init/budget is near a takeoff threshold, and
failing to cross it means prior collapse — the audit's "decoder ignores the
encoder" failure, the target of E5 (pose–text alignment).

**Separating chance from effect** needs replicate seeds of the same config.
If B's seeds reliably take off and E's don't, the text change matters; if B's
seeds also fail sometimes, takeoff is stochastic and arm comparisons need
either several seeds each or a training setup that crosses the threshold
reliably (longer schedule — B20 is running — or an alignment loss).

B20 (B with a 20-epoch schedule) launched 2026-09-19 12:05, tmux `krsl` window
`b20`, `output/e3_armB20_upstream_20ep/`, ETA ~22:10.

### B20 (B with a 20-epoch schedule) — never took off (checked 2026-09-19 19:40, epoch 16/20)

Same seed (0) and data order as B; only the cosine schedule differs. Full-dev CE:
3.24, 3.00, 2.87, 2.80, 2.84, 2.82, 2.78, 2.76, 2.75, 2.77, 2.74, 2.74, 2.74, 2.76, 2.76
(epochs 1–15) — flat since epoch 9 and creeping up while train CE keeps falling
(2.28): fitting the text prior, not using pose. Epoch-15 checkpoint: chrF 16.71,
recall 0.017, 353/600 distinct hypotheses, most-reused 68×.

**Both collapsed runs (E, B20) converge on the same sentence**, "енді әлемді елең
еткізген басты жаңалықтарға тоқталсақ" ("now let's turn to the main news that shook
the world") — a newscaster segue that is the single most repeated full reference in
training (17 identical copies, all qazaqstantv). Non-takeoff runs regress to the mode
of the target distribution. Deduplicating it would only move the mode; the failure
is the collapse itself.

**So far: 1 of 3 runs of the B configuration took off** (B yes; E, a 5.6%-target
perturbation, no; B20, a schedule change, no). A longer schedule does not make
takeoff reliable — it gives more time to overfit the prior. This strengthens the
case for an objective that forces use of the pose input (E5, pose–text alignment)
over further preprocessing or schedule variants. Seeds 1 and 2 of B (queued) will
say how often the unmodified configuration takes off.

---

## E3 CONCLUSION (2026-09-20): no preprocessing arm is distinguishable. Takeoff is the only variable that mattered.

Three seeds of the *same* B configuration, scored on the 600-clip selection set:

| B config | chrF | recall | BLEU | distinct | reuse |
|---|---|---|---|---|---|
| seed 0 | **29.79** | 0.128 | 2.63 | 598/600 | 2× |
| seed 1 | 21.02 | 0.031 | 0.42 | 503/600 | 32× |
| seed 2 | 19.83 | 0.025 | 0.27 | 523/600 | 14× |

Seed spread for one config: **19.8 – 29.8 chrF**. The single-run arms
(A 19.9, C 21.4, D 22.9, E 18.3, B20 16.7) all sit inside that spread, and inside
the non-takeoff part of it. **Every conclusion drawn from single-run arm
comparisons is withdrawn**: not "upstream preprocessing wins", not "SignSpace
helps", not "hand scaling helps", not "the ASR fix hurts". The data cannot
separate them.

Takeoff rate: **1 of 8 runs** (only B seed 0). Runs that don't take off converge
on the corpus's most frequent reference and score chrF 17–23 with high hypothesis
reuse; the one that did reached chrF 29.8 with 598/600 distinct outputs. Neither a
longer schedule (B20) nor any input variant changed this.

**What is established:**
1. At this init and budget, training is bimodal — it either starts using the pose
   input or regresses to the language prior, and which happens is mostly chance.
2. Single-seed comparisons here are worthless: seed noise (~10 chrF) dwarfs every
   arm difference measured (~3 chrF). Any future claim needs ≥3 seeds per arm, as
   the audit's E8 already required.
3. The bottleneck is not input representation. It is that nothing in the objective
   forces the decoder to use the encoder.

**Next: E5 (pose–text alignment).** A contrastive objective that makes pose and
text representations correspond attacks the collapse directly, rather than
hoping takeoff happens. Judge it by takeoff rate across seeds, not by a single
run's chrF. Re-testing preprocessing variants is only worthwhile once training
is reliable — and then with seeds.
