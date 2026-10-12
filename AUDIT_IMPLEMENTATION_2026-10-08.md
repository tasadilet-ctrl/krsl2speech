# Audit changes: what was implemented (2026-10-08)

Follows the audit of 2026-10-04 ("drastic changes"). Each item: what was built,
how it was verified, and what is still open. All code is synced to the box.

| # | Audit item | Status |
|---|---|---|
| 1 | Measure the ceiling with a human | **Tooling done, verified; needs an annotator** |
| 2 | Sign-faithful targets | Same tool collects them (question 4); scale-up needs annotators |
| 3 | Reframe as retrieval | **Done -- 86% top-1 of 200, controls passed** |
| 4a | Native frame rate | **Bug fixed, verified; E10 training** |
| 4b | Re-detect hands at higher resolution | **Not possible -- the pixels do not exist** |
| 5 | Stop re-rolling the lottery | **Done: --init-from, screener (re-fit), deterministic mode** |
| 6 | Minimal trainer | **Done -- exact parity with the old trainer, 400/400 batches** |
| 7 | RGB | Not touched: closed by the project owner on 2026-07-16 |

E8 (decoder corruption) stays parked; its flag is in the trainer, dormant at 0.

## 1-2. Ceiling study package

`scripts/build_annotation_package.py` -> `~/ceiling_study` on the box,
`~/Downloads/ceiling_study.zip` locally (60 MB). 100 clips from the frozen
selection set (34/33/33 by source), shuffled, **no audio** -- an interpreter
who hears the speech marks words "signed" because they heard them.

`scripts/annotation_tool.html` (the package's `index.html`): opens locally, no
server or upload. Per clip: video at 0.25-1x with frame stepping; every
transcript word clickable signed / not signed / unsure; does the signing match
the text; are the hands readable; free text "what was actually signed"; time
per clip recorded. Autosaves; Export / Import JSON. Bilingual EN/RU.
Verified in a browser: video decodes, labels cycle, completion and progress
update, state survives reload and resumes at the first unfinished clip, and
the export JSON is correct.

`scripts/analyze_annotations.py`: from the export, the signed fraction overall,
by source and by sentence position; every checkpoint's recall on signed vs
unsigned words using predictions already on disk; clip-level answers;
projected hours for 3-5k clips; Cohen's kappa for multiple annotators.
Verified against a synthetic annotator with a known answer (last 40% unsigned):
recovered 61% signed and a clean step in the position bins at 60%.

**To run:** give the zip to a KRSL interpreter (README in English and Russian),
then `PYTHONPATH=. python scripts/analyze_annotations.py <export>.json
--pred-dirs output/all_rescore output/e4b_rescore output/e9b_rescore
--shuf-dirs output/e8_shuffled output/e9b_rescore_shuf`.

## 3. Retrieval -- NOTES_retrieval.md

`scripts/eval_retrieval.py` scores every same-source reference by the
checkpoint's own likelihood, with PMI against other clips' poses to cancel the
text prior. B_s0: **85.5% (informburo) / 86.5% (khabar) top-1 of 200, median
rank 1**, chance 0.5%. Controls: duration alone 1.0%; nearest training
sentence shares a median 9-12% of content words (no memorisation path);
non-takeoff B_s1 1.5% top-1. Retrieval is real and tracks coupling. Replication
on E4b_s1 running.

## 4a. Frame rate -- EXPERIMENT_e10_native_fps.md

`downsample_every: 2` under "videos are 50 fps" trained 91% of the corpus at
12.5 / 15 fps in every one of 34 runs. Config default is now 1; the trainer
records the stride in `run_args`; the rescorer reads it from each checkpoint
(legacy 2). Verified: B_s0 rescored under the new config reproduced its
recorded chrF 29.79 / recall 0.128 / WER 0.9618 exactly; resuming a pre-fix
checkpoint under the new default is refused with
`{'downsample_every': (1, 2)}`. E10 (B config, stride 1, 3 seeds) is training;
stride 1 needs ~53 GB per run.

## 4b. Resolution -- NOTES_input_resolution.md

Videos are 204x186 to 314x282 crops, and the archive's "raw" videos are the
same size. Hands are 31-51 px; finger joints 2-3 px apart. The best source is
the one with the largest hands. Re-detection would need the original
broadcasts.

## 5. Lottery tooling

- `--init-from CKPT` (both trainers): full model, weights only, fresh
  optimizer. Refuses a different input pipeline unless
  `--init-allow-input-change`. Verified: refusal names the stride; a matching
  init starts at loss 2.14 at batch 500 against ~16.4 from scratch.
- `scripts/screen_epoch5.py`: attaches to any run directories, scores each
  epoch-5 checkpoint when written, reports distinct-output count; `--kill`
  opt-in. **Out-of-sample test changed the rule**: E9b_s3 took off from 547,
  below the old 550 line. Re-fit on 27 labelled runs to kill below 500 (zero
  false negatives so far). Attached to E10 in report-only mode.
- **Deterministic mode** (`train_min.py --deterministic`, old trainer
  `KRSL_DETERMINISTIC=1`). Found while testing parity: **the same seed does not
  reproduce a run.** Two default-mode runs with seed 0 reached 18.34 and 17.26
  at batch 400. "Seed 0 took off" therefore does not mean seed 0 would take
  off again.

## 6. Minimal trainer -- train/train_min.py

295 lines (old trainer: 2,360), plus `train/collate.py` (the collator, moved
verbatim and re-exported so existing imports work). Supports exactly the B
config plus stride, target_fps, init-from, resume, seed, LR overrides,
deterministic mode. Same checkpoint format, so the rescorer and screener read
it unchanged. Inference stays in the rescorer.

`scripts/parity_train_min.sh`, deterministic mode: identical to the old
trainer at **400/400 batches**, identical to itself on a rerun, and a
different-seed control differs by 1.27.

A first, non-deterministic parity run declared a bug: train_min ended 0.79
above the old trainer while an old-vs-old pair differed by 0.13. A second
train_min run landed 0.30 *below* it. The noise floor had been one sample. Same
lesson as the takeoff results: one comparison is not a distribution.

## Running when this was written

- E10 seeds 0 and 1 (stride 1), seed 2 queued; summary at
  `output/e10_rescore/summary.txt`; epoch-5 screen at `output/screen_e10.log`.
- Retrieval replication on E4b_s1: `output/retrieval_e4b/`.

## Needs a person

1. A KRSL interpreter for the 100-clip ceiling study (the zip). Median time
   per clip from the tool decides whether 3-5k faithful targets are affordable.
2. A decision on retrieval as a deliverable task alongside, or instead of,
   free generation.
