# Corpus audit: within-broadcast alignment of dev + test (2026-10-15)

`run_audit.sh` -> `scripts/audit_report.py` -> `output/audit_report/`
(`audit_all.csv`: every clip; `flagged_worst.json`: 640 flags, worst first).
Review page: `scripts/build_audit_viewer.py` -> `~/audit_viewer` (20 worst per
source, keypoints drawn, own transcript vs the better-fitting sentence).

## Method

Every usable dev and test clip in a broadcast with >= 2 clips (16,174 clips)
scored against every sentence of its own broadcast with B_s0, PMI as in E12.
**Train cannot be audited with these checkpoints**: they memorised its
clip-transcript pairs, misaligned or not. That needs cross-fitted models.

A clip is flagged as a **mutual mismatch** when another sentence fits the clip
better AND its sentence fits another clip better. Single-sided misses are
mostly ordinary model error (top-1 is ~0.82-0.97 on clean data).

## Result

| source | clips | own sentence ranks 1st | mutual mismatch | 95% CI |
|---|---|---|---|---|
| informburo | 2,417 | 0.935 | 1.5% | [1.1, 2.1] |
| khabar | 8,859 | 0.970 | 0.4% | [0.3, 0.6] |
| **qazaqstantv** | 4,898 | 0.805 | **11.5%** | [10.7, 12.4] |
| all | 16,174 | 0.915 | 4.0% | [3.7, 4.3] |

## The flags are real -- validated against an independent signal

Clips whose transcript implies an impossible speech rate (words/s outside
1-4) are flagged far more often, **within every source**:

| source | words/s 1-4 | outside 1-4 | ratio |
|---|---|---|---|
| informburo | 1.0% | 14.3% | x14 |
| khabar | 0.2% | 6.2% | x31 |
| qazaqstantv | 7.4% | **50.0%** | x7 |
| all | 2.4% | 30.1% | x12.5 |

So the audit detects genuine reference problems, not just model error.

## qazaqstantv's weakness is substantially its references

- It is flagged ~10-30x more than the other sources, and 9.8% of its clips have
  an implausible speech rate against 3.8% elsewhere.
- Even among normal-rate clips it is flagged 7.4% vs 0.2-1.0%.
- Part of its excess is ordinary model error (the model is weaker on
  qazaqstantv: smallest hands, 31 px), so 11.5% is an upper bound on the
  misaligned share, not an estimate of it.

**Counter-intuitive and worth checking with the data owner:** the ~6% of
qazaqstantv clips that use the speech-recognition fallback text are flagged
*less* (7.8% [5.2, 11.5], n=282) than clips with a human transcript (11.7%
[10.8, 12.7], n=4,616). The ASR text is a verbatim record of that time window;
`text_human` is the broadcast's article text matched onto time segments
(methodology.md, step 5), and that matching is where misalignment can enter.
Small n, overlapping-adjacent CIs: a lead, not a finding.

## Not found

No boundary-shift signature: 39% of flagged clips fit an adjacent sentence
better, against 40% expected for a random other sentence in pools this small
(~4). (An earlier version of the report labelled the 39% "boundary shift"
without the baseline; corrected.)

## Consequences

1. **Evaluation:** qazaqstantv dev/test scores are depressed by references
   that do not match the clip. Every per-source comparison in this project
   inherits that.
2. **Training:** the same pipeline produced the 32k qazaqstantv training clips,
   so a similar share is plausibly misaligned there -- unmeasured.
3. **Data owner:** the flag list is directly useful to whoever maintains the
   recollection's text_human matching.

## Auditing TRAIN: memorisation check, and why cross-fitting was dropped (2026-10-15)

The plan was to cross-fit (train two models on broadcast-disjoint halves of
train, each auditing the other half), on the assumption that B_s0 had memorised
its training pairs. That assumption was tested first, on 100 training
broadcasts per source (2,024 clips, `output/audit_train_memo*`):

| B_s0 within-broadcast top-1, pools ~6-7 | informburo | khabar | qazaqstantv |
|---|---|---|---|
| training clips (seen) | 0.938 | 0.954 | 0.689 |
| dev clips, E12 (unseen) | 0.925 | 0.967 | 0.670 |

**No advantage on seen clips** -- the model has not memorised its training
pairs. Equal aggregates could still hide memorised *wrong* pairs, so the
decisive check was enrichment: on the training sample, clips with implausible
speech rate are flagged 43.9% vs 2.3% (**x18.8**, stronger than dev's x12.5).
Memorisation is not masking bad pairs. **B_s0 audits train directly; cross-
fitting dropped** (saves ~10 GPU-hours and the risk that half-data models
never take off). `scripts/build_crossfit_halves.py` and `train_min.py
--train-videos` remain available.

The sample also replicated the dev reversal, more strongly: qazaqstantv clips
with human transcripts flagged 16.9% [14.0, 20.3] vs 3.9% [1.3, 10.8] for
speech-recognition text.

Full train audit launched: `run_train_audit.sh` -> `output/audit_train_report/`.
