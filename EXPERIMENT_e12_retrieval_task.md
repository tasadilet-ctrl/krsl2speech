# E12: sign -> text retrieval as the deliverable task

Status: **pre-registered 2026-10-12.** Protocol and readings are fixed before
any test-split number exists. Results go at the end.

## Why

Free generation has not left the text prior in 25+ runs (WER ~0.95). Retrieval
with the same checkpoints ranks the right sentence first of 200 ~85% of the time
on dev, replicated on two takeoff checkpoints from different arms, with
duration, memorisation and non-takeoff controls passed (NOTES_retrieval.md).
Before calling it a deliverable it needs (1) the realistic setting and (2) a
held-out number.

## Tasks

**T1 -- cross-broadcast search.** Query: one signed clip. Candidates: 200
sentences from 200 other videos of the same source, plus its own. As measured
on dev so far. Use: find which news item a signed clip belongs to.

**T2 -- within-broadcast alignment.** Query: one clip. Candidates: every
sentence of the *same* video (videos with >= 5 clips; median ~6). All
candidates share topic, names and vocabulary, so this is harder per candidate
than T1 even with far fewer candidates. Use: align an interpreter track to a
broadcast's transcript, line by line -- also a data-cleaning tool for this
corpus.

Both are scored by PMI: log p(text | pose) - mean log p(text | other poses),
using the checkpoint's own decoder; null poses come from other videos of the
same source. Candidates are always same-source (qazaqstantv text is cased and
punctuated, the others are not).

Metrics: top-1, top-5 (T1), MRR, median rank; chance reported alongside (T2's
chance varies with pool size: top-1 = mean of 1/n).

## Protocol

1. **Manifests** (`scripts/build_retrieval_manifests.py`, salt
   `krsl-canonical-v1`): T1 test mirrors the frozen dev selection exactly (one
   clip per video, four duration quartiles, 200 per source). T2 dev and T2 test
   take up to 60 videos per source with >= 5 clips, all their clips.
2. **Checkpoint selection on dev only**: B_s0, E4b_s1, E9b_s2 (the three best
   takeoff runs, two arms plus the 30 fps arm). Pick the highest **T2 dev MRR**
   -- T2 because it is the harder, realistic task. Tie (within 0.01): prefer
   the one with clean, stride-2 provenance (B_s0).
3. **Test split, once**, with the selected checkpoint: T1 and T2. No further
   choices made after seeing test numbers.

## Readings (pre-committed)

| Outcome on test | Reading |
|---|---|
| T1 top-1 within 0.05 of dev (~0.85) | Cross-broadcast search generalises; report as a result. |
| T1 top-1 more than 0.10 below dev | Dev was optimistic (selection effects); report the test number as the result. |
| T2 MRR >= 0.70 and top-1 >= 3x chance | Within-broadcast alignment works well enough to use for data cleaning and as a demo. |
| T2 top-1 < 2x chance | Same-topic discrimination fails: the model identifies the *story*, not the *sentence*. T1 is then a topic-matching result and must be described that way. |

## Not in scope

Speed. Scoring is one decoder pass per (clip, candidate) pair: fine for pools
of ~10-200, not for searching a whole archive. A dual-encoder index would be
the follow-up if T1/T2 hold.

## Results -- dev phase and checkpoint selection (2026-10-12, before any test number)

Manifests built by `scripts/build_retrieval_manifests.py`; the generalised
builder reproduces the frozen dev selection exactly. T2 dev: 1,085 clips in
180 broadcasts (60 per source), mean pool 6.4.

| checkpoint | T2 dev MRR | top-1 | informburo / khabar / qazaqstantv top-1 |
|---|---|---|---|
| B_s0 | 0.902 | 0.851 | 0.925 / 0.967 / 0.670 |
| E4b_s1 | 0.909 | 0.857 | 0.936 / 0.962 / 0.684 |
| E9b_s2 | **0.910** | 0.862 | 0.952 / 0.959 / 0.684 |
| chance | 0.404 | 0.166 | |

**Selection by the pre-registered rule:** E9b_s2 has the highest MRR, but all
three are within 0.008 -- inside the 0.01 tie band -- so the tie goes to the
clean stride-2 checkpoint: **B_s0**. Its deficit is noise-level and it needs no
non-default input pipeline.

**Error structure (dev, all three checkpoints):** 52-57% of wrong picks are the
adjacent sentence, against 33-34% if errors were random; only 22-27% are three
or more sentences away. Broadcasts -- khabar bulletins especially -- hold
several stories, so the model finds the story and its mistakes are mostly
confusions between neighbouring sentences of that story.

qazaqstantv is the weak source on T2 (0.67-0.68 top-1 against 0.93-0.97). Its
pools are not much larger (6.8 vs 5.7-6.4); its transcripts are cased and
punctuated text_human with ~10% fragment references, and its clips have the
smallest hands (31 px). Which of these matters is not separated here.

Also verified: `scripts/sign_retrieve.py` (the user-facing tool) ranked the
correct line first for all 3 clips of a 6-sentence khabar bulletin, run on CPU.
