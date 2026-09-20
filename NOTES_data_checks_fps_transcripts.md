# Data checks: frame rate and qazaqstantv transcripts — 2026-09-18

## 1. Frame rate differs by source; the pipeline treats all sources alike

ffprobe on 300 random canonical-TRAIN clips per source:

| source | native fps | after `downsample_every=2` | median frames (post-downsample) |
|---|---|---|---|
| informburo | **50** (100%) | 25 | 430 |
| khabar | **25** (98%); 29.97, 30, 60 rare | 12.5 | 202 |
| qazaqstantv | **30** (99%); 25 (1%) | 15 | 223 |

- Pose is extracted at the native rate: pose frames == video frames for every
  probed clip.
- Real clip durations are similar (~16–17 s median), so **the same second of
  signing is ~2× as many frames for informburo as for khabar**. The kernel-5
  temporal convs cover 0.2 s of motion on informburo but 0.4 s on khabar, and
  informburo feeds mT5 roughly twice as many input tokens.
- No truncation: `max_seq_len` is 1000 post-downsample frames; the longest
  clips (informburo p99) reach 500.
- `config.yaml`'s "videos are 50 fps → 25 fps" comment is true only for
  informburo (an earlier note calling it stale was wrong for informburo).
  `compute_velocity` also assumes 50 fps.

**Does not invalidate E3:** all arms share identical temporal handling.
**Is a confound for per-source comparisons:** informburo is the weakest source
in every E3 arm (arm B ep5: 20.6 chrF vs khabar 25.7, qazaqstantv 23.0), and it
is also the source whose temporal scale differs most. Plausible link, untested.

**Fix (needs its own arm):** resample every clip to one target rate using its
real fps — `qazaqstantv_kz_fixed.jsonl` carries `fps` per clip; the archive
sources need a one-off ffprobe pass (khabar has ~2% non-25 clips, so a per-source
constant is not enough). Target to choose: 12.5 fps halves informburo's cost;
25 fps keeps its resolution but doubles khabar/qazaqstantv sequence length.

## 2. qazaqstantv trains on `text_human`, with `text_asr` fallback

The recollection stores two transcripts per clip: `text_asr` (Whisper on the
broadcast audio — what the interpreter is actually signing) and `text_human`
(website article sentences fuzzy-aligned to the clip window). The training
manifest's single `text` field, verified clip-by-clip:

- **89.8%** `text_human`, **10.2%** `text_asr` (where no `text_human` match
  exists). 0 clips match neither. Canonical train: 28,740 / 3,528.
- So one source mixes two target styles: article prose and raw ASR (with ASR
  artifacts like "ммм"). khabar/informburo each use a single caption-style text.

**The §13 slicing bug is fixed in what we train on.** Clips sharing an identical
`text_human` with a sibling clip: pre-fix backup 43.2% → current file 4.2% →
canonical train 4.0%, dev 2.7%, test 2.3%. File timestamps agree (backup 22:29,
fixed files 23:35/23:54 on 2026-08-31).

**Residual problem: fragment references.**
- `text_human` vs `text_asr` on the same clips: corpus chrF 62.5 (same content,
  different wording, mostly); median per-clip chrF 68.6, but **14.1% of clips
  score below 30**.
- **3,283 canonical-train clips (10.2% of qazaqstantv train) have a
  `text_human` with fewer than half the words ASR heard.** Example: 19.0 s of
  speech, reference `"Бес есе көп"` (3 words). These train the model to emit a
  fragment for a full clip of signing.
- Visible in words/sec: qazaqstantv p10 **0.87** vs 1.34 informburo / 1.39 khabar;
  medians are similar (1.85 vs 1.91/1.95).
- Also in evaluation: 7.0% of canonical dev, 8.0% of test — but only **7 of the
  200 qazaqstantv selection clips**, so selection metrics are barely affected.

**Options, to test as a data arm rather than assume:** (a) for fragment clips
use `text_asr`; (b) drop them from train; (c) switch qazaqstantv to `text_asr`
entirely (matches what is signed, but ASR errors and a style unlike the other
sources). Any change to train targets must be mirrored in dev/test, or scored on
a filtered eval set, or the comparison measures a moving reference.
