# Pending: clean qazaqstantv re-collection

**Status: not yet accessible, deferred (not blocking the current ablation).**

The qazaqstantv portion of asan-dataset was reported dirty and has been
re-collected by a colleague. It lives at
`<colleague-home>/qazaqstantv_recollect/` on the training box,
alongside a `training_manifest_qazaqstantv.jsonl`.

**Why it matters:** qazaqstantv is 13,493 of our 40,240 training clips
(~33%). Dirty labels/alignment across a third of the training set is a
plausible contributor to our weak absolute numbers (WER 0.919).

**Blocker:** the colleague's home is `drwxr-x---` (owner only), so our
account cannot read it. Needs the owner to run:
```bash
chmod o+x <colleague-home>
chmod -R o+rX <colleague-home>/qazaqstantv_recollect
```
(or copy to a shared location / add a common group).

**Decision (2026-08-21):** launch the prosody-supervision ablation on the
CURRENT data rather than waiting. Both arms use identical data, so the A/B
comparison isolating the prosody effect stays valid regardless of data
quality. Absolute numbers may improve later with clean data.

## When access is granted — verify before swapping in

Do NOT assume it's a drop-in replacement. Check:
1. Structure matches what `AsanDataset` expects:
   `annotations/kz/{train,dev,test}.json`, `pose/kz/processed/...`,
   `videos/`, `audio/` — or is it only a flat manifest?
2. Clip IDs: same scheme? If they changed, the prosody extraction
   (`data/asan_prosody_v4`, keyed by `clip_id`) must be re-run for
   qazaqstantv or it will silently return blank samples.
3. Split integrity: are train/dev/test still video-disjoint, and do they
   overlap the old splits? A clip that was dev and is now train would
   leak.
4. Count delta vs the old 13,546 raw / 13,493 kept.

## Re-running the ablation on clean data

If the prosody technique shows an effect on current data, re-run both arms
on clean data. Demonstrating the effect holds across two data versions
would materially strengthen the paper (rules out a data-artifact
explanation). Requires re-extracting prosody for the new qazaqstantv clips
first.
