# Clean qazaqstantv recollection — INTEGRATED & VERIFIED (2026-09-17)

Access was granted by adding our account to the owning group (traverse ACL
on the parent; the `qazaqstantv_recollect` dir itself is world-readable).
Verification from the
old checklist is complete and the data is wired up.

## Result

```bash
export ASAN_ROOT=$HOME/asan_clean
```

Built by `scripts/build_qazaqstantv_recollect_root.py`. Layout:

| path | contents |
|---|---|
| `qazaqstantv/annotations/kz/{train,dev,test}.json` | generated, pointing at the recollection's absolute `.npz` paths |
| `khabar`, `informburo` | symlinks to `/data/archive/asan-dataset/...`, **unchanged** |

`/data/archive/asan-dataset` is our original training data and is intact.
`~/asan_local` (the old `ASAN_ROOT`) no longer exists — the box was rebuilt.

## Counts

| split | archive | clean | delta |
|---|---|---|---|
| train | 40,240 | **52,835** | +12,595 (+31.3%) |
| dev | 8,902 | 11,496 | +2,594 |
| test | 8,795 | 11,448 | +2,653 |

qazaqstantv train alone: 13,493 → 26,088 (+93%). The archive figures
reproduce our recorded numbers exactly, confirming the archive is what we
trained on.

## Checklist results

1. **Structure** — differs. New data is jsonl manifests + per-clip `.npz`
   (`wb_xy`, `wb_score`, `hand_*`); the archive is `annotations/kz/*.json` +
   per-clip `.pkl` (`keypoints`, `scores`). Handled by the builder script plus
   a `.npz` branch in `AsanDataset._load_pose`.
2. **Clip IDs** — changed: `qazaqstantv__kazsign_218288__seg00000` →
   `qazaqstantv_kazsign_218286_000`. Moot for prosody (dropped), but any
   future clip_id-keyed side data must be regenerated.
3. **Split integrity** — clean. Zero video-level overlap between train/dev/test
   in both the qazaqstantv and unified manifests.
4. **Count delta** — above.
5. **Frame rate** — unchanged, 30 fps in both (config.yaml's "videos are 50 fps"
   comment is stale; `downsample_every: 2` yields 15 fps, as before).
6. **Feature distribution** — matches. Enriched features from the clean root vs
   the archive: mean 0.2027 vs 0.2026, std 0.6033 vs 0.6031, dim 1410 both,
   no NaN. The swap introduces no preprocessing shift.
7. **`frame_start`/`frame_end`** — the recollection uses `0/0` as a "whole file"
   sentinel. Naively slicing on it yields an EMPTY clip; `_load_pose` only
   slices when `frame_end > frame_start`.

## Why qazaqstantv only, not `training_manifest_unified.jsonl`

The unified manifest also re-segments khabar (train 34,696 vs archive 21,524)
and informburo (10,205 vs 5,249), because it is built from the broadcasters'
raw manifests rather than the archive's filtered annotations. Using it would
change all three sources at once, so a quality change could not be attributed
to the qazaqstantv fix. Worth evaluating separately later.

## Caveat that matters for the paper

**The dev/test sets changed** (8,795 → 11,448 test clips). Metrics on the clean
root are NOT comparable to the recorded WER 0.919 / BLEU 3.90. Re-run the
baseline on the clean root before drawing any conclusion about whether the
data fix helped.

## Known follow-up

The `.npz` files carry `hand_l_xy`/`hand_r_xy` from a dedicated hand detector,
NaN where it failed — plausibly higher quality than the wholebody model's hand
joints, which is what `_load_pose` currently uses. Not adopted: it would change
hands for qazaqstantv only, confounding the comparison.

Bug found in the original data (from the colleague's `FINDINGS.md`): the old
pipeline called Whisper with no `chunk_length_s`, so on 45–90s audio the
timestamps degenerated and a 128-word paragraph could be attached to a 0.74s
window — 2,507/19,490 clips (12.9%) affected. The fix adds chunking plus a
words-per-second plausibility gate (0.3–6.0 wps).
