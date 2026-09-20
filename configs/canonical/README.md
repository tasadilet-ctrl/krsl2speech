# Canonical evaluation artifacts

The two files that every valid number in this repo depends on. They lived only
on the training box, which has already been rebuilt once (taking the old
`~/asan_local` with it), so they are committed here.

| file | what it is |
|---|---|
| `selection_manifest.json` | the frozen 600-clip selection set, 200 per source: `clip_id`, `source`, `video_id`, frame count `T` |
| `score_quantiles.json` | per-keypoint-group score quantiles fitted on train only, for `--score-quantiles` |

No transcripts, poses or media — identifiers and numbers only, so nothing from
the restricted corpora is redistributed. The clips themselves still have to
come from the dataset owners (see the README's data availability section).

Used as:

```bash
python train/train_encoder_mt5.py \
  --selection-manifest configs/canonical/selection_manifest.json \
  --score-quantiles   configs/canonical/score_quantiles.json
```

Regenerate with `scripts/build_canonical_splits.py` and
`scripts/fit_score_quantiles.py`. Regenerating changes the selection set, so
numbers computed against a new manifest are not comparable to the ones in the
README — treat these as frozen.
