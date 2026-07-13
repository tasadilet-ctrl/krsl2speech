"""
Raw keypoint detection-quality check, independent of the encoder/training
pipeline entirely.

diagnose_phase1.py's (B)/(B3) sections found the encoder's embeddings
collapse to near-identical across genuinely different clips (pairwise
cosine ~0.98-0.995). Six architecture/training-side explanations for that
have been ruled out by direct testing: CTC weight (0.0/0.3/1.0),
part_para/pose_proj bias, decoder cross-attention starvation, encoder LR,
and BatchNorm running-stats calibration -- see that script's git history
for each ablation.

This checks the one thing none of those could rule out: whether the
UPSTREAM pose estimator (COCO-WholeBody + dedicated hand models) is
actually tracking hand shape reliably during signing. A noisy or
low-confidence input would explain persistent collapse regardless of how
the encoder is trained -- no encoder change fixes a signal that isn't
there upstream.

Uses the exact same clip filtering as diagnose_phase1.py's AsanDataset
call (same sources/split/skip_low_quality/min_hand_cov), so clip indices
0-3 here correspond to the exact same 4 clips already tested there.

Usage:
  PYTHONPATH=. python scripts/inspect_pose_quality.py \
      --config configs/config.yaml --n 8
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np

from data.asan_dataset import AsanDataset, WB_LEFT_HAND, WB_RIGHT_HAND
from data.utils import BODY_IDX, FACE_IDX


def group_stats(sc_group, name, low_thresh):
    """
    sc_group: (T, n_joints) raw detection scores for one keypoint group.

    The raw scale of these scores is NOT assumed to be a normalized [0,1]
    confidence (asan-dataset's raw values turned out to run ~5-8, not
    0-1 -- a fixed low_thresh=0.3 would trivially pass everything and miss
    real dropouts). Prints the raw distribution (min/p5/median/p95/max) so
    the actual scale is visible, and ALSO flags per-frame dips RELATIVE to
    this clip's own median (scale-agnostic), alongside the fixed-threshold
    check for reference.
    """
    nan_frac = float(np.isnan(sc_group).mean())
    sc_clean = np.nan_to_num(sc_group, nan=0.0)
    frame_conf = sc_clean.mean(axis=-1)  # (T,) — per-frame group confidence

    lo, p5, p50, p95, hi = np.percentile(frame_conf, [0, 5, 50, 95, 100])
    low_frac = float((frame_conf < low_thresh).mean())

    # Relative dip: frames below 30% of this clip's own median -- catches
    # dropouts regardless of what absolute scale the scores are on.
    rel_thresh = 0.3 * p50 if p50 > 0 else low_thresh
    rel_mask = frame_conf < rel_thresh
    rel_frac = float(rel_mask.mean())
    longest_run, cur = 0, 0
    for v in rel_mask:
        cur = cur + 1 if v else 0
        longest_run = max(longest_run, cur)

    print(f"    {name:10s}: range=[{lo:.3f}, {p5:.3f}, {p50:.3f}, {p95:.3f}, "
          f"{hi:.3f}] (min/p5/median/p95/max)  nan_frac={nan_frac:.3f}")
    print(f"      fixed low_conf(<{low_thresh})_frac={low_frac:.3f}  |  "
          f"relative dip(<30% of own median)_frac={rel_frac:.3f}  "
          f"longest_run={longest_run} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--n', type=int, default=8, help='clips to inspect')
    ap.add_argument('--low-thresh', type=float, default=0.3,
                    help='per-frame group confidence below this counts as '
                         'a detection dropout')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    from utils.paths import apply_env_overrides
    cfg = apply_env_overrides(cfg)
    acfg = cfg['paths']['asan']

    ds = AsanDataset(root=acfg['root'], sources=acfg.get('sources'),
                     lang=acfg.get('lang', 'kz'), split='train',
                     downsample_every=acfg.get('downsample_every', 1),
                     skip_low_quality=True)

    n = min(args.n, len(ds))
    if n == 0:
        print(f"\n[inspect] ERROR: 0 clips in the dataset. Check "
              f"'paths.asan.root' ({acfg['root']}) exists on this machine, "
              f"or set ASAN_ROOT.")
        sys.exit(1)

    print(f"\nInspecting {n} clips for raw pose-detection quality "
          f"(low-confidence threshold={args.low_thresh})\n")

    for i in range(n):
        entry = ds.clips[i]
        try:
            wb, sc = ds._load_pose(entry)  # (T, 133, 2), (T, 133)
        except Exception as e:
            print(f"clip {i} ({entry.get('clip_id', '?')}): FAILED to load: {e}\n")
            continue
        T = wb.shape[0]
        print(f"clip {i} ({entry.get('clip_id', '?')}, T={T}): "
              f"\"{entry.get('text', '')[:60]}\"")
        group_stats(sc[:, WB_LEFT_HAND], 'left_hand', args.low_thresh)
        group_stats(sc[:, WB_RIGHT_HAND], 'right_hand', args.low_thresh)
        group_stats(sc[:, BODY_IDX], 'body', args.low_thresh)
        group_stats(sc[:, FACE_IDX], 'face', args.low_thresh)
        print()

    print("Interpretation: high nan_frac / low_conf_frac / long low_conf "
          "runs on left_hand or right_hand -- especially if they differ a "
          "lot between clips that the model confuses with each other -- "
          "would mean the upstream pose estimator isn't reliably capturing "
          "the hand shapes that actually distinguish signs, independent of "
          "anything in the encoder.")


if __name__ == '__main__':
    main()
