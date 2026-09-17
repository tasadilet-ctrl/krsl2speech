#!/usr/bin/env python3
"""
E2: fit per-group quantile tables for detector scores, on TRAIN only.

The validity/score channel was `clip(raw_score, 0, 1)`. RTMPose wholebody
scores are unbounded heatmap responses: across informburo, khabar and
qazaqstantv, 99.6-100% of body, face and hand scores are >= 1 and none are
<= 0. After clipping the channel was a constant 1.0 -- it encoded neither
graded confidence nor detection, even though raw scores genuinely vary
(hands p10 ~3 vs p90 ~8).

This writes a monotone empirical-CDF table per assembled group so the dataset
can map raw scores into a non-saturating [0, 1] rank. It is a RANK-NORMALIZED
DETECTOR RESPONSE, not a calibrated probability: nothing here is fitted
against labelled keypoint correctness, and it must not be described as one.

Fitted on the canonical TRAIN split only, with deterministic sampling, so no
dev/test statistics leak into features.

Usage:
    python scripts/fit_score_quantiles.py --root ~/asan_canonical \
        --out ~/asan_canonical/score_quantiles.json
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.utils import assemble_joint_scores  # noqa: E402

SOURCES = ('informburo', 'khabar', 'qazaqstantv')
# Groups in the assembled 282-dim layout (x,y duplicated per joint).
GROUP_SLICES = {'body': (0, 22), 'face': (22, 198), 'hands': (198, 282)}


def raw_scores(path, archive):
    p = path if os.path.isabs(path) else os.path.join(archive, path)
    if p.endswith('.npz'):
        return np.asarray(np.load(p)['wb_score'], dtype=np.float32)
    d = pickle.load(open(p, 'rb'))
    sc = np.asarray(d['scores'], dtype=np.float32)
    return sc[:, 0] if sc.ndim == 3 else sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='~/asan_canonical')
    ap.add_argument('--archive', default='/data/archive/asan-dataset')
    ap.add_argument('--out', default=None)
    ap.add_argument('--clips-per-source', type=int, default=150)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    root = os.path.expanduser(args.root)
    out = os.path.expanduser(args.out or os.path.join(root, 'score_quantiles.json'))

    rng = np.random.default_rng(args.seed)
    pooled = {g: [] for g in GROUP_SLICES}
    n_clips = 0
    for src in SOURCES:
        entries = json.load(open(os.path.join(root, src, 'annotations', 'kz', 'train.json')))
        for i in rng.choice(len(entries), size=min(args.clips_per_source, len(entries)),
                            replace=False):
            sc = assemble_joint_scores(raw_scores(entries[i]['pose'], args.archive))
            for g, (a, b) in GROUP_SLICES.items():
                v = sc[:, a:b:2].ravel()        # one value per joint, not per x/y
                pooled[g].append(v[np.isfinite(v) & (v > 0)])
            n_clips += 1

    probs = np.linspace(0.0, 1.0, 101)
    table = {'_meta': {
        'description': 'empirical CDF of raw RTMPose wholebody scores per assembled '
                       'group; rank-normalized detector response, NOT calibrated',
        'fit_split': 'train', 'root': root, 'clips': n_clips, 'seed': args.seed,
        'probs': probs.tolist()}}
    for g, parts in pooled.items():
        v = np.concatenate(parts)
        q = np.quantile(v, probs)
        # Strictly increasing knots so np.interp is well defined.
        q = np.maximum.accumulate(q + np.arange(len(q)) * 1e-6)
        table[g] = {'slice': GROUP_SLICES[g], 'knots': q.tolist(), 'n': int(len(v))}
        print(f"{g:6s} n={len(v):9d}  p1={q[1]:.2f} p10={q[10]:.2f} "
              f"p50={q[50]:.2f} p90={q[90]:.2f} p99={q[99]:.2f}")
    with open(out, 'w') as fh:
        json.dump(table, fh, indent=1)
    print(f"wrote {out}  ({n_clips} train clips)")


if __name__ == '__main__':
    main()
