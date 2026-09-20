#!/usr/bin/env python3
"""
E3 arm D: fit the typical hand size, in upstream Uni-Sign body-box units, on
TRAIN only.

Arm D rescales each hand per frame so its extent equals this ratio (see
data/utils.py::unisign_part_features, hand_ratio). Using the corpus median
rather than a fixed [-1, 1] box keeps hand coordinates in the value range the
pretrained Uni-Sign weights were trained on, while removing per-signer hand
size and camera distance.

Measured on the upstream pipeline's own output, so the ratio is expressed in
exactly the units the model will see.

Usage:
    python scripts/fit_hand_ratio.py --root ~/asan_canonical
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.utils import unisign_part_features, _hand_extent, _US_THR  # noqa: E402

SOURCES = ('informburo', 'khabar', 'qazaqstantv')


def load_pose(path, archive):
    p = path if os.path.isabs(path) else os.path.join(archive, path)
    if p.endswith('.npz'):
        d = np.load(p)
        return np.asarray(d['wb_xy'], np.float32), np.asarray(d['wb_score'], np.float32)
    d = pickle.load(open(p, 'rb'))
    wb, sc = np.asarray(d['keypoints'], np.float32), np.asarray(d['scores'], np.float32)
    return (wb[:, 0], sc[:, 0]) if wb.ndim == 4 else (wb, sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='~/asan_canonical')
    ap.add_argument('--archive', default='/data/archive/asan-dataset')
    ap.add_argument('--out', default=None)
    ap.add_argument('--clips-per-source', type=int, default=150)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    root = os.path.expanduser(args.root)
    out = os.path.expanduser(args.out or os.path.join(root, 'hand_ratio.json'))

    rng = np.random.default_rng(args.seed)
    per_source, pooled = {}, []
    for src in SOURCES:
        entries = json.load(open(os.path.join(root, src, 'annotations', 'kz', 'train.json')))
        vals = []
        for i in rng.choice(len(entries), size=min(args.clips_per_source, len(entries)),
                            replace=False):
            wb, sc = load_pose(entries[i]['pose'], args.archive)
            f = unisign_part_features(wb, sc).reshape(len(wb), 69, 3)
            for hand in (f[:, 9:30], f[:, 30:51]):
                e = _hand_extent(hand, _US_THR)
                vals.append(e[np.isfinite(e) & (e > 1e-6)])
        v = np.concatenate(vals)
        per_source[src] = {'median': float(np.median(v)), 'p10': float(np.percentile(v, 10)),
                           'p90': float(np.percentile(v, 90)), 'n_frames': int(len(v))}
        pooled.append(v)
        print(f"{src:11s} median={per_source[src]['median']:.4f} "
              f"p10={per_source[src]['p10']:.4f} p90={per_source[src]['p90']:.4f} "
              f"n={len(v)}")
    allv = np.concatenate(pooled)
    ratio = float(np.median(allv))
    print(f"pooled      median={ratio:.4f}  (p10={np.percentile(allv, 10):.4f} "
          f"p90={np.percentile(allv, 90):.4f})")
    with open(out, 'w') as fh:
        json.dump({'ratio': ratio, 'fit_split': 'train', 'root': root, 'seed': args.seed,
                   'units': 'upstream Uni-Sign body-box units (hand extent / body scale)',
                   'per_source': per_source}, fh, indent=1)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
