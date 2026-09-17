#!/usr/bin/env python3
"""
Build an ASAN_ROOT that swaps in the clean qazaqstantv recollection while
leaving khabar and informburo untouched.

Why only qazaqstantv: the colleague's `training_manifest_unified.jsonl` also
re-segments khabar and informburo (train 34,696 / 10,205 vs the archive's
21,524 / 5,249), because it is built from the broadcasters' raw manifests
rather than from the archive's filtered `annotations/kz/*.json`. Swapping all
three at once would change every source simultaneously, so a quality change
could not be attributed to the qazaqstantv fix. This script changes exactly
one variable.

Layout produced at --out:
    khabar       -> symlink to <archive>/khabar        (unchanged)
    informburo   -> symlink to <archive>/informburo    (unchanged)
    qazaqstantv/annotations/kz/{train,dev,test}.json   (generated)

The generated annotations use the AsanDataset entry schema (`pose`, `text`,
`clip_id`, `T`, `hand_l_cov`, `hand_r_cov`, `low_quality`), with `pose`
holding the recollection's ABSOLUTE .npz path — os.path.join(root, abs_path)
returns the absolute path unchanged, so no copying of the 18 GB keypoint
tree is needed. data/asan_dataset.py::_load_pose reads both .pkl and .npz.

Usage:
    python scripts/build_qazaqstantv_recollect_root.py \
        --manifest <colleague-home>/qazaqstantv_recollect/training_manifest_qazaqstantv.jsonl \
        --archive  /data/archive/asan-dataset \
        --out      ~/asan_clean \
        --workers 12
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

WB_LEFT_HAND = slice(91, 112)    # 21 points, matches data/asan_dataset.py
WB_RIGHT_HAND = slice(112, 133)

# Hand coverage = fraction of frames where the DEDICATED hand detector fired.
# In these .npz files `hand_l_score`/`hand_r_score` are NaN on frames where
# that detector found no hand, which is the only informative presence signal
# available: `wb_score` holds raw RTMPose heatmap responses (unbounded, median
# ~5.4, never exactly 0), so any "score > 0" test is trivially true and yields
# a useless constant 1.0.
#
# NOT COMPARABLE to the archive's hand_l_cov/hand_r_cov: those came from a
# pipeline whose code we do not have. Do not set `min_hand_cov` above 0.0
# while mixing sources — it would filter qazaqstantv on a different definition
# than khabar/informburo. It defaults to 0.0 in config.yaml.

# Both hands essentially invisible for the whole clip => it cannot carry sign
# content. Mirrors the intent of the archive's `low_quality` flag.
_LOW_QUALITY_COV = 0.1


def probe(rec):
    """Read one .npz and return the quality fields AsanDataset filters on."""
    path = rec['keypoints_path']
    try:
        with np.load(path) as d:
            sc = np.asarray(d['wb_score'], dtype=np.float32)       # (T, 133)
            hl = np.asarray(d['hand_l_score'], dtype=np.float32)   # (T, 21)
            hr = np.asarray(d['hand_r_score'], dtype=np.float32)
            idx = np.asarray(d['frame_idx'])
        fs, fe = rec.get('frame_start', 0), rec.get('frame_end', 0)
        if fe > fs:
            keep = (idx >= fs) & (idx < fe)
            sc, hl, hr = sc[keep], hl[keep], hr[keep]
        T = int(sc.shape[0])
        if T == 0:
            return rec['clip_id'], None
        l_cov = float((~np.isnan(hl).all(axis=1)).mean())
        r_cov = float((~np.isnan(hr).all(axis=1)).mean())
        return rec['clip_id'], {
            'pose': path,
            'text': rec['text'],
            'clip_id': rec['clip_id'],
            'video_id': rec.get('video_id'),
            'T': T,
            'hand_l_cov': round(l_cov, 4),
            'hand_r_cov': round(r_cov, 4),
            'low_quality': bool(max(l_cov, r_cov) < _LOW_QUALITY_COV),
            'frame_start': rec.get('frame_start', 0),
            'frame_end': rec.get('frame_end', 0),
        }
    except Exception as exc:                                   # noqa: BLE001
        return rec['clip_id'], {'_error': f'{type(exc).__name__}: {exc}'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--archive', default='/data/archive/asan-dataset')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()

    out = os.path.expanduser(args.out)
    archive = os.path.expanduser(args.archive)

    records = [json.loads(l) for l in open(args.manifest) if l.strip()]
    print(f"[read] {len(records)} clips from {args.manifest}")

    by_split = {}
    errors, empty = [], []
    done = 0
    # 12 workers, not 40: the prosody extraction on this box hit
    # BrokenProcessPool at 40 and ran clean at 12.
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(probe, r): r for r in records}
        for fut in as_completed(futs):
            clip_id, entry = fut.result()
            done += 1
            if done % 2000 == 0:
                print(f"  probed {done}/{len(records)}", flush=True)
            if entry is None:
                empty.append(clip_id)
                continue
            if '_error' in entry:
                errors.append((clip_id, entry['_error']))
                continue
            by_split.setdefault(futs[fut]['split'], []).append(entry)

    ann_dir = os.path.join(out, 'qazaqstantv', 'annotations', 'kz')
    os.makedirs(ann_dir, exist_ok=True)
    for split, entries in sorted(by_split.items()):
        entries.sort(key=lambda e: e['clip_id'])
        dest = os.path.join(ann_dir, f'{split}.json')
        with open(dest, 'w') as fh:
            json.dump(entries, fh, ensure_ascii=False)
        lq = sum(e['low_quality'] for e in entries)
        print(f"[write] {dest}: {len(entries)} clips ({lq} low_quality)")

    for src in ('khabar', 'informburo'):
        link, target = os.path.join(out, src), os.path.join(archive, src)
        if not os.path.exists(target):
            sys.exit(f"[fatal] missing archive source: {target}")
        if os.path.islink(link):
            os.unlink(link)
        elif os.path.exists(link):
            sys.exit(f"[fatal] {link} exists and is not a symlink; refusing to touch it")
        os.symlink(target, link)
        print(f"[link] {link} -> {target}")

    if empty:
        print(f"[warn] {len(empty)} clips had zero frames after slicing "
              f"(first 5: {empty[:5]})")
    if errors:
        print(f"[warn] {len(errors)} clips failed to load:")
        for clip_id, msg in errors[:10]:
            print(f"    {clip_id}: {msg}")

    print(f"\nDone. Use:  export ASAN_ROOT={out}")


if __name__ == '__main__':
    main()
