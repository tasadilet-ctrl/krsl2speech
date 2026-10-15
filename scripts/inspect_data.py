#!/usr/bin/env python3
"""
Human-readable data-quality report per source. Run on the box.

  python3 inspect_data.py                 # 8 clips per source
  python3 inspect_data.py --n 20          # more
  python3 inspect_data.py --source qazaqstantv --n 30
  python3 inspect_data.py --split dev

For each sampled clip it prints the reference text and the pose checks that
matter: how many frames are all-zero (detector gave up), NaN fraction, how
often the hand keypoints fall below the confidence floor, and the clip's
duration in SECONDS at its source frame rate -- which is the thing that is NOT
normalised anywhere in the pipeline.
"""
import argparse, json, os, random
import numpy as np

ROOT = os.path.expanduser('~/asan_canonical')
SOURCES = ('qazaqstantv', 'khabar', 'informburo')
# Inferred from words/frame (informburo 0.038, khabar 0.078, qazaqstantv 0.061)
# against a ~2 words/s speech rate; matches the fps recorded in the data notes.
FPS = {'informburo': 50, 'khabar': 25, 'qazaqstantv': 30}
# COCO-WholeBody: 0-16 body, 17-22 feet, 23-90 face, 91-111 left hand, 112-132 right
LH, RH = slice(91, 112), slice(112, 133)


def load_pose(root, entry):
    p = entry['pose']
    p = p if os.path.isabs(p) else os.path.join(root, p)
    if p.endswith('.npz'):
        d = np.load(p)
        return np.asarray(d['wb_xy'], np.float32), np.asarray(d['wb_score'], np.float32)
    import pickle
    with open(p, 'rb') as f:
        d = pickle.load(f)
    wb, sc = np.asarray(d['keypoints'], np.float32), np.asarray(d['scores'], np.float32)
    return (wb[:, 0], sc[:, 0]) if wb.ndim == 4 else (wb, sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--split', default='train')
    ap.add_argument('--source', default=None)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    srcs = [a.source] if a.source else list(SOURCES)

    for src in srcs:
        f = os.path.join(a.root, src, 'annotations', 'kz', f'{a.split}.json')
        if not os.path.exists(f):
            print(f'\n### {src}: no {a.split}.json'); continue
        d = json.load(open(f))
        fps = FPS.get(src, 30)
        print(f'\n{"=" * 78}\n### {src}  ({len(d)} clips in {a.split}, {fps} fps)\n{"=" * 78}')
        agg = []
        for e in rng.sample(d, min(a.n, len(d))):
            try:
                wb, sc = load_pose(a.root, e)
            except Exception as ex:
                print(f'\n  [{e.get("clip_id")}] POSE LOAD FAILED: {ex}'); continue
            T = len(wb)
            zero = float((np.abs(wb).sum(axis=(1, 2)) == 0).mean())
            nan = float(np.isnan(wb).mean())
            # RTMPose scores here are UNBOUNDED (observed ~0.4-11.5, median ~8),
            # not probabilities -- a fixed 0.3 floor reports 0% for everything
            # and tells you nothing. Use this clip's own weakest 5% instead.
            thr = np.percentile(sc, 5)
            lo_l = float((sc[:, LH] < thr).mean())
            lo_r = float((sc[:, RH] < thr).mean())
            secs = T / fps
            words = len(e.get('text', '').split())
            agg.append((secs, words / max(secs, 1e-6), zero, lo_l, lo_r))
            print(f'\n  clip  {e.get("clip_id")}   video {e.get("video_id") or e.get("video")}')
            print(f'  frames {T}  = {secs:5.1f} s at {fps} fps   |  {words} words '
                  f'= {words / max(secs, 1e-6):4.2f} words/s')
            print(f'  pose   all-zero frames {100 * zero:5.1f}%   NaN {100 * nan:5.2f}%   '
                  f'score<0.3  Lhand {100 * lo_l:4.1f}%  Rhand {100 * lo_r:4.1f}%')
            print(f'  TEXT   {e.get("text", "")[:150]}')
        if agg:
            A = np.array(agg)
            print(f'\n  -- {src} sample means: {A[:,0].mean():5.1f} s/clip, '
                  f'{A[:,1].mean():4.2f} words/s, all-zero {100*A[:,2].mean():4.1f}%, '
                  f'weak L {100*A[:,3].mean():4.1f}% R {100*A[:,4].mean():4.1f}%')


if __name__ == '__main__':
    main()
