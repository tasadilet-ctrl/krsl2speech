#!/usr/bin/env python3
"""
How often are the whole-body hand keypoints (the ones the model trains on)
GUESSED rather than seen?

The whole-body model (RTMW) reports both hands in 100% of frames and the
Uni-Sign masking threshold (score <= 0.3) never fires on its unbounded scores,
so the model cannot tell a seen hand from an invented one. Two independent
visibility signals:

  OUT    geometry: the hand's root (wrist) or most of its points lie outside the
         video frame -- certainly not visible.
  NODET  the dedicated hand detector (hand_l_xy / hand_r_xy, qazaqstantv only)
         found no hand in that frame. Its finger points were judged bad by eye,
         but whether it found a hand at all is a separate question -- verified
         here with a contact sheet, not assumed.

Then: does RTMW's own hand score separate those frames (AUC)? If so, a
calibrated threshold could mask guessed hands.

    PYTHONPATH=. python scripts/measure_guessed_hands.py --n 200 --sheet ~/qazaq_viewer/guessed_sheet.png
"""
import argparse
import json
import os
import pickle
import random

import numpy as np

ROOT = os.path.expanduser('~/asan_canonical')
SIZE = {'informburo': (250, 370), 'khabar': (314, 282), 'qazaqstantv': (204, 186)}   # W, H
HANDS = {'L': slice(91, 112), 'R': slice(112, 133)}


def load(entry):
    p = entry['pose'] if os.path.isabs(entry['pose']) else os.path.join(ROOT, entry['pose'])
    if p.endswith('.npz'):
        z = np.load(p)
        return z['wb_xy'], z['wb_score'], {'L': z['hand_l_xy'], 'R': z['hand_r_xy']}, p
    d = pickle.load(open(p, 'rb'))
    xy, sc = np.asarray(d['keypoints'], np.float32), np.asarray(d['scores'], np.float32)
    if xy.ndim == 4:
        xy, sc = xy[:, 0], sc[:, 0]
    return xy, sc, None, p


def auc(pos, neg):
    """P(score of a 'visible' frame > score of a 'guessed' frame)."""
    if not len(pos) or not len(neg):
        return float('nan')
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200, help='clips per source')
    ap.add_argument('--split', default='dev')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sheet', default=None, help='write a contact sheet of flagged frames')
    a = ap.parse_args()
    rng = random.Random(a.seed)
    examples = []

    for src, (W, H) in SIZE.items():
        ents = [e for e in json.load(open(os.path.join(ROOT, src, 'annotations', 'kz', f'{a.split}.json')))
                if e.get('text', '').strip()]
        rng.shuffle(ents)
        tot = out = nodet = either = 0
        s_vis, s_out, s_nodet = [], [], []
        swapped = same = 0
        for e in ents[:a.n]:
            xy, sc, ded, path = load(e)
            for h, sl in HANDS.items():
                hp, hs = xy[:, sl], sc[:, sl]
                inside = (hp[..., 0] >= 0) & (hp[..., 0] < W) & (hp[..., 1] >= 0) & (hp[..., 1] < H)
                o = ~inside[:, 0] | (inside.mean(1) < 0.5)
                n = np.zeros(len(hp), bool)
                if ded is not None:
                    n = np.isnan(ded[h]).all(axis=(1, 2))
                    other = 'R' if h == 'L' else 'L'
                    ok = ~n & ~np.isnan(ded[other]).all(axis=(1, 2))
                    if ok.any():   # check side convention: our hand nearer its namesake?
                        d_same = np.nanmedian(np.linalg.norm(hp[ok] - ded[h][ok], axis=-1))
                        d_swap = np.nanmedian(np.linalg.norm(hp[ok] - ded[other][ok], axis=-1))
                        same += d_same < d_swap; swapped += d_swap <= d_same
                ms = hs.mean(1)
                tot += len(hp); out += o.sum(); nodet += n.sum(); either += (o | n).sum()
                s_out.append(ms[o]); s_nodet.append(ms[n & ~o]); s_vis.append(ms[~o & ~n])
                for t in np.where(n | o)[0]:
                    examples.append((src, path, e['clip_id'], int(t), h, float(ms[t]),
                                     'OUT' if o[t] else 'NODET'))
        sv, so, sn = (np.concatenate(x) for x in (s_vis, s_out, s_nodet))
        print(f'\n=== {src} ({a.n} {a.split} clips, {tot:,} hand-frames, video {W}x{H})')
        print(f'  OUT   hand outside the frame            {100 * out / tot:5.2f}%')
        if src == 'qazaqstantv':
            print(f'  NODET dedicated detector found no hand  {100 * nodet / tot:5.2f}%')
            print(f'  either                                  {100 * either / tot:5.2f}%')
            print(f'  (side convention check: our L/R nearer the detector\'s same side in '
                  f'{same}/{same + swapped} clip-hands)')
        print(f'  RTMW mean hand score  visible: median {np.median(sv):.2f}  p10 {np.percentile(sv, 10):.2f}'
              + (f'  |  OUT: median {np.median(so):.2f}' if len(so) else '')
              + (f'  |  NODET: median {np.median(sn):.2f}' if len(sn) else ''))
        if len(so):
            print(f'  AUC, score separating visible from OUT:   {auc(sv, so):.3f}')
        if len(sn):
            print(f'  AUC, score separating visible from NODET: {auc(sv, sn):.3f}')
        frac_masked = np.mean(np.concatenate([sv, so, sn]) <= 0.3)
        print(f'  share of hand-frames the current 0.3 threshold masks: {100 * frac_masked:.2f}%')

    if a.sheet and examples:
        import cv2
        rng.shuffle(examples)
        pick = [x for x in examples if x[0] == 'qazaqstantv'][:24]
        tiles = []
        for src, path, cid, t, h, ms, why in pick:
            vid = path.replace('/keypoints/', '/videos/').replace('.npz', '.mp4')
            cap = cv2.VideoCapture(vid); cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok_, fr = cap.read()
            if not ok_:
                continue
            xy = np.load(path)['wb_xy'][t]
            img = cv2.resize(fr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            for j in range(*HANDS[h].indices(133)):
                if not np.isnan(xy[j]).any():
                    cv2.circle(img, (int(xy[j][0] * 2), int(xy[j][1] * 2)), 2, (0, 0, 255), -1)
            cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
            cv2.putText(img, f'{why} {h} s={ms:.1f}', (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(cv2.resize(img, (408, 372)))
        rows = [np.hstack(tiles[i:i + 6] + [np.zeros_like(tiles[0])] * (6 - len(tiles[i:i + 6])))
                for i in range(0, len(tiles), 6)]
        cv2.imwrite(os.path.expanduser(a.sheet), np.vstack(rows))
        print(f'\n[write] contact sheet: {a.sheet} ({len(tiles)} flagged qazaqstantv frames; '
              f'red dots = the whole-body hand points the model receives for that hand)')


if __name__ == '__main__':
    main()
