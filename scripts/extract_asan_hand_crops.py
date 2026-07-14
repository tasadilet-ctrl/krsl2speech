"""
Extract per-frame hand crops + reference points from asan-dataset video,
for Uni-Sign's real Prior-Guided Fusion (PGF, arXiv:2501.15187, ICLR 2025
-- see models/pgf_fusion.py for the fusion module itself).

Unlike scripts/extract_asan_rgb.py (whole-frame, frozen-backbone pooled
features), this crops each HAND region using that hand's own keypoint
coordinates, resizes to 112x112 (the paper's spec), and saves the RAW
(JPEG-compressed) crop -- no backbone runs here. The paper's EfficientNet-B0
hand backbone is TRAINABLE during Phase 1 training, so nothing past the
crop itself can be precomputed; the crop is the only thing that's the same
every epoch.

For every annotated clip this script:
  1. loads the pose .pkl for that clip (same file AsanDataset reads) to
     get ABSOLUTE pixel keypoint coordinates for both hands
  2. decodes video frames via scripts.extract_asan_rgb.read_frames (same
     downsample_every as pose, so hand crops stay frame-aligned 1:1)
  3. for each hand, each frame: bounding-box from the 21 hand keypoints
     (padded 30%, squared to avoid distorting the 112x112 resize), crop,
     resize, JPEG-encode. Also computes the normalized [-1,1] wrist
     reference point (grid_sample's native convention) and a per-frame
     mean keypoint confidence score, both needed at train time (the
     reference point for deformable attention, the score for score-aware
     sampling -- paper Appendix A.3 Algorithm 1).
  4. saves crops + metadata per clip; NO score-aware sampling happens
     here -- every frame is extracted so different --pgf-p-samp values can
     be experimented with later without re-extracting.

Usage:
  PYTHONPATH=. python scripts/extract_asan_hand_crops.py \
      --root /raid/shared/dataset --out ~/krsl2speech/data/asan_hand_crops \
      --splits train dev test --downsample-every 2

Note on --root: same caveat as scripts/extract_asan_rgb.py -- video files
were never symlinked into ASAN_ROOT's local workaround on Box B; point
--root at wherever {source}/videos/kz/processed/... actually lives (e.g.
/raid/shared/dataset), not $ASAN_ROOT.
"""
import os
import sys
import json
import pickle
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

from scripts.extract_asan_rgb import read_frames
from data.asan_dataset import WB_LEFT_HAND, WB_RIGHT_HAND

_SPLIT_FILES = {'train': 'train.json', 'dev': 'dev.json', 'test': 'test.json'}
_HANDS = {'left': WB_LEFT_HAND, 'right': WB_RIGHT_HAND}
# COCO-WholeBody hand layout: index 0 (local to the 21-point slice) is the
# wrist, the root of that hand's kinematic chain -- same parent-child
# convention data/utils.py::to_offset_keypoints already relies on.
_WRIST_LOCAL_IDX = 0


def _load_pose_pkl(root, entry):
    """Same file/format AsanDataset._load_pose reads: absolute pixel coords."""
    pkl_path = os.path.join(root, entry['pose'])
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    wb = np.asarray(d['keypoints'], dtype=np.float32)   # (T, 1, 133, 2)
    sc = np.asarray(d['scores'], dtype=np.float32)      # (T, 1, 133)
    if wb.ndim == 4:
        wb = wb[:, 0]
        sc = sc[:, 0]
    return wb, sc


def _hand_bbox(points, scores, min_hand_points, margin):
    """
    points: (21, 2) absolute pixel coords for one hand, one frame.
    scores: (21,) detection confidence for the same.
    Returns (x0, y0, x1, y1) padded+squared box, or None if too few
    points were detected to trust this frame's crop.
    """
    detected = (scores > 0) & ~np.isnan(points).any(axis=-1)
    if detected.sum() < min_hand_points:
        return None
    pts = points[detected]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(w, h) * (1.0 + 2 * margin)
    side = max(side, 1.0)  # guard against a degenerate zero-size box
    return cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2


def _crop_and_resize(frame, box, crop_size):
    """frame: (H, W, 3) uint8 RGB. box: (x0,y0,x1,y1) float, may be out of
    bounds. Returns (crop_size, crop_size, 3) uint8, black-padded where the
    box extends past the frame."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    bw, bh = x1 - x0, y1 - y0
    canvas = np.zeros((bh, bw, 3), dtype=np.uint8)
    src_x0, src_y0 = max(x0, 0), max(y0, 0)
    src_x1, src_y1 = min(x1, W), min(y1, H)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
        canvas[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = \
            frame[src_y0:src_y1, src_x0:src_x1]
    return cv2.resize(canvas, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)


def _normalized_ref_point(wrist_xy, box):
    """wrist_xy: (2,) absolute pixel. box: (x0,y0,x1,y1). -> [-1,1] normalized,
    grid_sample's native convention (judgment call #3, models/pgf_fusion.py)."""
    x0, y0, x1, y1 = box
    rx = 2 * (wrist_xy[0] - x0) / max(x1 - x0, 1e-6) - 1
    ry = 2 * (wrist_xy[1] - y0) / max(y1 - y0, 1e-6) - 1
    return float(np.clip(rx, -1, 1)), float(np.clip(ry, -1, 1))


def process_clip(entry, root, crops_dir, meta_dir, downsample_every,
                 crop_size, margin, min_hand_points, jpeg_quality):
    clip_id = entry['clip_id']
    left_npz = os.path.join(crops_dir, f'{clip_id}_left.npz')
    right_npz = os.path.join(crops_dir, f'{clip_id}_right.npz')
    meta_npz = os.path.join(meta_dir, f'{clip_id}.npz')
    if os.path.exists(left_npz) and os.path.exists(right_npz) and os.path.exists(meta_npz):
        return {'clip_id': clip_id, 'text': entry['text'],
                'left_path': os.path.abspath(left_npz),
                'right_path': os.path.abspath(right_npz),
                'meta_path': os.path.abspath(meta_npz),
                'T': int(np.load(meta_npz)['left_valid'].shape[0])}, None

    mp4 = os.path.join(root, entry['video'])
    if not os.path.exists(mp4):
        return None, f"{clip_id}: missing video at {mp4}"

    try:
        wb, sc = _load_pose_pkl(root, entry)
        if downsample_every > 1:
            wb = wb[::downsample_every]
            sc = sc[::downsample_every]
        frames = read_frames(mp4, downsample_every)
        if frames is None:
            return None, f"{clip_id}: no frames decoded"

        T = min(len(frames), wb.shape[0])
        if T == 0:
            return None, f"{clip_id}: zero frames after alignment"

        out = {}
        for side, sl in _HANDS.items():
            crops = np.zeros((T, crop_size, crop_size, 3), dtype=np.uint8)
            ref = np.zeros((T, 2), dtype=np.float32)
            valid = np.zeros((T,), dtype=bool)
            score = np.zeros((T,), dtype=np.float32)
            for t in range(T):
                pts = wb[t, sl]        # (21, 2) absolute pixel
                confs = sc[t, sl]      # (21,)
                score[t] = float(np.nan_to_num(confs, nan=0.0).mean())
                box = _hand_bbox(pts, confs, min_hand_points, margin)
                if box is None:
                    continue  # crop/ref/valid stay at their zero/False default
                crops[t] = _crop_and_resize(frames[t], box, crop_size)
                wrist = pts[_WRIST_LOCAL_IDX]
                if confs[_WRIST_LOCAL_IDX] > 0 and not np.isnan(wrist).any():
                    ref[t] = _normalized_ref_point(wrist, box)
                else:
                    # Wrist itself undetected -- fall back to the centroid
                    # of whatever points WERE detected, still in the box's
                    # normalized frame.
                    detected = (confs > 0) & ~np.isnan(pts).any(axis=-1)
                    centroid = pts[detected].mean(axis=0)
                    ref[t] = _normalized_ref_point(centroid, box)
                valid[t] = True
            out[side] = {'crops': crops, 'ref': ref, 'valid': valid, 'score': score}

        for side in ('left', 'right'):
            npz_path = os.path.join(crops_dir, f'{clip_id}_{side}.npz')
            jpg_bytes = np.empty(T, dtype=object)
            for t in range(T):
                ok, buf = cv2.imencode('.jpg', out[side]['crops'][t],
                                       [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                jpg_bytes[t] = buf.tobytes() if ok else b''
            np.savez(npz_path, jpg=jpg_bytes)

        np.savez(meta_npz,
                left_ref=out['left']['ref'], left_valid=out['left']['valid'],
                left_score=out['left']['score'],
                right_ref=out['right']['ref'], right_valid=out['right']['valid'],
                right_score=out['right']['score'])

        return {'clip_id': clip_id, 'text': entry['text'],
                'left_path': os.path.abspath(left_npz),
                'right_path': os.path.abspath(right_npz),
                'meta_path': os.path.abspath(meta_npz), 'T': T}, None
    except Exception:
        return None, f"{clip_id}: {traceback.format_exc(limit=1)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True,
                    help='Root containing {source}/videos/... AND '
                         '{source}/pose/... -- on Box B this is '
                         '/raid/shared/dataset, NOT $ASAN_ROOT.')
    ap.add_argument('--out', required=True, help='output root (writable)')
    ap.add_argument('--sources', nargs='+',
                    default=['informburo', 'khabar', 'qazaqstantv'])
    ap.add_argument('--lang', default='kz')
    ap.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'])
    ap.add_argument('--downsample-every', type=int, default=2,
                    help='Match paths.asan.downsample_every in config.yaml')
    ap.add_argument('--crop-size', type=int, default=112, help='Paper spec')
    ap.add_argument('--margin', type=float, default=0.3,
                    help='Bounding-box padding as a fraction of box side length')
    ap.add_argument('--min-hand-points', type=int, default=3,
                    help='Minimum detected keypoints to trust a crop')
    ap.add_argument('--jpeg-quality', type=int, default=90)
    ap.add_argument('--skip-low-quality', action='store_true', default=True)
    ap.add_argument('--max-consecutive-failures', type=int, default=20,
                    help='Abort if this many clips in a row fail -- almost '
                         'certainly a broken environment, not bad clips.')
    args = ap.parse_args()

    out = os.path.expanduser(args.out)
    for split in args.splits:
        entries = []
        for source in args.sources:
            ann = os.path.join(args.root, source, 'annotations', args.lang,
                               _SPLIT_FILES[split])
            if not os.path.exists(ann):
                print(f"[warn] missing {ann}")
                continue
            with open(ann) as f:
                for e in json.load(f):
                    if args.skip_low_quality and e.get('low_quality', False):
                        continue
                    if e.get('text', '').strip():
                        entries.append(e)

        crops_dir = os.path.join(out, 'hand_crops', split)
        meta_dir = os.path.join(out, 'hand_meta', split)
        os.makedirs(crops_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
        print(f"[{split}] {len(entries)} clips -> {out}")

        records, errors = [], []
        consecutive_failures = 0
        for i, entry in enumerate(entries):
            rec, err = process_clip(entry, args.root, crops_dir, meta_dir,
                                    args.downsample_every, args.crop_size,
                                    args.margin, args.min_hand_points,
                                    args.jpeg_quality)
            if rec:
                records.append(rec)
                consecutive_failures = 0
            elif err:
                errors.append(err)
                consecutive_failures += 1
                if consecutive_failures >= args.max_consecutive_failures:
                    print(f"\n[extract_asan_hand_crops] ABORTING: "
                          f"{consecutive_failures} consecutive failures -- "
                          f"this almost certainly means something is broken, "
                          f"not that individual clips are bad. Last error: {err}")
                    manifest = os.path.join(out, f'manifest_{split}.jsonl')
                    with open(manifest, 'w') as f:
                        for r in records:
                            f.write(json.dumps(r, ensure_ascii=False) + '\n')
                    sys.exit(1)
            if (i + 1) % 100 == 0:
                print(f"  [{split}] {i + 1}/{len(entries)} "
                      f"(ok={len(records)}, failed={len(errors)})")

        manifest = os.path.join(out, f'manifest_{split}.jsonl')
        with open(manifest, 'w') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"[{split}] done: {len(records)} ok, {len(errors)} failed "
              f"-> {manifest}")
        if errors:
            err_log = os.path.join(out, f'errors_{split}.log')
            with open(err_log, 'w') as f:
                f.write('\n'.join(errors))
            print(f"[{split}] errors logged to {err_log}")


if __name__ == '__main__':
    main()
