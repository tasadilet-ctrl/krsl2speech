#!/usr/bin/env python3
"""
Draw the keypoints onto the clip videos so their quality can be judged by eye.

Draws EXACTLY the 69 points the model receives (data/utils.py,
unisign_part_features), upscaled 3x (the source is ~204x186; hands ~31 px):

  body  9  nose, ears, shoulders, elbows, wrists  (_US_BODY -- no eyes, hips, legs)
  hands 21 + 21                                    (wb_xy[91:133])
  face  18 every 2nd jaw point, 8 mouth, nose tip  (_US_FACE)

An earlier version drew all 17 COCO body points. The knees and ankles are
below the interpreter crop, so the pose model guesses them anew each frame:
they jumped 49-73 px per frame and landed near the resting hands, looking like
stray hand points. The model never sees them, so they are no longer drawn.

The dedicated sign-language hand model (hand_l_xy / hand_r_xy, unused) was
compared side by side on 2026-10-14 and judged clearly worse on fingers; it is
no longer drawn either.

Keypoints and video frames correspond one-to-one, in the video's own pixels
(both come from the cropped signer.mp4).

    python scripts/render_keypoint_overlays.py ~/qazaq_viewer   # renders every clip there
"""
import os
import pickle
import subprocess
import sys

import cv2
import numpy as np

SCALE = 3
US_BODY = [0, 3, 4, 5, 6, 7, 8, 9, 10]          # nose, ears, shoulders, elbows, wrists
BODY_EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]
US_FACE = list(range(23, 40))[::2] + list(range(83, 91)) + [53]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
              (0, 17), (17, 18), (18, 19), (19, 20)]
WHITE, GREY = (235, 235, 235), (150, 150, 150)
TRAIN_L, TRAIN_R = (255, 200, 0), (0, 200, 255)         # cyan-ish / amber (BGR)


def ok(p):
    return p is not None and not np.isnan(p).any()


def pt(p):
    return int(round(p[0] * SCALE)), int(round(p[1] * SCALE))


def draw_edges(img, pts, edges, color, thick):
    for a, b in edges:
        if ok(pts[a]) and ok(pts[b]):
            cv2.line(img, pt(pts[a]), pt(pts[b]), color, thick, cv2.LINE_AA)
    for p in pts:
        if ok(p):
            cv2.circle(img, pt(p), thick + 1, color, -1, cv2.LINE_AA)


def draw_frame(frame, wb):
    img = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    body = np.full((17, 2), np.nan)
    body[US_BODY] = wb[US_BODY]
    draw_edges(img, body, BODY_EDGES, WHITE, 2)
    for j in US_FACE:
        if ok(wb[j]):
            cv2.circle(img, pt(wb[j]), 2, GREY, -1, cv2.LINE_AA)
    draw_edges(img, wb[91:112], HAND_EDGES, TRAIN_L, 1)
    draw_edges(img, wb[112:133], HAND_EDGES, TRAIN_R, 1)
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, 'exactly the 69 keypoints the model receives', (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TRAIN_L, 1, cv2.LINE_AA)
    return img


def keypoint_path(video):
    """qazaqstantv: videos/ -> keypoints/, .npz.  khabar/informburo: videos/ -> pose/, .pkl."""
    for sub, ext in (('/keypoints/', '.npz'), ('/pose/', '.pkl')):
        p = video.replace('/videos/', sub).replace('.mp4', ext)
        if os.path.exists(p):
            return p
    return None


def load_wb(path):
    if path.endswith('.npz'):
        return np.load(path)['wb_xy']
    d = pickle.load(open(path, 'rb'))
    xy = np.asarray(d['keypoints'], np.float32)
    return xy[:, 0] if xy.ndim == 4 else xy


def render(video, npz, out):
    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        ok_, f = cap.read()
        if not ok_:
            break
        frames.append(f)
    wb = load_wb(npz)
    n = min(len(frames), len(wb))
    if n == 0:
        return False
    h, w = frames[0].shape[:2]
    H, W = h * SCALE + (h * SCALE) % 2, w * SCALE + (w * SCALE) % 2
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                             '-s', f'{W}x{H}', '-r', '30', '-i', '-', '-c:v', 'libx264',
                             '-pix_fmt', 'yuv420p', '-crf', '23', '-movflags', '+faststart', out],
                            stdin=subprocess.PIPE)
    for i in range(n):
        a = draw_frame(frames[i], wb[i])
        panel = np.zeros((H, W, 3), np.uint8)
        panel[:a.shape[0], :a.shape[1]] = a
        proc.stdin.write(panel.tobytes())
    proc.stdin.close()
    return proc.wait() == 0


def main(viewer_dir):
    viewer_dir = os.path.expanduser(viewer_dir)
    clips = os.path.join(viewer_dir, 'clips')
    outdir = os.path.join(viewer_dir, 'overlays')
    os.makedirs(outdir, exist_ok=True)
    names = sorted(f for f in os.listdir(clips) if f.endswith('.mp4'))
    done = 0
    for k, name in enumerate(names, 1):
        video = os.path.realpath(os.path.join(clips, name))
        npz = keypoint_path(video)
        out = os.path.join(outdir, name)
        if os.path.exists(out):
            done += 1
            continue
        if npz and render(video, npz, out):
            done += 1
        if k % 10 == 0:
            print(f'  {k}/{len(names)}', flush=True)
    print(f'[write] {done}/{len(names)} overlays in {outdir}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '~/qazaq_viewer')
