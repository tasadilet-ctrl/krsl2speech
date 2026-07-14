"""
Extract per-clip RGB visual features from asan-dataset video, for the
RGB-as-second-modality experiment queued up after diagnose_phase1.py's
embedding-collapse investigation (seven architecture/training-side
hypotheses ruled out for the pose-only encoder; pose-only may be
discarding exactly the fine finger configuration / facial detail that
disambiguates otherwise-similar signs).

Precomputes with a FROZEN pretrained visual backbone (DINOv2-small via
transformers -- already a dependency for mT5, so this only adds
torchvision, needed by AutoImageProcessor) rather than a trainable
end-to-end branch. Keeps this fully decoupled from Phase 1's already-tight
GPU memory budget (batch 16 OOMs there with full mT5 fine-tuning) and lets
us test whether RGB helps at all before committing to a much more
expensive trainable branch.

For every annotated clip this script:
  1. reads video frames via cv2.VideoCapture from entry['video']
  2. downsamples frames at the SAME rate as the pose pipeline
     (--downsample-every, matching paths.asan.downsample_every in
     config.yaml) so RGB features stay frame-aligned with pose --
     video and pose share the same underlying per-clip frame count
     (confirmed directly: a sample clip's video and pose .pkl both
     have T=765 before downsampling)
  3. runs frames through the frozen backbone in batches -> per-frame
     pooled feature vector (384-dim for dinov2-small)
  4. saves (T, feature_dim) float32 to {out}/rgb/{split}/{clip_id}.npy
  5. appends {clip_id, text, rgb_path, T} to {out}/manifest_{split}.jsonl

Usage:
  PYTHONPATH=. python scripts/extract_asan_rgb.py \
      --root /raid/shared/dataset --out ~/krsl2speech/data/asan_rgb \
      --splits train dev test --downsample-every 2 --device cuda

Note on --root: unlike pose (symlinked into ASAN_ROOT's local workaround
on Box B), video files were NOT included in that symlink setup -- point
--root directly at wherever videos/{source}/videos/kz/... actually lives
(e.g. /raid/shared/dataset on Box B), not $ASAN_ROOT.
"""
import os
import sys
import json
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import torch

_SPLIT_FILES = {'train': 'train.json', 'dev': 'dev.json', 'test': 'test.json'}


def read_frames(mp4_path, downsample_every):
    """Decode all frames, keep every `downsample_every`-th one, BGR->RGB."""
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return None
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % downsample_every == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames if frames else None


@torch.no_grad()
def extract_features(frames, processor, model, device, batch_size=32):
    """frames: list of (H, W, 3) uint8 RGB arrays -> (T, feature_dim) float32."""
    feats = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        inputs = processor(images=batch, return_tensors='pt').to(device)
        out = model(**inputs)
        feats.append(out.pooler_output.float().cpu().numpy())
    return np.concatenate(feats, axis=0)


def process_clip(entry, root, rgb_dir, downsample_every, processor, model, device):
    clip_id = entry['clip_id']
    npy = os.path.join(rgb_dir, f'{clip_id}.npy')
    if os.path.exists(npy):
        return {'clip_id': clip_id, 'text': entry['text'],
                'rgb_path': os.path.abspath(npy),
                'T': int(np.load(npy, mmap_mode='r').shape[0])}, None
    mp4 = os.path.join(root, entry['video'])
    if not os.path.exists(mp4):
        return None, f"{clip_id}: missing video at {mp4}"
    try:
        frames = read_frames(mp4, downsample_every)
        if frames is None:
            return None, f"{clip_id}: no frames decoded"
        feats = extract_features(frames, processor, model, device)
        np.save(npy, feats)
        return {'clip_id': clip_id, 'text': entry['text'],
                'rgb_path': os.path.abspath(npy), 'T': int(feats.shape[0])}, None
    except Exception:
        return None, f"{clip_id}: {traceback.format_exc(limit=1)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True,
                    help='Root containing {source}/videos/... -- on Box B '
                         'this is /raid/shared/dataset, NOT $ASAN_ROOT '
                         '(videos were never symlinked into asan_local)')
    ap.add_argument('--out', required=True, help='output root (writable)')
    ap.add_argument('--sources', nargs='+',
                    default=['informburo', 'khabar', 'qazaqstantv'])
    ap.add_argument('--lang', default='kz')
    ap.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'])
    ap.add_argument('--downsample-every', type=int, default=2,
                    help='Match paths.asan.downsample_every in config.yaml '
                         'so RGB stays frame-aligned with pose (50fps -> 25fps)')
    ap.add_argument('--backbone', default='facebook/dinov2-small',
                    help='Any transformers vision model with a pooler_output')
    ap.add_argument('--batch-size', type=int, default=32,
                    help='Frames per backbone forward call')
    ap.add_argument('--device', default=None,
                    help='cuda/cpu; auto-detects if not set')
    ap.add_argument('--skip-low-quality', action='store_true', default=True)
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[extract_asan_rgb] backbone={args.backbone} device={device}")

    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(args.backbone)
    model = AutoModel.from_pretrained(args.backbone).to(device).eval()
    print(f"[extract_asan_rgb] hidden_size={model.config.hidden_size}")

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
                        e['_source'] = source
                        entries.append(e)

        rgb_dir = os.path.join(out, 'rgb', split)
        os.makedirs(rgb_dir, exist_ok=True)
        print(f"[{split}] {len(entries)} clips -> {out}")

        records, errors = [], []
        for i, entry in enumerate(entries):
            rec, err = process_clip(entry, args.root, rgb_dir,
                                    args.downsample_every, processor, model, device)
            if rec:
                records.append(rec)
            elif err:
                errors.append(err)
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
