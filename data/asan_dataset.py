"""
DataLoader for asan-dataset (360+ hours, umbrella of informburo / khabar /
qazaqstantv sources).

Layout (per source, e.g. /data/shared/asan-dataset/informburo):
  annotations/kz/{train,dev,test}.json   — clip-level annotations:
      {video, pose, text, clip_id, T, hand_l_cov, hand_r_cov, low_quality}
  pose/kz/processed/{split}/{video_id}/{clip_id}.pkl
      {'keypoints': (T, 1, 133, 2), 'scores': (T, 1, 133)}  — COCO-WholeBody
  keypoints/kz/raw/{video_id}.npz        — per-video npz (no clip offsets,
                                            so we use the per-clip .pkl)
  audio/kz, transcripts/kz, videos/kz    — for later phases

Notes:
  - Splits are PREDEFINED (train/dev/test json) and video-disjoint by
    construction — we respect them rather than re-splitting.
  - Hands are part of the 133 wholebody points: left 91-111, right 112-132.
  - Frames where a joint's score == 0 (or coords are NaN) are treated as
    undetected for the validity channel of enriched features.
"""
import os
import json
import pickle
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from data.utils import (
    assemble_keypoints,
    assemble_joint_scores,
    to_offset_keypoints,
    enrich_keypoints,
    preprocess_keypoints,
    resample_prosody,
    resample_hand_crops,
    resample_hand_meta,
    KEYPOINT_DIM,
    ENRICHED_DIM,
)
from data.collators import PoseTextCollator

# COCO-WholeBody hand slices inside the 133-point array
WB_LEFT_HAND = slice(91, 112)    # 21 points
WB_RIGHT_HAND = slice(112, 133)  # 21 points

_SPLIT_FILES = {'train': 'train.json', 'val': 'dev.json',
                'dev': 'dev.json', 'test': 'test.json'}


class AsanDataset(Dataset):
    """
    Clip-level dataset over one or more asan-dataset sources.

    Each sample matches the other loaders (PoseTextCollator-compatible):
      keypoints: (T, 282) or (T, 1128), text: str, input_length, clip_id.
    """

    def __init__(
        self,
        root,                    # e.g. /data/shared/asan-dataset
        sources=('informburo', 'khabar', 'qazaqstantv'),
        split='train',           # 'train' | 'val'/'dev' | 'test' (predefined)
        lang='kz',
        tokenizer=None,
        min_frames=25,           # skip clips shorter than this (pre-downsample)
        max_frames=1000,         # truncate after downsampling
        downsample_every=1,
        use_enriched=False,
        skip_low_quality=True,   # drop clips flagged low_quality
        min_hand_cov=0.0,        # drop clips with hand coverage below this
        load_prosody=False,      # Phase 2: load per-clip prosody .npy
        prosody_root=None,       # output root of scripts/extract_asan_prosody.py
        load_rgb=False,          # optional second modality: per-clip RGB features
        rgb_root=None,           # output root of scripts/extract_asan_rgb.py
        load_hand_crops=False,   # Prior-Guided Fusion: per-frame hand-crop JPEGs
        hand_crop_root=None,     # output root of scripts/extract_asan_hand_crops.py
        name=None,
    ):
        if split not in _SPLIT_FILES:
            raise ValueError(f"split must be one of {list(_SPLIT_FILES)}, got {split}")

        self.root = root
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.downsample_every = downsample_every
        self.use_enriched = use_enriched
        self.ds_name = name or f"asan_{lang}"
        self.load_prosody = load_prosody
        # extract_asan_prosody.py writes prosody/{train|dev|test}/{clip_id}.npy
        self._prosody_dir = None
        if load_prosody:
            if not prosody_root:
                raise ValueError("load_prosody=True requires prosody_root "
                                 "(run scripts/extract_asan_prosody.py first)")
            split_dir = _SPLIT_FILES[split].replace('.json', '')
            self._prosody_dir = os.path.join(
                os.path.expanduser(prosody_root), 'prosody', split_dir)

        self.load_rgb = load_rgb
        # extract_asan_rgb.py writes rgb/{train|dev|test}/{clip_id}.npy
        self._rgb_dir = None
        if load_rgb:
            if not rgb_root:
                raise ValueError("load_rgb=True requires rgb_root "
                                 "(run scripts/extract_asan_rgb.py first)")
            split_dir = _SPLIT_FILES[split].replace('.json', '')
            self._rgb_dir = os.path.join(
                os.path.expanduser(rgb_root), 'rgb', split_dir)

        self.load_hand_crops = load_hand_crops
        # extract_asan_hand_crops.py writes hand_crops/{split}/{clip_id}_{left,right}.npz
        # and hand_meta/{split}/{clip_id}.npz
        self._hand_crops_dir = None
        self._hand_meta_dir = None
        if load_hand_crops:
            if not hand_crop_root:
                raise ValueError("load_hand_crops=True requires hand_crop_root "
                                 "(run scripts/extract_asan_hand_crops.py first)")
            split_dir = _SPLIT_FILES[split].replace('.json', '')
            hcr = os.path.expanduser(hand_crop_root)
            self._hand_crops_dir = os.path.join(hcr, 'hand_crops', split_dir)
            self._hand_meta_dir = os.path.join(hcr, 'hand_meta', split_dir)

        self.clips = []
        n_filtered = 0
        for source in sources:
            ann_path = os.path.join(root, source, 'annotations', lang,
                                    _SPLIT_FILES[split])
            if not os.path.exists(ann_path):
                print(f"[{self.ds_name}] WARNING: missing {ann_path}, skipping source")
                continue
            with open(ann_path, 'r') as f:
                entries = json.load(f)
            kept = 0
            for e in entries:
                if not e.get('text', '').strip():
                    n_filtered += 1
                    continue
                if e.get('T', 0) < min_frames:
                    n_filtered += 1
                    continue
                if skip_low_quality and e.get('low_quality', False):
                    n_filtered += 1
                    continue
                if (min(e.get('hand_l_cov', 1.0), e.get('hand_r_cov', 1.0))
                        < min_hand_cov):
                    n_filtered += 1
                    continue
                e['_source'] = source
                self.clips.append(e)
                kept += 1
            print(f"[{self.ds_name}] {source}/{_SPLIT_FILES[split]}: "
                  f"{kept}/{len(entries)} clips kept")

        enrich_tag = " enriched" if use_enriched else ""
        print(f"[{self.ds_name}] split={split}{enrich_tag} Total: "
              f"{len(self.clips)} clips ({n_filtered} filtered), "
              f"downsample={downsample_every}x")

    def __len__(self):
        return len(self.clips)

    def _load_pose(self, entry):
        """Load per-clip wholebody keypoints from the .pkl."""
        pkl_path = os.path.join(self.root, entry['pose'])
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        wb = np.asarray(d['keypoints'], dtype=np.float32)   # (T, 1, 133, 2)
        sc = np.asarray(d['scores'], dtype=np.float32)      # (T, 1, 133)
        if wb.ndim == 4:  # drop the person axis
            wb = wb[:, 0]
            sc = sc[:, 0]
        return wb, sc

    def __getitem__(self, idx):
        entry = self.clips[idx]
        try:
            wb, sc = self._load_pose(entry)
        except Exception as e:
            print(f"[warn] Failed to load {entry.get('pose')}: {e}")
            return self._blank_sample()

        if len(wb) == 0:
            return self._blank_sample()

        # Undetected joints: score == 0 or NaN coords → NaN (drives the
        # validity channel of enriched features, then imputed to 0).
        undetected = (sc <= 0) | np.isnan(sc) | np.isnan(wb).any(axis=-1)
        wb = wb.copy()
        wb[undetected] = np.nan

        # Split into wholebody + hands (hands live at 91-111 / 112-132)
        hand_l = wb[:, WB_LEFT_HAND]    # (T, 21, 2)
        hand_r = wb[:, WB_RIGHT_HAND]   # (T, 21, 2)

        # Downsample / truncate (scores kept in lockstep)
        if self.downsample_every > 1:
            wb = wb[::self.downsample_every]
            hand_l = hand_l[::self.downsample_every]
            hand_r = hand_r[::self.downsample_every]
            sc = sc[::self.downsample_every]
        if len(wb) > self.max_frames:
            wb = wb[:self.max_frames]
            hand_l = hand_l[:self.max_frames]
            hand_r = hand_r[:self.max_frames]
            sc = sc[:self.max_frames]

        # Keep raw (NaN-marked) copies for the validity mask
        wb_raw, hl_raw, hr_raw = (wb.copy(), hand_l.copy(), hand_r.copy()) \
            if self.use_enriched else (None, None, None)

        # Impute and assemble → (T, 282)
        wb0 = np.nan_to_num(wb, nan=0.0)
        kps = assemble_keypoints(
            wb0, np.nan_to_num(hand_l, nan=0.0), np.nan_to_num(hand_r, nan=0.0))

        # Signer-scale normalization + detector-spike removal
        kps = preprocess_keypoints(kps)

        # Keep absolute coordinates for the dual-coord enriched channel
        kps_abs = kps.copy() if self.use_enriched else None

        # Offset features (translation-invariant), then optional enrichment
        kps = to_offset_keypoints(kps)
        if self.use_enriched:
            # Continuous per-joint confidence → validity channel
            # (score-aware, Uni-Sign). Undetected joints already have
            # score <= 0 or NaN, which clips to 0.
            joint_scores = assemble_joint_scores(np.nan_to_num(sc, nan=0.0))
            kps = enrich_keypoints(kps, wb_raw, hl_raw, hr_raw,
                                   kps_abs=kps_abs, joint_scores=joint_scores)

        # Prosody (Phase 2): 100 Hz [F0, energy] resampled to keypoint length
        prosody = None
        if self.load_prosody:
            npy = os.path.join(self._prosody_dir,
                               f"{entry.get('clip_id', '')}.npy")
            if not os.path.exists(npy):
                return self._blank_sample()  # skip clips without prosody
            try:
                prosody = resample_prosody(np.load(npy), len(kps))
            except Exception:
                return self._blank_sample()

        # RGB (optional second modality): per-frame frozen-backbone features
        # from scripts/extract_asan_rgb.py, resampled to keypoint length.
        # Extraction already downsamples video at the same rate as pose, so
        # this is normally a no-op -- resample_prosody returns the array
        # unchanged when lengths already match (its early-return path), so
        # no interpolation happens in the common case. Resampling only
        # actually kicks in for the max_frames-truncation case above, or a
        # rare video/pose frame-count mismatch from independent decoding.
        rgb = None
        if self.load_rgb:
            rgb_npy = os.path.join(self._rgb_dir, f"{entry.get('clip_id', '')}.npy")
            if not os.path.exists(rgb_npy):
                return self._blank_sample()  # skip clips without RGB features
            try:
                rgb = resample_prosody(np.load(rgb_npy), len(kps))
            except Exception:
                return self._blank_sample()

        # Hand crops (Prior-Guided Fusion): per-frame JPEG crops + reference
        # points/confidence from scripts/extract_asan_hand_crops.py,
        # nearest-neighbor resampled to keypoint length. Stacked into a
        # single (T, 2, ...) tensor per field, hand axis order [left, right]
        # -- matches models/unisign_encoder.py's KeypointEncoder.MODES
        # left/right processing order, and keeps collation to a single
        # pad-along-T operation (same convention as 'rgb' above).
        hand_crops = hand_ref = hand_valid = hand_score = None
        if self.load_hand_crops:
            clip_id = entry.get('clip_id', '')
            try:
                per_hand_crops, per_hand_meta = {}, {}
                for side in ('left', 'right'):
                    crop_npz = os.path.join(self._hand_crops_dir, f"{clip_id}_{side}.npz")
                    if not os.path.exists(crop_npz):
                        return self._blank_sample()
                    jpg = np.load(crop_npz, allow_pickle=True)['jpg']
                    decoded = np.stack([
                        cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                        for b in jpg
                    ], axis=0)  # (T_src, 112, 112, 3) uint8, RGB -- extract_asan_hand_crops.py
                                # crops from scripts.extract_asan_rgb.read_frames' output,
                                # which is already BGR->RGB converted; cv2.imencode/imdecode
                                # round-trip preserves array values regardless of cv2's own
                                # BGR labeling convention, so this stays true RGB. Matters
                                # because models/pgf_fusion.py's HandBackbone is ImageNet-
                                # pretrained and expects standard RGB + ImageNet norm stats.
                    per_hand_crops[side] = resample_hand_crops(decoded, len(kps))

                meta_npz = os.path.join(self._hand_meta_dir, f"{clip_id}.npz")
                if not os.path.exists(meta_npz):
                    return self._blank_sample()
                meta = np.load(meta_npz)
                for side in ('left', 'right'):
                    ref_r, valid_r, score_r = resample_hand_meta(
                        meta[f'{side}_ref'], meta[f'{side}_valid'],
                        meta[f'{side}_score'], len(kps))
                    per_hand_meta[side] = (ref_r, valid_r, score_r)
            except Exception:
                return self._blank_sample()

            hand_crops = np.stack([per_hand_crops['left'], per_hand_crops['right']], axis=1)
            hand_ref = np.stack([per_hand_meta['left'][0], per_hand_meta['right'][0]], axis=1)
            hand_valid = np.stack([per_hand_meta['left'][1], per_hand_meta['right'][1]], axis=1)
            hand_score = np.stack([per_hand_meta['left'][2], per_hand_meta['right'][2]], axis=1)

        text = entry['text']
        text_ids = self.tokenizer.encode(text) if self.tokenizer else None

        return {
            'keypoints': torch.tensor(kps, dtype=torch.float32),
            'prosody': (torch.tensor(prosody, dtype=torch.float32)
                        if prosody is not None else None),
            'rgb': (torch.tensor(rgb, dtype=torch.float32)
                   if rgb is not None else None),
            'hand_crops': (torch.tensor(hand_crops, dtype=torch.uint8)
                          if hand_crops is not None else None),
            'hand_ref': (torch.tensor(hand_ref, dtype=torch.float32)
                        if hand_ref is not None else None),
            'hand_valid': (torch.tensor(hand_valid, dtype=torch.bool)
                          if hand_valid is not None else None),
            'hand_score': (torch.tensor(hand_score, dtype=torch.float32)
                          if hand_score is not None else None),
            'text': text,
            'text_ids': (torch.tensor(text_ids, dtype=torch.long)
                         if text_ids is not None else None),
            'input_length': len(kps),
            'clip_id': entry.get('clip_id', ''),
        }

    def _blank_sample(self):
        dim = ENRICHED_DIM() if self.use_enriched else KEYPOINT_DIM
        return {
            'keypoints': torch.zeros(1, dim),
            'prosody': None,
            'rgb': None,
            'hand_crops': None,
            'hand_ref': None,
            'hand_valid': None,
            'hand_score': None,
            'text': '',
            'text_ids': torch.empty(0, dtype=torch.long),
            'input_length': 0,
            'clip_id': '',
        }


class AsanCollator(PoseTextCollator):
    """Collate function for AsanDataset — see data/collators.py."""
    pass
