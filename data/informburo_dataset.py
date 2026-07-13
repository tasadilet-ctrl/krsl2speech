"""
DataLoader for Informburo KZ dataset.
Loads paired keypoints (.npz) + Kazakh text (.txt).
Segments long videos into shorter clips for training.
"""
import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from data.utils import (
    load_npz_keypoints,
    assemble_keypoints,
    to_offset_keypoints,
    enrich_keypoints,
    preprocess_keypoints,
    KEYPOINT_DIM,
    ENRICHED_DIM,
)
from data.collators import PoseTextCollator


class InformburoDataset(Dataset):
    """
    Dataset for Informburo KZ (raw full videos).

    Each video:
      - keypoints/*.npz — full video keypoints (T frames)
      - transcripts/*.txt — Kazakh text
      - keypoints/*.json — metadata (fps, frames_total)

    Segments long videos into shorter clips for training.
    """

    def __init__(
        self,
        keypoints_root,
        transcripts_root,
        tokenizer=None,
        max_duration=60.0,
        min_duration=2.0,
        max_frames=1000,
        downsample_every=2,  # 50fps → 25fps
        name=None,
        split=None,       # 'train' | 'val' | None — signer-disjoint split by video_id
        split_ratio=0.9,  # fraction for train
        split_seed=42,    # random seed for reproducible split
        use_enriched=False,  # if True, return enriched keypoints (T, 1128)
    ):
        """
        Args:
            keypoints_root: path to keypoints/kz/ directory
            transcripts_root: path to transcripts/kz/ directory
            tokenizer: SentencePiece tokenizer (optional)
            max_duration: max clip duration in seconds
            min_duration: min clip duration in seconds
            max_frames: max frames per clip (truncate if longer)
            downsample_every: take every Nth frame
            name: dataset name for logging
            split: 'train', 'val', or None — signer-disjoint split by video_id
            split_ratio: fraction of video_ids for train set
            split_seed: random seed for reproducible split
            use_enriched: if True, return enriched keypoints (offset+vel+acc+valid)
        """
        self.keypoints_root = keypoints_root
        self.transcripts_root = transcripts_root
        self.tokenizer = tokenizer
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.max_frames = max_frames
        self.downsample_every = downsample_every
        self.ds_name = name or "informburo"
        self.use_enriched = use_enriched

        # Build clip list from all .npz files
        self.clips = self._build_clips()

        # Apply signer-disjoint split
        if split is not None:
            self.clips = self._apply_signer_split(self.clips, split, split_ratio, split_seed)

        ds_label = self.ds_name if self.ds_name else "dataset"
        split_tag = f" split={split}" if split else ""
        enrich_tag = f" enriched" if use_enriched else ""
        print(f"[{ds_label}]{split_tag}{enrich_tag} Loaded {len(self.clips)} clips (downsample={downsample_every}x)")

    def _build_clips(self):
        """Scan all .npz files and segment into clips."""
        # Check if keypoints_root has category subdirs (like Khabar raw)
        subdirs = [d for d in os.listdir(self.keypoints_root)
                   if os.path.isdir(os.path.join(self.keypoints_root, d))]
        if subdirs and not os.path.exists(os.path.join(self.keypoints_root, 'transcripts')):
            # Category-based structure: scan each subdir
            npz_files = []
            for sd in subdirs:
                npz_files.extend(glob.glob(os.path.join(self.keypoints_root, sd, '*.npz')))
        else:
            # Flat structure: scan directly
            npz_files = sorted(glob.glob(os.path.join(self.keypoints_root, '*.npz')))

        clips = []

        for npz_path in npz_files:
            video_id = os.path.basename(npz_path).replace('.npz', '')

            # Load metadata from .json
            json_path = npz_path.replace('.npz', '.json')
            fps = 50.0  # default
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                    fps = meta.get('fps', 50.0)

            # Load text from .txt
            # Try flat structure first, then category-based
            txt_path = os.path.join(self.transcripts_root, f'{video_id}.txt')
            text = ""
            if not os.path.exists(txt_path):
                # Try category-based: extract relative path from npz
                rel = os.path.relpath(npz_path, self.keypoints_root)
                txt_path = os.path.join(self.transcripts_root, rel.replace('.npz', '.txt'))
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    text = f.read().strip()

            if not text:
                continue  # skip if no text

            # Load keypoints to get frame count
            try:
                data = np.load(npz_path)
                wb_xy = data.get('wb_xy', None)
            except Exception:
                continue

            if wb_xy is None:
                continue

            total_frames = len(wb_xy)
            total_duration = total_frames / fps

            # Segment into clips of max_duration seconds
            clip_frames = int(self.max_duration * fps)
            clip_start = 0

            while clip_start < total_frames:
                clip_end = min(clip_start + clip_frames, total_frames)
                clip_duration = (clip_end - clip_start) / fps

                if clip_duration < self.min_duration:
                    break

                clips.append({
                    'video_id': video_id,
                    'npz_path': npz_path,
                    'frame_start': clip_start,
                    'frame_end': clip_end,
                    'fps': fps,
                    'duration': clip_duration,
                    'text': text,
                })

                clip_start = clip_end

        return clips

    def _apply_signer_split(self, clips, split, split_ratio, seed):
        """
        Split clips by video_id for signer-disjoint train/val.
        """
        import random
        rng = random.Random(seed)  # local RNG — don't reseed the global one

        # Group clips by video_id
        video_groups = {}
        for clip in clips:
            vid = clip['video_id']
            if vid not in video_groups:
                video_groups[vid] = []
            video_groups[vid].append(clip)

        # Shuffle and split video_ids
        video_ids = list(video_groups.keys())
        rng.shuffle(video_ids)

        split_idx = int(len(video_ids) * split_ratio)
        train_ids = set(video_ids[:split_idx])
        val_ids = set(video_ids[split_idx:])

        if split == 'train':
            keep_ids = train_ids
        elif split == 'val':
            keep_ids = val_ids
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split}")
        filtered = [c for c in clips if c['video_id'] in keep_ids]
        split_name = split
        n_videos = len(keep_ids)
        print(f"  [Signer split] {split_name}: {n_videos} videos → {len(filtered)} clips (out of {len(clips)} total)")
        return filtered

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]

        # Load keypoints for this clip
        try:
            data = np.load(clip['npz_path'])

            # Extract clip range from the raw arrays
            ws = slice(clip['frame_start'], clip['frame_end'])
            wb_xy = data['wb_xy'][ws]
            hand_l_xy = data['hand_l_xy'][ws]
            hand_r_xy = data['hand_r_xy'][ws]

            # Keep raw copies for validity mask (enriched features)
            if self.use_enriched:
                wb_raw = wb_xy.copy()
                hl_raw = hand_l_xy.copy()
                hr_raw = hand_r_xy.copy()

            # Handle NaN
            wb_xy = np.nan_to_num(wb_xy, nan=0.0)
            hand_l_xy = np.nan_to_num(hand_l_xy, nan=0.0)
            hand_r_xy = np.nan_to_num(hand_r_xy, nan=0.0)

            # Assemble into (T, 282)
            kps = assemble_keypoints(wb_xy, hand_l_xy, hand_r_xy)

            if kps is None or len(kps) == 0:
                return self._blank_sample()

            # Downsample if needed
            if self.downsample_every > 1 and len(kps) > 0:
                kps = kps[::self.downsample_every]
                if self.use_enriched:
                    wb_raw = wb_raw[::self.downsample_every]
                    hl_raw = hl_raw[::self.downsample_every]
                    hr_raw = hr_raw[::self.downsample_every]

            # Truncate if too long
            if len(kps) > self.max_frames:
                kps = kps[:self.max_frames]
                if self.use_enriched:
                    wb_raw = wb_raw[:self.max_frames]
                    hl_raw = hl_raw[:self.max_frames]
                    hr_raw = hr_raw[:self.max_frames]

            # Signer-scale normalization + detector-spike removal
            kps = preprocess_keypoints(kps)

            # Keep absolute coordinates for the dual-coord enriched channel
            kps_abs = kps.copy() if self.use_enriched else None

            # Convert to offset features (translation-invariant, from GloFE)
            kps = to_offset_keypoints(kps)

            # Enrich: offset + absolute + velocity + acceleration + validity
            if self.use_enriched:
                kps = enrich_keypoints(kps, wb_raw, hl_raw, hr_raw, kps_abs=kps_abs)
        except Exception:
            return self._blank_sample()

        # Text
        text = clip['text']
        text_ids = None
        if self.tokenizer:
            text_ids = self.tokenizer.encode(text)

        return {
            'keypoints': torch.tensor(kps, dtype=torch.float32),
            'prosody': None,
            'text': text,
            'text_ids': torch.tensor(text_ids, dtype=torch.long) if text_ids is not None else None,
            'input_length': len(kps),
            'clip_id': f"{clip['video_id']}_{clip['frame_start']}_{clip['frame_end']}",
        }

    def _blank_sample(self):
        """Return blank sample for failed loads."""
        dim = ENRICHED_DIM() if self.use_enriched else KEYPOINT_DIM
        return {
            'keypoints': torch.zeros(1, dim),
            'prosody': None,
            'text': '',
            'text_ids': torch.empty(0, dtype=torch.long),
            'input_length': 0,
            'clip_id': '',
        }


class InformburoCollator(PoseTextCollator):
    """Collate function for InformburoDataset — see data/collators.py."""
    pass
