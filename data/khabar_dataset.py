"""
DataLoader for srp-manifest/khabar_kz dataset.
Loads paired keypoints (from .npz) + Kazakh text + audio prosody.
"""
import json
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from data.utils import (
    load_npz_keypoints,
    extract_prosody_from_audio,
    to_offset_keypoints,
    enrich_keypoints,
    preprocess_keypoints,
    resample_prosody,
    KEYPOINT_DIM,
    ENRICHED_DIM,
)
from data.collators import PoseTextCollator


class KhabarKzDataset(Dataset):
    """
    Dataset for srp-manifest/khabar_kz.

    Each sample:
      - keypoints: (T, 304) — assembled from .npz
      - text: str — Kazakh text (norm_text)
      - prosody: (T, 3) — F0, energy, duration from audio (if available)
      - clip_id: str
    """

    def __init__(
        self,
        manifest_path,
        keypoints_root,
        audio_root=None,
        tokenizer=None,
        max_duration=60.0,
        min_duration=2.0,
        max_frames=1000,
        load_prosody=False,
        downsample_every=1,  # >1: take every Nth frame (e.g., 2 for 50fps → 25fps)
        name=None,
        split=None,       # 'train' | 'val' | None — signer-disjoint split by video_id
        split_ratio=0.9,  # fraction for train
        split_seed=42,    # random seed for reproducible split
        use_enriched=False,  # if True, return enriched keypoints (T, 1128)
    ):
        """
        Args:
            manifest_path:  path to khabar_kz.jsonl
            keypoints_root: path to keypoints/ directory
            audio_root:     path to khabar/audio/kz/ (optional, for prosody)
            tokenizer:      SentencePiece tokenizer (optional, for Phase 1)
            max_duration:   max clip duration in seconds
            min_duration:   min clip duration in seconds
            max_frames:     max frames per clip (truncate if longer)
            load_prosody:   whether to load prosody from audio
            downsample_every: take every Nth frame (e.g., 2 for 50fps → 25fps)
            name:           dataset name for logging
            split:          'train', 'val', or None — signer-disjoint split by video_id
            split_ratio:    fraction of video_ids for train set
            split_seed:     random seed for reproducible split
            use_enriched:   if True, return enriched keypoints (offset+vel+acc+valid)
        """
        self.manifest = self._load_manifest(manifest_path, max_duration, min_duration)
        self.keypoints_root = keypoints_root
        self.audio_root = audio_root
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.load_prosody = load_prosody
        self.downsample_every = downsample_every
        self.ds_name = name or os.path.basename(manifest_path)
        self.use_enriched = use_enriched

        # Category mapping for audio lookup
        self.categories = [
            'alemde', 'densaulyk', 'ekologiya', 'ekonomika',
            'kogam', 'madeniet', 'okiga', 'sayasat', 'sport'
        ]

        # Apply signer-disjoint split
        if split is not None:
            self.manifest = self._apply_signer_split(
                self.manifest, split, split_ratio, split_seed
            )

        ds_label = self.ds_name if self.ds_name else 'dataset'
        split_tag = f" split={split}" if split else ""
        enrich_tag = f" enriched" if use_enriched else ""
        print(f"[{ds_label}]{split_tag}{enrich_tag} Loaded {len(self.manifest)} clips (downsample={self.downsample_every}x)")

    def _load_manifest(self, path, max_duration, min_duration):
        """Load and filter manifest entries."""
        clips = []
        with open(path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                duration = entry.get('duration', entry.get('end', 0) - entry.get('start', 0))
                if duration > max_duration or duration < min_duration:
                    continue
                clips.append(entry)
        return clips

    def _get_video_id(self, entry):
        """Extract video_id from a manifest entry."""
        if 'video_id' in entry:
            return entry['video_id']
        # Fallback: extract from clip_id like "khabar__160745__seg00000"
        clip_id = entry.get('clip_id', '')
        if '__' in clip_id:
            parts = clip_id.split('__')
            if len(parts) >= 2:
                return parts[1]
        # Last resort: use clip_id itself
        return clip_id

    def _apply_signer_split(self, manifest, split, split_ratio, seed):
        """
        Split manifest by video_id for signer-disjoint train/val.

        Ensures clips from the same video (same signer/session) don't leak
        between train and validation sets.
        """
        import random
        rng = random.Random(seed)  # local RNG — don't reseed the global one

        # Group entries by video_id
        video_groups = {}
        for entry in manifest:
            vid = self._get_video_id(entry)
            if vid not in video_groups:
                video_groups[vid] = []
            video_groups[vid].append(entry)

        # Shuffle video_ids and split
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

        filtered = [e for e in manifest if self._get_video_id(e) in keep_ids]
        split_name = split
        n_videos = len(keep_ids)
        print(f"  [Signer split] {split_name}: {n_videos} videos → {len(filtered)} clips (out of {len(manifest)} total)")
        return filtered

    def _find_keypoint_file(self, entry):
        """Find the .npz keypoint file for a clip."""
        # khabar_kz.jsonl format: keypoints_path field or clip_id
        kp_path = entry.get('keypoints_path', '')
        if kp_path and os.path.exists(kp_path):
            return kp_path

        # Fallback: construct from clip_id
        clip_id = entry['clip_id']  # e.g. "khabar__160745__seg00000"
        video_id = clip_id.split('__')[1]  # "160745"

        # Try keypoints directory structure — prefer the file that matches
        # this exact clip; picking an arbitrary npz (old behaviour) could pair
        # keypoints from a different segment with this clip's text.
        kp_dir = os.path.join(self.keypoints_root, video_id)
        exact = os.path.join(kp_dir, f'{clip_id}.npz')
        if os.path.exists(exact):
            return exact
        npz_files = sorted(glob.glob(os.path.join(kp_dir, '*.npz')))
        if len(npz_files) == 1:
            return npz_files[0]

        return None

    def _find_audio_file(self, entry):
        """Find the audio file for a clip's parent video."""
        if not self.audio_root:
            return None

        video_id = entry.get('video_id', '')
        if not video_id:
            # Extract from clip_id: "khabar__160745__seg00000" → "160745"
            clip_id = entry['clip_id']
            video_id = clip_id.split('__')[1] if '__' in clip_id else ''

        if not video_id:
            return None

        # Search in category subdirectories
        for cat in self.categories:
            audio_file = os.path.join(self.audio_root, cat, f'{video_id}.opus')
            if os.path.exists(audio_file):
                return audio_file

        return None

    def _get_text(self, entry):
        """Extract normalized Kazakh text."""
        return entry.get('norm_text', entry.get('text', ''))

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        clip_id = entry['clip_id']

        # Load keypoints
        kp_path = self._find_keypoint_file(entry)
        if kp_path is None:
            return self._blank_sample()

        frame_start = entry.get('frame_start')
        frame_end = entry.get('frame_end')

        if self.use_enriched:
            # Load raw arrays for validity mask computation
            kps, scores, frame_idx, wb_raw, hl_raw, hr_raw = load_npz_keypoints(
                kp_path, frame_start, frame_end, raw_arrays=True
            )
        else:
            kps, scores, frame_idx = load_npz_keypoints(kp_path, frame_start, frame_end)
            wb_raw, hl_raw, hr_raw = None, None, None

        if kps is None or len(kps) == 0:
            return self._blank_sample()

        # Downsample if needed (e.g., 50fps → 25fps)
        if self.downsample_every > 1 and len(kps) > 0:
            kps = kps[::self.downsample_every]
            if wb_raw is not None:
                wb_raw = wb_raw[::self.downsample_every]
                hl_raw = hl_raw[::self.downsample_every]
                hr_raw = hr_raw[::self.downsample_every]

        # Truncate if too long
        if len(kps) > self.max_frames:
            kps = kps[:self.max_frames]
            if wb_raw is not None:
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

        # Load prosody (optional, Phase 2+)
        prosody = None
        if self.load_prosody:
            audio_path = self._find_audio_file(entry)
            if audio_path:
                start_sec = entry.get('start', 0)
                end_sec = entry.get('end')
                prosody = extract_prosody_from_audio(
                    audio_path,
                    sr_target=16000,
                    frame_start_sec=start_sec,
                    frame_end_sec=end_sec,
                )

        # Align prosody length with keypoints. Prosody is at 100 Hz while
        # keypoints are at the video frame rate, so resample rather than
        # truncate (truncation dropped the second half of the prosody).
        if prosody is not None:
            prosody = resample_prosody(prosody, len(kps))

        # Text
        text = self._get_text(entry)
        text_ids = None
        if self.tokenizer:
            text_ids = self.tokenizer.encode(text)

        return {
            'keypoints': torch.tensor(kps, dtype=torch.float32),       # (T, 282) or (T, 1128)
            'prosody': torch.tensor(prosody, dtype=torch.float32) if prosody is not None else None,  # (T, 3)
            'text': text,
            'text_ids': torch.tensor(text_ids, dtype=torch.long) if text_ids is not None else None,
            'input_length': len(kps),
            'clip_id': clip_id,
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


class KhabarKzCollator(PoseTextCollator):
    """Collate function for KhabarKzDataset — see data/collators.py."""
    pass
