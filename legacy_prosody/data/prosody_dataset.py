"""
Dataset for Phase 2: Prosody GAN training.
Loads paired keypoints + prosody features.
"""
import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from data.utils import (
    load_npz_keypoints,
    assemble_keypoints,
    to_offset_keypoints,
    resample_prosody,
    KEYPOINT_DIM,
)


class ProsodyDataset(Dataset):
    """
    Dataset for prosody GAN training.

    Each sample:
      - keypoints: (T, 282) — sign keypoints
      - prosody: (T, 2) — [F0, energy]
      - text: str (optional, for alignment)
    """

    def __init__(
        self,
        manifest_path,
        keypoints_root,
        prosody_root,
        max_duration=60.0,
        min_duration=2.0,
        max_frames=1000,
        downsample_every=1,
        name=None,
        split=None,       # 'train' | 'val' | None — signer-disjoint split by video_id
        split_ratio=0.9,
        split_seed=42,
    ):
        """
        Args:
            manifest_path: path to khabar_kz.jsonl
            keypoints_root: path to keypoints/ directory
            prosody_root: path to extracted prosody .npy files
            max_duration: max clip duration in seconds
            min_duration: min clip duration in seconds
            max_frames: max frames per clip
            downsample_every: take every Nth frame
            name: dataset name for logging
            split: 'train', 'val', or None — signer-disjoint split by video_id
            split_ratio: fraction of video_ids for train set
            split_seed: random seed for reproducible split
        """
        self.manifest = self._load_manifest(manifest_path, max_duration, min_duration)
        self.keypoints_root = keypoints_root
        self.prosody_root = prosody_root
        self.max_frames = max_frames
        self.downsample_every = downsample_every
        self.ds_name = name or "prosody"

        # Categories for audio lookup
        self.categories = [
            'alemde', 'densaulyk', 'ekologiya', 'ekonomika',
            'kogam', 'madeniet', 'okiga', 'sayasat', 'sport'
        ]

        if split is not None:
            self.manifest = self._apply_signer_split(
                self.manifest, split, split_ratio, split_seed
            )

        split_tag = f" split={split}" if split else ""
        print(f"[{self.ds_name}]{split_tag} Loaded {len(self.manifest)} clips for prosody training")

    def _get_video_id(self, entry):
        """Extract video_id from a manifest entry."""
        if 'video_id' in entry:
            return entry['video_id']
        clip_id = entry.get('clip_id', '')
        if '__' in clip_id:
            parts = clip_id.split('__')
            if len(parts) >= 2:
                return parts[1]
        return clip_id

    def _apply_signer_split(self, manifest, split, split_ratio, seed):
        """Signer-disjoint split by video_id (same scheme as KhabarKzDataset)."""
        import random
        rng = random.Random(seed)

        video_ids = sorted({self._get_video_id(e) for e in manifest})
        rng.shuffle(video_ids)

        split_idx = int(len(video_ids) * split_ratio)
        if split == 'train':
            keep_ids = set(video_ids[:split_idx])
        elif split == 'val':
            keep_ids = set(video_ids[split_idx:])
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split}")

        filtered = [e for e in manifest if self._get_video_id(e) in keep_ids]
        print(f"  [Signer split] {split}: {len(keep_ids)} videos → "
              f"{len(filtered)} clips (out of {len(manifest)} total)")
        return filtered

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

    def _find_keypoint_file(self, entry):
        """Find the .npz keypoint file."""
        kp_path = entry.get('keypoints_path', '')
        if kp_path and os.path.exists(kp_path):
            return kp_path

        clip_id = entry['clip_id']
        video_id = clip_id.split('__')[1]
        kp_dir = os.path.join(self.keypoints_root, video_id)
        # Prefer the npz matching this exact clip; an arbitrary npz could be
        # a different segment of the video.
        exact = os.path.join(kp_dir, f'{clip_id}.npz')
        if os.path.exists(exact):
            return exact
        npz_files = sorted(glob.glob(os.path.join(kp_dir, '*.npz')))
        if len(npz_files) == 1:
            return npz_files[0]
        return None

    def _find_prosody_file(self, entry):
        """Find the prosody .npy file for this clip's parent video."""
        clip_id = entry['clip_id']
        video_id = clip_id.split('__')[1]

        # Try in prosody root
        for cat in self.categories:
            prosody_path = os.path.join(self.prosody_root, cat, f'{video_id}.npy')
            if os.path.exists(prosody_path):
                return prosody_path

        # Try flat structure
        prosody_path = os.path.join(self.prosody_root, f'{video_id}.npy')
        if os.path.exists(prosody_path):
            return prosody_path

        return None

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
        kps, scores, frame_idx = load_npz_keypoints(kp_path, frame_start, frame_end)

        if kps is None or len(kps) == 0:
            return self._blank_sample()

        # Downsample
        if self.downsample_every > 1 and len(kps) > 0:
            kps = kps[::self.downsample_every]

        # Truncate
        if len(kps) > self.max_frames:
            kps = kps[:self.max_frames]

        # Convert to offset features (translation-invariant, from GloFE)
        kps = to_offset_keypoints(kps)

        # Load prosody
        prosody = None
        prosody_path = self._find_prosody_file(entry)
        if prosody_path:
            try:
                prosody_full = np.load(prosody_path)  # (T_parent, 2)

                # Align with keypoint frames
                start_sec = entry.get('start', 0)
                end_sec = entry.get('end')

                # Assuming prosody is at 100 Hz
                frame_rate = 100
                start_frame = int(start_sec * frame_rate)
                end_frame = int(end_sec * frame_rate) if end_sec else start_frame + len(kps) * self.downsample_every

                # Resample prosody (100 Hz) to the keypoint frame count.
                # Truncating to min length (old behaviour) misaligned the
                # sequences: 100 Hz prosody has ~2-4x the frames of the
                # 25-50 fps keypoints, so only the first half (or quarter)
                # of the clip's prosody was ever used.
                prosody_clip = prosody_full[start_frame:end_frame]
                if len(prosody_clip) > 0:
                    prosody = resample_prosody(prosody_clip, len(kps))

            except Exception:
                pass

        # No usable prosody → skip this sample. Training the GAN against
        # all-zero targets (old behaviour) teaches it to produce silence.
        if prosody is None or len(prosody) == 0:
            return self._blank_sample()

        return {
            'keypoints': torch.tensor(kps, dtype=torch.float32),       # (T, 282)
            'prosody': torch.tensor(prosody, dtype=torch.float32),     # (T, 2)
            'input_length': len(kps),
            'clip_id': clip_id,
        }

    def _blank_sample(self):
        return {
            'keypoints': torch.zeros(1, KEYPOINT_DIM),
            'prosody': torch.zeros(1, 2),
            'input_length': 0,
            'clip_id': '',
        }


class ProsodyCollator:
    """Collate function for ProsodyDataset."""

    def __call__(self, batch):
        batch = [b for b in batch if b['input_length'] > 0]
        if not batch:
            return None

        batch.sort(key=lambda x: x['input_length'], reverse=True)
        keypoint_lengths = [b['input_length'] for b in batch]
        max_kp_len = max(keypoint_lengths)

        # Pad keypoints
        kps_padded = []
        for b in batch:
            kps = b['keypoints']
            if kps.shape[0] < max_kp_len:
                pad = torch.zeros(max_kp_len - kps.shape[0], kps.shape[1])
                kps = torch.cat([kps, pad], dim=0)
            kps_padded.append(kps)
        kps_tensor = torch.stack(kps_padded, dim=0)

        # Pad prosody
        pros_padded = []
        for b in batch:
            p = b['prosody']
            if p.shape[0] < max_kp_len:
                pad = torch.zeros(max_kp_len - p.shape[0], p.shape[1])
                p = torch.cat([p, pad], dim=0)
            pros_padded.append(p)
        prosody_tensor = torch.stack(pros_padded, dim=0)

        return {
            'keypoints': kps_tensor,                          # (B, T, 282)
            'prosody': prosody_tensor,                        # (B, T, 2)
            'input_lengths': torch.tensor(keypoint_lengths),  # (B,)
            'clip_ids': [b['clip_id'] for b in batch],
        }
