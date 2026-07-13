"""
DataLoader for kazsign-dataset.
Loads paired keypoints + audio prosody + Kazakh text.
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


class KazSignDataset(Dataset):
    """
    Dataset for kazsign-dataset (10,037 entries).

    Each entry directory:
      kazsign_XXXXX/
        ├── signer.mp4       → raw signer video
        ├── audio.wav        → synchronized Kazakh speech
        ├── meta.json        → Whisper alignment, transcript
        ├── clips.json       → pre-segmented clips with text_asr

    After extract.py:
        └── signer.npz       → keypoints (T frames × COCO-WholeBody + hands)

    Each clip in clips.json has:
      - clip_id, video_id, start, end, frame_start, frame_end
      - text_asr: Kazakh text from Whisper
    """

    def __init__(
        self,
        data_root,
        keypoints_root=None,
        tokenizer=None,
        max_duration=60.0,
        min_duration=2.0,
        max_frames=1000,
        load_prosody=True,
        split=None,       # 'train' | 'val' | None — signer-disjoint split by video_id
        split_ratio=0.9,
        split_seed=42,
        use_enriched=False,
    ):
        """
        Args:
            data_root:      path to kazsign-dataset/
            keypoints_root: path to extracted keypoints (if different from data_root)
            tokenizer:      SentencePiece tokenizer (optional)
            max_duration:   max clip duration in seconds
            min_duration:   min clip duration in seconds
            max_frames:     max frames per clip
            load_prosody:   whether to extract prosody from audio
            split:          'train', 'val', or None — signer-disjoint split
            split_ratio:    fraction for train set
            split_seed:     random seed for reproducible split
            use_enriched:   if True, return enriched keypoints (T, 1128)
        """
        self.data_root = data_root
        self.keypoints_root = keypoints_root or data_root
        self.tokenizer = tokenizer
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.max_frames = max_frames
        self.load_prosody = load_prosody
        self.use_enriched = use_enriched

        # Load all clips from all entries
        self.clips = self._load_all_clips()

        # Apply signer-disjoint split
        if split is not None:
            self.clips = self._apply_signer_split(self.clips, split, split_ratio, split_seed)

        split_tag = f" split={split}" if split else ""
        enrich_tag = f" enriched" if use_enriched else ""
        print(f"[KazSignDataset]{split_tag}{enrich_tag} Loaded {len(self.clips)} clips from kazsign-dataset")

    def _load_all_clips(self):
        """Load clips from all kazsign_XXXXX/clips.json files."""
        clips = []
        entry_dirs = sorted(glob.glob(os.path.join(self.data_root, 'kazsign_*')))

        for entry_dir in entry_dirs:
            # Skip non-entry directories
            basename = os.path.basename(entry_dir)
            if not basename.startswith('kazsign_'):
                continue

            clips_json = os.path.join(entry_dir, 'clips.json')
            if not os.path.exists(clips_json):
                continue

            try:
                with open(clips_json, 'r') as f:
                    entry_clips = json.load(f)
            except Exception as e:
                print(f"[warn] Failed to load {clips_json}: {e}")
                continue

            for clip in entry_clips:
                duration = clip.get('duration', clip.get('end', 0) - clip.get('start', 0))
                if duration > self.max_duration or duration < self.min_duration:
                    continue
                clip['_entry_dir'] = entry_dir
                clips.append(clip)

        return clips

    def _apply_signer_split(self, clips, split, split_ratio, seed):
        """Split clips by video_id for signer-disjoint train/val."""
        import random
        rng = random.Random(seed)  # local RNG — don't reseed the global one

        video_groups = {}
        for clip in clips:
            vid = clip.get('video_id', clip.get('clip_id', ''))
            if vid not in video_groups:
                video_groups[vid] = []
            video_groups[vid].append(clip)

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
        filtered = [c for c in clips if c.get('video_id', c.get('clip_id', '')) in keep_ids]
        split_name = split
        n_videos = len(keep_ids)
        print(f"  [Signer split] {split_name}: {n_videos} videos → {len(filtered)} clips (out of {len(clips)} total)")
        return filtered

    def _find_keypoint_file(self, clip):
        """Find .npz keypoint file for this clip's entry."""
        entry_dir = clip['_entry_dir']

        # Check if keypoints are co-located with the entry
        npz_path = os.path.join(entry_dir, 'signer.npz')
        if os.path.exists(npz_path):
            return npz_path

        # Check separate keypoints root
        video_id = clip.get('video_id', '')
        if video_id:
            npz_path = os.path.join(self.keypoints_root, f'{video_id}.npz')
            if os.path.exists(npz_path):
                return npz_path

        return None

    def _find_audio_file(self, clip):
        """Find audio.wav for this clip's entry."""
        entry_dir = clip['_entry_dir']
        audio_path = os.path.join(entry_dir, 'audio.wav')
        if os.path.exists(audio_path):
            return audio_path
        return None

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        entry_dir = clip['_entry_dir']

        # Load keypoints
        npz_path = self._find_keypoint_file(clip)
        if npz_path is None:
            return self._blank_sample()

        frame_start = clip.get('frame_start')
        frame_end = clip.get('frame_end')

        if self.use_enriched:
            kps, scores, frame_idx, wb_raw, hl_raw, hr_raw = load_npz_keypoints(
                npz_path, frame_start, frame_end, raw_arrays=True
            )
        else:
            kps, scores, frame_idx = load_npz_keypoints(npz_path, frame_start, frame_end)
            wb_raw, hl_raw, hr_raw = None, None, None

        if kps is None or len(kps) == 0:
            return self._blank_sample()

        # Truncate if needed
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

        # Load prosody from audio.wav
        prosody = None
        if self.load_prosody:
            audio_path = self._find_audio_file(clip)
            if audio_path:
                start_sec = clip.get('start', 0)
                end_sec = clip.get('end')
                prosody = extract_prosody_from_audio(
                    audio_path,
                    sr_target=16000,
                    frame_start_sec=start_sec,
                    frame_end_sec=end_sec,
                )

        # Align prosody with keypoints. Prosody is at 100 Hz while keypoints
        # are at the video frame rate, so resample rather than truncate
        # (truncation dropped the second half of the prosody contour).
        if prosody is not None:
            prosody = resample_prosody(prosody, len(kps))

        # Text
        text = clip.get('text_asr', '')
        text_ids = None
        if self.tokenizer:
            text_ids = self.tokenizer.encode(text)

        return {
            'keypoints': torch.tensor(kps, dtype=torch.float32),      # (T, 282) or (T, 1128)
            'prosody': torch.tensor(prosody, dtype=torch.float32) if prosody is not None else None,
            'text': text,
            'text_ids': torch.tensor(text_ids, dtype=torch.long) if text_ids is not None else None,
            'input_length': len(kps),
            'clip_id': clip.get('clip_id', ''),
        }

    def _blank_sample(self):
        dim = ENRICHED_DIM() if self.use_enriched else KEYPOINT_DIM
        return {
            'keypoints': torch.zeros(1, dim),
            'prosody': None,
            'text': '',
            'text_ids': torch.empty(0, dtype=torch.long),
            'input_length': 0,
            'clip_id': '',
        }


class KazSignCollator(PoseTextCollator):
    """Collate function for KazSignDataset — see data/collators.py."""
    pass
