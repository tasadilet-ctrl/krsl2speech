"""
Dataset for Phase 3: TTS training (FastSpeech2).
Loads audio → mel spectrograms, text → MT5 tokens, prosody from .npy.
"""
import os
import glob
import json
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset


# Default mel extraction config matching typical FastSpeech2
MEL_CONFIG = {
    'n_fft': 1024,
    'hop_length': 256,
    'win_length': 1024,
    'n_mel': 80,
    'fmin': 0,
    'fmax': 8000,
    'sr': 22050,
}


def wav_to_mel(audio_path, config=None):
    """
    Convert audio file to log-mel spectrogram.

    Returns:
        mel: (T_frames, n_mel) or None on failure
    """
    cfg = config or MEL_CONFIG
    try:
        audio, _ = librosa.load(audio_path, sr=cfg['sr'])
    except Exception:
        return None

    # librosa mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=cfg['sr'],
        n_fft=cfg['n_fft'],
        hop_length=cfg['hop_length'],
        win_length=cfg['win_length'],
        n_mels=cfg['n_mel'],
        fmin=cfg['fmin'],
        fmax=cfg['fmax'],
    )

    # Log-scale with clipping
    mel = np.log10(np.maximum(mel, 1e-10))

    # librosa returns (n_mel, T); the rest of the pipeline expects (T, n_mel).
    # Without this transpose, "frame" truncation/padding sliced mel BINS and
    # every sample appeared to be exactly n_mel frames long.
    return mel.T.astype(np.float32)


def extract_prosody_simple(audio_path, config=None):
    """
    Extract prosody (F0, energy) from audio using librosa.
    Simpler than extract_prosody.py — doesn't need parselmouth.

    Returns:
        prosody: (T_frames, 2) or None
    """
    cfg = config or MEL_CONFIG
    try:
        audio, sr = librosa.load(audio_path, sr=cfg['sr'])
    except Exception:
        return None

    if len(audio) < sr * 0.1:
        return None

    # F0 via pyin
    f0, _, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        frame_length=cfg['n_fft'],
        hop_length=cfg['hop_length'],
    )
    f0 = np.nan_to_num(f0, nan=0.0)

    # Energy via RMS
    energy = librosa.feature.rms(
        y=audio,
        frame_length=cfg['n_fft'],
        hop_length=cfg['hop_length'],
    ).squeeze()

    # Align
    min_len = min(len(f0), len(energy))
    f0 = f0[:min_len]
    energy = energy[:min_len]

    # Normalize
    voiced = f0 > 0
    if voiced.sum() > 0:
        mu, std = f0[voiced].mean(), f0[voiced].std()
        if std > 0:
            f0[voiced] = (f0[voiced] - mu) / std
    if energy.max() > 0:
        energy = energy / energy.max()

    return np.stack([f0, energy], axis=-1).astype(np.float32)


class TTSDataset(Dataset):
    """
    TTS training dataset.

    Each sample:
      - text_ids: (L,) — MT5 token IDs
      - mel: (T, n_mel) — log-mel spectrogram
      - prosody: (T, 2) — [F0, energy]
      - duration: (L,) — token-level alignment (optional)
    """

    def __init__(self, manifest_path, tokenizer, mel_config=None, max_mel_frames=2000,
                 split=None, split_ratio=0.95, split_seed=42):
        """
        Args:
            manifest_path: JSONL with {text, audio_path, mel_path (optional)}
            tokenizer: MT5 T5Tokenizer
            mel_config: mel extraction kwargs
            max_mel_frames: truncate samples longer than this
            split: 'train', 'val', or None — video-disjoint split
            split_ratio: fraction of video_ids for train set
            split_seed: random seed for reproducible split
        """
        self.manifest = self._load_manifest(manifest_path, max_mel_frames)
        self.tokenizer = tokenizer
        self.mel_config = mel_config or MEL_CONFIG
        self.max_mel_frames = max_mel_frames

        if split is not None:
            self.manifest = self._apply_video_split(
                self.manifest, split, split_ratio, split_seed
            )

        split_tag = f" split={split}" if split else ""
        print(f"[TTSDataset]{split_tag} Loaded {len(self.manifest)} samples")

    @staticmethod
    def _get_video_id(entry):
        """Source-video id for disjoint splitting."""
        if entry.get('video_id'):
            return entry['video_id']
        clip_id = entry.get('clip_id', '')
        if '__' in clip_id:  # e.g. "khabar__160745__seg00000" → "160745"
            parts = clip_id.split('__')
            if len(parts) >= 2:
                return parts[1]
        # Fallback: one audio file == one source video
        audio_path = entry.get('audio_path', '')
        if audio_path:
            return os.path.splitext(os.path.basename(audio_path))[0]
        return clip_id or ''

    def _apply_video_split(self, manifest, split, split_ratio, seed):
        """
        Video-disjoint train/val split. Clip-level random splitting (the old
        trainer behaviour) lets adjacent clips of the same broadcast span
        train and val, inflating metrics.
        """
        import random
        rng = random.Random(seed)  # local RNG — don't reseed the global one

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
        print(f"  [Video split] {split}: {len(keep_ids)} videos → "
              f"{len(filtered)} clips (out of {len(manifest)} total)")
        return filtered

    def _load_manifest(self, path, max_mel_frames):
        entries = []
        with open(path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                audio_path = entry.get('audio_path', '')
                if not audio_path or not os.path.exists(audio_path):
                    continue
                entries.append(entry)
        return entries

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        audio_path = entry['audio_path']

        # Load mel spectrogram (from cache or compute)
        mel_path = entry.get('mel_path', '')
        if mel_path and os.path.exists(mel_path):
            try:
                mel = np.load(mel_path)
            except Exception:
                mel = wav_to_mel(audio_path, self.mel_config)
        else:
            mel = wav_to_mel(audio_path, self.mel_config)

        if mel is None or mel.shape[0] == 0:
            return self._blank_sample()

        # Normalize orientation of cached mels to (T, n_mel).
        n_mel = self.mel_config['n_mel']
        if mel.ndim == 2 and mel.shape[0] == n_mel and mel.shape[1] != n_mel:
            mel = mel.T

        # Truncate
        if mel.shape[0] > self.max_mel_frames:
            mel = mel[:self.max_mel_frames]

        # Load prosody (or compute from audio)
        prosody = None
        prosody_path = entry.get('prosody_path', '')
        if prosody_path and os.path.exists(prosody_path):
            try:
                prosody = np.load(prosody_path)
            except Exception:
                prosody = extract_prosody_simple(audio_path, self.mel_config)
        else:
            prosody = extract_prosody_simple(audio_path, self.mel_config)

        # Align prosody with mel frames. Cached prosody is at 100 Hz while
        # mel is at sr/hop ≈ 86 Hz — resample rather than truncate (the
        # old min-length cut misaligned the last ~14% of every clip).
        if prosody is not None and len(prosody) > 0:
            from data.utils import resample_prosody
            prosody = resample_prosody(prosody, mel.shape[0])
        else:
            # Fallback: zeros
            prosody = np.zeros((mel.shape[0], 2), dtype=np.float32)

        # Tokenize text
        text = entry.get('text', '')
        if not text:
            return self._blank_sample()

        tokens = self.tokenizer.encode(text, max_length=256, truncation=True)
        if not tokens:
            return self._blank_sample()

        return {
            'text_ids': torch.tensor(tokens, dtype=torch.long),
            'mel': torch.tensor(mel, dtype=torch.float32),     # (T, n_mel)
            'prosody': torch.tensor(prosody, dtype=torch.float32),  # (T, 2)
            'mel_length': mel.shape[0],
            'text_length': len(tokens),
            'text': text,
        }

    def _blank_sample(self):
        return {
            'text_ids': torch.empty(0, dtype=torch.long),
            'mel': torch.zeros(1, self.mel_config['n_mel']),
            'prosody': torch.zeros(1, 2),
            'mel_length': 0,
            'text_length': 0,
            'text': '',
        }


class TTSCollator:
    """Collate for TTS: pad text, mel, prosody."""

    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # Filter blanks
        batch = [b for b in batch if b['text_length'] > 0 and b['mel_length'] > 0]
        if not batch:
            return None

        # Sort by mel length descending
        batch.sort(key=lambda x: x['mel_length'], reverse=True)

        # Pad text
        text_lens = [b['text_length'] for b in batch]
        max_text = max(text_lens)
        text_ids = torch.zeros(len(batch), max_text, dtype=torch.long)
        for i, b in enumerate(batch):
            text_ids[i, :b['text_length']] = b['text_ids']

        # Pad mel
        mel_lens = [b['mel_length'] for b in batch]
        max_mel = max(mel_lens)
        n_mel = batch[0]['mel'].shape[1]
        mels = torch.zeros(len(batch), max_mel, n_mel, dtype=torch.float32)
        for i, b in enumerate(batch):
            mels[i, :b['mel_length'], :] = b['mel']

        # Pad prosody
        prosody = torch.zeros(len(batch), max_mel, 2, dtype=torch.float32)
        for i, b in enumerate(batch):
            p_len = min(b['prosody'].shape[0], b['mel_length'])
            prosody[i, :p_len, :] = b['prosody'][:p_len]

        return {
            'text_ids': text_ids,                    # (B, L)
            'text_lengths': torch.tensor(text_lens), # (B,)
            'mel': mels,                             # (B, T, n_mel)
            'mel_lengths': torch.tensor(mel_lens),   # (B,)
            'prosody': prosody,                      # (B, T, 2)
            'texts': [b['text'] for b in batch],
        }
