"""
Extract prosody features (F0, energy, duration) from audio files.
Saves as .npy files alongside the audio.

Usage:
    python data/extract_prosody.py \
        --audio-dir /raid/shared/khabar/audio/kz \
        --output-dir /raid/shared/alikhan_datasets/khabar_kz/prosody \
        --manifest /raid/shared/alikhan_datasets/khabar_kz/khabar_kz.jsonl

Or for Informburo (no manifest):
    python data/extract_prosody.py \
        --audio-dir /raid/shared/informburo/audio/kz \
        --output-dir /raid/shared/informburo/prosody/kz
"""
import os
import glob
import json
import argparse
import numpy as np
import torch
import librosa
try:
    import praat_parselmouth as parselmouth
except ImportError:
    import parselmouth


def extract_f0(audio, sr, frame_rate=100):
    """
    Extract F0 (fundamental frequency) using parselmouth.

    Args:
        audio: (T,) audio samples
        sr: sample rate
        frame_rate: target frame rate (Hz)

    Returns:
        f0: (T_frames,) F0 in Hz, 0 = unvoiced
    """
    # Use parselmouth for robust F0 estimation
    duration = len(audio) / sr
    sound = parselmouth.Sound(audio, sampling_frequency=sr)

    # Pitch extraction
    pitch = sound.to_pitch(
        time_step=1.0 / frame_rate,
        pitch_floor=50.0,
        pitch_ceiling=500.0,  # Kazakh speech range
    )

    f0 = pitch.selected_array['frequency']

    # Replace NaN with 0 (unvoiced)
    f0 = np.nan_to_num(f0, nan=0.0)

    # Smooth: replace isolated voiced frames with 0 if surrounded by unvoiced
    f0_smooth = f0.copy()
    for i in range(1, len(f0_smooth) - 1):
        if f0_smooth[i] > 0 and f0_smooth[i-1] == 0 and f0_smooth[i+1] == 0:
            f0_smooth[i] = 0

    return f0_smooth


def extract_energy(audio, sr, frame_rate=100):
    """
    Extract energy (RMS amplitude) per frame.

    Args:
        audio: (T,) audio samples
        sr: sample rate
        frame_rate: target frame rate (Hz)

    Returns:
        energy: (T_frames,) RMS energy
    """
    hop_length = int(sr / frame_rate)
    frame_length = hop_length  # non-overlapping

    # RMS energy per frame
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    return rms


def extract_prosody(audio_path, sr_target=16000, frame_rate=100):
    """
    Extract full prosody vector (F0, energy) from audio file.

    Args:
        audio_path: path to audio file (.opus, .wav, .mp3)
        sr_target: target sample rate
        frame_rate: target frame rate

    Returns:
        prosody: (T_frames, 2) — [F0, energy]
    """
    try:
        audio, sr = librosa.load(audio_path, sr=sr_target)
    except Exception as e:
        print(f"  [warn] Failed to load {audio_path}: {e}")
        return None

    if len(audio) < sr_target * 0.1:  # skip very short files
        return None

    f0 = extract_f0(audio, sr_target, frame_rate)
    energy = extract_energy(audio, sr_target, frame_rate)

    # Align lengths
    min_len = min(len(f0), len(energy))
    f0 = f0[:min_len]
    energy = energy[:min_len]

    # Normalize F0
    voiced_mask = f0 > 0
    if voiced_mask.sum() > 0:
        f0_mean = f0[voiced_mask].mean()
        f0_std = f0[voiced_mask].std()
        if f0_std > 0:
            f0[voiced_mask] = (f0[voiced_mask] - f0_mean) / f0_std

    # Normalize energy
    if energy.max() > 0:
        energy = energy / energy.max()

    prosody = np.stack([f0, energy], axis=-1)  # (T, 2)
    return prosody


def main():
    parser = argparse.ArgumentParser(description="Extract prosody from audio files")
    parser.add_argument('--audio-dir', required=True, help="Path to audio directory")
    parser.add_argument('--output-dir', required=True, help="Path to save prosody files")
    parser.add_argument('--manifest', default=None, help="Path to JSONL manifest (optional)")
    parser.add_argument('--frame-rate', type=int, default=100, help="Target frame rate")
    parser.add_argument('--sr', type=int, default=16000, help="Sample rate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find audio files
    audio_exts = ['*.opus', '*.wav', '*.mp3', '*.flac']
    audio_files = []
    for ext in audio_exts:
        audio_files.extend(glob.glob(os.path.join(args.audio_dir, '**', ext), recursive=True))
        audio_files.extend(glob.glob(os.path.join(args.audio_dir, ext)))
    audio_files = sorted(set(audio_files))

    print(f"Found {len(audio_files)} audio files")

    success = 0
    failed = 0

    for i, audio_path in enumerate(audio_files):
        if i % 100 == 0 and i > 0:
            print(f"  Processed {i}/{len(audio_files)}...")

        # Determine output path
        rel = os.path.relpath(audio_path, args.audio_dir)
        rel_name = os.path.splitext(rel)[0]
        # Replace backslashes for Windows paths
        rel_name = rel_name.replace('\\', '/')
        output_path = os.path.join(args.output_dir, rel_name + '.npy')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.exists(output_path):
            success += 1
            continue

        prosody = extract_prosody(audio_path, args.sr, args.frame_rate)
        if prosody is not None:
            np.save(output_path, prosody)
            success += 1
        else:
            failed += 1

    print(f"\nDone! Success: {success}, Failed: {failed}")
    print(f"Saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
