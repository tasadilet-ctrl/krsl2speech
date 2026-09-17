"""
Extract prosody features (F0, energy, duration) from audio files in bulk.
Preprocess for Phase 2 training.
"""
import os
import json
import glob
import argparse
import numpy as np
import torch
import librosa
from tqdm import tqdm

from data.utils import extract_prosody_from_audio


def extract_batch(audio_dir, output_dir, categories=None):
    """Extract prosody from all audio files in a directory."""
    if categories:
        audio_files = []
        for cat in categories:
            cat_dir = os.path.join(audio_dir, cat)
            if os.path.exists(cat_dir):
                audio_files.extend(glob.glob(os.path.join(cat_dir, '*.opus')))
                audio_files.extend(glob.glob(os.path.join(cat_dir, '*.wav')))
                audio_files.extend(glob.glob(os.path.join(cat_dir, '*.mp3')))
    else:
        audio_files = glob.glob(os.path.join(audio_dir, '**', '*.opus'), recursive=True)
        audio_files += glob.glob(os.path.join(audio_dir, '**', '*.wav'), recursive=True)

    audio_files = sorted(set(audio_files))
    print(f"Found {len(audio_files)} audio files")

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for audio_path in tqdm(audio_files, desc="Extracting prosody"):
        prosody = extract_prosody_from_audio(audio_path, sr_target=16000)

        if prosody is not None and len(prosody) > 10:
            # Save as .npz
            rel_path = os.path.relpath(audio_path, audio_dir)
            out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '_prosody.npz')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savez_compressed(out_path, prosody=prosody)
            results.append({
                'audio_path': audio_path,
                'prosody_path': out_path,
                'frames': len(prosody),
                'f0_mean': float(np.nanmean(prosody[:, 0])),
                'energy_mean': float(np.nanmean(prosody[:, 1])),
            })

    # Save manifest
    manifest_path = os.path.join(output_dir, 'prosody_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Extracted prosody for {len(results)}/{len(audio_files)} files")
    print(f"Manifest saved to {manifest_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--categories', nargs='+', default=None,
                        help='Category subdirectories (default: all)')
    args = parser.parse_args()

    extract_batch(args.audio_dir, args.output_dir, args.categories)


if __name__ == '__main__':
    main()
