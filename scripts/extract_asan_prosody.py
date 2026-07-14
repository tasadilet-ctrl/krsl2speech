"""
Extract per-clip audio + prosody from asan-dataset for Phases 2-3.

The per-clip mp4s under videos/kz/processed/ carry an AAC audio track,
so no clip-offset math is needed: audio comes straight from each clip.

For every annotated clip this script:
  1. extracts mono audio via ffmpeg → {out}/wavs/{split}/{clip_id}.wav
  2. computes prosody [F0, energy] at 100 Hz (16 kHz, hop 160), with the
     same normalization as data/utils.extract_prosody_from_audio
     (standardized voiced F0, max-normalized energy)
     → {out}/prosody/{split}/{clip_id}.npy   (T, 2) float32
  3. appends {clip_id, text, audio_path, prosody_path, duration} to
     {out}/manifest_{split}.jsonl  — ready for train/train_tts.py

Usage:
  python scripts/extract_asan_prosody.py \
      --root /data/shared/asan-dataset \
      --out ~/krsl2speech/data/asan_audio \
      --splits train dev test --workers 8

Disk note: ~360 h of 22.05 kHz mono 16-bit wav ≈ 57 GB. Use --sr 16000
(~41 GB) if space is tight — mel extraction will resample as needed.
"""
import os
import sys
import json
import argparse
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_SPLIT_FILES = {'train': 'train.json', 'dev': 'dev.json', 'test': 'test.json'}
PROSODY_SR = 16000
PROSODY_HOP = 160  # 10 ms → 100 Hz


def _resolve_ffmpeg():
    """Prefer a system ffmpeg on PATH; fall back to the static binary
    bundled by the imageio-ffmpeg wheel (no system install / sudo needed,
    which is the situation on Box B). Returns the executable path or 'ffmpeg'."""
    from shutil import which
    exe = which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'  # let subprocess raise a clear error if truly absent


_FFMPEG = _resolve_ffmpeg()


def extract_wav(mp4_path, wav_path, sr):
    """ffmpeg: mp4 → mono wav. Returns (ok, error_detail)."""
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
        return True, None
    cmd = [_FFMPEG, '-y', '-loglevel', 'error', '-i', mp4_path,
           '-vn', '-ac', '1', '-ar', str(sr), wav_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
        if proc.returncode != 0:
            detail = (proc.stderr or '').strip().splitlines()
            return False, detail[-1] if detail else f'exit {proc.returncode}'
        if not (os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000):
            return False, 'empty output (video has no audio stream?)'
        return True, None
    except Exception as e:
        return False, str(e)


def compute_prosody(wav_path):
    """[F0, energy] at 100 Hz. Returns (T, 2) float32 or None."""
    import librosa
    audio, _ = librosa.load(wav_path, sr=PROSODY_SR)
    if len(audio) < PROSODY_SR * 0.2:
        return None

    f0, _, _ = librosa.pyin(
        audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=PROSODY_SR, frame_length=1024, hop_length=PROSODY_HOP)
    f0 = np.nan_to_num(f0, nan=0.0)
    energy = librosa.feature.rms(
        y=audio, frame_length=1024, hop_length=PROSODY_HOP).squeeze()

    n = min(len(f0), len(energy))
    f0, energy = f0[:n], energy[:n]

    voiced = f0 > 0
    if voiced.sum() > 1:
        mu, std = f0[voiced].mean(), f0[voiced].std()
        if std > 0:
            f0[voiced] = (f0[voiced] - mu) / std
    if energy.max() > 0:
        energy = energy / energy.max()

    return np.stack([f0, energy], axis=-1).astype(np.float32)


def process_clip(job):
    """One clip end-to-end. Returns (manifest_record | None, error | None)."""
    entry, root, wav_dir, pros_dir, sr, delete_wavs = job
    clip_id = entry['clip_id']
    try:
        npy = os.path.join(pros_dir, f'{clip_id}.npy')
        wav = os.path.join(wav_dir, f'{clip_id}.wav')

        if not os.path.exists(npy):
            mp4 = os.path.join(root, entry['video'])
            if not os.path.exists(mp4):
                return None, f"{clip_id}: missing video"
            ok, detail = extract_wav(mp4, wav, sr)
            if not ok:
                return None, f"{clip_id}: ffmpeg failed ({detail})"
            prosody = compute_prosody(wav)
            if prosody is None:
                return None, f"{clip_id}: audio too short"
            np.save(npy, prosody)

        # Prosody-only mode (Phase 2 doesn't need the audio itself):
        # ~57 GB of wavs shrink to ~200 MB of .npy files.
        if delete_wavs and os.path.exists(wav):
            os.remove(wav)

        return {
            'clip_id': clip_id,
            'video_id': clip_id.split('__')[1] if '__' in clip_id else clip_id,
            'text': entry['text'],
            'audio_path': (os.path.abspath(wav)
                           if os.path.exists(wav) else None),
            'prosody_path': os.path.abspath(npy),
        }, None
    except Exception:
        return None, f"{clip_id}: {traceback.format_exc(limit=1)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True, help='asan-dataset root')
    ap.add_argument('--out', required=True, help='output root (writable)')
    ap.add_argument('--sources', nargs='+',
                    default=['informburo', 'khabar', 'qazaqstantv'])
    ap.add_argument('--lang', default='kz')
    ap.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'])
    ap.add_argument('--sr', type=int, default=22050,
                    help='wav sample rate (22050 matches TTS MEL_CONFIG)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--skip-low-quality', action='store_true', default=True)
    ap.add_argument('--delete-wavs', action='store_true',
                    help='remove each wav after prosody is computed '
                         '(Phase 2 only needs the .npy; saves ~57 GB)')
    args = ap.parse_args()

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
                        entries.append(e)

        wav_dir = os.path.join(out, 'wavs', split)
        pros_dir = os.path.join(out, 'prosody', split)
        os.makedirs(wav_dir, exist_ok=True)
        os.makedirs(pros_dir, exist_ok=True)

        jobs = [(e, args.root, wav_dir, pros_dir, args.sr, args.delete_wavs)
                for e in entries]
        print(f"[{split}] {len(jobs)} clips → {out}")

        records, errors = [], []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_clip, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures)):
                rec, err = fut.result()
                if rec:
                    records.append(rec)
                elif err:
                    errors.append(err)
                if (i + 1) % 500 == 0:
                    print(f"  [{split}] {i + 1}/{len(jobs)} "
                          f"(ok={len(records)}, failed={len(errors)})")

        manifest = os.path.join(out, f'manifest_{split}.jsonl')
        with open(manifest, 'w') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"[{split}] done: {len(records)} ok, {len(errors)} failed "
              f"→ {manifest}")
        if errors:
            err_log = os.path.join(out, f'errors_{split}.log')
            with open(err_log, 'w') as f:
                f.write('\n'.join(errors))
            print(f"[{split}] errors logged to {err_log}")


if __name__ == '__main__':
    main()
