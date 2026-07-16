"""
Extract per-CLIP prosody (raw F0 + energy) from asan-dataset for Phases 2-3.

Supersedes extract_asan_prosody_v2.py (per-video), which was numerically correct
but ~5x slower than necessary: librosa.pyin's Viterbi thrashes on full-video
signals because clips cover ~72% of parent recordings, so running pyin on
unused audio wastes cycles and evicts cache.

The original extract_asan_prosody.py had two bugs:
  1. It read audio from per-clip mp4s (silent for khabar/qazaqstantv).
  2. It normalized per-file (erasing cross-clip dynamics).

This script fixes both:
  - Slices raw audio at [start, end] via ffmpeg (one clip at a time).
  - Stores RAW F0 (Hz) + RAW RMS energy — no normalization baked in.
  - Corpus stats computed in a streaming second pass -> prosody_stats.json.
  - Normalization happens at train time in data/prosody_dataset.py.

Output:
  {out}/prosody/{split}/{clip_id}.npy   (T, 2) float32 -- [F0_hz, energy_rms],
                                         F0=0 on unvoiced frames, 100 Hz.
  {out}/prosody_stats.json               corpus mean/std for F0 (voiced)
                                         and energy.
  {out}/manifest_{split}.jsonl           one record per clip.
  {out}/errors_{split}.log               per-clip failures.

Usage:
  python scripts/extract_asan_prosody_v3.py \
      --root /raid/shared/dataset \
      --out ~/krsl2speech/data/asan_prosody_v3 \
      --langs kz --splits train dev test --workers 44

Resumable: a clip whose .npy already exists is skipped.
"""
import os
import json
import argparse
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

PROSODY_SR = 16000
PROSODY_HOP = 160  # 10 ms -> 100 Hz


def _resolve_ffmpeg():
    """Prefer system ffmpeg on PATH; fall back to imageio-ffmpeg static binary."""
    from shutil import which
    exe = which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


_FFMPEG = _resolve_ffmpeg()


def slice_and_load_audio(src_audio, start, end):
    """ffmpeg: raw recording [start, end] -> mono 16 kHz float array.

    -ss/-to placed BEFORE -i so ffmpeg seeks on the input (fast, exact
    for the formats here); -vn drops any video the container carries.
    """
    import tempfile, librosa
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        wav = tmp.name
    try:
        cmd = [_FFMPEG, '-y', '-loglevel', 'error',
               '-ss', f'{start:.3f}', '-to', f'{end:.3f}',
               '-i', src_audio, '-vn', '-ac', '1', '-ar', str(PROSODY_SR),
               wav]
        proc = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
        if proc.returncode != 0:
            detail = (proc.stderr or '').strip().splitlines()
            raise RuntimeError(detail[-1] if detail else f'exit {proc.returncode}')
        if not (os.path.exists(wav) and os.path.getsize(wav) > 1000):
            raise RuntimeError('empty slice (bad start/end?)')
        audio, _ = librosa.load(wav, sr=PROSODY_SR)
        return audio
    finally:
        if os.path.exists(wav):
            os.remove(wav)


def compute_prosody_raw(audio):
    """RAW [F0_hz, energy_rms] at 100 Hz. Returns (T, 2) float32 or None.

    Deliberately NOT normalized -- see module docstring. F0 is 0.0 where
    pyin reports unvoiced, so a voiced mask is recoverable as (f0 > 0).
    """
    import librosa
    if len(audio) < PROSODY_SR * 0.2:
        return None

    f0, _, _ = librosa.pyin(
        audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=PROSODY_SR, frame_length=1024, hop_length=PROSODY_HOP)
    f0 = np.nan_to_num(f0, nan=0.0)
    energy = librosa.feature.rms(
        y=audio, frame_length=1024, hop_length=PROSODY_HOP).squeeze()

    n = min(len(f0), len(energy))
    return np.stack([f0[:n], energy[:n]], axis=-1).astype(np.float32)


def process_clip(job):
    """One clip end-to-end. Returns (manifest_record | None, error | None)."""
    entry, pros_dir, sr = job
    clip_id = entry['clip_id']
    try:
        npy = os.path.join(pros_dir, f'{clip_id}.npy')
        if not os.path.exists(npy):
            if not os.path.exists(entry['audio_src']):
                return None, f"{clip_id}: missing raw audio {entry['audio_src']}"
            audio = slice_and_load_audio(
                entry['audio_src'], entry['start'], entry['end'])
            prosody = compute_prosody_raw(audio)
            if prosody is None:
                return None, f"{clip_id}: audio too short"
            np.save(npy, prosody)

        arr = np.load(npy, mmap_mode='r')
        return {
            'clip_id': clip_id,
            'video_id': entry['video_id'],
            'lang': entry['lang'],
            'source': entry['source'],
            'split': entry['split'],
            'text': entry['text'],
            'duration': entry['duration'],
            'frames': int(arr.shape[0]),
            'prosody_path': os.path.abspath(npy),
        }, None
    except Exception:
        return None, f"{clip_id}: {traceback.format_exc(limit=1)}"


def load_jobs(root, sources, langs, splits):
    """Join the per-clip master manifest with the raw-recording manifest.

    {source}/{lang}/processed/{source}_{lang}.jsonl  -> clip start/end/split
    {source}/{lang}/raw/manifest.jsonl               -> video_id -> audio path
    """
    by_split = {s: [] for s in splits}
    for source in sources:
        for lang in langs:
            clips_f = os.path.join(root, source, lang, 'processed',
                                   f'{source}_{lang}.jsonl')
            raw_f = os.path.join(root, source, lang, 'raw', 'manifest.jsonl')
            if not os.path.exists(clips_f):
                print(f'[warn] no clip manifest: {clips_f}')
                continue
            if not os.path.exists(raw_f):
                print(f'[warn] no raw manifest: {raw_f}')
                continue

            audio_of = {}
            with open(raw_f) as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec.get('audio'):
                        audio_of[str(rec['id'])] = rec['audio']

            kept = missing = 0
            with open(clips_f) as fh:
                for line in fh:
                    c = json.loads(line)
                    if c.get('split') not in by_split:
                        continue
                    rel = audio_of.get(str(c['video_id']))
                    if not rel:
                        missing += 1
                        continue
                    by_split[c['split']].append({
                        'clip_id': c['clip_id'],
                        'video_id': str(c['video_id']),
                        'source': source,
                        'lang': lang,
                        'split': c['split'],
                        'text': c['text'],
                        'start': float(c['start']),
                        'end': float(c['end']),
                        'duration': round(float(c['end']) - float(c['start']), 3),
                        'audio_src': os.path.join(root, rel),
                    })
                    kept += 1
            print(f"[{source}/{lang}] {kept} clips mapped to raw audio"
                  + (f", {missing} with no audio entry" if missing else ""))
    return by_split


def corpus_stats(npy_paths):
    """Streaming corpus mean/std: F0 over voiced frames, energy over all.

    Phase 3 (FastSpeech2) normalizes with corpus-level statistics, not
    per-utterance ones -- these are those statistics.
    """
    f0_sum = f0_sq = f0_n = 0.0
    e_sum = e_sq = e_n = 0.0
    f0_min, f0_max = np.inf, -np.inf
    for p in npy_paths:
        a = np.load(p)
        f0, e = a[:, 0], a[:, 1]
        v = f0[f0 > 0]
        if v.size:
            f0_sum += float(v.sum())
            f0_sq += float((v.astype(np.float64) ** 2).sum())
            f0_n += v.size
            f0_min = min(f0_min, float(v.min()))
            f0_max = max(f0_max, float(v.max()))
        e_sum += float(e.sum())
        e_sq += float((e.astype(np.float64) ** 2).sum())
        e_n += e.size

    def mstd(s, sq, n):
        if n < 2:
            return 0.0, 1.0
        mu = s / n
        var = max(sq / n - mu * mu, 0.0)
        return mu, (var ** 0.5) or 1.0

    f0_mu, f0_std = mstd(f0_sum, f0_sq, f0_n)
    e_mu, e_std = mstd(e_sum, e_sq, e_n)
    return {
        'f0_mean': f0_mu, 'f0_std': f0_std,
        'f0_min': None if f0_min == np.inf else f0_min,
        'f0_max': None if f0_max == -np.inf else f0_max,
        'energy_mean': e_mu, 'energy_std': e_std,
        'voiced_frames': int(f0_n), 'total_frames': int(e_n),
        'voiced_ratio': (f0_n / e_n) if e_n else 0.0,
        'note': 'F0 stats over voiced frames only (f0 > 0). Apply as '
                '(f0 - f0_mean)/f0_std on voiced frames, 0 elsewhere; '
                '(energy - energy_mean)/energy_std.',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True, help='asan-dataset root')
    ap.add_argument('--out', required=True, help='output root (writable)')
    ap.add_argument('--sources', nargs='+',
                    default=['informburo', 'khabar', 'qazaqstantv'])
    ap.add_argument('--langs', nargs='+', default=['kz'],
                    help='kz, ru, or both (qazaqstantv is kz-only)')
    ap.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'])
    ap.add_argument('--workers', type=int, default=44)
    ap.add_argument('--limit', type=int, default=0,
                    help='process only the first N clips per split (smoke test)')
    args = ap.parse_args()

    out = os.path.expanduser(args.out)
    jobs_by_split = load_jobs(args.root, args.sources, args.langs, args.splits)

    all_done_paths = []

    for split in args.splits:
        entries = jobs_by_split.get(split, [])
        if args.limit:
            entries = entries[:args.limit]
        if not entries:
            print(f"[{split}] no clips, skipping")
            continue

        pros_dir = os.path.join(out, 'prosody', split)
        os.makedirs(pros_dir, exist_ok=True)

        man_path = os.path.join(out, f'manifest_{split}.jsonl')
        err_path = os.path.join(out, f'errors_{split}.log')
        ok = failed = 0

        jobs = [(e, pros_dir, PROSODY_SR) for e in entries]

        with open(man_path, 'w') as man, open(err_path, 'w') as errf, \
                ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_clip, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                rec, err = fut.result()
                if rec:
                    man.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    all_done_paths.append(rec['prosody_path'])
                    ok += 1
                else:
                    errf.write(err + '\n')
                    failed += 1
                if i % 500 == 0:
                    print(f"  [{split}] {i}/{len(jobs)} "
                          f"(ok={ok}, failed={failed})", flush=True)

        print(f"[{split}] done: {ok} ok, {failed} failed -> {man_path}")
        if failed:
            print(f"[{split}] errors logged to {err_path}")

    if all_done_paths:
        print(f'\n[stats] computing corpus statistics over {len(all_done_paths)} clips...',
              flush=True)
        stats = corpus_stats(all_done_paths)
        stats_path = os.path.join(out, 'prosody_stats.json')
        with open(stats_path, 'w') as fh:
            json.dump(stats, fh, indent=2)
        print(f'[stats] F0   mean={stats["f0_mean"]:.2f} Hz  '
              f'std={stats["f0_std"]:.2f}  '
              f'range=[{stats["f0_min"]:.1f}, {stats["f0_max"]:.1f}]')
        print(f'[stats] energy mean={stats["energy_mean"]:.5f}  '
              f'std={stats["energy_std"]:.5f}')
        print(f'[stats] voiced {100 * stats["voiced_ratio"]:.1f}% of frames')
        print(f'[stats] -> {stats_path}')


if __name__ == '__main__':
    main()