"""
Convert the KazakhTTS2 corpus (github.com/IS2AI/Kazakh_TTS) into a
{text, audio_path} JSONL manifest for train/train_tts.py.

The corpus is organized per speaker; each utterance is a .wav with a
matching .txt transcription. This script pairs them recursively, so it
tolerates layout variations between corpus versions.

Our FastSpeech2 has no speaker embedding — train on ONE speaker
(--speaker filters directories whose path contains the given substring,
e.g. F1 / M1), otherwise the model averages five voices.

Usage:
  python scripts/kazakhtts2_to_manifest.py \
      --corpus /path/to/KazakhTTS2 --speaker F1 \
      --out data/kazakhtts2_F1.jsonl

Then:
  PYTHONPATH=. python train/train_tts.py --config configs/config.yaml \
      --manifest data/kazakhtts2_F1.jsonl --save-dir output/phase3_ktts2
"""
import os
import json
import argparse
import glob


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus', required=True, help='KazakhTTS2 root')
    ap.add_argument('--speaker', default=None,
                    help='keep only paths containing this substring (e.g. F1)')
    ap.add_argument('--out', required=True, help='output manifest .jsonl')
    ap.add_argument('--min-chars', type=int, default=5)
    args = ap.parse_args()

    wavs = glob.glob(os.path.join(args.corpus, '**', '*.wav'), recursive=True)
    if args.speaker:
        wavs = [w for w in wavs if args.speaker in w]

    n_ok, n_missing, n_short = 0, 0, 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        for wav in sorted(wavs):
            stem = os.path.splitext(wav)[0]
            txt = stem + '.txt'
            if not os.path.exists(txt):
                # common alternative: transcripts live in a sibling dir
                alt = txt.replace('/wavs/', '/transcripts/')
                txt = alt if os.path.exists(alt) else None
            if txt is None:
                n_missing += 1
                continue
            with open(txt, encoding='utf-8') as t:
                text = t.read().strip()
            if len(text) < args.min_chars:
                n_short += 1
                continue
            f.write(json.dumps({
                'clip_id': os.path.basename(stem),
                'video_id': os.path.basename(stem),  # utterance-level split
                'text': text,
                'audio_path': os.path.abspath(wav),
            }, ensure_ascii=False) + '\n')
            n_ok += 1

    print(f"{n_ok} utterances → {args.out} "
          f"(missing transcript: {n_missing}, too short: {n_short})")
    if n_ok == 0:
        print("Nothing matched — check --corpus path and --speaker filter, "
              "and whether transcripts sit next to wavs or in a parallel dir.")


if __name__ == '__main__':
    main()
