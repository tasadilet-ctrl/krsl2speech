#!/usr/bin/env python3
"""
Build the ceiling-study annotation package: N clips + a self-contained HTML tool
a KRSL interpreter opens locally (no server, no upload) to mark, word by word,
which parts of each speech transcript were actually SIGNED.

Why: the decoder is trained to produce the speech transcript, but simultaneous
interpreters compress and drop material. Pose-attributable recall falls 2.2x
from sentence start to end and the missing content is not in the next clip
(NOTES_positional_recall_and_lag.md). If a large share of every reference was
never signed, every metric in this project has the wrong denominator. Only a
human who reads KRSL can measure that.

Clips come from the FROZEN SELECTION SET, so once annotations return, the
predictions already on disk for every checkpoint can be re-scored against the
signed words only (scripts/analyze_annotations.py) -- no re-decoding needed.

Methodology choices:
  * No audio. The stored clips have no audio track (verified with ffprobe) and
    are stream-copied with -an anyway: an interpreter who hears the speech will
    mark words "signed" because they heard them.
  * Clip order is shuffled across sources so fatigue does not fall on one source.
  * The package is LOCAL ONLY. These are recordings of real interpreters from a
    dataset with its own terms; do not publish them to a hosted page.

Usage:
    python scripts/build_annotation_package.py --n 100 --out ~/ceiling_study
    (then: cd ~ && zip -r ceiling_study.zip ceiling_study)
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys

ROOT = os.path.expanduser('~/asan_canonical')
SOURCES = ('informburo', 'khabar', 'qazaqstantv')
HERE = os.path.dirname(os.path.abspath(__file__))


def video_path(root, entry):
    v = entry.get('video')
    if v:
        return v if os.path.isabs(v) else os.path.join(root, v)
    # qazaqstantv recollection: videos mirror the keypoint tree
    return entry['pose'].replace('/keypoints/', '/videos/').replace('.npz', '.mp4')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='~/ceiling_study')
    a = ap.parse_args()
    root, out = os.path.expanduser(a.root), os.path.expanduser(a.out)
    rng = random.Random(a.seed)

    sel = json.load(open(os.path.join(root, 'selection_manifest.json')))
    ann = {}
    for src in SOURCES:
        for e in json.load(open(os.path.join(root, src, 'annotations', 'kz', 'dev.json'))):
            ann[e['clip_id']] = (src, e)

    per = {s: [m['clip_id'] for m in sel if m['source'] == s] for s in SOURCES}
    k = {s: a.n // len(SOURCES) + (1 if i < a.n % len(SOURCES) else 0)
         for i, s in enumerate(SOURCES)}
    picked = []
    for s in SOURCES:
        ids = sorted(per[s])
        rng.shuffle(ids)
        got = 0
        for cid in ids:
            if got >= k[s]:
                break
            src, e = ann[cid]
            vp = video_path(root, e)
            if not e.get('text', '').strip() or not os.path.exists(vp):
                continue
            picked.append((cid, src, e, vp))
            got += 1
        if got < k[s]:
            sys.exit(f'[fatal] only {got}/{k[s]} {s} clips have a video file')
    rng.shuffle(picked)

    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, 'clips'))
    clips = []
    for i, (cid, src, e, vp) in enumerate(picked):
        dst = os.path.join(out, 'clips', f'{cid}.mp4')
        r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', vp, '-an', '-c:v', 'copy',
                            '-movflags', '+faststart', dst], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f'[fatal] ffmpeg failed on {vp}: {r.stderr[:300]}')
        text = e['text'].strip()
        clips.append({'n': i + 1, 'clip_id': cid, 'source': src, 'text': text,
                      'tokens': text.split()})
    meta = {'built_from': 'frozen selection set (dev)', 'seed': a.seed,
            'n': len(clips), 'by_source': {s: sum(c['source'] == s for c in clips)
                                           for s in SOURCES}}
    with open(os.path.join(out, 'data.js'), 'w', encoding='utf-8') as fh:
        fh.write('window.STUDY = ' + json.dumps({'meta': meta, 'clips': clips},
                                                 ensure_ascii=False) + ';\n')
    shutil.copy(os.path.join(HERE, 'annotation_tool.html'), os.path.join(out, 'index.html'))
    with open(os.path.join(out, 'README.txt'), 'w', encoding='utf-8') as fh:
        fh.write(README)
    size = sum(os.path.getsize(os.path.join(out, 'clips', f)) for f in os.listdir(os.path.join(out, 'clips')))
    print(f'[write] {out}: {len(clips)} clips {meta["by_source"]}, {size / 1e6:.1f} MB of video')


README = """CEILING STUDY -- what part of the speech transcript was actually signed?
==========================================================================

Open index.html in Chrome or Firefox (double-click; no internet needed).

For each clip:
  1. Watch the interpreter (there is no sound, on purpose). Use 0.5x speed and
     the frame-step buttons as needed.
  2. The transcript below the video is what was SAID on air. Click words to mark
     them:  green = this was signed,  red = not signed,  grey = unsure.
     Shortcuts: "all signed" / "all not signed", then fix the exceptions.
  3. Answer the two questions under the transcript.
  4. Optional but very valuable: in the box, write in Kazakh or Russian what the
     interpreter actually signed, if it differs from the transcript.

Progress saves automatically in this browser. When finished (or to hand over
part-way), press "Export results" and send the downloaded .json file back.
"Import" restores a previous export, e.g. on another computer.

------------------------------------------------------------------------------
ИССЛЕДОВАНИЕ: какая часть речи диктора была действительно переведена на жестовый
язык?

Откройте index.html в Chrome или Firefox (интернет не нужен). Звука нет
намеренно. Для каждого фрагмента отметьте слова текста: зелёный = показано
жестами, красный = не показано, серый = не уверен(а). Ответьте на два вопроса и,
по возможности, напишите, что именно было показано. В конце нажмите
"Export results" и отправьте полученный файл .json.
"""

if __name__ == '__main__':
    main()
