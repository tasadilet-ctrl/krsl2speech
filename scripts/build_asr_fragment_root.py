#!/usr/bin/env python3
"""
Data arm E: replace qazaqstantv FRAGMENT references with the clip's Whisper
transcript (text_asr). Everything else is identical to the canonical root.

qazaqstantv's training text is `text_human` (website article sentences
fuzzy-aligned to the clip, 89.8%) with `text_asr` fallback. After the
colleague's slicing fix, 10.2% of canonical-train clips still carry a
`text_human` with fewer than half the words ASR heard in the same clip --
e.g. 19 s of speech labelled "Бес есе көп". Those references teach the model to
emit a fragment for a full clip of signing.

Rule (fixed before any result exists):
    fragment := text_human and text_asr both non-empty
                AND words(text_human) < FRAG_RATIO * words(text_asr)
    -> text := text_asr
Applied to train, dev AND test, so training and evaluation use the same
reference convention. Clip IDs, splits, poses and the selection manifest are
unchanged, so arm E is comparable clip-for-clip with arm B.

Layout at --out (khabar/informburo and all pose-derived tables are shared):
    khabar, informburo            -> symlinks into the canonical root
    qazaqstantv/annotations/kz/*  -> rewritten
    selection_manifest.json, score_quantiles.json, hand_ratio.json -> symlinks
    switched_clips.json           -> every changed clip, old and new text

Usage:
    python scripts/build_asr_fragment_root.py \
        --canonical ~/asan_canonical --out ~/asan_canonical_asrfrag
"""
import argparse
import json
import os
import sys

FRAG_RATIO = 0.5
_SPLITS = ('train', 'dev', 'test')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical', default='~/asan_canonical')
    # Lives in a colleague's home directory, which differs per box and is not
    # this repo's to publish. Set QAZ_RECOLLECT, or pass --recollect.
    ap.add_argument('--recollect',
                    default=os.environ.get('QAZ_RECOLLECT'),
                    help='qazaqstantv_kz_fixed.jsonl from the recollection '
                         '(default: $QAZ_RECOLLECT)')
    ap.add_argument('--out', default='~/asan_canonical_asrfrag')
    args = ap.parse_args()
    if not args.recollect:
        ap.error('--recollect is required (or set QAZ_RECOLLECT)')
    canon, out = os.path.expanduser(args.canonical), os.path.expanduser(args.out)
    # expanduser here too; the other paths get it and this one silently did not
    recollect = os.path.expanduser(args.recollect)

    fixed = {}
    for line in open(recollect):
        d = json.loads(line)
        fixed[d['clip_id']] = d

    ann = os.path.join(out, 'qazaqstantv', 'annotations', 'kz')
    os.makedirs(ann, exist_ok=True)
    switched, summary = [], {}
    for split in _SPLITS:
        rows = json.load(open(os.path.join(canon, 'qazaqstantv', 'annotations', 'kz',
                                           f'{split}.json')))
        n_sw = 0
        for e in rows:
            f = fixed.get(e['clip_id'])
            if f is None:
                sys.exit(f"[fatal] {e['clip_id']} missing from {args.recollect}")
            h = (f.get('text_human') or '').strip()
            a = (f.get('text_asr') or '').strip()
            if h and a and len(h.split()) < FRAG_RATIO * len(a.split()):
                if e['text'].strip() != h:
                    sys.exit(f"[fatal] {e['clip_id']}: canonical text is not its "
                             f"text_human -- manifest and recollection disagree")
                switched.append({'clip_id': e['clip_id'], 'split': split,
                                 'old_text_human': h, 'new_text_asr': a,
                                 'words': [len(h.split()), len(a.split())]})
                e['text'] = a
                e['text_source'] = 'text_asr (fragment text_human replaced)'
                n_sw += 1
        with open(os.path.join(ann, f'{split}.json'), 'w') as fh:
            json.dump(rows, fh, ensure_ascii=False)
        summary[split] = {'clips': len(rows), 'switched': n_sw}
        print(f"[qazaqstantv] {split:5s}: {n_sw:5d} / {len(rows):6d} switched to text_asr "
              f"({100 * n_sw / len(rows):.1f}%)")

    def link(name, target):
        p = os.path.join(out, name)
        if os.path.islink(p):
            os.unlink(p)
        elif os.path.exists(p):
            sys.exit(f"[fatal] {p} exists and is not a symlink")
        os.symlink(target, p)

    for src in ('khabar', 'informburo'):
        link(src, os.path.realpath(os.path.join(canon, src)))
    for f in ('selection_manifest.json', 'score_quantiles.json', 'hand_ratio.json'):
        if os.path.exists(os.path.join(canon, f)):
            link(f, os.path.join(canon, f))

    with open(os.path.join(out, 'switched_clips.json'), 'w') as fh:
        json.dump(switched, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out, 'provenance.json'), 'w') as fh:
        json.dump({'built': '2026-09-18', 'from': canon, 'rule':
                   f'words(text_human) < {FRAG_RATIO} * words(text_asr) -> use text_asr',
                   'splits': summary}, fh, indent=1)
    print(f"\nDone.  export ASAN_ROOT={out}")


if __name__ == '__main__':
    main()
