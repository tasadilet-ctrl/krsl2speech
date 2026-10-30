#!/usr/bin/env python3
"""
A data root identical to the canonical one except that listed qazaqstantv
clips carry a new reference text, in the listed splits only.

Unlike build_asr_fragment_root.py (which rewrote every split), the split list
is explicit, so a TRAINING arm can change train targets while dev and test
keep their original references -- evaluation then measures the same thing in
every arm. Everything except qazaqstantv/annotations is a symlink into the
canonical root (khabar, informburo, poses, selection and retrieval manifests).

    python scripts/build_text_override_root.py \
        --overrides output/e15/overrides_train.json --splits train \
        --out ~/asan_canonical_llmref
"""
import argparse
import json
import os
import sys

_SPLITS = ('train', 'dev', 'test')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical', default='~/asan_canonical')
    ap.add_argument('--overrides', required=True, help='json {clip_id: text}')
    ap.add_argument('--splits', nargs='+', required=True, choices=_SPLITS)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    canon, out = os.path.expanduser(a.canonical), os.path.expanduser(a.out)
    over = json.load(open(a.overrides))

    ann = os.path.join(out, 'qazaqstantv', 'annotations', 'kz')
    os.makedirs(ann, exist_ok=True)
    summary, used = {}, set()
    for split in _SPLITS:
        rows = json.load(open(os.path.join(canon, 'qazaqstantv', 'annotations', 'kz', f'{split}.json')))
        n = 0
        if split in a.splits:
            for e in rows:
                t = over.get(e['clip_id'])
                if t is not None:
                    used.add(e['clip_id'])
                    if t.strip() != e['text'].strip():
                        e['text_original'] = e['text']
                        e['text'] = t
                        n += 1
        with open(os.path.join(ann, f'{split}.json'), 'w') as fh:
            json.dump(rows, fh, ensure_ascii=False)
        summary[split] = {'clips': len(rows), 'changed': n}
        print(f'[qazaqstantv] {split:5s}: {n:6d} / {len(rows):6d} references changed')
    unused = len(set(over) - used)
    if unused:
        print(f'[note] {unused} overrides name clips outside the chosen splits; ignored')

    for name in os.listdir(canon):
        if name == 'qazaqstantv':
            continue
        p = os.path.join(out, name)
        if os.path.islink(p):
            os.unlink(p)
        elif os.path.exists(p):
            sys.exit(f'[fatal] {p} exists and is not a symlink')
        os.symlink(os.path.realpath(os.path.join(canon, name)), p)
    for name in os.listdir(os.path.join(canon, 'qazaqstantv')):
        if name == 'annotations':
            continue
        p = os.path.join(out, 'qazaqstantv', name)
        if not os.path.lexists(p):
            os.symlink(os.path.realpath(os.path.join(canon, 'qazaqstantv', name)), p)
    with open(os.path.join(out, 'provenance_override.json'), 'w') as fh:
        json.dump({'from': canon, 'overrides': os.path.abspath(a.overrides),
                   'splits_changed': a.splits, 'summary': summary}, fh, indent=1)
    print(f'\nDone.  --root {out}')


if __name__ == '__main__':
    main()
