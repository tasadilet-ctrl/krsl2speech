#!/usr/bin/env python3
"""
Cross-fit halves of the TRAIN split, by broadcast (video), for auditing train.

A clip is audited against the other sentences of its own broadcast, so the
whole broadcast must be unseen by the model that scores it: halves are split by
video, never by clip. Deterministic (salted SHA-256, not Python's hash()).

Writes <root>/crossfit_half_A.json and _B.json: lists of "source/video_id".
"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_canonical_splits as bcs   # noqa: E402

root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/asan_canonical')
halves = {'A': set(), 'B': set()}
clips = collections.Counter()
for src in bcs._SOURCES:
    for e in bcs.load(root, src, 'train'):
        if not bcs._passes_dataset_filters(e):
            continue
        key = f'{src}/{bcs.video_id(e)}'
        h = 'A' if bcs.stable_frac(key, 'crossfit-v1') < 0.5 else 'B'
        halves[h].add(key); clips[(h, src)] += 1
for h, keys in halves.items():
    json.dump(sorted(keys), open(os.path.join(root, f'crossfit_half_{h}.json'), 'w'))
    print(f'half {h}: {len(keys)} broadcasts, clips ' +
          ', '.join(f'{s} {clips[(h, s)]}' for s in bcs._SOURCES))
assert not (halves['A'] & halves['B'])
