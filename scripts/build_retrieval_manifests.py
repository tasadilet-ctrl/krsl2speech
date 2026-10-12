#!/usr/bin/env python3
"""
E12 manifests (EXPERIMENT_e12_retrieval_task.md).

  T1 test: mirror of the frozen DEV selection -- one clip per video, four
           duration quartiles, 200 per source -- built on the test split with
           the same salt and the same code path (build_canonical_splits).
  T2 dev / T2 test: within-broadcast pools -- up to --videos per source with
           >= --min-clips usable clips, ALL of each video's clips.

Writes into the canonical root:
  selection_manifest_test.json, within_video_dev.json, within_video_test.json
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_canonical_splits as bcs   # noqa: E402

SALT = 'krsl-canonical-v1'


def selection(root, split, per_source):
    """build_canonical_splits.build_selection with the split as a parameter."""
    orig = bcs.load
    try:
        bcs.load = lambda r, s, _split: orig(r, s, split)
        return bcs.build_selection(root, per_source, SALT)
    finally:
        bcs.load = orig


def within_video(root, split, n_videos, min_clips):
    out = []
    for src in bcs._SOURCES:
        by_v = collections.defaultdict(list)
        for e in bcs.load(root, src, split):
            if bcs._passes_dataset_filters(e):
                by_v[bcs.video_id(e)].append(e)
        vids = sorted((v for v, es in by_v.items() if len(es) >= min_clips),
                      key=lambda v: bcs.stable_frac(v, SALT))[:n_videos]
        for v in vids:
            for e in sorted(by_v[v], key=lambda e: e['clip_id']):
                out.append({'clip_id': e['clip_id'], 'source': src, 'video_id': v,
                            'T': e.get('T', 0)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='~/asan_canonical')
    ap.add_argument('--videos', type=int, default=60)
    ap.add_argument('--min-clips', type=int, default=5)
    a = ap.parse_args()
    root = os.path.expanduser(a.root)

    # Sanity: the generalised builder must reproduce the frozen dev selection.
    dev = selection(root, 'dev', 200)
    frozen = json.load(open(os.path.join(root, 'selection_manifest.json')))
    assert [m['clip_id'] for m in dev] == [m['clip_id'] for m in frozen], \
        'generalised builder does not reproduce the frozen dev selection'
    print('[check] generalised builder reproduces the frozen dev selection exactly')

    outs = {'selection_manifest_test.json': selection(root, 'test', 200),
            'within_video_dev.json': within_video(root, 'dev', a.videos, a.min_clips),
            'within_video_test.json': within_video(root, 'test', a.videos, a.min_clips)}
    for name, m in outs.items():
        json.dump(m, open(os.path.join(root, name), 'w'), ensure_ascii=False, indent=1)
        by_src = collections.Counter(x['source'] for x in m)
        vids = len({(x['source'], x['video_id']) for x in m})
        print(f'[write] {name}: {len(m)} clips, {vids} videos, {dict(by_src)}')


if __name__ == '__main__':
    main()
