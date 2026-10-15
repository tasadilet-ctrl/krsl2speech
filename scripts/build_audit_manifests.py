#!/usr/bin/env python3
"""
Corpus-audit manifests: every usable dev / test clip in a broadcast with >= 2
clips (a lone clip has no neighbours to be compared against).

Only dev and test are auditable with the current checkpoints: they were trained
on the train split and have memorised its clip-transcript pairs, misaligned or
not, so a train audit would flag nothing. Auditing train needs models that did
not see the clips (cross-fitting).

Writes <root>/audit_dev.json and <root>/audit_test.json.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_canonical_splits as bcs   # noqa: E402

root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/asan_canonical')
for split in ('dev', 'test'):
    out, skipped = [], 0
    for src in bcs._SOURCES:
        by_v = collections.defaultdict(list)
        for e in bcs.load(root, src, split):
            if bcs._passes_dataset_filters(e):
                by_v[bcs.video_id(e)].append(e)
        for v, es in sorted(by_v.items()):
            if len(es) < 2:
                skipped += 1
                continue
            for e in sorted(es, key=lambda e: e['clip_id']):
                out.append({'clip_id': e['clip_id'], 'source': src, 'video_id': v, 'T': e.get('T', 0)})
    json.dump(out, open(os.path.join(root, f'audit_{split}.json'), 'w'), ensure_ascii=False)
    print(f'[write] audit_{split}.json: {len(out)} clips '
          f'{dict(collections.Counter(m["source"] for m in out))}; '
          f'{skipped} single-clip broadcasts skipped')
