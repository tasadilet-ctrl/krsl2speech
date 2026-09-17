#!/usr/bin/env python3
"""
E0: build a canonical, leak-free dataset root plus a frozen selection manifest.

Fixes two defects found in the 2026-09-17 audit:

1. **Cross-version split leakage.** Archive qazaqstantv TRAIN shares 563 video
   IDs with the recollection's DEV (64.2% of its clips) and 533 with its TEST.
   Each manifest is internally split-disjoint -- which is why checking them
   separately missed this -- but the recollection re-split the same source
   videos, so a model trained on archive train is evaluated on videos it saw.

2. **Unrepresentative checkpoint selection.** The trainer generates on the
   first 25 unshuffled batches; sources concatenate in list order, so those 200
   clips are 100% informburo (10.8% of dev) and qazaqstantv (48.6%) is never
   generated on. This writes a frozen manifest stratified across all three
   sources and across duration, sampling distinct source VIDEOS so adjacent
   clips from one broadcast cannot dominate.

Held-out purity
---------------
Every video the initializer's fine-tuning saw (archive qazaqstantv TRAIN) is
forced into canonical TRAIN, so canonical dev/test are unseen by it. khabar and
informburo are byte-identical across versions and already train-disjoint, so
their splits are carried over unchanged.

This does NOT clear the initializer entirely: `ours_enriched_friend_mt5.pth` is
`pose_pretrain_v3 encoder + a colleague's mT5`, and that mT5's training history
is not recorded anywhere we hold. Runs from it stay EXPLORATORY until that
provenance is established -- see --require-provenance.

Usage:
    python scripts/build_canonical_splits.py \
        --archive /data/archive/asan-dataset \
        --clean   ~/asan_clean \
        --out     ~/asan_canonical
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys

_SPLITS = ('train', 'dev', 'test')
_SOURCES = ('informburo', 'khabar', 'qazaqstantv')


def video_id(entry):
    """Video ID from an explicit field, else from the `src__video__segNNN` id."""
    if entry.get('video_id'):
        return entry['video_id']
    m = re.match(r'[^_]+__(.+?)__seg', entry.get('clip_id', ''))
    if not m:
        raise ValueError(f"cannot derive video_id from {entry.get('clip_id')!r}")
    return m.group(1)


def load(root, source, split):
    p = os.path.join(root, source, 'annotations', 'kz', f'{split}.json')
    return json.load(open(p)) if os.path.exists(p) else []


def stable_frac(key, salt):
    """Deterministic [0,1) from a string -- reproducible across runs/machines
    (Python's hash() is salted per process and must not be used here)."""
    h = hashlib.sha256(f'{salt}:{key}'.encode()).digest()
    return int.from_bytes(h[:8], 'big') / 2 ** 64


def build_qazaqstantv(archive, clean, dev_clips, test_clips, salt):
    """Canonical qazaqstantv split with initializer-seen videos pinned to train."""
    seen = {video_id(x) for x in load(archive, 'qazaqstantv', 'train')}

    by_video = collections.defaultdict(list)
    for split in _SPLITS:
        for e in load(clean, 'qazaqstantv', split):
            by_video[video_id(e)].append(e)

    eligible = sorted(v for v in by_video if v not in seen)
    pinned = sorted(v for v in by_video if v in seen)

    # Deterministic shuffle, then fill dev and test to their clip targets.
    eligible.sort(key=lambda v: stable_frac(v, salt))

    assign, counts = {}, {'dev': 0, 'test': 0}
    for v in eligible:
        n = len(by_video[v])
        if counts['dev'] + n <= dev_clips:
            assign[v] = 'dev'; counts['dev'] += n
        elif counts['test'] + n <= test_clips:
            assign[v] = 'test'; counts['test'] += n
        else:
            assign[v] = 'train'
    for v in pinned:
        assign[v] = 'train'

    out = {s: [] for s in _SPLITS}
    for v, entries in by_video.items():
        out[assign[v]].extend(entries)
    for s in _SPLITS:
        out[s].sort(key=lambda e: e['clip_id'])
    return out, assign, len(pinned), len(eligible)


def _passes_dataset_filters(e, min_frames=25):
    """
    Mirror AsanDataset's own filtering. The manifest must list only clips that
    will actually exist in the loaded dataset -- otherwise the trainer's
    coverage check (correctly) refuses to run on a subset different from the
    one intended.
    """
    if not e.get('text', '').strip():
        return False
    if e.get('T', 0) < min_frames:
        return False
    if e.get('low_quality', False):
        return False
    return True


def build_selection(root, per_source, salt):
    """
    Frozen selection manifest: per source, sample distinct videos across
    duration quartiles, one clip per video. Selection must never be dominated
    by adjacent clips from a single broadcast.
    """
    manifest = []
    for source in _SOURCES:
        entries = load(root, source, 'dev')
        if not entries:
            continue
        by_video = collections.defaultdict(list)
        for e in entries:
            if _passes_dataset_filters(e):
                by_video[video_id(e)].append(e)

        # One representative clip per video (deterministic pick), then bucket
        # those by duration so short and long clips are both represented.
        reps = []
        for v, es in by_video.items():
            reps.append(min(es, key=lambda e: (stable_frac(e['clip_id'], salt))))
        reps.sort(key=lambda e: e.get('T', 0))

        n_buckets = 4
        per_bucket = max(1, per_source // n_buckets)
        size = max(1, len(reps) // n_buckets)
        picked = []
        for b in range(n_buckets):
            lo = b * size
            hi = len(reps) if b == n_buckets - 1 else (b + 1) * size
            bucket = sorted(reps[lo:hi], key=lambda e: stable_frac(e['clip_id'], salt))
            picked.extend(bucket[:per_bucket])

        for e in picked[:per_source]:
            manifest.append({'clip_id': e['clip_id'], 'source': source,
                             'video_id': video_id(e), 'T': e.get('T', 0)})
    manifest.sort(key=lambda m: m['clip_id'])
    return manifest


def verify(out_root, archive, clean):
    """Hard assertions -- this is the whole point of E0."""
    problems = []
    vids = {}
    for s in _SPLITS:
        vids[s] = set()
        for source in _SOURCES:
            vids[s] |= {video_id(x) for x in load(out_root, source, s)}

    for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
        ov = vids[a] & vids[b]
        if ov:
            problems.append(f'{a}/{b} share {len(ov)} video IDs')

    # The real check: nothing the initializer trained on may appear in eval.
    seen = set()
    for source in _SOURCES:
        seen |= {video_id(x) for x in load(archive, source, 'train')}
    for s in ('dev', 'test'):
        ov = vids[s] & seen
        if ov:
            problems.append(f'{len(ov)} canonical {s} videos were in ARCHIVE TRAIN '
                            f'(initializer has seen them)')
    return problems, vids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', default='/data/archive/asan-dataset')
    ap.add_argument('--clean', default='~/asan_clean')
    ap.add_argument('--out', default='~/asan_canonical')
    ap.add_argument('--dev-clips', type=int, default=2500,
                    help='target qazaqstantv dev clips drawn from unseen videos')
    ap.add_argument('--test-clips', type=int, default=2500)
    ap.add_argument('--selection-per-source', type=int, default=200,
                    help='clips per source in the frozen selection manifest')
    ap.add_argument('--salt', default='krsl-canonical-v1',
                    help='changing this reshuffles the split; keep it fixed')
    args = ap.parse_args()

    archive = os.path.expanduser(args.archive)
    clean = os.path.expanduser(args.clean)
    out = os.path.expanduser(args.out)

    splits, assign, n_pinned, n_eligible = build_qazaqstantv(
        archive, clean, args.dev_clips, args.test_clips, args.salt)

    ann = os.path.join(out, 'qazaqstantv', 'annotations', 'kz')
    os.makedirs(ann, exist_ok=True)
    for s in _SPLITS:
        with open(os.path.join(ann, f'{s}.json'), 'w') as fh:
            json.dump(splits[s], fh, ensure_ascii=False)
        n_vid = len({video_id(e) for e in splits[s]})
        print(f'[qazaqstantv] {s:5s}: {len(splits[s]):6d} clips / {n_vid:5d} videos')
    print(f'[qazaqstantv] {n_pinned} videos pinned to train (seen by initializer), '
          f'{n_eligible} eligible for eval')

    # khabar / informburo are identical in both versions and already
    # train-disjoint -- carry them over untouched.
    for source in ('khabar', 'informburo'):
        link, target = os.path.join(out, source), os.path.join(archive, source)
        if not os.path.exists(target):
            sys.exit(f'[fatal] missing {target}')
        if os.path.islink(link):
            os.unlink(link)
        elif os.path.exists(link):
            sys.exit(f'[fatal] {link} exists and is not a symlink')
        os.symlink(target, link)
        print(f'[link] {source} -> {target}')

    problems, vids = verify(out, archive, clean)
    print('\n=== verification ===')
    for s in _SPLITS:
        print(f'  {s:5s}: {len(vids[s])} videos')
    if problems:
        print('\nFAILED:')
        for p in problems:
            print(f'  - {p}')
        sys.exit(1)
    print('  no split overlap; no eval video seen in archive train  OK')

    selection = build_selection(out, args.selection_per_source, args.salt)
    sel_path = os.path.join(out, 'selection_manifest.json')
    with open(sel_path, 'w') as fh:
        json.dump(selection, fh, ensure_ascii=False, indent=1)
    by_src = collections.Counter(m['source'] for m in selection)
    print(f'\n[selection] {len(selection)} clips -> {sel_path}')
    print(f'[selection] by source: {dict(by_src)}')

    prov = {
        'built': '2026-09-17', 'salt': args.salt,
        'archive': archive, 'clean': clean,
        'dev_clips_target': args.dev_clips, 'test_clips_target': args.test_clips,
        'qazaqstantv_pinned_videos': n_pinned,
        'qazaqstantv_eligible_videos': n_eligible,
        'khabar_informburo': 'carried over from archive unchanged',
        'initializer_caveat': (
            "ours_enriched_friend_mt5.pth = pose_pretrain_v3 encoder + a "
            "colleague's mT5 whose training history is unrecorded. Canonical "
            "dev/test exclude everything archive TRAIN contained, but that mT5 "
            "component cannot be cleared from here. Label runs EXPLORATORY "
            "until its provenance is established."),
        'counts': {s: {'clips': len(splits[s]) if s in splits else None,
                       'videos': len(vids[s])} for s in _SPLITS},
    }
    with open(os.path.join(out, 'split_provenance.json'), 'w') as fh:
        json.dump(prov, fh, indent=1, ensure_ascii=False)

    print(f'\nDone.  export ASAN_ROOT={out}')


if __name__ == '__main__':
    main()
