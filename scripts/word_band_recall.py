#!/usr/bin/env python3
"""
What a checkpoint's translations get right, by word type (NOTES_e16_error_analysis.md).

For each predictions.jsonl (rescore_checkpoints.py output):
  recall by band   share of reference content words (len >= 4, or numbers) that
                   appear anywhere in the hypothesis, split by how often the word
                   occurs in the TRAINING references: common (500+), mid
                   (20-500), rare (< 20), unseen, number
  precision        share of hypothesis content words found in the reference
                   (1 - precision = invented content)

    python scripts/word_band_recall.py --train-root ~/asan_canonical_llmref \
        E16_s0=output/e16early_rescore/E16_s0.predictions.jsonl ...
"""
import argparse
import collections
import json
import os
import re

SOURCES = ('informburo', 'khabar', 'qazaqstantv')
BANDS = ('common', 'mid', 'rare', 'unseen', 'number')


def words(s):
    return re.sub(r'[^\w\s]', ' ', s.lower()).split()


def band(w, freq):
    if w.isdigit():
        return 'number'
    f = freq[w]
    return 'unseen' if f == 0 else 'rare' if f < 20 else 'mid' if f < 500 else 'common'


def analyse(path, freq):
    rows = [json.loads(l) for l in open(path)]
    hk = [k for k in rows[0] if 'hyp' in k][0]
    rec = collections.defaultdict(lambda: [0, 0])
    p_hit = p_n = 0
    for r in rows:
        hw, rw = set(words(r[hk])), set(words(r['ref']))
        for w in rw:
            if len(w) >= 4 or w.isdigit():
                b = band(w, freq)
                rec[b][0] += w in hw
                rec[b][1] += 1
        for w in words(r[hk]):
            if len(w) >= 4:
                p_hit += w in rw
                p_n += 1
    out = {b: rec[b][0] / rec[b][1] for b in BANDS if rec[b][1]}
    out['precision'] = p_hit / max(p_n, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('preds', nargs='+', help='label=predictions.jsonl')
    ap.add_argument('--train-root', default='~/asan_canonical_llmref')
    ap.add_argument('--out', default=None, help='write results as json')
    a = ap.parse_args()
    freq = collections.Counter()
    for src in SOURCES:
        p = os.path.join(os.path.expanduser(a.train_root), src, 'annotations', 'kz', 'train.json')
        for e in json.load(open(p)):
            freq.update(words(e['text']))
    res = {}
    print(f'{"run":<14}' + ''.join(f'{b:>9}' for b in BANDS) + f'{"precision":>11}')
    for spec in a.preds:
        label, path = spec.split('=', 1)
        res[label] = r = analyse(path, freq)
        print(f'{label:<14}' + ''.join(f'{r.get(b, float("nan")):9.3f}' for b in BANDS)
              + f'{r["precision"]:11.3f}')
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
