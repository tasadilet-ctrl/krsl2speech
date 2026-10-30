#!/usr/bin/env python3
"""
E15 step 2: accept or reject each LLM-rewritten reference, and write the
accepted ones as an override map {clip_id: text}.

The LLM is told to cover exactly what SPEECH (the clip's ASR) covers. Two
guards check that against the ASR, since the ASR is the only text whose
boundaries are trustworthy:
  length   words(llm) / words(asr) within [LO, HI]
  overlap  chrF(llm, asr), both lower-cased with punctuation stripped, >= MIN_CHRF
           (character n-grams tolerate the spelling and numeral fixes the LLM
           is asked to make; a different sentence scores far lower)
Whisper sometimes emits a passage twice; the LLM is told to write it once, so
both guards compare against the ASR with repeated passages (>= 5 words)
collapsed. Ellipses, which the LLM adds despite being told not to, are removed.
A rejected clip keeps its current reference -- the conservative default.

    .venv/bin/python scripts/realign_guard.py output/e15/raw_*.jsonl \
        --out output/e15/overrides_train.json --split train
"""
import argparse
import collections
import json
import random
import re

from sacrebleu.metrics import CHRF

LO, HI, MIN_CHRF = 0.67, 1.5, 45.0
_chrf = CHRF()


def norm(s):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', s.lower())).strip()


def dedup_repeats(text, min_n=5):
    """Drop later copies of any passage of >= min_n words seen earlier."""
    w = text.split()
    changed = True
    while changed:
        changed = False
        for n in range(len(w) // 2, min_n - 1, -1):
            seen = {}
            for i in range(len(w) - n + 1):
                key = tuple(w[i:i + n])
                if key in seen and seen[key] + n <= i:
                    del w[i:i + n]
                    changed = True
                    break
                seen.setdefault(key, i)
            if changed:
                break
    return ' '.join(w)


def clean(t):
    t = re.sub(r'\.{3,}|…', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def chrf(h, r):
    return _chrf.sentence_score(norm(h), [norm(r)]).score


def judge(r):
    if not r.get('llm'):
        return 'no output', None
    if r.get('finish') != 'stop':
        return 'truncated', None
    r['llm'] = clean(r['llm'])
    ref = dedup_repeats(norm(r['speech']))
    w = len(r['llm'].split()) / max(1, len(ref.split()))
    c = chrf(r['llm'], ref)
    if not LO <= w <= HI:
        return 'length', (w, c)
    if c < MIN_CHRF:
        return 'overlap', (w, c)
    return 'ok', (w, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw', nargs='+')
    ap.add_argument('--split', default=None, help='keep only this split')
    ap.add_argument('--out', default=None)
    ap.add_argument('--show', type=int, default=0, help='print N random accepted + all rejected')
    a = ap.parse_args()

    rows = {}
    for p in a.raw:
        for l in open(p):
            r = json.loads(l)
            if a.split is None or r['split'] == a.split:
                rows[r['clip_id']] = r
    verdicts, over, stats = collections.Counter(), {}, []
    for cid, r in rows.items():
        v, m = judge(r)
        r['verdict'], r['metrics'] = v, m
        verdicts[v] += 1
        if v == 'ok':
            over[cid] = r['llm']
            stats.append((chrf(r['current'], r['speech']), m[1]))
    n = len(rows)
    print(f'{n} clips: ' + ', '.join(f'{k} {c} ({c / n:.1%})' for k, c in verdicts.most_common()))
    if stats:
        cur, new = zip(*stats)
        print(f'accepted: chrF vs ASR  current ref {sum(cur) / len(cur):.1f}  ->  LLM {sum(new) / len(new):.1f}')
        changed = sum(over[c].strip() != rows[c]['current'] for c in over)
        print(f'accepted and different from the current reference: {changed}/{len(over)}')
    if a.show:
        acc = [r for r in rows.values() if r['verdict'] == 'ok']
        random.Random(0).shuffle(acc)
        for r in acc[:a.show] + [r for r in rows.values() if r['verdict'] not in ('ok',)]:
            print(f"\n=== {r['clip_id']}  [{r['verdict']}]  {r['metrics']}")
            print(f"  ASR    : {r['speech']}")
            print(f"  CURRENT: {r['current']}")
            print(f"  LLM    : {r['llm']}")
    if a.out:
        json.dump(over, open(a.out, 'w'), ensure_ascii=False)
        print(f'[write] {a.out}: {len(over)} overrides')


if __name__ == '__main__':
    main()
