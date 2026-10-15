#!/usr/bin/env python3
"""
Corpus audit report from within-broadcast PMI scores (eval_retrieval --pool video).

For every clip i of a broadcast with sentences 1..n, using PMI scores S[clip, text]:
  row margin   S[i,i] - max_{j!=i} S[i,j]   does the CLIP prefer another sentence?
  col margin   S[i,i] - max_{k!=i} S[k,i]   does its SENTENCE prefer another clip?
A clip whose pairing is wrong should look bad from both sides; "mutual" =
both margins < 0. Single-sided losses are mostly ordinary model error (the
model misses top-1 on ~17% of clean test clips), so the strength of evidence
is the margin, not the rank.

Validation, because a flag rate alone cannot separate data errors from model
errors: are flags ENRICHED among clips with independent signs of a bad
reference?
  * implausible speech rate (words/s outside 1-4: text and clip cover
    different stretches of speech)
  * qazaqstantv clips whose training text is the speech-recognition fallback
    (no human transcript) -- the ~10% the colleague's manifest flags implicitly.
If the audit sees real problems, both groups are flagged more often.

    python scripts/audit_report.py --out output/audit_report
"""
import argparse
import collections
import csv
import json
import math
import os

import numpy as np

ROOT = os.path.expanduser('~/asan_canonical')
# Lives in a colleague's home directory, which differs per box and is not
# this repo's to publish. Set QAZ_RECOLLECT to that jsonl; the section of
# the report that needs it is skipped when it is unset (see os.path.exists).
QZ = os.path.expanduser(os.environ.get('QAZ_RECOLLECT', ''))
FPS = {'informburo': 50, 'khabar': 25, 'qazaqstantv': 30}


def rate_ci(k, n):
    """Wilson 95% interval for k/n."""
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    p = k / n; z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dumps', nargs='+', default=['output/audit_dev/B_s0.within_video.json',
                                                   'output/audit_test/B_s0.within_video.json'])
    ap.add_argument('--out', default='output/audit_report')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ann = {}
    for sp in ('dev', 'test', 'train'):
        for src in FPS:
            for e in json.load(open(os.path.join(ROOT, src, 'annotations', 'kz', f'{sp}.json'))):
                ann[e['clip_id']] = (sp, src, e)
    human = {}
    if os.path.exists(QZ):
        for l in open(QZ):
            r = json.loads(l)
            human[r['clip_id']] = (r.get('text_human') or '').strip()

    rows = []
    for path in a.dumps:
        for g in json.load(open(path)):
            S = np.array(g['raw']) - np.array(g['null'])[None, :]
            n = len(g['ids'])
            for i, cid in enumerate(g['ids']):
                others = [j for j in range(n) if j != i]
                rm = S[i, i] - S[i, others].max()
                cm = S[i, i] - S[others, i].max()
                best = others[int(np.argmax(S[i, others]))]
                sp, src, e = ann[cid]
                secs = e['T'] / FPS[src]
                wps = len(e['text'].split()) / max(secs, 1e-6)
                tsrc = ''
                if src == 'qazaqstantv':
                    tsrc = 'human' if human.get(cid) and human[cid] == e['text'].strip() else 'asr'
                rows.append({'clip_id': cid, 'split': sp, 'source': src, 'video': g['video'],
                             'pool': n, 'own_rank': 1 + int((S[i, others] >= S[i, i]).sum()),
                             'row_margin': float(rm), 'col_margin': float(cm),
                             'mutual': bool(rm < 0 and cm < 0),
                             'best_other_is_adjacent': abs(best - i) == 1,
                             'best_other_id': g['ids'][best], 'wps': wps,
                             'wps_outlier': wps < 1 or wps > 4, 'text_source': tsrc,
                             'text': e['text'].strip(),
                             'best_other_text': ann[g['ids'][best]][2]['text'].strip()})

    with open(os.path.join(a.out, 'audit_all.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f'clips audited: {len(rows)}  (broadcasts with >= 2 clips)\n')
    print(f'{"source":<12}{"clips":>7}{"own top-1":>11}{"mutual mismatch":>18}   95% CI')
    for src in list(FPS) + ['ALL']:
        R = [r for r in rows if src == 'ALL' or r['source'] == src]
        top1 = np.mean([r['own_rank'] == 1 for r in R])
        p, lo, hi = rate_ci(sum(r['mutual'] for r in R), len(R))
        print(f'{src:<12}{len(R):>7}{top1:>11.3f}{p:>17.1%}   [{lo:.1%}, {hi:.1%}]')

    print('\nValidation -- are flags enriched where the reference is independently suspect?')
    def enrich(name, sel):
        A = [r for r in rows if sel(r)]; B = [r for r in rows if sel(r) is False]
        pa = rate_ci(sum(r['mutual'] for r in A), len(A))
        pb = rate_ci(sum(r['mutual'] for r in B), len(B))
        ratio = pa[0] / pb[0] if pb[0] else float('nan')
        print(f'  {name:<44} {pa[0]:6.1%} [{pa[1]:.1%},{pa[2]:.1%}] n={len(A):<5} vs '
              f'{pb[0]:6.1%} [{pb[1]:.1%},{pb[2]:.1%}] n={len(B):<6}  x{ratio:.1f}')
    enrich('words/s outside 1-4  vs  inside', lambda r: r['wps_outlier'])
    enrich('qazaqstantv: ASR-fallback text  vs  human',
           lambda r: (r['text_source'] == 'asr') if r['source'] == 'qazaqstantv' else None)

    M = [r for r in rows if r['mutual']]
    if M:
        # Compare with chance: in small pools a random other sentence is often adjacent.
        pos = collections.defaultdict(list)
        for r in rows:
            pos[(r['source'], r['video'])].append(r['clip_id'])
        exp = []
        for r in M:
            ids = sorted(pos[(r['source'], r['video'])]); i = ids.index(r['clip_id']); k = len(ids)
            exp.append(((i > 0) + (i < k - 1)) / (k - 1))
        print(f'\nflagged clips whose better-fitting sentence is adjacent: '
              f'{np.mean([r["best_other_is_adjacent"] for r in M]):.0%} '
              f'(a random other sentence would be adjacent {np.mean(exp):.0%}; only an excess '
              f'over that would indicate boundary shifts)')
    worst = sorted(M, key=lambda r: r['row_margin'] + r['col_margin'])
    with open(os.path.join(a.out, 'flagged_worst.json'), 'w', encoding='utf-8') as fh:
        json.dump(worst, fh, ensure_ascii=False, indent=1)
    print(f'\n[write] {a.out}/audit_all.csv ({len(rows)} clips), '
          f'{a.out}/flagged_worst.json ({len(M)} mutual mismatches, worst first)')


if __name__ == '__main__':
    main()
