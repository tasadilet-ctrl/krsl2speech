#!/usr/bin/env python3
"""
E15 step 2 summary: did B_s0 (never trained on dev) find the rewritten
qazaqstantv dev references a better fit for the signing than the originals?

Reads output/e15/precheck/<version>/summary.json and the matching
<label>.within_video.json dumps (eval_retrieval --pool video) and prints, per
reference version: T2 top-1 per source, and qazaqstantv's mutual-mismatch rate
(both row and column PMI margins < 0, as in audit_report.py).

Exit code 0 = proceed to step 3, 3 = stop (rewrite scored >= 0.03 below the
original), per EXPERIMENT_e15_llm_references.md.
"""
import json
import os
import sys

import numpy as np

D = 'output/e15/precheck'
VERSIONS = ['original', 'llm', 'asr']


def mutual_rate(dump, src='qazaqstantv'):
    k = n = 0
    for g in json.load(open(dump)):
        if g.get('source', src) != src:
            continue
        S = np.array(g['raw']) - np.array(g['null'])[None, :]
        m = len(g['ids'])
        for i in range(m):
            o = [j for j in range(m) if j != i]
            k += (S[i, i] < S[i, o].max()) and (S[i, i] < S[o, i].max())
            n += 1
    return k / max(n, 1), n


res = {}
print(f'{"refs":<10}{"ALL":>7}{"informburo":>12}{"khabar":>8}{"qazaq":>8}{"qazaq mutual":>14}')
for v in VERSIONS:
    p = os.path.join(D, v, 'summary.json')
    if not os.path.exists(p):
        print(f'{v:<10} missing'); continue
    s = json.load(open(p))['B_s0']
    ps = s['per_source']
    top = {k: ps[k]['pmi_top1'] for k in ps}
    allq = sum(ps[k]['pmi_top1'] * ps[k]['queries'] for k in ps) / sum(ps[k]['queries'] for k in ps)
    mr, n = mutual_rate(os.path.join(D, v, 'B_s0.within_video.json'))
    res[v] = top['qazaqstantv']
    print(f'{v:<10}{allq:>7.3f}{top["informburo"]:>12.3f}{top["khabar"]:>8.3f}'
          f'{top["qazaqstantv"]:>8.3f}{mr:>13.1%}  (n={n})')

d = res.get('llm', 0) - res.get('original', 0)
print(f'\nqazaqstantv: llm - original = {d:+.3f}')
if d >= 0.03:
    print('READING: rewritten references fit the signing better. Proceed to step 3.')
elif d > -0.03:
    print('READING: no evidence the rewrite fits better (within 0.03). Step 3 runs; a null is expected.')
else:
    print('READING: the rewrite damages references. STOP.')
    sys.exit(3)
