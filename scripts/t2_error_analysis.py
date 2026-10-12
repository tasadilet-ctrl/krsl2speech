#!/usr/bin/env python3
"""
E12 T2 error analysis: when within-broadcast retrieval is wrong, does it pick a
NEIGHBOURING sentence (likely the same news story) more often than random
errors would?

Broadcasts (especially khabar bulletins) contain several stories, so a T2 pool
mixes "other stories" (easy) with "other sentences of this story" (hard).
Neighbours are the best available proxy for the second. Clip ids are
zero-padded, so sorted order is broadcast order.

Usage: python scripts/t2_error_analysis.py output/e12_dev_t2/B_s0.within_video.json
"""
import json
import sys

import numpy as np


def main(path):
    groups = json.load(open(path))
    wrong = adj = 0
    expected = 0.0
    dist = []
    for g in groups:
        raw, null = np.array(g['raw']), np.array(g['null'])
        pmi = raw - null[None, :]
        n = len(g['ids'])
        for i in range(n):
            j = int(np.argmax(pmi[i]))
            if j == i:
                continue
            wrong += 1
            adj += abs(i - j) == 1
            dist.append(abs(i - j))
            neighbours = (i > 0) + (i < n - 1)
            expected += neighbours / (n - 1)      # adjacency rate if errors were random
    print(f'{path}')
    if not wrong:
        print('  no errors'); return
    print(f'  wrong picks: {wrong}')
    print(f'  wrong pick is the adjacent sentence: {adj / wrong:.1%}   '
          f'(random errors would give {expected / wrong:.1%})')
    d = np.array(dist)
    print(f'  distance of wrong pick: 1: {np.mean(d == 1):.0%}  2: {np.mean(d == 2):.0%}  '
          f'3+: {np.mean(d >= 3):.0%}')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        main(p)
