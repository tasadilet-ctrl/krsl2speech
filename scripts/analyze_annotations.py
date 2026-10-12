#!/usr/bin/env python3
"""
Analyse ceiling-study exports (scripts/annotation_tool.html) against the
predictions already on disk. No decoding.

Reports:
  1. What fraction of reference content words was actually SIGNED -- the
     ceiling on content recall for any model trained on speech transcripts.
  2. Signed fraction by position in the sentence. If the end of each sentence
     is mostly unsigned, that explains the 2.2x positional recall decay
     (NOTES_positional_recall_and_lag.md) without any lag.
  3. Per checkpoint: recall of SIGNED vs NOT-SIGNED words, raw and
     pose-attributable (clean minus pose-shuffled, when shuffled predictions
     exist). If unsigned-word recall sits at the text-prior level, the model is
     doing as well as the signal allows on those words.
  4. Clip-level answers (does the signing match the text; are hands readable),
     time per clip, and the projected cost of annotating 3-5k clips.
  5. With two or more annotators: Cohen's kappa on word labels.

Usage:
    PYTHONPATH=. python scripts/analyze_annotations.py ceiling_study_*.json \
        --pred-dirs output/all_rescore output/e4b_rescore output/e9b_rescore \
        --shuf-dirs output/e8_shuffled output/e9b_rescore_shuf
"""
import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import normalize_kazakh  # noqa: E402

BINS = 5


def words_with_labels(tokens, labels):
    """Expand raw whitespace tokens to normalised words, carrying each token's
    label and its relative position. Punctuation can split one token into
    several words ("Достық–Мойынты"); each inherits the token's label."""
    out = []
    flat = []
    for tok, lab in zip(tokens, labels):
        for w in normalize_kazakh(tok).split():
            flat.append((w, lab))
    n = len(flat)
    for i, (w, lab) in enumerate(flat):
        out.append({'w': w, 'label': lab, 'pos': i / max(n, 1), 'content': len(w) >= 4})
    return out


def load_exports(paths):
    anns = []
    for p in paths:
        d = json.load(open(p))
        if d.get('tool') != 'krsl_ceiling_v1':
            sys.exit(f'[fatal] {p} is not a ceiling-study export')
        anns.append((d.get('annotator') or os.path.basename(p), d))
    return anns


def load_preds(dirs):
    preds = {}
    for d in dirs or []:
        for f in sorted(glob.glob(os.path.join(d, '*.predictions.jsonl'))):
            label = os.path.basename(f)[:-len('.predictions.jsonl')]
            if '.sweep_' in label:
                continue
            preds.setdefault(label, {}).update(
                {json.loads(l)['clip_id']: json.loads(l)['hyp'] for l in open(f)})
    return preds


def kappa(a, b):
    cats = sorted({x for x in a + b})
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exports', nargs='+')
    ap.add_argument('--pred-dirs', nargs='*', default=[])
    ap.add_argument('--shuf-dirs', nargs='*', default=[])
    a = ap.parse_args()

    anns = load_exports(a.exports)
    name, primary = anns[0]
    done = [r for r in primary['results'] if r.get('complete')]
    print(f'annotator: {name}   complete clips: {len(done)} / {primary["n_total"]}')
    if not done:
        sys.exit('no complete clips yet')

    # ---- 1 + 2: signed fraction, overall / by source / by position --------
    per_src = collections.defaultdict(collections.Counter)
    pos = [collections.Counter() for _ in range(BINS)]
    clip_words = {}
    for r in done:
        ws = words_with_labels(r['tokens'], r['labels'])
        clip_words[r['clip_id']] = ws
        for w in ws:
            if not w['content']:
                continue
            per_src[r['source']][w['label']] += 1
            per_src['ALL'][w['label']] += 1
            pos[min(BINS - 1, int(BINS * w['pos']))][w['label']] += 1

    def frac(c):
        t = sum(c.values())
        return {k: c[k] / t for k in ('signed', 'not', 'unsure')}, t

    print('\n1. Content words (>= 4 chars) actually signed')
    print(f'   {"source":<12}{"signed":>8}{"not":>8}{"unsure":>8}{"words":>8}')
    for s in ('informburo', 'khabar', 'qazaqstantv', 'ALL'):
        if s in per_src:
            f, t = frac(per_src[s])
            print(f'   {s:<12}{f["signed"]:8.3f}{f["not"]:8.3f}{f["unsure"]:8.3f}{t:8d}')
    ceil = frac(per_src['ALL'])[0]['signed']
    print(f'   -> a model trained on speech transcripts can recover at most ~{ceil:.0%} '
          f'of reference content words from the signing')

    print('\n2. Signed fraction by position in the sentence (start -> end)')
    print('   ' + '  '.join(f'{frac(p)[0]["signed"]:.3f}' for p in pos))
    print('   compare: pose-attributable model recall by position was '
          '+0.173 +0.139 +0.132 +0.124 +0.080')

    # ---- 3: model recall on signed vs unsigned words -----------------------
    clean, shuf = load_preds(a.pred_dirs), load_preds(a.shuf_dirs)
    if clean:
        print('\n3. Model recall on SIGNED vs NOT-SIGNED content words '
              '(same clips; pose-attributable = clean - shuffled)')
        print(f'   {"checkpoint":<12}{"signed":>8}{"not":>8}{"ratio":>7}'
              f'{"pose|signed":>13}{"pose|not":>10}')
        for label in sorted(clean):
            got = collections.Counter(); tot = collections.Counter()
            sgot = collections.Counter()
            for cid, ws in clip_words.items():
                if cid not in clean[label]:
                    continue
                hyp = set(normalize_kazakh(clean[label][cid]).split())
                shy = set(normalize_kazakh(shuf.get(label, {}).get(cid, '')).split())
                for w in ws:
                    if w['content'] and w['label'] in ('signed', 'not'):
                        tot[w['label']] += 1
                        got[w['label']] += w['w'] in hyp
                        sgot[w['label']] += w['w'] in shy
            if not tot['signed'] or not tot['not']:
                continue
            rs, rn = got['signed'] / tot['signed'], got['not'] / tot['not']
            has_shuf = label in shuf
            ps = rs - sgot['signed'] / tot['signed'] if has_shuf else float('nan')
            pn = rn - sgot['not'] / tot['not'] if has_shuf else float('nan')
            print(f'   {label:<12}{rs:8.3f}{rn:8.3f}{rs / max(rn, 1e-9):7.1f}'
                  f'{ps:13.3f}{pn:10.3f}')
        print('   signed = share of the SIGNED content the model recovers (the honest recall).')
        print('   not    = recall of words that were never signed; anything above the')
        print('            pose-shuffled level there is the text prior guessing, not reading.')

    # ---- 4: clip-level answers and cost -------------------------------------
    al = collections.Counter(r['align'] for r in done)
    qu = collections.Counter(r['quality'] for r in done)
    secs = [r['seconds'] for r in done if r.get('seconds')]
    print('\n4. Does the signing match the text?  ' +
          ', '.join(f'{k}: {v}' for k, v in al.most_common()))
    print('   Are the hands readable?             ' +
          ', '.join(f'{k}: {v}' for k, v in qu.most_common()))
    st = sum(1 for r in done if (r.get('signed_text') or '').strip())
    print(f'   free-text "what was signed" provided for {st}/{len(done)} clips')
    if secs:
        med = sorted(secs)[len(secs) // 2]
        print(f'   median {med} s per clip -> 3,000 clips ~{3000 * med / 3600:.0f} h, '
              f'5,000 clips ~{5000 * med / 3600:.0f} h of annotator time '
              f'(word marking; writing full glosses will take longer)')

    # ---- 5: agreement -------------------------------------------------------
    if len(anns) > 1:
        print('\n5. Inter-annotator agreement (Cohen\'s kappa, word labels, '
              'clips both completed)')
        for (n2, other) in anns[1:]:
            o = {r['clip_id']: r for r in other['results'] if r.get('complete')}
            A, B = [], []
            for r in done:
                if r['clip_id'] in o:
                    A += r['labels']; B += o[r['clip_id']]['labels']
            if A:
                print(f'   {name} vs {n2}: kappa {kappa(A, B):.3f} over {len(A)} words')


if __name__ == '__main__':
    main()
