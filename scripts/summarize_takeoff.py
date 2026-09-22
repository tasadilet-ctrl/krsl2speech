#!/usr/bin/env python3
"""
Apply the pre-registered takeoff rule to a rescore_checkpoints.py output dir.

E3 found training here is bimodal: a run either starts using the pose input or
settles into fluent, content-free broadcast boilerplate. The rule, fixed in
EXPERIMENT_e6_token_alignment.md from the six runs scored before E6
(chrF 29.8 vs <= 22.1, content recall 0.128 vs <= 0.037):

    takeoff  <=>  chrF >= 26  and  content recall >= 0.08

Reads each <label>.metrics.json (the `clean` split, i.e. contamination
excluded) and <label>.predictions.jsonl, and groups labels by the prefix
before the first underscore (B_s0, E5_s1, E6_s2 ...) to report a count per arm.

    python scripts/summarize_takeoff.py --dir output/e6_rescore
"""
import argparse
import collections
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--chrf', type=float, default=26.0)
    ap.add_argument('--recall', type=float, default=0.08)
    args = ap.parse_args()

    rows = []
    for mpath in sorted(glob.glob(os.path.join(args.dir, '*.metrics.json'))):
        label = os.path.basename(mpath)[:-len('.metrics.json')]
        if '.sweep_' in label:
            continue
        m = json.load(open(mpath))['clean']['all']
        hyps = [json.loads(l)['hyp'].strip()
                for l in open(os.path.join(args.dir, f'{label}.predictions.jsonl'))]
        c = collections.Counter(hyps)
        took = m['chrf'] >= args.chrf and m['content_recall'] >= args.recall
        rows.append((label, m['chrf'], m['content_recall'], m['bleu'], len(c),
                     len(hyps), c.most_common(1)[0][1] / len(hyps), took))
    if not rows:
        raise SystemExit(f"[fatal] no *.metrics.json in {args.dir}")

    print(f"takeoff rule: chrF >= {args.chrf:g} and content recall >= {args.recall:g}\n")
    print(f"{'checkpoint':12}{'chrF':>7}{'recall':>8}{'BLEU':>6}{'distinct':>11}"
          f"{'top-1':>7}  takeoff")
    for label, chrf, rec, bleu, nd, n, top1, took in rows:
        print(f"{label:12}{chrf:7.2f}{rec:8.3f}{bleu:6.2f}{nd:>7}/{n:<3}{top1:7.1%}  "
              f"{'YES' if took else 'no'}")

    arms = collections.defaultdict(list)
    for r in rows:
        arms[r[0].split('_')[0]].append(r)
    print("\nper arm:")
    for arm, rs in arms.items():
        k = sum(r[-1] for r in rs)
        mean_chrf = sum(r[1] for r in rs) / len(rs)
        mean_rec = sum(r[2] for r in rs) / len(rs)
        print(f"  {arm:6} {k}/{len(rs)} take off   mean chrF {mean_chrf:5.2f}   "
              f"mean recall {mean_rec:.3f}")


if __name__ == '__main__':
    main()
