#!/usr/bin/env python3
"""
Early screen for training runs: score each run's epoch-5 checkpoint on the
frozen selection set as soon as it is written, and report its distinct-output
count -- the signal that separated takeoff from no-takeoff 18/18 in-sample and
called E7 and E9 correctly out-of-sample (NOTES_ep5_takeoff_predictor.md).

Record over 33 labelled runs (2026-10-11): under 500 distinct at epoch 5, 1 of
27 took off (E11_s2 from 419); 500-576, 1 of 3 (E9b_s3 from 547); 577+, 5 of 5.
Two cut-offs fitted in-sample ("never below 550", then "never below 500") were
each broken by the next experiment, so this reports a PROBABILITY, not a rule.

It is a MONITOR, not a launcher: it attaches to any running experiment's
output directories, so it composes with the existing run_*.sh schedulers.
Killing is opt-in (--kill), because the line was fitted on one input pipeline
and must be re-validated before it is trusted on another (e.g. stride 1).

Usage:
    PYTHONPATH=. python scripts/screen_epoch5.py \
        --runs output/e10_stride1_seed0 output/e10_stride1_seed1 output/e10_stride1_seed2 \
        --rescore-flags "--unisign-preprocess --block-padding-mask"          # report only
    ... --kill --kill-below 550                                           # also stop losers
"""
import argparse
import collections
import json
import os
import shlex
import signal
import subprocess
import sys
import time


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_gpu(need_mib):
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    best = None
    for line in out.strip().splitlines():
        i, f = [int(x) for x in line.split(',')]
        if f >= need_mib and (best is None or f > best[1]):
            best = (i, f)
    return None if best is None else best[0]


def trainer_pid(run_dir):
    """PID of the python trainer writing into run_dir (matched on --save-dir)."""
    want = os.path.abspath(run_dir)
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            argv = open(f'/proc/{pid}/cmdline', 'rb').read().split(b'\0')
        except OSError:
            continue
        argv = [a.decode(errors='ignore') for a in argv if a]
        if not any('train_encoder_mt5.py' in a for a in argv) or '--save-dir' not in argv:
            continue
        d = argv[argv.index('--save-dir') + 1]
        if os.path.abspath(os.path.join(os.readlink(f'/proc/{pid}/cwd'), d)) == want:
            # DataLoader workers share the cmdline; the parent is the lowest PID.
            return int(pid)
    return None


def score(run_dir, ckpt, args):
    out = os.path.join(run_dir, f'screen_ep{args.epoch}')
    gpu = None
    while gpu is None:
        gpu = free_gpu(args.need_mib)
        if gpu is None:
            log(f'  waiting for a GPU with {args.need_mib} MiB free to score {ckpt}')
            time.sleep(120)
    root = os.path.expanduser(args.root)
    cmd = ([sys.executable, 'scripts/rescore_checkpoints.py', '--root', root, '--out', out,
            '--ckpt', f'{ckpt}=ep{args.epoch}:{root}'] + shlex.split(args.rescore_flags))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH='.')
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        log(f'  RESCORE FAILED for {ckpt}: {r.stderr.strip().splitlines()[-1:]}')
        return None
    hyps = [json.loads(l)['hyp'] for l in open(os.path.join(out, f'ep{args.epoch}.predictions.jsonl'))]
    c = collections.Counter(hyps)
    m = json.load(open(os.path.join(out, f'ep{args.epoch}.metrics.json')))['clean']['all']
    return {'distinct': len(c), 'n': len(hyps),
            'top1_share': c.most_common(1)[0][1] / len(hyps),
            'chrf': m['chrf'], 'content_recall': m['content_recall'], 'gpu': gpu}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--epoch', type=int, default=5)
    ap.add_argument('--root', default='~/asan_canonical')
    ap.add_argument('--rescore-flags', default='--unisign-preprocess --block-padding-mask')
    ap.add_argument('--kill', action='store_true')
    ap.add_argument('--kill-below', type=int, default=500)
    ap.add_argument('--need-mib', type=int, default=25000)
    ap.add_argument('--poll', type=int, default=120)
    a = ap.parse_args()

    pending = list(a.runs)
    log(f'screening {len(pending)} runs at epoch {a.epoch}; '
        f'{"KILL below " + str(a.kill_below) if a.kill else "report only"}')
    if a.kill:
        log('WARNING: this is a probability, not a rule. Two cut-offs have been broken '
            'out of sample (E9b_s3 took off from 547, E11_s2 from 419); below 500 about '
            '1 run in 27 has still taken off. --kill trades those runs for GPU time.')
    while pending:
        for run in list(pending):
            ck = os.path.join(run, f'phase1_mt5_epoch{a.epoch}.pth')
            done_marker = os.path.join(run, f'screen_ep{a.epoch}.json')
            if os.path.exists(done_marker):
                pending.remove(run)
                continue
            if not os.path.exists(ck):
                continue   # not started, or not at this epoch yet
            if time.time() - os.path.getmtime(ck) < 90:
                continue   # still being written
            log(f'{run}: epoch-{a.epoch} checkpoint present, scoring')
            res = score(run, ck, a)
            if res is None:
                continue
            below = res['distinct'] < a.kill_below
            dct = res['distinct']
            res['verdict'] = ('under 500: historically 1 of 27 runs took off (~4%)'
                              if dct < 500 else
                              '500-576: historically 1 of 3 took off -- let it finish'
                              if dct < 577 else
                              '577+: historically 5 of 5 took off')
            res['killed'] = False
            if below and a.kill:
                pid = trainer_pid(run)
                if pid:
                    os.kill(pid, signal.SIGTERM)
                    res['killed'] = True
            json.dump(res, open(done_marker, 'w'), indent=1)
            log(f'{run}: distinct {res["distinct"]}/{res["n"]}, top-1 '
                f'{res["top1_share"]:.1%}, chrF {res["chrf"]:.2f}, recall '
                f'{res["content_recall"]:.3f} -> {res["verdict"]}'
                + ('  [KILLED]' if res['killed'] else ''))
            pending.remove(run)
        if pending:
            time.sleep(a.poll)
    log('screen DONE')


if __name__ == '__main__':
    main()
