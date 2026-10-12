#!/usr/bin/env bash
# Parity: does train/train_min.py reproduce train/train_encoder_mt5.py?
#
# Both trainers run the B config for N batches in DETERMINISTIC mode (old:
# KRSL_DETERMINISTIC=1, new: --deterministic), logging the running-mean loss at
# every batch. Same seed + same construction order + deterministic kernels =>
# the curves must coincide exactly, not approximately.
#
# Why deterministic: without it the same seed does not reproduce a run. On
# 2026-10-08 two default-mode train_min runs with seed 0 reached 18.34 and 17.26
# at batch 400, and two old-trainer runs 17.56 and 17.44 -- a first parity test
# that used one old-vs-old pair as its noise floor wrongly declared a bug.
#
# Runs: old, new, new again (reproducibility), new with another seed (negative
# control: proves the comparison can see a difference at all).
#
# Usage: GPU=0 N=400 STRIDE=2 bash scripts/parity_train_min.sh
set -uo pipefail
cd ~/krsl2speech
export ASAN_ROOT=$HOME/asan_canonical PYTHONPATH=. PYTHONUNBUFFERED=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${GPU:-0}; N=${N:-400}; STRIDE=${STRIDE:-2}
OUT=/tmp/parity_det; rm -rf $OUT; mkdir -p $OUT
U=checkpoints/unisign/csl_stage1_weight.pth
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "old trainer (deterministic), seed 0, $N batches"
CUDA_VISIBLE_DEVICES=$GPU KRSL_DETERMINISTIC=1 KRSL_LOG_EVERY=1 .venv/bin/python \
  train/train_encoder_mt5.py --pretrained-unisign $U --block-padding-mask \
  --unisign-preprocess --epochs 10 --seed 0 --downsample-every $STRIDE \
  --save-dir $OUT/old > $OUT/old.log 2>&1 &
pid=$!
until grep -q "Batch $N/" $OUT/old.log 2>/dev/null || ! kill -0 $pid 2>/dev/null; do sleep 5; done
kill $pid 2>/dev/null; wait $pid 2>/dev/null

for run in min0a:0 min0b:0 min1:1; do
  name=${run%%:*}; seed=${run##*:}
  log "train_min (deterministic) $name, seed $seed, $N batches"
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python train/train_min.py --pretrained-unisign $U \
    --epochs 10 --seed $seed --downsample-every $STRIDE --save-dir $OUT/$name \
    --deterministic --log-every 1 --max-batches $((N + 1)) > $OUT/$name.log 2>&1
done

python3 - "$OUT" "$N" <<'PY'
import re, sys
out, N = sys.argv[1], int(sys.argv[2])
def curve(f):
    return {int(m.group(1)): float(m.group(2))
            for m in re.finditer(r'Batch (\d+)/\d+ \| Loss: ([0-9.]+)', open(f).read())}
r = {k: curve(f'{out}/{k}.log') for k in ('old', 'min0a', 'min0b', 'min1')}
for k, c in r.items():
    if len(c) < N:
        sys.exit(f'[fatal] {k} logged {len(c)}/{N} batches -- see {out}/{k}.log '
                 f'(a "deterministic implementation" error means an op lacks one)')
ks = range(1, N + 1)
def cmp(a, b):
    d = [abs(r[a][k] - r[b][k]) for k in ks]
    return max(d), sum(x == 0 for x in d)
for a, b, label in (('old', 'min0a', 'old vs train_min'),
                    ('min0a', 'min0b', 'train_min vs itself'),
                    ('old', 'min1', 'CONTROL: other seed')):
    mx, ex = cmp(a, b)
    print(f'{label:<22} max |diff| {mx:.4f}   exact at {ex}/{N} batches')
print(f'batch {N}: ' + '  '.join(f'{k} {r[k][N]:.4f}' for k in r))
same = cmp('old', 'min0a')[1] == N and cmp('min0a', 'min0b')[1] == N
ctl = cmp('old', 'min1')[0] > 0.05
print('PARITY PASS: identical to the logged precision at every batch, reproducible, '
      'and the control differs' if (same and ctl) else 'PARITY FAIL')
PY
