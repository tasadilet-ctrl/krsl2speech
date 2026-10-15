#!/usr/bin/env bash
# Score E9's epoch-5 checkpoints on the validated early signal (distinct-output
# count; below ~550 no run in 19 has recovered). Waits for both to be written.
set -uo pipefail
cd ~/krsl2speech
export ASAN_ROOT=$HOME/asan_canonical PYTHONPATH=. PYTHONUNBUFFERED=1
log(){ echo "[$(date +%H:%M:%S)] $*"; }
for i in $(seq 1 90); do
  n=0
  for s in 0 1; do [ -f output/e9_fps25_seed$s/phase1_mt5_epoch5.pth ] && n=$((n+1)); done
  [ "$n" -ge 2 ] && break
  sleep 60
done
log "epoch-5 checkpoints present: $(ls output/e9_fps25_seed*/phase1_mt5_epoch5.pth 2>/dev/null | wc -l)"
sleep 60   # let the final write flush
C=$HOME/asan_canonical
CK=()
for s in 0 1; do
  f=output/e9_fps25_seed$s/phase1_mt5_epoch5.pth
  [ -f "$f" ] && CK+=(--ckpt "$f=E9_s${s}_ep5:$C")
done
[ ${#CK[@]} -eq 0 ] && { log "no checkpoints"; exit 1; }
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F', ' '$2>=25000{print $1}' | head -1)
[ -z "$gpu" ] && { log "no GPU with 25GB free"; exit 1; }
log "scoring on GPU $gpu"
rm -rf output/e9_ep5
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/rescore_checkpoints.py \
  --root "$C" --unisign-preprocess --block-padding-mask --target-fps 25 \
  --out output/e9_ep5 "${CK[@]}"
log "distinct-output counts (takeoff band 577-591; below ~550 has never recovered)"
python3 - <<'PY'
import json, glob, collections, os
for f in sorted(glob.glob('output/e9_ep5/*.predictions.jsonl')):
    h=[json.loads(l)['hyp'] for l in open(f)]
    c=collections.Counter(h)
    m=json.load(open(f.replace('.predictions.jsonl','.metrics.json')))['clean']['all']
    print(f"  {os.path.basename(f).split('.')[0]:<12} distinct={len(c):>3}/{len(h)}  "
          f"top1={100*c.most_common(1)[0][1]/len(h):5.1f}%  chrF={m['chrf']:5.2f}  "
          f"recall={m['content_recall']:.3f}")
PY
log "DONE"
