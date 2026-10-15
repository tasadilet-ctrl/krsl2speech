#!/usr/bin/env bash
# The e9 scheduler was stopped so seed 2 never launches; seeds 0 and 1 keep
# running. This waits for them, rescores clean + pose-shuffled (coupling delta),
# writes the summary, then launches E9b at 30 fps with three seeds.
#
# Why 30 instead of 25: 25 fps DOWNSAMPLES 91% of the corpus -- informburo loses
# half its frames, qazaqstantv a sixth -- and the epoch-5 read showed every
# source getting worse, with informburo (the 2x change) not improving at all.
# 30 fps leaves qazaqstantv (55% of clips) untouched, downsamples informburo by
# 5/3 instead of 2, and upsamples khabar only 25->30. This is the fallback the
# E9 readings pre-registered for a null.
set -uo pipefail
cd ~/krsl2speech
export ASAN_ROOT=$HOME/asan_canonical PYTHONPATH=. PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log(){ echo "[$(date +%H:%M:%S)] $*"; }
C=$HOME/asan_canonical

log "waiting for e9 seeds 0 and 1 to finish"
while pgrep -f "train_encoder_mt5.py.*e9_fps25_seed" >/dev/null; do sleep 300; done
log "e9 training done"
for s in 0 1; do
  d=output/e9_fps25_seed$s
  n=$(grep -acE "^Epoch [0-9]+/10" "$d/train.log" 2>/dev/null || echo 0)
  log "seed $s: $n/10 epochs"
done

CK=()
for s in 0 1; do
  f=output/e9_fps25_seed$s/phase1_mt5_epoch10.pth
  [ -f "$f" ] && CK+=(--ckpt "$f=E9_s$s:$C") || log "missing $f"
done
if [ ${#CK[@]} -gt 0 ]; then
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', ' '$2>=25000{print $1}' | head -1)
  log "rescoring E9 on GPU ${gpu:-0}"
  rm -rf output/e9_rescore output/e9_rescore_shuf
  CUDA_VISIBLE_DEVICES=${gpu:-0} .venv/bin/python scripts/rescore_checkpoints.py \
    --root "$C" --unisign-preprocess --block-padding-mask --target-fps 25 \
    --out output/e9_rescore "${CK[@]}" > output/e9_rescore.log 2>&1 \
    && CUDA_VISIBLE_DEVICES=${gpu:-0} .venv/bin/python scripts/rescore_checkpoints.py \
    --root "$C" --unisign-preprocess --block-padding-mask --target-fps 25 \
    --shuffle-pose --out output/e9_rescore_shuf "${CK[@]}" > output/e9_rescore_shuf.log 2>&1 \
    && python3 scripts/summarize_takeoff.py --dir output/e9_rescore | tee output/e9_rescore/summary.txt \
    || log "RESCORE FAILED -- see output/e9_rescore*.log"
else
  log "no E9 epoch-10 checkpoints to score"
fi

log "launching E9b at 30 fps, three seeds"
exec ./run_e9b.sh
