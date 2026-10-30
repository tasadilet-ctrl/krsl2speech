#!/usr/bin/env bash
# Replaces run_e16.sh's tail: wait for seed 2, then score ONLY seed 2, split across both GPUs.
# Seeds 0/1 were scored by e16_early.sh into output/e16early_*.
set -uo pipefail
cd ~/krsl2speech
export PYTHONPATH=. PYTHONUNBUFFERED=1
C=$HOME/asan_canonical
log(){ echo "[$(date +%H:%M:%S)] $*"; }
f=output/e16_seed2/phase1_mt5_epoch10.pth
while pgrep -f "[t]rain_min.py.*e16_seed2" >/dev/null; do sleep 60; done
[ -f "$f" ] || { log "missing $f"; exit 1; }
while tmux has-session -t e16early 2>/dev/null; do sleep 30; done   # GPU 1 free of early T2
log "seed 2 done; scoring"
( CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/rescore_checkpoints.py --root "$C" --unisign-preprocess \
    --block-padding-mask --out output/e16s2_rescore --ckpt "$f=E16_s2:$C" > output/e16s2_rescore.log 2>&1
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/rescore_checkpoints.py --root "$C" --unisign-preprocess \
    --block-padding-mask --shuffle-pose --out output/e16s2_rescore_shuf --ckpt "$f=E16_s2:$C" > output/e16s2_rescore_shuf.log 2>&1 ) &
( for r in "$C:e16s2_t2" "$HOME/asan_e15_devllm:e16s2_t2_llmdev"; do
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/eval_retrieval.py --root "${r%%:*}" --manifest "$C/within_video_dev.json" \
      --split val --pool video --unisign-preprocess --block-padding-mask --chunk 20 --out output/${r##*:} \
      --ckpt "$f=E16_s2" > output/${r##*:}.log 2>&1
  done ) &
wait
log "E16 DONE"
