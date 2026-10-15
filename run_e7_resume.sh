#!/usr/bin/env bash
# Resume E7 seed 0 from its epoch-5 checkpoint to epoch 10, then rescore
# against the three B seeds and apply the takeoff rule.
#
# The checkpoint carries optimizer state (3 param groups, lr 2.847e-05 /
# 2.847e-04 = the epoch-5 logged values) and global_step 9220, so the LR
# schedule continues mid-decay rather than restarting from warmup. Flags below
# replicate run_e7.sh exactly, minus --pretrained-unisign, which --resume
# overrides anyway.
set -uo pipefail
cd ~/krsl2speech
export ASAN_ROOT=$HOME/asan_canonical PYTHONPATH=. PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log(){ echo "[$(date +%H:%M:%S)] $*"; }
D=output/e7_adapted_seed0
C=$HOME/asan_canonical

log "resuming E7 seed 0 from epoch 5"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train/train_encoder_mt5.py \
  --selection-manifest $C/selection_manifest.json \
  --block-padding-mask --unisign-preprocess --epochs 10 \
  --mt5-path $HOME/mt5_kaznews --seed 0 --save-dir "$D" \
  --resume "$D/phase1_mt5_epoch5.pth" >> "$D/train.log" 2>&1
rc=$?
n=$(grep -cE "^Epoch [0-9]+/10" "$D/train.log" 2>/dev/null || echo 0)
log "training exited rc=$rc, $n epoch lines in log"
if [ ! -f "$D/phase1_mt5_epoch10.pth" ]; then
  log "FATAL: no epoch-10 checkpoint -- run did not finish"; tail -20 "$D/train.log"; exit 1
fi

log "rescoring"
rm -rf output/e7_resume_rescore
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/rescore_checkpoints.py \
  --root "$C" --unisign-preprocess --block-padding-mask --out output/e7_resume_rescore \
  --ckpt output/e3_armB_upstream/phase1_mt5_epoch10.pth=B_s0:$C \
  --ckpt output/e3_armS1_upstream_seed1/phase1_mt5_epoch10.pth=B_s1:$C \
  --ckpt output/e3_armS2_upstream_seed2/phase1_mt5_epoch10.pth=B_s2:$C \
  --ckpt "$D/phase1_mt5_epoch10.pth=E7_s0:$C" > output/e7_resume_rescore.log 2>&1 \
  || { log "RESCORE FAILED"; tail -20 output/e7_resume_rescore.log; exit 1; }
python3 scripts/summarize_takeoff.py --dir output/e7_resume_rescore | tee output/e7_resume_rescore/summary.txt
log "DONE -- output/e7_resume_rescore/summary.txt"
