#!/usr/bin/env bash
# E5: pose-text contrastive alignment (weight 0.5) on the arm-B config, 3 seeds.
# Baseline = the same config without alignment (B seeds 0/1/2), whose takeoff
# rate was 1/3. Judge E5 by takeoff rate across seeds, not one run's chrF.
set -uo pipefail
cd ~/krsl2speech
export ASAN_ROOT=$HOME/asan_canonical PYTHONPATH=. PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CACHE=$HOME/asan_canonical/text_teacher_mt5base.npz
COMMON="--pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth \
--selection-manifest $HOME/asan_canonical/selection_manifest.json \
--block-padding-mask --epochs 10 --unisign-preprocess \
--align-cache $CACHE --align-weight 0.5"
NEED_MIB=36000
QUEUE=(0 1 2)
declare -A GPU_PID=()
log(){ echo "[$(date +%H:%M:%S)] $*"; }
busy(){ local p=${GPU_PID[$1]:-}; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

launch(){
  local seed=$1 gpu=$2 dir=output/e5_align_seed$1
  mkdir -p "$dir"; log "launch align seed $seed on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu setsid .venv/bin/python train/train_encoder_mt5.py \
      $COMMON --seed "$seed" --save-dir "$dir" > "$dir/train.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 60); do
    sleep 20
    kill -0 $pid 2>/dev/null || { log "seed $seed died during startup:"; tail -5 "$dir/train.log"; return 1; }
    grep -q "Batch " "$dir/train.log" && { log "seed $seed training (pid $pid)"; GPU_PID[$gpu]=$pid; return 0; }
  done
  log "seed $seed no first batch after 20 min -- killing"; kill $pid 2>/dev/null; return 1
}

declare -A TRIES=()
while [ ${#QUEUE[@]} -gt 0 ]; do
  seed=${QUEUE[0]}; started=0
  for gpu in 1 0; do
    busy $gpu && continue
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu | tr -d ' ')
    if [ "$free" -ge "$NEED_MIB" ]; then
      TRIES[$seed]=$(( ${TRIES[$seed]:-0} + 1 ))
      if launch $seed $gpu; then QUEUE=("${QUEUE[@]:1}"); started=1
      elif [ ${TRIES[$seed]} -ge 3 ]; then log "seed $seed failed 3x -- dropping"; QUEUE=("${QUEUE[@]:1}"); fi
      break
    fi
  done
  [ $started -eq 0 ] && sleep 300
done
log "all seeds launched; waiting"
while pgrep -f "train_encoder_mt5.py.*e5_align_seed" >/dev/null; do sleep 600; done
log "E5 DONE"
for s in 0 1 2; do echo "== seed $s"; grep -E "^Epoch" output/e5_align_seed$s/train.log | tail -10; done
