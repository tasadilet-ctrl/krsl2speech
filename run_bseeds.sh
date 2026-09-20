#!/usr/bin/env bash
# Replicate seeds of arm B (seed 0 = arm B itself); identical config, only --seed differs
# Arms differ ONLY in spatial preprocessing; everything else is shared.
# Scheduler: start each arm when a GPU has >= NEED_MIB free; up to one arm per GPU.
set -uo pipefail
cd ~/krsl2speech
export ASAN_ROOT=$HOME/asan_canonical PYTHONPATH=. PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=$HOME/asan_canonical/score_quantiles.json
COMMON="--pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth --selection-manifest $HOME/asan_canonical/selection_manifest.json --block-padding-mask --epochs 10"
declare -A ARGS=(
  [S1]="--unisign-preprocess --seed 1"
  [S2]="--unisign-preprocess --seed 2"
)
declare -A NAME=( [S1]=upstream_seed1 [S2]=upstream_seed2 )
NEED_MIB=36000
QUEUE=(S1 S2)
declare -A GPU_PID=()
log(){ echo "[$(date +%H:%M:%S)] $*"; }

gpu_busy_by_us(){ local g=$1; local p=${GPU_PID[$g]:-}; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

launch(){
  local arm=$1 gpu=$2 attempt=$3
  local dir=output/e3_arm${arm}_${NAME[$arm]}; mkdir -p "$dir"
  log "launch arm $arm (${NAME[$arm]}) on GPU $gpu, attempt $attempt"
  CUDA_VISIBLE_DEVICES=$gpu setsid .venv/bin/python train/train_encoder_mt5.py $COMMON ${ARGS[$arm]} \
      --save-dir "$dir" > "$dir/train.log" 2>&1 &
  local pid=$!
  # Wait for the first training batch (or death) before launching anything else:
  # simultaneous CUDA inits on this box have stalled before.
  for i in $(seq 1 60); do
    sleep 20
    if ! kill -0 $pid 2>/dev/null; then log "arm $arm died during startup:"; tail -5 "$dir/train.log"; return 1; fi
    if grep -q "Batch " "$dir/train.log"; then log "arm $arm training (pid $pid)"; GPU_PID[$gpu]=$pid; return 0; fi
  done
  log "arm $arm no first batch after 20 min -- killing (suspected CUDA-init stall)"
  kill $pid 2>/dev/null; return 1
}

log "waiting for B20 to be confirmed training before competing for a GPU"
until grep -q "arm B20 training" output/b20_scheduler.log 2>/dev/null; do sleep 120; done
log "B20 is training; seeds now wait for a free GPU"
declare -A ATTEMPTS=()
while [ ${#QUEUE[@]} -gt 0 ]; do
  arm=${QUEUE[0]}
  started=0
  for gpu in 1 0; do
    gpu_busy_by_us $gpu && continue
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu | tr -d " ")
    if [ "$free" -ge "$NEED_MIB" ]; then
      ATTEMPTS[$arm]=$(( ${ATTEMPTS[$arm]:-0} + 1 ))
      if launch $arm $gpu ${ATTEMPTS[$arm]}; then
        QUEUE=("${QUEUE[@]:1}"); started=1
      elif [ ${ATTEMPTS[$arm]} -ge 3 ]; then
        log "arm $arm failed 3 times -- dropping from queue"; QUEUE=("${QUEUE[@]:1}")
      fi
      break
    fi
  done
  [ $started -eq 0 ] && sleep 300
done
log "all arms launched; waiting"
while pgrep -f "train_encoder_mt5.py.*e3_armS" >/dev/null; do sleep 600; done
log "E3 DONE"
for arm in S1 S2; do d=output/e3_arm${arm}_${NAME[$arm]}; echo "== $arm ${NAME[$arm]}"; grep -E "^Epoch" $d/train.log | tail -10; done
