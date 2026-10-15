#!/usr/bin/env bash
# E9b: baseline B config + --target-fps 30. E9 vs B isolates frame-rate
# normalisation. See EXPERIMENT_e9_framerate.md for the pre-registered readings.
set -uo pipefail
cd ~/krsl2speech
export ASAN_ROOT=$HOME/asan_canonical PYTHONPATH=. PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth \
--selection-manifest $HOME/asan_canonical/selection_manifest.json \
--block-padding-mask --epochs 10 --unisign-preprocess --target-fps 30"
NEED_MIB=36000
SEEDS=(3 4 5)
QUEUE=("${SEEDS[@]}")
declare -A GPU_PID=()
log(){ echo "[$(date +%H:%M:%S)] $*"; }
busy(){ local p=${GPU_PID[$1]:-}; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

launch(){
  local seed=$1 gpu=$2 dir=output/e9_fps30_seed$1
  mkdir -p "$dir"; log "launch fps30 seed $seed on GPU $gpu"
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
  for gpu in 0 1; do
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
while pgrep -f "train_encoder_mt5.py.*e9_fps30_seed" >/dev/null; do sleep 600; done
log "E9 training DONE"
for s in "${SEEDS[@]}"; do
  d=output/e9_fps30_seed$s
  n=$(grep -cE "^Epoch [0-9]+/10" "$d/train.log" 2>/dev/null || echo 0)
  echo "== seed $s ($n/10 epochs)"; grep -E "^Epoch" "$d/train.log" 2>/dev/null | tail -3
  [ "$n" -lt 10 ] && log "WARNING: seed $s ended at $n/10 -- incomplete, excluded"
done

C=$HOME/asan_canonical
# E9 checkpoints MUST be scored with --target-fps 30 and B WITHOUT it: the
# rescorer enforces target_fps exactly, so they cannot go in one invocation.
log "rescoring E9 (25 fps)"
CK=()
for s in "${SEEDS[@]}"; do
  f=output/e9_fps30_seed$s/phase1_mt5_epoch10.pth
  [ -f "$f" ] && CK+=(--ckpt "$f=E9b_s$s:$C") || log "missing: $f"
done
[ ${#CK[@]} -eq 0 ] && { log "FATAL: no E9 checkpoints"; exit 1; }
gpu=""
while [ -z "$gpu" ]; do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', ' '$2>=25000{print $1}' | sort -r | head -1)
  [ -z "$gpu" ] && { log "waiting for a GPU with 25 GB free"; sleep 300; }
done
rm -rf output/e9b_rescore_345 output/e9b_rescore_345_shuf
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/rescore_checkpoints.py \
  --root "$C" --unisign-preprocess --block-padding-mask --target-fps 30 \
  --out output/e9b_rescore_345 "${CK[@]}" > output/e9b_rescore_345.log 2>&1 \
  || { log "RESCORE FAILED"; tail -20 output/e9b_rescore_345.log; exit 1; }

log "rescoring E9 pose-shuffled (coupling delta)"
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/rescore_checkpoints.py \
  --root "$C" --unisign-preprocess --block-padding-mask --target-fps 30 \
  --shuffle-pose --out output/e9b_rescore_345_shuf "${CK[@]}" \
  > output/e9b_rescore_345_shuf.log 2>&1 \
  || { log "SHUFFLE RESCORE FAILED"; tail -20 output/e9b_rescore_345_shuf.log; exit 1; }

python3 scripts/summarize_takeoff.py --dir output/e9b_rescore_345 | tee output/e9b_rescore_345/summary.txt
log "E9 DONE -- output/e9b_rescore_345/summary.txt (B's numbers are in output/all_rescore)"
