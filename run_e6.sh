#!/usr/bin/env bash
# E6: token-level pose-text alignment (weight 0.5) on the arm-B config, 3 seeds.
# Identical to run_e5.sh except --align-mode token and the per-token cache.
# Pre-registration, takeoff rule and readings: EXPERIMENT_e6_token_alignment.md.
#
# Runs entirely on the box: trains the seeds (one per free GPU), then rescores
# all nine epoch-10 checkpoints (B, E5, E6) through one metric path and writes
# output/e6_rescore/summary.txt. Nothing depends on a session staying open.
set -uo pipefail
cd ~/krsl2speech
export ASAN_ROOT=$HOME/asan_canonical PYTHONPATH=. PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=$HOME/asan_canonical
CACHE=$C/text_teacher_mt5base_tokens.npz
COMMON="--pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth \
--selection-manifest $C/selection_manifest.json \
--block-padding-mask --epochs 10 --unisign-preprocess \
--align-mode token --align-cache $CACHE --align-weight 0.5"
NEED_MIB=38000   # smoke run peaked at 34.4 GB; E5 used 36000 for ~33 GB
QUEUE=(0 1 2)
declare -A GPU_PID=()
log(){ echo "[$(date +%H:%M:%S)] $*"; }
busy(){ local p=${GPU_PID[$1]:-}; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

launch(){
  local seed=$1 gpu=$2 dir=output/e6_tokalign_seed$1
  mkdir -p "$dir"; log "launch token-align seed $seed on GPU $gpu"
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
while pgrep -f "train_encoder_mt5.py.*e6_tokalign_seed" >/dev/null; do sleep 600; done
log "E6 training DONE"
for s in 0 1 2; do echo "== seed $s"; grep -E "^Epoch" output/e6_tokalign_seed$s/train.log | tail -10; done

# ---- rescore B, E5 and E6 together, same scorer, epoch 10 ----------------
CKPTS=()
add(){ [ -f "$1" ] && CKPTS+=(--ckpt "$1=$2:$C") || log "missing checkpoint, skipped: $1"; }
add output/e3_armB_upstream/phase1_mt5_epoch10.pth       B_s0
add output/e3_armS1_upstream_seed1/phase1_mt5_epoch10.pth B_s1
add output/e3_armS2_upstream_seed2/phase1_mt5_epoch10.pth B_s2
for s in 0 1 2; do add output/e5_align_seed$s/phase1_mt5_epoch10.pth E5_s$s; done
for s in 0 1 2; do add output/e6_tokalign_seed$s/phase1_mt5_epoch10.pth E6_s$s; done

gpu=""
while [ -z "$gpu" ]; do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', ' '$2>=25000{print $1}' | sort -r | head -1)
  [ -z "$gpu" ] && { log "rescore: waiting for a GPU with 25 GB free"; sleep 300; }
done
log "rescoring ${#CKPTS[@]} args on GPU $gpu"
rm -rf output/e6_rescore
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/rescore_checkpoints.py \
  --root "$C" --unisign-preprocess --block-padding-mask --out output/e6_rescore \
  "${CKPTS[@]}" > output/e6_rescore.log 2>&1 \
  || { log "RESCORE FAILED -- see output/e6_rescore.log"; tail -20 output/e6_rescore.log; exit 1; }
python3 scripts/summarize_takeoff.py --dir output/e6_rescore | tee output/e6_rescore/summary.txt
log "E6 DONE -- output/e6_rescore/summary.txt"
