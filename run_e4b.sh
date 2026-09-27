#!/usr/bin/env bash
# E4b: encoder-rate screen. Encoder 1e-4 with the decoder UNCHANGED at 5e-4
# (ratio 5, not 0.5). E4 confounded a slower decoder with less decoder training.
# Judge by takeoff rate over the three seeds. See EXPERIMENT_e4_lr.md.
set -uo pipefail
cd ~/krsl2speech
export ASAN_ROOT=$HOME/asan_canonical PYTHONPATH=. PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth \
--selection-manifest $HOME/asan_canonical/selection_manifest.json \
--block-padding-mask --epochs 10 --unisign-preprocess \
--encoder-lr 1e-4 --mt5-lr 5e-4"
NEED_MIB=36000
# One list, used by the queue, the end-of-run summary and the rescore, so they
# cannot drift apart: the summary used to read seeds 0 1 2 while the queue ran
# 3 4 5, and so reported the previous run's seeds.
SEEDS=(3 4 5)
QUEUE=("${SEEDS[@]}")
declare -A GPU_PID=()
log(){ echo "[$(date +%H:%M:%S)] $*"; }
busy(){ local p=${GPU_PID[$1]:-}; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

launch(){
  local seed=$1 gpu=$2 dir=output/e4b_encfast_seed$1
  mkdir -p "$dir"; log "launch encfast seed $seed on GPU $gpu"
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
while pgrep -f "train_encoder_mt5.py.*e4b_encfast_seed" >/dev/null; do sleep 600; done
log "E4b training DONE"
for s in "${SEEDS[@]}"; do
  d=output/e4b_encfast_seed$s
  n=$(grep -cE "^Epoch [0-9]+/10" "$d/train.log" 2>/dev/null || echo 0)
  echo "== seed $s ($n/10 epochs)"
  grep -E "^Epoch" "$d/train.log" 2>/dev/null | tail -10
  # Seeds 3 and 4 of the first attempt stopped at epochs 3 and 2 with nothing
  # in the log and no scheduler message. Say so here rather than leaving an
  # incomplete run to look finished.
  [ "$n" -lt 10 ] && log "WARNING: seed $s ended at $n/10 epochs -- incomplete, excluded from the rescore"
done

# ---- rescore and apply the takeoff rule, on the box ---------------------
# run_e6.sh does this; this script did not, so its numbers had to be produced
# by a manual rescore, which was interrupted part-way and left a partial dir.
C=$HOME/asan_canonical
CK=()
add(){ [ -f "$1" ] && CK+=(--ckpt "$1=$2:$C") || log "missing checkpoint, skipped: $1"; }
add output/e3_armB_upstream/phase1_mt5_epoch10.pth        B_s0
add output/e3_armS1_upstream_seed1/phase1_mt5_epoch10.pth B_s1
add output/e3_armS2_upstream_seed2/phase1_mt5_epoch10.pth B_s2
for s in 0 1 2 "${SEEDS[@]}"; do
  f=output/e4b_encfast_seed$s/phase1_mt5_epoch10.pth
  [ -f "$f" ] && add "$f" E4b_s$s
done

gpu=""
while [ -z "$gpu" ]; do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', ' '$2>=25000{print $1}' | sort -r | head -1)
  [ -z "$gpu" ] && { log "rescore: waiting for a GPU with 25 GB free"; sleep 300; }
done
log "rescoring on GPU $gpu"
rm -rf output/e4b_rescore
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/rescore_checkpoints.py \
  --root "$C" --unisign-preprocess --block-padding-mask --out output/e4b_rescore \
  "${CK[@]}" > output/e4b_rescore.log 2>&1 \
  || { log "RESCORE FAILED -- see output/e4b_rescore.log"; tail -20 output/e4b_rescore.log; exit 1; }
python3 scripts/summarize_takeoff.py --dir output/e4b_rescore | tee output/e4b_rescore/summary.txt
log "E4b DONE -- output/e4b_rescore/summary.txt"
