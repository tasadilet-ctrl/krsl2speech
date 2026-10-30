#!/usr/bin/env bash
# E16b (seeds 3-5): B config from scratch on LLM-rewritten qazaqstantv train refs. See EXPERIMENT_e16_scratch_llm_refs.md.
set -uo pipefail
cd ~/krsl2speech
export ASAN_ROOT=$HOME/asan_canonical_llmref PYTHONPATH=. PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth --epochs 10"
NEED_MIB=40000
SEEDS=(3 4 5)
QUEUE=("${SEEDS[@]}")
declare -A GPU_PID=()
log(){ echo "[$(date +%H:%M:%S)] $*"; }
busy(){ local p=${GPU_PID[$1]:-}; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

launch(){
  local seed=$1 gpu=$2 dir=output/e16_seed$1
  mkdir -p "$dir"; log "launch seed $seed on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu setsid .venv/bin/python train/train_min.py \
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
while pgrep -f "[t]rain_min.py.*e16_seed" >/dev/null; do sleep 600; done
log "E16b training DONE"

C=$HOME/asan_canonical
CK=()
for s in "${SEEDS[@]}"; do
  d=output/e16_seed$s
  n=$(grep -acE "^Epoch [0-9]+/10" "$d/train.log" 2>/dev/null || echo 0)
  log "seed $s: $n/10 epochs"
  f=$d/phase1_mt5_epoch10.pth
  [ -f "$f" ] && CK+=(--ckpt "$f=E16_s$s:$C") || log "missing $f -- excluded"
done
[ ${#CK[@]} -eq 0 ] && { log "FATAL: no E16 checkpoints"; exit 1; }
gpu=""
while [ -z "$gpu" ]; do
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', ' '$2>=25000{print $1}' | head -1)
  [ -z "$gpu" ] && { log "waiting for a GPU with 25 GB free"; sleep 300; }
done
# Stride and cap come from each checkpoint's run_args -- no flags needed.
rm -rf output/e16b_rescore output/e16b_rescore_shuf
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/rescore_checkpoints.py \
  --root "$C" --unisign-preprocess --block-padding-mask \
  --out output/e16b_rescore "${CK[@]}" > output/e16b_rescore.log 2>&1 \
  || { log "RESCORE FAILED"; tail -20 output/e16b_rescore.log; exit 1; }
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/rescore_checkpoints.py \
  --root "$C" --unisign-preprocess --block-padding-mask --shuffle-pose \
  --out output/e16b_rescore_shuf "${CK[@]}" > output/e16b_rescore_shuf.log 2>&1 \
  || { log "SHUFFLE RESCORE FAILED"; tail -20 output/e16b_rescore_shuf.log; exit 1; }
python3 scripts/summarize_takeoff.py --dir output/e16b_rescore | tee output/e16b_rescore/summary.txt
log "E16b rescore done -- output/e16b_rescore/summary.txt"
# T2 dev retrieval (primary measure) for every seed.
log "T2 dev retrieval"
CK2=()
for s in "${SEEDS[@]}"; do
  f=output/e16_seed$s/phase1_mt5_epoch10.pth; [ -f "$f" ] && CK2+=(--ckpt "$f=E16_s$s")
done
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/eval_retrieval.py --root "$C" \
  --manifest "$C/within_video_dev.json" --split val --pool video --unisign-preprocess \
  --block-padding-mask --chunk 20 --out output/e16b_t2 "${CK2[@]}" > output/e16b_t2.log 2>&1 \
  || { log "T2 FAILED"; tail -20 output/e16b_t2.log; exit 1; }
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/eval_retrieval.py --root "$HOME/asan_e15_devllm" \
  --manifest "$C/within_video_dev.json" --split val --pool video --unisign-preprocess \
  --block-padding-mask --chunk 20 --out output/e16b_t2_llmdev "${CK2[@]}" > output/e16b_t2_llmdev.log 2>&1 \
  || { log "T2 llmdev FAILED"; tail -20 output/e16b_t2_llmdev.log; exit 1; }
grep -E "===|T2" output/e16b_t2.log output/e16b_t2_llmdev.log
log "E16b DONE"
