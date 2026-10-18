#!/usr/bin/env bash
# E14: fine-tune B_s0 on cleaned / random-reduced / full data. EXPERIMENT_e14_finetune_clean.md
set -uo pipefail
cd ~/krsl2speech
export ASAN_ROOT=$HOME/asan_canonical PYTHONPATH=. PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=$HOME/asan_canonical
B=output/e3_armB_upstream/phase1_mt5_epoch10.pth
COMMON="--init-from $B --base-lr 1e-4 --epochs 3"
declare -A EXTRA=( [clean]="--seed 0 --exclude-clips $C/clean_exclude_qz_v1.json"
                   [random]="--seed 0 --exclude-clips $C/random_exclude_qz_v1.json"
                   [full_a]="--seed 0" [full_b]="--seed 1" )
QUEUE=(clean random full_a full_b)
declare -A GPU_PID=()
log(){ echo "[$(date +%H:%M:%S)] $*"; }
busy(){ local p=${GPU_PID[$1]:-}; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }
launch(){
  local arm=$1 gpu=$2 dir=output/e14_$1
  mkdir -p "$dir"; log "launch $arm on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu setsid .venv/bin/python train/train_min.py $COMMON ${EXTRA[$arm]} \
      --save-dir "$dir" > "$dir/train.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 60); do
    sleep 20
    kill -0 $pid 2>/dev/null || { log "$arm died during startup:"; tail -5 "$dir/train.log"; return 1; }
    grep -q "Batch " "$dir/train.log" && { log "$arm training (pid $pid)"; GPU_PID[$gpu]=$pid; return 0; }
  done
  kill $pid 2>/dev/null; return 1
}
while [ ${#QUEUE[@]} -gt 0 ]; do
  arm=${QUEUE[0]}; started=0
  for gpu in 0 1; do
    busy $gpu && continue
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu | tr -d ' ')
    if [ "$free" -ge 40000 ] && launch $arm $gpu; then QUEUE=("${QUEUE[@]:1}"); started=1; break; fi
  done
  [ $started -eq 0 ] && sleep 120
done
log "all arms launched; waiting"
while pgrep -f "[t]rain_min.py.*e14_" >/dev/null; do sleep 300; done
log "E14 training DONE"
CK=(); CK2=()
for arm in clean random full_a full_b; do
  f=output/e14_$arm/phase1_mt5_epoch3.pth
  [ -f "$f" ] && { CK+=(--ckpt "$f=E14_$arm:$C"); CK2+=(--ckpt "$f=E14_$arm"); } || log "missing $f"
done
rm -rf output/e14_rescore output/e14_rescore_shuf output/e14_t2
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/rescore_checkpoints.py --root "$C" --unisign-preprocess \
  --block-padding-mask --out output/e14_rescore "${CK[@]}" > output/e14_rescore.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/eval_retrieval.py --root "$C" --manifest "$C/within_video_dev.json" \
  --split val --pool video --unisign-preprocess --block-padding-mask --chunk 20 --out output/e14_t2 \
  "${CK2[@]}" > output/e14_t2.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/rescore_checkpoints.py --root "$C" --unisign-preprocess \
  --block-padding-mask --shuffle-pose --out output/e14_rescore_shuf "${CK[@]}" > output/e14_rescore_shuf.log 2>&1
grep -E "===|T2" output/e14_t2.log
log "E14 DONE"
