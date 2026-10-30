#!/usr/bin/env bash
# E17: text-adapted mT5 (on rewritten refs) as decoder init, E16 config otherwise.
# EXPERIMENT_e17_adapted_decoder.md. Starts only after E16b has finished.
set -uo pipefail
cd ~/krsl2speech
export PYTHONPATH=. PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$HOME/asan_canonical_llmref
C=$HOME/asan_canonical
M=$HOME/mt5_kaznews_llmref
log(){ echo "[$(date +%H:%M:%S)] $*"; }

until grep -q "E16b DONE\|FATAL\|FAILED" output/e16b_scheduler.log 2>/dev/null; do sleep 300; done
log "E16b finished: $(tail -1 output/e16b_scheduler.log)"

# ---- 1. adapt ----
if [ ! -f "$M/adaptation.json" ]; then
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/adapt_mt5_text.py --root "$L" --out "$M" \
    > output/e17_adapt.log 2>&1 || { log "ADAPT FAILED"; tail -20 output/e17_adapt.log; exit 1; }
fi
log "adapted: $(cat $M/adaptation.json | tr -d '\n ')"

# ---- 2. train: seeds 0,1 in parallel, then 2 ----
train(){ # seed gpu
  local d=output/e17_seed$1; mkdir -p $d
  ASAN_ROOT=$L CUDA_VISIBLE_DEVICES=$2 .venv/bin/python train/train_min.py \
    --pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth --epochs 10 \
    --mt5-path "$M" --seed $1 --save-dir $d > $d/train.log 2>&1 || log "seed $1 FAILED"
}
score(){ # seed: rescore on GPU 0, T2 on GPU 1
  local s=$1 f=output/e17_seed$1/phase1_mt5_epoch10.pth
  [ -f "$f" ] || { log "missing $f"; return; }
  ( CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/rescore_checkpoints.py --root $C --unisign-preprocess \
      --block-padding-mask --out output/e17s${s}_rescore --ckpt "$f=E17_s$s:$C" > output/e17s${s}_rescore.log 2>&1
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/rescore_checkpoints.py --root $C --unisign-preprocess \
      --block-padding-mask --shuffle-pose --out output/e17s${s}_rescore_shuf --ckpt "$f=E17_s$s:$C" \
      > output/e17s${s}_rescore_shuf.log 2>&1 ) &
  ( for r in "$C:e17s${s}_t2" "$HOME/asan_e15_devllm:e17s${s}_t2_llmdev"; do
      CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/eval_retrieval.py --root "${r%%:*}" \
        --manifest $C/within_video_dev.json --split val --pool video --unisign-preprocess \
        --block-padding-mask --chunk 20 --out output/${r##*:} --ckpt "$f=E17_s$s" > output/${r##*:}.log 2>&1
    done ) &
  wait
  log "seed $s scored"
}
train 0 0 & P0=$!
train 1 1 & P1=$!
wait $P0; log "seed 0 trained"
train 2 0 & P2=$!          # seed 2 takes GPU 0 as soon as seed 0 is done
wait $P1; log "seed 1 trained"
# score seeds 0/1 while seed 2 trains: rescore shares GPU 0 with training, T2 uses GPU 1
score 0; score 1
wait $P2; log "seed 2 trained"
score 2

# ---- 3. summaries ----
for s in 0 1 2; do
  [ -d output/e17s${s}_rescore ] && .venv/bin/python scripts/e16_selection_summary.py \
    output/e17s${s}_rescore output/e17s${s}_rescore_shuf
done | tee output/e17_summary.txt
P=(); for s in 0 1 2; do f=output/e17s${s}_rescore/E17_s$s.predictions.jsonl; [ -f $f ] && P+=("E17_s$s=$f"); done
P+=("E16_s0=output/e16early_rescore/E16_s0.predictions.jsonl" "E16_s2=output/e16s2_rescore/E16_s2.predictions.jsonl")
for s in 3 4 5; do f=output/e16b_rescore/E16_s$s.predictions.jsonl; [ -f $f ] && P+=("E16_s$s=$f"); done
.venv/bin/python scripts/word_band_recall.py --out output/e17_word_bands.json "${P[@]}" | tee -a output/e17_summary.txt
grep -h "T2 " output/e17s*_t2*.log | tee -a output/e17_summary.txt
log "E17 DONE"
