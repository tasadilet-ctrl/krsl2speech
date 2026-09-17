#!/usr/bin/env bash
# E1: re-score checkpoints on the canonical selection set, then sweep decoding.
# Inference only -- no training. Log: output/e1/e1.log
set -uo pipefail
cd ~/krsl2speech
A=/data/archive/asan-dataset
C=$HOME/asan_clean
export ASAN_ROOT=$HOME/asan_canonical PYTHONPATH=. PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
.venv/bin/python scripts/rescore_checkpoints.py \
  --root "$ASAN_ROOT" --use-enriched --out output/e1 --batch-size 16 \
  --ckpt output/ours_enriched_friend_mt5.pth=init:$A \
  --ckpt output/clean_treat/phase1_mt5_epoch5.pth=treat_e5:$A,$C \
  --ckpt output/clean_treat/phase1_mt5_best.pth=treat_best:$A,$C \
  --ckpt output/clean_treat/phase1_mt5_epoch10.pth=treat_e10:$A,$C \
  --sweep-on auto 2>&1 | grep --line-buffered -vE "Warning|warnings.warn|Loading weights|tie_word_embeddings"
echo "[exit $?] $(date)"
