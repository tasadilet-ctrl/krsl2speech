#!/usr/bin/env bash
# E12 step 2: checkpoint selection on DEV only (T2 within-broadcast).
cd ~/krsl2speech
export ASAN_ROOT=$HOME/asan_canonical PYTHONPATH=. PYTHONUNBUFFERED=1
C=$HOME/asan_canonical
COMMON="--root $C --manifest $C/within_video_dev.json --split val --pool video --unisign-preprocess --block-padding-mask --chunk 10"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval_retrieval.py $COMMON --out output/e12_dev_t2 \
  --ckpt output/e3_armB_upstream/phase1_mt5_epoch10.pth=B_s0 \
  --ckpt output/e4b_encfast_seed1/phase1_mt5_epoch10.pth=E4b_s1
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval_retrieval.py $COMMON --target-fps 30 --out output/e12_dev_t2_fps30 \
  --ckpt output/e9_fps30_seed2/phase1_mt5_epoch10.pth=E9b_s2
echo E12-DEV-DONE
