#!/usr/bin/env bash
# Does top-1 share at epoch 5 predict takeoff at epoch 10?
# Scores the epoch-5 checkpoint of every arm whose epoch-10 verdict is known
# (B, E4, E4b, E5, E6 -- 18 runs, 4 of which took off). E7 is excluded from the
# predictor test: it was killed at epoch 5, so it has no epoch-10 label.
set -uo pipefail
cd "$(dirname "$0")"
C=$HOME/asan_canonical
A=()
for s in 0 1 2; do A+=(--ckpt output/e4_lr_seed$s/phase1_mt5_epoch5.pth=E4_s${s}_ep5:$C); done
for s in 0 1 2 3 4 5; do A+=(--ckpt output/e4b_encfast_seed$s/phase1_mt5_epoch5.pth=E4b_s${s}_ep5:$C); done
for s in 0 1 2; do A+=(--ckpt output/e5_align_seed$s/phase1_mt5_epoch5.pth=E5_s${s}_ep5:$C); done
for s in 0 1 2; do A+=(--ckpt output/e6_tokalign_seed$s/phase1_mt5_epoch5.pth=E6_s${s}_ep5:$C); done
PYTHONPATH=. .venv/bin/python scripts/rescore_checkpoints.py \
  --root $C --unisign-preprocess --block-padding-mask \
  --out output/ep5_predictor "${A[@]}"
