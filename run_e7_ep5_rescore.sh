#!/usr/bin/env bash
# Rescore the surviving epoch-5 checkpoints: E7 (killed at ep5) against all
# three B seeds at the SAME epoch. B seed 0 is the run that went on to take
# off at epoch 10, so it is the reference for "on the takeoff trajectory".
set -uo pipefail
cd "$(dirname "$0")"
C=$HOME/asan_canonical
PYTHONPATH=. .venv/bin/python scripts/rescore_checkpoints.py \
  --root $C --unisign-preprocess --block-padding-mask \
  --out output/e7_ep5_rescore \
  --ckpt output/e3_armB_upstream/phase1_mt5_epoch5.pth=B_s0_ep5:$C \
  --ckpt output/e3_armS1_upstream_seed1/phase1_mt5_epoch5.pth=B_s1_ep5:$C \
  --ckpt output/e3_armS2_upstream_seed2/phase1_mt5_epoch5.pth=B_s2_ep5:$C \
  --ckpt output/e7_adapted_seed0/phase1_mt5_epoch5.pth=E7_s0_ep5:$C \
  --ckpt output/e7_adapted_seed1/phase1_mt5_epoch5.pth=E7_s1_ep5:$C
