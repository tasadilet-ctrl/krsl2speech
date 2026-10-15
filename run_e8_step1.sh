#!/usr/bin/env bash
# E8 step 1: pose-shuffle sensitivity on every epoch-10 checkpoint with a known
# takeoff label. Clean scores already exist (output/all_rescore,
# output/e4b_rescore, output/e7_resume_rescore); this produces the shuffled half.
#   delta = chrF(clean) - chrF(shuffled)
set -uo pipefail
cd ~/krsl2speech
export ASAN_ROOT=$HOME/asan_canonical PYTHONPATH=. PYTHONUNBUFFERED=1
C=$HOME/asan_canonical
CK=()
add(){ [ -f "$1" ] && CK+=(--ckpt "$1=$2:$C") || echo "MISSING: $1"; }
add output/e3_armB_upstream/phase1_mt5_epoch10.pth        B_s0
add output/e3_armS1_upstream_seed1/phase1_mt5_epoch10.pth B_s1
add output/e3_armS2_upstream_seed2/phase1_mt5_epoch10.pth B_s2
for s in 0 1 2;       do add output/e4_lr_seed$s/phase1_mt5_epoch10.pth      E4_s$s;  done
for s in 0 1 2 3 4 5; do add output/e4b_encfast_seed$s/phase1_mt5_epoch10.pth E4b_s$s; done
for s in 0 1 2;       do add output/e5_align_seed$s/phase1_mt5_epoch10.pth    E5_s$s;  done
for s in 0 1 2;       do add output/e6_tokalign_seed$s/phase1_mt5_epoch10.pth E6_s$s;  done
add output/e7_adapted_seed0/phase1_mt5_epoch10.pth E7_s0

echo "scoring ${#CK[@]} checkpoints with --shuffle-pose"
rm -rf output/e8_shuffled
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/rescore_checkpoints.py \
  --root "$C" --unisign-preprocess --block-padding-mask --shuffle-pose \
  --out output/e8_shuffled "${CK[@]}"
