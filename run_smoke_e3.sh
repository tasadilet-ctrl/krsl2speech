#!/usr/bin/env bash
set -uo pipefail
cd ~/krsl2speech
export ASAN_ROOT=$HOME/asan_smoke PYTHONPATH=. PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=$HOME/asan_canonical/score_quantiles.json
COMMON="--pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth --selection-manifest $HOME/asan_smoke/selection_manifest.json --block-padding-mask --epochs 1 --seed 0"
declare -A ARMS=(
  [A]="--use-enriched --real-wrists --score-quantiles $Q"
  [B]="--unisign-preprocess"
  [C]="--use-enriched --signspace --real-wrists --score-quantiles $Q"
)
for arm in A B C; do
  echo "===== SMOKE ARM $arm ====="
  .venv/bin/python train/train_encoder_mt5.py $COMMON ${ARMS[$arm]} --save-dir /tmp/e3_smoke_$arm 2>&1 \
    | grep -E "Traceback|Error|error:|Unisign|Uni-Sign|loaded|skipped|Total:|Selection|Epoch 1/1|Input dim|WER|Saved|saved|best" | grep -v "Warning"
  echo "exit=${PIPESTATUS[0]}  ckpt: $(ls /tmp/e3_smoke_$arm 2>/dev/null | tr "\n" " ")"
done
