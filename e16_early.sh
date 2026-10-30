#!/usr/bin/env bash
# Score E16 seeds 0 and 1 while seed 2 trains (separate dirs; run_e16.sh still rescoring all three at the end).
set -uo pipefail
cd ~/krsl2speech
export PYTHONPATH=. PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1
C=$HOME/asan_canonical
CK=(); CK2=()
for s in 0 1; do f=output/e16_seed$s/phase1_mt5_epoch10.pth; CK+=(--ckpt "$f=E16_s$s:$C"); CK2+=(--ckpt "$f=E16_s$s"); done
.venv/bin/python scripts/rescore_checkpoints.py --root "$C" --unisign-preprocess --block-padding-mask \
  --out output/e16early_rescore "${CK[@]}" > output/e16early_rescore.log 2>&1
.venv/bin/python scripts/rescore_checkpoints.py --root "$C" --unisign-preprocess --block-padding-mask --shuffle-pose \
  --out output/e16early_rescore_shuf "${CK[@]}" > output/e16early_rescore_shuf.log 2>&1
echo RESCORE_DONE
for r in "$C:e16early_t2" "$HOME/asan_e15_devllm:e16early_t2_llmdev"; do
  .venv/bin/python scripts/eval_retrieval.py --root "${r%%:*}" --manifest "$C/within_video_dev.json" --split val \
    --pool video --unisign-preprocess --block-padding-mask --chunk 20 --out output/${r##*:} "${CK2[@]}" > output/${r##*:}.log 2>&1
done
echo EARLY_DONE
