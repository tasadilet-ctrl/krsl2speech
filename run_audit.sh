#!/usr/bin/env bash
# Corpus audit: within-broadcast PMI scores for every dev/test clip, B_s0.
cd ~/krsl2speech
export ASAN_ROOT=$HOME/asan_canonical PYTHONPATH=. PYTHONUNBUFFERED=1
C=$HOME/asan_canonical
for sp in dev test; do
  split=$([ $sp = dev ] && echo val || echo test)
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval_retrieval.py --root $C \
    --manifest $C/audit_$sp.json --split $split --pool video --unisign-preprocess \
    --block-padding-mask --chunk 20 --out output/audit_$sp \
    --ckpt output/e3_armB_upstream/phase1_mt5_epoch10.pth=B_s0
done
echo AUDIT-SCORING-DONE
