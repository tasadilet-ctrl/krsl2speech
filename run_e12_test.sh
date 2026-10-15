#!/usr/bin/env bash
# E12 step 3: the TEST split, once, with the checkpoint selected on dev (B_s0 by
# the pre-registered tie rule). No choices are made after this runs.
cd ~/krsl2speech
export ASAN_ROOT=$HOME/asan_canonical PYTHONPATH=. PYTHONUNBUFFERED=1
C=$HOME/asan_canonical
CK="--ckpt output/e3_armB_upstream/phase1_mt5_epoch10.pth=B_s0"
BASE="--root $C --split test --unisign-preprocess --block-padding-mask --chunk 10"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval_retrieval.py $BASE \
  --manifest $C/within_video_test.json --pool video --out output/e12_test_t2 $CK
.venv/bin/python scripts/t2_error_analysis.py output/e12_test_t2/B_s0.within_video.json
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval_retrieval.py $BASE \
  --manifest $C/selection_manifest_test.json --pool source --out output/e12_test_t1 $CK
echo E12-TEST-DONE
