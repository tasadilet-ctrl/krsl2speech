#!/usr/bin/env bash
# E16 test: the TEST split, once, for E16_s0 (best on dev: chrF 35.57) with B_s0
# as the reference point (B_s0's test numbers were first reported in E12).
# Reporting only -- no choices are made after this runs.
# Test references are scored in two versions: original, and rewritten with the
# FROZEN E15 prompt/guards (rewrite done here; test was never rewritten before).
set -uo pipefail
cd ~/krsl2speech
export PYTHONPATH=. PYTHONUNBUFFERED=1
# Lives in a colleague's home directory, which differs per box and is not
# this repo's to publish. Export it before running, e.g.
#   export QAZ_RECOLLECT=<their-home>/qazaqstantv_recollect/qazaqstantv_kz_fixed.jsonl
: "${QAZ_RECOLLECT:?set QAZ_RECOLLECT to the qazaqstantv recollection jsonl}"
C=$HOME/asan_canonical
TL=$HOME/asan_e16_testllm
VENV=~/skillopt_work/vllm_env
O=output/e16_test
log(){ echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p $O
CK_R=(--ckpt output/e16_seed0/phase1_mt5_epoch10.pth=E16_s0:$C --ckpt output/e3_armB_upstream/phase1_mt5_epoch10.pth=B_s0:$C)
CK=(--ckpt output/e16_seed0/phase1_mt5_epoch10.pth=E16_s0 --ckpt output/e3_armB_upstream/phase1_mt5_epoch10.pth=B_s0)
gen(){ # gpu root out [extra]
  local g=$1 r=$2 o=$3; shift 3
  CUDA_VISIBLE_DEVICES=$g .venv/bin/python scripts/rescore_checkpoints.py --root "$r" --split test \
    --manifest $C/selection_manifest_test.json --unisign-preprocess --block-padding-mask \
    --out $O/$o "$@" "${CK_R[@]}" > $O/$o.log 2>&1 || log "FAILED $o"
}
ret(){ # gpu root out manifest pool
  CUDA_VISIBLE_DEVICES=$1 .venv/bin/python scripts/eval_retrieval.py --root "$2" --split test \
    --manifest $C/$4 --pool $5 --unisign-preprocess --block-padding-mask --chunk 10 \
    --out $O/$3 "${CK[@]}" > $O/$3.log 2>&1 || log "FAILED $3"
}

# GPU 1: original references
( gen 1 $C gen_orig
  gen 1 $C gen_orig_shuf --shuffle-pose
  ret 1 $C t2_orig within_video_test.json video
  ret 1 $C t1_orig selection_manifest_test.json source
  log "original-ref evals done" ) &

# GPU 0: rewrite test refs, then rewritten-reference evals
( PATH=$VENV/bin:$PATH CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $VENV/bin/python scripts/llm_realign_refs.py \
    --splits test --out output/e15/raw_test.jsonl > output/e15/raw_test.log 2>&1
  .venv/bin/python scripts/realign_guard.py output/e15/raw_test.jsonl --split test \
    --out output/e15/overrides_test.json > output/e15/guard_test.txt
  head -4 output/e15/guard_test.txt
  .venv/bin/python scripts/build_text_override_root.py --overrides output/e15/overrides_test.json \
    --splits test --out $TL
  gen 0 $TL gen_llm
  ret 0 $TL t2_llm within_video_test.json video
  ret 0 $TL t1_llm selection_manifest_test.json source
  log "rewritten-ref evals done" ) &
wait
log "E16 TEST DONE"
