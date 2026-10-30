#!/usr/bin/env bash
# E15: LLM-rewritten qazaqstantv references. EXPERIMENT_e15_llm_references.md
set -uo pipefail
cd ~/krsl2speech
export PYTHONPATH=. PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Lives in a colleague's home directory, which differs per box and is not
# this repo's to publish. Export it before running, e.g.
#   export QAZ_RECOLLECT=<their-home>/qazaqstantv_recollect/qazaqstantv_kz_fixed.jsonl
: "${QAZ_RECOLLECT:?set QAZ_RECOLLECT to the qazaqstantv recollection jsonl}"
C=$HOME/asan_canonical
B=output/e3_armB_upstream/phase1_mt5_epoch10.pth
VENV=~/skillopt_work/vllm_env
O=output/e15
log(){ echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p $O

# ---- step 1: rewrite dev + train (resumable), one shard per GPU ----
for i in 0 1; do
  PATH=$VENV/bin:$PATH CUDA_VISIBLE_DEVICES=$i HF_HUB_OFFLINE=1 $VENV/bin/python scripts/llm_realign_refs.py \
    --splits dev train --shard $i/2 --out $O/raw_$i.jsonl > $O/raw_$i.log 2>&1 &
done
wait
log "rewrite done: $(cat $O/raw_*.jsonl | wc -l) rows"
for s in dev train; do
  .venv/bin/python scripts/realign_guard.py $O/raw_*.jsonl --split $s --out $O/overrides_$s.json > $O/guard_$s.txt
  head -4 $O/guard_$s.txt
done
.venv/bin/python - <<'EOF'
import json, os
rec = {}
for l in open(os.environ['QAZ_RECOLLECT']):
    d = json.loads(l); rec[d['clip_id']] = (d.get('text_asr') or '').strip()
dev = json.load(open(os.path.expanduser('~/asan_canonical/qazaqstantv/annotations/kz/dev.json')))
json.dump({e['clip_id']: rec[e['clip_id']] for e in dev if rec.get(e['clip_id'])},
          open('output/e15/overrides_dev_asr.json', 'w'), ensure_ascii=False)
EOF
.venv/bin/python scripts/build_text_override_root.py --overrides $O/overrides_dev.json --splits dev --out ~/asan_e15_devllm
.venv/bin/python scripts/build_text_override_root.py --overrides $O/overrides_dev_asr.json --splits dev --out ~/asan_e15_devasr
.venv/bin/python scripts/build_text_override_root.py --overrides $O/overrides_train.json --splits train --out ~/asan_canonical_llmref

# ---- step 2: pre-check with B_s0 on three versions of the dev references ----
t2(){  # gpu root out ckpt-specs...
  local g=$1 r=$2 o=$3; shift 3
  CUDA_VISIBLE_DEVICES=$g .venv/bin/python scripts/eval_retrieval.py --root "$r" --manifest "$C/within_video_dev.json" \
    --split val --pool video --unisign-preprocess --block-padding-mask --chunk 20 --out "$o" "$@" > "$o.log" 2>&1
}
rm -rf $O/precheck; mkdir -p $O/precheck
( t2 0 $C $O/precheck/original --ckpt $B=B_s0; t2 0 ~/asan_e15_devasr $O/precheck/asr --ckpt $B=B_s0 ) &
t2 1 ~/asan_e15_devllm $O/precheck/llm --ckpt $B=B_s0 &
wait
.venv/bin/python scripts/e15_precheck.py | tee $O/precheck.txt
[ "${PIPESTATUS[0]}" = 3 ] && { log "E15 STOPPED at pre-check"; exit 0; }

# ---- step 3: fine-tune from B_s0 on rewritten train references (E14 recipe) ----
for s in 0 1; do
  d=output/e15_llm_s$s; mkdir -p $d
  ASAN_ROOT=$HOME/asan_canonical_llmref CUDA_VISIBLE_DEVICES=$s setsid .venv/bin/python train/train_min.py \
    --init-from $B --base-lr 1e-4 --epochs 3 --seed $s --save-dir $d > $d/train.log 2>&1 &
done
log "training llm_s0, llm_s1"
wait
log "E15 training DONE"

# ---- step 4: score ----
CK=(); CK2=(); CKALL=()
for s in 0 1; do
  f=output/e15_llm_s$s/phase1_mt5_epoch3.pth
  [ -f "$f" ] && { CK+=(--ckpt "$f=E15_llm_s$s:$C"); CK2+=(--ckpt "$f=E15_llm_s$s"); } || log "missing $f"
done
CKALL=("${CK2[@]}" --ckpt output/e14_full_a/phase1_mt5_epoch3.pth=E14_full_a \
       --ckpt output/e14_full_b/phase1_mt5_epoch3.pth=E14_full_b --ckpt $B=B_s0)
rm -rf output/e15_rescore output/e15_rescore_shuf output/e15_t2 output/e15_t2_llmdev
( CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/rescore_checkpoints.py --root "$C" --unisign-preprocess \
    --block-padding-mask --out output/e15_rescore "${CK[@]}" > output/e15_rescore.log 2>&1
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/rescore_checkpoints.py --root "$C" --unisign-preprocess \
    --block-padding-mask --shuffle-pose --out output/e15_rescore_shuf "${CK[@]}" > output/e15_rescore_shuf.log 2>&1 ) &
( t2 1 $C output/e15_t2 "${CK2[@]}"
  t2 1 ~/asan_e15_devllm output/e15_t2_llmdev "${CKALL[@]}" ) &
wait
grep -E "===|T2" output/e15_t2.log output/e15_t2_llmdev.log
log "E15 DONE"
