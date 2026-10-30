#!/usr/bin/env bash
# Speed-up for run_e16_test.sh's T1: B_s0's original-ref T1 is E12's (same code/manifest), so stop
# the re-run; on rewritten refs only qazaqstantv changes, so score B_s0 on those 200 clips alone.
set -uo pipefail
cd ~/krsl2speech
export PYTHONPATH=. PYTHONUNBUFFERED=1
C=$HOME/asan_canonical
kill 1422361
python3 - <<'PY'
import json, os
m = json.load(open(os.path.expanduser('~/asan_canonical/selection_manifest_test.json')))
q = [e for e in m if e['source'] == 'qazaqstantv']
json.dump(q, open('output/e16_test/selection_test_qazaq.json', 'w'))
print(len(q), 'qazaq test selection clips')
PY
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/eval_retrieval.py --root $HOME/asan_e16_testllm --split test \
  --manifest output/e16_test/selection_test_qazaq.json --pool source --unisign-preprocess --block-padding-mask \
  --chunk 10 --out output/e16_test/t1_llm_Bqz --ckpt output/e3_armB_upstream/phase1_mt5_epoch10.pth=B_s0 \
  > output/e16_test/t1_llm_Bqz.log 2>&1 &
# stop t1_llm once E16_s0 is done (its B_s0 pass is replaced by the qazaq-only run above)
until python3 -c "import json,sys; sys.exit(0 if 'E16_s0' in json.load(open('output/e16_test/t1_llm/summary.json')) else 1)" 2>/dev/null; do sleep 30; done
kill 1430990
wait
echo T1_FAST_DONE
