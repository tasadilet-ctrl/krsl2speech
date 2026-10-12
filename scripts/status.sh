#!/usr/bin/env bash
# One-shot status of krsl2speech experiments on this box.
# Usage: bash ~/krsl2speech/scripts/status.sh [experiment]   e.g. e11
cd ~/krsl2speech
EXP=${1:-$(ls -t output/*_scheduler.log 2>/dev/null | head -1 | sed 's|output/||;s|_scheduler.log||')}
echo "=== GPUs ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo; echo "=== my training / scoring jobs ==="
ps -u "$USER" -o pid,etime,args= | awk '/train_min|train_encoder_mt5|rescore_check|eval_retrieval/ && !/awk/ {print $1, $2, $4, $5, $NF}' | head
echo; echo "=== $EXP: scheduler (last lines) ==="
tail -4 "output/${EXP}_scheduler.log" 2>/dev/null
echo; echo "=== $EXP: latest epoch per seed ==="
for d in $(ls -d output/${EXP}_*seed* 2>/dev/null); do
  printf '%-32s ' "$d"; grep -aE '^Epoch [0-9]+/' "$d/train.log" 2>/dev/null | tail -1 | cut -c1-70
  [ -z "$(grep -aE '^Epoch' "$d/train.log" 2>/dev/null)" ] && echo "(no epoch yet) $(grep -aE 'Batch' "$d/train.log" | tail -1)"
done
echo; echo "=== $EXP: epoch-5 screen ==="
grep -aE 'distinct' "output/screen_${EXP}.log" 2>/dev/null | sed 's/^\[[^]]*\] //' || echo "(none)"
echo; echo "=== $EXP: final summary ==="
cat "output/${EXP}_rescore/summary.txt" 2>/dev/null || echo "not finished yet"
