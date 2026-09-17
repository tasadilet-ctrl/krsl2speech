#!/usr/bin/env bash
# One-screen progress check, sized for a phone.
echo "== GPU =="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "== scheduler =="; tail -4 ~/krsl2speech/output/e3_scheduler.log 2>/dev/null
for d in ~/krsl2speech/output/e3_arm*; do
  [ -f "$d/train.log" ] || continue
  echo "== $(basename $d) =="
  grep -E "^Epoch" "$d/train.log" | tail -3 | cut -c1-110
  tail -1 "$d/train.log" | cut -c1-80
done
