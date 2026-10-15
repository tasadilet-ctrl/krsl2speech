#!/usr/bin/env bash
# Full TRAIN-split audit with B_s0 (memorisation check showed no advantage on
# training clips and x18.8 enrichment, so cross-fitting is unnecessary).
cd ~/krsl2speech
export ASAN_ROOT=$HOME/asan_canonical PYTHONPATH=. PYTHONUNBUFFERED=1
C=$HOME/asan_canonical
python3 - <<'PY'
import collections, json, os, sys
sys.path.insert(0, 'scripts'); import build_canonical_splits as bcs
root = os.path.expanduser('~/asan_canonical')
groups = {'qz': ['qazaqstantv'], 'kh_ib': ['khabar', 'informburo']}
for name, srcs in groups.items():
    out = []
    for src in srcs:
        by_v = collections.defaultdict(list)
        for e in bcs.load(root, src, 'train'):
            if bcs._passes_dataset_filters(e): by_v[bcs.video_id(e)].append(e)
        for v, es in sorted(by_v.items()):
            if len(es) >= 2:
                for e in sorted(es, key=lambda e: e['clip_id']):
                    out.append({'clip_id': e['clip_id'], 'source': src, 'video_id': v, 'T': e.get('T', 0)})
    json.dump(out, open(f'{root}/audit_train_{name}.json', 'w'))
    print(name, len(out), dict(collections.Counter(m['source'] for m in out)), flush=True)
PY
run(){ CUDA_VISIBLE_DEVICES=$1 .venv/bin/python scripts/eval_retrieval.py --root $C \
  --manifest $C/audit_train_$2.json --split train --pool video --unisign-preprocess \
  --block-padding-mask --chunk 20 --out output/audit_train_$2 \
  --ckpt output/e3_armB_upstream/phase1_mt5_epoch10.pth=B_s0 > output/audit_train_$2.log 2>&1; }
run 0 qz & run 1 kh_ib & wait
.venv/bin/python scripts/audit_report.py --dumps output/audit_train_qz/B_s0.within_video.json \
  output/audit_train_kh_ib/B_s0.within_video.json --out output/audit_train_report > output/audit_train_report.log 2>&1
echo TRAIN-AUDIT-DONE >> output/audit_train_report.log
