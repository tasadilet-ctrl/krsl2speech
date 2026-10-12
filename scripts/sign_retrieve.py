#!/usr/bin/env python3
"""
Rank candidate sentences for signed clips.  (E12: retrieval as a deliverable.)

Given one or more KRSL clips as pose files and a text file of candidate
sentences (one per line), score every (clip, sentence) pair with a trained
translation checkpoint and print each clip's best-matching sentences.

    PYTHONPATH=. python scripts/sign_retrieve.py \
        --ckpt output/e3_armB_upstream/phase1_mt5_epoch10.pth \
        --queries clip1.pkl clip2.npz --candidates broadcast_transcript.txt \
        --source khabar --top 3 --out ranking.json

Score = PMI: log p(sentence | clip) - mean log p(sentence | K null clips).
Raw likelihood alone mostly measures how generic a sentence is (near chance
in every test); subtracting the null removes that. Null clips are drawn from
the frozen dev selection of the given source (or all sources).

What to expect (dev and test numbers: EXPERIMENT_e12_retrieval_task.md):
pick the right sentence from ~200 unrelated news sentences most of the time;
from the sentences of the same broadcast, see E12 T2. Use a checkpoint that
took off -- a non-takeoff checkpoint retrieves at near chance.

Inputs:
  * pose files in the corpus formats: .pkl {'keypoints': (T,1,133,2),
    'scores': (T,1,133)} or .npz {'wb_xy': (T,133,2), 'wb_score': (T,133)}
    (COCO-WholeBody, RTMPose), at the source's native frame rate.
  * candidates written the way that source's training text is written:
    qazaqstantv text is cased and punctuated, khabar/informburo are lowercase
    without punctuation. A mismatch in style lowers the score of the right line.
Preprocessing, stride, frame cap and target fps are taken from the
checkpoint's run_args and applied by the same AsanDataset code used in training.
"""
import argparse
import contextlib
import io
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402
import yaml                                                                # noqa: E402

from data.asan_dataset import AsanDataset                                  # noqa: E402
from data.utils import UNISIGN_DIM                                         # noqa: E402
from models.unisign_encoder import KeypointEncoder                         # noqa: E402
from scripts.eval_retrieval import encode_all, labels_for, score           # noqa: E402
from scripts.rescore_checkpoints import load_weights                       # noqa: E402
from train.collate import SimpleCollator                                   # noqa: E402
from train.train_encoder_mt5 import UniSignMT5                             # noqa: E402

SOURCES = ('informburo', 'khabar', 'qazaqstantv')
LEGACY_STRIDE = 2


def build_dataset(entries, ra, root):
    with contextlib.redirect_stdout(io.StringIO()):
        ds = AsanDataset(root=root, sources=[], split='val', lang='kz',
                         unisign_preprocess=True,
                         downsample_every=ra.get('downsample_every', LEGACY_STRIDE),
                         target_fps=ra.get('target_fps'), cap_frames=ra.get('cap_frames'))
    ds.clips = entries
    return ds


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--queries', nargs='+', required=True, help='pose files (.pkl / .npz)')
    ap.add_argument('--candidates', required=True, help='text file, one sentence per line')
    ap.add_argument('--source', choices=SOURCES, default=None,
                    help='broadcast source: picks null clips (and is required by '
                         'checkpoints trained with --target-fps)')
    ap.add_argument('--null-k', type=int, default=8)
    ap.add_argument('--top', type=int, default=5)
    ap.add_argument('--root', default='~/asan_canonical')
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--chunk', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='write full rankings as JSON')
    a = ap.parse_args()
    root = os.path.expanduser(a.root)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cands = [l.strip() for l in open(a.candidates, encoding='utf-8') if l.strip()]
    if len(cands) < 2:
        sys.exit('[fatal] need at least two candidate sentences')

    cfg = yaml.safe_load(open(a.config))
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    ra = dict(ck.get('run_args') or {})
    del ck
    if not ra.get('unisign_preprocess', False):
        sys.exit('[fatal] only checkpoints trained with --unisign-preprocess are supported')
    if ra.get('target_fps') and not a.source:
        sys.exit('[fatal] this checkpoint resamples by source frame rate: pass --source')

    with contextlib.redirect_stdout(io.StringIO()):
        enc = KeypointEncoder(hidden_dim=cfg['model']['d_model'], input_dim=UNISIGN_DIM,
                              block_padding_mask=bool(ra.get('block_padding_mask')),
                              unisign_input=True)
        model = UniSignMT5(encoder=enc, lang='Kazakh').to(device).eval()
    load_weights(model, a.ckpt, device)
    model.eval()
    tok = model.mt5_tokenizer
    collate = SimpleCollator(mt5_tokenizer=tok, max_text_tokens=128)

    # Queries. The collator drops samples without text, so a placeholder is set;
    # it is never scored.
    q_entries = [{'pose': os.path.abspath(p), 'text': '.', 'clip_id': f'q{i}:{os.path.basename(p)}',
                  '_source': a.source or 'khabar'} for i, p in enumerate(a.queries)]
    # Null clips: frozen dev selection, given source if known.
    sel = json.load(open(os.path.join(root, 'selection_manifest.json')))
    pool = [m for m in sel if a.source is None or m['source'] == a.source]
    rng = random.Random(a.seed)
    picked = rng.sample(pool, min(a.null_k, len(pool)))
    ann = {}
    for s in {m['source'] for m in picked}:
        for e in json.load(open(os.path.join(root, s, 'annotations', 'kz', 'dev.json'))):
            ann[e['clip_id']] = dict(e, _source=s)
    n_entries = [dict(ann[m['clip_id']], text='.') for m in picked]

    def loader(entries):
        return torch.utils.data.DataLoader(build_dataset(entries, ra, root), batch_size=8,
                                           shuffle=False, collate_fn=collate)
    Hq = encode_all(model, loader(q_entries), device)
    Hn = encode_all(model, loader(n_entries), device)
    missing = [e['clip_id'] for e in q_entries if e['clip_id'] not in Hq]
    if missing:
        sys.exit(f'[fatal] could not load: {missing}')

    lab = labels_for(tok, cands, device)
    null = torch.stack([score(model, h, lab, a.chunk) for h in Hn.values()]).mean(0).cpu().numpy()
    report = []
    for e in q_entries:
        raw = score(model, Hq[e['clip_id']], lab, a.chunk).cpu().numpy()
        pmi = raw - null
        order = np.argsort(-pmi)
        name = e['clip_id'].split(':', 1)[1]
        print(f'\n== {name}')
        for r, j in enumerate(order[:a.top], 1):
            print(f'  {r}. [{pmi[j]:+7.2f}] {cands[j][:110]}')
        report.append({'query': e['pose'],
                       'ranking': [{'rank': r, 'line': int(j) + 1, 'pmi': float(pmi[j]),
                                    'raw_logp': float(raw[j]), 'text': cands[j]}
                                   for r, j in enumerate(order, 1)]})
    if a.out:
        json.dump({'checkpoint': a.ckpt, 'run_args': ra, 'null_clips': [m['clip_id'] for m in picked],
                   'results': report}, open(a.out, 'w'), ensure_ascii=False, indent=1)
        print(f'\n[write] {a.out}')


if __name__ == '__main__':
    main()
