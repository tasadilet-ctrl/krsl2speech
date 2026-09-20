#!/usr/bin/env python3
"""
E5: cache a frozen text-teacher embedding per clip, for pose-text alignment.

Why cached and frozen:
  * No second copy of mT5 on the GPU (other users leave us ~36 GB; a run
    already needs ~33 GB).
  * The alignment target does not drift while the model trains.
  * Negatives become free: with batch size 8, an in-batch contrastive loss has
    only 7 negatives, and gradient accumulation does not add any. Sampling rows
    from this table gives hundreds of negatives at no memory cost.

Embedding = mean over non-pad tokens of the BASE (pretrained, unmodified) mT5
encoder's last hidden state for the clip's reference text. Base mT5 has never
seen KRSL, so this adds no data leakage.

Also stores a hash of the normalized text per clip, so the training loop can
drop false negatives: 2.6% of training references are exact duplicates of
another clip's (repeated broadcast boilerplate), and pushing those apart would
be actively wrong.

Usage:
    python scripts/cache_text_embeddings.py --root ~/asan_canonical \
        --out ~/asan_canonical/text_teacher_mt5base.npz
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import normalize_kazakh  # noqa: E402

SOURCES = ('informburo', 'khabar', 'qazaqstantv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='~/asan_canonical')
    ap.add_argument('--mt5', default=None, help='defaults to configs/config.yaml MT5 path')
    ap.add_argument('--out', default=None)
    ap.add_argument('--splits', default='train,dev')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--max-tokens', type=int, default=128)
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    out = os.path.expanduser(args.out or os.path.join(root, 'text_teacher_mt5base.npz'))
    mt5_path = args.mt5
    if mt5_path is None:
        import yaml
        cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..',
                                               'configs', 'config.yaml')))
        mt5_path = cfg['paths'].get('mt5') or cfg['model'].get('mt5_path')
    print(f'[teacher] {mt5_path}')

    from transformers import MT5EncoderModel, T5Tokenizer
    tok = T5Tokenizer.from_pretrained(mt5_path, legacy=False)
    enc = MT5EncoderModel.from_pretrained(mt5_path).eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = enc.to(device)

    clip_ids, texts = [], []
    for split in args.splits.split(','):
        for src in SOURCES:
            p = os.path.join(root, src, 'annotations', 'kz', f'{split}.json')
            if not os.path.exists(p):
                continue
            for e in json.load(open(p)):
                if e.get('text', '').strip():
                    clip_ids.append(e['clip_id'])
                    texts.append(e['text'])
    print(f'[texts] {len(clip_ids)} clips over splits {args.splits}')

    vecs = np.zeros((len(texts), enc.config.d_model), dtype=np.float16)
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i + args.batch_size]
            t = tok(batch, padding=True, truncation=True, max_length=args.max_tokens,
                    return_tensors='pt').to(device)
            h = enc(**t).last_hidden_state                      # (B, L, D)
            m = t['attention_mask'].unsqueeze(-1).to(h.dtype)   # mean over real tokens
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            vecs[i:i + len(batch)] = v.float().cpu().numpy().astype(np.float16)
            if (i // args.batch_size) % 100 == 0:
                print(f'  {i}/{len(texts)}', flush=True)

    hashes = np.array([int(hashlib.sha1(normalize_kazakh(t).encode()).hexdigest()[:15], 16)
                       for t in texts], dtype=np.int64)
    np.savez(out, clip_ids=np.array(clip_ids), vecs=vecs, text_hash=hashes,
             teacher=str(mt5_path), splits=args.splits)
    uniq = len(set(hashes.tolist()))
    print(f'[write] {out}\n  {len(clip_ids)} vectors, dim {vecs.shape[1]}, '
          f'{uniq} distinct texts ({100 * (1 - uniq / len(hashes)):.1f}% share a text with another clip)')


if __name__ == '__main__':
    main()
