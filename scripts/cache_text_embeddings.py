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

E6 adds --per-token: the same encoder states WITHOUT the mean, one row per
content token (EOS and padding dropped), for a token-level alignment objective.
Rows go to <out>.tokens.npy (flat, memory-mappable) and the per-clip index to
<out>. Same tokenizer, teacher, clip order and batching as the sequence cache,
so the only difference between E5 and E6 is the pooling. --check-against
recomputes the E5 mean from the same states and compares it with the existing
sequence cache, which verifies that.

Usage:
    python scripts/cache_text_embeddings.py --root ~/asan_canonical \
        --out ~/asan_canonical/text_teacher_mt5base.npz
    python scripts/cache_text_embeddings.py --root ~/asan_canonical --per-token \
        --out ~/asan_canonical/text_teacher_mt5base_tokens.npz \
        --check-against ~/asan_canonical/text_teacher_mt5base.npz
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
    ap.add_argument('--per-token', action='store_true',
                    help='store unpooled per-token states (E6) instead of the mean')
    ap.add_argument('--check-against', default=None,
                    help='with --per-token: an E5 sequence cache to verify against')
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    out = os.path.expanduser(args.out or os.path.join(root, 'text_teacher_mt5base.npz'))
    mt5_path = args.mt5
    if mt5_path is None:
        import yaml
        cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..',
                                               'configs', 'config.yaml')))
        mt5_path = cfg['paths'].get('mt5') or cfg['model'].get('mt5_path')
    if mt5_path is None:
        # config.yaml carries no mT5 path, so the lookup above yields None and
        # from_pretrained(None) fails. The trainer's MT5_PATH is what E5 used.
        mt5_path = 'google/mt5-base'
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

    hashes = np.array([int(hashlib.sha1(normalize_kazakh(t).encode()).hexdigest()[:15], 16)
                       for t in texts], dtype=np.int64)
    if args.per_token:
        return write_per_token(tok, enc, device, texts, clip_ids, hashes, mt5_path,
                               out, args)

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

    np.savez(out, clip_ids=np.array(clip_ids), vecs=vecs, text_hash=hashes,
             teacher=str(mt5_path), splits=args.splits)
    uniq = len(set(hashes.tolist()))
    print(f'[write] {out}\n  {len(clip_ids)} vectors, dim {vecs.shape[1]}, '
          f'{uniq} distinct texts ({100 * (1 - uniq / len(hashes)):.1f}% share a text with another clip)')


def write_per_token(tok, enc, device, texts, clip_ids, hashes, mt5_path, out, args):
    """E6 cache: one row per content token, EOS and padding dropped."""
    eos = tok.eos_token_id
    # Pass 1: exact token counts, so the flat array can be preallocated on disk.
    counts = np.array([sum(1 for i in tok(t, truncation=True, max_length=args.max_tokens)
                           ['input_ids'] if i != eos) for t in texts], dtype=np.int64)
    if (counts < 1).any():
        sys.exit(f'[fatal] {int((counts < 1).sum())} texts have no content tokens')
    offsets = np.concatenate([[0], np.cumsum(counts)])
    tok_path = out[:-4] + '.tokens.npy' if out.endswith('.npz') else out + '.tokens.npy'
    rows = np.lib.format.open_memmap(tok_path, mode='w+', dtype=np.float16,
                                     shape=(int(offsets[-1]), enc.config.d_model))
    print(f'[per-token] {int(offsets[-1]):,} rows -> {tok_path} '
          f'({rows.nbytes / 1e9:.2f} GB)')

    ref = None
    if args.check_against:
        z = np.load(os.path.expanduser(args.check_against), allow_pickle=True)
        if z['clip_ids'].tolist() != clip_ids:
            sys.exit('[fatal] --check-against cache has a different clip order')
        ref = z['vecs'].astype(np.float32)
    worst = 0.0

    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i + args.batch_size]
            t = tok(batch, padding=True, truncation=True, max_length=args.max_tokens,
                    return_tensors='pt').to(device)
            h = enc(**t).last_hidden_state                      # (B, L, D)
            attn, ids = t['attention_mask'].bool(), t['input_ids']
            if ref is not None:
                # The E5 mean includes EOS; recompute it exactly to verify the
                # two caches come from the same states.
                m = attn.unsqueeze(-1).to(h.dtype)
                pooled = ((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy()
                worst = max(worst, float(np.abs(pooled - ref[i:i + len(batch)]).max()))
            keep = attn & (ids != eos)
            for j in range(len(batch)):
                k = i + j
                v = h[j][keep[j]].float().cpu().numpy().astype(np.float16)
                if len(v) != counts[k]:
                    sys.exit(f'[fatal] clip {clip_ids[k]}: {len(v)} rows, expected {counts[k]}')
                rows[offsets[k]:offsets[k + 1]] = v
            if (i // args.batch_size) % 100 == 0:
                print(f'  {i}/{len(texts)}', flush=True)
    rows.flush()

    np.savez(out, clip_ids=np.array(clip_ids), offsets=offsets, text_hash=hashes,
             tokens_file=os.path.basename(tok_path), teacher=str(mt5_path),
             splits=args.splits, dim=enc.config.d_model)
    print(f'[write] {out}\n  {len(clip_ids)} clips, tokens/clip mean {counts.mean():.1f} '
          f'max {counts.max()}')
    if ref is not None:
        # fp16 storage of the E5 vectors bounds agreement at ~1e-3.
        print(f'[check] max |pooled - E5 cache| = {worst:.2e}')
        if worst > 5e-3:
            sys.exit('[fatal] per-token states do not reproduce the E5 cache')


if __name__ == '__main__':
    main()
