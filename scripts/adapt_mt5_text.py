#!/usr/bin/env python3
"""
E7 step 1: domain-adapt base mT5 on the canonical TRAIN transcripts (text only).

Why: initialization is the one thing that has ever moved this project's numbers.
The colleague's KRSL-tuned mT5 reached chrF 36 / recall 0.207 (E1) where every
base-mT5 run since has capped at ~29.8–31.5 and taken off in 4/19 runs. That
checkpoint's training history is unrecorded, so it cannot back a published
claim. This reaches for the same effect with clean provenance: nothing but our
own canonical TRAIN split's text.

**Objective: T5 span corruption, not language modelling.** Training the decoder
to emit corpus text unconditionally is precisely the failure mode under
investigation (the decoder ignoring its encoder and reciting broadcast
boilerplate), so teaching it to do that first would be self-defeating. Span
corruption keeps the task conditional — the model must reconstruct masked spans
from surrounding context — and is mT5's own pretraining objective, so it adapts
to Kazakh broadcast text without rehearsing unconditional generation.

Leakage: canonical TRAIN transcripts only. Dev/test text is used solely to
report perplexity before and after, never trained on.

Usage:
    python scripts/adapt_mt5_text.py --root ~/asan_canonical \
        --out ~/mt5_kaznews --epochs 3
"""
import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SOURCES = ('informburo', 'khabar', 'qazaqstantv')


def load_texts(root, split):
    out = []
    for src in SOURCES:
        p = os.path.join(root, src, 'annotations', 'kz', f'{split}.json')
        if os.path.exists(p):
            out += [e['text'] for e in json.load(open(p)) if e.get('text', '').strip()]
    return out


class SpanCorruption(Dataset):
    """
    T5 span corruption: mask ~`ratio` of tokens in spans of mean length
    `mean_span`, replace each masked span with a sentinel, and make the target
    the sentinels followed by the spans they stand for.
    """

    def __init__(self, texts, tok, max_len=128, ratio=0.15, mean_span=3.0, seed=0):
        self.texts, self.tok, self.max_len = texts, tok, max_len
        self.ratio, self.mean_span = ratio, mean_span
        self.seed = seed
        # mT5 sentinels are the last tokens of the vocabulary: <extra_id_0> is
        # the highest id (250099 for mt5-base), <extra_id_1> the next, and so on.
        # Resolve by ENCODING the literal: on this tokenizer version
        # convert_tokens_to_ids('<extra_id_0>') and get_vocab() both miss them
        # and return <unk> (id 2), which walks into negative ids as k grows.
        self.sentinel0 = tok('<extra_id_0>', add_special_tokens=False)['input_ids'][0]
        assert tok.decode([self.sentinel0]) == '<extra_id_0>', \
            f'sentinel lookup failed: id {self.sentinel0} decodes to ' \
            f'{tok.decode([self.sentinel0])!r}'
        assert tok.decode([self.sentinel0 - 99]) == '<extra_id_99>', \
            'sentinels are not contiguous descending from <extra_id_0>'

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        rng = random.Random((self.seed, i).__hash__())
        ids = self.tok(self.texts[i], truncation=True,
                       max_length=self.max_len)['input_ids']
        if ids and ids[-1] == self.tok.eos_token_id:
            ids = ids[:-1]
        if len(ids) < 4:
            ids = ids + [self.tok.pad_token_id] * (4 - len(ids))

        n_mask = max(1, int(round(len(ids) * self.ratio)))
        n_spans = max(1, int(round(n_mask / self.mean_span)))
        n_spans = min(n_spans, 99, len(ids) // 2)          # 100 sentinels exist

        # Choose non-overlapping span starts, left to right.
        starts = sorted(rng.sample(range(len(ids)), n_spans))
        spans, used, prev_end = [], 0, -1
        for s in starts:
            if s <= prev_end:
                continue
            length = max(1, min(int(rng.expovariate(1 / self.mean_span)) + 1,
                                len(ids) - s, n_mask - used))
            if length < 1:
                break
            spans.append((s, s + length))
            used += length
            prev_end = s + length - 1
            if used >= n_mask:
                break

        src, tgt, cur, k = [], [], 0, 0
        for s, e in spans:
            src += ids[cur:s] + [self.sentinel0 - k]
            tgt += [self.sentinel0 - k] + ids[s:e]
            cur, k = e, k + 1
        src += ids[cur:] + [self.tok.eos_token_id]
        tgt += [self.sentinel0 - k, self.tok.eos_token_id]
        return {'input_ids': src[:self.max_len], 'labels': tgt[:self.max_len]}


def collate(batch, pad):
    def pack(key, fill):
        n = max(len(b[key]) for b in batch)
        out = torch.full((len(batch), n), fill, dtype=torch.long)
        for i, b in enumerate(batch):
            out[i, :len(b[key])] = torch.tensor(b[key], dtype=torch.long)
        return out
    ids = pack('input_ids', pad)
    return {'input_ids': ids, 'attention_mask': (ids != pad).long(),
            'labels': pack('labels', -100)}


@torch.no_grad()
def perplexity(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        loss = model(**b).loss
        if torch.isfinite(loss):
            tot += loss.item(); n += 1
    model.train()
    return math.exp(tot / max(n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='~/asan_canonical')
    ap.add_argument('--base', default='google/mt5-base')
    ap.add_argument('--out', default='~/mt5_kaznews')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=24)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    root, out = os.path.expanduser(args.root), os.path.expanduser(args.out)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    from transformers import MT5ForConditionalGeneration, T5Tokenizer, get_cosine_schedule_with_warmup
    tok = T5Tokenizer.from_pretrained(args.base, legacy=False)
    model = MT5ForConditionalGeneration.from_pretrained(args.base)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).train()

    train_texts, dev_texts = load_texts(root, 'train'), load_texts(root, 'dev')
    print(f'[data] train {len(train_texts)} transcripts, dev {len(dev_texts)} '
          f'(dev used only for perplexity, never trained on)')

    pad = tok.pad_token_id
    tr = DataLoader(SpanCorruption(train_texts, tok, seed=args.seed),
                    batch_size=args.batch_size, shuffle=True, num_workers=4,
                    collate_fn=lambda b: collate(b, pad), drop_last=True)
    dv = DataLoader(SpanCorruption(dev_texts, tok, seed=1234),
                    batch_size=args.batch_size, shuffle=False, num_workers=2,
                    collate_fn=lambda b: collate(b, pad))

    before = perplexity(model, dv, device)
    print(f'[dev] span-corruption perplexity BEFORE: {before:.2f}', flush=True)

    steps = len(tr) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)

    step = 0
    for ep in range(args.epochs):
        for b in tr:
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b).loss
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 200 == 0:
                print(f'  epoch {ep + 1}/{args.epochs} step {step}/{steps} '
                      f'loss {loss.item():.4f}', flush=True)
        print(f'[epoch {ep + 1}] dev perplexity {perplexity(model, dv, device):.2f}',
              flush=True)

    after = perplexity(model, dv, device)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    with open(os.path.join(out, 'adaptation.json'), 'w') as fh:
        json.dump({'base': args.base, 'root': root, 'split': 'train only',
                   'objective': 'T5 span corruption (15%, mean span 3)',
                   'epochs': args.epochs, 'lr': args.lr, 'seed': args.seed,
                   'train_texts': len(train_texts),
                   'dev_perplexity_before': before, 'dev_perplexity_after': after},
                  fh, indent=1)
    print(f'[dev] span-corruption perplexity AFTER:  {after:.2f}  '
          f'({100 * (1 - after / before):.1f}% lower)')
    print(f'[write] {out}')


if __name__ == '__main__':
    main()
