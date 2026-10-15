#!/usr/bin/env python3
"""
Sign -> text RETRIEVAL with an existing translation checkpoint. No training.

Why: generation fails (WER ~0.95) but E5's pose-text alignment loss fell to
~1.9 against a chance of ln(264) = 5.58, i.e. the encoder can tell its own
transcript from others far above chance. If that holds held-out, "pick the
right sentence / keywords" is a task this project can already do, even while
"write the sentence" is not.

Method: for each clip x in the frozen selection set, score every reference t
of the SAME source (200 candidates) by the checkpoint's own likelihood
log p(t | x), teacher-forced exactly as in training (same prefix, same
tokenisation, max 128 tokens incl. EOS). Same-source only, because
qazaqstantv text is cased and punctuated while khabar/informburo is not --
mixing sources would let the model "retrieve" by spotting casing style.

Two scores are reported:
  raw  log p(t | x)                    -- favours short, generic texts
  PMI  log p(t | x) - log p(t | null)  -- null = mean over K other clips' poses
                                          of the same source, which removes
                                          how likely t is regardless of pose.
PMI is the primary score; raw is shown because the gap between them is itself
a measure of how much the text prior dominates.

Chance for 200-way retrieval: top-1 0.5%, top-5 2.5%, MRR ~0.029.

Usage (stride and target_fps are read from each checkpoint's run_args):
    PYTHONPATH=. python scripts/eval_retrieval.py --root ~/asan_canonical \
        --unisign-preprocess --block-padding-mask --out output/retrieval \
        --ckpt output/e3_armB_upstream/phase1_mt5_epoch10.pth=B_s0
"""
import argparse
import collections
import contextlib
import io
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers.modeling_outputs import BaseModelOutput

from data.utils import ENRICHED_DIM, KEYPOINT_DIM, UNISIGN_DIM
from models.unisign_encoder import KeypointEncoder
from scripts.rescore_checkpoints import build_loader, check_run_args, load_weights, log
from train.train_encoder_mt5 import UniSignMT5

LEGACY_STRIDE = 2   # every checkpoint without the run_args key trained at 2


@torch.no_grad()
def encode_all(model, loader, device):
    """mT5-encoder states per clip, trimmed to valid positions. Mirrors
    UniSignMT5.generate up to the encoder exactly."""
    out = {}
    for batch in loader:
        if batch is None:
            continue
        kps = batch['keypoints'].to(device)
        lens = batch['input_lengths'].to(device)
        B = kps.size(0)
        pose = model.pose_norm(model.encoder(kps, input_lengths=lens))
        pre = model.mt5.shared(model.prefix_ids.unsqueeze(0).expand(B, -1))
        pre_m = model.prefix_attn.unsqueeze(0).expand(B, -1)
        mask = torch.cat([pre_m, model._pose_mask(kps, lens)], dim=1)
        h = model.mt5.encoder(inputs_embeds=torch.cat([pre, pose], dim=1),
                              attention_mask=mask).last_hidden_state
        for i, cid in enumerate(batch['clip_ids']):
            n = int(mask[i].sum())
            # Padding is at the end of the pose segment, so the first n
            # positions are exactly the valid ones.
            assert bool(mask[i, :n].all()), 'non-contiguous mask'
            out[cid] = h[i, :n].float()
    return out


@torch.no_grad()
def score(model, h, labels, chunk):
    """log p(text | encoder states h) for each row of labels. h: (L, D)."""
    res = []
    for s in range(0, labels.size(0), chunk):
        lab = labels[s:s + chunk]
        n = lab.size(0)
        hh = h.unsqueeze(0).expand(n, -1, -1)
        logits = model.mt5(encoder_outputs=BaseModelOutput(last_hidden_state=hh),
                           attention_mask=torch.ones(hh.shape[:2], device=h.device,
                                                     dtype=torch.long),
                           labels=lab).logits
        nll = F.cross_entropy(logits.float().transpose(1, 2), lab,
                              ignore_index=-100, reduction='none').sum(1)
        res.append(-nll)
    return torch.cat(res)


def ranks_of(scores, gold):
    """1-based rank of gold in each row (ties broken pessimistically)."""
    g = scores[np.arange(len(gold)), gold][:, None]
    return (scores >= g).sum(1)


def summarise(rk):
    rk = np.asarray(rk)
    return {'n': int(len(rk)), 'top1': float((rk <= 1).mean()),
            'top5': float((rk <= 5).mean()), 'top10': float((rk <= 10).mean()),
            'mrr': float((1.0 / rk).mean()), 'median_rank': float(np.median(rk))}


def labels_for(tok, texts, device):
    """Teacher-forcing labels exactly as SimpleCollator builds them."""
    ids = [tok(t.strip(), truncation=True, max_length=128)['input_ids'] for t in texts]
    lab = torch.full((len(ids), max(len(x) for x in ids)), -100, dtype=torch.long)
    for j, x in enumerate(ids):
        lab[j, :len(x)] = torch.tensor(x)
    return lab.to(device)


def within_video(model, H, refs, meta, tok, args, rng, device, label):
    """T2: each clip ranked against the sentences of its own broadcast."""
    vid = {m['clip_id']: m['video_id'] for m in meta}
    src = {m['clip_id']: m['source'] for m in meta}
    groups, by_source = collections.defaultdict(list), collections.defaultdict(list)
    for c in H:
        groups[(src[c], vid[c])].append(c)
        by_source[src[c]].append(c)
    per_query, dump = [], []
    for (s, v), ids in sorted(groups.items()):
        if len(ids) < 2:
            continue
        ids = sorted(ids)
        lab = labels_for(tok, [refs[c] for c in ids], device)
        raw = torch.stack([score(model, H[c], lab, args.chunk) for c in ids]).cpu().numpy()
        # Null poses from OTHER broadcasts of the same source: they share the
        # source's style but none of this story's content.
        others = [c for c in by_source[s] if vid[c] != v]
        nulls = rng.sample(others, min(args.null_k, len(others)))
        null = torch.stack([score(model, H[c], lab, args.chunk) for c in nulls]).mean(0).cpu().numpy()
        pmi = raw - null[None, :]
        gold = np.arange(len(ids))
        for r_raw, r_pmi in zip(ranks_of(raw, gold), ranks_of(pmi, gold)):
            per_query.append((s, len(ids), int(r_raw), int(r_pmi)))
        dump.append({'source': s, 'video': v, 'ids': ids, 'raw': raw.tolist(),
                     'null': null.tolist()})
    with open(os.path.join(args.out, f'{label}.within_video.json'), 'w') as fh:
        json.dump(dump, fh)

    def summ(rows):
        n = np.array([r[1] for r in rows], float)
        rr, rp = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
        chance_mrr = np.mean([np.mean(1.0 / np.arange(1, k + 1)) for k in n.astype(int)])
        return {'queries': len(rows), 'mean_pool': float(n.mean()),
                'pmi_top1': float(np.mean(rp == 1)), 'pmi_mrr': float(np.mean(1.0 / rp)),
                'raw_top1': float(np.mean(rr == 1)), 'raw_mrr': float(np.mean(1.0 / rr)),
                'chance_top1': float(np.mean(1.0 / n)), 'chance_mrr': float(chance_mrr)}
    out = {'per_source': {s: summ([r for r in per_query if r[0] == s])
                          for s in sorted({r[0] for r in per_query})},
           'all': summ(per_query)}
    for s, x in list(out['per_source'].items()) + [('ALL', out['all'])]:
        log(f'  T2 {s:<12} {x["queries"]:>4} queries, pool ~{x["mean_pool"]:.1f}  '
            f'PMI top1 {x["pmi_top1"]:.3f} MRR {x["pmi_mrr"]:.3f}  | raw top1 {x["raw_top1"]:.3f}'
            f'  | chance top1 {x["chance_top1"]:.3f} MRR {x["chance_mrr"]:.3f}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--root', default='~/asan_canonical')
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--ckpt', action='append', required=True, help='path=label')
    ap.add_argument('--unisign-preprocess', action='store_true')
    ap.add_argument('--block-padding-mask', action='store_true')
    ap.add_argument('--target-fps', type=int, default=None)
    ap.add_argument('--null-k', type=int, default=3)
    ap.add_argument('--split', default='val', choices=['val', 'test', 'train'])
    ap.add_argument('--pool', default='source', choices=['source', 'video'],
                    help='source: rank against every same-source clip in the manifest '
                         '(T1). video: rank only against sentences of the SAME broadcast '
                         '(T2, within-broadcast alignment); null poses from other videos.')
    ap.add_argument('--chunk', type=int, default=25)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='output/retrieval')
    # check_run_args compares against these; retrieval never uses them
    for flag in ('use_enriched', 'signspace', 'real_wrists'):
        ap.add_argument('--' + flag.replace('_', '-'), action='store_true')
    ap.add_argument('--score-quantiles', default=None)
    ap.add_argument('--unisign-hand-scale', default=None)
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    manifest = os.path.expanduser(args.manifest or os.path.join(root, 'selection_manifest.json'))
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = yaml.safe_load(open(args.config))
    rng = random.Random(args.seed)

    input_dim = UNISIGN_DIM if args.unisign_preprocess else KEYPOINT_DIM
    with contextlib.redirect_stdout(io.StringIO()):
        enc = KeypointEncoder(hidden_dim=cfg['model']['d_model'], input_dim=input_dim,
                              block_padding_mask=args.block_padding_mask,
                              unisign_input=args.unisign_preprocess)
        model = UniSignMT5(encoder=enc, lang='Kazakh').to(device).eval()
    tok = model.mt5_tokenizer

    loaders = {}
    results = {}
    for spec in args.ckpt:
        path, label = spec.split('=', 1)
        path = os.path.expanduser(path)
        log(f'=== {label}: {path}')
        info = load_weights(model, path, device)
        model.eval()
        args.downsample_every = None
        check_run_args(info, args, path)
        stride = (info.get('run_args') or {}).get('downsample_every', LEGACY_STRIDE)
        cap = (info.get('run_args') or {}).get('cap_frames')
        if (stride, cap) not in loaders:
            loaders[(stride, cap)] = build_loader(root, manifest, False, False, stride, tok,
                                                  args.batch_size,
                                                  unisign_preprocess=args.unisign_preprocess,
                                                  target_fps=args.target_fps, cap_frames=cap,
                                                  split=args.split)
        loader, meta = loaders[(stride, cap)]
        log(f'  stride {stride}, target_fps {args.target_fps}, {len(meta)} clips')

        t0 = time.time()
        H = encode_all(model, loader, device)
        refs = {}
        for batch in loader:
            if batch is None:
                continue
            for cid, txt in zip(batch['clip_ids'], batch['texts']):
                refs[cid] = txt
        by_src = collections.defaultdict(list)
        for m in meta:
            if m['clip_id'] in H:
                by_src[m['source']].append(m['clip_id'])
        if args.pool == 'video':
            results[label] = within_video(model, H, refs, meta, tok, args, rng, device, label)
            results[label]['stride'] = stride
            results[label]['seconds'] = round(time.time() - t0, 1)
            with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
                json.dump(results, fh, indent=1)
            continue

        per_src, all_raw, all_pmi = {}, [], []
        for src, ids in sorted(by_src.items()):
            ids = sorted(ids)
            ids_tok = [tok(refs[c].strip(), truncation=True, max_length=128)['input_ids']
                       for c in ids]
            T = max(len(x) for x in ids_tok)
            lab = torch.full((len(ids), T), -100, dtype=torch.long)
            for j, x in enumerate(ids_tok):
                lab[j, :len(x)] = torch.tensor(x)
            lab = lab.to(device)

            raw = torch.stack([score(model, H[c], lab, args.chunk) for c in ids]).cpu().numpy()
            # null[j]: mean log p(text_j | pose of K OTHER clips of this source)
            null = np.zeros(len(ids))
            for j in range(len(ids)):
                others = rng.sample([k for k in range(len(ids)) if k != j], args.null_k)
                null[j] = np.mean([raw[k, j] for k in others])
            pmi = raw - null[None, :]
            gold = np.arange(len(ids))
            r_raw, r_pmi = ranks_of(raw, gold), ranks_of(pmi, gold)
            per_src[src] = {'raw': summarise(r_raw), 'pmi': summarise(r_pmi),
                            'candidates': len(ids)}
            all_raw += list(r_raw)
            all_pmi += list(r_pmi)
            log(f'  {src:<12} {len(ids)}-way  PMI top1 {per_src[src]["pmi"]["top1"]:.3f} '
                f'top5 {per_src[src]["pmi"]["top5"]:.3f} MRR {per_src[src]["pmi"]["mrr"]:.3f}'
                f'  | raw top1 {per_src[src]["raw"]["top1"]:.3f}')
            np.save(os.path.join(args.out, f'{label}.{src}.scores.npy'), raw)
            with open(os.path.join(args.out, f'{label}.{src}.ids.json'), 'w') as fh:
                json.dump(ids, fh)

        results[label] = {'stride': stride, 'per_source': per_src,
                          'all_raw': summarise(all_raw), 'all_pmi': summarise(all_pmi),
                          'seconds': round(time.time() - t0, 1)}
        a = results[label]['all_pmi']
        log(f'  ALL  PMI top1 {a["top1"]:.3f}  top5 {a["top5"]:.3f}  top10 {a["top10"]:.3f}  '
            f'MRR {a["mrr"]:.3f}  median rank {a["median_rank"]:.0f}   '
            f'(chance top1 0.005, top5 0.025)  [{results[label]["seconds"]}s]')
        with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
            json.dump(results, fh, indent=1)
    log('RETRIEVAL DONE')


if __name__ == '__main__':
    main()
