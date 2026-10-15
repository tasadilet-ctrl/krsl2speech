#!/usr/bin/env python3
"""
Minimal trainer: pose keypoints -> ST-GCN (Uni-Sign) -> mT5. One pipeline.

Why it exists: train/train_encoder_mt5.py grew to 2,341 lines carrying PGF,
prosody, CTC, LoRA, masked-pose, alignment, enriched-feature and DDP paths that
no current experiment uses. Each was a surface for silent bugs -- two optimizer
param-group bugs, a generator-exhaustion bug, and a config default wrong for
two of three sources all produced plausible numbers. This file supports exactly
the configuration every live experiment uses (the E3 "B" config) and nothing
else, so the whole training path fits on a few screens.

Supported: upstream Uni-Sign preprocessing + padding-masked ST-GCN (always on),
stride / target_fps, Uni-Sign or --init-from initialisation, --resume, seed,
LR overrides. Dropped: everything listed above, in-loop beam search (score with
scripts/rescore_checkpoints.py or scripts/screen_epoch5.py instead), and
best-by-WER selection (WER is ~1.0 for most runs, so it selected noise).

Equivalence: construction order, optimizer groups, schedule, accumulation,
clipping and the non-finite-gradient rule match the old trainer exactly, so with
the same seed the two produce the same first-epoch batches and losses -- see
scripts/parity_train_min.sh. Checkpoints use the same keys, so every existing
tool reads them. Later epochs draw a different shuffle from the old trainer
because the old one also iterates a generation loader each epoch, which
consumes the global RNG; that is a different seed, not a different method.

Usage:
    PYTHONPATH=. python train/train_min.py --seed 0 --save-dir output/run \
        --pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (MT5ForConditionalGeneration, T5Tokenizer,
                          get_cosine_schedule_with_warmup)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.asan_dataset import AsanDataset          # noqa: E402
from data.utils import UNISIGN_DIM                 # noqa: E402
from models.unisign_encoder import KeypointEncoder, load_unisign_weights  # noqa: E402
from train.collate import SimpleCollator           # noqa: E402
from utils.paths import apply_env_overrides        # noqa: E402

LEGACY_STRIDE = 2        # every checkpoint without the run_args key used 2
INPUT_KEYS = ('unisign_preprocess', 'block_padding_mask', 'real_wrists', 'signspace',
              'use_enriched', 'target_fps', 'downsample_every', 'cap_frames')


def log(msg):
    print(msg, flush=True)


def enable_determinism():
    """Must run before the first CUDA op. Raises if any op lacks a
    deterministic implementation rather than silently falling back."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SignMT5(nn.Module):
    """Baseline model. Submodule names match UniSignMT5 (encoder, pose_norm,
    mt5, prefix_ids, prefix_attn) so checkpoints are interchangeable."""

    def __init__(self, encoder, mt5_path, lang='Kazakh'):
        super().__init__()
        self.encoder = encoder
        # Raw encoder outputs were ~6x mT5's embedding scale with a shared
        # direction across clips; without this cross-attention saturates.
        self.pose_norm = nn.LayerNorm(encoder.hidden_dim)
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(mt5_path)
        self.mt5_tokenizer = T5Tokenizer.from_pretrained(mt5_path, legacy=False)
        tok = self.mt5_tokenizer([f"Translate sign language video to {lang}: "],
                                 padding='longest', truncation=True, return_tensors='pt')
        self.register_buffer('prefix_ids', tok['input_ids'].squeeze(0))
        self.register_buffer('prefix_attn', tok['attention_mask'].squeeze(0))

    def forward(self, kps, label_ids, input_lengths):
        B = kps.size(0)
        pose = self.pose_norm(self.encoder(kps, input_lengths=input_lengths))
        prefix = self.mt5.shared(self.prefix_ids.unsqueeze(0).expand(B, -1))
        pose_mask = (torch.arange(kps.size(1), device=kps.device)[None, :]
                     < input_lengths[:, None]).long()
        mask = torch.cat([self.prefix_attn.unsqueeze(0).expand(B, -1), pose_mask], dim=1)
        return self.mt5(inputs_embeds=torch.cat([prefix, pose], dim=1),
                        attention_mask=mask, labels=label_ids, return_dict=True).loss


def input_pipeline(args, cfg):
    stride = (args.downsample_every if args.downsample_every is not None
              else cfg['paths']['asan'].get('downsample_every', 1))
    return {'unisign_preprocess': True, 'block_padding_mask': True, 'real_wrists': False,
            'signspace': False, 'use_enriched': False, 'target_fps': args.target_fps,
            'downsample_every': int(stride), 'cap_frames': args.cap_frames}


def check_pipeline(src_args, cur, path, allow=False):
    src = dict(src_args or {})
    src.setdefault('downsample_every', LEGACY_STRIDE)
    src.setdefault('cap_frames', None)           # no checkpoint before E11 had a cap
    diff = {k: (src.get(k), cur[k]) for k in INPUT_KEYS if k in src and src.get(k) != cur[k]}
    if diff and not allow:
        sys.exit(f'[fatal] {path} used a different input pipeline (checkpoint, this run): '
                 f'{diff}' + ('' if allow is None else
                              '. Pass --init-allow-input-change to do it deliberately.'))
    return diff


def load_model_weights(model, ck):
    model.encoder.load_state_dict(ck['encoder'])
    if 'pose_norm' in ck:
        model.pose_norm.load_state_dict(ck['pose_norm'])
    if 'mt5' not in ck:
        sys.exit('[fatal] checkpoint has no full mT5 weights')
    model.mt5.load_state_dict(ck['mt5'])


def video_of(e):
    """video_id field, else the middle of 'src__video__segNNN'."""
    return e.get('video_id') or e['clip_id'].split('__')[1]


def build_loaders(cfg, pipe, collator, batch_size, keep=None):
    a = cfg['paths']['asan']
    common = dict(root=a['root'], sources=a.get('sources', ['informburo', 'khabar', 'qazaqstantv']),
                  lang=a.get('lang', 'kz'), tokenizer=None,
                  max_frames=cfg['training']['phase1']['max_seq_len'],
                  downsample_every=pipe['downsample_every'], unisign_preprocess=True,
                  target_fps=pipe['target_fps'], cap_frames=pipe['cap_frames'],
                  skip_low_quality=a.get('skip_low_quality', True),
                  min_hand_cov=a.get('min_hand_cov', 0.0))
    train = AsanDataset(split='train', **common)
    if keep is not None:
        before = len(train.clips)
        train.clips = [e for e in train.clips if f"{e['_source']}/{video_of(e)}" in keep]
        log(f'[train-videos] kept {len(train.clips)} of {before} train clips')
        if not train.clips:
            sys.exit('[fatal] --train-videos matched no training clips')
    val = AsanDataset(split='val', **common)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4,
                       collate_fn=collator, pin_memory=True, persistent_workers=True),
            DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2,
                       collate_fn=collator, pin_memory=True))


def step(model, optimizer, scheduler):
    """Clip, step only on a finite norm, always zero (drops non-finite grads)."""
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if torch.isfinite(norm):
        optimizer.step()
    optimizer.zero_grad()
    scheduler.step()


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for b in loader:
        if b is None:
            continue
        loss = model(b['keypoints'].to(device), b['label_ids'].to(device),
                     b['input_lengths'].to(device))
        if torch.isfinite(loss):
            tot += loss.item(); n += 1
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=None, help='total, incl. resumed')
    ap.add_argument('--save-dir', required=True)
    ap.add_argument('--pretrained-unisign', default=None)
    ap.add_argument('--init-from', default=None,
                    help='full model, weights only, fresh optimizer/schedule')
    ap.add_argument('--init-allow-input-change', action='store_true')
    ap.add_argument('--resume', default=None, help='continue a train_min run')
    ap.add_argument('--downsample-every', type=int, default=None)
    ap.add_argument('--target-fps', type=int, default=None)
    ap.add_argument('--train-videos', default=None,
                    help='JSON list of "source/video_id": train only on these broadcasts '
                         '(cross-fitting for the train-set audit). Dev is unaffected.')
    ap.add_argument('--cap-frames', type=int, default=None,
                    help='Uni-Sign-style cap: clips longer than this keep a random sorted '
                         'subset of frames in training, evenly spaced frames in eval. '
                         'Uni-Sign pretrained with 256.')
    ap.add_argument('--mt5-path', default='google/mt5-base')
    ap.add_argument('--encoder-lr', type=float, default=None)
    ap.add_argument('--mt5-lr', type=float, default=None)
    ap.add_argument('--grad-accum', type=int, default=None)
    ap.add_argument('--log-every', type=int, default=500)
    ap.add_argument('--save-every', type=int, default=5)
    ap.add_argument('--deterministic', action='store_true',
                    help='Deterministic GPU kernels. Without this, the same seed does NOT\n'
                         'reproduce a run: nondeterministic cuDNN/cuBLAS reductions diverge\n'
                         'from the second optimizer step and early training amplifies it\n'
                         '(same-seed runs differed by ~1.1 loss within 400 batches).')
    ap.add_argument('--max-batches', type=int, default=None,
                    help='stop after this many batches (smoke / parity tests only)')
    args = ap.parse_args()
    if args.init_from and args.resume:
        ap.error('--init-from and --resume are exclusive')

    if args.deterministic:
        enable_determinism()
    # Seed exactly as the old trainer's main() does, before anything is built.
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    cfg = apply_env_overrides(yaml.safe_load(open(args.config)))
    p1 = cfg['training']['phase1']
    device = torch.device('cuda:0')
    pipe = input_pipeline(args, cfg)
    os.makedirs(args.save_dir, exist_ok=True)

    # Construction order matches the old trainer (encoder -> Uni-Sign weights ->
    # model -> device -> optimizer -> data) so the RNG stream, and therefore
    # the first epoch's shuffle and dropout, is identical for the same seed.
    encoder = KeypointEncoder(hidden_dim=cfg['model']['d_model'], input_dim=UNISIGN_DIM,
                              block_padding_mask=True, real_wrists=False, unisign_input=True)
    if args.pretrained_unisign:
        load_unisign_weights(encoder, args.pretrained_unisign)
    model = SignMT5(encoder, args.mt5_path)
    run_args = dict(pipe, mt5_path=args.mt5_path, selection_manifest=None,
                    score_quantiles=None, unisign_hand_scale=None, trainer='train_min',
                    seed=args.seed, deterministic=args.deterministic)
    if args.init_from:
        ck = torch.load(args.init_from, map_location='cpu', weights_only=False)
        diff = check_pipeline(ck.get('run_args'), pipe, args.init_from,
                              args.init_allow_input_change)
        load_model_weights(model, ck)
        run_args['init_from'] = os.path.abspath(args.init_from)
        log(f'[init] weights from {args.init_from} (epoch {ck.get("epoch")})'
            + (f'; INPUT CHANGE {diff}' if diff else ''))
        del ck
    model.to(device)

    base_lr = p1.get('learning_rate', 5e-4)
    groups = [{'params': list(model.encoder.parameters()),
               'lr': args.encoder_lr if args.encoder_lr is not None else base_lr / 10},
              {'params': list(model.mt5.parameters()),
               'lr': args.mt5_lr if args.mt5_lr is not None else base_lr},
              {'params': list(model.pose_norm.parameters()), 'lr': base_lr}]
    n_opt = sum(p.numel() for g in groups for p in g['params'])
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_opt == n_train, f'optimizer covers {n_opt:,} of {n_train:,} trainable params'
    optimizer = AdamW(groups, weight_decay=0.01)
    log(f'[optimizer] {n_opt:,} params in 3 groups, lrs '
        + str([f"{g['lr']:.2e}" for g in groups]))

    start_epoch, global_step = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        check_pipeline(ck.get('run_args'), pipe, args.resume, allow=None)
        load_model_weights(model, ck)
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch, global_step = ck['epoch'] + 1, ck['global_step']
        run_args = dict(ck['run_args'], **{k: run_args[k] for k in ('seed',)})
        log(f'[resume] {args.resume}: continuing at epoch {start_epoch + 1}, step {global_step}')
        del ck

    collator = SimpleCollator(mt5_tokenizer=model.mt5_tokenizer,
                              max_text_tokens=p1.get('max_text_len', 128))
    keep = set(json.load(open(args.train_videos))) if args.train_videos else None
    if args.train_videos:
        run_args['train_videos'] = os.path.abspath(args.train_videos)
    train_loader, val_loader = build_loaders(cfg, pipe, collator, p1['batch_size'], keep)
    accum = max(1, args.grad_accum or p1.get('grad_accum', 1))
    epochs = args.epochs or p1['max_epochs']
    total = math.ceil(len(train_loader) / accum) * epochs
    warmup = min(p1.get('warmup_steps', 1500), max(total // 10, 1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total,
        last_epoch=global_step - 1 if global_step > 0 else -1)
    log(f'[run] {pipe}  epochs {epochs}  accum {accum}  steps {total}  warmup {warmup}')

    seen = 0
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        tot, n, pending = 0.0, 0, 0
        optimizer.zero_grad()
        for i, b in enumerate(train_loader):
            if b is None:
                continue
            loss = model(b['keypoints'].to(device), b['label_ids'].to(device),
                         b['input_lengths'].to(device))
            (loss / accum).backward()
            pending += 1
            if torch.isfinite(loss):
                tot += loss.item(); n += 1
            if pending >= accum:
                step(model, optimizer, scheduler)
                pending = 0; global_step += 1
            if i % args.log_every == 0 and i > 0:
                log(f'  Batch {i}/{len(train_loader)} | Loss: {tot / n:.4f}')
            seen += 1
            if args.max_batches and seen >= args.max_batches:
                log(f'[stop] --max-batches {args.max_batches} reached')
                return
        if pending:                       # flush the partial accumulation window
            step(model, optimizer, scheduler)
            global_step += 1
        val = validate(model, val_loader, device)
        log(f'Epoch {epoch + 1}/{epochs} | Train: {tot / max(n, 1):.4f} | Val: {val:.4f} | '
            f'LR_enc: {optimizer.param_groups[0]["lr"]:.6f} | '
            f'LR_mt5: {optimizer.param_groups[1]["lr"]:.6f} | Time: {time.time() - t0:.1f}s')
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == epochs:
            path = os.path.join(args.save_dir, f'phase1_mt5_epoch{epoch + 1}.pth')
            torch.save({'encoder': model.encoder.state_dict(),
                        'pose_norm': model.pose_norm.state_dict(),
                        'mt5': model.mt5.state_dict(), 'epoch': epoch, 'val_loss': val,
                        'optimizer': optimizer.state_dict(), 'global_step': global_step,
                        'run_args': run_args, 'use_lora': False}, path)
            log(f'  saved {path}')


if __name__ == '__main__':
    main()
