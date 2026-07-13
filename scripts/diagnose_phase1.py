"""
Phase-1 pipeline diagnostic. Answers, in ~10 minutes on one GPU, which of
these is true:

  (A) DATA BUG — clips are indistinguishable / mostly zeros / constant
  (B) ENCODER COLLAPSE — features differ but pose embeddings don't
  (C) OPTIMIZATION — features and embeddings are fine and a tiny
      fixed-LR memorization run drives CE to ~0  → pipeline is sound,
      the full run needs capacity/schedule, not debugging

Usage:
  PYTHONPATH=. python scripts/diagnose_phase1.py \
      --config configs/config.yaml \
      --pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth \
      --use-enriched
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch

from data.asan_dataset import AsanDataset
from data.utils import ENRICHED_DIM, KEYPOINT_DIM
from models.unisign_encoder import KeypointEncoder, load_unisign_weights
from train.train_encoder_mt5 import UniSignMT5, SimpleCollator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--pretrained-unisign', default=None)
    ap.add_argument('--use-enriched', action='store_true')
    ap.add_argument('--n', type=int, default=4, help='clips to memorize')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-4)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    acfg = cfg['paths']['asan']
    input_dim = ENRICHED_DIM() if args.use_enriched else KEYPOINT_DIM

    # ---- Model ----
    encoder = KeypointEncoder(hidden_dim=cfg['model']['d_model'],
                              input_dim=input_dim)
    if args.pretrained_unisign:
        load_unisign_weights(encoder, args.pretrained_unisign)
    model = UniSignMT5(encoder=encoder, lang="Kazakh").to(device)

    # ---- Data: first N usable clips ----
    ds = AsanDataset(root=acfg['root'], sources=acfg.get('sources'),
                     lang=acfg.get('lang', 'kz'), split='train',
                     downsample_every=acfg.get('downsample_every', 1),
                     use_enriched=args.use_enriched,
                     skip_low_quality=True)
    samples, i = [], 0
    while len(samples) < args.n and i < len(ds):
        s = ds[i]; i += 1
        if s['input_length'] > 1 and s['text'].strip():
            samples.append(s)
    collator = SimpleCollator(mt5_tokenizer=model.mt5_tokenizer,
                              max_text_tokens=128)
    batch = collator(samples)
    kps = batch['keypoints'].to(device)
    label_ids = batch['label_ids'].to(device)
    label_attn = batch['label_attn_mask'].to(device)
    lengths = batch['input_lengths'].to(device)
    B, T, D = kps.shape
    print(f"\n=== batch: {B} clips, T={T}, D={D} ===")

    # ---- (A) Feature sanity ----
    print("\n--- (A) input features ---")
    zero_frac = (kps == 0).float().mean().item()
    print(f"zero fraction: {zero_frac:.3f}  (mostly-padding is fine; ~1.0 is a data bug)")
    for b in range(B):
        v = kps[b, :lengths[b], :KEYPOINT_DIM]
        print(f"clip {b}: offset std={v.std().item():.4f}  "
              f"temporal std={v.std(dim=0).mean().item():.4f}  "
              f"len={lengths[b].item()}")
    flat = torch.stack([kps[b, :lengths[b]].mean(0) for b in range(B)])
    dists = torch.cdist(flat, flat)
    print(f"pairwise clip distance (mean pose): min offdiag="
          f"{(dists + torch.eye(B, device=device) * 1e9).min().item():.4f} "
          f"(≈0 → clips indistinguishable → DATA BUG)")

    # ---- (B) Encoder output sanity ----
    print("\n--- (B) encoder embeddings ---")
    model.eval()
    with torch.no_grad():
        emb = model.encoder(kps)  # (B, T, 768)
    for b in range(B):
        e = emb[b, :lengths[b]]
        print(f"clip {b}: emb std={e.std().item():.4f}  "
              f"temporal std={e.std(dim=0).mean().item():.4f} "
              f"(≈0 temporal std → encoder collapses time → ENCODER BUG)")
    m = torch.nn.functional.normalize(
        torch.stack([emb[b, :lengths[b]].mean(0) for b in range(B)]), dim=-1)
    cos = m @ m.T
    print(f"pairwise clip cosine: max offdiag="
          f"{(cos - torch.eye(B, device=device)).max().item():.4f} "
          f"(≈1.0 → embeddings identical across clips → ENCODER BUG)")

    # ---- (C) Fixed-LR memorization ----
    print(f"\n--- (C) memorize {B} clips, {args.steps} steps @ lr={args.lr} ---")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for step in range(args.steps):
        loss, _, _ = model(kps, label_ids, label_attn, input_lengths=lengths)
        opt.zero_grad(); loss.backward()
        if step == 0:
            g_enc = sum(p.grad.abs().sum().item()
                        for p in model.encoder.parameters() if p.grad is not None)
            print(f"step 0: CE={loss.item():.4f}  encoder |grad|={g_enc:.2e} "
                  f"(0 → gradients not reaching encoder → GRAPH BUG)")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}: CE={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        hyps = model.generate(kps, input_lengths=lengths)
    print("\n--- after memorization ---")
    for b in range(B):
        print(f"REF: {batch['texts'][b][:70]}")
        print(f"HYP: {hyps[b][:70]}\n")
    print("VERDICT: CE < 0.5 and HYP≈REF → pipeline sound (scale/schedule "
          "issue). CE stuck > 2 → structural bug; send this full output back.")


if __name__ == '__main__':
    main()
