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
    ap.add_argument('--pretrained-unisign', default=None,
                    help='Raw Uni-Sign/CSL checkpoint (never seen KRSL data)')
    ap.add_argument('--pretrained-encoder', default=None,
                    help="Our fine-tuned encoder checkpoint (e.g. a "
                         "phase1_mt5_*.pth from a real training run) -- lets "
                         "you check whether actual KRSL fine-tuning has "
                         "changed the (B) embedding-collapse cosine at all, "
                         "vs. testing the cold CSL-pretrained-only encoder.")
    ap.add_argument('--use-enriched', action='store_true')
    ap.add_argument('--n', type=int, default=4, help='clips to memorize')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42,
                    help='Fixed seed so section (C) is reproducible across '
                         'runs -- (A)/(B)/(B2) run under eval()+no_grad() '
                         'and are already deterministic, but mT5 has active '
                         'dropout during the (C) training loop.')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    from utils.paths import apply_env_overrides
    cfg = apply_env_overrides(cfg)
    acfg = cfg['paths']['asan']
    input_dim = ENRICHED_DIM() if args.use_enriched else KEYPOINT_DIM

    # ---- Model ----
    encoder = KeypointEncoder(hidden_dim=cfg['model']['d_model'],
                              input_dim=input_dim)
    if args.pretrained_unisign:
        load_unisign_weights(encoder, args.pretrained_unisign)
    if args.pretrained_encoder:
        print(f"\n[diagnose] Loading fine-tuned encoder from "
              f"{args.pretrained_encoder}")
        ckpt = torch.load(args.pretrained_encoder, map_location='cpu')
        encoder.load_state_dict(ckpt['encoder'] if 'encoder' in ckpt else ckpt)
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
    if not samples:
        print(f"\n[diagnose] ERROR: 0 usable clips out of {len(ds)} in the "
              f"dataset. Likely cause: 'paths.asan.root' in {args.config} "
              f"({acfg['root']}) doesn't exist on this machine — check for "
              f"'WARNING: missing ...' lines above, or set ASAN_ROOT.")
        sys.exit(1)
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

    # ---- (B2) ablation: does the additive part_para / pose_proj bias
    #      dominate the final embedding, drowning out the per-clip signal
    #      from the actual pose features? Both are added IDENTICALLY to
    #      every sample and frame, right before the final projection. ----
    print("\n--- (B2) part_para / pose_proj bias ablation ---")
    captured = {}
    def _capture_pre_proj(module, mod_args):
        captured['pre_proj'] = mod_args[0].detach().clone()  # (B, T, 1024)
    hook = model.encoder.pose_proj.register_forward_pre_hook(_capture_pre_proj)
    with torch.no_grad():
        _ = model.encoder(kps)
    hook.remove()
    pre_proj = captured['pre_proj']

    # ---- (B3) per-group breakdown, before concatenation ----
    # Hands load their CSL-pretrained weights exactly; body/face are mostly
    # reinitialized (see load_unisign_weights). If the aggregate 1024-dim
    # collapse is hiding a split -- hands actually discriminative, body/face
    # collapsed -- concatenating them still gives ~1.0 aggregate cosine if
    # body/face's near-constant output dominates in raw magnitude.
    print("\n--- (B3) per-group embedding collapse (pre-concatenation) ---")
    for gi, gname in enumerate(KeypointEncoder.MODES):
        g = pre_proj[:, :, gi * 256:(gi + 1) * 256]
        gm = torch.nn.functional.normalize(
            torch.stack([g[b, :lengths[b]].mean(0) for b in range(B)]), dim=-1)
        goffdiag = (gm @ gm.T - torch.eye(B, device=device)).max().item()
        print(f"  {gname:10s}: pairwise cosine max offdiag={goffdiag:.4f}")

    part_para_norm = model.encoder.part_para.norm().item()
    per_clip_mean = torch.stack([pre_proj[b, :lengths[b]].mean(0) for b in range(B)])
    signal_spread = (per_clip_mean - per_clip_mean.mean(0, keepdim=True)).norm(dim=-1).mean().item()
    print(f"part_para norm: {part_para_norm:.4f}  |  typical per-clip signal "
          f"spread pre-pose_proj: {signal_spread:.4f} "
          f"(part_para >> spread → the additive offset swamps the per-clip signal)")

    saved_part_para = model.encoder.part_para.data.clone()
    saved_bias = (model.encoder.pose_proj.bias.data.clone()
                  if model.encoder.pose_proj.bias is not None else None)
    model.encoder.part_para.data.zero_()
    if saved_bias is not None:
        model.encoder.pose_proj.bias.data.zero_()
    with torch.no_grad():
        emb_ablated = model.encoder(kps)
    model.encoder.part_para.data.copy_(saved_part_para)
    if saved_bias is not None:
        model.encoder.pose_proj.bias.data.copy_(saved_bias)

    m2 = torch.nn.functional.normalize(
        torch.stack([emb_ablated[b, :lengths[b]].mean(0) for b in range(B)]), dim=-1)
    cos2 = m2 @ m2.T
    before = (cos - torch.eye(B, device=device)).max().item()
    after = (cos2 - torch.eye(B, device=device)).max().item()
    print(f"pairwise clip cosine WITHOUT part_para/bias: max offdiag={after:.4f} "
          f"(was {before:.4f} with them) — a big drop confirms these additive "
          f"terms were swamping the signal; still ≈1.0 here → the collapse "
          f"happens earlier (GCN/BatchNorm layers), not at this final step")

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
