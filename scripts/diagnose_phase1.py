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
    ap.add_argument('--encoder-lr', type=float, default=None,
                    help='Separate LR for the encoder in section (C)/(E) '
                         '(default: same as --lr). Test whether a much '
                         'higher encoder LR than the real trainer\'s '
                         'base_lr/10 default actually un-collapses the '
                         'embeddings (see (E) below), before committing to '
                         'a full slow training run.')
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

    # ---- (D) Decoder cross-attention: mean-pooling over T (used by (B)/(B3)
    #      above) is NOT what the real model does -- UniSignMT5.forward feeds
    #      mT5 the FULL per-frame sequence and lets cross-attention pick out
    #      whatever's relevant. A collapsed mean-pool doesn't prove per-frame
    #      information is unusable; this checks what the decoder actually
    #      does with it, on the untouched loaded checkpoint (run before (C)
    #      mutates the weights via memorization). ----
    print("\n--- (D) decoder cross-attention: prefix vs. pose frames ---")
    model.eval()
    with torch.no_grad():
        pose_emb = model.pose_norm(model.encoder(kps, input_lengths=lengths))
        prefix_embeds = model.mt5.shared(model.prefix_ids.unsqueeze(0).expand(B, -1))
        prefix_attn = model.prefix_attn.unsqueeze(0).expand(B, -1)
        pose_mask = model._pose_mask(kps, lengths)
        inputs_embeds = torch.cat([prefix_embeds, pose_emb], dim=1)
        attn_mask = torch.cat([prefix_attn, pose_mask], dim=1)
        out = model.mt5(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                        labels=label_ids, output_attentions=True, return_dict=True)
    P = prefix_embeds.size(1)
    last_layer_attn = out.cross_attentions[-1].mean(dim=1)  # avg over heads -> (B, dec_len, P+T)
    for b in range(B):
        valid_dec = int((label_ids[b] != -100).sum().item())
        a = last_layer_attn[b, :valid_dec]  # (dec_len, P+T)
        prefix_mass = a[:, :P].sum(-1).mean().item()
        pose_mass = a[:, P:P + int(lengths[b])].sum(-1).mean().item()
        print(f"clip {b}: attention mass on prefix={prefix_mass:.3f}  "
              f"pose={pose_mass:.3f} (pose≈0 → decoder ignores the video "
              f"entirely, regardless of what the encoder produces)")

    # ---- (F) BatchNorm running-stats reset ablation ----
    # Every GCN_unit's nn.BatchNorm2d tracks running_mean/running_var via a
    # slow momentum update (default 0.1 -> ~90% weight on OLD stats per
    # batch). These were calibrated on CSL data during Uni-Sign pretraining;
    # 13 real KRSL epochs may not have been enough to shift them much. This
    # hard-resets them to the untrained default (mean=0, var=1) -- NO
    # gradient steps, a single forward pass -- to test whether stale BN
    # calibration alone is contributing to the collapse, independent of
    # weight training.
    print("\n--- (F) BatchNorm running-stats reset ablation ---")
    bn_modules = [m for m in model.encoder.modules()
                  if isinstance(m, torch.nn.BatchNorm2d)]
    saved_bn_state = [(m.running_mean.clone(), m.running_var.clone(),
                       m.num_batches_tracked.clone()) for m in bn_modules]
    for m in bn_modules:
        m.running_mean.zero_()
        m.running_var.fill_(1.0)
        m.num_batches_tracked.zero_()
    model.eval()
    with torch.no_grad():
        emb_bn_reset = model.encoder(kps, input_lengths=lengths)
    for m, (rm, rv, nb) in zip(bn_modules, saved_bn_state):
        m.running_mean.copy_(rm)
        m.running_var.copy_(rv)
        m.num_batches_tracked.copy_(nb)
    m4 = torch.nn.functional.normalize(
        torch.stack([emb_bn_reset[b, :lengths[b]].mean(0) for b in range(B)]), dim=-1)
    bn_offdiag = (m4 @ m4.T - torch.eye(B, device=device)).max().item()
    print(f"{len(bn_modules)} BatchNorm2d modules reset | pairwise clip "
          f"cosine: max offdiag={bn_offdiag:.4f} "
          f"(was {(cos - torch.eye(B, device=device)).max().item():.4f} with "
          f"the loaded running stats -- a real drop implicates stale BN "
          f"calibration; unchanged rules it out too)")

    # ---- (C) Fixed-LR memorization ----
    # Differential LR (encoder vs. rest) so --encoder-lr can test whether
    # base_lr/10 (the real trainer's default) is too conservative for what
    # the encoder needs to learn, without waiting on a full training run.
    enc_lr = args.encoder_lr if args.encoder_lr is not None else args.lr
    print(f"\n--- (C) memorize {B} clips, {args.steps} steps @ "
          f"lr={args.lr} (encoder_lr={enc_lr}) ---")
    model.train()
    encoder_params = list(model.encoder.parameters())
    encoder_param_ids = {id(p) for p in encoder_params}
    other_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
    opt = torch.optim.AdamW([
        {'params': encoder_params, 'lr': enc_lr},
        {'params': other_params, 'lr': args.lr},
    ])
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

    # ---- (E) Re-check embedding collapse AFTER memorization ----
    # Directly tests whether this encoder_lr let the embeddings actually
    # become more distinguishable across clips, vs. section (C)'s CE/HYP
    # check alone (which can crash to ~0 via decoder-side memorization of a
    # handful of sentences without the encoder discriminating anything --
    # see (D)'s cross-attention finding).
    print(f"\n--- (E) embedding collapse after memorization (encoder_lr={enc_lr}) ---")
    model.eval()
    with torch.no_grad():
        emb_post = model.encoder(kps, input_lengths=lengths)
    m3 = torch.nn.functional.normalize(
        torch.stack([emb_post[b, :lengths[b]].mean(0) for b in range(B)]), dim=-1)
    post_offdiag = (m3 @ m3.T - torch.eye(B, device=device)).max().item()
    print(f"pairwise clip cosine after memorization: max offdiag={post_offdiag:.4f} "
          f"(was {(cos - torch.eye(B, device=device)).max().item():.4f} before "
          f"-- a real drop means this encoder_lr helps the encoder actually "
          f"discriminate; unchanged means raising it alone isn't the fix)")


if __name__ == '__main__':
    main()
