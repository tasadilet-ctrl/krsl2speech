"""
Fast standalone check: does Prior-Guided Fusion (PGF) reduce the
cross-clip embedding-collapse cosine metric from scripts/diagnose_phase1.py's
section (B), using the SAME real code path production training uses
(UniSignMT5._make_pgf_hook), before any training on our own data?

Motivation: diagnose_phase1.py found the pose-only encoder collapses to
cosine ~0.976 across genuinely different clips even after full dedicated
pretraining (pose_pretrain_v3, MSE 0.0082), with the collapse independently
confirmed in all four pose sub-groups -- worst in the hands (left 0.994,
right 0.982). PGF injects RGB specifically into those two groups. This
script answers whether that fix shows up immediately -- using the
colleague's already-trained hand_backbone/pgf_hand_fusion weights (see
scripts/convert_friend_checkpoint.py --convert-pgf) -- or whether it needs
real training on our data first. Fast (a handful of clips, no training
loop), so it's meant to run alongside the ongoing baseline/PGF comparison
training jobs, not instead of them.

Usage:
  PYTHONPATH=. python scripts/diagnose_pgf_collapse.py \
      --config configs/config.yaml \
      --pretrained-encoder output/pose_pretrain_v3/pose_pretrain_best.pth \
      --pretrained-pgf output/friend_pgf.pth \
      --hand-crop-root ~/krsl2speech/data/asan_hand_crops \
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
from models.unisign_encoder import KeypointEncoder
from train.train_encoder_mt5 import UniSignMT5, SimpleCollator


def _collapse_cosine(emb, lengths):
    """Same metric as diagnose_phase1.py's section (B): mean-pool over time,
    L2-normalize, max off-diagonal pairwise cosine (higher = more collapsed)."""
    B = emb.size(0)
    m = torch.nn.functional.normalize(
        torch.stack([emb[b, :lengths[b]].mean(0) for b in range(B)]), dim=-1)
    cos = m @ m.T
    return (cos - torch.eye(B, device=emb.device)).max().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--pretrained-encoder', required=True,
                    help='e.g. output/pose_pretrain_v3/pose_pretrain_best.pth')
    ap.add_argument('--pretrained-pgf', required=True,
                    help='Converted colleague checkpoint, see '
                         'scripts/convert_friend_checkpoint.py --convert-pgf')
    ap.add_argument('--hand-crop-root', required=True)
    ap.add_argument('--use-enriched', action='store_true')
    ap.add_argument('--n', type=int, default=8, help='clips to compare')
    ap.add_argument('--pgf-p-samp', type=float, default=1.0,
                    help='Sample every frame for this check (default 0.5 in '
                         'real training trades RGB coverage for speed; here '
                         'we want the clearest possible read).')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    from utils.paths import apply_env_overrides
    cfg = apply_env_overrides(cfg)
    acfg = cfg['paths']['asan']
    input_dim = ENRICHED_DIM() if args.use_enriched else KEYPOINT_DIM

    # ---- Model: same construction as train_encoder_mt5.py's --use-pgf
    # --pretrained-encoder --pretrained-pgf combination ----
    encoder = KeypointEncoder(hidden_dim=cfg['model']['d_model'], input_dim=input_dim)
    print(f"[diagnose-pgf] Loading encoder from {args.pretrained_encoder}")
    ckpt = torch.load(args.pretrained_encoder, map_location='cpu')
    encoder.load_state_dict(ckpt['encoder'] if 'encoder' in ckpt else ckpt)

    model = UniSignMT5(encoder=encoder, lang="Kazakh", use_pgf=True,
                       pgf_p_samp=args.pgf_p_samp).to(device)

    print(f"[diagnose-pgf] Loading PGF submodules from {args.pretrained_pgf}")
    pgf_ckpt = torch.load(args.pretrained_pgf, map_location='cpu')
    pgf_submodules = {
        'hand_backbone': model.hand_backbone,
        'pgf_hand_fusion': model.pgf_hand_fusion,
        'pgf_gate': model.pgf_gate,
        'pgf_keypoint_adapter': model.pgf_keypoint_adapter,
    }
    for key, module in pgf_submodules.items():
        if key in pgf_ckpt:
            module.load_state_dict(pgf_ckpt[key])
            print(f"  {key}: loaded")
        else:
            print(f"  {key}: not in checkpoint, keeping fresh init")

    # ---- Data: first N clips that have real hand-crop data (not every
    # clip in the dataset necessarily has extraction output, though after
    # a full extraction run almost all should) ----
    ds = AsanDataset(root=acfg['root'], sources=acfg.get('sources'),
                     lang=acfg.get('lang', 'kz'), split='train',
                     downsample_every=acfg.get('downsample_every', 1),
                     use_enriched=args.use_enriched, skip_low_quality=True,
                     load_hand_crops=True, hand_crop_root=args.hand_crop_root)
    samples, i = [], 0
    while len(samples) < args.n and i < len(ds):
        s = ds[i]; i += 1
        if s['input_length'] > 1 and s['text'].strip() and s['hand_crops'] is not None:
            samples.append(s)
    if not samples:
        print(f"\n[diagnose-pgf] ERROR: 0 usable clips with hand-crop data "
              f"out of {len(ds)} in the dataset. Check --hand-crop-root "
              f"points at scripts/extract_asan_hand_crops.py's output.")
        sys.exit(1)
    print(f"[diagnose-pgf] Using {len(samples)} clips with real hand-crop data\n")

    collator = SimpleCollator(mt5_tokenizer=model.mt5_tokenizer, max_text_tokens=128)
    batch = collator(samples)
    kps = batch['keypoints'].to(device)
    lengths = batch['input_lengths'].to(device)
    hand_crops = batch['hand_crops'].to(device)
    hand_ref = batch['hand_ref'].to(device)
    hand_valid = batch['hand_valid'].to(device)
    hand_score = batch['hand_score'].to(device)

    # A bit-for-bit IDENTICAL pose-only vs PGF-active result (not just
    # close) usually means _make_pgf_hook's "no valid samples this step"
    # fallback triggered for both hands -- i.e. the extracted crops for
    # these specific frames had too few detected hand keypoints (see
    # scripts/extract_asan_hand_crops.py's min_hand_points), not that the
    # gate's near-zero init washed out a real but tiny contribution. Report
    # the actual valid fraction so that's diagnosed directly, not guessed.
    valid_frac_per_clip = []
    for b in range(len(samples)):
        v = hand_valid[b, :lengths[b]]  # (T_valid, 2) bool, [left, right]
        valid_frac_per_clip.append((v[:, 0].float().mean().item(),
                                    v[:, 1].float().mean().item()))
    print("hand_valid fraction per clip (left, right) -- 0.0 means the hook's "
          "'no valid samples' fallback WILL trigger for that hand/clip:")
    for b, (lf, rf) in enumerate(valid_frac_per_clip):
        print(f"  clip {b}: left={lf:.3f}  right={rf:.3f}")
    # Length-aware (previous version wrongly averaged over padded frames,
    # which are always False, silently dragging this number down).
    valid_time = (torch.arange(hand_valid.size(1), device=device)[None, :]
                 < lengths[:, None])  # (B, T)
    overall_valid = hand_valid[valid_time].float().mean().item()
    print(f"overall hand_valid fraction across the whole batch (length-aware): "
          f"{overall_valid:.3f}\n")

    model.eval()
    with torch.no_grad():
        # Pose-only: no hook, exactly what diagnose_phase1.py's section (B)
        # measures, on this same encoder checkpoint.
        emb_pose_only = model.encoder(kps, input_lengths=lengths)
        cos_pose_only = _collapse_cosine(emb_pose_only, lengths)

        # PGF-active: the real hook, same code path UniSignMT5.forward() uses.
        pgf_hook = model._make_pgf_hook(hand_crops, hand_ref, hand_valid,
                                        hand_score, lengths)
        emb_pgf = model.encoder(kps, input_lengths=lengths, hand_fusion_fn=pgf_hook)
        cos_pgf = _collapse_cosine(emb_pgf, lengths)

    # A cosine delta rounded to 4 decimals can hide a real-but-tiny effect
    # (the gate blends in ~0.25% RGB at init by design) behind display
    # precision. Report the raw embedding difference directly so "PGF had
    # literally zero effect" (a bug -- the hook silently not firing) and
    # "PGF had a real but currently-too-small-to-move-this-coarse-a-metric
    # effect" (expected, needs training) are never confused for each other.
    diff = (emb_pgf - emb_pose_only)
    diff_l2 = diff.norm().item()
    diff_max = diff.abs().max().item()
    print(f"raw embedding difference (pose-only vs PGF-active): "
          f"L2 norm={diff_l2:.8f}  max abs diff={diff_max:.8f}")
    if diff_l2 == 0.0:
        print("  L2 norm is EXACTLY zero -- the hook had literally no numeric "
              "effect on the output. That's NOT consistent with a small-but-"
              "real gate blend (sigmoid(-6) is representable in float32, not "
              "zero) -- this points at a real bug (hook not actually being "
              "invoked, or the scatter not landing), not just 'needs training'.")
    else:
        print(f"  Nonzero -- PGF IS measurably changing the output ({diff_l2:.8f}), "
              f"just not by enough to move the 4-decimal cosine metric above. "
              f"Consistent with the gate's near-zero init working as designed.")

    print(f"\n=== collapse metric on {len(samples)} clips (higher = more collapsed) ===")
    print(f"pose-only (no PGF):  max offdiag cosine = {cos_pose_only:.8f}")
    print(f"PGF-active:          max offdiag cosine = {cos_pgf:.8f}")
    delta = cos_pgf - cos_pose_only
    if delta < -0.005:
        print(f"\nDROPPED by {-delta:.4f} -- PGF measurably reduces collapse "
              f"on these clips, even before any training on our own data. "
              f"The friend's hand-vision weights transfer useful signal.")
    elif delta > 0.005:
        print(f"\nROSE by {delta:.4f} -- PGF does not help (or actively hurts) "
              f"at this untrained-on-our-data starting point. Not necessarily "
              f"bad news -- the gate starts near-zero by design (pose-dominant "
              f"until it learns to trust RGB), so this may simply require "
              f"real training, which is exactly what's running right now.")
    else:
        print(f"\n~UNCHANGED -- consistent with the gate's near-zero init "
              f"(sigmoid(-6)~=0.0025): at this point PGF is barely blending "
              f"in RGB yet regardless of whether it would help, by design.")


if __name__ == '__main__':
    main()
