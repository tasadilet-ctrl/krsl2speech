"""
Convert a colleague's best_checkpoint.pth (see scripts/
inspect_friend_checkpoint.py's reverse-engineering notes) into this repo's
own checkpoint format, for the parts we can use with full confidence:
the pose encoder and mT5 decoder, plus (with --convert-pgf) the RGB
Prior-Guided Fusion weights now that the paper (arXiv:2501.15187 Sec 3.3 +
Appendix A.3) has confirmed the actual mechanism, not just module shapes
-- see models/pgf_fusion.py for the real, wired-up reimplementation.

What this converts with full confidence (verified as an exact structural
match against this repo's own classes in inspect_friend_checkpoint.py --
343/343 and 284/284 keys+shapes):
  - The pose encoder (343 keys) -- bit-for-bit models.unisign_encoder.
    KeypointEncoder(hidden_dim=768, input_dim=282). Standard (non-enriched)
    282-dim input -- load the result WITHOUT --use-enriched.
  - mT5 (284 keys, under 'mt5_model.' in their checkpoint) -- standard
    google/mt5-base, just needs the prefix stripped to match this repo's
    'mt5.' naming.

All were stored in bfloat16; converts to float32 to match this repo's
training precision.

--convert-pgf additionally converts rgb_support_backbone/rgb_proj (->
models.pgf_fusion.HandBackbone), fusion_pose_rgb_DA + the sibling
fusion_pose_rgb_linear (-> models.pgf_fusion.DeformablePoseRGBAttention),
and fusion_gate (-> models.pgf_fusion.FusionGate), via
load_state_dict(strict=False). Two of our modules have NO corresponding
source key and are EXPECTED to show up as missing every time (judgment
call #4, models/pgf_fusion.py's module docstring): rgb_adapter_down
(inside DeformablePoseRGBAttention) and pgf_keypoint_adapter (built by
UniSignMT5 itself, not converted here at all -- it has no equivalent in
the colleague's architecture since their 256<->768 reconciliation seam is
different from ours). Both stay at their fresh, randomly-initialized
weights after conversion; this is correct, not a bug to chase.

Output is loadable via train_encoder_mt5.py's --resume (encoder + mt5 +
pose_norm/PGF all optional -- see the soft-fail paths in
_load_resume_checkpoint for exactly this scenario: an external checkpoint
with no pose_norm/optimizer/run_args, and (without --convert-pgf) no PGF
weights at all). --resume's architecture-mismatch guard is a no-op here
since there's no run_args to compare against, so double check --use-
enriched is NOT passed (their encoder is standard 282-dim), --use-pgf is
only passed if this WAS converted with --convert-pgf, and any other flags
are otherwise compatible before resuming a full run with this.

Usage:
  PYTHONPATH=. python scripts/convert_friend_checkpoint.py \
      --input ~/Downloads/best_checkpoint.pth \
      --output output/friend_encoder_mt5.pth \
      --reported-wer 0.90 --convert-pgf
"""
import argparse

import torch

from models.pgf_fusion import HandBackbone, DeformablePoseRGBAttention, FusionGate


def _convert_pgf(sd):
    """
    Returns (hand_backbone_sd, pgf_hand_fusion_sd, pgf_gate_sd, reports)
    where reports is a list of (name, missing, unexpected) tuples from
    each load_state_dict(strict=False) call, for the caller to print.
    """
    reports = []

    # HandBackbone's own attribute names (rgb_support_backbone, rgb_proj)
    # match the checkpoint's top-level key names exactly -- no remap needed.
    hb_prefixes = ('rgb_support_backbone', 'rgb_proj')
    hb_sd = {k: v.float() for k, v in sd.items() if k.startswith(hb_prefixes)}
    hb = HandBackbone(out_channels=256, pretrained=False)
    missing, unexpected = hb.load_state_dict(hb_sd, strict=False)
    reports.append(('HandBackbone', missing, unexpected))

    # DeformablePoseRGBAttention: fusion_pose_rgb_DA.* keys strip their
    # prefix (our class's to_offsets/to_q/.../cross_attn are direct
    # attributes, same names as the checkpoint's DeformableCrossAttention);
    # fusion_pose_rgb_linear is a SEPARATE top-level key in the checkpoint
    # (sibling of fusion_pose_rgb_DA, not nested under it) but is also a
    # direct attribute on our class under the identical name, so no remap
    # needed there either.
    da_sd = {}
    for k, v in sd.items():
        if k.startswith('fusion_pose_rgb_DA.'):
            da_sd[k[len('fusion_pose_rgb_DA.'):]] = v.float()
        elif k.startswith('fusion_pose_rgb_linear.'):
            da_sd[k] = v.float()
    da = DeformablePoseRGBAttention(embed_dim=256, adapter_dim=32, num_heads=8)
    missing, unexpected = da.load_state_dict(da_sd, strict=False)
    reports.append(('DeformablePoseRGBAttention', missing, unexpected))

    # FusionGate wraps its Sequential in self.net; checkpoint's fusion_gate
    # keys are top-level Sequential indices (fusion_gate.0.*, fusion_gate.2.*)
    # -- remap the prefix to net.
    gate_sd = {}
    for k, v in sd.items():
        if k.startswith('fusion_gate.'):
            gate_sd['net.' + k[len('fusion_gate.'):]] = v.float()
    gate = FusionGate(embed_dim=256)
    missing, unexpected = gate.load_state_dict(gate_sd, strict=False)
    reports.append(('FusionGate', missing, unexpected))

    return hb.state_dict(), da.state_dict(), gate.state_dict(), reports


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--reported-wer', type=float, default=None,
                    help="Informational only (not verified by us) -- "
                         "whatever WER your colleague reported for this "
                         "checkpoint, stored in the output for reference.")
    ap.add_argument('--convert-pgf', action='store_true',
                    help='Also convert the RGB Prior-Guided Fusion weights '
                         '(rgb_support_backbone/rgb_proj/fusion_pose_rgb_DA/'
                         'fusion_gate/fusion_pose_rgb_linear) into '
                         'models/pgf_fusion.py\'s classes. Load the result '
                         'with --resume --use-pgf.')
    args = ap.parse_args()

    ckpt = torch.load(args.input, map_location='cpu', weights_only=False)
    sd = ckpt['model']

    encoder_prefixes = ('proj_linear', 'gcn_modules', 'fusion_gcn_modules',
                        'pose_proj', 'part_para')
    encoder_sd = {k: v.float() for k, v in sd.items() if k.startswith(encoder_prefixes)}
    mt5_sd = {k[len('mt5_model.'):]: v.float()
             for k, v in sd.items() if k.startswith('mt5_model.')}

    print(f"encoder: {len(encoder_sd)} keys (expect 343)")
    print(f"mt5: {len(mt5_sd)} keys (expect 284)")
    if len(encoder_sd) != 343 or len(mt5_sd) != 284:
        print("[WARN] unexpected key count -- the source checkpoint may "
              "not match the architecture this converter assumes. Double "
              "check with scripts/inspect_friend_checkpoint.py before "
              "trusting the output.")

    out_ckpt = {
        'encoder': encoder_sd,
        'mt5': mt5_sd,
        # No 'pose_norm' key -- their architecture has no equivalent bridge
        # layer; train_encoder_mt5.py's --resume now handles this (starts
        # pose_norm from default init instead of raising).
        # No 'optimizer'/'run_args' -- --resume already degrades gracefully
        # for both (soft-resume: fresh LR schedule, no architecture check).
        'epoch': -1,
        'val_loss': None,
        'wer': args.reported_wer,
        'use_lora': False,
        'source': 'converted from a colleague\'s best_checkpoint.pth -- '
                  'see scripts/inspect_friend_checkpoint.py',
    }

    if args.convert_pgf:
        hb_sd, da_sd, gate_sd, reports = _convert_pgf(sd)
        out_ckpt['hand_backbone'] = hb_sd
        out_ckpt['pgf_hand_fusion'] = da_sd
        out_ckpt['pgf_gate'] = gate_sd
        print("\n--convert-pgf key remapping report:")
        for name, missing, unexpected in reports:
            print(f"  {name}: {len(missing)} missing, {len(unexpected)} unexpected")
            if missing:
                print(f"    missing (expected: rgb_adapter_down.* has no "
                      f"source key): {missing}")
            if unexpected:
                print(f"    unexpected (unmapped source keys -- investigate!): "
                      f"{unexpected}")
        print("  Note: pgf_keypoint_adapter is NOT converted (no equivalent "
              "in the colleague's architecture) -- stays randomly "
              "initialized; this is expected, not an error.")

    torch.save(out_ckpt, args.output)
    print(f"\nSaved to {args.output}")
    print("Load with --resume (NOT --use-enriched -- this encoder is "
         "standard 282-dim)." + (" Add --use-pgf to load the converted "
         "PGF weights." if args.convert_pgf else ""))


if __name__ == '__main__':
    main()
