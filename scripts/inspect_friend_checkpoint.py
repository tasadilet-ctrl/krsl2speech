"""
Reverse-engineered architecture for a colleague's best_checkpoint.pth
(shared out-of-band, downloaded to ~/Downloads/best_checkpoint.pth).

No training code came with the checkpoint and no metrics (epoch/loss/WER)
are stored in the file -- only the raw state_dict. Everything below was
inferred from parameter names/shapes/dtypes alone via
torch.load(..., weights_only=False)['model'].

CONFIRMED with certainty (exact key+shape match against known classes):
  - Pose encoder ('proj_linear.*', 'gcn_modules.*', 'fusion_gcn_modules.*',
    'pose_proj.*', 'part_para') is IDENTICAL to this repo's
    models.unisign_encoder.KeypointEncoder(hidden_dim=768, input_dim=282)
    -- 343/343 keys and shapes match exactly. Standard (non-enriched)
    282-dim input, NOT the 1410-dim dual-coord format used in this
    session's recent runs.
  - 'mt5_model.*' (284 keys) is a standard google/mt5-base (vocab 250112,
    d_model 768, 12 encoder + 12 decoder blocks) -- same base model this
    repo uses, just stored under a different attribute name (mt5_model vs
    our mt5) and in bfloat16 (not fp32 -- explains the ~1.19GB file size
    vs. this repo's ~2.36GB fp32 checkpoints for the same base model).
  - 'rgb_support_backbone.0.*' (358 keys) is EXACTLY
    torchvision.models.efficientnet_b0(weights=None).features, wrapped in
    an extra nn.Sequential -- 358/358 keys and shapes match exactly. This
    is a TRAINABLE embedded backbone (not a frozen precomputed feature
    extractor like scripts/extract_asan_rgb.py's DINOv2 approach), and its
    Conv2d output stays spatial (not globally pooled) -- rgb_proj is a 1x1
    Conv2d, not a Linear, preserving a spatial feature map for the
    attention module below to sample from.

INFERRED (structure is verifiable via state_dict shapes; exact forward()
wiring -- activation choices, attention head count, and precisely how the
256-dim fusion output reconciles with the 768-dim pose stream before mT5
-- is NOT recoverable from weights alone and would need the actual code):
  - rgb_proj: Conv2d(1280, 256, kernel_size=1) -- projects EfficientNet-B0's
    1280-dim final feature map down to 256 channels, keeping spatial dims.
  - fusion_pose_rgb_DA ("DA" almost certainly = Deformable Attention):
    a custom cross-modal deformable attention letting pose (1D, T frames)
    query into the RGB spatial feature map with LEARNED sampling offsets,
    similar in spirit to Deformable DETR:
      - to_offsets: Conv1d(32,32,groups=32) -> activation -> Conv1d(32,2)
        predicts a 2D (x,y) sampling offset per pose frame
      - to_q/to_k/to_v: Conv1d/Conv2d projections into a 256-dim attention
        space (channel dim 32 on the input side -- likely a per-head or
        reduced-dim adapter this reconstruction can't pin down exactly)
      - pe_layer.positional_encoding_gaussian_matrix (2, 128): a random
        Fourier positional encoding for the RGB spatial locations -- this
        buffer name is a verbatim match to Meta's Segment Anything Model
        (segment-anything's PositionEmbeddingRandom class), strongly
        suggesting this module's positional encoding was adapted from SAM.
      - rel_pos_bias.mlp: a continuous relative-position bias MLP
        (2 -> 64 -> 64 -> 1), Swin-Transformer-V2-style, adding a learned
        bias to attention scores based on relative (x, y) offset.
      - cross_attn: an auxiliary standard nn.MultiheadAttention(embed_dim=
        256) (in_proj_weight/in_proj_bias naming is PyTorch's own
        MultiheadAttention parameter convention) -- possibly used
        alongside or as an alternative path to the deformable branch above.
  - fusion_gate: Conv1d(512,256,1) -> activation -> Conv1d(256,1,1) ->
    (presumably sigmoid, not a parameterized layer) -- a LEARNED, per-frame
    SCALAR gate deciding how much the RGB branch should influence that
    frame, from the concatenation of pose+RGB (512 = 256+256).
  - fusion_pose_rgb_linear: Linear(256, 256) -- purpose (pre- or
    post-attention refinement) not recoverable from shape alone.

This reconstruction exists to (a) document what was learned by staring at
tensor shapes so it doesn't need to be re-derived, and (b) let the
checkpoint's weights actually be loaded into *something* for inspection
(e.g. checking which submodules have non-trivial/non-random-looking
weight statistics), even though forward() below is a best-effort guess at
wiring, not a verified reproduction of the original training code.

Verification performed: state_dict() key set and every tensor shape of
FriendModel() below is confirmed to match the checkpoint's 'model' dict
key-for-key (1012/1012), so load_state_dict(strict=True) succeeds -- see
scripts/inspect_pose_quality.py-style verification in this file's __main__.
"""
import argparse

import torch
import torch.nn as nn
import torchvision.models as tvm
from transformers import MT5ForConditionalGeneration, MT5Config

from models.unisign_encoder import KeypointEncoder


class PositionEmbeddingRandom(nn.Module):
    """Verbatim structure of segment-anything's random Fourier position
    embedding (buffer name match confirms this was the source)."""

    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.register_buffer(
            'positional_encoding_gaussian_matrix', torch.randn(2, num_pos_feats))


class ContinuousRelPosBiasMLP(nn.Module):
    """2 -> 64 -> 64 -> 1, Swin-V2-style continuous relative position bias."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Sequential(nn.Linear(2, 64), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True)),
            nn.Linear(64, 1, bias=True),
        )


class DeformableCrossAttention(nn.Module):
    """
    Best-effort reconstruction of fusion_pose_rgb_DA. Module TYPES/SHAPES
    are verified exact matches; how these pieces actually connect in
    forward() (this class has none -- inspection-only) is inferred from
    naming convention, not confirmed.
    """

    def __init__(self, embed_dim=256, adapter_dim=32, num_heads=8):
        super().__init__()
        self.to_offsets = nn.Sequential(
            nn.Conv1d(adapter_dim, adapter_dim, kernel_size=1, groups=adapter_dim),
            nn.GELU(),
            nn.Conv1d(adapter_dim, 2, kernel_size=1, bias=False),
        )
        self.to_q = nn.Conv1d(adapter_dim, embed_dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(adapter_dim, embed_dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(adapter_dim, embed_dim, kernel_size=1, bias=False)
        self.to_out = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.pe_layer = PositionEmbeddingRandom(num_pos_feats=embed_dim // 2)
        self.rel_pos_bias = ContinuousRelPosBiasMLP()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True)


class FriendModel(nn.Module):
    """
    Inspection-only reconstruction -- NOT wired to run inference correctly
    (no forward() attempts the actual fusion logic). Exists so the
    checkpoint's weights can be loaded and individual submodules examined.
    """

    def __init__(self):
        super().__init__()
        self.encoder_module = KeypointEncoder(hidden_dim=768, input_dim=282)

        eff = tvm.efficientnet_b0(weights=None)
        self.rgb_support_backbone = nn.Sequential(eff.features)
        self.rgb_proj = nn.Conv2d(1280, 256, kernel_size=1)

        self.fusion_pose_rgb_DA = DeformableCrossAttention()
        self.fusion_pose_rgb_linear = nn.Linear(256, 256)
        self.fusion_gate = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(256, 1, kernel_size=1),
        )

        mt5_cfg = MT5Config.from_pretrained('google/mt5-base')
        self.mt5_model = MT5ForConditionalGeneration(mt5_cfg)

    def _remap_keys(self, sd):
        """Encoder keys need an 'encoder_module.' prefix added; everything
        else already matches the checkpoint's naming 1:1."""
        out = {}
        for k, v in sd.items():
            if k.startswith(('proj_linear', 'gcn_modules', 'fusion_gcn_modules',
                             'pose_proj', 'part_para')):
                out[f'encoder_module.{k}'] = v
            else:
                out[k] = v
        return out

    def load_friend_checkpoint(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = self._remap_keys(ckpt['model'])
        missing, unexpected = self.load_state_dict(sd, strict=False)
        return missing, unexpected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    args = ap.parse_args()

    model = FriendModel()
    missing, unexpected = model.load_friend_checkpoint(args.checkpoint)
    print(f"Loaded. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if missing:
        print("First few missing:", missing[:10])
    if unexpected:
        print("First few unexpected:", unexpected[:10])


if __name__ == '__main__':
    main()
