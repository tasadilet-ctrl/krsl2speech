"""
Prior-Guided Fusion (PGF) -- Uni-Sign's real RGB-pose fusion mechanism
(arXiv:2501.15187, ICLR 2025, Section 3.3 + Appendix A.3).

This replaces the earlier placeholder whole-frame RGB fusion (frozen
DINOv2, simple concat+project) with the paper's actual hand-crop
deformable-attention design, confirmed both by reading the paper directly
and by reverse-engineering a colleague's shared checkpoint (see
scripts/inspect_friend_checkpoint.py for the full key/shape analysis).

Architecture, confirmed:
  - RGB is HAND-ONLY: each hand cropped via that hand's own keypoints,
    resized 112x112, encoded by EfficientNet-B0 (ImageNet-pretrained).
    Face/body never touch RGB.
  - Two-stage attention per hand, per sampled frame: (1) standard
    multi-head cross-attention, pose query attends to the full RGB spatial
    map ("global RGB"); (2) deformable attention -- the keypoint's own
    (x,y) initializes a reference point, a learned offset refines it,
    F.grid_sample bilinearly samples the RGB feature map there.
  - Gate (paper Eq. 3-4): g = Gate([F_pose, F_fused]),
    F_final = (1-g)*F_pose + g*F_fused, gate starts near g=0 (pose-only at
    the start of RGB fine-tuning, learns to trust RGB gradually).
  - Score-aware sampling (Appendix A.3 Algorithm 1): only a random subset
    of frames per clip get RGB each step, weighted by
    (1 - mean_keypoint_confidence).

Judgment calls made where the paper/checkpoint don't fully pin down the
wiring (each flagged at its point of use, not hidden):
  1. `to_q`/`to_k`/`to_v` operate in a 32-dim "adapter" space, not the
     256-dim rgb_proj/pose output -- two small bridging modules
     (`pgf_keypoint_adapter` here is built by the caller, `rgb_adapter_down`
     below) have NO corresponding checkpoint key and stay randomly
     initialized even after loading a colleague's weights.
  2. Gate init: true g=0 needs an infinite pre-sigmoid logit -- zero-init
     the last conv's weight, large negative bias (-6.0) instead, so it
     starts numerically pose-only while keeping a live gradient.
  3. `fusion_pose_rgb_linear` applied as a single post-fusion refinement
     (checkpoint names it singular, not per-attention-stage).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HandBackbone(nn.Module):
    """
    EfficientNet-B0 wrapper. Matches a colleague's checkpoint keys
    `rgb_support_backbone.0.*` (358/358 keys+shapes confirmed exact) and
    `rgb_proj.*` (Conv2d 1x1) -- state_dict-compatible with that file.
    """

    def __init__(self, out_channels=256, pretrained=True):
        super().__init__()
        import torchvision.models as tvm
        weights = 'IMAGENET1K_V1' if pretrained else None
        eff = tvm.efficientnet_b0(weights=weights)
        # Wrapped in an extra nn.Sequential to match the checkpoint's
        # 'rgb_support_backbone.0.*' naming (confirmed via exact key/shape
        # comparison in scripts/inspect_friend_checkpoint.py).
        self.rgb_support_backbone = nn.Sequential(eff.features)
        self.rgb_proj = nn.Conv2d(1280, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x: (N, 3, 112, 112) float32, ImageNet-normalized (caller's job --
           kept out of this class so it stays testable on plain synthetic
           tensors).
        Returns: (N, out_channels, H', W') -- 112x112 input to
        EfficientNet-B0's stride-32 feature extractor gives H'=W'=4.
        """
        feat = self.rgb_support_backbone(x)   # (N, 1280, 4, 4)
        return self.rgb_proj(feat)            # (N, out_channels, 4, 4)


class PositionEmbeddingRandom(nn.Module):
    """
    Random Fourier positional encoding. Buffer name
    'positional_encoding_gaussian_matrix' is a verbatim match to Meta's
    Segment Anything Model's PositionEmbeddingRandom class -- Uni-Sign's
    deformable attention borrows this scheme (confirmed via the colleague's
    checkpoint carrying this exact buffer name and shape).

    Deviates slightly from SAM's own convention: SAM's forward_with_coords
    expects coords in [0,1] and remaps internally (2*coords-1); here
    coords already arrive in [-1,1] (this repo's reference points are
    normalized to grid_sample's native convention at precompute time), so
    that remap is skipped.
    """

    def __init__(self, num_pos_feats=128, scale=None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            'positional_encoding_gaussian_matrix', scale * torch.randn(2, num_pos_feats))

    def forward_with_coords(self, coords_input):
        """coords_input: (..., 2) already in [-1, 1] -> (..., 2*num_pos_feats)."""
        coords = coords_input @ self.positional_encoding_gaussian_matrix
        coords = 2 * math.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)


class ContinuousRelPosBiasMLP(nn.Module):
    """
    2 -> 64 -> 64 -> 1 continuous relative-position bias, Swin-Transformer-
    V2-style. Matches checkpoint's rel_pos_bias.mlp (3 Linear layers, first
    two followed by ReLU, confirmed exact shapes).
    """

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Sequential(nn.Linear(2, 64), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True)),
            nn.Linear(64, 1, bias=True),
        )

    def forward(self, rel_pos):
        """rel_pos: (..., 2) relative (dx, dy) -> (..., 1)."""
        return self.mlp(rel_pos)


class DeformablePoseRGBAttention(nn.Module):
    """
    The two-stage pose<->RGB attention (paper Section 3.3, Fig 5b).
    Matches colleague checkpoint's fusion_pose_rgb_DA.* (19 keys) plus the
    sibling fusion_pose_rgb_linear.* key (applied here as a shared
    post-fusion refinement -- judgment call #3, see module docstring).

    Stage 1: standard multi-head cross-attention, pose query attends over
             the full RGB spatial map (flattened H*W tokens) -- "global
             RGB information".
    Stage 2: deformable sampling -- the keypoint coordinate is the
             reference point, a learned offset refines it, grid_sample
             bilinearly samples the RGB map there, producing F_hat_p.

    Judgment call #1 (flagged in the module docstring): to_q/to_k/to_v
    operate on a 32-dim "adapter" representation, not the 256-dim
    embed_dim directly -- `rgb_adapter_down` bridges the RGB side (no
    corresponding checkpoint key; the pose side's equivalent bridge,
    `pgf_keypoint_adapter`, is built by the caller in train_encoder_mt5.py
    since it's shared across calls to this module).
    """

    def __init__(self, embed_dim=256, adapter_dim=32, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter_dim = adapter_dim

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.pe_layer = PositionEmbeddingRandom(num_pos_feats=embed_dim // 2)
        self.rel_pos_bias = ContinuousRelPosBiasMLP()

        self.to_offsets = nn.Sequential(
            nn.Conv1d(adapter_dim, adapter_dim, kernel_size=1, groups=adapter_dim, bias=True),
            nn.GELU(),
            nn.Conv1d(adapter_dim, 2, kernel_size=1, bias=False),
        )
        self.to_q = nn.Conv1d(adapter_dim, embed_dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(adapter_dim, embed_dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(adapter_dim, embed_dim, kernel_size=1, bias=False)
        self.to_out = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=True)

        # Judgment call #1: bridges rgb_proj's embed_dim output down to the
        # adapter_dim to_k/to_v actually consume. No corresponding
        # checkpoint key -- stays randomly initialized even when loading a
        # colleague's converted weights (see scripts/convert_friend_checkpoint.py).
        self.rgb_adapter_down = nn.Conv2d(embed_dim, adapter_dim, kernel_size=1, bias=False)

        # Judgment call #3: applied once, after combining both attention
        # stages (checkpoint names this key singular, not per-stage).
        self.fusion_pose_rgb_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, pose_feat, rgb_map, ref_point):
        """
        pose_feat: (N, adapter_dim) -- N = number of (batch,frame,hand)
                   tokens selected for fusion this step, already projected
                   to adapter_dim by the caller's shared keypoint adapter.
        rgb_map:   (N, embed_dim, Hs, Ws) -- per-sample RGB spatial map
                   from HandBackbone.
        ref_point: (N, 2) -- normalized [-1,1] reference point (the hand's
                   wrist keypoint, precomputed at extraction time).

        Returns: (N, embed_dim) fused feature F_hat_p.
        """
        N = pose_feat.size(0)
        device = pose_feat.device

        pose_1d = pose_feat.unsqueeze(-1)                      # (N, adapter_dim, 1)
        q = self.to_q(pose_1d).transpose(1, 2)                 # (N, 1, embed_dim)

        rgb_adapted = self.rgb_adapter_down(rgb_map)            # (N, adapter_dim, Hs, Ws)
        k = self.to_k(rgb_adapted)                              # (N, embed_dim, Hs, Ws)
        v = self.to_v(rgb_adapted)                              # (N, embed_dim, Hs, Ws)
        Hs, Ws = k.shape[-2:]
        k_tokens = k.flatten(2).transpose(1, 2)                 # (N, Hs*Ws, embed_dim)
        v_tokens = v.flatten(2).transpose(1, 2)                 # (N, Hs*Ws, embed_dim)

        # --- Stage 1: global cross-attention, pose queries all RGB tokens ---
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, Hs, device=device),
            torch.linspace(-1, 1, Ws, device=device), indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # (Hs*Ws, 2)
        pos_emb = self.pe_layer.forward_with_coords(grid)              # (Hs*Ws, embed_dim)
        k_tokens = k_tokens + pos_emb.unsqueeze(0)

        attn_out, _ = self.cross_attn(query=q, key=k_tokens, value=v_tokens)  # (N,1,embed_dim)
        f_global = attn_out.squeeze(1)                                        # (N, embed_dim)

        # --- Stage 2: deformable sampling around the keypoint reference point ---
        offset = self.to_offsets(pose_1d).squeeze(-1)           # (N, 2), learned delta
        sample_xy = (ref_point + torch.tanh(offset) * 0.5).clamp(-1, 1)
        grid_pts = sample_xy.view(N, 1, 1, 2)
        sampled = F.grid_sample(v, grid_pts, mode='bilinear', align_corners=False)  # (N,embed_dim,1,1)
        f_deform = sampled.view(N, self.embed_dim)

        rel = sample_xy - ref_point                              # (N, 2)
        bias = self.rel_pos_bias(rel).squeeze(-1)                 # (N,)
        f_deform = f_deform * torch.sigmoid(bias).unsqueeze(-1)

        fused = f_global + f_deform
        out = self.to_out(fused.unsqueeze(-1)).squeeze(-1)        # (N, embed_dim)
        return self.fusion_pose_rgb_linear(out)


class FusionGate(nn.Module):
    """
    Matches checkpoint fusion_gate.*: Conv1d(2*embed_dim,embed_dim,1) ->
    act -> Conv1d(embed_dim,1,1) -> sigmoid. Implements paper Eq 3-4:
    g = Gate([F_p, F_hat_p]); F_final = (1-g)*F_p + g*F_hat_p.

    Judgment call #2: true g=0 needs logit=-inf. Zero-init the final
    conv's weight AND set its bias to a large negative constant so g starts
    numerically negligible (pose-only) while keeping a working gradient --
    a hard-clamped g=0 would kill backprop into this module entirely.
    """

    def __init__(self, embed_dim=256, init_bias=-6.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(embed_dim, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, f_pose, f_fused):
        """f_pose, f_fused: (N, embed_dim) -> f_final: (N, embed_dim), g: (N, 1)."""
        x = torch.cat([f_pose, f_fused], dim=-1).unsqueeze(-1)   # (N, 2*embed_dim, 1)
        g = torch.sigmoid(self.net(x)).squeeze(-1)                 # (N, 1)
        f_final = (1 - g) * f_pose + g * f_fused
        return f_final, g


def score_aware_sample_indices(hand_scores, input_lengths, p_samp, generator=None):
    """
    Batched score-aware sampling (paper Appendix A.3, Algorithm 1):
    randomly sample int(T * p_samp) frame indices per clip, weighted by
    (1 - mean keypoint confidence) -- i.e. frames with less reliable pose
    get RGB compute preferentially. Equivalent to the paper's
    `random.choices(range(T), weights=sampling_scores, k=int(T*P_samp))`,
    translated to torch.multinomial(..., replacement=True) (same weighted-
    with-replacement semantics).

    hand_scores:   (B, T) float -- per-frame hand detection confidence.
    input_lengths: (B,) long -- true (unpadded) frame counts.
    p_samp: float in (0, 1] -- fraction of true frames sampled per clip.

    Returns:
        sampled_idx:  (B, K) long -- K = per-batch max of each sample's own
                      ceil(input_lengths[b] * p_samp) (the paper's k is
                      PER CLIP, using that clip's own true length -- rows
                      shorter than K are padded, see sampled_mask).
        sampled_mask: (B, K) bool -- True where sampled_idx[b, :] is a real
                      sample for that row (False in the padding region).

    The weighted DRAW itself is fully vectorized; only an unavoidable
    Python loop over the (small) batch dimension is used, since
    torch.multinomial's num_samples is a single scalar valid across the
    whole batch, not a per-row value, and each row's weight vector /
    sample count both vary.
    """
    B, T = hand_scores.shape
    device = hand_scores.device
    weights = (1.0 - hand_scores).clamp(min=1e-6)
    valid = (torch.arange(T, device=device)[None, :] < input_lengths[:, None])
    weights = weights * valid  # padded frames never sampled

    k_per_sample = (input_lengths.float() * p_samp).ceil().long().clamp(min=1)
    K = int(k_per_sample.max().item())

    sampled_idx = torch.zeros(B, K, dtype=torch.long, device=device)
    sampled_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
    for b in range(B):
        k_b = int(k_per_sample[b].item())
        idx = torch.multinomial(weights[b], num_samples=k_b, replacement=True,
                                generator=generator)
        sampled_idx[b, :k_b] = idx
        sampled_mask[b, :k_b] = True

    return sampled_idx, sampled_mask
