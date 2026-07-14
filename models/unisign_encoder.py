"""
Uni-Sign Pose Encoder — Exact Architecture Match

Adapted from Uni-Sign (ICLR 2025): https://github.com/ZechengLi19/Uni-Sign
Original code derived from CoSign + GFSLT-VLP.

Purpose: Load Uni-Sign pretrained weights from HuggingFace, then fine-tune
on KRSL (Kazakh Sign Language) data.

Architecture (matches Uni-Sign exactly):
  Input per group: (B, T, V, 3) — (x, y, score) per joint
    ↓
  Linear(3 → 64) per group
    ↓
  Spatial STGCN: [[64,1], [128,1], [256,1]]  (3 blocks)
    ↓
  Body feature fusion: detach body root → add to face/hands
    ↓
  Temporal STGCN: [[256,3]]  (3 blocks with kernel_size=5)
    ↓
  Mean pool over nodes → (B, T, 256) per group
    ↓
  Concatenate 4 groups → (B, T, 1024) → Linear(1024, 768)

Pretrained checkpoint:
  https://huggingface.co/ZechengLi19/Uni-Sign/tree/main
  Recommended: csl_stage1_weight.pth (pose-only pretraining)
               or csl_daily_pose_only_slt.pth (SLT fine-tuned, pose-only)

Weight compatibility:
  - left/right hand (21 nodes): 100% compatible, loads exactly
  - body (9 nodes from 11): partially compatible (node mapping needed)
  - face (18 nodes from 68): not compatible, reinitialized
"""
import math
import torch
import torch.nn as nn
import numpy as np

# ============================================================
# Graph construction (from Uni-Sign stgcn_layers/gcn_utils.py)
# ============================================================

class Graph:
    """
    Skeleton graph for each sub-pose group.
    Exact copy from Uni-Sign gcn_utils.py.
    """

    def __init__(self, layout='custom', strategy='uniform', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def get_edge(self, layout):
        if layout in ('left', 'right'):
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor = [
                [0, 1], [1, 2], [2, 3], [3, 4],
                [0, 5], [5, 6], [6, 7], [7, 8],
                [0, 9], [9, 10], [10, 11], [11, 12],
                [0, 13], [13, 14], [14, 15], [15, 16],
                [0, 17], [17, 18], [18, 19], [19, 20],
            ]
            self.edge = self_link + neighbor
            self.center = 0
        elif layout == 'body':
            self.num_node = 9
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor = [
                [0, 1], [0, 2], [0, 3], [0, 4],
                [3, 5], [5, 7],
                [4, 6], [6, 8],
            ]
            self.edge = self_link + neighbor
            self.center = 0
        elif layout == 'face_all':
            self.num_node = 18  # 9 + 8 + 1
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor = (
                [[i, i + 1] for i in range(9 - 1)] +
                [[i, i + 1] for i in range(9, 9 + 8 - 1)] +
                [[9 + 8 - 1, 9]] +
                [[17, i] for i in range(17)]
            )
            self.edge = self_link + neighbor
            self.center = self.num_node - 1
        else:
            raise ValueError(f"Unknown layout: {layout}")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if (self.hop_dis[j, self.center]
                                == self.hop_dis[i, self.center]):
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif (self.hop_dis[j, self.center]
                                  > self.hop_dis[i, self.center]):
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return np.dot(A, Dn)


# ============================================================
# ST-GCN Layers (from Uni-Sign stgcn_layers/stgcn_block.py)
# ============================================================

class GCN_unit(nn.Module):
    """
    Graph Convolution Unit with adaptive adjacency.
    Exact copy from Uni-Sign.
    """

    def __init__(self, in_channels, out_channels, kernel_size, A,
                 adaptive=True, t_kernel_size=1, t_stride=1,
                 t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        assert A.size(0) == self.kernel_size
        self.conv = nn.Conv2d(
            in_channels, out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0), stride=(t_stride, 1),
            dilation=(t_dilation, 1), bias=bias,
        )
        self.adaptive = adaptive
        if self.adaptive:
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, len_x=None):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A)).contiguous()
        return self.relu(self.bn(x))


class STGCN_block(nn.Module):
    """
    Spatio-Temporal GCN Block.
    Exact copy from Uni-Sign.
    """

    def __init__(self, in_channels, out_channels, kernel_size, A,
                 adaptive=True, stride=1, dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = GCN_unit(in_channels, out_channels, kernel_size[1], A,
                            adaptive=adaptive)

        if kernel_size[0] > 1:
            self.tcn = nn.Sequential(
                nn.Conv2d(out_channels, out_channels,
                          (kernel_size[0], 1), (stride, 1), padding),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout, inplace=True),
            )
        else:
            self.tcn = nn.Identity()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, len_x=None):
        res = self.residual(x)
        x = self.gcn(x, len_x)
        x = self.tcn(x) + res
        return self.relu(x)


class STGCNChain(nn.Sequential):
    """Chain of STGCN blocks. Exact copy from Uni-Sign."""

    def __init__(self, in_dim, block_args, kernel_size, A, adaptive):
        super().__init__()
        last_dim = in_dim
        for i, [channel, depth] in enumerate(block_args):
            for j in range(depth):
                self.add_module(
                    f'layer{i}_{j}',
                    STGCN_block(last_dim, channel, kernel_size,
                                A.clone(), adaptive),
                )
                last_dim = channel


def get_stgcn_chain(in_dim, level, kernel_size, A, adaptive):
    """
    Build STGCN chain for spatial or temporal level.
    Exact copy from Uni-Sign.
    """
    if level == 'spatial':
        block_args = [[64, 1], [128, 1], [256, 1]]
    elif level == 'temporal':
        block_args = [[256, 3]]
    else:
        raise NotImplementedError
    return STGCNChain(in_dim, block_args, kernel_size, A, adaptive), block_args[-1][0]


# ============================================================
# Key Point Mapper: COCO-WholeBody → Uni-Sign format
# ============================================================

def _map_coco_to_unisign_body(coco_body_xy):
    """
    Map COCO-WholeBody body points (11 nodes) → Uni-Sign body (9 nodes).

    COCO body (11 points from BODY_IDX = [0,1,2,3,4,5,6,7,8,11,12]):
      0: nose,       1: l_eye,    2: r_eye,
      3: l_ear,      4: r_ear,    5: l_shoulder,
      6: r_shoulder, 7: l_elbow,  8: r_elbow,
      9: l_hip,      10: r_hip
    (COCO wrists 9/10 are NOT in BODY_IDX — wrist motion is carried by the
    hand groups, whose root node is the wrist.)

    Uni-Sign body graph (see Graph 'body': edges 0-1, 0-2, 0-3, 0-4,
    3-5, 5-7, 4-6, 6-8) implies the layout:
      0: neck (root)
      1: l_hip,      2: r_hip
      3: l_shoulder, 4: r_shoulder
      5: l_elbow,    6: r_elbow
      7: l_wrist,    8: r_wrist

    The previous mapping ignored this layout (it fed hips into "wrist"
    slots and fabricated hips as "shoulder − 80 px", which is meaningless
    on the offset features this function actually receives). Wrist nodes
    have no source joint here, so they repeat the elbow features — the
    shoulder→elbow→wrist chain then still propagates arm motion.
    """
    B, T = coco_body_xy.shape[0], coco_body_xy.shape[1]
    body_out = torch.zeros(B, T, 9, 2, device=coco_body_xy.device,
                           dtype=coco_body_xy.dtype)
    body_out[:, :, 0] = (coco_body_xy[:, :, 5] + coco_body_xy[:, :, 6]) / 2.0  # neck
    body_out[:, :, 1] = coco_body_xy[:, :, 9]    # l_hip
    body_out[:, :, 2] = coco_body_xy[:, :, 10]   # r_hip
    body_out[:, :, 3] = coco_body_xy[:, :, 5]    # l_shoulder
    body_out[:, :, 4] = coco_body_xy[:, :, 6]    # r_shoulder
    body_out[:, :, 5] = coco_body_xy[:, :, 7]    # l_elbow
    body_out[:, :, 6] = coco_body_xy[:, :, 8]    # r_elbow
    body_out[:, :, 7] = coco_body_xy[:, :, 7]    # l_wrist ≈ l_elbow (no wrist in BODY_IDX)
    body_out[:, :, 8] = coco_body_xy[:, :, 8]    # r_wrist ≈ r_elbow
    return body_out


def _map_coco_to_unisign_face(coco_face_xy):
    """
    Map COCO-WholeBody face (68 points) → Uni-Sign face_all (18 nodes).
    Handles 3D (T, 68, 2) and 4D (B, T, 68, 2) input.
    """
    # 18 landmarks: jaw (0, 8, 16), eyes (36, 39, 42, 45), nose (27, 30, 33),
    # mouth corners (48, 54) and lip contour (49, 52, 55, 60, 63, 67).
    # Note: index 11 was previously a duplicate of 39 (left-eye inner corner);
    # 54 (right mouth corner) completes the mouth symmetrically.
    idx_map = [
        0, 8, 16, 36, 39, 42, 27, 30, 33,
        45, 48, 54, 49, 52, 55, 60, 63, 67,
    ]
    if coco_face_xy.dim() == 4:
        # (B, T, 68, 2) → (B, T, 18, 2)
        face_out = coco_face_xy[:, :, idx_map, :]
    else:
        # (T, 68, 2) → (T, 18, 2)
        face_out = coco_face_xy[:, idx_map, :]
    return face_out


def map_keypoints_to_unisign_format(kps_raw):
    """
    Convert raw COCO-WholeBody keypoints to Uni-Sign format.

    Args:
        kps_raw: (B, T, D) — keypoint features
                 D=282:  offset keypoints [body(11×2), face(68×2), lips(20×2), hand_l(21×2), hand_r(21×2)]
                 D=1128: legacy enriched [offset, velocity, acceleration, validity]
                 D=1410: dual-coord enriched [offset, absolute, velocity, acceleration, validity]
                 The offset portion (first 282) is used for skeleton mapping.
                 With D=1410 the absolute portion (282:564) is appended as two
                 extra input channels per joint. Validity (last 282) is used
                 as the score channel when present.

    Returns:
        dict with keys (C=3 for D<=1128: x_off, y_off, score;
                        C=5 for D=1410: x_off, y_off, x_abs, y_abs, score):
          'body': (B, T, 9, C), 'left'/'right': (B, T, 21, C),
          'face_all': (B, T, 18, C)
    """
    B, T, D = kps_raw.shape
    dual = D >= 1410

    # Extract groups from the offset portion (first 282 dims)
    body_xy = kps_raw[:, :, 0:22].reshape(B, T, 11, 2)
    face_xy = kps_raw[:, :, 22:198].reshape(B, T, 88, 2)
    hand_l_xy = kps_raw[:, :, 198:240].reshape(B, T, 21, 2)
    hand_r_xy = kps_raw[:, :, 240:282].reshape(B, T, 21, 2)

    # Map to Uni-Sign format
    body_mapped = _map_coco_to_unisign_body(body_xy)
    face_mapped = _map_coco_to_unisign_face(face_xy[:, :, :68, :])

    # Dual-coords: append the absolute portion through the same mappings
    if dual:
        abs_part = kps_raw[:, :, 282:564]
        body_abs = _map_coco_to_unisign_body(
            abs_part[:, :, 0:22].reshape(B, T, 11, 2))
        face_abs = _map_coco_to_unisign_face(
            abs_part[:, :, 22:198].reshape(B, T, 88, 2)[:, :, :68, :])
        hand_l_abs = abs_part[:, :, 198:240].reshape(B, T, 21, 2)
        hand_r_abs = abs_part[:, :, 240:282].reshape(B, T, 21, 2)
        body_mapped = torch.cat([body_mapped, body_abs], dim=-1)
        face_mapped = torch.cat([face_mapped, face_abs], dim=-1)
        hand_l_xy = torch.cat([hand_l_xy, hand_l_abs], dim=-1)
        hand_r_xy = torch.cat([hand_r_xy, hand_r_abs], dim=-1)

    # Score: use validity channel from enriched features if available, else 1.0
    if D >= 1128:
        # Validity is the LAST 282 channels of the enriched vector
        validity = kps_raw[:, :, D - 282:D]
        body_valid = validity[:, :, 0:22].reshape(B, T, 11, 2)
        # Map validity through the same joint mapping as body
        body_score = torch.zeros(B, T, 9, 1, device=kps_raw.device, dtype=kps_raw.dtype)
        # Use mean validity of source joints for each mapped joint
        # Same node layout as _map_coco_to_unisign_body.
        body_score[:, :, 0] = ((body_valid[:, :, 5] + body_valid[:, :, 6]) / 2.0).mean(dim=-1, keepdim=True)  # neck (avg of shoulders)
        body_score[:, :, 1] = body_valid[:, :, 9].mean(dim=-1, keepdim=True)   # l_hip
        body_score[:, :, 2] = body_valid[:, :, 10].mean(dim=-1, keepdim=True)  # r_hip
        body_score[:, :, 3] = body_valid[:, :, 5].mean(dim=-1, keepdim=True)   # l_shoulder
        body_score[:, :, 4] = body_valid[:, :, 6].mean(dim=-1, keepdim=True)   # r_shoulder
        body_score[:, :, 5] = body_valid[:, :, 7].mean(dim=-1, keepdim=True)   # l_elbow
        body_score[:, :, 6] = body_valid[:, :, 8].mean(dim=-1, keepdim=True)   # r_elbow
        body_score[:, :, 7] = body_score[:, :, 5]  # l_wrist ≈ l_elbow validity
        body_score[:, :, 8] = body_score[:, :, 6]  # r_wrist ≈ r_elbow validity

        hand_l_valid = validity[:, :, 198:240].reshape(B, T, 21, 2)
        hand_r_valid = validity[:, :, 240:282].reshape(B, T, 21, 2)
        hand_l_score = hand_l_valid.mean(dim=-1, keepdim=True)
        hand_r_score = hand_r_valid.mean(dim=-1, keepdim=True)

        face_valid = validity[:, :, 22:198].reshape(B, T, 88, 2)
        face_score = face_valid.mean(dim=(-2, -1), keepdim=True)  # single mean over all face validity
        face_score = face_score.expand(B, T, 18, 1)  # broadcast to 18 Uni-Sign face nodes

        result = {
            'body': torch.cat([body_mapped, body_score], dim=-1),
            'left': torch.cat([hand_l_xy, hand_l_score], dim=-1),
            'right': torch.cat([hand_r_xy, hand_r_score], dim=-1),
            'face_all': torch.cat([face_mapped, face_score], dim=-1),
        }
    else:
        # Default: use 1.0 as confidence score for each group
        ones = lambda n: torch.ones(B, T, n, 1, device=kps_raw.device, dtype=kps_raw.dtype)
        result = {
            'body': torch.cat([body_mapped, ones(9)], dim=-1),
            'left': torch.cat([hand_l_xy, ones(21)], dim=-1),
            'right': torch.cat([hand_r_xy, ones(21)], dim=-1),
            'face_all': torch.cat([face_mapped, ones(18)], dim=-1),
        }
    return result


# ============================================================
# Weight Loading Utilities
# ============================================================

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Truncated normal initialization (from Uni-Sign)."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def load_unisign_weights(model, path):
    """
    Load pretrained weights from a Uni-Sign checkpoint.

    Handles different checkpoint formats:
      - Uni-Sign HuggingFace: {'model': {...}}
      - Our saved checkpoints: {'encoder': {...}}

    Weight mapping (our keys → Uni-Sign keys):
      proj_linear.<mode>.*        → proj_linear.<mode>.*
      gcn_modules.<mode>.*        → gcn_modules.<mode>.*
      fusion_gcn_modules.<mode>.* → fusion_gcn_modules.<mode>.*
      pose_proj.*                  → pose_proj.*
      part_para                    → part_para

    Compatibility:
      - left/right hands (21 nodes): loads exactly
      - body (9 nodes): loads if shapes match
      - face_all (18 nodes): loads if shapes match
    """
    print(f"\n[Loading pretrained weights from {path}]")
    checkpoint = torch.load(path, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'encoder' in checkpoint:
        state_dict = checkpoint['encoder']
    else:
        state_dict = checkpoint

    loaded = 0
    skipped_shape = 0
    skipped_missing = 0

    our_sd = model.state_dict()

    for our_key in our_sd.keys():
        if our_key not in state_dict:
            skipped_missing += 1
            continue

        our_tensor = our_sd[our_key]
        pt_tensor = state_dict[our_key]

        if our_tensor.shape != pt_tensor.shape:
            # Dual-coords projection: our Linear is (64, 5) with channel
            # order (x_off, y_off, x_abs, y_abs, score); pretrained is
            # (64, 3) = (x, y, score). Copy the matching columns and
            # zero-init the absolute-coordinate columns so the layer
            # computes exactly the pretrained function at init.
            if ('proj_linear' in our_key and our_key.endswith('.weight')
                    and our_tensor.shape == (64, 5)
                    and pt_tensor.shape == (64, 3)):
                pt = pt_tensor.to(our_tensor.dtype)
                our_tensor.zero_()
                our_tensor[:, 0] = pt[:, 0]  # x_off ← x
                our_tensor[:, 1] = pt[:, 1]  # y_off ← y
                our_tensor[:, 4] = pt[:, 2]  # score ← score
                loaded += 1
                print(f"  [DUAL-COORD] {our_key}: 3-ch pretrained → 5-ch "
                      f"(abs columns zero-init)")
                continue
            skipped_shape += 1
            print(f"  [SKIP shape] {our_key}: "
                  f"ours={our_tensor.shape}, pt={pt_tensor.shape}")
            continue

        # Handle dtype mismatch (checkpoint may be BF16)
        if pt_tensor.dtype != our_tensor.dtype:
            pt_tensor = pt_tensor.to(our_tensor.dtype)

        our_sd[our_key].copy_(pt_tensor)
        loaded += 1

    print(f"  Loaded: {loaded} keys")
    print(f"  Skipped (shape mismatch): {skipped_shape} keys")
    print(f"  Skipped (not in checkpoint): {skipped_missing} keys")

    # Per-group summary
    for mode in ['body', 'left', 'right', 'face_all']:
        proj_key = f'proj_linear.{mode}.weight'
        if proj_key in state_dict and proj_key in our_sd:
            ours, pt = our_sd[proj_key], state_dict[proj_key]
            if ours.shape == (64, 5) and pt.shape == (64, 3):
                status = ("✓ loaded (dual-coord)" if torch.allclose(
                    ours[:, [0, 1, 4]], pt.to(ours.dtype))
                    else "✗ reinitialized")
            elif ours.shape == pt.shape and torch.allclose(ours, pt.to(ours.dtype)):
                status = "✓ loaded"
            else:
                status = "✗ reinitialized (shape mismatch)"
        else:
            status = "✗ reinitialized (not in checkpoint)"
        print(f"  {mode}: {status}")


# ============================================================
# Uni-Sign Encoder (matches Uni-Sign models.py exactly)
# ============================================================

class KeypointEncoder(nn.Module):
    """
    Pose encoder matching Uni-Sign architecture exactly.

    Can load pretrained weights from Uni-Sign checkpoints.
    Uses the same ST-GCN layers, channel dimensions, and block counts.

    Differences from original Uni-Sign:
      - No MT5 language model (we use our own decoder)
      - No RGB branch (pose-only)
      - Accepts COCO-WholeBody keypoints (mapped internally)

    Input:  (B, T, 282) — COCO-WholeBody offset keypoints
    Output: (B, T, 768) — pose embeddings

    Usage:
      # Load from scratch:
      encoder = KeypointEncoder()

      # Load with pretrained weights:
      encoder = KeypointEncoder(pretrained_path='path/to/checkpoint.pth')

      # Or load after creation:
      encoder = KeypointEncoder()
      load_unisign_weights(encoder, 'path/to/checkpoint.pth')

      # Fine-tuning with frozen spatial:
      encoder.freeze_spatial()  # freeze spatial STGCN
    """

    MODES = ['body', 'left', 'right', 'face_all']

    def __init__(self, hidden_dim=768, pretrained_path=None, input_dim=282):
        """
        Build Uni-Sign ST-GCN encoder.

        Args:
            hidden_dim: output embedding dimension (default 768, matches MT5-base)
            pretrained_path: path to Uni-Sign checkpoint (optional)
            input_dim: input keypoint dimension per frame
                      282 — standard offset keypoints (default)
                      1128 — enriched features (offset + velocity + acceleration + validity)

        Note on enriched features: When input_dim=1128, the mapper extracts:
          - Offset coordinates (first 282 dims) for skeleton mapping
          - Validity channels (dims 846-1127) as joint confidence scores
          - Velocity and acceleration are implicitly captured by the temporal
            STGCN (kernel_size=5, 3 blocks) through sequence processing
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # Build graphs and adjacency matrices
        self.graph = {}
        A_list = []
        for mode in self.MODES:
            self.graph[mode] = Graph(layout=mode, strategy='distance', max_hop=1)
            A_list.append(torch.tensor(
                self.graph[mode].A, dtype=torch.float32, requires_grad=False,
            ))

        # Projection per group:
        #   3 channels (x_off, y_off, score) for standard/legacy input
        #   5 channels (x_off, y_off, x_abs, y_abs, score) for dual-coords
        # Pretrained 3-ch weights load into the matching columns; the two
        # absolute-coordinate columns start at zero (see load_unisign_weights)
        # so pretrained behaviour is preserved at init.
        self.proj_in_channels = 5 if input_dim >= 1410 else 3
        self.proj_linear = nn.ModuleDict()
        for mode in self.MODES:
            self.proj_linear[mode] = nn.Linear(self.proj_in_channels, 64)

        # Spatial STGCN: [[64,1], [128,1], [256,1]]
        self.gcn_modules = nn.ModuleDict()
        self.fusion_gcn_modules = nn.ModuleDict()
        spatial_kernel_size = A_list[0].size(0)

        for idx, mode in enumerate(self.MODES):
            self.gcn_modules[mode], final_dim = get_stgcn_chain(
                64, 'spatial', (1, spatial_kernel_size),
                A_list[idx].clone(), adaptive=True,
            )
            self.fusion_gcn_modules[mode], _ = get_stgcn_chain(
                final_dim, 'temporal', (5, spatial_kernel_size),
                A_list[idx].clone(), adaptive=True,
            )

        # Share left/right hand weights (matches Uni-Sign)
        self.gcn_modules['left'] = self.gcn_modules['right']
        self.fusion_gcn_modules['left'] = self.fusion_gcn_modules['right']
        self.proj_linear['left'] = self.proj_linear['right']

        # Feature fusion
        # part_para matches concatenated feature dim (256 per group × 4 groups)
        self.part_para = nn.Parameter(torch.zeros(256 * len(self.MODES)))
        self.pose_proj = nn.Linear(256 * 4, hidden_dim)

        # Initialization (from Uni-Sign)
        self.apply(self._init_weights)

        # Load pretrained weights if provided
        if pretrained_path is not None:
            load_unisign_weights(self, pretrained_path)

        n_params = sum(p.numel() for p in self.parameters())
        enrich_tag = " (enriched input)" if input_dim > 282 else ""
        print(f"\n[KeypointEncoder] Uni-Sign architecture{enrich_tag}: {n_params:,} parameters")
        print(f"  Input dim: {input_dim} ({'enriched' if input_dim > 282 else 'standard offset'})")
        print(f"  Spatial STGCN: [[64,1], [128,1], [256,1]] per group")
        print(f"  Temporal STGCN: [[256,3]] per group")
        print(f"  Left/right hands share weights")
        print(f"  Output: (B, T, {hidden_dim})")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, kps_raw, scores=None, freeze_spatial=False, input_lengths=None,
                hand_fusion_fn=None):
        """
        Forward pass matching Uni-Sign's pose branch.

        Args:
            kps_raw: (B, T, D) — keypoint features
                     D=282: standard offset keypoints
                     D=1128: enriched (offset + velocity + acc + validity)
            scores: (B, T, 133) — detection confidence (optional, unused)
            freeze_spatial: deprecated, use encoder.freeze_spatial() instead
            input_lengths: (B,) — true (unpadded) frame count per sample.
                Batches are zero-padded to the longest clip, but Conv2d bias
                and BatchNorm2d affine shift turn those zero frames into a
                nonzero, batch-dependent constant after the first spatial
                block. The temporal STGCN (kernel_size=5) then convolves
                that constant into the last two REAL frames of every
                shorter-than-max clip. Re-zeroing at pad positions before
                the time-mixing step keeps the boundary artifact-free; if
                omitted, forward runs exactly as before (no masking).
            hand_fusion_fn: optional callable
                (mode: str, pool_feat: (B,T,256), kps_raw, input_lengths) -> (B,T,256)
                Called only for mode in {'left', 'right'}, right after
                mean-pooling over graph nodes and BEFORE the 4-group
                concatenation + pose_proj. Lets a caller (e.g. UniSignMT5's
                Prior-Guided RGB Fusion, see models/pgf_fusion.py) inject
                hand-crop RGB fusion at the exact 256-dim seam the
                architecture already provides per group, without this
                class knowing anything about RGB/PGF -- default None means
                this encoder's behavior is EXACTLY unchanged (pose-only,
                per this file's own docstring) for every other caller
                (train_pose_pretrain.py, scripts/diagnose_phase1.py, etc.).

        Returns:
            pose_emb: (B, T, 768)
        """
        B, T, _ = kps_raw.shape

        pad_mask = None  # (B, 1, T, 1), True at padded (invalid) positions
        if input_lengths is not None:
            valid = (torch.arange(T, device=kps_raw.device)[None, :]
                     < input_lengths.to(kps_raw.device)[:, None])  # (B, T)
            pad_mask = (~valid)[:, None, :, None]

        def zero_pad(x):
            if pad_mask is None:
                return x
            return x.masked_fill(pad_mask, 0.0)

        # Map to Uni-Sign format
        parts = map_keypoints_to_unisign_format(kps_raw)

        # Process each group
        features = []
        body_feat = None

        for part in self.MODES:
            feat = parts[part]  # (B, T, V, 3)
            proj_feat = self.proj_linear[part](feat)  # (B, T, V, 64)

            # Rearrange to (B, C, T, V) for Conv2d
            proj_feat = proj_feat.permute(0, 3, 1, 2)  # (B, 64, T, V)
            proj_feat = zero_pad(proj_feat)  # cancel proj_linear's bias at pad frames

            # Spatial STGCN: [[64,1], [128,1], [256,1]]. No temporal kernel
            # here (t_kernel_size=1), so this only mixes across graph nodes —
            # padded frames stay independent of real ones through this stage.
            gcn_feat = self.gcn_modules[part](proj_feat)
            gcn_feat = zero_pad(gcn_feat)  # cancel 3 chained BN/bias shifts

            # Body feature fusion (matches Uni-Sign)
            # body_feat is (B, 256, T, 9). gcn_feat is (B, 256, T, V_group).
            # Select a specific body node, expand to match target group node count.
            if part == 'body':
                body_feat = gcn_feat
            else:
                if body_feat is not None:
                    if part == 'left':
                        # Add detached left wrist (body node 7 in the graph
                        # layout) — the anchor the left hand hangs from
                        feat_add = body_feat[:, :, :, 7].detach().unsqueeze(-1)  # (B, 256, T, 1)
                        gcn_feat = gcn_feat + feat_add.expand_as(gcn_feat)
                    elif part == 'right':
                        # Add detached right wrist (body node 8)
                        feat_add = body_feat[:, :, :, 8].detach().unsqueeze(-1)  # (B, 256, T, 1)
                        gcn_feat = gcn_feat + feat_add.expand_as(gcn_feat)
                    elif part == 'face_all':
                        # Add detached neck (body node 0)
                        feat_add = body_feat[:, :, :, 0].detach().unsqueeze(-1)  # (B, 256, T, 1)
                        gcn_feat = gcn_feat + feat_add.expand_as(gcn_feat)
                    gcn_feat = zero_pad(gcn_feat)  # body_feat is already clean, but re-assert

            # Temporal STGCN: [[256,3]], kernel_size=5 — this is the one block
            # that actually mixes neighbouring TIME steps, so its input must
            # be true zero at pad positions (guaranteed by zero_pad above)
            # or the conv blends a learned pad-artifact into the last ~2
            # real frames of every clip shorter than the batch max.
            gcn_feat = self.fusion_gcn_modules[part](gcn_feat)  # (B, 256, T, V)
            gcn_feat = zero_pad(gcn_feat)  # keep pad frames clean for downstream consumers

            # Mean pool over nodes
            pool_feat = gcn_feat.mean(dim=3)  # (B, 256, T)
            pool_feat = pool_feat.transpose(1, 2)  # (B, T, 256)
            if hand_fusion_fn is not None and part in ('left', 'right'):
                pool_feat = hand_fusion_fn(part, pool_feat, kps_raw, input_lengths)
            features.append(pool_feat)

        # Concatenate: 4 × (B, T, 256) → (B, T, 1024)
        inputs = torch.cat(features, dim=-1)  # (B, T, 1024)
        inputs = inputs + self.part_para  # trainable offset
        inputs = self.pose_proj(inputs)  # (B, T, 768)
        if pad_mask is not None:
            inputs = inputs.masked_fill(pad_mask.squeeze(1), 0.0)

        return inputs

    def freeze_spatial(self):
        """
        Freeze spatial STGCN blocks (transfer learning strategy).

        This preserves pretrained spatial features while allowing
        temporal adaptation to KRSL data.
        """
        for mode in self.MODES:
            for p in self.proj_linear[mode].parameters():
                p.requires_grad = False
            for p in self.gcn_modules[mode].parameters():
                p.requires_grad = False
            # Temporal STGCN stays trainable
        print("[KeypointEncoder] Frozen: projection + spatial STGCN")
        print("[KeypointEncoder] Trainable: temporal STGCN, pose_proj, part_para")

    def freeze_all(self):
        """Freeze entire encoder (use as fixed feature extractor)."""
        for p in self.parameters():
            p.requires_grad = False
        print("[KeypointEncoder] Fully frozen")

    def trainable_params(self):
        """Count trainable vs frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return total, trainable, frozen


def build_masked_pose_decoder(d_model, output_dim):
    """
    Reconstruction head for masked-pose self-supervision: predicts the
    clean keypoint vector from a (possibly masked-input) encoder embedding.

    Shared between UniSignMT5 (train/train_encoder_mt5.py, where it's a
    small 0.1-weighted auxiliary loss during Phase 1) and
    train/train_pose_pretrain.py (where it's the sole training objective,
    for a dedicated pretraining phase run before Phase 1 ever touches mT5).
    """
    return nn.Sequential(
        nn.Linear(d_model, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, output_dim),
    )
