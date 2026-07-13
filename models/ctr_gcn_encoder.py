"""
Keypoint Encoder — CTR-GCN + MS-TCN Architecture

Adapted from:
  - GloFE (ACL 2023): Cycle-Time Relational GCN + Multi-Scale Temporal Conv
  - Uni-Sign (ICLR 2025): Sub-pose division (body, face, L-hand, R-hand)
  - S2PFormer (2026): CTR-GCN + MS-TCN as visual backbone

Key improvements over previous architecture:
  1. CTR-GCN: Learns relational edge weights dynamically (not fixed adjacency)
  2. MS-TCN: Multi-scale temporal conv with dilations {1,2,4,8} captures
     gestures from 1 to 15+ frames, not just kernel_size=3
  3. Offset features: Translation-invariant skeleton-relative coordinates
  4. Sub-pose groups: Independent encoding per body part

Input:  (B, T, 282) — offset keypoint coordinates
Output: (B, T+1, d_model), cls_out: (B, d_model)

Layout: (B, C, T, V) — batch, channel, time, vertices
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# ============================================================
# Sub-pose group definitions
# ============================================================
GROUPS = [
    {'name': 'body',      'num_nodes': 11, 'dim': 22,  'start': 0},
    {'name': 'face',      'num_nodes': 88, 'dim': 176, 'start': 22},
    {'name': 'left_hand', 'num_nodes': 21, 'dim': 42,  'start': 198},
    {'name': 'right_hand','num_nodes': 21, 'dim': 42,  'start': 240},
]

# ============================================================
# Utility functions (from GloFE)
# ============================================================

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


# ============================================================
# Graph construction (from Uni-Sign stgcn_layers/gcn_utils.py)
# ============================================================

def build_adjacency(num_nodes, layout, strategy='spatial', max_hop=1, dilation=1):
    """
    Build adjacency matrix for a sub-pose group.

    Returns:
        A: (kernel_size, N, N) — adjacency matrix for GCN
    """
    edge = get_edge(layout, num_nodes)
    hop_dis = get_hop_distance(num_nodes, edge, max_hop=max_hop)

    valid_hop = range(0, max_hop + 1, dilation)
    adjacency = np.zeros((num_nodes, num_nodes))
    for hop in valid_hop:
        adjacency[hop_dis == hop] = 1
    normalize_adjacency = normalize_digraph(adjacency)

    if strategy == 'uniform':
        A = np.zeros((1, num_nodes, num_nodes))
        A[0] = normalize_adjacency
    elif strategy == 'spatial':
        A = []
        for hop in valid_hop:
            a_root = np.zeros((num_nodes, num_nodes))
            a_close = np.zeros((num_nodes, num_nodes))
            a_further = np.zeros((num_nodes, num_nodes))
            center = 0  # root node
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if hop_dis[j, i] == hop:
                        if hop_dis[j, center] == hop_dis[i, center]:
                            a_root[j, i] = normalize_adjacency[j, i]
                        elif hop_dis[j, center] > hop_dis[i, center]:
                            a_close[j, i] = normalize_adjacency[j, i]
                        else:
                            a_further[j, i] = normalize_adjacency[j, i]
            if hop == 0:
                A.append(a_root)
            else:
                A.append(a_root + a_close)
                A.append(a_further)
        A = np.stack(A)
    else:
        A = np.zeros((1, num_nodes, num_nodes))
        A[0] = normalize_adjacency

    return A


def get_edge(layout, num_nodes):
    """Get skeleton edges for a sub-pose group."""
    self_link = [(i, i) for i in range(num_nodes)]

    if layout in ('left_hand', 'right_hand'):
        neighbor = [
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [0, 9], [9, 10], [10, 11], [11, 12],
            [0, 13], [13, 14], [14, 15], [15, 16],
            [0, 17], [17, 18], [18, 19], [19, 20],
        ]
    elif layout == 'body':
        # COCO body: neck(0), shoulders(1,6), elbows(5,8), wrists(7,10), hips(2,3), knees(4,5)
        neighbor = [
            [0, 1], [1, 5], [5, 7],
            [0, 6], [6, 8], [8, 10],
            [0, 2], [2, 4],
            [0, 3], [3, 5],
            [7, 9], [10, 9],  # wrists to center
        ]
    elif layout == 'face':
        # Chain connections for face landmarks
        neighbor = [[i, i + 1] for i in range(num_nodes - 1)]
        # Contour closures (every 17 points)
        for seg_start in range(0, num_nodes - 17, 17):
            neighbor.append([seg_start, seg_start + 16])

    return self_link + neighbor


def get_hop_distance(num_nodes, edge, max_hop=1):
    A = np.zeros((num_nodes, num_nodes))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_nodes, num_nodes)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_nodes = A.shape[0]
    Dn = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


# ============================================================
# CTR-GCN: Cycle-Time Relational Graph Convolution (from GloFE)
# ============================================================

class CTRGC(nn.Module):
    """
    Cycle-Time Relational Graph Convolution.

    Unlike fixed adjacency GCN, CTR-GCN learns relational edge weights
    dynamically based on input features. This captures which joints move
    together in each frame.

    From GloFE (ACL 2023).

    Input:  (B, C, T, V)
    Output: (B, out_channels, T, V)
    """

    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels <= 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction

        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        """
        Args:
            x: (B, C, T, V)
            A: (num_subsets, V, V) — static adjacency (optional prior)
            alpha: scalar — weight for learned vs static adjacency

        Returns:
            (B, out_channels, T, V)
        """
        # x: [B, C, T, V]
        # x1, x2: [B, rel_channels, V] — Theta and Phi mappings
        # x3: [B, out_channels, T, V] — Feature transformation
        x1 = self.conv1(x).mean(-2)  # (B, rel_channels, V)
        x2 = self.conv2(x).mean(-2)  # (B, rel_channels, V)
        x3 = self.conv3(x)            # (B, out_channels, T, V)

        # Learn relational matrix M: (B, out_channels, V, V)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))  # (B, rel_channels, V, V)
        x1 = self.conv4(x1)  # (B, out_channels, V, V)

        # Combine learned + static adjacency
        x1 = x1 * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)

        # Graph conv: (B, out_channels, V, V) x (B, out_channels, T, V) → (B, out_channels, T, V)
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1


# ============================================================
# Multi-Scale Temporal Convolution (from GloFE)
# ============================================================

class TemporalConv(nn.Module):
    """Single temporal convolution with padding."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class MultiScale_TemporalConv(nn.Module):
    """
    Multi-Scale Temporal Convolution (MS-TCN).

    Parallel branches with different dilation rates capture temporal
    context at multiple scales simultaneously:
      - Dilation 1: 1 frame context
      - Dilation 2: 2-3 frame context
      - Dilation 4: 4-7 frame context
      - Dilation 8: 8-15 frame context
      - MaxPool: broad context
      - 1x1: identity

    From GloFE (ACL 2023), used by S2PFormer.

    Input:  (B, C, T, V)
    Output: (B, out_channels, T, V)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilations=[1, 2, 4, 8], residual=True, residual_kernel_size=1):
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, \
            f'# out channels ({out_channels}) should be multiples of # branches ({len(dilations) + 2})'

        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(branch_channels, branch_channels,
                            kernel_size=ks, stride=stride, dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional MaxPool branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels),
        ))

        # Identity branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0,
                      stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels),
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels,
                                         kernel_size=residual_kernel_size, stride=stride)

        self.apply(weights_init)

    def forward(self, x):
        res = self.residual(x)
        branch_outs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


# ============================================================
# Single TCN+GCN Unit (from GloFE)
# ============================================================

class unit_tcn(nn.Module):
    """Temporal convolution unit."""

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class unit_gcn(nn.Module):
    """
    GCN unit with CTR-GC (adaptive relational edges).

    From GloFE (ACL 2023).
    """

    def __init__(self, in_channels, out_channels, A, coff_embedding=4,
                 adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.register_buffer('A', torch.from_numpy(A.astype(np.float32)))
        self.alpha = nn.Parameter(torch.zeros(1))  # zero init — starts as static
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        A = self.PA if self.adaptive else self.A
        y = None
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y


class TCN_GCN_unit(nn.Module):
    """
    Single stack: CTR-GCN (spatial) → MS-TCN (temporal) → residual.

    From GloFE (ACL 2023).
    """

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True,
                 adaptive=True, kernel_size=5, dilations=[1, 2, 4, 8]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(
            out_channels, out_channels, kernel_size=kernel_size,
            stride=stride, dilations=dilations, residual=False)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        return self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))


# ============================================================
# Per-Group Pose Encoder
# ============================================================

class GroupEncoder(nn.Module):
    """
    Encodes one sub-pose group independently.

    Architecture (from GloFE):
      Input: (B, 2, T, V) — offset coordinates (x, y) per node
        ↓
      2× TCN_GCN_unit (64 channels) — no residual (input dim change)
      2× TCN_GCN_unit (64 channels)
        ↓
      Mean pool over nodes → (B, 64, T) → (B, T, 64)

    Each TCN_GCN_unit: CTR-GCN (learned relational edges) → MS-TCN (multi-scale temporal)
    """

    def __init__(self, num_nodes, adj_matrix, base_channels=64):
        super().__init__()
        A = adj_matrix  # (kernel_size, N, N)

        self.l1 = TCN_GCN_unit(2, base_channels, A, residual=False, adaptive=True)
        self.l2 = TCN_GCN_unit(base_channels, base_channels, A, adaptive=True)
        self.l3 = TCN_GCN_unit(base_channels, base_channels, A, adaptive=True)
        self.l4 = TCN_GCN_unit(base_channels, base_channels, A, adaptive=True)

    def forward(self, x):
        """
        Args:
            x: (B, T, num_nodes, 2)

        Returns:
            (B, T, base_channels)
        """
        # Reshape to (B, C, T, V) for Conv2d-based layers
        x = x.permute(0, 3, 1, 2)  # (B, 2, T, num_nodes)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        # Mean pool over nodes (V is dim=3)
        x = x.mean(dim=3)  # (B, base_channels, T)
        x = x.permute(0, 2, 1)  # (B, T, base_channels)
        return x


# ============================================================
# Full Keypoint Encoder
# ============================================================

class KeypointEncoder(nn.Module):
    """
    Full encoder combining:
      - GloFE/S2PFormer: CTR-GCN + MS-TCN per sub-pose group
      - Uni-Sign: Sub-pose division (body, face, L-hand, R-hand)

    Architecture:
      Input: (B, T, 282) — raw keypoint coordinates
        ↓
      Offset encoding: skeleton-relative coordinates (translation invariant)
        ↓
      Split into 4 sub-pose groups:
        - Body (11 nodes × 2 = 22 dims)
        - Face+Lips (88 nodes × 2 = 176 dims)
        - Left Hand (21 nodes × 2 = 42 dims)
        - Right Hand (21 nodes × 2 = 42 dims)
        ↓
      Each group → 4×(CTR-GCN + MS-TCN) → mean pool → (B, T, 64)
        ↓
      Concatenate: (B, T, 256) → Linear(256 → d_model)
        ↓
      Positional Encoding
        ↓
      [CLS] Token prepended
        ↓
      Output: (B, T+1, d_model), cls_out: (B, d_model)
    """

    def __init__(self, d_model=512, nhead=8, num_layers=3,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.base_channels = 64

        # Build adjacency matrices and per-group encoders
        self.group_encoders = nn.ModuleDict()
        for g in GROUPS:
            A = build_adjacency(g['num_nodes'], g['name'], strategy='spatial')
            A_tensor = torch.tensor(A, dtype=torch.float32)
            self.group_encoders[g['name']] = GroupEncoder(
                num_nodes=g['num_nodes'],
                adj_matrix=A_tensor,
                base_channels=self.base_channels,
            )

        # Score-aware weighting (from Uni-Sign)
        self.score_scales = nn.Parameter(torch.ones(len(GROUPS)))

        # Projection: concat(4×64=256) → d_model
        total_group_channels = len(GROUPS) * self.base_channels
        self.proj = nn.Sequential(
            nn.Linear(total_group_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding (learnable)
        self.pos_enc = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Output normalization
        self.output_norm = nn.LayerNorm(d_model)

        print(f"[KeypointEncoder] CTR-GCN+MS-TCN: 4 groups × {self.base_channels}ch "
              f"→ {total_group_channels} → {d_model}")
        print(f"  MS-TCN dilations: [1,2,4,8] + MaxPool + Identity = {len([1,2,4,8])+2} branches")
        print(f"  CTR-GCN: learned relational edges (adaptive)")

    def forward(self, kps, scores=None):
        """
        Args:
            kps: (B, T, 282) — raw keypoint coordinates
            scores: (B, T, 133) — detection confidence (optional)

        Returns:
            latent: (B, T+1, d_model)
            cls_out: (B, d_model)
        """
        B, T, _ = kps.shape

        # Split into groups and encode independently
        features = []
        for g in GROUPS:
            raw = kps[:, :, g['start']:g['start'] + g['dim']]
            raw = raw.reshape(B, T, g['num_nodes'], 2)
            feat = self.group_encoders[g['name']](raw)
            features.append(feat)

        # Score-aware scaling
        features = [f * self.score_scales[i] for i, f in enumerate(features)]

        # Concatenate: 4×(B, T, 64) → (B, T, 256)
        x = torch.cat(features, dim=-1)

        # Project to d_model
        x = self.proj(x)  # (B, T, d_model)

        # Positional encoding
        x = x + self.pos_enc[:, :T, :]  # (B, T, d_model)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        out = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)

        out = self.output_norm(out)
        cls_out = out[:, 0, :]  # (B, d_model)

        return out, cls_out
