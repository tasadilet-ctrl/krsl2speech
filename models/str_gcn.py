"""
STRGCN: Spatio-Temporal Relational Graph Convolutional Network
Adapted from: "STRGCN: Capturing Asynchronous Spatio-Temporal Dependencies
for Irregular Multivariate Time Series Forecasting" (arXiv:2505.04167)

ADAPTATION for sign language:
- Sparse temporal graph (each frame connects to ±W neighbors, not all T²)
- Sinusoidal positional encoding before STRGCN (preserves temporal order)
- No sandwich structure (global pooling collapses temporal variation)
- Temporal + spatial relational embeddings on edges

Input: (B, T, 282)
Output: (B, T, d_model)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class STRGCNLayer(nn.Module):
    """
    Spatio-Temporal Relational Graph Convolution Layer on a sparse graph.

    For each node (frame) i:
        h_i = σ( Σ_{j∈neighbors(i)} a*_{i,j} (W^t * W^s) h_j + h_i )

    where:
        W^t = P_i · Q(P_j)  — temporal relation from position embeddings
        W^s = S_i · S_j      — spatial relation from keypoint features
        a*_{i,j} = softmax over neighbors
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Temporal transformation
        self.Q = nn.Linear(d_model, d_model, bias=False)

        # Spatial relation projection
        self.spatial_proj = nn.Linear(d_model, d_model, bias=False)

        # Self-connection weight
        self.W_0 = nn.Linear(d_model, d_model)

        # Normalization
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, h, P, S, adj_mask):
        """
        Args:
            h: (B, T, D) — node features
            P: (B, T, D) — positional/temporal embeddings per node
            S: (B, T, D) — spatial embeddings from keypoint features
            adj_mask: (B, T, N_neigh) — indices of neighbors for each node

        Returns:
            (B, T, D)
        """
        B, T, D = h.shape
        residual = h
        N_neigh = adj_mask.shape[-1]

        # Gather neighbor features using advanced indexing:
        # h_neigh[b, t, n, d] = h[b, adj_mask[b,t,n], d]
        b_idx = torch.arange(B, device=h.device)[:, None, None]  # (B, 1, 1)
        h_neigh = h[b_idx, adj_mask]  # (B, T, N_neigh, D)
        P_neigh = P[b_idx, adj_mask]  # (B, T, N_neigh, D)
        S_neigh = S[b_idx, adj_mask]  # (B, T, N_neigh, D)

        # Temporal relation: W^t_{ij} = P_i · Q(P_j)
        Q_P = self.Q(P_neigh)  # (B, T, N, D)
        P_expand = P.unsqueeze(2)  # (B, T, 1, D)
        W_t = (P_expand * Q_P).sum(dim=-1)  # (B, T, N)

        # Spatial relation: W^s_{ij} = S_i · S_j
        S_proj = self.spatial_proj(S_neigh)  # (B, T, N, D)
        S_expand = S.unsqueeze(2)  # (B, T, 1, D)
        W_s = (S_expand * S_proj).sum(dim=-1)  # (B, T, N)

        # Combined edge weights
        edge_weights = W_t * W_s  # (B, T, N)

        # Normalization: softmax over neighbors
        alpha = F.softmax(edge_weights, dim=-1)  # (B, T, N)

        # Weighted message aggregation
        msg = torch.sum(alpha.unsqueeze(-1) * h_neigh, dim=2)  # (B, T, D)

        # Transform + residual
        out = self.W_0(msg) + residual
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)

        return out


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (B, T, D) → add positional encoding."""
        T = x.size(1)
        return x + self.pe[:T]


class STRGCNEncoder(nn.Module):
    """
    STRGCN encoder with sparse temporal graph.

    Architecture:
      Input: (B, T, 282)
        ↓
      LayerNorm(282) + Linear(282 → D) + Positional Encoding
        ↓
      K × STRGCN Layers (sparse temporal graph, ±W neighbors)
        ↓
      Output: (B, T, D)
    """

    def __init__(self, d_model=512, num_layers=3, dropout=0.1,
                 input_dim=282, window_size=10):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.window_size = window_size

        # Input normalization + projection
        self.input_norm = nn.LayerNorm(input_dim, eps=1e-5)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding (strong temporal signal)
        self.pos_enc = PositionalEncoding(d_model)

        # Spatial projection from keypoint features
        self.spatial_proj_embed = nn.Linear(input_dim, d_model)

        # STRGCN layers (sparse)
        self.layers = nn.ModuleList([
            STRGCNLayer(d_model, dropout)
            for _ in range(num_layers)
        ])

        # Final normalization
        self.output_norm = nn.LayerNorm(d_model)

        # Precompute adjacency: each frame connects to ±W neighbors (plus self)
        self._build_adjacency()

        print(f"[STRGCNEncoder] {num_layers} layers, window={window_size} "
              f"(graph: {2*window_size+1} neighbors per node)")

    def _build_adjacency(self):
        """Build sparse adjacency: each frame connects to ±W neighbors."""
        # We'll build this dynamically based on T, but register a template
        pass

    def _get_adjacency(self, T, device):
        """Build adjacency indices for sequence length T."""
        W = self.window_size
        N_neigh = 2 * W + 1  # ±W + self

        # For each position t, neighbors are [max(0,t-W), ..., min(T-1,t+W)]
        neighbors = []
        for t in range(T):
            start = max(0, t - W)
            end = min(T - 1, t + W)
            neigh = list(range(start, end + 1))
            # Pad to N_neigh if needed (at boundaries)
            while len(neigh) < N_neigh:
                neigh.append(t)  # repeat self
            neighbors.append(neigh)

        adj = torch.tensor(neighbors, dtype=torch.long, device=device)  # (T, N_neigh)
        return adj

    def forward(self, kps):
        """
        Args:
            kps: (B, T, 282) — raw keypoint coordinates

        Returns:
            (B, T, d_model)
        """
        B, T, _ = kps.shape

        # Normalize + project
        x = self.input_norm(kps)       # (B, T, 282)
        x = self.input_proj(x)         # (B, T, D)

        # Add positional encoding
        x = self.pos_enc(x)            # (B, T, D)

        # Spatial embeddings from keypoint features
        S = self.spatial_proj_embed(self.input_norm(kps))  # (B, T, D)

        # Temporal embeddings = the features themselves (after pos enc)
        P = x  # (B, T, D)

        # Build sparse adjacency
        adj_mask = self._get_adjacency(T, x.device).unsqueeze(0).expand(B, -1, -1)

        # STRGCN layers
        for layer in self.layers:
            x = layer(x, P, S, adj_mask)

        x = self.output_norm(x)
        return x


class KeypointEncoder(nn.Module):
    """
    Full encoder: STRGCN + [CLS] token.

    Input: (B, T, 282)
    Output: latent (B, T+1, d_model), cls_out (B, d_model)
    """

    def __init__(self, d_model=512, nhead=8, num_layers=3,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.str_gcn = STRGCNEncoder(
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
        )

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.output_norm = nn.LayerNorm(d_model)

        print(f"[KeypointEncoder] Sparse STRGCN({num_layers}layers) + [CLS]: 282 → {d_model}")

    def forward(self, kps, scores=None):
        """
        Args:
            kps: (B, T, 282)
        Returns:
            latent: (B, T+1, d_model)
            cls_out: (B, d_model)
        """
        # STRGCN encoding
        x = self.str_gcn(kps)  # (B, T, D)

        # Prepend [CLS] token
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        out = torch.cat([cls, x], dim=1)  # (B, T+1, D)

        out = self.output_norm(out)
        cls_out = out[:, 0, :]

        return out, cls_out
