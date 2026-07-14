"""
Prosody GAN (SignRecGAN) — maps sign latents to prosody features.
Adapted from S2PFormer: Sign-to-Speech Prosody Transfer via Sign Reconstruction-based GAN
(Manabe et al., arXiv:2604.10413).

Beyond the base generator/discriminator/L1-prosody design, this module
implements the paper's two signature reconstruction losses (Sec 3.2-3.3),
both of which operate purely on prosody + keypoints (no FastSpeech2 needed):
  - SignRec loss: a small ProsodyEstimator reconstructs *sign-motion
    histogram labels* (hand/face velocity+acceleration magnitude
    distributions) FROM the generated prosody. If the generated prosody
    can't reconstruct the sign's motion statistics, the two aren't
    correlated -- this forces sign information into the prosody.
  - ProMo loss: cross-modal prior regularizer aligning speech-energy with
    hand-motion magnitude and speech-pitch with face-motion speed
    (paper Eq. 5), so prosody moves in sympathy with the signing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.utils import KEYPOINT_DIM

# 282-dim keypoint layout (see data/utils.py KEYPOINT_GROUPS): body 0:22,
# face 22:198 (88 nodes incl. lips), left_hand 198:240, right_hand 240:282.
# The paper's sign-motion labels are computed over HANDS and FACE only.
_FACE_SLICE = slice(22, 198)    # 88 nodes -> 176 dims
_HANDS_SLICE = slice(198, 282)  # both hands, 42 nodes -> 84 dims


def sign_motion_labels(keypoints, input_lengths=None, num_bins=16, eps=1e-6):
    """
    Paper Sec 3.2 "sign language prosody label": per-frame squared-magnitude
    of hand/face velocity and acceleration, histogrammed over time into a
    normalized distribution. Produces 4 target distributions per clip:
    (hand-velocity, hand-accel, face-velocity, face-accel).

    keypoints: (B, T, 282) -- any consistent per-frame keypoint
        representation (offset/enriched-first-block both work; velocity is a
        temporal difference, which stays a valid motion signal either way).
    input_lengths: (B,) valid frame counts; padded frames excluded.

    Returns: (B, 4, num_bins) float, each [:, m, :] a probability
    distribution over bins (sums to 1). Computed under no_grad -- this is a
    TARGET label for the SignRec cross-entropy, not a differentiable path.
    """
    with torch.no_grad():
        B, T, _ = keypoints.shape
        device = keypoints.device
        parts = {'hand': keypoints[:, :, _HANDS_SLICE],
                 'face': keypoints[:, :, _FACE_SLICE]}

        labels = []
        for name in ('hand', 'face'):
            p = parts[name]                              # (B, T, 2*nodes)
            p = p.reshape(B, T, -1, 2)                   # (B, T, nodes, 2)
            vel = p[:, 1:] - p[:, :-1]                   # (B, T-1, nodes, 2)
            acc = vel[:, 1:] - vel[:, :-1]               # (B, T-2, nodes, 2)
            # squared magnitude summed over the point set -> scalar per frame
            v_mag = (vel ** 2).sum(dim=-1).sum(dim=-1)   # (B, T-1)
            a_mag = (acc ** 2).sum(dim=-1).sum(dim=-1)   # (B, T-2)

            for mag, offset in ((v_mag, 1), (a_mag, 2)):
                # valid frames: differencing drops `offset` frames off the end
                if input_lengths is not None:
                    valid_len = (input_lengths - offset).clamp(min=1)
                else:
                    valid_len = torch.full((B,), mag.size(1), device=device)
                hist = torch.zeros(B, num_bins, device=device)
                for b in range(B):
                    L = int(valid_len[b].item())
                    m = mag[b, :L]
                    if m.numel() == 0:
                        hist[b, 0] = 1.0
                        continue
                    # per-sample min-max normalize to [0,1] then bin
                    m_norm = (m - m.min()) / (m.max() - m.min() + eps)
                    idx = (m_norm * num_bins).long().clamp(0, num_bins - 1)
                    hist[b].scatter_add_(0, idx, torch.ones_like(m))
                    hist[b] /= hist[b].sum().clamp(min=eps)
                labels.append(hist)

        # order: hand-vel, hand-acc, face-vel, face-acc
        return torch.stack(labels, dim=1)  # (B, 4, num_bins)


class ProsodyEstimator(nn.Module):
    """
    Paper Sec 3.2: reconstructs the 4 sign-motion histogram labels FROM the
    generated prosody (pitch, energy). A shallow temporal conv encoder over
    the prosody sequence, mean-pooled, then a head per (part, motion-type)
    producing a distribution over bins. Kept small on purpose -- its job is
    to provide a reconstruction *signal* back into the generator, not to be
    a strong model in its own right.
    """

    def __init__(self, prosody_dim=2, num_bins=16, hidden=64):
        super().__init__()
        self.num_bins = num_bins
        self.encoder = nn.Sequential(
            nn.Conv1d(prosody_dim, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
        )
        # 4 heads: hand-vel, hand-acc, face-vel, face-acc
        self.heads = nn.ModuleList([nn.Linear(hidden, num_bins) for _ in range(4)])

    def forward(self, prosody, input_lengths=None):
        """
        prosody: (B, T, prosody_dim) -> (B, 4, num_bins) log-probabilities
        (log-softmax over bins, ready for the SignRec cross-entropy).
        """
        x = prosody.transpose(1, 2)          # (B, prosody_dim, T)
        h = self.encoder(x)                  # (B, hidden, T)
        h = h.transpose(1, 2)                # (B, T, hidden)
        if input_lengths is not None:
            T = h.size(1)
            mask = (torch.arange(T, device=h.device)[None, :]
                    < input_lengths[:, None]).unsqueeze(-1)  # (B, T, 1)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = h.mean(1)               # (B, hidden)
        logits = torch.stack([head(pooled) for head in self.heads], dim=1)  # (B,4,num_bins)
        return F.log_softmax(logits, dim=-1)


class ProsodyGenerator(nn.Module):
    """
    Generator for prosody prediction from sign latents.

    Architecture:
      Input: (B, T, d_model) — sign latents from Uni-Sign encoder (no [CLS])
        ↓
      Mean-pooled global query + full sequence via cross-attention
        ↓
      Transformer decoder → prosody sequence
        ↓
      ┌─────────────────────────────────────────┐
      │  Prosody Head  │  Reconstruction Head  │
      │  → (B, T, 2)   │  → (B, T, 282)        │
      └─────────────────────────────────────────┘

    The reconstruction constraint preserves temporal structure,
    preventing the generator from learning trivial statistics.
    """

    def __init__(self, d_model=512, prosody_dim=2, keypoint_dim=KEYPOINT_DIM,
                 num_layers=4, nhead=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Global query: mean-pooled representation of the entire sequence
        self.global_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Positional encoding for output sequence
        self.pos_enc = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)

        # Transformer decoder (query attends to encoder memory)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Prosody head: (B, T, d_model) → (B, T, prosody_dim)
        self.prosody_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, prosody_dim),
        )

        # Reconstruction head: (B, T, d_model) → (B, T, keypoint_dim)
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, keypoint_dim),
        )

        print(f"[ProsodyGenerator] {num_layers} layers, d_model={d_model}, prosody_dim={prosody_dim}")

    def forward(self, sign_latent, input_lengths=None):
        """
        Args:
            sign_latent: (B, T, d_model) — from Uni-Sign encoder (no [CLS])
            input_lengths: (B,) — number of frames

        Returns:
            prosody:  (B, T, prosody_dim)
            recon_kp: (B, T, keypoint_dim)
        """
        B, T, _ = sign_latent.shape

        # Global query: length-aware mean pool (exclude padded frames),
        # expanded to sequence length
        if input_lengths is not None:
            pad_mask = (torch.arange(T, device=sign_latent.device)[None, :]
                        >= input_lengths[:, None])            # (B, T) True = pad
            denom = input_lengths.clamp(min=1).view(B, 1, 1).to(sign_latent.dtype)
            global_ctx = (sign_latent.masked_fill(pad_mask.unsqueeze(-1), 0.0)
                          .sum(dim=1, keepdim=True)) / denom  # (B, 1, d_model)
        else:
            pad_mask = None
            global_ctx = sign_latent.mean(dim=1, keepdim=True)  # (B, 1, d_model)
        global_ctx = self.global_proj(global_ctx)           # (B, 1, d_model)
        query = global_ctx.expand(B, T, -1)                 # (B, T, d_model)

        # Positional encoding — guard against sequences longer than the table
        if T > self.pos_enc.size(1):
            pos = F.interpolate(
                self.pos_enc.transpose(1, 2), size=T, mode='linear',
                align_corners=False,
            ).transpose(1, 2)
        else:
            pos = self.pos_enc[:, :T, :]
        query = query + pos

        # Memory: full encoder sequence (for cross-attention context)
        memory = sign_latent  # (B, T, d_model)

        # Transformer decoder (padded frames masked out of both streams)
        out = self.transformer(
            query, memory,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=pad_mask,
        )  # (B, T, d_model)

        prosody = self.prosody_head(out)   # (B, T, prosody_dim)
        recon_kp = self.recon_head(out)    # (B, T, keypoint_dim)

        return prosody, recon_kp


class ProsodyDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN discriminator for prosody realism.

    Evaluates whether prosody sequences (F0, energy) are
    real (extracted from speech) or generated (from sign latents).

    Uses multiple scales to catch both local and global patterns.
    """

    def __init__(self, in_channels=2):
        super().__init__()
        # Multi-scale discriminators
        self.scales = nn.ModuleList([
            self._make_disc(in_channels, kernel_size=k, stride=s)
            for k, s in [(3, 1), (5, 2), (7, 3)]
        ])

    def _make_disc(self, in_ch, kernel_size, stride):
        """Build a single-scale discriminator."""
        return nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 1, kernel_size=1),
        )

    def forward(self, prosody):
        """
        Args:
            prosody: (B, T, prosody_dim)

        Returns:
            scores: list of tensors, each (B, 1, T') — real/fake scores per scale
        """
        x = prosody.transpose(1, 2)  # (B, prosody_dim, T)
        return [d(x) for d in self.scales]


class ProsodyGAN(nn.Module):
    """
    Combined Prosody GAN (Generator + Discriminator).

    Training loop:
      1. Generator: G(sign_latent) → prosody_gen, recon_kp
      2. Discriminator: D(prosody_real) vs D(prosody_gen)
      3. Losses:
         - L_adv: hinge adversarial loss (multi-scale)
         - L_prosody: L1 between prosody_gen and prosody_real
         - L_recon: L1 between recon_kp and original keypoints
    """

    def __init__(self, d_model=512, prosody_dim=2, keypoint_dim=KEYPOINT_DIM,
                 num_layers=4, nhead=8, dropout=0.1, num_motion_bins=16):
        super().__init__()
        self.generator = ProsodyGenerator(
            d_model=d_model, prosody_dim=prosody_dim,
            keypoint_dim=keypoint_dim, num_layers=num_layers,
            nhead=nhead, dropout=dropout,
        )
        self.discriminator = ProsodyDiscriminator(in_channels=prosody_dim)
        # Paper SignRec: reconstructs sign-motion histograms from prosody.
        self.num_motion_bins = num_motion_bins
        self.prosody_estimator = ProsodyEstimator(
            prosody_dim=prosody_dim, num_bins=num_motion_bins)

    def forward(self, sign_latent, input_lengths=None):
        """Forward pass for generator."""
        return self.generator(sign_latent, input_lengths)

    @staticmethod
    def _masked_l1(pred, target, input_lengths=None):
        """L1 loss over valid (non-padded) frames only."""
        if input_lengths is None:
            return F.l1_loss(pred, target)
        B, T = pred.shape[0], pred.shape[1]
        mask = (torch.arange(T, device=pred.device)[None, :]
                < input_lengths[:, None]).unsqueeze(-1)      # (B, T, 1)
        diff = (pred - target).abs() * mask
        denom = (mask.sum() * pred.shape[-1]).clamp(min=1)
        return diff.sum() / denom

    @staticmethod
    def _masked_mean(x, input_lengths=None):
        """Per-clip mean over valid frames. x: (B, T) -> (B,)."""
        if input_lengths is None:
            return x.mean(dim=1)
        B, T = x.shape
        mask = (torch.arange(T, device=x.device)[None, :] < input_lengths[:, None]).float()
        return (x * mask).sum(1) / mask.sum(1).clamp(min=1)

    @staticmethod
    def _standardize(v, eps=1e-6):
        """Zero-mean/unit-std across the batch (differentiable)."""
        return (v - v.mean()) / (v.std() + eps)

    def signrec_loss(self, prosody_gen, keypoints, input_lengths=None, est_module=None):
        """
        Paper Eq. 4: L_SignRec = -1/4 sum_M sum_k P_M(k) log P_hat_M(k).
        Target histograms P_M come from the keypoints (detached label);
        P_hat_M is the ProsodyEstimator reading the GENERATED prosody, so
        the gradient flows prosody_estimator -> prosody_gen -> generator.

        est_module: optional DDP-wrapped ProsodyEstimator (same convention
            as gen_module/disc_module) so the estimator's gradients sync
            across ranks in distributed training; defaults to
            self.prosody_estimator.
        """
        est = est_module if est_module is not None else self.prosody_estimator
        target = sign_motion_labels(keypoints, input_lengths,
                                    num_bins=self.num_motion_bins)  # (B,4,bins), no grad
        log_pred = est(prosody_gen, input_lengths)  # (B,4,bins) log-probs
        # cross-entropy per (part,motion), averaged over the 4 and the batch
        ce = -(target * log_pred).sum(dim=-1)  # (B, 4)
        return ce.mean()

    def promo_loss(self, prosody_gen, keypoints, input_lengths=None, margin=0.5):
        """
        Paper Eq. 5 (cross-modal prior): speech ENERGY should track HAND
        motion magnitude, speech PITCH should track FACE motion speed.
        Margin-clipped L1 between batch-standardized per-clip means; the
        prosody side carries gradient into the generator, the keypoint side
        is a detached target. prosody_gen channels: [0]=F0/pitch, [1]=energy
        (matches scripts/extract_asan_prosody.py's [F0, energy] convention).
        """
        pitch_mean = self._masked_mean(prosody_gen[:, :, 0], input_lengths)   # (B,)
        energy_mean = self._masked_mean(prosody_gen[:, :, 1], input_lengths)  # (B,)

        with torch.no_grad():
            hands = keypoints[:, :, _HANDS_SLICE].reshape(keypoints.size(0), keypoints.size(1), -1, 2)
            face = keypoints[:, :, _FACE_SLICE].reshape(keypoints.size(0), keypoints.size(1), -1, 2)
            hand_vmag = ((hands[:, 1:] - hands[:, :-1]) ** 2).sum(-1).sum(-1)  # (B, T-1)
            face_vmag = ((face[:, 1:] - face[:, :-1]) ** 2).sum(-1).sum(-1)    # (B, T-1)
            vlen = (input_lengths - 1).clamp(min=1) if input_lengths is not None else None
            hand_mean = self._masked_mean(hand_vmag, vlen)  # (B,)
            face_mean = self._masked_mean(face_vmag, vlen)  # (B,)

        z_energy, z_pitch = self._standardize(energy_mean), self._standardize(pitch_mean)
        z_hand, z_face = self._standardize(hand_mean), self._standardize(face_mean)
        e_term = F.relu((z_energy - z_hand).abs() - margin).mean()
        p_term = F.relu((z_pitch - z_face).abs() - margin).mean()
        return e_term + p_term

    def generator_loss(self, sign_latent, prosody_real, keypoints,
                       lambda_adv=0.1, lambda_prosody=5.0, lambda_recon=1.0,
                       lambda_signrec=1.0, lambda_promo=1.0, promo_margin=0.5,
                       input_lengths=None, gen_module=None, est_module=None):
        """
        Compute generator losses.

        Args:
            gen_module: optional module to run the generator forward through
                (pass the DDP-wrapped generator in distributed training so
                gradients synchronize; defaults to self.generator).
            lambda_signrec / lambda_promo: weights for the paper's SignRec
                and ProMo losses (set to 0 to disable either).

        Returns:
            total_loss, loss_dict
        """
        gen = gen_module if gen_module is not None else self.generator
        prosody_gen, recon_kp = gen(sign_latent, input_lengths)

        # Adversarial loss (hinge: generator wants discriminator to output +1)
        fake_scores = self.discriminator(prosody_gen)
        adv_loss = -torch.mean(torch.cat([s.flatten(1) for s in fake_scores], dim=1))

        # Prosody reconstruction loss (masked by input length)
        prosody_loss = self._masked_l1(prosody_gen, prosody_real, input_lengths)

        # Keypoint reconstruction loss (prevents mode collapse)
        recon_loss = self._masked_l1(recon_kp, keypoints, input_lengths)

        # Paper's SignRec + ProMo losses (both on prosody_gen + keypoints)
        signrec = self.signrec_loss(prosody_gen, keypoints, input_lengths,
                                    est_module=est_module)
        promo = self.promo_loss(prosody_gen, keypoints, input_lengths, margin=promo_margin)

        total = (lambda_adv * adv_loss + lambda_prosody * prosody_loss
                 + lambda_recon * recon_loss + lambda_signrec * signrec
                 + lambda_promo * promo)

        return total, {
            'gen': total.item(),
            'adv': adv_loss.item(),
            'prosody': prosody_loss.item(),
            'recon': recon_loss.item(),
            'signrec': signrec.item(),
            'promo': promo.item(),
        }

    def discriminator_loss(self, sign_latent, prosody_real, input_lengths=None,
                           disc_module=None):
        """
        Compute discriminator losses.

        Hinge loss applied per score BEFORE averaging (the standard
        formulation). The previous relu(1 - mean(scores)) collapses as soon
        as the AVERAGE clears the margin, letting individual scores drift
        arbitrarily and weakening the discriminator signal.

        Real and fake batches go through the discriminator in a SINGLE
        forward (concatenated on the batch axis) — required for DDP, which
        only supports one forward per backward.

        Args:
            disc_module: optional module to run the discriminator through
                (pass the DDP-wrapped discriminator in distributed training).

        Returns:
            total_loss, loss_dict
        """
        disc = disc_module if disc_module is not None else self.discriminator

        with torch.no_grad():
            prosody_gen, _ = self.generator(sign_latent, input_lengths)

        B = prosody_real.size(0)
        scores = disc(torch.cat([prosody_real.detach(), prosody_gen], dim=0))
        real_flat = torch.cat([s[:B].flatten(1) for s in scores], dim=1)
        fake_flat = torch.cat([s[B:].flatten(1) for s in scores], dim=1)

        # Real → > +1, fake → < -1
        real_loss = F.relu(1.0 - real_flat).mean()
        fake_loss = F.relu(1.0 + fake_flat).mean()
        d_loss = real_loss + fake_loss

        return d_loss, {
            'disc': d_loss.item(),
            'real': real_flat.mean().item(),
            'fake': fake_flat.mean().item(),
        }
