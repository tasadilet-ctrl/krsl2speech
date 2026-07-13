"""
Prosody GAN (SignRecGAN) — maps sign latents to prosody features.
Adapted from S2PFormer: Sign-to-Speech Prosody Transfer via Sign Reconstruction-based GAN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.utils import KEYPOINT_DIM


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
                 num_layers=4, nhead=8, dropout=0.1):
        super().__init__()
        self.generator = ProsodyGenerator(
            d_model=d_model, prosody_dim=prosody_dim,
            keypoint_dim=keypoint_dim, num_layers=num_layers,
            nhead=nhead, dropout=dropout,
        )
        self.discriminator = ProsodyDiscriminator(in_channels=prosody_dim)

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

    def generator_loss(self, sign_latent, prosody_real, keypoints,
                       lambda_adv=0.1, lambda_prosody=5.0, lambda_recon=1.0,
                       input_lengths=None, gen_module=None):
        """
        Compute generator losses.

        Args:
            gen_module: optional module to run the generator forward through
                (pass the DDP-wrapped generator in distributed training so
                gradients synchronize; defaults to self.generator).

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

        total = lambda_adv * adv_loss + lambda_prosody * prosody_loss + lambda_recon * recon_loss

        return total, {
            'gen': total.item(),
            'adv': adv_loss.item(),
            'prosody': prosody_loss.item(),
            'recon': recon_loss.item(),
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
