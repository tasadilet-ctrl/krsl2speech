"""
Loss functions for KRSL → Kazakh Speech pipeline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HingeAdversarialLoss(nn.Module):
    """Hinge loss for GAN discriminator/generator."""

    def discriminator_loss(self, real_out, fake_out):
        """
        real_out: discriminator output on real prosody  (should be → 1)
        fake_out: discriminator output on fake prosody  (should be → 0)
        """
        real_loss = F.binary_cross_entropy(real_out, torch.ones_like(real_out))
        fake_loss = F.binary_cross_entropy(fake_out, torch.zeros_like(fake_out))
        return real_loss + fake_loss

    def generator_loss(self, fake_out):
        """Generator tries to make discriminator output → 1."""
        return F.binary_cross_entropy(fake_out, torch.ones_like(fake_out))


class ProsodyL1Loss(nn.Module):
    """Weighted L1 loss for prosody (F0, energy, duration)."""

    def __init__(self, f0_weight=1.0, energy_weight=1.0, duration_weight=0.5):
        super().__init__()
        self.f0_weight = f0_weight
        self.energy_weight = energy_weight
        self.duration_weight = duration_weight

    def forward(self, pred, target):
        """
        pred:   (B, 3, T) — predicted prosody [F0, energy, duration]
        target: (B, 3, T) — ground-truth prosody
        """
        f0_loss = F.l1_loss(pred[:, 0:1, :], target[:, 0:1, :])
        energy_loss = F.l1_loss(pred[:, 1:2, :], target[:, 1:2, :])
        duration_loss = F.l1_loss(pred[:, 2:3, :], target[:, 2:3, :])
        return (
            self.f0_weight * f0_loss
            + self.energy_weight * energy_loss
            + self.duration_weight * duration_loss
        )


def compute_ctc_loss(logits, text_ids, input_lengths, text_lengths, blank):
    """
    CTC loss for gloss decoding.

    Args:
        logits:      (B, T, vocab_size+1) — encoder logits per frame
        text_ids:    (B, L) — target token ids
        input_lengths: (B,) — valid frame count per sample
        text_lengths:  (B,) — valid token count per sample
        blank: int — CTC blank token id

    Returns:
        ctc_loss: scalar
    """
    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, vocab)
    loss = F.ctc_loss(
        log_probs, text_ids, input_lengths, text_lengths,
        blank=blank, zero_infinity=True, reduction='mean'
    )
    return loss


def compute_combined_loss(
    ctc_loss=None,
    adv_loss=None,
    recon_loss=None,
    prosody_loss=None,
    mel_loss=None,
    lambda_ctc=1.0,
    lambda_adv=0.1,
    lambda_recon=1.0,
    lambda_prosody=5.0,
    lambda_mel=1.0,
):
    """Weighted sum of all losses for end-to-end training."""
    total = torch.tensor(0.0, device=ctc_loss.device if ctc_loss is not None else 'cpu')
    losses = {}

    if ctc_loss is not None:
        total = total + lambda_ctc * ctc_loss
        losses['ctc'] = ctc_loss.item()

    if adv_loss is not None:
        total = total + lambda_adv * adv_loss
        losses['adv'] = adv_loss.item()

    if recon_loss is not None:
        total = total + lambda_recon * recon_loss
        losses['recon'] = recon_loss.item()

    if prosody_loss is not None:
        total = total + lambda_prosody * prosody_loss
        losses['prosody'] = prosody_loss.item()

    if mel_loss is not None:
        total = total + lambda_mel * mel_loss
        losses['mel'] = mel_loss.item()

    return total, losses
