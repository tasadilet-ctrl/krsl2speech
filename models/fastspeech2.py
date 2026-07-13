"""
FastSpeech2 for Kazakh TTS.
Text + Prosody → Mel-spectrogram.
Adapted for Kazakh subword tokens with prosody control from SignRecGAN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionEncoder(nn.Module):
    """Positional encoding for text tokens."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TextEncoder(nn.Module):
    """Encoder for Kazakh subword tokens."""

    def __init__(self, vocab_size=8000, d_model=256, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model)
        self.pos_enc = PositionEncoder(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, text_ids, text_mask=None):
        """
        Args:
            text_ids: (B, L) — token ids
            text_mask: (B, L) — boolean mask (True = pad)
        Returns:
            (B, L, d_model)
        """
        x = self.embedding(text_ids) * np.sqrt(self.embedding.embedding_dim)
        x = self.pos_enc(x)
        if text_mask is not None:
            x = self.encoder(x, src_key_padding_mask=text_mask)
        else:
            x = self.encoder(x)
        return x


class VarianceAdaptor(nn.Module):
    """Injects prosody (F0, energy) into text encoder output."""

    def __init__(self, d_model=256, prosody_dim=2):
        super().__init__()
        self.f0_conv = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )
        self.energy_conv = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )

    def forward(self, x, prosody):
        """
        Args:
            x:       (B, L, d_model) — text encoder output
            prosody: (B, 2, T') — F0 and energy (may differ in length)

        Returns:
            (B, L, d_model) — prosody-enhanced text representation
        """
        f0 = self.f0_conv(prosody[:, 0:1, :])    # (B, d_model, T')
        energy = self.energy_conv(prosody[:, 1:2, :])  # (B, d_model, T')

        # Interpolate to text length
        L = x.size(1)
        f0 = F.interpolate(f0, size=L, mode='linear', align_corners=False)
        energy = F.interpolate(energy, size=L, mode='linear', align_corners=False)

        # Add to text representation
        x = x + f0.transpose(1, 2) + energy.transpose(1, 2)
        return x


class DurationPredictor(nn.Module):
    """Predicts log duration for each token."""

    def __init__(self, d_model=256):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
            nn.Conv1d(d_model, 1, kernel_size=1),
        )

    def forward(self, x):
        """
        Args: x: (B, L, d_model)
        Returns: (B, L) — log durations
        """
        x = self.predictor(x.transpose(1, 2)).squeeze(1)
        return x


class Decoder(nn.Module):
    """Decoder: prosody-enhanced text → mel-spectrogram."""

    def __init__(self, d_model=256, n_mel=80, nhead=2, num_layers=4, dropout=0.1):
        super().__init__()
        self.pos_enc = PositionEncoder(d_model)

        # FastSpeech2's mel decoder is non-autoregressive: it refines the
        # length-regulated sequence with SELF-attention only. nn.Transformer
        # Decoder is cross-attention (needs a `memory` arg) and is the wrong
        # module here — use a TransformerEncoder stack.
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # Mel projection
        self.mel_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_mel),
        )

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: (B, T_dec, d_model) — length-regulated sequence
            key_padding_mask: (B, T_dec) — True at padded frames (optional)
        Returns: (B, T_dec, n_mel) — mel-spectrogram
        """
        x = self.pos_enc(x)
        out = self.decoder(x, src_key_padding_mask=key_padding_mask)
        mel = self.mel_proj(out)
        return mel


class FastSpeech2(nn.Module):
    """
    FastSpeech2: Text + Prosody → Mel-spectrogram.

    Architecture:
      Text Tokens → TextEncoder → VarianceAdaptor (with prosody)
        ↓
      DurationPredictor (for length regulation)
        ↓
      Token Expansion (based on duration)
        ↓
      Decoder → Mel-spectrogram
    """

    def __init__(
        self,
        vocab_size=8000,
        d_model=256,
        n_mel=80,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        self.text_encoder = TextEncoder(vocab_size, d_model, num_layers=num_encoder_layers, dropout=dropout)
        self.variance_adaptor = VarianceAdaptor(d_model)
        self.duration_predictor = DurationPredictor(d_model)
        self.decoder = Decoder(d_model, n_mel, num_layers=num_decoder_layers, dropout=dropout)

        # Length regulator
        self.scaling_factor = nn.Parameter(torch.tensor(1.0))

    def forward(self, text_ids, prosody, text_lengths=None, duration=None):
        """
        Args:
            text_ids:      (B, L) — Kazakh subword tokens
            prosody:       (B, 2, T') — F0 and energy from GAN
            text_lengths:  (B,) — valid token lengths
            duration:      (B, L) — ground-truth log durations (for training)

        Returns:
            mel:       (B, T_dec, n_mel)
            pred_dur:  (B, L) — predicted durations
        """
        # Encode text
        text_mask = None
        if text_lengths is not None:
            text_mask = torch.arange(text_ids.size(1), device=text_ids.device).unsqueeze(0) >= text_lengths.unsqueeze(1)

        enc_out = self.text_encoder(text_ids, text_mask)

        # Adapt with prosody
        enc_out = self.variance_adaptor(enc_out, prosody[:, :2, :])  # F0 + energy

        # Predict duration
        pred_dur = self.duration_predictor(enc_out)

        # Use ground-truth duration for training, predicted for inference
        if duration is not None:
            # Length regulation: expand tokens based on duration
            mel = self._expand_and_decode(enc_out, duration, prosody)
        else:
            # Inference: use predicted duration
            mel = self._expand_and_decode(enc_out, pred_dur, prosody)

        return mel, pred_dur

    def _expand_and_decode(self, enc_out, log_durations, prosody):
        """Expand tokens based on duration and decode to mel."""
        # Convert log duration to integer duration
        durations = torch.clamp(
            torch.round((log_durations * self.scaling_factor).exp() - 1), min=0
        ).long()

        B, L, D = enc_out.shape

        # Repeat each token according to its duration
        expanded = []
        for b in range(B):
            token_durs = durations[b].tolist()
            reps = []
            for i, d in enumerate(token_durs):
                if d > 0:
                    reps.append(enc_out[b, i:i+1].expand(d, D))
            if reps:
                expanded.append(torch.cat(reps, dim=0))
            else:
                # All durations rounded to 0 — keep one frame so the batch
                # stays aligned (previously this sample was silently dropped,
                # shifting every following sample in the batch).
                expanded.append(enc_out[b, 0:1])

        # Pad to same length
        max_len = max(e.shape[0] for e in expanded)
        padded = []
        for e in expanded:
            if e.shape[0] < max_len:
                pad = torch.zeros(max_len - e.shape[0], D, device=e.device)
                e = torch.cat([e, pad], dim=0)
            padded.append(e)

        expanded_tensor = torch.stack(padded, dim=0)  # (B, T_dec, d_model)

        # Decode to mel
        mel = self.decoder(expanded_tensor)
        return mel

    def forward_train(self, text_ids, prosody, mel_lengths, text_lengths=None):
        """
        Training forward with KNOWN target mel length (no ground-truth token
        durations required).

        Rationale: our corpus has frame-level prosody + mel spectrograms but no
        token-level alignment (no MFA). Instead of a hard length regulator we
        upsample the (prosody-adapted) text encoding to the target mel length by
        interpolation, so predicted mel always matches the ground-truth shape.
        The duration predictor is still trained (auxiliary) against a uniform
        proxy so inference can run without a known length.

        Args:
            text_ids:     (B, L)
            prosody:      (B, 2, T') — F0, energy per mel frame
            mel_lengths:  (B,) — target mel frame counts
            text_lengths: (B,) — valid token counts

        Returns:
            mel:      (B, T_max, n_mel)  where T_max = max(mel_lengths)
            pred_dur: (B, L) — predicted log durations (auxiliary)
            dur_tgt:  (B, L) — proxy log-duration target (mel_len / text_len)
        """
        text_mask = None
        if text_lengths is not None:
            text_mask = (torch.arange(text_ids.size(1), device=text_ids.device)
                         .unsqueeze(0) >= text_lengths.unsqueeze(1))

        enc_out = self.text_encoder(text_ids, text_mask)          # (B, L, D)
        enc_out = self.variance_adaptor(enc_out, prosody[:, :2, :])
        pred_dur = self.duration_predictor(enc_out)              # (B, L)

        # Upsample encoder output (B, L, D) -> (B, T_max, D) by interpolation.
        T_max = int(mel_lengths.max().item())
        up = F.interpolate(
            enc_out.transpose(1, 2), size=T_max, mode='linear', align_corners=False
        ).transpose(1, 2)                                        # (B, T_max, D)
        mel = self.decoder(up)                                   # (B, T_max, n_mel)

        # Proxy duration target: on average each token spans mel_len/text_len
        # frames. log1p to match the predictor's log-space output.
        if text_lengths is not None:
            avg = (mel_lengths.float() / text_lengths.float().clamp(min=1))
            dur_tgt = torch.log1p(avg).unsqueeze(1).expand_as(pred_dur)
        else:
            dur_tgt = torch.zeros_like(pred_dur)

        return mel, pred_dur, dur_tgt

    def forward_inference(self, text_ids, prosody):
        """Inference: text + prosody → mel-spectrogram (uses duration predictor)."""
        mel, _ = self.forward(text_ids, prosody)
        return mel


class FastSpeech2Loss(nn.Module):
    """Combined loss for FastSpeech2 training."""

    def __init__(self):
        super().__init__()
        self.mel_loss = nn.L1Loss()
        self.duration_loss = nn.MSELoss()

    def forward(self, mel_pred, mel_gt, dur_pred, dur_gt, text_lengths=None,
                mel_lengths=None):
        """
        Args:
            mel_pred: (B, T_dec, n_mel)
            mel_gt:   (B, T_dec, n_mel)
            dur_pred: (B, L)
            dur_gt:   (B, L)
            text_lengths: (B,) — valid token counts (optional)
            mel_lengths:  (B,) — valid mel frame counts (optional; masks
                          padded frames out of the mel loss)
        """
        if mel_lengths is not None:
            T = mel_pred.size(1)
            frame_mask = (torch.arange(T, device=mel_pred.device)[None, :]
                          < mel_lengths[:, None]).unsqueeze(-1)  # (B, T, 1)
            diff = (mel_pred - mel_gt).abs() * frame_mask
            mel_l1 = diff.sum() / (frame_mask.sum() * mel_pred.size(-1)).clamp(min=1)
        else:
            mel_l1 = self.mel_loss(mel_pred, mel_gt)

        # Duration loss (only on non-padded tokens)
        if text_lengths is not None:
            mask = torch.arange(dur_pred.size(1), device=dur_pred.device).unsqueeze(0) < text_lengths.unsqueeze(1)
            dur_l1 = torch.mean(torch.abs(dur_pred[mask] - dur_gt[mask]))
        else:
            dur_l1 = torch.mean(torch.abs(dur_pred - dur_gt))

        return mel_l1 + dur_l1, {'mel': mel_l1.item(), 'dur': dur_l1.item()}
