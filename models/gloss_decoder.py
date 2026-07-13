"""
Autoregressive Gloss Decoder (Transformer Decoder)

Adapted from:
  - GloFE (ACL 2023): Transformer decoder with future mask, teacher forcing,
    cross-attention to visual backbone output, label smoothing loss
  - Uni-Sign (ICLR 2025): Subword vocabulary, label smoothing

Replaces the 2-layer MLP + CTC decoder.

Architecture:
  Encoder (KeypointEncoder): visual features → (B, T+1, d_model) with [CLS]
  Decoder: text tokens → cross-attend to visual features → next token

  Input: (B, T+1, d_model) — sign latents from encoder (with [CLS])
  Target: (B, L) — text token IDs (autoregressive)

  Forward:
    text_emb = self.emb(text_input) + self.pos_enc
    out = self.transformer(text_emb, memory=sign_latent, mask=future_mask)
    logits = self.logits_proj(out)  # (B, L, vocab_size)

  Training: CrossEntropy with label smoothing + teacher forcing
  Inference: Autoregressive generation with optional beam search
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FutureMask(nn.Module):
    """
    Causal mask: prevents attending to future tokens during training.

    From GloFE trans_model.py.
    """

    def __init__(self, max_len=512):
        super().__init__()
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(1, max_len, max_len), diagonal=1).bool(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, L, L) — attention weights

        Returns:
            (B, L, L) with future positions masked to -inf
        """
        return x.masked_fill(self.mask[:, :x.size(1), :x.size(2)], float('-inf'))


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for cross-entropy training.

    From GloFE trans_model.py.
    Reduces overconfidence on correct labels, improving generalization.
    """

    def __init__(self, size, padding_idx, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        Args:
            x: (B, L, vocab_size) — logits
            target: (B, L) — token IDs

        Returns:
            smoothed_loss: scalar
        """
        assert x.size(0) * x.size(1) == target.size(0) * target.size(1)
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(
            F.log_softmax(x, dim=-1),
            true_view(true_dist, x),
        )


def true_view(data, size):
    return data.view(size[0], size[1], size[2])


class GlossDecoder(nn.Module):
    """
    Autoregressive Transformer Decoder for gloss (text) generation.

    Uses cross-attention to attend to sign latents from the encoder,
    and self-attention with future mask for autoregressive text generation.

    From GloFE (ACL 2023) trans_model.py, adapted for sign-to-speech.

    Training:
      - Teacher forcing: ground truth text fed as input
      - Cross-entropy with label smoothing
      - Noam learning rate schedule

    Inference:
      - Autoregressive: <sos> → generate → feed back → repeat
      - Beam search supported
    """

    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=3,
                 dim_feedforward=2048, dropout=0.1, max_len=512,
                 label_smoothing=0.1, padding_idx=0):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        # Embedding layer
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

        # Positional encoding (learnable)
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        # Future mask for autoregressive training
        self.future_mask = FutureMask(max_len=max_len)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection: d_model → vocab_size
        self.logits_proj = nn.Linear(d_model, vocab_size)

        # Loss function
        self.loss_fn = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=padding_idx,
            smoothing=label_smoothing,
        )

        # Weight initialization (Gluon/Xavier uniform)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        print(f"[GlossDecoder] vocab={vocab_size}, {num_layers} layers, "
              f"d_model={d_model}, label_smoothing={label_smoothing}")

    def forward(self, sign_latent, text_input, text_lengths,
                input_lengths=None):
        """
        Forward pass with teacher forcing (training mode).

        Args:
            sign_latent: (B, T+1, d_model) — encoder output with [CLS]
            text_input: (B, L) — ground truth text token IDs (shifted right)
            text_lengths: (B,) — number of tokens per sequence
            input_lengths: (B,) — number of sign frames (optional, for masking)

        Returns:
            logits: (B, L, vocab_size)
            loss: scalar (smoothed cross-entropy)
            ppl: perplexity
        """
        B, L = text_input.shape

        # Embedding + positional encoding
        text_emb = self.emb(text_input)  # (B, L, d_model)
        text_emb = text_emb + self.pos_enc[:, :L, :]

        # Build padding mask for decoder
        padding_mask = text_input == self.padding_idx

        # Source key padding mask (for cross-attention to sign latents)
        src_key_padding_mask = None
        if input_lengths is not None:
            T = sign_latent.size(1) - 1
            src_mask = (
                torch.arange(T + 1, device=sign_latent.device).unsqueeze(0)
                >= (input_lengths.unsqueeze(1) + 1)
            )
            src_key_padding_mask = src_mask

        # Transformer decoder with future mask
        out = self.transformer(
            tgt=text_emb,
            memory=sign_latent,
            tgt_mask=self.future_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )  # (B, L, d_model)

        # Project to vocabulary
        logits = self.logits_proj(out)  # (B, L, vocab_size)

        # Compute loss
        loss, ppl = self.compute_loss(logits, text_input)

        return logits, loss, ppl

    def compute_loss(self, logits, target):
        """Compute smoothed cross-entropy loss and perplexity."""
        loss = self.loss_fn(logits, target)

        # Perplexity (effective)
        total_words = target.ne(self.padding_idx).sum().item()
        non_pad_tokens = total_words if total_words > 0 else 1
        loss = loss / non_pad_tokens
        ppl = torch.exp(torch.clamp(loss, max=20))

        return loss, ppl

    def generate(self, sign_latent, max_len=200, sos_idx=1,
                 eos_idx=2, beam_size=1):
        """
        Autoregressive generation (inference mode).

        Args:
            sign_latent: (B, T+1, d_model)
            max_len: maximum sequence length
            sos_idx: start-of-sequence token index
            eos_idx: end-of-sequence token index
            beam_size: beam search width (1 = greedy)

        Returns:
            sequences: (B, L) or (B*beam_size, L)
            scores: (B,) or (B*beam_size,)
        """
        if beam_size > 1:
            return self._beam_search(
                sign_latent, max_len, sos_idx, eos_idx, beam_size,
            )

        # Greedy decoding
        B = sign_latent.shape[0]
        device = sign_latent.device

        # Start with <sos> token
        current = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)
        sequences = [current.squeeze(1)]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for i in range(1, max_len):
            # Embedding + positional encoding
            text_emb = (
                self.emb(current) + self.pos_enc[:, :current.size(1), :]
            )

            # Decoder forward
            out = self.transformer(
                tgt=text_emb,
                memory=sign_latent,
                tgt_mask=self.future_mask,
            )  # (B, current_len, d_model)

            # Last position logits
            logits = self.logits_proj(out[:, -1, :])  # (B, vocab_size)

            # Greedy selection
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)

            sequences.append(next_token.squeeze(1))
            finished = finished | (next_token.squeeze(1) == eos_idx)

            current = next_token

            if finished.all():
                break

        result = torch.stack(sequences, dim=1)  # (B, L)
        return result

    def _beam_search(self, sign_latent, max_len, sos_idx, eos_idx, beam_size):
        """
        Beam search decoding.

        From GloFE trans_model.py.
        """
        B = sign_latent.shape[0]
        device = sign_latent.device

        # Initialize beams: (B * beam_size, 1)
        sos = torch.full(
            (B * beam_size, 1), sos_idx, dtype=torch.long, device=device,
        )
        beam_scores = torch.zeros(B * beam_size, device=device)
        batch_idx = torch.arange(B, device=device).repeat_interleave(beam_size)

        sequences = [sos.squeeze(1)]
        finished_beams = torch.zeros(B, dtype=torch.bool, device=device)
        final_scores = torch.zeros(B, device=device)

        for i in range(1, max_len):
            # Embedding + positional encoding
            text_emb = self.emb(sos) + self.pos_enc[:, :sos.size(1), :]

            # Group by batch for cross-attention
            unique_batch = torch.unique(batch_idx)
            out_groups = {}
            for b in unique_batch:
                mask = batch_idx == b
                out = self.transformer(
                    tgt=text_emb[mask],
                    memory=sign_latent[b:b + 1],
                    tgt_mask=self.future_mask,
                )
                out_groups[b.item()] = out

            # Get logits for last position
            logits_groups = {}
            for b, out in out_groups.items():
                logits = self.logits_proj(out[:, -1, :])
                logits_groups[b] = logits

            # Expand to beam_size
            log_probs = F.log_softmax(
                torch.cat(
                    [logits_groups[b] for b in unique_batch], dim=0,
                ),
                dim=-1,
            )
            vocab_size = log_probs.size(1)

            # Score: beam_score + log_prob
            scores = beam_scores.unsqueeze(1) + log_probs

            new_sequences = []
            new_scores = []
            new_batch_idx = []

            for b in range(B):
                if finished_beams[b]:
                    continue
                indices = range(b * beam_size, (b + 1) * beam_size)
                top_scores, top_words = scores[indices].view(-1).topk(beam_size)

                for j in range(beam_size):
                    word = top_words[j].item()
                    score = top_scores[j].item()
                    src_beam = top_words[j] // vocab_size
                    src_seq = sequences[-1][b * beam_size + src_beam]

                    new_sequences.append(torch.cat([src_seq, word.unsqueeze(0)]))
                    new_scores.append(score)
                    new_batch_idx.append(b)

                    if word == eos_idx and not finished_beams[b]:
                        final_scores[b] = score
                        finished_beams[b] = True

            if all(finished_beams):
                break

            sequences = new_sequences
            beam_scores = torch.tensor(new_scores, device=device)
            batch_idx = torch.tensor(
                new_batch_idx, dtype=torch.long, device=device,
            )
            sos = torch.stack(sequences).unsqueeze(-1)

        return torch.stack(sequences).view(B, beam_size, -1)[:, 0], final_scores


# ============================================================
# Legacy CTC Decoder (kept for backward compatibility)
# ============================================================

class GlossDecoderCTC(nn.Module):
    """
    Transitional CTC decoder with LSTM backbone.

    NOTE: The autoregressive GlossDecoder above is preferred.
    This is kept only for compatibility with existing checkpoints.
    """

    def __init__(self, vocab_size, d_model=512, dropout=0.1, nlayers=4):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.drop = nn.Dropout(dropout)

        self.word_lut = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=nlayers,
            batch_first=True,
            dropout=dropout if nlayers > 1 else 0,
            bidirectional=False,
        )

        self.pre_output = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, vocab_size + 1)  # +1 for CTC blank
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, sign_latent):
        """
        Args:
            sign_latent: (B, T+1, d_model)

        Returns:
            (T+1, B, vocab_size + 1) — logits for CTC loss
        """
        x = self.drop(sign_latent)
        out, _ = self.word_lut(x)
        out = self.pre_output(out)
        out = self.layer_norm(out)
        out = F.relu(out)
        out = self.output(out)  # (B, T+1, vocab_size+1)
        out = out.permute(1, 0, 2)  # (T+1, B, vocab_size+1)
        return out

    def decode_greedy(self, logits, input_lengths):
        """Greedy CTC decoding."""
        token_seqs = []
        B = logits.size(1)
        blank = self.vocab_size  # last token is blank

        for b in range(B):
            probs = logits[:input_lengths[b], b, :].softmax(dim=-1)
            pred = probs.argmax(dim=-1)

            decoded = []
            prev = -1
            for t in range(input_lengths[b]):
                tok = pred[t].item()
                if tok != blank and tok != prev:
                    decoded.append(tok)
                prev = tok

            token_seqs.append(torch.tensor(decoded, dtype=torch.long))

        return token_seqs
