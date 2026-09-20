"""
Phase 1: Uni-Sign Encoder + MT5 Decoder.

Replaces our custom Transformer decoder with MT5 (multilingual T5),
matching Uni-Sign's architecture exactly.

MT5 is pretrained on Kazakh + 100+ languages, so it already knows
grammar, vocabulary, and spelling. We only train the encoder + adapter.

Usage:
  # From Uni-Sign pretrained weights (recommended):
  PYTHONPATH=. python train/train_encoder_mt5.py \
      --config configs/config.yaml \
      --pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth

  # Multi-GPU (2 GPUs):
  CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. torchrun --nproc_per_node=2 train/train_encoder_mt5.py \
      --config configs/config.yaml \
      --pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth

Architecture (matches Uni-Sign models.py):
  Keypoints → Uni-Sign Encoder → pose_proj (1024→768)
    ↓
  Prefix: "Translate sign language video to Kazakh: "
    ↓
  MT5 Encoder (inputs_embeds) → MT5 Decoder → text
"""
import os
import time
import yaml
import argparse
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW

from data.khabar_dataset import KhabarKzDataset
from data.kazsign_dataset import KazSignDataset
from data.informburo_dataset import InformburoDataset
from data.asan_dataset import AsanDataset
from data.utils import ENRICHED_DIM, KEYPOINT_DIM
from utils.metrics import compute_bleu, compute_corpus_wer, compute_rouge, compute_bertscore
from models.unisign_encoder import (
    KeypointEncoder, load_unisign_weights, build_masked_pose_decoder,
    build_prosody_aux_head)
from models.pgf_fusion import (
    HandBackbone, DeformablePoseRGBAttention, FusionGate, score_aware_sample_indices)


# ============================================================
# MT5 Path (from Uni-Sign config.py)
# ============================================================
# Checkpoint-selection metrics and their direction. Val CE is included
# for backwards compatibility but is NOT recommended: on this task val CE
# rises while generation quality (WER/BLEU) keeps improving, so selecting
# on it reliably picks a WORSE model -- observed across multiple runs.
SELECT_METRICS = {
    'wer': 'lower',
    'bleu': 'higher',
    'rouge1': 'higher',
    'rougeL': 'higher',
    'bertscore_f1': 'higher',
    'val_loss': 'lower',
}

MT5_PATH = "google/mt5-base"  # 388M params, d_model=768 (matches Uni-Sign encoder)


# ============================================================
# Dataset (no tokenizer needed — MT5 has its own)
# ============================================================

class SimpleCollator:
    """Collator for MT5 training. Tokenizes text here (not in forward!) for speed."""

    def __init__(self, mt5_tokenizer=None, max_text_tokens=128):
        self.mt5_tokenizer = mt5_tokenizer
        self.max_text_tokens = max_text_tokens

    def __call__(self, batch):
        valid = [b for b in batch if b is not None and b.get('input_length', 0) > 0]
        if not valid:
            return None

        # Filter samples with text
        valid = [b for b in valid if b.get('text') and b['text'].strip()]
        if not valid:
            return None

        # Sort by keypoint length (descending) for efficient padding
        valid.sort(key=lambda x: x['input_length'], reverse=True)

        # Pad keypoints
        kps = [b['keypoints'] for b in valid]
        max_t = max(k.shape[0] for k in kps)
        input_lengths = torch.tensor([k.shape[0] for k in kps], dtype=torch.long)

        kps_padded = torch.zeros(len(valid), max_t, kps[0].shape[1], dtype=torch.float32)
        for i, k in enumerate(kps):
            kps_padded[i, :k.shape[0], :] = k

        # Pad RGB features (only if every sample in the batch has them --
        # AsanDataset.__getitem__ already skips clips missing an rgb file
        # entirely when load_rgb=True, via _blank_sample(), so partial
        # coverage within a kept batch shouldn't normally happen once
        # extraction covers all clips).
        rgb_list = [b.get('rgb') for b in valid]
        rgb_padded = None
        if all(r is not None for r in rgb_list) and rgb_list:
            rgb_padded = torch.zeros(len(valid), max_t, rgb_list[0].shape[1], dtype=torch.float32)
            for i, r in enumerate(rgb_list):
                rgb_padded[i, :r.shape[0], :] = r

        # Prosody ([F0, energy], (T,2)) for the prosody-as-supervision
        # ablation. Same all-or-nothing convention as rgb above:
        # AsanDataset returns a blank sample for clips missing prosody, so a
        # partial batch shouldn't occur once extraction covers the corpus.
        prosody_list = [b.get('prosody') for b in valid]
        prosody_padded = None
        if all(p is not None for p in prosody_list) and prosody_list:
            prosody_padded = torch.zeros(len(valid), max_t,
                                         prosody_list[0].shape[1],
                                         dtype=torch.float32)
            for i, p_ in enumerate(prosody_list):
                prosody_padded[i, :p_.shape[0], :] = p_

        # Pad hand-crop fields for Prior-Guided Fusion (only if every
        # sample in the batch has them -- same "skip clips missing the
        # file entirely via _blank_sample()" convention as rgb above, so
        # partial coverage within a kept batch shouldn't normally happen
        # once extraction covers all clips).
        hand_crops_list = [b.get('hand_crops') for b in valid]
        hand_crops_padded = hand_ref_padded = hand_valid_padded = hand_score_padded = None
        if all(h is not None for h in hand_crops_list) and hand_crops_list:
            def _pad_time(tensors, extra_shape, dtype):
                out = torch.zeros((len(valid), max_t) + extra_shape, dtype=dtype)
                for i, t in enumerate(tensors):
                    out[i, :t.shape[0]] = t
                return out

            hand_crops_padded = _pad_time(hand_crops_list, (2, 112, 112, 3), torch.uint8)
            hand_ref_padded = _pad_time([b['hand_ref'] for b in valid], (2, 2), torch.float32)
            hand_valid_padded = _pad_time([b['hand_valid'] for b in valid], (2,), torch.bool)
            hand_score_padded = _pad_time([b['hand_score'] for b in valid], (2,), torch.float32)

        # Tokenize text ONCE here (not every forward pass)
        texts = [b['text'].strip() for b in valid]
        if self.mt5_tokenizer is not None:
            label_tokens = self.mt5_tokenizer(
                texts, padding="longest", truncation=True,
                max_length=self.max_text_tokens, return_tensors="pt",
            )
            label_ids = label_tokens['input_ids']
            label_ids[label_ids == self.mt5_tokenizer.pad_token_id] = -100
            label_attn = label_tokens['attention_mask']
        else:
            label_ids = None
            label_attn = None

        return {
            'keypoints': kps_padded,
            'input_lengths': input_lengths,
            'label_ids': label_ids,            # (B, L_text) — -100 for pads
            'label_attn_mask': label_attn,     # (B, L_text)
            'rgb': rgb_padded,                  # (B, T, rgb_dim) or None
            'prosody': prosody_padded,          # (B, T, 2) or None
            'hand_crops': hand_crops_padded,    # (B, T, 2, 112, 112, 3) uint8 or None
            'hand_ref': hand_ref_padded,        # (B, T, 2, 2) or None
            'hand_valid': hand_valid_padded,    # (B, T, 2) bool or None
            'hand_score': hand_score_padded,    # (B, T, 2) or None
            'texts': texts,                     # raw strings for generation eval
            # In the SAME (length-sorted) order as every tensor above. Any
            # consumer attaching per-clip metadata must key on these, never on
            # dataset order: this collator sorts by length and drops invalid
            # samples, so batch position != dataset position.
            'clip_ids': [b.get('clip_id', '') for b in valid],
        }


# ============================================================
# Helpers
# ============================================================

def build_pose_mask(kps, input_lengths, ratio):
    """
    Multi-granularity pose masking (SignBERT+, arXiv:2305.04868):
    the masking budget is split between three corruption types that target
    different keypoint-detector failure modes:
      - FRAME masking (50% of budget): whole frames zeroed — wrong pose.
      - SPAN masking (25%): a contiguous run of frames — wrong motion
        over a transition.
      - JOINT masking (25%): individual joints across frames — wrong
        single-landmark detections.

    Args:
        kps: (B, T, D) with D a multiple of 282
        input_lengths: (B,) valid frame counts
        ratio: total fraction of entries to corrupt

    Returns:
        bool mask (B, T, D), True = masked. Never masks every valid frame
        of a sample.
    """
    B, T, D = kps.shape
    device = kps.device
    n_blocks = max(D // 282, 1)

    valid = (torch.arange(T, device=device)[None, :]
             < input_lengths[:, None])                       # (B, T)

    # Frame-level
    frame_mask = torch.rand(B, T, device=device) < (ratio * 0.5)

    # Span-level: one contiguous span per sample, length ≈ ratio*0.25*T_valid
    span_len = (input_lengths.float() * ratio * 0.25).long().clamp(min=1)
    span_start = (torch.rand(B, device=device)
                  * (input_lengths - span_len).clamp(min=1).float()).long()
    t_idx = torch.arange(T, device=device)[None, :]
    span_mask = (t_idx >= span_start[:, None]) & (t_idx < (span_start + span_len)[:, None])

    frame_level = (frame_mask | span_mask) & valid           # (B, T)

    # Joint-level: mask whole joints (x,y pairs) at random frame-joint cells
    n_joints = 141  # 282 / 2
    joint_mask = torch.rand(B, T, n_joints, device=device) < (ratio * 0.25)
    joint_mask = joint_mask & valid[:, :, None]
    joint_cols = joint_mask.repeat_interleave(2, dim=2)      # (B, T, 282)

    # Combine and tile across all feature blocks (offset/abs/vel/acc/valid)
    mask282 = joint_cols | frame_level[:, :, None]
    mask = mask282.repeat(1, 1, n_blocks)[:, :, :D]

    # Keep at least one unmasked valid frame per sample
    fully_masked_frames = mask.all(dim=2)                    # (B, T)
    all_gone = (fully_masked_frames | ~valid).all(dim=1)     # (B,)
    if all_gone.any():
        first_valid = torch.zeros(B, dtype=torch.long, device=device)
        mask[all_gone.nonzero(as_tuple=True)[0], first_valid[all_gone]] = False

    return mask


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0

def log(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)


# ============================================================
# MT5 Wrapper Model
# ============================================================

class UniSignMT5(nn.Module):
    """
    Uni-Sign + MT5 wrapper.

    Matches Uni-Sign models.py forward pass:
      1. Pose encoder → pose_proj
      2. Prefix token embedding ("Translate sign language video to Kazakh: ")
      3. Concatenate prefix + pose embeddings
      4. Feed to MT5 as inputs_embeds
      5. MT5 generates text autoregressively
    """

    def __init__(self, encoder, mt5_path=MT5_PATH, lang="Kazakh", align_dim=None,
                 masked_pose_dim=None, ctc_vocab_size=None,
                 use_pgf=False, pgf_p_samp=0.5, prosody_aux_dim=None):
        """
        Args:
            encoder: KeypointEncoder
            mt5_path: HF path for MT5
            lang: target language for the task prefix
            masked_pose_dim: if set, build a masked-pose reconstruction head
                (d_model → masked_pose_dim). Living inside this module keeps
                it covered by DDP and lets the aux loss backprop into the
                encoder.
            ctc_vocab_size: if set, build a character-CTC head
                (d_model → vocab+1, blank=0). Forces frame-level features to
                align with the transcript — the standard fix when the decoder
                degenerates into a pure language model (fluent output, wrong
                content) because CE alone lets it ignore the video.
            use_pgf: if True, build Uni-Sign's real Prior-Guided Fusion
                (arXiv:2501.15187 Sec 3.3 + Appendix A.3 — see
                models/pgf_fusion.py). Pose-only diagnostics
                (diagnose_phase1.py) found the encoder's embeddings collapse
                to near-identical across genuinely different clips even
                after real training, with six architecture/training-side
                explanations ruled out by direct testing — this tests
                whether pose-only input was discarding exactly the fine
                finger detail RGB can supply. Fusion happens INSIDE the
                encoder (models/unisign_encoder.py's hand_fusion_fn hook,
                at the pre-pose_proj 256-dim per-group seam), not as a
                post-hoc concat like the earlier whole-frame placeholder —
                see _make_pgf_hook below.
            pgf_p_samp: fraction of frames per clip that get RGB fusion each
                step (paper Appendix A.3 Algorithm 1's score-aware sampling;
                the rest of that clip's frames stay pure pose that step).
        """
        super().__init__()
        self.encoder = encoder
        self.lang = lang

        self.ctc_head = None
        if ctc_vocab_size is not None:
            self.ctc_head = nn.Linear(encoder.hidden_dim, ctc_vocab_size + 1)

        # LayerNorm on pose embeddings before MT5. Diagnostics showed raw
        # encoder outputs at ~6x MT5's embedding scale with a 0.999-cosine
        # shared direction across clips — cross-attention saturates on the
        # constant and the decoder degenerates to corpus-prior loops.
        self.pose_norm = nn.LayerNorm(encoder.hidden_dim)

        # E5 pose-text alignment heads (train-only; unused at inference).
        # Deliberately NOT saved in checkpoints -- see save logic.
        self.align_dim = align_dim
        self.align_pose_head = None
        if align_dim:
            self.align_pose_head = nn.Linear(encoder.hidden_dim, align_dim)
            self.align_text_head = nn.Linear(encoder.hidden_dim, align_dim)
            # log(1/0.07), the CLIP initialisation; learnable and clamped in forward.
            self.align_logit_scale = nn.Parameter(torch.tensor(2.6593))

        self.masked_pose_decoder = None
        if masked_pose_dim is not None:
            self.masked_pose_decoder = build_masked_pose_decoder(
                encoder.hidden_dim, masked_pose_dim)

        # Prosody-as-supervision (ablation treatment arm). Predicts per-frame
        # speech [F0, energy] from the encoder embedding purely to pressure
        # the encoder into clip-discriminative representations -- see
        # build_prosody_aux_head's docstring. None => baseline arm.
        self.prosody_aux_head = None
        if prosody_aux_dim is not None:
            self.prosody_aux_head = build_prosody_aux_head(
                encoder.hidden_dim, prosody_aux_dim)

        # Prior-Guided Fusion (real Uni-Sign RGB branch). Operates at the
        # paper's C=256, matching KeypointEncoder's per-group pool_feat dim
        # BEFORE pose_proj (1024→768) -- the hook is called from inside
        # KeypointEncoder.forward for the left/right groups only, so no
        # 256↔768 reconciliation layer is needed here at all.
        self.use_pgf = use_pgf
        self.pgf_p_samp = pgf_p_samp
        self.hand_backbone = None
        self.pgf_hand_fusion = None
        self.pgf_gate = None
        self.pgf_keypoint_adapter = None
        if use_pgf:
            pgf_dim = 256
            self.hand_backbone = HandBackbone(out_channels=pgf_dim, pretrained=True)
            self.pgf_hand_fusion = DeformablePoseRGBAttention(
                embed_dim=pgf_dim, adapter_dim=32, num_heads=8)
            self.pgf_gate = FusionGate(embed_dim=pgf_dim)
            # Judgment call #4 (models/pgf_fusion.py docstring): bridges
            # pool_feat's 256-dim pose representation down to the 32-dim
            # adapter space to_q/to_k/to_v actually consume. No
            # corresponding checkpoint key in a colleague's shared weights
            # -- stays randomly initialized even after --convert-pgf.
            self.pgf_keypoint_adapter = nn.Linear(pgf_dim, 32)
            # ImageNet normalization stats -- HandBackbone is ImageNet-
            # pretrained (or converted from a colleague's checkpoint, also
            # ImageNet-pretrained per the paper), and hand crops are stored
            # as true RGB uint8 (see data/asan_dataset.py's load_hand_crops).
            self.register_buffer(
                '_pgf_imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer(
                '_pgf_imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            log(f"[PGF] Prior-Guided Fusion enabled (p_samp={pgf_p_samp})")

        # MT5
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(mt5_path)
        self.mt5_tokenizer = T5Tokenizer.from_pretrained(mt5_path, legacy=False)

        # Cache prefix token IDs as buffers (no grad, follows .to(device)/DDP)
        prefix = [f"Translate sign language video to {lang}: "]
        prefix_token = self.mt5_tokenizer(
            prefix, padding="longest", truncation=True, return_tensors="pt",
        )
        self.register_buffer('prefix_ids', prefix_token['input_ids'].squeeze(0))
        self.register_buffer('prefix_attn', prefix_token['attention_mask'].squeeze(0))

        log(f"[MT5] Loaded {mt5_path}")
        log(f"[MT5] Encoder params: {sum(p.numel() for p in self.encoder.parameters()):,}")
        log(f"[MT5] MT5 params: {sum(p.numel() for p in self.mt5.parameters()):,}")

    def _pose_mask(self, kps, input_lengths):
        """
        Build the (B, T) attention mask over pose frames.

        Prefer explicit input_lengths from the collator. The old fallback —
        counting non-zero frames — undercounts whenever frames inside the
        sequence are zero (e.g. masked-pose training), which silently cut
        the END of every sequence out of attention.
        """
        device = kps.device
        if input_lengths is None:
            input_lengths = (kps.abs().sum(dim=-1) > 0).int().sum(dim=1)  # (B,)
        t_max = kps.size(1)
        return (torch.arange(t_max, device=device)[None, :]
                < input_lengths.to(device)[:, None]).long()  # (B, T)

    def _make_pgf_hook(self, hand_crops, hand_ref, hand_valid, hand_score, input_lengths):
        """
        Builds the per-batch closure passed as KeypointEncoder's
        hand_fusion_fn (models/unisign_encoder.py), called once each for
        the 'left'/'right' groups during that forward pass.

        hand_crops: (B, T, 2, 112, 112, 3) uint8, hand axis [left, right]
        hand_ref:   (B, T, 2, 2) float32, normalized [-1,1] wrist reference
        hand_valid: (B, T, 2) bool -- whether that frame's crop is trustworthy
        hand_score: (B, T, 2) float32 -- mean keypoint confidence

        Returns None if hand_crops is None (no RGB data this batch, or
        --use-pgf wasn't set) -- KeypointEncoder.forward treats hand_fusion_fn=None
        as pose-only, exactly as before.
        """
        if not self.use_pgf or hand_crops is None:
            return None

        B, T = hand_crops.shape[:2]
        device = hand_crops.device
        hand_axis = {'left': 0, 'right': 1}

        # Score-aware sampling (paper Appendix A.3 Algorithm 1) is done
        # ONCE per batch, shared across both hands -- the paper samples
        # per CLIP, not per hand-per-clip. Average left/right confidence
        # since a single per-clip sampling decision needs one score.
        combined_score = hand_score.mean(dim=-1)  # (B, T)
        sampled_idx, sampled_mask = score_aware_sample_indices(
            combined_score, input_lengths, self.pgf_p_samp)  # (B, K), (B, K)
        K = sampled_idx.shape[1]
        b_grid = torch.arange(B, device=device).unsqueeze(1).expand(B, K)  # (B, K)

        def hook(mode, pool_feat, kps_raw, _input_lengths):
            h = hand_axis[mode]
            crops_bt = hand_crops[b_grid, sampled_idx, h]   # (B, K, 112, 112, 3) uint8
            ref_bt = hand_ref[b_grid, sampled_idx, h]        # (B, K, 2)
            valid_bt = hand_valid[b_grid, sampled_idx, h] & sampled_mask  # (B, K)
            pose_bt = pool_feat[b_grid, sampled_idx]         # (B, K, 256)

            flat_valid = valid_bt.reshape(-1)
            if not flat_valid.any():
                # No valid RGB samples this step for this hand (can happen
                # if every sampled frame had an undetected hand). Still
                # touch every PGF param with a zero-valued term so DDP's
                # find_unused_parameters=False doesn't choke on unused
                # parameters (same trick this file already uses for
                # masked_pose_decoder/ctc_head above).
                zero = (self.hand_backbone.rgb_proj.weight.sum() * 0.0
                        + self.pgf_hand_fusion.to_out.weight.sum() * 0.0
                        + self.pgf_gate.net[-1].weight.sum() * 0.0
                        + self.pgf_keypoint_adapter.weight.sum() * 0.0)
                return pool_feat + zero

            b_flat = b_grid.reshape(-1)[flat_valid]
            t_flat = sampled_idx.reshape(-1)[flat_valid]
            crops_flat = crops_bt.reshape(-1, 112, 112, 3)[flat_valid]
            ref_flat = ref_bt.reshape(-1, 2)[flat_valid]
            pose_flat = pose_bt.reshape(-1, pose_bt.shape[-1])[flat_valid]  # (N, 256)

            rgb_in = crops_flat.permute(0, 3, 1, 2).float() / 255.0  # (N, 3, 112, 112)
            rgb_in = (rgb_in - self._pgf_imagenet_mean) / self._pgf_imagenet_std
            rgb_map = self.hand_backbone(rgb_in)  # (N, 256, 4, 4)

            pose_adapter = self.pgf_keypoint_adapter(pose_flat)  # (N, 32)
            f_hat = self.pgf_hand_fusion(pose_adapter, rgb_map, ref_flat)  # (N, 256)
            f_final, _ = self.pgf_gate(pose_flat, f_hat)  # (N, 256)

            # Scatter fused features back; frames not selected here (either
            # never sampled, or sampled but invalid) stay bit-identical to
            # the pure-pose pool_feat this hook received as input.
            out = pool_feat.clone()
            out[b_flat, t_flat] = f_final
            return out

        return hook

    def _alignment_loss(self, pose_emb, kps, input_lengths, text_vecs,
                        neg_vecs=None, neg_mask=None):
        """
        InfoNCE between a clip's pooled pose embedding and its frozen text-teacher
        vector (scripts/cache_text_embeddings.py).

        Why this exists: E3 showed training is bimodal -- the decoder either starts
        using the pose input ("takeoff") or regresses to the corpus's most frequent
        sentence. Only 1 of 8 runs took off. Cross-entropy alone never requires the
        encoder to carry information, so this adds a term that is only minimised by
        pose representations that identify their own transcript.

        Pose side: mean over VALID frames only (padding would otherwise dilute
        short clips differently depending on their batch).
        Negatives: in-batch plus optional rows sampled from the cached table, so the
        count is not capped by the batch size of 8. `neg_mask` marks sampled
        negatives whose text is identical to the positive (2.6% of references repeat)
        -- those are masked out instead of being pushed apart.
        """
        B, T, _ = pose_emb.shape
        if input_lengths is not None:
            valid = (torch.arange(T, device=pose_emb.device)[None, :]
                     < input_lengths.to(pose_emb.device)[:, None]).unsqueeze(-1)
        else:
            valid = (kps.abs().sum(-1, keepdim=True) > 0)
        v = valid.to(pose_emb.dtype)
        pooled = (pose_emb * v).sum(1) / v.sum(1).clamp(min=1.0)          # (B, D)

        zp = F.normalize(self.align_pose_head(pooled), dim=-1)            # (B, d)
        zt = F.normalize(self.align_text_head(text_vecs.to(pooled.dtype)), dim=-1)
        scale = self.align_logit_scale.clamp(max=4.6052).exp()            # <= 100

        logits = scale * zp @ zt.t()                                      # (B, B)
        if neg_vecs is not None and neg_vecs.numel():
            zn = F.normalize(self.align_text_head(neg_vecs.to(pooled.dtype)), dim=-1)
            extra = scale * zp @ zn.t()                                   # (B, K)
            if neg_mask is not None:
                extra = extra.masked_fill(neg_mask, float('-inf'))
            logits = torch.cat([logits, extra], dim=1)                    # (B, B+K)

        target = torch.arange(B, device=logits.device)
        # Pose->text only: the text side is frozen and cached, so the symmetric
        # text->pose direction would just train the text head against itself.
        return F.cross_entropy(logits, target)

    def forward(self, kps, label_ids, label_attn_mask, input_lengths=None,
                kps_target=None, frame_mask=None,
                hand_crops=None, hand_ref=None, hand_valid=None, hand_score=None,
                prosody_target=None, align_text=None, align_negatives=None,
                align_neg_mask=None):
        """
        Training forward pass.

        Args:
            kps: (B, T, D) — keypoints (282 or 1128); may be frame-masked
            label_ids: (B, L) — pre-tokenized label ids (-100 for pads)
            label_attn_mask: (B, L) — attention mask for labels
            input_lengths: (B,) — true frame counts from the collator
            kps_target: (B, T, D) — clean keypoints (masked-pose aux target)
            frame_mask: (B, T, 1) or (B, T, D) — True at masked entries
                (broadcasts over D; supports frame- and joint-level masks)
            hand_crops, hand_ref, hand_valid, hand_score: Prior-Guided
                Fusion inputs from scripts/extract_asan_hand_crops.py via
                the collator (all None unless the model was built with
                use_pgf=True — see _make_pgf_hook above)

            prosody_target: (B, T, 2) — per-frame [F0, energy] from
                data/asan_dataset.py's corpus-normalized prosody, used ONLY
                as auxiliary encoder supervision (never synthesized). None
                in the baseline arm of the ablation.

        Returns:
            loss: scalar CE loss
            mse_loss: masked-pose reconstruction loss (or None)
            ctc_log_probs: (T, B, V+1) log-probs for CTC (or None)
            prosody_aux_loss: prosody-supervision loss (or None)
            align_loss: pose-text InfoNCE (or None). See _alignment_loss.
        """
        B = kps.size(0)

        # Pose encoder — runs on the (possibly masked) input. Reconstructing
        # the masked frames from THESE embeddings gives the encoder a
        # gradient signal to encode temporal context (the previous version
        # reconstructed from clean, no-grad embeddings, so the aux loss only
        # ever trained the small MLP head).
        # input_lengths re-zeroes pad frames before the encoder's temporal
        # conv so batch padding can't bleed into real boundary frames.
        # Prior-Guided Fusion happens INSIDE the encoder (hand_fusion_fn
        # hook, called for the left/right groups at the pre-pose_proj
        # 256-dim seam) rather than as a post-hoc concat.
        pgf_hook = self._make_pgf_hook(hand_crops, hand_ref, hand_valid, hand_score, input_lengths)
        pose_emb = self.pose_norm(self.encoder(
            kps, input_lengths=input_lengths, hand_fusion_fn=pgf_hook))  # (B, T, 768)

        # E5: pose-text alignment on the pooled pose embedding.
        align_loss = None
        if align_text is not None and self.align_pose_head is not None:
            align_loss = self._alignment_loss(pose_emb, kps, input_lengths,
                                              align_text, align_negatives,
                                              align_neg_mask)

        # Prefix embeds: re-embed each forward for grad correctness
        # (~10 token lookup is free, avoids backward-through-cached-graph bugs)
        prefix_embeds = self.mt5.shared(self.prefix_ids.unsqueeze(0).expand(B, -1))
        prefix_attn = self.prefix_attn.unsqueeze(0).expand(B, -1)

        pose_mask = self._pose_mask(kps, input_lengths)

        # Concatenate: prefix + pose embeddings
        inputs_embeds = torch.cat([prefix_embeds, pose_emb], dim=1)

        # Attention mask: prefix + pose (encoder inputs only)
        attention_mask = torch.cat([prefix_attn, pose_mask], dim=1)

        # MT5 forward
        out = self.mt5(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=label_ids,
            return_dict=True,
        )

        # Masked-pose reconstruction aux loss. The decoder always runs when
        # a frame_mask is supplied so that DDP (find_unused_parameters=False)
        # sees its params participate even if the random mask selected zero
        # frames this step.
        mse_loss = None
        if (self.masked_pose_decoder is not None and frame_mask is not None
                and kps_target is not None):
            reconstructed = self.masked_pose_decoder(pose_emb)  # (B, T, D)
            sel = frame_mask.expand_as(kps_target)
            if sel.any():
                mse_loss = F.mse_loss(reconstructed[sel], kps_target[sel])
            else:
                mse_loss = reconstructed.sum() * 0.0

        # CTC head runs whenever it exists (keeps DDP happy); the trainer
        # decides whether/how to weight the loss.
        ctc_log_probs = None
        if self.ctc_head is not None:
            ctc_log_probs = self.ctc_head(pose_emb).log_softmax(-1).transpose(0, 1)

        # Prosody-as-supervision aux loss (ablation treatment arm). Masked to
        # valid frames so padding can't dominate; runs whenever the head
        # exists so DDP sees its params participate every step.
        prosody_aux_loss = None
        if self.prosody_aux_head is not None:
            prosody_pred = self.prosody_aux_head(pose_emb)  # (B, T, 2)
            if prosody_target is not None:
                T_min = min(prosody_pred.size(1), prosody_target.size(1))
                pred = prosody_pred[:, :T_min]
                tgt = prosody_target[:, :T_min]
                valid = (torch.arange(T_min, device=pred.device)[None, :]
                         < input_lengths.to(pred.device)[:, None]).unsqueeze(-1)
                if valid.any():
                    prosody_aux_loss = (F.mse_loss(pred, tgt, reduction='none')
                                        * valid).sum() / (valid.sum() * pred.size(-1))
                else:
                    prosody_aux_loss = prosody_pred.sum() * 0.0
            else:
                prosody_aux_loss = prosody_pred.sum() * 0.0

        return out.loss, mse_loss, ctc_log_probs, prosody_aux_loss, align_loss

    def generate(self, kps, input_lengths=None, max_new_tokens=128, num_beams=4,
                hand_crops=None, hand_ref=None, hand_valid=None, hand_score=None,
                no_repeat_ngram_size=3, repetition_penalty=1.3):
        """
        Inference: generate text from keypoints.

        Args:
            kps: (B, T, D)
            input_lengths: (B,) — true frame counts (optional)
            max_new_tokens: max output tokens
            num_beams: beam width
            hand_crops, hand_ref, hand_valid, hand_score: Prior-Guided
                Fusion inputs, or None (see forward()/_make_pgf_hook above)

        Returns:
            list of decoded strings
        """
        B = kps.size(0)

        pgf_hook = self._make_pgf_hook(hand_crops, hand_ref, hand_valid, hand_score, input_lengths)
        pose_emb = self.pose_norm(self.encoder(
            kps, input_lengths=input_lengths, hand_fusion_fn=pgf_hook))  # (B, T, 768)

        prefix_embeds = self.mt5.shared(self.prefix_ids.unsqueeze(0).expand(B, -1))
        prefix_attn = self.prefix_attn.unsqueeze(0).expand(B, -1)

        pose_mask = self._pose_mask(kps, input_lengths)
        attention_mask = torch.cat([prefix_attn, pose_mask], dim=1)
        inputs_embeds = torch.cat([prefix_embeds, pose_emb], dim=1)

        output_ids = self.mt5.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            # Suppress degenerate loops ("екінші кезеңде екінші кезеңде…"):
            # they dominate early-training beams and inflate WER via
            # insertions far past 1.0. Defaults preserve the historical
            # behaviour; they are parameters so E1's dev-only decoding sweep
            # can test whether they also suppress legitimate repetition.
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
        )

        decoded = self.mt5_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return decoded


# ============================================================
# Trainer
# ============================================================

class MT5Trainer:
    def __init__(self, config_path='configs/config.yaml', local_rank=0,
                 pretrained_encoder=None, pretrained_unisign=None,
                 freeze_spatial=False, use_lora=False, lora_r=16, lora_alpha=32,
                 use_enriched=False, signspace=False,
                 selection_manifest=None, block_padding_mask=False,
                 real_wrists=False, score_quantiles=None,
                 unisign_preprocess=False, unisign_hand_scale=None,
                 align_cache=None, align_weight=0.0, align_negatives=256,
                 align_dim=256,
                 masked_pose_ratio=0.0, overfit_n=0,
                 ctc_weight=0.0, ctc_vocab_size=2000, resume=None,
                 grad_accum=None, encoder_lr=None,
                 use_pgf=False, hand_crop_root=None, pgf_p_samp=0.5,
                 pretrained_pgf=None, prosody_aux_weight=0.0, prosody_root=None,
                 select_metric='wer'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        from utils.paths import apply_env_overrides
        self.config = apply_env_overrides(self.config)

        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{local_rank}')
        self.cfg = self.config['model']
        self.train_cfg = self.config['training']['phase1']
        self.use_enriched = use_enriched
        self.signspace = signspace
        self.real_wrists = real_wrists
        self.score_quantiles = score_quantiles
        self.unisign_preprocess = unisign_preprocess
        self.unisign_hand_scale = unisign_hand_scale
        self.align_weight = align_weight
        self.align_negatives = align_negatives
        self._align_running = 0.0
        self._align_vecs = self._align_hash = self._align_index = None
        if align_cache:
            import numpy as _np
            z = _np.load(os.path.expanduser(align_cache), allow_pickle=True)
            self._align_vecs = torch.from_numpy(z['vecs'].astype('float32'))
            self._align_hash = torch.from_numpy(z['text_hash'])
            self._align_index = {c: i for i, c in enumerate(z['clip_ids'].tolist())}
            log(f"[Align] text teacher {z['teacher']}: {len(self._align_index)} clips, "
                f"dim {self._align_vecs.shape[1]}, weight {align_weight}, "
                f"{align_negatives} sampled negatives")
        elif align_weight > 0:
            raise ValueError("--align-weight needs --align-cache")
        self.selection_manifest = selection_manifest
        self.gen_loader = None
        self._gen_target = None
        self.masked_pose_ratio = masked_pose_ratio
        self.overfit_n = overfit_n
        self.ctc_weight = ctc_weight
        # Prosody-as-supervision ablation: 0.0 = baseline arm (head not
        # even built), >0 = treatment arm.
        self.prosody_aux_weight = prosody_aux_weight
        self.prosody_root = prosody_root
        self.freeze_spatial = freeze_spatial
        self.use_pgf = use_pgf
        self.hand_crop_root = hand_crop_root
        self.pgf_p_samp = pgf_p_samp
        # Flags that determine model/optimizer structure — saved into the
        # checkpoint so --resume can verify the resuming run uses the same
        # architecture (a mismatch here breaks state_dict loads far less
        # clearly than this explicit check does).
        self._run_args = dict(
            use_enriched=use_enriched, signspace=signspace,
            selection_manifest=selection_manifest,
            block_padding_mask=block_padding_mask,
            real_wrists=real_wrists,
            score_quantiles=score_quantiles,
            unisign_preprocess=unisign_preprocess,
            unisign_hand_scale=unisign_hand_scale,
            align_weight=align_weight, align_negatives=align_negatives,
            align_dim=align_dim if align_weight > 0 else None,
            masked_pose_ratio=masked_pose_ratio,
            ctc_weight=ctc_weight, ctc_vocab_size=ctc_vocab_size,
            freeze_spatial=freeze_spatial, use_lora=use_lora,
            use_pgf=use_pgf, pgf_p_samp=pgf_p_samp if use_pgf else None,
            prosody_aux_weight=prosody_aux_weight,
        )

        # Subword-BPE vocabulary for the CTC auxiliary loss (id 0 = blank).
        # Character-level CTC is unusable here: sign clips are ~223 frames vs
        # ~258 transcript chars, so >2/3 of samples have T < L and are zeroed
        # by ctc_loss(zero_infinity=True). BPE pieces (~2.5 chars each) cut L
        # to ~100, giving frames > targets so CTC actually receives gradient.
        self.ctc_tokenizer = None
        self.ctc_vocab_size = None
        if ctc_weight > 0:
            self.ctc_tokenizer = self._build_bpe_tokenizer(
                self.config['paths'], vocab_size=ctc_vocab_size)
            self.ctc_vocab_size = self.ctc_tokenizer.get_piece_size()
            log(f"[CTC] subword-BPE vocab: {self.ctc_vocab_size} pieces, "
                f"weight={ctc_weight}")

        # Determine input dimension
        if unisign_preprocess:
            from data.utils import UNISIGN_DIM
            input_dim = UNISIGN_DIM
        else:
            input_dim = ENRICHED_DIM() if use_enriched else KEYPOINT_DIM

        # Build encoder
        self.encoder = KeypointEncoder(
            hidden_dim=self.cfg['d_model'],
            input_dim=input_dim,
            block_padding_mask=block_padding_mask,
            real_wrists=real_wrists,
            unisign_input=unisign_preprocess,
        )

        # Option 1: Load raw Uni-Sign pretrained weights
        if pretrained_unisign:
            log(f"Loading Uni-Sign pretrained weights from {pretrained_unisign}")
            load_unisign_weights(self.encoder, pretrained_unisign)

        # Option 2: Load our fine-tuned encoder checkpoint
        if pretrained_encoder:
            log(f"Loading encoder from {pretrained_encoder}")
            checkpoint = torch.load(pretrained_encoder, map_location='cpu')
            if 'encoder' in checkpoint:
                self.encoder.load_state_dict(checkpoint['encoder'])
            else:
                self.encoder.load_state_dict(checkpoint)
            log("  Encoder loaded")

        # Wrap with MT5. The masked-pose decoder lives inside the model so
        # that (a) DDP keeps it in sync across ranks and (b) the aux loss
        # backprops into the encoder.
        self.model = UniSignMT5(
            align_dim=align_dim if align_weight > 0 else None,
            encoder=self.encoder, lang="Kazakh",
            masked_pose_dim=input_dim if masked_pose_ratio > 0 else None,
            ctc_vocab_size=self.ctc_vocab_size if self.ctc_tokenizer else None,
            use_pgf=use_pgf, pgf_p_samp=pgf_p_samp,
            prosody_aux_dim=2 if prosody_aux_weight > 0 else None,
        )
        if masked_pose_ratio > 0:
            log(f"[Masked Pose] Reconstruction decoder: {self.cfg['d_model']} → {input_dim}")
            log(f"[Masked Pose] Mask ratio: {masked_pose_ratio}")
        if use_pgf:
            log(f"[PGF] Prior-Guided Fusion enabled, p_samp={pgf_p_samp}")
            log(f"[PGF] Hand-crop root: {hand_crop_root}")

        # Option 3: seed PGF submodules from a converted colleague checkpoint
        # (scripts/convert_friend_checkpoint.py --convert-pgf) while still
        # using OUR OWN encoder (--pretrained-encoder above) -- unlike
        # --resume, this loads each PGF submodule independently rather than
        # all-or-nothing, since the converted checkpoint deliberately never
        # has pgf_keypoint_adapter (no equivalent in the colleague's
        # architecture -- see that script's docstring) and an all-or-nothing
        # check would always reject it. hand_backbone/pgf_hand_fusion are
        # the two submodules actually worth transferring (verified genuinely
        # trained against the real checkpoint); pgf_gate is included too in
        # case a future source checkpoint has a real one, but
        # convert_friend_checkpoint.py already keeps our own safe init in
        # place of an untrained source gate, so loading it here is a no-op
        # for that specific file.
        if use_pgf and pretrained_pgf:
            log(f"[PGF] Loading PGF submodules from {pretrained_pgf}")
            pgf_ckpt = torch.load(pretrained_pgf, map_location='cpu')
            pgf_submodules = {
                'hand_backbone': self.model.hand_backbone,
                'pgf_hand_fusion': self.model.pgf_hand_fusion,
                'pgf_gate': self.model.pgf_gate,
                'pgf_keypoint_adapter': self.model.pgf_keypoint_adapter,
            }
            for key, module in pgf_submodules.items():
                if key in pgf_ckpt:
                    module.load_state_dict(pgf_ckpt[key])
                    log(f"  {key}: loaded")
                else:
                    log(f"  {key}: not in checkpoint, keeping fresh init")

        # --- LoRA setup (optional, via peft) ---
        # MUST happen BEFORE the DDP wrap: DDP registers parameters at wrap
        # time, so swapping modules afterwards breaks gradient sync.
        self.use_lora = use_lora
        self.lora_params = []
        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                log(f"[LoRA] Applying LoRA to MT5 (r={lora_r}, alpha={lora_alpha})")
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=["q", "v"],
                    lora_dropout=0.1,
                    bias="none",
                )
                self.model.mt5 = get_peft_model(self.model.mt5, lora_config)
                self.model.mt5.print_trainable_parameters()
                self.lora_params = [p for p in self.model.mt5.parameters() if p.requires_grad]
            except ImportError:
                log("[WARN] peft not installed. Install with: pip install peft")
                log("[WARN] Falling back to full MT5 fine-tuning")
                self.use_lora = False

        self.model.to(self.device)

        self.distributed = dist.is_initialized()
        if self.distributed:
            self.model = DDP(
                self.model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False,
            )

        # Optimizer — differential LR
        base_lr = self.train_cfg.get('learning_rate', 5e-4)
        # Default: base_lr/10, on the theory that the CSL-pretrained encoder
        # only needs gentle adaptation. diagnose_phase1.py's (B)/(B3)/(D)
        # checks found the encoder still produces near-collapsed embeddings
        # across genuinely different clips (cosine ~0.98-0.995) even after
        # 13 real epochs at this rate, while the decoder DOES attend to the
        # pose frames substantially and still can't discriminate them --
        # i.e. base_lr/10 may be too conservative for what the encoder
        # actually needs to learn (KRSL-specific discrimination, not just
        # light CSL adaptation). --encoder-lr overrides this directly.
        encoder_lr = encoder_lr if encoder_lr is not None else base_lr / 10

        if freeze_spatial:
            self.encoder.freeze_spatial()
            encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        else:
            encoder_params = list(self.encoder.parameters())

        param_groups = [{'params': encoder_params, 'lr': encoder_lr}]

        core = self.model.module if self.distributed else self.model
        if self.use_lora:
            param_groups.append({'params': self.lora_params, 'lr': base_lr})
        else:
            param_groups.append({'params': core.mt5.parameters(), 'lr': base_lr})

        # pose_norm bridges encoder → MT5; param groups are explicit, so it
        # must be added or it would silently never train
        param_groups.append({'params': core.pose_norm.parameters(), 'lr': base_lr})

        if core.masked_pose_decoder is not None:
            param_groups.append({'params': core.masked_pose_decoder.parameters(), 'lr': base_lr})

        if core.ctc_head is not None:
            param_groups.append({'params': core.ctc_head.parameters(), 'lr': base_lr})

        if core.prosody_aux_head is not None:
            param_groups.append({'params': core.prosody_aux_head.parameters(), 'lr': base_lr})

        if core.hand_backbone is not None:
            param_groups.append({'params': core.hand_backbone.parameters(), 'lr': base_lr})
            param_groups.append({'params': core.pgf_hand_fusion.parameters(), 'lr': base_lr})
            param_groups.append({'params': core.pgf_gate.parameters(), 'lr': base_lr})
            param_groups.append({'params': core.pgf_keypoint_adapter.parameters(), 'lr': base_lr})

        self.optimizer = AdamW(param_groups, weight_decay=0.01)

        self.warmup_steps = self.train_cfg.get('warmup_steps', 1500)
        # config.yaml's grad_accum (4) is sized for the full asan-dataset
        # run. With --overfit-n on a handful of clips, len(train_loader) is
        # tiny (e.g. 4 batches for 30 clips at batch_size=8), so grad_accum=4
        # collapses to ~1 optimizer step per epoch — nowhere near enough
        # updates to memorize anything, regardless of loss weighting. This
        # override lets sanity checks and other ad-hoc runs set their own
        # value without editing the shared config.
        effective_grad_accum = grad_accum if grad_accum is not None \
            else self.train_cfg.get('grad_accum', 1)
        self.grad_accum = max(1, int(effective_grad_accum))
        self.global_step = 0
        self.start_epoch = 0
        self.scheduler = None
        self.encoder_total_params = sum(p.numel() for p in self.encoder.parameters())
        core = self.model.module if self.distributed else self.model
        self.mt5_params = sum(p.numel() for p in core.mt5.parameters())

        log(f"\n[MT5 Trainer] Differential LR training:")
        log(f"  Encoder LR: {encoder_lr:.6f} ({self.encoder_total_params:,} params)")
        if self.use_lora:
            lora_n = sum(p.numel() for p in self.lora_params)
            log(f"  MT5 LoRA LR: {base_lr:.6f} (r={lora_r}, alpha={lora_alpha}, {lora_n:,} trainable)")
        else:
            log(f"  MT5 LR:     {base_lr:.6f} ({self.mt5_params:,} params)")
        log(f"  Warmup:     {self.warmup_steps} steps, then cosine decay")
        if freeze_spatial:
            log(f"  Frozen:     spatial STGCN + projection")
        if use_enriched:
            log(f"  Features:   enriched (offset + velocity + acc + validity)")
        if masked_pose_ratio > 0:
            log(f"  Aux loss:   masked-pose reconstruction (ratio={masked_pose_ratio})")

        self.max_epochs = self.train_cfg.get('max_epochs', 20)
        self.best_loss = float('inf')
        # Best-checkpoint selection metric. Default WER (generation
        # quality) rather than val CE -- see SELECT_METRICS.
        self.select_metric = select_metric
        self.best_score = (float('-inf')
                           if SELECT_METRICS[select_metric] == 'higher'
                           else float('inf'))

        # --- Resume: full trainer state (model + optimizer + step count) ---
        # Unlike --pretrained-encoder (which loads ONLY the encoder submodule
        # and is meant for starting a NEW phase/run), --resume restores the
        # exact state of an interrupted or completed run so training can
        # continue past it — including the fine-tuned MT5/LoRA weights and
        # the CTC/masked-pose heads that --pretrained-encoder silently drops.
        # Placed after best_loss/max_epochs are set above so resume can
        # override best_loss with the checkpoint's actual value.
        if resume:
            self._load_resume_checkpoint(resume)

        log(f"\n[MT5 Trainer] Encoder: {self.encoder_total_params:,} | MT5: {self.mt5_params:,}")
        log(f"[MT5 Trainer] Device: {self.device}")

    def _load_resume_checkpoint(self, path):
        """
        Restore a full trainer state saved by _build_checkpoint: model
        weights (encoder, pose_norm, mT5/LoRA, CTC head, masked-pose
        decoder), optimizer state, and step/epoch counters — so training
        continues exactly where it left off instead of quietly restarting
        the decoder from its pretrained-HuggingFace state, which is what
        --pretrained-encoder does (it only ever touches the encoder).
        """
        log(f"[Resume] Loading trainer state from {path}")
        ckpt = torch.load(path, map_location='cpu')

        saved_args = ckpt.get('run_args', {})
        mismatches = {
            k: (self._run_args[k], saved_args[k]) for k in saved_args
            if k in self._run_args and self._run_args[k] != saved_args[k]
        }
        if mismatches:
            raise ValueError(
                f"[Resume] Architecture flags differ from the checkpoint's "
                f"run (current, saved): {mismatches}. Resume must use the "
                f"same flags the checkpoint was trained with — a mismatch "
                f"here means the model/optimizer structure won't line up.")

        core = self.model.module if self.distributed else self.model
        core.encoder.load_state_dict(ckpt['encoder'])
        # Unlike ctc_head/masked_pose_decoder below (hard-fail if missing),
        # this used to be unconditional -- broke loading any checkpoint
        # that never had a
        # pose_norm bridge layer at all (e.g. an externally-sourced encoder+
        # mt5 checkpoint converted from a different architecture that
        # normalizes the pose embedding some other way, or doesn't need to).
        # A missing key here just means pose_norm starts from its default
        # (identity-ish) init and adapts during training, same tradeoff as
        # the other optional components.
        if 'pose_norm' in ckpt:
            core.pose_norm.load_state_dict(ckpt['pose_norm'])
        else:
            log("[Resume] WARNING: checkpoint has no pose_norm -- starting "
                "it from default init.")

        if self.use_lora:
            if 'mt5_lora' not in ckpt:
                raise ValueError("[Resume] --use-lora is set but the "
                                 "checkpoint has no 'mt5_lora' weights.")
            core.mt5.load_state_dict(ckpt['mt5_lora'], strict=False)
        else:
            if 'mt5' not in ckpt:
                raise ValueError("[Resume] checkpoint has no full 'mt5' "
                                 "weights (was it saved with --use-lora?).")
            core.mt5.load_state_dict(ckpt['mt5'])

        if core.prosody_aux_head is not None:
            if 'prosody_aux_head' in ckpt:
                core.prosody_aux_head.load_state_dict(ckpt['prosody_aux_head'])
            else:
                log('[Resume] NOTE: --prosody-aux-weight > 0 but checkpoint has '
                    'no prosody head -- starting it fresh (expected when '
                    'branching the treatment arm off a baseline checkpoint).')

        if core.ctc_head is not None:
            if 'ctc_head' not in ckpt:
                raise ValueError("[Resume] --ctc-weight > 0 but the "
                                 "checkpoint has no CTC head.")
            core.ctc_head.load_state_dict(ckpt['ctc_head'])

        if core.masked_pose_decoder is not None:
            if 'masked_pose_decoder' not in ckpt:
                raise ValueError("[Resume] --masked-pose-ratio > 0 but the "
                                 "checkpoint has no masked-pose decoder.")
            core.masked_pose_decoder.load_state_dict(ckpt['masked_pose_decoder'])

        if core.hand_backbone is not None:
            # Soft-fail (unlike ctc_head/masked_pose_decoder above): a
            # checkpoint converted from a colleague's weights via
            # scripts/convert_friend_checkpoint.py WITHOUT --convert-pgf is
            # a normal, expected resume target that simply has no PGF
            # weights yet -- not a user error like a genuine --ctc-weight/
            # --masked-pose-ratio mismatch would be.
            pgf_keys = ('hand_backbone', 'pgf_hand_fusion', 'pgf_gate', 'pgf_keypoint_adapter')
            if all(k in ckpt for k in pgf_keys):
                core.hand_backbone.load_state_dict(ckpt['hand_backbone'])
                core.pgf_hand_fusion.load_state_dict(ckpt['pgf_hand_fusion'])
                core.pgf_gate.load_state_dict(ckpt['pgf_gate'])
                core.pgf_keypoint_adapter.load_state_dict(ckpt['pgf_keypoint_adapter'])
            else:
                log("[Resume] WARNING: --use-pgf is set but the checkpoint "
                    "has no PGF weights -- starting PGF modules from their "
                    "default init (hand_backbone from ImageNet pretrain, "
                    "everything else random). Expected if resuming from a "
                    "checkpoint converted without --convert-pgf.")

        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.global_step = ckpt.get('global_step', 0)
        else:
            # No optimizer state to restore momentum/variance from, and
            # crucially no 'initial_lr' seeded into param_groups (only a
            # scheduler construction or a loaded optimizer state_dict sets
            # that) — train() would hit a hard KeyError if it then tried to
            # fast-forward a scheduler via last_epoch=global_step-1 on a
            # fresh, scheduler-naive optimizer. Force global_step to 0 so
            # train() takes its normal last_epoch=-1 path instead: a "soft"
            # resume that restarts the LR schedule from warmup but still
            # skips the epochs already completed.
            log("[Resume] WARNING: checkpoint has no optimizer state (saved "
                "by an older run) — Adam momentum/variance restart at zero, "
                "and the LR schedule restarts from warmup instead of "
                "continuing mid-decay (the epoch counter still resumes "
                "correctly).")
            self.global_step = 0
        self.start_epoch = ckpt.get('epoch', -1) + 1
        # Coerce None to inf explicitly: a converted external checkpoint
        # (scripts/convert_friend_checkpoint.py) stores val_loss=None, and
        # dict.get's default only fires for a MISSING key, not a present-
        # but-None one -- so best_loss would otherwise stay None and crash
        # the :.4f format below (and later best-checkpoint comparisons).
        self.best_loss = ckpt.get('val_loss')
        if self.best_loss is None:
            self.best_loss = float('inf')
        # Restore the best-so-far score only if the checkpoint was selected
        # on the SAME metric; otherwise the numbers aren't comparable and we
        # restart the search (a stale best would block all future saves).
        ckpt_metric = ckpt.get('select_metric')
        if ckpt_metric == self.select_metric and ckpt.get('best_score') is not None:
            self.best_score = ckpt['best_score']
            log(f"[Resume] best_score ({self.select_metric}) restored: "
                f"{self.best_score:.4f}")
        elif ckpt_metric is not None and ckpt_metric != self.select_metric:
            log(f"[Resume] checkpoint was selected on '{ckpt_metric}' but this "
                f"run uses '{self.select_metric}' -- restarting best-score "
                f"search from scratch.")
        else:
            log(f"[Resume] checkpoint predates metric-based selection -- "
                f"restarting best-score search on '{self.select_metric}'.")

        log(f"[Resume] checkpoint was at epoch {ckpt.get('epoch')} -> "
            f"continuing from epoch {self.start_epoch + 1}, "
            f"global_step={self.global_step}, best_loss={self.best_loss:.4f}")

    @staticmethod
    def _build_bpe_tokenizer(paths, vocab_size=2000):
        """
        SentencePiece BPE tokenizer for the CTC auxiliary loss, trained on the
        asan training transcripts. Trained once and cached at
        ctc_bpe_<vocab>.model in the working dir; reused on later runs.

        Piece ids are used directly as CTC targets AFTER a +1 shift (id 0 is
        reserved for the CTC blank), so the CTC head has vocab_size + 1 outputs.
        BPE (vs characters) is the fix for the frames<chars length problem:
        ~2.5 chars/piece roughly halves the target length so T > L holds.
        """
        import json as _json
        import sentencepiece as spm

        model_path = os.path.join(os.getcwd(), f'ctc_bpe_{vocab_size}.model')
        if not os.path.exists(model_path):
            asan = paths.get('asan', {})
            root = asan.get('root', '')
            corpus = os.path.join(os.getcwd(), f'ctc_bpe_{vocab_size}_corpus.txt')
            n = 0
            with open(corpus, 'w') as out:
                for source in asan.get('sources', []):
                    ann = os.path.join(root, source, 'annotations',
                                       asan.get('lang', 'kz'), 'train.json')
                    if os.path.exists(ann):
                        with open(ann) as f:
                            for e in _json.load(f):
                                t = e.get('text', '').strip()
                                if t:
                                    out.write(t.lower() + '\n')
                                    n += 1
            log(f"[CTC] training BPE (vocab={vocab_size}) on {n} transcripts...")
            spm.SentencePieceTrainer.train(
                input=corpus, model_prefix=model_path[:-6],
                vocab_size=vocab_size, model_type='bpe',
                character_coverage=1.0,
                # unk maps to id 0 within SP; after our +1 shift it becomes 1,
                # leaving CTC blank=0 free. No bos/eos/pad in CTC targets.
                unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1,
            )
        return spm.SentencePieceProcessor(model_file=model_path)

    def create_datasets(self):
        split_ratio = 0.9
        paths = self.config['paths']
        all_train = []
        all_val = []

        # asan-dataset — predefined video-disjoint train/dev/test splits
        if 'asan' in paths and os.path.exists(paths['asan'].get('root', '')):
            asan_cfg = paths['asan']
            asan_common = dict(
                root=asan_cfg['root'],
                sources=asan_cfg.get('sources',
                                     ['informburo', 'khabar', 'qazaqstantv']),
                lang=asan_cfg.get('lang', 'kz'),
                tokenizer=None,
                max_frames=self.train_cfg['max_seq_len'],
                downsample_every=asan_cfg.get('downsample_every', 1),
                use_enriched=self.use_enriched,
                signspace=self.signspace,
                real_wrists=self.real_wrists,
                score_quantiles=self.score_quantiles,
                unisign_preprocess=self.unisign_preprocess,
                unisign_hand_scale=self.unisign_hand_scale,
                skip_low_quality=asan_cfg.get('skip_low_quality', True),
                min_hand_cov=asan_cfg.get('min_hand_cov', 0.0),
                # Hand crops are only extracted for asan-dataset (khabar_kz/
                # informburo below have no PGF support) -- a batch mixing
                # sources would just see hand_crops=None for that batch
                # (both collators only stack it when every sample in the
                # batch has it), not a crash, but in practice asan is ~10x
                # the other sources so this is a non-issue.
                load_hand_crops=self.use_pgf, hand_crop_root=self.hand_crop_root,
                # Prosody only loaded for the ablation's treatment arm;
                # baseline arm never touches it (identical data pipeline
                # otherwise, so the arms stay comparable).
                load_prosody=self.prosody_aux_weight > 0,
                prosody_root=self.prosody_root,
            )
            all_train.append(AsanDataset(split='train', **asan_common))
            all_val.append(AsanDataset(split='val', **asan_common))

        # Khabar KZ — signer-disjoint split
        if 'khabar_kz' in paths and os.path.exists(
                paths['khabar_kz'].get('manifest', '')):
            khabar_train = KhabarKzDataset(
                manifest_path=paths['khabar_kz']['manifest'],
                keypoints_root=paths['khabar_kz']['keypoints'],
                tokenizer=None, max_duration=60.0, min_duration=2.0,
                max_frames=self.train_cfg['max_seq_len'], downsample_every=1,
                name='khabar_kz', split='train', split_ratio=split_ratio,
                use_enriched=self.use_enriched,
            )
            khabar_val = KhabarKzDataset(
                manifest_path=paths['khabar_kz']['manifest'],
                keypoints_root=paths['khabar_kz']['keypoints'],
                tokenizer=None, max_duration=60.0, min_duration=2.0,
                max_frames=self.train_cfg['max_seq_len'], downsample_every=1,
                name='khabar_kz', split='val', split_ratio=split_ratio,
                use_enriched=self.use_enriched,
            )
            all_train.append(khabar_train)
            all_val.append(khabar_val)

        # Informburo KZ — signer-disjoint split
        if 'informburo' in paths:
            informburo_kps = paths['informburo'].get('keypoints', '')
            informburo_txt = paths['informburo'].get('transcripts', '')
            if informburo_kps and os.path.exists(informburo_kps):
                inf_train = InformburoDataset(
                    keypoints_root=informburo_kps,
                    transcripts_root=informburo_txt,
                    tokenizer=None, max_duration=60.0, min_duration=2.0,
                    max_frames=self.train_cfg['max_seq_len'], downsample_every=2,
                    name='informburo_kz', split='train', split_ratio=split_ratio,
                    use_enriched=self.use_enriched,
                )
                inf_val = InformburoDataset(
                    keypoints_root=informburo_kps,
                    transcripts_root=informburo_txt,
                    tokenizer=None, max_duration=60.0, min_duration=2.0,
                    max_frames=self.train_cfg['max_seq_len'], downsample_every=2,
                    name='informburo_kz', split='val', split_ratio=split_ratio,
                    use_enriched=self.use_enriched,
                )
                all_train.append(inf_train)
                all_val.append(inf_val)

        if not all_train:
            raise RuntimeError(
                "No datasets found — check that the paths in configs/config.yaml "
                "exist on this machine (asan.root, khabar_kz.manifest, ...)")

        train_dataset = ConcatDataset(all_train) if len(all_train) > 1 else all_train[0]
        val_dataset = ConcatDataset(all_val) if len(all_val) > 1 else all_val[0]

        # Sanity mode: memorize N clips (train == val). A healthy pipeline
        # drives CE near 0 and reproduces the references verbatim within
        # ~50-100 epochs on 100 clips; failure to do so means a structural
        # bug, and no amount of full-data training will help.
        if self.overfit_n > 0:
            from torch.utils.data import Subset
            n = min(self.overfit_n, len(train_dataset))
            idx = list(range(n))
            train_dataset = Subset(train_dataset, idx)
            val_dataset = Subset(train_dataset.dataset, idx) \
                if hasattr(train_dataset, 'dataset') else train_dataset
            log(f"[OVERFIT SANITY] train == val == first {n} clips")

        log(f"[Datasets] Signer-disjoint split: train={len(train_dataset)}, val={len(val_dataset)}")

        core = self.model.module if self.distributed else self.model
        collator = SimpleCollator(
            mt5_tokenizer=core.mt5_tokenizer,
            max_text_tokens=self.train_cfg.get('max_text_len', 128),
        )

        if self.distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            train_loader = DataLoader(
                train_dataset, batch_size=self.train_cfg['batch_size'],
                sampler=train_sampler, num_workers=2, collate_fn=collator,
                pin_memory=True, persistent_workers=True,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.train_cfg['batch_size'],
                shuffle=False, num_workers=2, collate_fn=collator, pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=self.train_cfg['batch_size'],
                shuffle=True, num_workers=4, collate_fn=collator,
                pin_memory=True, persistent_workers=True,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.train_cfg['batch_size'],
                shuffle=False, num_workers=2, collate_fn=collator, pin_memory=True,
            )

        # Frozen selection subset (E0). Without this, validate() generates on
        # the first 25 unshuffled batches, and because sources concatenate in
        # list order those 200 clips are 100% informburo -- 10.8% of dev, and
        # qazaqstantv (48.6%) never generated on at all. Checkpoints were
        # therefore selected on a sample that could not see most of the data.
        self.gen_loader = None
        if self.selection_manifest:
            with open(os.path.expanduser(self.selection_manifest)) as fh:
                wanted = {m['clip_id'] for m in json.load(fh)}
            subsets = all_val if isinstance(val_dataset, ConcatDataset) else [val_dataset]
            indices, offset = [], 0
            for ds in subsets:
                clips = getattr(ds, 'clips', [])
                for i, c in enumerate(clips):
                    if c.get('clip_id') in wanted:
                        indices.append(offset + i)
                offset += len(ds)
            missing = len(wanted) - len(indices)
            if missing:
                raise RuntimeError(
                    f"selection manifest lists {len(wanted)} clips but only "
                    f"{len(indices)} are present in this val set ({missing} "
                    f"missing). The manifest and ASAN_ROOT disagree -- refusing "
                    f"to silently select on a different subset than intended.")
            from torch.utils.data import Subset
            self.gen_loader = DataLoader(
                Subset(val_dataset, sorted(indices)),
                batch_size=self.train_cfg['batch_size'], shuffle=False,
                num_workers=2, collate_fn=collator, pin_memory=True)
            self._gen_target = len(indices)
            log(f"[Selection] frozen manifest: {len(indices)} clips "
                f"({self.selection_manifest})")
        else:
            self._gen_target = None
            log("[Selection] WARNING: no --selection-manifest; falling back to "
                "the first 25 val batches, which are NOT source-representative")

        return train_loader, val_loader

    def _align_batch(self, batch):
        """
        Per-batch tensors for the E5 alignment loss, or (None, None, None).

        Looks up each clip's frozen text-teacher vector by clip_id (never by
        batch position: SimpleCollator sorts by length and drops invalid
        samples). Then samples extra negatives from the cached table, masking
        any whose normalized text equals the positive's -- 2.6% of references
        are exact duplicates of another clip's, and those are not negatives.
        """
        if self._align_vecs is None or self.align_weight <= 0:
            return None, None, None
        ids = batch.get('clip_ids')
        if not ids:
            return None, None, None
        rows = [self._align_index.get(c) for c in ids]
        if any(r is None for r in rows):                 # clip absent from cache
            return None, None, None
        idx = torch.tensor(rows, dtype=torch.long)
        text = self._align_vecs[idx].to(self.device, non_blocking=True).float()

        negs = neg_mask = None
        if self.align_negatives > 0:
            k = min(self.align_negatives, self._align_vecs.shape[0])
            nidx = torch.randint(0, self._align_vecs.shape[0], (k,))
            negs = self._align_vecs[nidx].to(self.device, non_blocking=True).float()
            pos_h = self._align_hash[idx].to(self.device)      # (B,)
            neg_h = self._align_hash[nidx].to(self.device)     # (K,)
            neg_mask = pos_h[:, None] == neg_h[None, :]        # (B, K) true = same text
        return text, negs, neg_mask

    _align_epoch_mean = 0.0

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        total_mse = 0
        self._ctc_running = 0.0
        self._prosody_aux_running = 0.0
        self._align_running = 0.0
        num_batches = 0
        pending = 0  # batches accumulated since the last optimizer step
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            # Rank-consistent skip: if ANY rank got an empty batch, all ranks
            # skip this step. A one-sided `continue` desyncs DDP — the other
            # rank blocks in its gradient all-reduce until the NCCL watchdog
            # kills the job (SIGABRT).
            if self.distributed:
                ok = torch.tensor(
                    [0.0 if batch is None else 1.0], device=self.device)
                dist.all_reduce(ok, dist.ReduceOp.MIN)
                if ok.item() < 1:
                    continue
            elif batch is None:
                continue

            kps = batch['keypoints'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            label_attn = batch['label_attn_mask'].to(self.device)
            input_lengths = batch['input_lengths'].to(self.device)
            hand_crops = batch['hand_crops'].to(self.device) if batch.get('hand_crops') is not None else None
            hand_ref = batch['hand_ref'].to(self.device) if batch.get('hand_ref') is not None else None
            hand_valid = batch['hand_valid'].to(self.device) if batch.get('hand_valid') is not None else None
            hand_score = batch['hand_score'].to(self.device) if batch.get('hand_score') is not None else None
            prosody_target = (batch['prosody'].to(self.device)
                             if batch.get('prosody') is not None else None)

            # --- Masked-pose reconstruction (multi-granularity:
            #     joint / frame / span, SignBERT+-style) ---
            kps_train = kps
            mask = None
            if self.masked_pose_ratio > 0:
                mask = build_pose_mask(kps, input_lengths, self.masked_pose_ratio)
                kps_train = torch.where(mask, torch.zeros_like(kps), kps)

            # Forward pass (CE + optional aux losses, single encoder pass)
            align_text, align_negs, align_neg_mask = self._align_batch(batch)
            loss, mse_loss, ctc_log_probs, prosody_aux_loss, align_loss = self.model(
                kps_train, label_ids, label_attn,
                input_lengths=input_lengths,
                kps_target=kps if mask is not None else None,
                frame_mask=mask,
                hand_crops=hand_crops, hand_ref=hand_ref,
                hand_valid=hand_valid, hand_score=hand_score,
                prosody_target=prosody_target,
                align_text=align_text, align_negatives=align_negs,
                align_neg_mask=align_neg_mask,
            )
            if align_loss is None:
                align_loss = torch.tensor(0.0, device=self.device)
            elif self.align_weight > 0:
                loss = loss + self.align_weight * align_loss
            self._align_running += float(align_loss.detach())
            if mse_loss is None:
                mse_loss = torch.tensor(0.0, device=self.device)
            else:
                loss = loss + 0.1 * mse_loss

            # Prosody-as-supervision aux loss (ablation treatment arm).
            if prosody_aux_loss is None:
                prosody_aux_loss = torch.tensor(0.0, device=self.device)
            elif self.prosody_aux_weight > 0:
                loss = loss + self.prosody_aux_weight * prosody_aux_loss
            self._prosody_aux_running += float(prosody_aux_loss.detach())

            # CTC auxiliary loss: align encoder frames to transcript chars
            if ctc_log_probs is not None and self.ctc_weight > 0:
                targets, tgt_lens = [], []
                for text in batch['texts']:
                    # +1 shift: SP ids start at 0, CTC blank occupies 0
                    ids = [i + 1 for i in
                           self.ctc_tokenizer.encode(text.lower())]
                    targets.extend(ids)
                    tgt_lens.append(len(ids))
                targets = torch.tensor(targets, dtype=torch.long,
                                       device=self.device)
                tgt_lens = torch.tensor(tgt_lens, dtype=torch.long,
                                        device=self.device)
                in_lens = input_lengths.clamp(max=ctc_log_probs.size(0))
                ctc_loss = F.ctc_loss(
                    ctc_log_probs, targets, in_lens, tgt_lens,
                    blank=0, zero_infinity=True,  # inf when T < target len
                )
                loss = loss + self.ctc_weight * ctc_loss
                self._ctc_running += ctc_loss.item()

            # Always run backward — skipping it on one rank (the old
            # `continue` on non-finite loss) deadlocks the other rank's
            # all-reduce. After DDP averaging, gradients are identical on all
            # ranks, so the finiteness of the clipped grad norm is a
            # rank-consistent signal for whether to step.
            # Gradient accumulation: backward every batch (scaled), optimizer
            # step every grad_accum batches.
            (loss / self.grad_accum).backward()
            pending += 1

            if torch.isfinite(loss):
                total_loss += loss.item()
                if mse_loss.item() > 0:
                    total_mse += mse_loss.item()
                num_batches += 1

            if pending >= self.grad_accum:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                if torch.isfinite(total_norm):
                    self.optimizer.step()
                self.optimizer.zero_grad()  # also drops non-finite grads
                pending = 0
                self.global_step += 1
                self.scheduler.step()

            if is_main() and batch_idx % 500 == 0 and batch_idx > 0:
                log(f"  Batch {batch_idx}/{len(train_loader)} | "
                     f"Loss: {total_loss / num_batches:.4f}"
                     + (f" | MSE: {total_mse / num_batches:.4f}" if total_mse > 0 else "")
                     + (f" | CTC: {self._ctc_running / num_batches:.4f}"
                        if self._ctc_running > 0 else "")
                     + (f" | ProsAux: {self._prosody_aux_running / num_batches:.4f}"
                        if self._prosody_aux_running > 0 else "")
                     + (f" | Align: {self._align_running / num_batches:.4f}"
                        if self._align_running > 0 else ""))

        # Flush a leftover partial accumulation window at epoch end
        if pending > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            if torch.isfinite(total_norm):
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1
            self.scheduler.step()

        if self.distributed:
            loss_tensor = torch.tensor([total_loss], device=self.device)
            count_tensor = torch.tensor([num_batches], device=self.device)
            dist.all_reduce(loss_tensor, dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, dist.ReduceOp.SUM)
            total_loss = loss_tensor.item()
            num_batches = count_tensor.item()

        self._align_epoch_mean = self._align_running / max(num_batches, 1)

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader, max_gen_batches=25):
        """
        CE loss over the full val set; generation metrics over the frozen
        selection subset (self.gen_loader) when one is configured.

        The legacy path -- first `max_gen_batches` batches of an unshuffled
        val_loader -- is kept only for runs without a manifest, and is NOT
        source-representative: see create_datasets for why.
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_refs = []
        all_hyps = []

        for batch in val_loader:
            if batch is None:
                continue

            kps = batch['keypoints'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            label_attn = batch['label_attn_mask'].to(self.device)
            input_lengths = batch['input_lengths'].to(self.device)
            hand_crops = batch['hand_crops'].to(self.device) if batch.get('hand_crops') is not None else None
            hand_ref = batch['hand_ref'].to(self.device) if batch.get('hand_ref') is not None else None
            hand_valid = batch['hand_valid'].to(self.device) if batch.get('hand_valid') is not None else None
            hand_score = batch['hand_score'].to(self.device) if batch.get('hand_score') is not None else None
            texts = batch['texts']

            loss, _, _, _, _ = self.model(kps, label_ids, label_attn,
                                    input_lengths=input_lengths,
                                    hand_crops=hand_crops, hand_ref=hand_ref,
                                    hand_valid=hand_valid, hand_score=hand_score)
            if not torch.isfinite(loss):
                continue

            total_loss += loss.item()
            num_batches += 1

            if (is_main() and self.gen_loader is None
                    and num_batches <= max_gen_batches):
                try:
                    core = self.model.module if self.distributed else self.model
                    hyps = core.generate(kps, input_lengths=input_lengths,
                                         hand_crops=hand_crops, hand_ref=hand_ref,
                                         hand_valid=hand_valid, hand_score=hand_score)
                    all_hyps.extend(hyps)
                    all_refs.extend(texts)
                except Exception as e:
                    log(f"  [WARN] Generation failed: {e}")

        # Frozen selection subset: generate over ALL of it, and report actual
        # coverage so a partial run can never masquerade as a full one.
        if is_main() and self.gen_loader is not None:
            n_failed = 0
            core = self.model.module if self.distributed else self.model
            for batch in self.gen_loader:
                if batch is None:
                    continue
                try:
                    hyps = core.generate(
                        batch['keypoints'].to(self.device),
                        input_lengths=batch['input_lengths'].to(self.device))
                    all_hyps.extend(hyps)
                    all_refs.extend(batch['texts'])
                except Exception as e:
                    n_failed += len(batch['texts'])
                    log(f"  [WARN] Generation failed on a selection batch: {e}")
            covered = len(all_hyps)
            log(f"  [Selection] generated {covered}/{self._gen_target} clips"
                + (f" ({n_failed} failed)" if n_failed else ""))
            if covered < self._gen_target:
                log(f"  [WARN] selection coverage incomplete -- metrics this "
                    f"epoch are NOT comparable to a full-coverage epoch")

        avg_loss = total_loss / max(num_batches, 1)

        # Translation-quality metrics over the same generated subset used
        # for WER (utils/metrics.py) -- WER catches exact-transcription
        # accuracy (what's been tracked all along); BLEU/ROUGE catch n-gram
        # overlap (standard MT-quality metrics); BERTScore catches semantic
        # similarity via embeddings, robust to paraphrasing/morphological
        # variation that WER/BLEU/ROUGE penalize harshly for an agglutinative
        # language like Kazakh. Each degrades independently (missing package
        # -> that one metric stays 0.0 with a one-time warning) so a single
        # missing pip install doesn't block the others or crash validation.
        # n_gen: how many hypotheses were actually generated. Critical for
        # checkpoint selection -- if generation fails, every metric stays at
        # its 0.0 default, and a WER of 0.0 would otherwise look PERFECT and
        # be saved as the best model.
        metrics = {'wer': 0.0, 'bleu': 0.0, 'rouge1': 0.0, 'rouge2': 0.0,
                  'rougeL': 0.0, 'bertscore_f1': 0.0, 'n_gen': 0}
        if is_main() and all_refs and all_hyps:
            metrics['n_gen'] = len(all_hyps)
            try:
                # Shared implementation (E0) -- identical to the one
                # scripts/evaluate_phase1.py calls, and normalized the same
                # way as BLEU/ROUGE below. Previously this was raw
                # whitespace-token WER alongside a normalized BLEU.
                metrics['wer'] = compute_corpus_wer(all_refs, all_hyps)
                metrics['wer_raw'] = compute_corpus_wer(
                    all_refs, all_hyps, normalize=False)
            except ImportError:
                log("[WARN] pip install editdistance for WER")

            try:
                metrics['bleu'] = compute_bleu(all_refs, all_hyps)
            except ImportError:
                log("[WARN] pip install sacrebleu for BLEU")

            try:
                r1, r2, rl = compute_rouge(all_refs, all_hyps)
                metrics['rouge1'], metrics['rouge2'], metrics['rougeL'] = r1, r2, rl
            except ImportError:
                log("[WARN] pip install rouge_score for ROUGE")

            try:
                metrics['bertscore_f1'] = compute_bertscore(
                    all_refs, all_hyps, device=str(self.device))
            except ImportError:
                log("[WARN] pip install bert_score for BERTScore")

            # Print a few examples for debugging
            for i in range(min(3, len(all_refs))):
                log(f"  REF: {all_refs[i][:80]}")
                log(f"  HYP: {all_hyps[i][:80]}")

        return avg_loss, metrics

    def _build_checkpoint(self, epoch, val_loss, metrics):
        """
        Checkpoint payload. Saves the MT5 decoder as well as the encoder —
        the encoder alone is NOT enough to reproduce the model at inference
        time (the decoder was fine-tuned / adapted along with it).
        """
        core = self.model.module if self.distributed else self.model
        ckpt = {
            'encoder': core.encoder.state_dict(),
            'pose_norm': core.pose_norm.state_dict(),
            'epoch': epoch, 'val_loss': val_loss,
            'wer': metrics['wer'],  # top-level shortcut, kept for convenience
            'metrics': metrics,     # full WER/BLEU/ROUGE/BERTScore dict
            'use_lora': self.use_lora,
            # Full trainer state for --resume: optimizer state (Adam
            # momentum/variance) and the optimizer-step counter the LR
            # scheduler needs to continue its cosine curve mid-decay
            # instead of restarting from warmup.
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'run_args': self._run_args,
            'select_metric': self.select_metric,
            'best_score': self.best_score,
        }
        if core.ctc_head is not None:
            ckpt['ctc_head'] = core.ctc_head.state_dict()
            ckpt['ctc_bpe_vocab_size'] = self.ctc_vocab_size
        if core.prosody_aux_head is not None:
            ckpt['prosody_aux_head'] = core.prosody_aux_head.state_dict()
        if self.use_lora:
            # Adapters only (small); base MT5 is reproducible from the hub.
            ckpt['mt5_lora'] = {
                k: v.detach().cpu()
                for k, v in core.mt5.named_parameters() if v.requires_grad
            }
        else:
            ckpt['mt5'] = core.mt5.state_dict()
        if core.masked_pose_decoder is not None:
            ckpt['masked_pose_decoder'] = core.masked_pose_decoder.state_dict()
        if core.hand_backbone is not None:
            ckpt['hand_backbone'] = core.hand_backbone.state_dict()
            ckpt['pgf_hand_fusion'] = core.pgf_hand_fusion.state_dict()
            ckpt['pgf_gate'] = core.pgf_gate.state_dict()
            ckpt['pgf_keypoint_adapter'] = core.pgf_keypoint_adapter.state_dict()
        return ckpt

    def train(self, num_epochs=None, save_dir=None):
        if num_epochs is None:
            num_epochs = self.max_epochs
        if save_dir is None:
            save_dir = self.config['paths']['output']
        if is_main():
            os.makedirs(save_dir, exist_ok=True)

        train_loader, val_loader = self.create_datasets()

        # Set up scheduler (one step per OPTIMIZER step, not per batch).
        # Cap warmup at 10% of the run — otherwise short runs (e.g.
        # --overfit-n sanity checks) spend their entire budget inside
        # warmup and train at a tiny LR, which looks exactly like a
        # broken pipeline.
        total_steps = math.ceil(len(train_loader) / self.grad_accum) * num_epochs
        warmup_eff = min(self.warmup_steps, max(total_steps // 10, 1))
        if warmup_eff < self.warmup_steps:
            log(f"[Scheduler] warmup capped: {self.warmup_steps} → {warmup_eff} "
                f"(10% of {total_steps} total steps)")
        # When resuming, num_epochs is the NEW total (e.g. 30 after an
        # original 25-epoch run) — the cosine curve is recomputed to span
        # that full total, then last_epoch fast-forwards it to global_step
        # so the LR continues mid-decay instead of restarting from warmup.
        # (This does mean extending the total changes the shape of the
        # decay — expected when you decide to train longer than planned.)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_eff,
            num_training_steps=total_steps,
            last_epoch=self.global_step - 1 if self.global_step > 0 else -1,
        )

        log(f"\n{'='*60}")
        log(f"Phase 1: Uni-Sign Encoder + MT5")
        if self.start_epoch > 0:
            log(f"Resuming at epoch {self.start_epoch + 1}, "
                f"global_step={self.global_step}")
        log(f"Epochs: {num_epochs} (total, including any already completed)")
        log(f"{'='*60}\n")

        for epoch in range(self.start_epoch, num_epochs):
            epoch_start = time.time()

            if self.distributed and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, metrics = self.validate(val_loader)
            # Rank 0 runs beam-search WER during validate; other ranks wait
            # here instead of timing out inside next epoch's first all-reduce.
            if self.distributed:
                dist.barrier()
            epoch_time = time.time() - epoch_start

            lr_mt5 = self.optimizer.param_groups[-1]['lr']
            lr_enc = self.optimizer.param_groups[0]['lr']
            log(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"WER: {metrics['wer']:.4f} | BLEU: {metrics['bleu']:.2f} | "
                  f"ROUGE-1/2/L: {metrics['rouge1']:.3f}/{metrics['rouge2']:.3f}/"
                  f"{metrics['rougeL']:.3f} | BERTScore: {metrics['bertscore_f1']:.4f} | "
                  f"LR_enc: {lr_enc:.6f} | LR_mt5: {lr_mt5:.6f} | "
                + (f"Align: {self._align_epoch_mean:.4f} | "
                   if self._align_epoch_mean else "")
                + f"Time: {epoch_time:.1f}s")

            # ---- Best-checkpoint selection ----
            # Selects on --select-metric (default WER), NOT val CE. On this
            # task val CE rises while WER/BLEU keep improving, so val-CE
            # selection reliably saves a worse model.
            if is_main():
                score = (val_loss if self.select_metric == 'val_loss'
                         else metrics.get(self.select_metric))
                # Guard: if generation produced nothing, every metric is at its
                # 0.0 default and a WER of 0.0 would look perfect. Never select
                # on metrics that no generation backed (val_loss is exempt --
                # it doesn't depend on generation).
                gen_backed = (self.select_metric == 'val_loss'
                              or metrics.get('n_gen', 0) > 0)
                if score is None:
                    log(f"  [WARN] select-metric '{self.select_metric}' missing "
                        f"from metrics; skipping best-checkpoint update.")
                elif not gen_backed:
                    log(f"  [WARN] no hypotheses generated this epoch -- metrics "
                        f"are placeholders; skipping best-checkpoint update.")
                else:
                    higher_better = SELECT_METRICS[self.select_metric] == 'higher'
                    improved = (score > self.best_score if higher_better
                                else score < self.best_score)
                    if improved:
                        self.best_score = score
                        ckpt = self._build_checkpoint(epoch, val_loss, metrics)
                        torch.save(ckpt,
                                   os.path.join(save_dir, 'phase1_mt5_best.pth'))
                        log(f"  Saved best by {self.select_metric}={score:.4f} "
                            f"(CE: {val_loss:.4f}, WER: {metrics['wer']:.4f}, "
                            f"BLEU: {metrics['bleu']:.2f}, "
                            f"BERTScore: {metrics['bertscore_f1']:.4f})")

            if is_main() and (epoch + 1) % 5 == 0:
                ckpt = self._build_checkpoint(epoch, val_loss, metrics)
                torch.save(ckpt, os.path.join(save_dir, f'phase1_mt5_epoch{epoch+1}.pth'))

        if self.distributed:
            dist.barrier()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--pretrained-encoder', default=None,
                        help='Path to our encoder checkpoint')
    parser.add_argument('--pretrained-unisign', default=None,
                        help='Path to raw Uni-Sign weights (csl_stage1_weight.pth)')
    parser.add_argument('--freeze-spatial', action='store_true',
                        help='Freeze spatial STGCN, only train temporal + MT5')
    parser.add_argument('--use-lora', action='store_true',
                        help='Use LoRA for MT5 fine-tuning (requires: pip install peft)')
    parser.add_argument('--lora-r', type=int, default=16,
                        help='LoRA rank (default: 16)')
    parser.add_argument('--lora-alpha', type=int, default=32,
                        help='LoRA alpha scaling (default: 32)')
    parser.add_argument('--selection-manifest', default=None,
                        help='JSON list of {clip_id,...} defining the FROZEN '
                             'checkpoint-selection subset (see '
                             'scripts/build_canonical_splits.py). Without it '
                             'the trainer falls back to the first 25 val '
                             'batches, which are 100%% informburo and cannot '
                             'see most of the data.')
    parser.add_argument('--block-padding-mask', action='store_true',
                        help='Re-zero padded frames inside EVERY temporal '
                             'ST-GCN block (E2). Without it, conv bias and '
                             'BatchNorm shift make padding nonzero and the '
                             'kernel-5 temporal convs mix it into real frames, '
                             'so a clip embedding depends on batch padding. '
                             'Makes eval exactly padding-invariant; training '
                             'BatchNorm statistics still count padded frames.')
    parser.add_argument('--unisign-preprocess', action='store_true',
                        help='E3 arm B: exact port of upstream Uni-Sign pose '
                             'preprocessing (9-joint body with real wrists, '
                             'clip-level body box, wrist-relative hands, '
                             'nose-relative face, (x,y,score) with score<=0.3 '
                             'masked). Input dim 207. Exclusive with '
                             '--use-enriched/--signspace/--real-wrists/--score-quantiles.')
    parser.add_argument('--align-cache', default=None,
                        help='E5: frozen text-teacher vectors from '
                             'scripts/cache_text_embeddings.py.')
    parser.add_argument('--align-weight', type=float, default=0.0,
                        help='E5: weight of the pose-text InfoNCE added to CE. '
                             '0 disables (default). Targets the bimodal failure '
                             'found in E3, where 7 of 8 runs never used the pose '
                             'input and collapsed to the corpus prior.')
    parser.add_argument('--align-negatives', type=int, default=256,
                        help='E5: negatives sampled from the cached table per '
                             'step, on top of the 7 in-batch ones.')
    parser.add_argument('--align-dim', type=int, default=256,
                        help='E5: projection dim for the shared pose/text space.')
    parser.add_argument('--unisign-hand-scale', default=None,
                        help='E3 arm D: hand_ratio.json from scripts/fit_hand_ratio.py. '
                             'Rescales each hand per frame to the train-median '
                             'hand size in upstream body-box units. Requires '
                             '--unisign-preprocess.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seeds python, numpy and torch (CPU+CUDA). cuDNN '
                             'kernels remain nondeterministic.')
    parser.add_argument('--score-quantiles', default=None,
                        help='Train-fitted score table (scripts/fit_score_quantiles.py). '
                             'Replaces clip(score,0,1), which pins ~100%% of '
                             'joints to 1.0 and makes the confidence channel '
                             'constant. Changes encoder input; do not --resume '
                             'a checkpoint trained without it.')
    parser.add_argument('--real-wrists', action='store_true',
                        help='Feed the body graph real wrist joints (E2). The '
                             'legacy mapping repeats ELBOW values into the '
                             'wrist nodes, so hand-anchor fusion reads a '
                             'pseudo-wrist. Sets dataset and encoder together; '
                             'changes encoder input, so do not --resume a '
                             'checkpoint trained without it.')
    parser.add_argument('--signspace', action='store_true',
                        help='SignSpace pose normalization (arXiv:2507.01532): '
                             'body scaled globally into a 3x-shoulder box, '
                             'hands and face normalized locally and '
                             'independently. Removes signer hand-size / '
                             'camera-distance nuisance variation that the '
                             'default single global shoulder-width divisor '
                             'leaves in. Changes the input distribution, so '
                             'do NOT --resume a checkpoint trained without it.')
    parser.add_argument('--use-enriched', action='store_true',
                        help='Use enriched pose features (offset+vel+acc+valid, 1128 dims)')
    parser.add_argument('--masked-pose-ratio', type=float, default=0.0,
                        help='Fraction of frames to mask for aux reconstruction loss')
    parser.add_argument('--overfit-n', type=int, default=0,
                        help='Sanity mode: train and validate on the first N '
                             'clips (expect near-0 CE if the pipeline is sound)')
    parser.add_argument('--grad-accum', type=int, default=None,
                        help="Override config.yaml's training.phase1.grad_accum. "
                             "Needed for --overfit-n: with few clips, "
                             "len(train_loader) is tiny, so the config's "
                             "grad_accum=4 can collapse to ~1 optimizer step "
                             "per epoch. Pass --grad-accum 1 for sanity checks.")
    parser.add_argument('--encoder-lr', type=float, default=None,
                        help='Override the encoder LR (default: base_lr/10, '
                             "a 'gentle adaptation' choice for the "
                             'CSL-pretrained encoder). diagnose_phase1.py '
                             'found the encoder still produces near-'
                             'collapsed embeddings across genuinely '
                             'different clips even after real training at '
                             'that rate -- try something close to or equal '
                             'to the base LR to test whether base_lr/10 is '
                             'too conservative for what the encoder needs '
                             'to learn.')
    parser.add_argument('--ctc-weight', type=float, default=0.0,
                        help='Weight of the subword-CTC auxiliary loss on '
                             'encoder frames (0.3 is a good start); forces '
                             'visual grounding when the decoder drifts into '
                             'pure language modeling')
    parser.add_argument('--ctc-vocab-size', type=int, default=2000,
                        help='SentencePiece BPE vocab size for CTC targets. '
                             'Larger = longer pieces = shorter targets (helps '
                             'the frames>=targets constraint), but a bigger '
                             'CTC head. 2000 gives ~2.5 chars/piece.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Total epoch count for this run. With --resume, '
                             'this is the NEW total (e.g. 30 to add 5 epochs '
                             'to a completed 25-epoch run), not an increment.')
    parser.add_argument('--save-dir', default=None)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--resume', default=None,
                        help='Path to a phase1_mt5_*.pth checkpoint to fully '
                             'resume from (model + optimizer + step count), '
                             'e.g. to train more epochs than originally '
                             'planned. Must be run with the SAME '
                             '--use-enriched/--masked-pose-ratio/--ctc-weight/'
                             '--ctc-vocab-size/--freeze-spatial/--use-lora '
                             'flags the checkpoint was trained with. Overrides '
                             '--pretrained-encoder/--pretrained-unisign.')
    parser.add_argument('--use-pgf', action='store_true',
                        help='Enable Uni-Sign\'s real Prior-Guided Fusion '
                             '(arXiv:2501.15187 -- hand-crop RGB via a '
                             'trainable EfficientNet-B0 + deformable '
                             'attention, see models/pgf_fusion.py). Requires '
                             '--hand-crop-root pointing at '
                             'scripts/extract_asan_hand_crops.py\'s output.')
    parser.add_argument('--hand-crop-root', default=None,
                        help='Output root of scripts/extract_asan_hand_crops.py '
                             '(the --out passed to that script).')
    parser.add_argument('--pgf-p-samp', type=float, default=0.5,
                        help='Fraction of frames per clip that get RGB '
                             'fusion each step (paper Appendix A.3 '
                             'score-aware sampling).')
    parser.add_argument('--select-metric', default='wer',
                        choices=list(SELECT_METRICS.keys()),
                        help="Metric for best-checkpoint selection "
                             "(default: wer). NOT val_loss -- on this "
                             "task val CE rises while WER/BLEU improve, "
                             "so val_loss selection saves a worse model.")
    parser.add_argument('--prosody-aux-weight', type=float, default=0.0,
                        help='Weight for the prosody-as-supervision auxiliary '
                             'loss (ablation). 0 = baseline arm (head not built, '
                             'prosody not even loaded); >0 = treatment arm. '
                             'Requires --prosody-root.')
    parser.add_argument('--prosody-root', default=None,
                        help='Output root of scripts/extract_asan_prosody_v3.py '
                             '(needs prosody_stats.json for normalization).')
    parser.add_argument('--pretrained-pgf', default=None,
                        help='Seed PGF submodules (hand_backbone, '
                             'pgf_hand_fusion, pgf_gate) from a checkpoint '
                             'produced by scripts/convert_friend_checkpoint.py '
                             '--convert-pgf, while still using our own '
                             '--pretrained-encoder for the pose encoder. '
                             'Requires --use-pgf. Each submodule loads '
                             'independently -- pgf_keypoint_adapter has no '
                             'equivalent in a converted colleague checkpoint '
                             'and always stays at its fresh init.')
    args = parser.parse_args()

    import random as _random
    _random.seed(args.seed)
    import numpy as _np
    _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.unisign_hand_scale and not args.unisign_preprocess:
        parser.error('--unisign-hand-scale requires --unisign-preprocess')
    if args.unisign_preprocess and (args.use_enriched or args.signspace
                                    or args.real_wrists or args.score_quantiles):
        parser.error('--unisign-preprocess is exclusive with --use-enriched, '
                     '--signspace, --real-wrists and --score-quantiles')

    if args.prosody_aux_weight > 0 and not args.prosody_root:
        parser.error('--prosody-aux-weight > 0 requires --prosody-root')

    if args.pretrained_pgf and not args.use_pgf:
        parser.error("--pretrained-pgf requires --use-pgf")

    if args.use_pgf and not args.hand_crop_root:
        parser.error("--use-pgf requires --hand-crop-root")

    if args.resume and (args.pretrained_encoder or args.pretrained_unisign):
        print("[WARN] --resume overrides --pretrained-encoder/--pretrained-unisign "
              "(their weights would be loaded then immediately replaced).")

    # Generous NCCL timeout: rank 0 does beam-search WER at validation while
    # other ranks wait; the 10-minute default watchdog is too tight.
    from datetime import timedelta
    nccl_timeout = timedelta(hours=2)

    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', timeout=nccl_timeout)
    elif args.local_rank >= 0:
        local_rank = args.local_rank
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', timeout=nccl_timeout)
    else:
        local_rank = 0

    trainer = MT5Trainer(
        config_path=args.config, local_rank=local_rank,
        pretrained_encoder=args.pretrained_encoder,
        pretrained_unisign=args.pretrained_unisign,
        freeze_spatial=args.freeze_spatial,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_enriched=args.use_enriched,
        signspace=args.signspace,
        selection_manifest=args.selection_manifest,
        block_padding_mask=args.block_padding_mask,
        real_wrists=args.real_wrists,
        score_quantiles=args.score_quantiles,
        unisign_preprocess=args.unisign_preprocess,
        unisign_hand_scale=args.unisign_hand_scale,
        align_dim=args.align_dim,
        align_negatives=args.align_negatives,
        align_weight=args.align_weight,
        align_cache=args.align_cache,
        masked_pose_ratio=args.masked_pose_ratio,
        overfit_n=args.overfit_n,
        ctc_weight=args.ctc_weight,
        ctc_vocab_size=args.ctc_vocab_size,
        resume=args.resume,
        grad_accum=args.grad_accum,
        encoder_lr=args.encoder_lr,
        use_pgf=args.use_pgf,
        hand_crop_root=args.hand_crop_root,
        pgf_p_samp=args.pgf_p_samp,
        pretrained_pgf=args.pretrained_pgf,
        prosody_aux_weight=args.prosody_aux_weight,
        prosody_root=args.prosody_root,
        select_metric=args.select_metric,
    )

    trainer.train(num_epochs=args.epochs, save_dir=args.save_dir)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
