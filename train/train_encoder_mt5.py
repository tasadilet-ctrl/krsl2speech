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
from models.unisign_encoder import KeypointEncoder, load_unisign_weights


# ============================================================
# MT5 Path (from Uni-Sign config.py)
# ============================================================
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
            'texts': texts,                     # raw strings for generation eval
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

    def __init__(self, encoder, mt5_path=MT5_PATH, lang="Kazakh",
                 masked_pose_dim=None, ctc_vocab_size=None):
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

        self.masked_pose_decoder = None
        if masked_pose_dim is not None:
            d_model = encoder.hidden_dim
            self.masked_pose_decoder = nn.Sequential(
                nn.Linear(d_model, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, masked_pose_dim),
            )

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

    def forward(self, kps, label_ids, label_attn_mask, input_lengths=None,
                kps_target=None, frame_mask=None):
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

        Returns:
            loss: scalar CE loss
            mse_loss: masked-pose reconstruction loss (or None)
            ctc_log_probs: (T, B, V+1) log-probs for CTC (or None)
        """
        B = kps.size(0)

        # Pose encoder — runs on the (possibly masked) input. Reconstructing
        # the masked frames from THESE embeddings gives the encoder a
        # gradient signal to encode temporal context (the previous version
        # reconstructed from clean, no-grad embeddings, so the aux loss only
        # ever trained the small MLP head).
        # input_lengths re-zeroes pad frames before the encoder's temporal
        # conv so batch padding can't bleed into real boundary frames.
        pose_emb = self.pose_norm(self.encoder(kps, input_lengths=input_lengths))  # (B, T, 768)

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

        return out.loss, mse_loss, ctc_log_probs

    def generate(self, kps, input_lengths=None, max_new_tokens=128, num_beams=4):
        """
        Inference: generate text from keypoints.

        Args:
            kps: (B, T, D)
            input_lengths: (B,) — true frame counts (optional)
            max_new_tokens: max output tokens
            num_beams: beam width

        Returns:
            list of decoded strings
        """
        B = kps.size(0)

        pose_emb = self.pose_norm(self.encoder(kps, input_lengths=input_lengths))  # (B, T, 768)

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
            # insertions far past 1.0.
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
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
                 use_enriched=False, masked_pose_ratio=0.0, overfit_n=0,
                 ctc_weight=0.0, ctc_vocab_size=2000, resume=None,
                 grad_accum=None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        from utils.paths import apply_env_overrides
        self.config = apply_env_overrides(self.config)

        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{local_rank}')
        self.cfg = self.config['model']
        self.train_cfg = self.config['training']['phase1']
        self.use_enriched = use_enriched
        self.masked_pose_ratio = masked_pose_ratio
        self.overfit_n = overfit_n
        self.ctc_weight = ctc_weight
        self.freeze_spatial = freeze_spatial
        # Flags that determine model/optimizer structure — saved into the
        # checkpoint so --resume can verify the resuming run uses the same
        # architecture (a mismatch here breaks state_dict loads far less
        # clearly than this explicit check does).
        self._run_args = dict(
            use_enriched=use_enriched, masked_pose_ratio=masked_pose_ratio,
            ctc_weight=ctc_weight, ctc_vocab_size=ctc_vocab_size,
            freeze_spatial=freeze_spatial, use_lora=use_lora,
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
        input_dim = ENRICHED_DIM() if use_enriched else KEYPOINT_DIM

        # Build encoder
        self.encoder = KeypointEncoder(
            hidden_dim=self.cfg['d_model'],
            input_dim=input_dim,
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
            encoder=self.encoder, lang="Kazakh",
            masked_pose_dim=input_dim if masked_pose_ratio > 0 else None,
            ctc_vocab_size=self.ctc_vocab_size if self.ctc_tokenizer else None,
        )
        if masked_pose_ratio > 0:
            log(f"[Masked Pose] Reconstruction decoder: {self.cfg['d_model']} → {input_dim}")
            log(f"[Masked Pose] Mask ratio: {masked_pose_ratio}")

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
        encoder_lr = base_lr / 10  # 5e-5 — gentle adaptation

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
        core.pose_norm.load_state_dict(ckpt['pose_norm'])

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
        self.best_loss = ckpt.get('val_loss', float('inf'))
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
                skip_low_quality=asan_cfg.get('skip_low_quality', True),
                min_hand_cov=asan_cfg.get('min_hand_cov', 0.0),
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

        return train_loader, val_loader

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        total_mse = 0
        self._ctc_running = 0.0
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

            # --- Masked-pose reconstruction (multi-granularity:
            #     joint / frame / span, SignBERT+-style) ---
            kps_train = kps
            mask = None
            if self.masked_pose_ratio > 0:
                mask = build_pose_mask(kps, input_lengths, self.masked_pose_ratio)
                kps_train = torch.where(mask, torch.zeros_like(kps), kps)

            # Forward pass (CE + optional aux losses, single encoder pass)
            loss, mse_loss, ctc_log_probs = self.model(
                kps_train, label_ids, label_attn,
                input_lengths=input_lengths,
                kps_target=kps if mask is not None else None,
                frame_mask=mask,
            )
            if mse_loss is None:
                mse_loss = torch.tensor(0.0, device=self.device)
            else:
                loss = loss + 0.1 * mse_loss

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
                        if self._ctc_running > 0 else ""))

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

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader, max_gen_batches=25):
        """
        Validation: CE loss over the full val set; WER over the first
        `max_gen_batches` batches (beam search over the whole set every
        epoch dominates epoch time; val_loader is not shuffled, so this is
        a fixed, comparable subset across epochs).
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
            texts = batch['texts']

            loss, _, _ = self.model(kps, label_ids, label_attn,
                                    input_lengths=input_lengths)
            if not torch.isfinite(loss):
                continue

            total_loss += loss.item()
            num_batches += 1

            if is_main() and num_batches <= max_gen_batches:
                try:
                    core = self.model.module if self.distributed else self.model
                    hyps = core.generate(kps, input_lengths=input_lengths)
                    all_hyps.extend(hyps)
                    all_refs.extend(texts)
                except Exception as e:
                    log(f"  [WARN] Generation failed: {e}")

        avg_loss = total_loss / max(num_batches, 1)

        # Compute WER
        wer = 0.0
        if is_main() and all_refs and all_hyps:
            try:
                import editdistance
                total_dist, total_words = 0, 0
                for r, h in zip(all_refs, all_hyps):
                    rd = r.strip().split()
                    hd = h.strip().split()
                    total_dist += editdistance.eval(rd, hd)
                    total_words += len(rd)
                wer = total_dist / max(total_words, 1)
                # Print a few examples for debugging
                for i in range(min(3, len(all_refs))):
                    log(f"  REF: {all_refs[i][:80]}")
                    log(f"  HYP: {all_hyps[i][:80]}")
            except ImportError:
                log("[WARN] pip install editdistance for WER")

        return avg_loss, wer

    def _build_checkpoint(self, epoch, val_loss, wer):
        """
        Checkpoint payload. Saves the MT5 decoder as well as the encoder —
        the encoder alone is NOT enough to reproduce the model at inference
        time (the decoder was fine-tuned / adapted along with it).
        """
        core = self.model.module if self.distributed else self.model
        ckpt = {
            'encoder': core.encoder.state_dict(),
            'pose_norm': core.pose_norm.state_dict(),
            'epoch': epoch, 'val_loss': val_loss, 'wer': wer,
            'use_lora': self.use_lora,
            # Full trainer state for --resume: optimizer state (Adam
            # momentum/variance) and the optimizer-step counter the LR
            # scheduler needs to continue its cosine curve mid-decay
            # instead of restarting from warmup.
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'run_args': self._run_args,
        }
        if core.ctc_head is not None:
            ckpt['ctc_head'] = core.ctc_head.state_dict()
            ckpt['ctc_bpe_vocab_size'] = self.ctc_vocab_size
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
            val_loss, wer = self.validate(val_loader)
            # Rank 0 runs beam-search WER during validate; other ranks wait
            # here instead of timing out inside next epoch's first all-reduce.
            if self.distributed:
                dist.barrier()
            epoch_time = time.time() - epoch_start

            lr_mt5 = self.optimizer.param_groups[-1]['lr']
            lr_enc = self.optimizer.param_groups[0]['lr']
            log(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"WER: {wer:.4f} | LR_enc: {lr_enc:.6f} | LR_mt5: {lr_mt5:.6f} | "
                  f"Time: {epoch_time:.1f}s")

            if is_main() and val_loss < self.best_loss:
                self.best_loss = val_loss
                ckpt = self._build_checkpoint(epoch, val_loss, wer)
                torch.save(ckpt, os.path.join(save_dir, 'phase1_mt5_best.pth'))
                log(f"  Saved best (CE: {val_loss:.4f}, WER: {wer:.4f})")

            if is_main() and (epoch + 1) % 5 == 0:
                ckpt = self._build_checkpoint(epoch, val_loss, wer)
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
    args = parser.parse_args()

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
        masked_pose_ratio=args.masked_pose_ratio,
        overfit_n=args.overfit_n,
        ctc_weight=args.ctc_weight,
        ctc_vocab_size=args.ctc_vocab_size,
        resume=args.resume,
        grad_accum=args.grad_accum,
    )

    trainer.train(num_epochs=args.epochs, save_dir=args.save_dir)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
