"""
DEPRECATED — superseded by train/train_encoder_mt5.py (Uni-Sign encoder + MT5).
Kept for reference only. Uses the old CTR-GCN encoder / CTC or AE decoder path
and is not part of the current pipeline. Do not run for new experiments.
"""
"""
Phase 1: Train Keypoint Encoder + Gloss Decoder (Cross-Entropy).

New architecture (paper-backed):
  Encoder: CTR-GCN + MS-TCN (from GloFE/S2PFormer) with offset encoding
  Decoder: Autoregressive Transformer (from GloFE trans_model.py)
  Loss: Cross-entropy with label smoothing + teacher forcing
  Scheduler: Noam (warmup + inverse sqrt)

Multi-GPU:
  Single GPU: python train/train_encoder_ce.py --config ... --tokenizer ...
  Multi GPU:  torchrun --nproc_per_node=2 train/train_encoder_ce.py ...
"""
import os
import time
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from data.khabar_dataset import KhabarKzDataset, KhabarKzCollator
from data.kazsign_dataset import KazSignDataset, KazSignCollator
from data.informburo_dataset import InformburoDataset, InformburoCollator
from torch.utils.data import ConcatDataset
from models.ctr_gcn_encoder import KeypointEncoder
from models.gloss_decoder import GlossDecoder


# ============================================================
# Special token IDs (SentencePiece convention)
# ============================================================
PAD_IDX = 0  # <pad>
SOS_IDX = 1  # <s>
EOS_IDX = 2  # </s>
UNK_IDX = 3  # <unk>


# ============================================================
# Noam Learning Rate Scheduler (from GloFE / Attention is All You Need)
# ============================================================

class NoamScheduler:
    """
    Noam learning rate schedule: warmup → inverse sqrt decay.

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    From GloFE trans_model.py.
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.noam_lr(self.step_num)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def noam_lr(self, step):
        if step == 0:
            return 0
        return (
            self.d_model ** (-0.5) *
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )

    def get_lr(self):
        return self.noam_lr(self.step_num)


# ============================================================
# Autoregressive Collator
# ============================================================

class AECollator:
    """
    Collator for autoregressive training.

    Takes tokenized text and creates:
      - src: text shifted right (with <sos> at start)
      - tgt: original text (with <eos> at end)

    Example:
      text = [a, b, c]
      src  = [<sos>, a, b, c]
      tgt  = [a, b, c, <eos>]
    """

    def __init__(self, max_text_len=256, pad_idx=PAD_IDX,
                 sos_idx=SOS_IDX, eos_idx=EOS_IDX):
        self.max_text_len = max_text_len
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def __call__(self, batch):
        """
        Args:
            batch: list of dicts with keys:
              - 'keypoints': (T, 282)
              - 'scores': (T, 133) or None
              - 'frame_idx': (T,) or None
              - 'text_ids': (L,) tokenized text
              - 'texts': raw text string

        Returns:
            dict with batched tensors
        """
        # Filter out None
        valid = [b for b in batch if b is not None]
        if not valid:
            return None

        # Key points
        kps = [b['keypoints'] for b in valid]
        max_t = max(k.shape[0] for k in kps)
        input_lengths = torch.tensor([k.shape[0] for k in kps], dtype=torch.long)

        # Pad keypoints to max length
        kps_padded = torch.zeros(len(valid), max_t, kps[0].shape[1], dtype=torch.float32)
        for i, k in enumerate(kps):
            kps_padded[i, :k.shape[0], :] = k

        # Scores (optional)
        scores_list = [b.get('scores') for b in valid]
        if scores_list[0] is not None:
            scores_raw = [s for s in scores_list]
            scores_padded = torch.zeros(
                len(valid), max_t, scores_raw[0].shape[1], dtype=torch.float32,
            )
            for i, s in enumerate(scores_raw):
                scores_padded[i, :s.shape[0], :] = s
        else:
            scores_padded = None

        # Autoregressive text preparation
        src_list = []
        tgt_list = []
        text_lengths = []

        for b in valid:
            raw_ids = b['text_ids'].tolist()
            # Truncate
            raw_ids = raw_ids[:self.max_text_len]
            L = len(raw_ids)

            # Source: <sos> + tokens
            src = [self.sos_idx] + raw_ids
            # Target: tokens + <eos>
            tgt = raw_ids + [self.eos_idx]

            # Pad to max
            src_padded = src + [self.pad_idx] * (L + 1 - len(src))
            tgt_padded = tgt + [self.pad_idx] * (L + 1 - len(tgt))

            src_list.append(torch.tensor(src_padded, dtype=torch.long))
            tgt_list.append(torch.tensor(tgt_padded, dtype=torch.long))
            text_lengths.append(L + 1)  # include <eos>

        src_tensor = torch.stack(src_list)
        tgt_tensor = torch.stack(tgt_list)
        text_lens = torch.tensor(text_lengths, dtype=torch.long)

        # Raw texts for WER computation
        texts = [b.get('texts', '') for b in valid]

        result = {
            'keypoints': kps_padded,
            'scores': scores_padded,
            'input_lengths': input_lengths,
            'text_src': src_tensor,
            'text_tgt': tgt_tensor,
            'text_lengths': text_lens,
            'texts': texts,
        }

        return result


# ============================================================
# Helper functions
# ============================================================

def load_tokenizer(model_path):
    """Load pre-trained tokenizer."""
    from sentencepiece import SentencePieceProcessor
    sp = SentencePieceProcessor(model_path)
    print(f"  vocab_size={sp.vocab_size()}, pad_id={sp.pad_id()}, unk_id={sp.unk_id()}")
    return sp


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(*args, **kwargs):
    """Print only on main process."""
    if is_main():
        print(*args, **kwargs)


# ============================================================
# WER computation
# ============================================================

def compute_wer_refs_hyps(refs, hyps):
    """Simple WER computation using edit distance (word-level)."""
    try:
        import editdistance
        total_dist = 0
        total_words = 0
        for r, h in zip(refs, hyps):
            r_words = r.strip().split()
            h_words = h.strip().split()
            dist = editdistance.eval(r_words, h_words)
            total_dist += dist
            total_words += len(r_words)
        return total_dist / max(total_words, 1)
    except ImportError:
        return 0.0


# ============================================================
# Trainer
# ============================================================

class EncoderTrainerAE:
    """
    Trainer for Phase 1: CTR-GCN Encoder + Autoregressive Gloss Decoder.

    Architecture:
      - KeypointEncoder (CTR-GCN + MS-TCN)
      - GlossDecoder (Transformer with teacher forcing)
      - Cross-entropy loss with label smoothing
      - Noam scheduler
    """

    def __init__(self, config_path='configs/config.yaml', local_rank=0):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{local_rank}')
        self.cfg = self.config['model']
        self.train_cfg = self.config['training']['phase1']

        # Build models
        self.encoder = KeypointEncoder(
            d_model=self.cfg['d_model'],
            nhead=self.cfg['nhead'],
            num_layers=self.cfg['num_encoder_layers'],
            dim_feedforward=self.cfg['dim_feedforward'],
            dropout=self.cfg['dropout'],
        )

        self.decoder = GlossDecoder(
            vocab_size=self.cfg['vocab_size'],
            d_model=self.cfg['d_model'],
            nhead=self.cfg['nhead'],
            num_layers=3,
            dim_feedforward=self.cfg['dim_feedforward'],
            dropout=self.cfg['dropout'],
            label_smoothing=0.1,
            padding_idx=PAD_IDX,
        )

        self.model = nn.ModuleDict({
            'encoder': self.encoder,
            'decoder': self.decoder,
        })
        self.model.to(self.device)

        # DistributedDataParallel wrap
        self.distributed = dist.is_initialized()
        if self.distributed:
            self.model = DDP(
                self.model, device_ids=[local_rank], output_device=local_rank,
            )
            self.encoder = self.model.module.encoder
            self.decoder = self.model.module.decoder
        else:
            self.encoder = self.model.encoder
            self.decoder = self.model.decoder

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=0.0,  # Noam scheduler sets LR
            weight_decay=self.train_cfg.get('weight_decay', 0.01),
        )

        # Noam scheduler
        warmup_steps = self.train_cfg.get('warmup_steps', 4000)
        self.scheduler = NoamScheduler(
            self.optimizer,
            d_model=self.cfg['d_model'],
            warmup_steps=warmup_steps,
        )

        self.max_epochs = self.train_cfg.get('max_epochs', 50)
        self.global_step = 0

        # Tokenizer
        self.tokenizer = None

        # Metrics
        self.best_loss = float('inf')

        log(f"[Phase 1 AE] Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        log(f"[Phase 1 AE] Device: {self.device}")
        n_gpu = dist.get_world_size() if self.distributed else 1
        log(f"[Phase 1 AE] GPUs: {n_gpu}, Noam warmup: {warmup_steps} steps")

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def create_datasets(self):
        """Create training/validation datasets from all available Kazakh datasets."""
        split = 0.9
        paths = self.config['paths']

        all_datasets = []

        # Khabar KZ
        khabar = KhabarKzDataset(
            manifest_path=paths['khabar_kz']['manifest'],
            keypoints_root=paths['khabar_kz']['keypoints'],
            tokenizer=self.tokenizer,
            max_duration=60.0,
            min_duration=2.0,
            max_frames=self.train_cfg['max_seq_len'],
            downsample_every=1,
            name='khabar_kz',
        )
        all_datasets.append(khabar)

        # KazSign
        if 'kazsign' in paths:
            import glob
            kazsign_root = paths['kazsign'].get('root', '')
            kazsign_kps = paths['kazsign'].get('keypoints', '')
            if kazsign_kps and glob.glob(os.path.join(kazsign_kps, '*.npz')):
                kazsign = KazSignDataset(
                    data_root=kazsign_root,
                    keypoints_root=kazsign_kps,
                    tokenizer=self.tokenizer,
                    max_duration=60.0,
                    min_duration=2.0,
                    max_frames=self.train_cfg['max_seq_len'],
                    load_prosody=False,
                )
                all_datasets.append(kazsign)
            else:
                log("[INFO] KazSign: keypoints not extracted yet, skipping")

        # Informburo KZ
        if 'informburo' in paths:
            informburo_kps = paths['informburo'].get('keypoints', '')
            informburo_txt = paths['informburo'].get('transcripts', '')
            if informburo_kps and os.path.exists(informburo_kps):
                informburo = InformburoDataset(
                    keypoints_root=informburo_kps,
                    transcripts_root=informburo_txt,
                    tokenizer=self.tokenizer,
                    max_duration=60.0,
                    min_duration=2.0,
                    max_frames=self.train_cfg['max_seq_len'],
                    downsample_every=2,
                    name='informburo_kz',
                )
                all_datasets.append(informburo)
            else:
                log("[INFO] Informburo KZ: keypoints not found, skipping")

        # Raw Khabar KZ
        if 'khabar_source' in paths:
            khabar_src_kps = paths['khabar_source'].get('keypoints', '')
            khabar_src_txt = paths['khabar_source'].get('transcripts', '')
            if khabar_src_kps and os.path.exists(khabar_src_kps):
                categories = ['alemde', 'densaulyk', 'ekologiya', 'ekonomika',
                              'kogam', 'madeniet', 'okiga', 'sayasat', 'sport']
                import glob
                total_npz = 0
                for cat in categories:
                    cat_dir = os.path.join(khabar_src_kps, cat)
                    if os.path.exists(cat_dir):
                        total_npz += len(glob.glob(os.path.join(cat_dir, '*.npz')))
                if total_npz > 0:
                    khabar_raw = InformburoDataset(
                        keypoints_root=khabar_src_kps,
                        transcripts_root=khabar_src_txt,
                        tokenizer=self.tokenizer,
                        max_duration=60.0,
                        min_duration=2.0,
                        max_frames=self.train_cfg['max_seq_len'],
                        downsample_every=1,
                        name='khabar_raw_kz',
                    )
                    all_datasets.append(khabar_raw)
            else:
                log("[INFO] Raw Khabar KZ: keypoints not found, skipping")

        # Combine
        if len(all_datasets) == 1:
            dataset = all_datasets[0]
        else:
            dataset = ConcatDataset(all_datasets)

        total = len(dataset)
        n_train = int(total * split)
        n_val = total - n_train

        collator = AECollator(max_text_len=self.train_cfg['max_text_len'])

        if self.distributed:
            train_sampler = DistributedSampler(dataset, shuffle=True)
            val_subset = torch.utils.data.Subset(
                dataset, list(range(n_train, total)),
            )
            train_loader = DataLoader(
                dataset, batch_size=self.train_cfg['batch_size'],
                sampler=train_sampler, num_workers=2, collate_fn=collator,
                pin_memory=True, persistent_workers=True,
            )
            val_loader = DataLoader(
                val_subset, batch_size=self.train_cfg['batch_size'],
                shuffle=False, num_workers=2, collate_fn=collator,
                pin_memory=True,
            )
            log(f"[Datasets] Combined: {total} clips → train {n_train}, val {n_val} (distributed)")
        else:
            train_set, val_set = torch.utils.data.random_split(
                dataset, [n_train, n_val],
            )
            train_loader = DataLoader(
                train_set, batch_size=self.train_cfg['batch_size'],
                shuffle=True, num_workers=4, collate_fn=collator,
                pin_memory=True, persistent_workers=True,
            )
            val_loader = DataLoader(
                val_set, batch_size=self.train_cfg['batch_size'],
                shuffle=False, num_workers=2, collate_fn=collator,
                pin_memory=True,
            )
            log(f"[Datasets] Combined: {total} clips → train {n_train}, val {n_val}")

        return train_loader, val_loader

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        total_grad_norm = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            kps = batch['keypoints'].to(self.device)
            scores = batch['scores'].to(self.device) if batch['scores'] is not None else None
            text_src = batch['text_src'].to(self.device)
            text_tgt = batch['text_tgt'].to(self.device)
            input_lens = batch['input_lengths'].to(self.device)
            text_lens = batch['text_lengths'].to(self.device)

            # Filter empty targets
            valid_mask = text_lens > 0
            if not valid_mask.all():
                kps = kps[valid_mask]
                text_src = text_src[valid_mask]
                text_tgt = text_tgt[valid_mask]
                input_lens = input_lens[valid_mask]
                text_lens = text_lens[valid_mask]
                if scores is not None:
                    scores = scores[valid_mask]

            if kps.size(0) < 1:
                continue

            # Forward: encoder
            latent, cls_out = self.encoder(kps, scores)

            # Forward: decoder (teacher forcing)
            logits, loss, ppl = self.decoder(
                sign_latent=latent,
                text_input=text_src,
                text_lengths=text_lens,
                input_lengths=input_lens,
            )

            if loss.isnan() or loss.isinf():
                if batch_idx < 5:
                    log(f"  [WARN] NaN/Inf loss at batch {batch_idx}, skipping.")
                continue

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0,
            ).item()
            self.optimizer.step()

            # Noam scheduler step
            lr = self.scheduler.step()

            total_loss += loss.item()
            total_grad_norm += grad_norm
            num_batches += 1
            self.global_step += 1

            if is_main() and batch_idx % 500 == 0 and batch_idx > 0:
                avg_loss = total_loss / num_batches
                avg_gn = total_grad_norm / num_batches
                log(f"  Batch {batch_idx}/{len(train_loader)} | "
                     f"CE: {avg_loss:.4f} | PPL: {total_loss / max(num_batches, 1):.2f} | "
                     f"LR: {lr:.6f} | GradNorm: {avg_gn:.4f}")

        # Sync across ranks
        if self.distributed:
            loss_tensor = torch.tensor([total_loss], device=self.device)
            count_tensor = torch.tensor([num_batches], device=self.device)
            dist.all_reduce(loss_tensor, dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, dist.ReduceOp.SUM)
            total_loss = loss_tensor.item()
            num_batches = count_tensor.item()

        avg_loss = total_loss / max(num_batches, 1)
        avg_grad = total_grad_norm / max(num_batches, 1)
        return avg_loss, avg_grad

    @torch.no_grad()
    def validate(self, val_loader):
        """Validate on val set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_refs = []
        all_hyps = []

        for batch in val_loader:
            if batch is None:
                continue

            kps = batch['keypoints'].to(self.device)
            scores = batch['scores'].to(self.device) if batch['scores'] is not None else None
            text_src = batch['text_src'].to(self.device)
            text_tgt = batch['text_tgt'].to(self.device)
            input_lens = batch['input_lengths'].to(self.device)
            text_lens = batch['text_lengths'].to(self.device)
            texts = batch.get('texts', [])

            valid_mask = text_lens > 0
            kps_v = kps[valid_mask]
            text_src_v = text_src[valid_mask]
            text_tgt_v = text_tgt[valid_mask]
            input_lens_v = input_lens[valid_mask]
            text_lens_v = text_lens[valid_mask]
            valid_texts = [texts[i] for i in range(len(texts)) if valid_mask[i]]

            if kps_v.size(0) < 1:
                continue

            latent, cls_out = self.encoder(kps_v, scores)
            logits, loss, ppl = self.decoder(
                sign_latent=latent,
                text_input=text_src_v,
                text_lengths=text_lens_v,
                input_lengths=input_lens_v,
            )

            if loss.isnan() or loss.isinf():
                continue

            total_loss += loss.item()
            num_batches += 1

            # Decode for WER (rank 0 only)
            if is_main() and self.tokenizer:
                gen_seqs = self.decoder.generate(
                    latent, max_len=200, sos_idx=SOS_IDX, eos_idx=EOS_IDX,
                )
                for seq in gen_seqs:
                    # Remove padding and EOS
                    tokens = seq.cpu().tolist()
                    tokens = [t for t in tokens if t not in (PAD_IDX, EOS_IDX)]
                    text = self.tokenizer.decode(tokens) if tokens else ""
                    all_hyps.append(text)
                all_refs.extend(valid_texts)

        avg_loss = total_loss / max(num_batches, 1)

        wer = 0.0
        if is_main() and all_refs and all_hyps and len(all_refs) == len(all_hyps):
            wer = compute_wer_refs_hyps(all_refs, all_hyps)

        return avg_loss, wer

    def train(self, num_epochs=None, save_dir=None):
        """Main training loop."""
        if num_epochs is None:
            num_epochs = self.train_cfg['max_epochs']
        if save_dir is None:
            save_dir = self.config['paths']['output']

        if is_main():
            os.makedirs(save_dir, exist_ok=True)

        train_loader, val_loader = self.create_datasets()

        log(f"\n{'='*60}")
        log(f"Phase 1: CTR-GCN Encoder + Autoregressive Gloss Decoder")
        log(f"Loss: Cross-Entropy + Label Smoothing (0.1)")
        log(f"Scheduler: Noam (warmup + inverse sqrt)")
        log(f"Epochs: {num_epochs}")
        log(f"{'='*60}\n")

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Set epoch for distributed sampler
            if self.distributed and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            # Train
            train_loss, grad_norm = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, wer = self.validate(val_loader)

            epoch_time = time.time() - epoch_start
            lr = self.scheduler.get_lr()

            log(f"\nEpoch {epoch+1}/{num_epochs} | "
                  f"Train CE: {train_loss:.4f} | "
                  f"Val CE: {val_loss:.4f} | "
                  f"WER: {wer:.4f} | "
                  f"LR: {lr:.6f} | "
                  f"GradNorm: {grad_norm:.4f} | "
                  f"Time: {epoch_time:.1f}s")

            # Save best (rank 0 only)
            if is_main() and val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save({
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler_step': self.scheduler.step_num,
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'wer': wer,
                }, os.path.join(save_dir, 'phase1_best_ce.pth'))
                log(f"  Saved best model (CE: {val_loss:.4f}, WER: {wer:.4f})")

            # Checkpoint every 5 epochs
            if is_main() and (epoch + 1) % 5 == 0:
                torch.save({
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler_step': self.scheduler.step_num,
                    'epoch': epoch,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, f'phase1_ce_epoch{epoch+1}.pth'))

        if self.distributed:
            dist.barrier()

    def save(self, path):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
        }, path)
        log(f"Saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        log(f"Loaded from {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--tokenizer', default=None,
                        help='Path to SentencePiece model')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--save-dir', default=None)
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local process rank (set by torchrun)')
    args = parser.parse_args()

    # Initialize DDP if launched by torchrun
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    elif args.local_rank >= 0:
        local_rank = args.local_rank
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    else:
        local_rank = 0

    trainer = EncoderTrainerAE(
        config_path=args.config, local_rank=local_rank,
    )

    if args.tokenizer:
        tokenizer = load_tokenizer(args.tokenizer)
        trainer.set_tokenizer(tokenizer)
        log(f"Loaded tokenizer from {args.tokenizer}")

    if args.epochs:
        trainer.max_epochs = args.epochs

    trainer.train(num_epochs=args.epochs, save_dir=args.save_dir)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
