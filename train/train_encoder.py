"""
DEPRECATED — superseded by train/train_encoder_mt5.py (Uni-Sign encoder + MT5).
Kept for reference only. Uses the old CTR-GCN encoder / CTC or AE decoder path
and is not part of the current pipeline. Do not run for new experiments.
"""
"""
Phase 1: Train Keypoint Encoder + Gloss Decoder (CTC).
Maps sign language keypoints → Kazakh text.

Multi-GPU:
  Single GPU: python train/train_encoder.py --config ...
  Multi GPU:  torchrun --nproc_per_node=2 train/train_encoder.py --config ...
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
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

from data.khabar_dataset import KhabarKzDataset, KhabarKzCollator
from data.kazsign_dataset import KazSignDataset, KazSignCollator
from data.informburo_dataset import InformburoDataset, InformburoCollator
from torch.utils.data import ConcatDataset
from models.keypoint_encoder import KeypointEncoder
from models.gloss_decoder import GlossDecoderCTC
from utils.losses import compute_ctc_loss
from utils.metrics import compute_batch_wer_cer


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


class EncoderTrainer:
    """Trainer for Phase 1: Encoder + Gloss Decoder."""

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

        self.decoder = GlossDecoderCTC(
            d_model=self.cfg['d_model'],
            vocab_size=self.cfg['vocab_size'],
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
                self.model, device_ids=[local_rank], output_device=local_rank
            )
            self.encoder = self.model.module.encoder
            self.decoder = self.model.module.decoder
        else:
            self.encoder = self.model.encoder
            self.decoder = self.model.decoder

        # Optimizer
        base_lr = self.train_cfg.get('learning_rate', 5e-4)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=self.train_cfg.get('weight_decay', 0.01),
        )

        # Scheduler: warmup (100 steps) → cosine annealing over max_epochs
        self.max_epochs = self.train_cfg.get('max_epochs', 50)
        self.warmup_steps = 100
        self.scheduler_warmup = LinearLR(
            self.optimizer,
            start_factor=0.05,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        self.scheduler_cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epochs,
            eta_min=1e-6,
        )
        self.warmup_done = False
        self.global_step = 0

        # Tokenizer
        self.tokenizer = None

        # Metrics
        self.best_ctc_loss = float('inf')

        log(f"[Phase 1] Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        log(f"[Phase 1] Device: {self.device}")
        n_gpu = dist.get_world_size() if self.distributed else 1
        log(f"[Phase 1] GPUs: {n_gpu}, Base LR: {base_lr}, Warmup: {self.warmup_steps} steps, Cosine: {self.max_epochs} epochs")

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def create_datasets(self):
        """Create training/validation datasets from all available Kazakh datasets."""
        split = 0.9
        paths = self.config['paths']

        all_datasets = []

        # Khabar KZ (25fps, no downsampling)
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

        # KazSign — if path exists and keypoints extracted
        if 'kazsign' in paths:
            import glob
            kazsign_root = paths['kazsign'].get('root', '')
            kazsign_kps = paths['kazsign'].get('keypoints', '')
            # Check if extracted keypoints exist (look for *.npz)
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

        # Informburo KZ (50fps → downsample to 25fps)
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

        # Also try raw Khabar KZ (if available alongside the segmented version)
        if 'khabar_source' in paths:
            khabar_src_kps = paths['khabar_source'].get('keypoints', '')
            khabar_src_txt = paths['khabar_source'].get('transcripts', '')
            if khabar_src_kps and os.path.exists(khabar_src_kps):
                # Check if it has category subdirs (raw Khabar format)
                categories = ['alemde', 'densaulyk', 'ekologiya', 'ekonomika',
                              'kogam', 'madeniet', 'okiga', 'sayasat', 'sport']
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
                        downsample_every=1,  # Khabar is already 25fps
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

        if self.distributed:
            # Use DistributedSampler — no manual random_split needed
            collator = KhabarKzCollator(max_text_len=self.train_cfg['max_text_len'])
            train_sampler = DistributedSampler(dataset, shuffle=True)
            val_subset = torch.utils.data.Subset(dataset, list(range(n_train, total)))
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
            if is_main():
                log(f"[Datasets] Combined: {total} clips → train {n_train}, val {n_val} (distributed)")
            return train_loader, val_loader
        else:
            # Single GPU: manual split
            train_set, val_set = torch.utils.data.random_split(
                dataset, [n_train, n_val]
            )
            collator = KhabarKzCollator(max_text_len=self.train_cfg['max_text_len'])
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
            text_ids = batch['text_ids'].to(self.device)
            input_lens = batch['input_lengths'].to(self.device)
            text_lens = batch['text_lengths'].to(self.device)

            # Filter empty targets
            valid_mask = text_lens > 0
            if not valid_mask.all():
                kps = kps[valid_mask]
                text_ids = text_ids[valid_mask]
                input_lens = input_lens[valid_mask]
                text_lens = text_lens[valid_mask]

            if kps.size(0) < 1:
                continue

            # Forward
            latent, _ = self.encoder(kps)
            logits = self.decoder(latent)

            # CTC loss
            input_lens_cls = input_lens + 1  # +1 for [CLS]
            ctc_loss = compute_ctc_loss(
                logits=logits,
                text_ids=text_ids,
                input_lengths=input_lens_cls,
                text_lengths=text_lens,
                blank=self.decoder.blank_token,
            )

            if ctc_loss.isnan() or ctc_loss.isinf():
                if batch_idx < 5:
                    log(f"  [WARN] NaN/Inf CTC loss at batch {batch_idx}, skipping.")
                continue

            # Backward
            self.optimizer.zero_grad()
            ctc_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=20.0
            ).item()
            self.optimizer.step()

            # Warmup scheduler
            if not self.warmup_done:
                self.scheduler_warmup.step()
                if self.global_step >= self.warmup_steps:
                    self.warmup_done = True
                    log(f"  [INFO] Warmup complete at step {self.global_step}. Switching to cosine annealing.")
            self.global_step += 1

            total_loss += ctc_loss.item()
            total_grad_norm += grad_norm
            num_batches += 1

            # Debug at batch 50 (rank 0 only)
            if is_main() and batch_idx == 49:
                log(f"\n  [Debug Batch 50, Epoch {epoch+1}]")
                with torch.no_grad():
                    log(f"    KPS shape: {kps.shape}, mean: {kps.mean():.4f}, std: {kps.std():.4f}, max: {kps.max():.4f}")
                    log(f"    KPS nonzero frac: {(kps != 0).sum().item() / kps.numel():.4f}")
                    log(f"    Text lens: {text_lens.cpu().numpy().tolist()[:5]}")
                    log(f"    Input lens: {input_lens.cpu().numpy().tolist()[:5]}")

                    latent, _ = self.encoder(kps)
                    latent_t = latent[0, 1:, :]
                    log(f"    Latent shape: {latent.shape}")
                    log(f"    Latent[0,1:] mean: {latent_t.mean(dim=0)}")
                    log(f"    Latent[0,1:] std across time: {latent_t.std(dim=0).mean():.6f}")
                    a = latent_t[:10].mean(dim=0)
                    b = latent_t[-10:].mean(dim=0)
                    cos_sim = (a * b).sum() / (a.norm() * b.norm())
                    log(f"    CosSim(first 10, last 10 time steps): {cos_sim:.4f}")

                    probs = logits[0, :input_lens_cls[0], :].softmax(dim=-1)
                    top5_vals_s, top5_idx_s = probs[:5].topk(5, dim=-1)
                    for t in range(min(5, top5_idx_s.size(0))):
                        tokens = top5_idx_s[t].cpu().tolist()
                        scores = top5_vals_s[t].cpu().tolist()
                        decoded = []
                        for tid in tokens:
                            if tid == self.decoder.blank_token:
                                decoded.append("<blank>")
                            elif self.tokenizer and tid < self.tokenizer.vocab_size():
                                decoded.append(self.tokenizer.decode([tid]))
                            else:
                                decoded.append(f"[{tid}]")
                        log(f"    t={t}: {list(zip(tokens, decoded))}  scores={[f'{s:.3f}' for s in scores]}")

                    token_seqs = self.decoder.decode_greedy(logits, input_lens_cls)
                    for i in range(min(3, len(token_seqs))):
                        pred_tokens = token_seqs[i].tolist()
                        pred_text = self.tokenizer.decode(pred_tokens) if self.tokenizer else str(pred_tokens)
                        ref_text = batch['texts'][i] if i < len(batch['texts']) else "N/A"
                        log(f"    Pred [{len(pred_tokens)} tokens]: [{pred_text}]")
                        log(f"    Ref:  [{ref_text}]")
                log()

            if batch_idx % 500 == 0 and batch_idx > 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                avg_gn = total_grad_norm / num_batches
                log(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {avg_loss:.4f} | LR: {lr:.6f} | GradNorm: {avg_gn:.4f}")

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
            text_ids = batch['text_ids'].to(self.device)
            input_lens = batch['input_lengths'].to(self.device)
            text_lens = batch['text_lengths'].to(self.device)
            texts = batch.get('texts', [])

            valid_mask = text_lens > 0
            kps_v = kps[valid_mask]
            text_ids_v = text_ids[valid_mask]
            input_lens_v = input_lens[valid_mask]
            text_lens_v = text_lens[valid_mask]

            if kps_v.size(0) < 1:
                continue

            latent, _ = self.encoder(kps_v)
            logits = self.decoder(latent)

            input_lens_cls = input_lens_v + 1
            ctc_loss = compute_ctc_loss(
                logits=logits,
                text_ids=text_ids_v,
                input_lengths=input_lens_cls,
                text_lengths=text_lens_v,
                blank=self.decoder.blank_token,
            )

            if ctc_loss.isnan() or ctc_loss.isinf():
                continue

            total_loss += ctc_loss.item()
            num_batches += 1

            # Decode for WER (rank 0 only)
            if is_main() and self.tokenizer:
                token_seqs = self.decoder.decode_greedy(logits, input_lens_cls)
                valid_texts = [texts[i] for i in range(len(texts)) if valid_mask[i]]
                for tokens in token_seqs:
                    text = self.tokenizer.decode(tokens.tolist())
                    all_hyps.append(text)
                all_refs.extend(valid_texts)

        avg_loss = total_loss / max(num_batches, 1)

        wer = 0.0
        if is_main() and all_refs and all_hyps and len(all_refs) == len(all_hyps):
            wer, _ = compute_batch_wer_cer(all_refs, all_hyps)

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
        log(f"Phase 1: Encoder + Gloss Decoder (CTC)")
        log(f"Epochs: {num_epochs} | Warmup: {self.warmup_steps} steps")
        log(f"Scheduler: LinearLR → CosineAnnealingLR")
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

            # Cosine scheduler step (after warmup)
            if self.warmup_done:
                self.scheduler_cosine.step(epoch)

            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]['lr']

            log(f"\nEpoch {epoch+1}/{num_epochs} | "
                  f"Train CTC: {train_loss:.4f} | "
                  f"Val CTC: {val_loss:.4f} | "
                  f"WER: {wer:.4f} | "
                  f"LR: {lr:.6f} | "
                  f"GradNorm: {grad_norm:.4f} | "
                  f"Time: {epoch_time:.1f}s")

            # Save best (rank 0 only)
            if is_main() and val_loss < self.best_ctc_loss:
                self.best_ctc_loss = val_loss
                torch.save({
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'wer': wer,
                }, os.path.join(save_dir, 'phase1_best.pth'))
                log(f"  Saved best model (CTC: {val_loss:.4f}, WER: {wer:.4f})")

            # Checkpoint every 5 epochs
            if is_main() and (epoch + 1) % 5 == 0:
                torch.save({
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, f'phase1_epoch{epoch+1}.pth'))

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--tokenizer', default=None, help='Path to SentencePiece model')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--save-dir', default=None)
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local process rank (set by torchrun)')
    args = parser.parse_args()

    # Initialize DDP if launched by torchrun
    # IMPORTANT: set_device MUST come before init_process_group!
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

    trainer = EncoderTrainer(config_path=args.config, local_rank=local_rank)

    if args.tokenizer:
        tokenizer = load_tokenizer(args.tokenizer)
        trainer.set_tokenizer(tokenizer)
        log(f"Loaded tokenizer from {args.tokenizer}")

    if args.epochs:
        trainer.max_epochs = args.epochs
        trainer.scheduler_cosine = CosineAnnealingLR(
            trainer.optimizer, T_max=args.epochs, eta_min=1e-6
        )

    trainer.train(num_epochs=args.epochs, save_dir=args.save_dir)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
