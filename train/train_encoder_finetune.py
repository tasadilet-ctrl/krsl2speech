"""
DEPRECATED — superseded by train/train_encoder_mt5.py (Uni-Sign encoder + MT5).
Kept for reference only. Uses the old CTR-GCN encoder / CTC or AE decoder path
and is not part of the current pipeline. Do not run for new experiments.
"""
"""
Phase 1 Fine-Tuning: Uni-Sign Pretrained Encoder + Autoregressive Decoder.

Loads Uni-Sign pretrained weights, fine-tunes on KRSL data.

Usage:
  # Download pretrained weights:
  # From https://huggingface.co/ZechengLi19/Uni-Sign/tree/main
  # Recommended: csl_stage1_weight.pth  (pose-only pretraining, ~1.19 GB)
  #              or csl_daily_pose_only_slt.pth (SLT fine-tuned, ~1.19 GB)

  # Single GPU:
  python train/train_encoder_finetune.py \
      --config configs/config.yaml \
      --tokenizer /path/to/sp_model.model \
      --pretrained /path/to/csl_stage1_weight.pth \
      --freeze-spatial  # optional: freeze spatial STGCN

  # Multi-GPU:
  torchrun --nproc_per_node=2 train/train_encoder_finetune.py \
      --config configs/config.yaml \
      --tokenizer /path/to/sp_model.model \
      --pretrained /path/to/csl_stage1_weight.pth

Architecture:
  Encoder: Uni-Sign ST-GCN (pretrained, partial load)
  Decoder: Autoregressive Transformer (trained from scratch)
  Loss: Cross-entropy with label smoothing
  Scheduler: Noam (warmup + inverse sqrt)
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

from data.khabar_dataset import KhabarKzDataset
from data.kazsign_dataset import KazSignDataset
from data.informburo_dataset import InformburoDataset
from torch.utils.data import ConcatDataset
from models.unisign_encoder import KeypointEncoder, load_unisign_weights
from models.gloss_decoder import GlossDecoder


# ============================================================
# Special token IDs
# ============================================================
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


# ============================================================
# Noam Scheduler
# ============================================================

class NoamScheduler:
    """
    Noam learning rate schedule: warmup → inverse sqrt decay.
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
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
    Creates source/target pairs with <sos>/<eos> tokens.
    """

    def __init__(self, max_text_len=256, pad_idx=PAD_IDX,
                 sos_idx=SOS_IDX, eos_idx=EOS_IDX):
        self.max_text_len = max_text_len
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def __call__(self, batch):
        valid = [b for b in batch if b is not None]
        if not valid:
            return None

        kps = [b['keypoints'] for b in valid]
        max_t = max(k.shape[0] for k in kps)
        input_lengths = torch.tensor([k.shape[0] for k in kps], dtype=torch.long)

        kps_padded = torch.zeros(
            len(valid), max_t, kps[0].shape[1], dtype=torch.float32,
        )
        for i, k in enumerate(kps):
            kps_padded[i, :k.shape[0], :] = k

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

        src_list = []
        tgt_list = []
        text_lengths = []

        # Filter out samples with no text
        valid_with_text = []
        for b in valid:
            raw_ids = []
            if b.get('text_ids') is not None:
                raw_ids = b['text_ids'].tolist()
            raw_ids = raw_ids[:self.max_text_len]
            if len(raw_ids) == 0:
                continue
            valid_with_text.append(b)
            src_list.append([self.sos_idx] + raw_ids)
            tgt_list.append(raw_ids + [self.eos_idx])
            text_lengths.append(len(raw_ids) + 1)

        if not valid_with_text:
            return None

        # Pad to max length in batch
        max_src_len = max(len(s) for s in src_list)
        for i in range(len(src_list)):
            pad = [self.pad_idx] * (max_src_len - len(src_list[i]))
            src_list[i] = torch.tensor(src_list[i] + pad, dtype=torch.long)
            tgt_list[i] = torch.tensor(tgt_list[i] + pad, dtype=torch.long)

        # Rebuild keypoints/scores for filtered batch
        valid = valid_with_text
        kps = [b['keypoints'] for b in valid]
        max_t = max(k.shape[0] for k in kps)
        input_lengths = torch.tensor([k.shape[0] for k in kps], dtype=torch.long)
        kps_padded = torch.zeros(len(valid), max_t, kps[0].shape[1], dtype=torch.float32)
        for i, k in enumerate(kps):
            kps_padded[i, :k.shape[0], :] = k

        scores_list = [b.get('scores') for b in valid]
        if scores_list[0] is not None:
            scores_raw = [s for s in scores_list]
            scores_padded = torch.zeros(len(valid), max_t, scores_raw[0].shape[1], dtype=torch.float32)
            for i, s in enumerate(scores_raw):
                scores_padded[i, :s.shape[0], :] = s
        else:
            scores_padded = None

        return {
            'keypoints': kps_padded,
            'scores': scores_padded,
            'input_lengths': input_lengths,
            'text_src': torch.stack(src_list),
            'text_tgt': torch.stack(tgt_list),
            'text_lengths': torch.tensor(text_lengths, dtype=torch.long),
            'texts': [b.get('text', '') for b in valid],
        }


# ============================================================
# Helpers
# ============================================================

def load_tokenizer(model_path):
    from sentencepiece import SentencePieceProcessor
    sp = SentencePieceProcessor(model_path)
    print(f"  vocab_size={sp.vocab_size()}, pad_id={sp.pad_id()}, unk_id={sp.unk_id()}")
    return sp


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)


def compute_wer_refs_hyps(refs, hyps):
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

class FinetuneTrainer:
    """
    Fine-tunes Uni-Sign pretrained encoder + autoregressive decoder on KRSL.

    Strategy:
      Phase A: Freeze spatial STGCN, train temporal + decoder (10 epochs)
      Phase B: Unfreeze all, train end-to-end (remaining epochs)
    """

    def __init__(self, config_path='configs/config.yaml', local_rank=0,
                 pretrained_path=None, freeze_spatial=False):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{local_rank}')
        self.cfg = self.config['model']
        self.train_cfg = self.config['training']['phase1']

        # Build encoder with pretrained weights
        self.encoder = KeypointEncoder(
            hidden_dim=self.cfg['d_model'],
            pretrained_path=pretrained_path,
        )

        if freeze_spatial:
            self.encoder.freeze_spatial()

        # Build decoder (from scratch)
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

        # DDP wrap
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

        # Optimizer (differential LR for encoder vs decoder)
        encoder_lr = self.train_cfg.get('learning_rate', 5e-4) / 10
        decoder_lr = self.train_cfg.get('learning_rate', 5e-4)
        param_groups = [
            {'params': self.encoder.parameters(), 'lr': encoder_lr},
            {'params': self.decoder.parameters(), 'lr': decoder_lr},
        ]
        self.optimizer = AdamW(
            param_groups,
            weight_decay=self.train_cfg.get('weight_decay', 0.01),
        )

        # Noam scheduler
        warmup_steps = self.train_cfg.get('warmup_steps', 4000)
        self.scheduler = NoamScheduler(
            self.optimizer, d_model=self.cfg['d_model'],
            warmup_steps=warmup_steps,
        )

        self.max_epochs = self.train_cfg.get('max_epochs', 50)
        self.global_step = 0
        self.tokenizer = None
        self.best_loss = float('inf')
        self.freeze_spatial = freeze_spatial

        total, trainable, frozen = self.encoder.trainable_params()
        log(f"\n[Finetune] Encoder: {trainable:,}/{total:,} trainable "
            f"({frozen:,} frozen)")
        log(f"[Finetune] Encoder LR: {encoder_lr:.6f}, Decoder LR: {decoder_lr:.6f}")
        log(f"[Finetune] Device: {self.device}")
        n_gpu = dist.get_world_size() if self.distributed else 1
        log(f"[Finetune] GPUs: {n_gpu}")

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def create_datasets(self):
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
            kazsign_kps = paths['kazsign'].get('keypoints', '')
            if kazsign_kps and glob.glob(os.path.join(kazsign_kps, '*.npz')):
                kazsign = KazSignDataset(
                    data_root=paths['kazsign'].get('root', ''),
                    keypoints_root=kazsign_kps,
                    tokenizer=self.tokenizer,
                    max_duration=60.0,
                    min_duration=2.0,
                    max_frames=self.train_cfg['max_seq_len'],
                    load_prosody=False,
                )
                all_datasets.append(kazsign)
            else:
                log("[INFO] KazSign: keypoints not extracted, skipping")

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

            # Forward
            pose_emb = self.encoder(kps, scores)  # (B, T, 768)

            # Add [CLS]-style prefix (learnable position for cross-attention)
            # The decoder treats pose_emb as memory
            logits, loss, ppl = self.decoder(
                sign_latent=pose_emb,
                text_input=text_src,
                text_lengths=text_lens,
                input_lengths=input_lens,
            )

            if loss.isnan() or loss.isinf():
                continue

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0,
            ).item()
            self.optimizer.step()
            lr = self.scheduler.step()

            total_loss += loss.item()
            total_grad_norm += grad_norm
            num_batches += 1
            self.global_step += 1

            if is_main() and batch_idx % 500 == 0 and batch_idx > 0:
                avg_loss = total_loss / num_batches
                log(f"  Batch {batch_idx}/{len(train_loader)} | "
                     f"CE: {avg_loss:.4f} | LR: {lr:.6f} | "
                     f"GradNorm: {total_grad_norm / num_batches:.4f}")

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

            pose_emb = self.encoder(kps_v, scores)
            logits, loss, ppl = self.decoder(
                sign_latent=pose_emb,
                text_input=text_src_v,
                text_lengths=text_lens_v,
                input_lengths=input_lens_v,
            )

            if loss.isnan() or loss.isinf():
                continue

            total_loss += loss.item()
            num_batches += 1

            if is_main() and self.tokenizer:
                gen_seqs = self.decoder.generate(
                    pose_emb, max_len=200, sos_idx=SOS_IDX, eos_idx=EOS_IDX,
                )
                for seq in gen_seqs:
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
        if num_epochs is None:
            num_epochs = self.train_cfg['max_epochs']
        if save_dir is None:
            save_dir = self.config['paths']['output']

        if is_main():
            os.makedirs(save_dir, exist_ok=True)

        train_loader, val_loader = self.create_datasets()

        log(f"\n{'='*60}")
        log(f"Fine-Tuning: Uni-Sign Pretrained Encoder + AE Decoder")
        log(f"Loss: Cross-Entropy + Label Smoothing (0.1)")
        log(f"Scheduler: Noam (warmup + inverse sqrt)")
        log(f"Freeze spatial: {self.freeze_spatial}")
        log(f"Encoder LR: {self.train_cfg.get('learning_rate', 5e-4) / 10:.6f}")
        log(f"Decoder LR: {self.train_cfg.get('learning_rate', 5e-4):.6f}")
        log(f"Epochs: {num_epochs}")
        log(f"{'='*60}\n")

        for epoch in range(num_epochs):
            epoch_start = time.time()

            if self.distributed and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            train_loss, grad_norm = self.train_epoch(train_loader, epoch)
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

            # Save best
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
                }, os.path.join(save_dir, 'phase1_finetune_best.pth'))
                log(f"  Saved best (CE: {val_loss:.4f}, WER: {wer:.4f})")

            # Checkpoint every 5 epochs
            if is_main() and (epoch + 1) % 5 == 0:
                torch.save({
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler_step': self.scheduler.step_num,
                    'epoch': epoch,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, f'phase1_finetune_epoch{epoch+1}.pth'))

        if self.distributed:
            dist.barrier()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--tokenizer', default=None,
                        help='Path to SentencePiece model')
    parser.add_argument('--pretrained', default=None,
                        help='Path to Uni-Sign pretrained checkpoint')
    parser.add_argument('--freeze-spatial', action='store_true',
                        help='Freeze spatial STGCN during fine-tuning')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--save-dir', default=None)
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local process rank (set by torchrun)')
    args = parser.parse_args()

    # DDP init
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

    if not args.pretrained:
        log("[WARN] No --pretrained path provided. Training from scratch.")
        log("[INFO] Download from: https://huggingface.co/ZechengLi19/Uni-Sign/tree/main")
        log("[INFO] Recommended: csl_stage1_weight.pth or csl_daily_pose_only_slt.pth")

    trainer = FinetuneTrainer(
        config_path=args.config, local_rank=local_rank,
        pretrained_path=args.pretrained, freeze_spatial=args.freeze_spatial,
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
