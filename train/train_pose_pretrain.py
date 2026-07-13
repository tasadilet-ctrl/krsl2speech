"""
Phase 1 pre-step: dedicated self-supervised masked-pose pretraining for the
Uni-Sign ST-GCN encoder alone -- no mT5, no text, no CE/CTC loss.

Why this exists: in the main Phase 1 run (train_encoder_mt5.py), masked-pose
reconstruction is only a small 0.1-weighted auxiliary loss riding alongside
cross-entropy. diagnose_phase1.py found the encoder's embeddings still
collapse to near-identical across genuinely different clips (pairwise
cosine ~0.98-0.995) even after 13 real KRSL epochs, with seven separate
architecture/training hypotheses ruled out by direct testing (CTC weight,
part_para/pose_proj bias, cross-attention starvation, encoder LR,
BatchNorm calibration, detection dropouts, freeze-spatial+temporal LR --
see diagnose_phase1.py's git history for each ablation). The Uni-Sign hands
load pretrained CSL weights exactly; body and face are mostly reinitialized
(load_unisign_weights' per-group load summary) and may simply need far more
dedicated exposure to real KRSL pose data -- via the SAME masking
objective, just as the SOLE training signal for many more epochs -- than a
0.1-weighted side loss buried under cross-entropy ever gave them.

The saved checkpoint is a drop-in --pretrained-encoder for
train_encoder_mt5.py (same 'encoder' key, same state_dict layout), so
Phase 1 needs no code changes to consume it.

Single-GPU only (matches this project's one-GPU-per-box infrastructure;
no DDP here, unlike train_encoder_mt5.py/train_prosody.py).

Usage:
  PYTHONPATH=. python train/train_pose_pretrain.py \
      --config configs/config.yaml \
      --pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth \
      --use-enriched --masked-pose-ratio 0.2 \
      --epochs 100 --save-dir output/pose_pretrain_v1

Then feed the result into Phase 1:
  PYTHONPATH=. python train/train_encoder_mt5.py \
      --pretrained-encoder output/pose_pretrain_v1/pose_pretrain_best.pth \
      --use-enriched --ctc-weight 0.3 ...
"""
import os
import time
import math
import yaml
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from data.asan_dataset import AsanDataset, AsanCollator
from data.utils import ENRICHED_DIM, KEYPOINT_DIM
from models.unisign_encoder import (
    KeypointEncoder, load_unisign_weights, build_masked_pose_decoder)
from train.train_encoder_mt5 import build_pose_mask


class PosePretrainer:
    def __init__(self, config_path='configs/config.yaml', pretrained_unisign=None,
                 pretrained_encoder=None, use_enriched=False, masked_pose_ratio=0.2,
                 grad_accum=None, freeze_spatial=False):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        from utils.paths import apply_env_overrides
        self.config = apply_env_overrides(self.config)

        if masked_pose_ratio <= 0:
            raise ValueError("--masked-pose-ratio must be > 0 -- it's the "
                             "only training signal in this script")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = self.config['model']
        self.train_cfg = self.config['training'].get('pose_pretrain', {})
        self.use_enriched = use_enriched
        self.masked_pose_ratio = masked_pose_ratio

        input_dim = ENRICHED_DIM() if use_enriched else KEYPOINT_DIM
        self.encoder = KeypointEncoder(hidden_dim=self.cfg['d_model'], input_dim=input_dim)

        if pretrained_unisign:
            print(f"Loading Uni-Sign pretrained weights from {pretrained_unisign}")
            load_unisign_weights(self.encoder, pretrained_unisign)
        if pretrained_encoder:
            print(f"Resuming encoder from {pretrained_encoder}")
            ckpt = torch.load(pretrained_encoder, map_location='cpu')
            self.encoder.load_state_dict(ckpt['encoder'] if 'encoder' in ckpt else ckpt)

        if freeze_spatial:
            self.encoder.freeze_spatial()

        self.masked_pose_decoder = build_masked_pose_decoder(self.cfg['d_model'], input_dim)
        self.encoder.to(self.device)
        self.masked_pose_decoder.to(self.device)

        base_lr = self.train_cfg.get('learning_rate', 1e-3)
        trainable_params = ([p for p in self.encoder.parameters() if p.requires_grad]
                            + list(self.masked_pose_decoder.parameters()))
        self.optimizer = AdamW(trainable_params, lr=base_lr,
                               weight_decay=self.train_cfg.get('weight_decay', 0.01))

        self.warmup_steps = self.train_cfg.get('warmup_steps', 1000)
        self.grad_accum = (grad_accum if grad_accum is not None
                          else max(1, int(self.train_cfg.get('grad_accum', 4))))
        self.max_epochs = self.train_cfg.get('max_epochs', 100)
        self.best_loss = float('inf')
        self.global_step = 0
        self.scheduler = None

        n_enc = sum(p.numel() for p in self.encoder.parameters())
        n_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(f"\n[PosePretrain] Encoder: {n_enc:,} total, {n_trainable:,} trainable")
        print(f"[PosePretrain] Masking ratio: {masked_pose_ratio} (sole objective)")
        print(f"[PosePretrain] LR: {base_lr}, grad_accum: {self.grad_accum}")
        if freeze_spatial:
            print("[PosePretrain] Spatial STGCN frozen -- only temporal blocks, "
                  "pose_proj, part_para, and the reconstruction decoder train")

    def create_datasets(self):
        """Same asan_common construction as MT5Trainer.create_datasets, minus
        the tokenizer/text machinery this script has no use for."""
        asan_cfg = self.config['paths']['asan']
        common = dict(
            root=asan_cfg['root'],
            sources=asan_cfg.get('sources', ['informburo', 'khabar', 'qazaqstantv']),
            lang=asan_cfg.get('lang', 'kz'),
            tokenizer=None,
            max_frames=self.train_cfg.get('max_seq_len', 1000),
            downsample_every=asan_cfg.get('downsample_every', 1),
            use_enriched=self.use_enriched,
            skip_low_quality=asan_cfg.get('skip_low_quality', True),
            min_hand_cov=asan_cfg.get('min_hand_cov', 0.0),
        )
        train_set = AsanDataset(split='train', **common)
        val_set = AsanDataset(split='val', **common)
        collator = AsanCollator()
        batch_size = self.train_cfg.get('batch_size', 8)
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=4,
            collate_fn=collator, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=2,
            collate_fn=collator, pin_memory=True)
        print(f"[PosePretrain Datasets] train={len(train_set)}, val={len(val_set)}")
        return train_loader, val_loader

    def _step(self, batch):
        kps = batch['keypoints'].to(self.device)
        input_lengths = batch['input_lengths'].to(self.device)
        mask = build_pose_mask(kps, input_lengths, self.masked_pose_ratio)
        kps_masked = torch.where(mask, torch.zeros_like(kps), kps)
        emb = self.encoder(kps_masked, input_lengths=input_lengths)
        reconstructed = self.masked_pose_decoder(emb)
        sel = mask.expand_as(kps)
        if sel.any():
            return F.mse_loss(reconstructed[sel], kps[sel])
        return reconstructed.sum() * 0.0

    def _clip_grad(self):
        return torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.masked_pose_decoder.parameters()), 1.0)

    def train_epoch(self, train_loader):
        self.encoder.train()
        self.masked_pose_decoder.train()
        total_loss, num_batches, pending = 0.0, 0, 0
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            loss = self._step(batch)
            (loss / self.grad_accum).backward()
            pending += 1
            if torch.isfinite(loss):
                total_loss += loss.item()
                num_batches += 1
            if pending >= self.grad_accum:
                total_norm = self._clip_grad()
                if torch.isfinite(total_norm):
                    self.optimizer.step()
                self.optimizer.zero_grad()
                pending = 0
                self.global_step += 1
                self.scheduler.step()
            if batch_idx % 500 == 0 and batch_idx > 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} | "
                      f"MSE: {total_loss / max(num_batches, 1):.4f}")
        if pending > 0:
            total_norm = self._clip_grad()
            if torch.isfinite(total_norm):
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1
            self.scheduler.step()
        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader):
        self.encoder.eval()
        self.masked_pose_decoder.eval()
        total_loss, num_batches = 0.0, 0
        for batch in val_loader:
            if batch is None:
                continue
            loss = self._step(batch)
            if torch.isfinite(loss):
                total_loss += loss.item()
                num_batches += 1
        return total_loss / max(num_batches, 1)

    def _build_checkpoint(self, epoch, val_loss):
        return {
            'encoder': self.encoder.state_dict(),
            'masked_pose_decoder': self.masked_pose_decoder.state_dict(),
            'epoch': epoch, 'val_loss': val_loss,
        }

    def train(self, num_epochs=None, save_dir=None):
        if num_epochs is None:
            num_epochs = self.max_epochs
        if save_dir is None:
            save_dir = self.config['paths']['output']
        os.makedirs(save_dir, exist_ok=True)

        train_loader, val_loader = self.create_datasets()
        total_steps = math.ceil(len(train_loader) / self.grad_accum) * num_epochs
        warmup_eff = min(self.warmup_steps, max(total_steps // 10, 1))
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_eff, num_training_steps=total_steps)

        print(f"\n{'=' * 60}\nPose pretraining (masked-pose reconstruction only)")
        print(f"Epochs: {num_epochs}\n{'=' * 60}\n")

        for epoch in range(num_epochs):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            print(f"Epoch {epoch + 1}/{num_epochs} | Train MSE: {train_loss:.4f} | "
                  f"Val MSE: {val_loss:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                  f"Time: {time.time() - t0:.1f}s")
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self._build_checkpoint(epoch, val_loss),
                          os.path.join(save_dir, 'pose_pretrain_best.pth'))
                print(f"  Saved best (MSE: {val_loss:.4f})")
            if (epoch + 1) % 10 == 0:
                torch.save(self._build_checkpoint(epoch, val_loss),
                          os.path.join(save_dir, f'pose_pretrain_epoch{epoch + 1}.pth'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--pretrained-unisign', default=None,
                        help='Raw Uni-Sign/CSL checkpoint to start from')
    parser.add_argument('--pretrained-encoder', default=None,
                        help='Resume from a previous pose-pretrain checkpoint')
    parser.add_argument('--use-enriched', action='store_true')
    parser.add_argument('--masked-pose-ratio', type=float, default=0.2,
                        help="Masking ratio -- the SOLE training signal here, "
                             "so higher than Phase 1's 0.05 auxiliary-loss "
                             "default. SignBERT+-style pretraining typically "
                             "uses 0.15-0.4.")
    parser.add_argument('--freeze-spatial', action='store_true',
                        help='Freeze the CSL-pretrained spatial (hand-shape) '
                             'blocks; only pretrain the temporal blocks + '
                             'pose_proj + part_para + reconstruction decoder.')
    parser.add_argument('--grad-accum', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--save-dir', default=None)
    args = parser.parse_args()

    trainer = PosePretrainer(
        config_path=args.config,
        pretrained_unisign=args.pretrained_unisign,
        pretrained_encoder=args.pretrained_encoder,
        use_enriched=args.use_enriched,
        masked_pose_ratio=args.masked_pose_ratio,
        grad_accum=args.grad_accum,
        freeze_spatial=args.freeze_spatial,
    )
    trainer.train(num_epochs=args.epochs, save_dir=args.save_dir)


if __name__ == '__main__':
    main()
