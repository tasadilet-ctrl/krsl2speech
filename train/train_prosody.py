"""
Phase 2: Train Prosody GAN.
Maps sign encoder features → prosody (F0, energy).

Uses frozen encoder from Phase 1.
"""
import os
import time
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.khabar_dataset import KhabarKzDataset
from data.prosody_dataset import ProsodyDataset, ProsodyCollator
from data.asan_dataset import AsanDataset, AsanCollator
from data.utils import KEYPOINT_DIM, ENRICHED_DIM
from models.unisign_encoder import KeypointEncoder
from models.prosody_gan import ProsodyGAN


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)


class ProsodyTrainer:
    """Trainer for Phase 2: Prosody GAN."""

    def __init__(self, config_path='configs/config.yaml', local_rank=0,
                 encoder_path=None, ckpt_path=None, use_enriched=False):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        from utils.paths import apply_env_overrides
        self.config = apply_env_overrides(self.config)

        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{local_rank}')
        self.cfg = self.config['model']
        self.train_cfg = self.config['training']['phase2']
        self.use_enriched = use_enriched

        # Build frozen encoder from Phase 1 (Uni-Sign ST-GCN).
        # input_dim MUST match how Phase 1 was trained — an enriched
        # (1410-dim) checkpoint has a (64, 5) projection that won't load
        # into the default 282-dim encoder.
        input_dim = ENRICHED_DIM() if use_enriched else KEYPOINT_DIM
        self.encoder = KeypointEncoder(hidden_dim=self.cfg['d_model'],
                                       input_dim=input_dim)

        # LayerNorm applied to encoder output in Phase 1 — part of the
        # frozen interface, so the GAN must see the same normalized space.
        self.pose_norm = nn.LayerNorm(self.cfg['d_model'])

        # Load Phase 1 checkpoint
        if encoder_path and os.path.exists(encoder_path):
            ckpt = torch.load(encoder_path, map_location='cpu')
            self.encoder.load_state_dict(ckpt['encoder'])
            if 'pose_norm' in ckpt:
                self.pose_norm.load_state_dict(ckpt['pose_norm'])
                log("[Phase 2] Loaded pose_norm from Phase 1")
            else:
                log("[Phase 2] WARNING: checkpoint has no pose_norm — "
                    "was Phase 1 trained with the current code?")
            log(f"[Phase 2] Loaded encoder from {encoder_path}")
        else:
            log("[Phase 2] WARNING: No encoder checkpoint, using random weights")

        # Freeze encoder (+ its output norm) and move to device
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.pose_norm.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.pose_norm.eval()
        self.encoder.to(self.device)
        self.pose_norm.to(self.device)

        # Build Prosody GAN
        self.gan = ProsodyGAN(
            d_model=self.cfg['d_model'],
            prosody_dim=2,
            keypoint_dim=self.cfg['keypoint_dim'],
            num_layers=4,
            nhead=self.cfg['nhead'],
            dropout=self.cfg['dropout'],
        ).to(self.device)

        self.distributed = dist.is_initialized()
        self.core = self.gan

        # For a GAN with two optimizers, wrap generator and discriminator in
        # SEPARATE DDP instances. Wrapping the whole ProsodyGAN (old code) is
        # broken twice over: custom methods like generator_loss aren't
        # reachable through a DDP wrapper (AttributeError in multi-GPU runs),
        # and calling them on .module would bypass gradient sync entirely.
        self.gen_ddp = None
        self.disc_ddp = None
        if self.distributed:
            self.gen_ddp = DDP(self.core.generator, device_ids=[local_rank],
                               output_device=local_rank)
            self.disc_ddp = DDP(self.core.discriminator, device_ids=[local_rank],
                                output_device=local_rank)

        # Optimizers
        g_params = list(self.core.generator.parameters())
        d_params = list(self.core.discriminator.parameters())

        self.optimizer_g = AdamW(g_params, lr=self.train_cfg.get('lr_generator', 1e-4), weight_decay=1e-5)
        self.optimizer_d = AdamW(d_params, lr=self.train_cfg.get('lr_discriminator', 4e-4), weight_decay=1e-5)

        # Schedulers
        self.max_epochs = self.train_cfg.get('max_epochs', 100)
        self.scheduler_g = CosineAnnealingLR(self.optimizer_g, T_max=self.max_epochs, eta_min=1e-6)
        self.scheduler_d = CosineAnnealingLR(self.optimizer_d, T_max=self.max_epochs, eta_min=1e-6)

        # Loss weights
        self.lambda_adv = self.train_cfg.get('lambda_adv', 0.1)
        self.lambda_prosody = self.train_cfg.get('lambda_prosody', 5.0)
        self.lambda_recon = self.train_cfg.get('lambda_recon', 1.0)

        # Metrics
        self.best_prosody_loss = float('inf')

        n_gpu = dist.get_world_size() if self.distributed else 1
        log(f"[Phase 2] GPUs: {n_gpu}")
        log(f"[Phase 2] Generator params: {sum(p.numel() for p in self.core.generator.parameters()):,}")
        log(f"[Phase 2] Discriminator params: {sum(p.numel() for p in self.core.discriminator.parameters()):,}")

    def create_datasets(self):
        """Create prosody training datasets (asan preferred, khabar fallback)."""
        split = 0.9
        paths = self.config['paths']

        # --- asan-dataset (per-clip prosody from extract_asan_prosody.py) ---
        asan_cfg = paths.get('asan', {})
        asan_prosody = os.path.expanduser(asan_cfg.get('prosody_root', ''))
        if asan_cfg and asan_prosody and os.path.exists(asan_prosody):
            common = dict(
                root=asan_cfg['root'],
                sources=asan_cfg.get('sources',
                                     ['informburo', 'khabar', 'qazaqstantv']),
                lang=asan_cfg.get('lang', 'kz'),
                tokenizer=None,
                max_frames=self.train_cfg.get('max_seq_len', 1000),
                downsample_every=asan_cfg.get('downsample_every', 1),
                use_enriched=self.use_enriched,
                skip_low_quality=asan_cfg.get('skip_low_quality', True),
                min_hand_cov=asan_cfg.get('min_hand_cov', 0.0),
                load_prosody=True,
                prosody_root=asan_cfg['prosody_root'],
            )
            train_set = AsanDataset(split='train', **common)
            val_set = AsanDataset(split='val', **common)
            collator = AsanCollator()

            if self.distributed:
                sampler = DistributedSampler(train_set, shuffle=True)
                train_loader = DataLoader(
                    train_set, batch_size=self.train_cfg['batch_size'],
                    sampler=sampler, num_workers=2, collate_fn=collator,
                    pin_memory=True, persistent_workers=True)
            else:
                train_loader = DataLoader(
                    train_set, batch_size=self.train_cfg['batch_size'],
                    shuffle=True, num_workers=4, collate_fn=collator,
                    pin_memory=True, persistent_workers=True)
            val_loader = DataLoader(
                val_set, batch_size=self.train_cfg['batch_size'],
                shuffle=False, num_workers=2, collate_fn=collator,
                pin_memory=True)
            log(f"[Phase 2 Datasets] asan: train {len(train_set)}, "
                f"val {len(val_set)}")
            return train_loader, val_loader

        # --- Khabar KZ fallback (legacy path) ---
        prosody_root = paths.get('prosody', {}).get('khabar_kz',
                          os.path.join(paths['data_root'], 'khabar_kz', 'prosody'))

        # Signer-disjoint split inside the dataset (same scheme as Phase 1).
        # The previous code trained on the FULL dataset in distributed mode
        # while validating on its last 10% — i.e. validation data leaked into
        # training — and used clip-level random_split (not signer-disjoint)
        # in single-GPU mode.
        common = dict(
            manifest_path=paths['khabar_kz']['manifest'],
            keypoints_root=paths['khabar_kz']['keypoints'],
            prosody_root=prosody_root,
            max_duration=60.0,
            min_duration=2.0,
            max_frames=self.train_cfg.get('max_seq_len', 1000),
            downsample_every=1,
            name='khabar_kz_prosody',
            split_ratio=split,
        )
        train_set = ProsodyDataset(split='train', **common)
        val_set = ProsodyDataset(split='val', **common)

        collator = ProsodyCollator()

        if self.distributed:
            train_sampler = DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(
                train_set, batch_size=self.train_cfg['batch_size'],
                sampler=train_sampler, num_workers=2, collate_fn=collator,
                pin_memory=True, persistent_workers=True,
            )
        else:
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

        log(f"[Phase 2 Datasets] train {len(train_set)}, val {len(val_set)}")
        log(f"[Phase 2 Datasets] Prosody root: {prosody_root}")
        return train_loader, val_loader

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.encoder.eval()  # frozen
        self.gan.train()

        total_gen_loss = 0
        total_disc_loss = 0
        num_batches = 0
        loss_details = {'adv': 0, 'prosody': 0, 'recon': 0, 'real': 0, 'fake': 0}

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            kps = batch['keypoints'].to(self.device)
            prosody = batch['prosody'].to(self.device)
            input_lens = batch['input_lengths'].to(self.device)

            # PoseTextCollator emits (B, 2, T); GAN losses expect (B, T, 2)
            if prosody.dim() == 3 and prosody.size(1) == 2 and prosody.size(2) != 2:
                prosody = prosody.transpose(1, 2)

            # Recon target is the OFFSET portion only — with enriched
            # (1410-dim) inputs the extra channels are derived features,
            # not something the generator should reproduce.
            kps_target = kps[:, :, :KEYPOINT_DIM]

            # Forward through frozen encoder (+ Phase-1 output norm).
            # input_lengths re-zeroes pad frames before the encoder's
            # temporal conv so batch padding can't bleed into real frames.
            with torch.no_grad():
                encoder_out = self.pose_norm(
                    self.encoder(kps, input_lengths=input_lens))  # (B, T, d_model)

            # ---- Discriminator step (forward through DDP wrapper if any) ----
            self.optimizer_d.zero_grad()
            d_loss, d_details = self.core.discriminator_loss(
                encoder_out, prosody, input_lengths=input_lens,
                disc_module=self.disc_ddp)
            d_loss.backward()
            self.optimizer_d.step()

            # ---- Generator step ----
            self.optimizer_g.zero_grad()
            g_loss, g_details = self.core.generator_loss(
                encoder_out, prosody, kps_target,
                self.lambda_adv, self.lambda_prosody, self.lambda_recon,
                input_lengths=input_lens,
                gen_module=self.gen_ddp,
            )
            g_loss.backward()
            # Discriminator grads from the adversarial term are discarded by
            # optimizer_d.zero_grad() at the next iteration.
            torch.nn.utils.clip_grad_norm_(self.core.generator.parameters(), max_norm=10.0)
            self.optimizer_g.step()

            total_gen_loss += g_details['gen']
            total_disc_loss += d_details['disc']
            for k in loss_details:
                if k in g_details:
                    loss_details[k] += g_details[k]
                if k in d_details:
                    loss_details[k] += d_details[k]
            num_batches += 1

            if is_main() and batch_idx % 500 == 0 and batch_idx > 0:
                avg_gen = total_gen_loss / num_batches
                lr_g = self.optimizer_g.param_groups[0]['lr']
                log(f"  Batch {batch_idx}/{len(train_loader)} | "
                    f"Gen: {avg_gen:.4f} | LR: {lr_g:.6f}")

        avg_gen = total_gen_loss / max(num_batches, 1)
        avg_disc = total_disc_loss / max(num_batches, 1)
        for k in loss_details:
            loss_details[k] /= max(num_batches, 1)

        return avg_gen, avg_disc, loss_details

    @torch.no_grad()
    def validate(self, val_loader):
        """Validate prosody generation."""
        self.encoder.eval()
        self.gan.eval()

        total_prosody_l1 = 0
        total_recon_l1 = 0
        num_batches = 0

        for batch in val_loader:
            if batch is None:
                continue

            kps = batch['keypoints'].to(self.device)
            prosody = batch['prosody'].to(self.device)
            input_lens = batch['input_lengths'].to(self.device)

            if prosody.dim() == 3 and prosody.size(1) == 2 and prosody.size(2) != 2:
                prosody = prosody.transpose(1, 2)
            kps_target = kps[:, :, :KEYPOINT_DIM]

            # Forward through frozen encoder (+ Phase-1 output norm)
            encoder_out = self.pose_norm(
                self.encoder(kps, input_lengths=input_lens))  # (B, T, d_model)

            # Generate prosody
            prosody_gen, recon_kp = self.core.generator(encoder_out, input_lens)

            # Length-masked L1 losses (padded frames excluded)
            prosody_l1 = self.core._masked_l1(prosody_gen, prosody, input_lens).item()
            recon_l1 = self.core._masked_l1(recon_kp, kps_target, input_lens).item()

            total_prosody_l1 += prosody_l1
            total_recon_l1 += recon_l1
            num_batches += 1

        avg_prosody = total_prosody_l1 / max(num_batches, 1)
        avg_recon = total_recon_l1 / max(num_batches, 1)
        return avg_prosody, avg_recon

    def train(self, num_epochs=None, save_dir=None):
        """Main training loop."""
        if num_epochs is None:
            num_epochs = self.max_epochs
        if save_dir is None:
            save_dir = self.config['paths']['output']

        if is_main():
            os.makedirs(save_dir, exist_ok=True)

        train_loader, val_loader = self.create_datasets()

        log(f"\n{'='*60}")
        log(f"Phase 2: Prosody GAN")
        log(f"Epochs: {num_epochs}")
        log(f"{'='*60}\n")

        for epoch in range(num_epochs):
            epoch_start = time.time()

            if self.distributed and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            # Train
            gen_loss, disc_loss, details = self.train_epoch(train_loader, epoch)

            # Validate
            val_prosody, val_recon = self.validate(val_loader)

            # Scheduler step
            self.scheduler_g.step()
            self.scheduler_d.step()

            epoch_time = time.time() - epoch_start
            lr_g = self.optimizer_g.param_groups[0]['lr']
            lr_d = self.optimizer_d.param_groups[0]['lr']

            log(f"\nEpoch {epoch+1}/{num_epochs} | "
                  f"Gen: {gen_loss:.4f} | Disc: {disc_loss:.4f} | "
                  f"Val Prosody: {val_prosody:.4f} | Val Recon: {val_recon:.4f} | "
                  f"LR_G: {lr_g:.6f} | LR_D: {lr_d:.6f} | "
                  f"Time: {epoch_time:.1f}s")

            # Save best
            if is_main() and val_prosody < self.best_prosody_loss:
                self.best_prosody_loss = val_prosody
                gen = self.core.generator
                disc = self.core.discriminator
                torch.save({
                    'generator': gen.state_dict(),
                    'discriminator': disc.state_dict(),
                    'optimizer_g': self.optimizer_g.state_dict(),
                    'optimizer_d': self.optimizer_d.state_dict(),
                    'epoch': epoch,
                    'val_prosody': val_prosody,
                }, os.path.join(save_dir, 'phase2_best.pth'))
                log(f"  Saved best (Prosody L1: {val_prosody:.4f})")

            if is_main() and (epoch + 1) % 10 == 0:
                gen = self.core.generator
                disc = self.core.discriminator
                torch.save({
                    'generator': gen.state_dict(),
                    'discriminator': disc.state_dict(),
                    'epoch': epoch,
                    'val_prosody': val_prosody,
                }, os.path.join(save_dir, f'phase2_epoch{epoch+1}.pth'))

        if self.distributed:
            dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--encoder-ckpt', default=None,
                        help='Path to Phase 1 best checkpoint')
    parser.add_argument('--use-enriched', action='store_true',
                        help='Encoder trained on enriched (1410-dim) features '
                             '(must match Phase 1)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--save-dir', default=None)
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
    elif args.local_rank >= 0:
        dist.init_process_group(backend='nccl')
        local_rank = args.local_rank
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    trainer = ProsodyTrainer(
        config_path=args.config, local_rank=local_rank,
        encoder_path=args.encoder_ckpt,
        use_enriched=args.use_enriched,
    )

    if args.epochs:
        trainer.max_epochs = args.epochs
        trainer.scheduler_g = CosineAnnealingLR(trainer.optimizer_g, T_max=args.epochs, eta_min=1e-6)
        trainer.scheduler_d = CosineAnnealingLR(trainer.optimizer_d, T_max=args.epochs, eta_min=1e-6)

    trainer.train(num_epochs=args.epochs, save_dir=args.save_dir)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
