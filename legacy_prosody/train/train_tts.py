"""
Phase 3: Standalone FastSpeech2 TTS trainer (Kazakh text + prosody -> mel).

This trainer is INDEPENDENT of the sign encoder / Phases 1-2. It learns a
text-to-speech acoustic model on paired (text, audio) data, with frame-level
prosody (F0, energy) injected through the variance adaptor. At inference time
the Phase 2 ProsodyGAN supplies the prosody; here we use prosody extracted from
the ground-truth audio (teacher-forced), which is the standard FastSpeech2
training setup.

Why no ground-truth token durations: our corpus has no MFA alignment, so we
train with KNOWN target mel length (FastSpeech2.forward_train upsamples the text
encoding to the mel length by interpolation) and train the duration predictor
against a uniform proxy. Swap in real durations later if you add an aligner.

Vocoder: this model outputs log-mel spectrograms. Convert to waveform with a
separate vocoder (e.g. HiFi-GAN) at inference — see inference/sign2speech.py.

Usage:
  # Single GPU:
  PYTHONPATH=. python train/train_tts.py \
      --config configs/config.yaml \
      --manifest /raid/shared/alikhan_datasets/khabar_kz/khabar_kz.jsonl \
      --epochs 100 --save-dir output

  # Multi-GPU (2 GPUs):
  CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. torchrun --nproc_per_node=2 \
      train/train_tts.py --config configs/config.yaml \
      --manifest /raid/shared/alikhan_datasets/khabar_kz/khabar_kz.jsonl \
      --epochs 100 --save-dir output

Manifest (JSONL) rows must contain at least:
  {"text": "...", "audio_path": "/abs/path.wav"}
Optional cached features speed things up:
  {"mel_path": "...npy", "prosody_path": "...npy"}
"""
import os
import time
import yaml
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import T5Tokenizer, get_cosine_schedule_with_warmup

from data.tts_dataset import TTSDataset, TTSCollator, MEL_CONFIG
from models.fastspeech2 import FastSpeech2, FastSpeech2Loss


MT5_PATH = "google/mt5-base"


# ------------------------------------------------------------------
# Distributed helpers
# ------------------------------------------------------------------
def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------
class TTSTrainer:
    def __init__(self, config_path, manifest, local_rank=0, resume=None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.mcfg = self.config['model']
        # phase3 block holds TTS hyperparameters; fall back to sane defaults.
        self.tcfg = self.config['training'].get('phase3', {})
        self.manifest = manifest

        self.local_rank = local_rank
        self.distributed = dist.is_initialized()
        self.device = torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        )

        # Tokenizer (shared with Phase 1 so token ids are consistent).
        self.tokenizer = T5Tokenizer.from_pretrained(MT5_PATH, legacy=False)

        # Model. vocab must cover MT5's tokenizer range.
        self.model = FastSpeech2(
            vocab_size=self.tokenizer.vocab_size,
            d_model=256,
            n_mel=self.mcfg['n_mel'],
        ).to(self.device)

        if resume and os.path.exists(resume):
            ckpt = torch.load(resume, map_location='cpu')
            self.model.load_state_dict(ckpt['model'])
            log(f"[TTS] Resumed from {resume}")

        if self.distributed:
            self.model = DDP(
                self.model, device_ids=[local_rank],
                output_device=local_rank, find_unused_parameters=True,
            )

        self.loss_fn = FastSpeech2Loss()

        self.lr = float(self.tcfg.get('learning_rate', 1e-4))
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        self.warmup_steps = int(self.tcfg.get('warmup_steps', 1000))
        self.max_epochs = int(self.tcfg.get('max_epochs', 100))
        self.batch_size = int(self.tcfg.get('batch_size', 16))
        self.lambda_dur = float(self.tcfg.get('lambda_dur', 0.1))
        self.scheduler = None
        self.best_loss = float('inf')

        n_params = sum(p.numel() for p in self.model.parameters())
        log(f"[TTS] FastSpeech2 params: {n_params:,} | LR {self.lr} | device {self.device}")

    # --------------------------------------------------------------
    def create_loaders(self, split=0.95):
        # Video-disjoint split (clips from the same source video never span
        # train/val — the old clip-level randperm split inflated val metrics).
        common = dict(
            manifest_path=self.manifest,
            tokenizer=self.tokenizer,
            mel_config=MEL_CONFIG,
            split_ratio=split,
        )
        train_set = TTSDataset(split='train', **common)
        val_set = TTSDataset(split='val', **common)
        total = len(train_set) + len(val_set)
        if total == 0:
            raise RuntimeError(f"No usable samples in manifest: {self.manifest}")

        collator = TTSCollator(pad_token_id=self.tokenizer.pad_token_id)

        if self.distributed:
            sampler = DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(
                train_set, batch_size=self.batch_size, sampler=sampler,
                num_workers=4, collate_fn=collator, pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=self.batch_size, shuffle=True,
                num_workers=4, collate_fn=collator, pin_memory=True,
            )
        val_loader = DataLoader(
            val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=2, collate_fn=collator, pin_memory=True,
        )
        log(f"[TTS] {total} clips -> train {len(train_set)}, val {len(val_set)}")
        return train_loader, val_loader

    # --------------------------------------------------------------
    def _step(self, batch):
        """One forward pass -> (loss, logs). Shared by train/val."""
        text_ids = batch['text_ids'].to(self.device)          # (B, L)
        text_lengths = batch['text_lengths'].to(self.device)  # (B,)
        mel_gt = batch['mel'].to(self.device)                 # (B, T, n_mel)
        mel_lengths = batch['mel_lengths'].to(self.device)    # (B,)
        prosody = batch['prosody'].to(self.device)            # (B, T, 2)

        core = self.model.module if self.distributed else self.model
        # FastSpeech2 variance adaptor wants (B, 2, T).
        mel_pred, pred_dur, dur_tgt = core.forward_train(
            text_ids, prosody.transpose(1, 2), mel_lengths, text_lengths
        )

        # Align predicted/target mel on the time axis (interpolation gives
        # T_max = max(mel_lengths); GT is padded to the same T_max by collator).
        T = min(mel_pred.size(1), mel_gt.size(1))
        mel_pred, mel_gt = mel_pred[:, :T], mel_gt[:, :T]

        # Masked mel L1 (ignore padded frames).
        frame_mask = (torch.arange(T, device=self.device)[None, :]
                      < mel_lengths[:, None]).unsqueeze(-1)     # (B, T, 1)
        mel_l1 = (torch.abs(mel_pred - mel_gt) * frame_mask).sum() / frame_mask.sum().clamp(min=1)

        # Masked duration loss (auxiliary).
        tok_mask = (torch.arange(pred_dur.size(1), device=self.device)[None, :]
                    < text_lengths[:, None])
        dur_l1 = torch.abs(pred_dur[tok_mask] - dur_tgt[tok_mask]).mean()

        loss = mel_l1 + self.lambda_dur * dur_l1
        return loss, {'mel': mel_l1.item(), 'dur': dur_l1.item()}

    def train_epoch(self, loader):
        self.model.train()
        total, n = 0.0, 0
        for batch in loader:
            if batch is None:
                continue
            loss, _ = self._step(batch)
            if not torch.isfinite(loss):
                continue
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            total += loss.item()
            n += 1
        if self.distributed:
            t = torch.tensor([total, n], device=self.device)
            dist.all_reduce(t, dist.ReduceOp.SUM)
            total, n = t[0].item(), t[1].item()
        return total / max(n, 1)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total, n = 0.0, 0
        for batch in loader:
            if batch is None:
                continue
            loss, _ = self._step(batch)
            if not torch.isfinite(loss):
                continue
            total += loss.item()
            n += 1
        return total / max(n, 1)

    # --------------------------------------------------------------
    def train(self, num_epochs=None, save_dir=None):
        num_epochs = num_epochs or self.max_epochs
        save_dir = save_dir or self.config['paths']['output']
        if is_main():
            os.makedirs(save_dir, exist_ok=True)

        train_loader, val_loader = self.create_loaders()
        total_steps = len(train_loader) * num_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

        log(f"\n{'='*60}\nPhase 3: FastSpeech2 TTS (standalone)\nEpochs: {num_epochs}\n{'='*60}\n")

        for epoch in range(num_epochs):
            t0 = time.time()
            if self.distributed and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            lr = self.optimizer.param_groups[0]['lr']
            log(f"Epoch {epoch+1}/{num_epochs} | Train {train_loss:.4f} | "
                f"Val {val_loss:.4f} | LR {lr:.6f} | {time.time()-t0:.1f}s")

            if is_main() and val_loss < self.best_loss:
                self.best_loss = val_loss
                core = self.model.module if self.distributed else self.model
                torch.save({'model': core.state_dict(), 'epoch': epoch,
                            'val_loss': val_loss, 'mel_config': MEL_CONFIG},
                           os.path.join(save_dir, 'tts_fastspeech2_best.pth'))
                log(f"  Saved best (Val {val_loss:.4f})")

            if is_main() and (epoch + 1) % 20 == 0:
                core = self.model.module if self.distributed else self.model
                torch.save({'model': core.state_dict(), 'epoch': epoch},
                           os.path.join(save_dir, f'tts_fastspeech2_epoch{epoch+1}.pth'))

        if self.distributed:
            dist.barrier()


# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--manifest', required=True,
                        help='JSONL with text + audio_path (+ optional mel/prosody paths)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--save-dir', default=None)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

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

    trainer = TTSTrainer(
        config_path=args.config, manifest=args.manifest,
        local_rank=local_rank, resume=args.resume,
    )
    trainer.train(num_epochs=args.epochs, save_dir=args.save_dir)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
