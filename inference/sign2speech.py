"""
End-to-end inference: KRSL keypoints -> Kazakh text -> prosody -> mel -> audio.

Pipeline (matches the current architecture):
  keypoints (B,T,282)
    -> Uni-Sign encoder + MT5  (Phase 1)     -> Kazakh text
    -> Uni-Sign encoder + ProsodyGAN (Phase 2) -> [F0, energy] per frame
    -> FastSpeech2 (Phase 3)                  -> log-mel spectrogram
    -> HiFi-GAN vocoder                       -> waveform

Checkpoints are loaded per-phase (they are trained separately), not from a
single fused checkpoint:
  --phase1  output/phase1_mt5_best.pth          (encoder state_dict)
  --phase2  output/phase2_prosody_best.pth      (generator/discriminator)
  --phase3  output/tts_fastspeech2_best.pth     (FastSpeech2 state_dict)
"""
import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from tqdm import tqdm

from models.unisign_encoder import KeypointEncoder
from models.prosody_gan import ProsodyGAN
from models.fastspeech2 import FastSpeech2
from data.utils import load_npz_keypoints
from data.tts_dataset import MEL_CONFIG
from train.train_encoder_mt5 import UniSignMT5


class SignToSpeechPipeline:
    def __init__(self, config_path, phase1, phase2=None, phase3=None,
                 device='cuda', fps=50.0, vocoder=None):
        """
        Args:
            fps: video frame rate of the input keypoints — used to resample
                sign-frame prosody to the mel frame rate.
            vocoder: path to a HiFi-GAN generator checkpoint (TorchScript or
                a pickled nn.Module). Falls back to config
                inference.vocoder_path if not given.
        """
        import yaml
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = device if torch.cuda.is_available() else 'cpu'
        self.fps = fps
        d_model = self.config['model']['d_model']
        n_mel = self.config['model']['n_mel']

        # ---- Phase 1: encoder + MT5 ----
        encoder = KeypointEncoder(hidden_dim=d_model)
        self.model = UniSignMT5(encoder=encoder, lang="Kazakh").to(self.device)
        ckpt1 = torch.load(phase1, map_location='cpu')
        enc_state = ckpt1.get('encoder', ckpt1)
        self.model.encoder.load_state_dict(enc_state)
        if isinstance(ckpt1, dict) and 'pose_norm' in ckpt1:
            self.model.pose_norm.load_state_dict(ckpt1['pose_norm'])
        # Load the fine-tuned MT5 decoder if the checkpoint carries it.
        # (Without it, generation runs with the raw pretrained MT5 that was
        # never trained on pose embeddings.)
        if isinstance(ckpt1, dict) and 'mt5' in ckpt1:
            self.model.mt5.load_state_dict(ckpt1['mt5'])
            print("[Pipeline] Loaded fine-tuned MT5 weights")
        elif isinstance(ckpt1, dict) and 'mt5_lora' in ckpt1:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32,
                target_modules=["q", "v"], lora_dropout=0.1, bias="none",
            )
            self.model.mt5 = get_peft_model(self.model.mt5, lora_config).to(self.device)
            missing = self.model.mt5.load_state_dict(ckpt1['mt5_lora'], strict=False)
            print("[Pipeline] Loaded MT5 LoRA adapters")
        else:
            print("[Pipeline] WARNING: checkpoint has no MT5 weights — "
                  "generation uses the base pretrained MT5")
        self.model.eval()
        self.tokenizer = self.model.mt5_tokenizer
        print(f"[Pipeline] Phase 1 loaded from {phase1}")

        # ---- Phase 2: prosody GAN (optional) ----
        self.prosody_gan = None
        if phase2 and os.path.exists(phase2):
            # d_model MUST match the encoder output (768), not the GAN default.
            self.prosody_gan = ProsodyGAN(d_model=d_model, prosody_dim=2).to(self.device)
            ckpt2 = torch.load(phase2, map_location='cpu')
            self.prosody_gan.generator.load_state_dict(ckpt2['generator'])
            self.prosody_gan.eval()
            print(f"[Pipeline] Phase 2 loaded from {phase2}")

        # ---- Phase 3: FastSpeech2 (optional) ----
        self.tts = None
        if phase3 and os.path.exists(phase3):
            self.tts = FastSpeech2(
                vocab_size=self.tokenizer.vocab_size, d_model=256, n_mel=n_mel
            ).to(self.device)
            ckpt3 = torch.load(phase3, map_location='cpu')
            self.tts.load_state_dict(ckpt3['model'])
            self.tts.eval()
            print(f"[Pipeline] Phase 3 loaded from {phase3}")

        # ---- Vocoder (optional) ----
        vocoder_path = vocoder or self.config.get('inference', {}).get('vocoder_path')
        self.vocoder = self._load_vocoder(vocoder_path) if vocoder_path else None
        if self.vocoder is None:
            print("[Pipeline] No vocoder — pipeline stops at mel spectrograms")

    def _load_vocoder(self, path):
        """
        Load a HiFi-GAN generator. Supports TorchScript exports and pickled
        nn.Module checkpoints. The generator must map (B, n_mel, T) log-mel
        → (B, 1, T_wav) or (B, T_wav) waveform at MEL_CONFIG['sr'].
        """
        if not os.path.exists(path):
            print(f"[Pipeline] WARNING: vocoder not found at {path}")
            return None
        try:
            voc = torch.jit.load(path, map_location=self.device)
        except Exception:
            try:
                voc = torch.load(path, map_location=self.device)
            except Exception as e:
                print(f"[Pipeline] WARNING: failed to load vocoder: {e}")
                return None
            if not isinstance(voc, torch.nn.Module):
                print("[Pipeline] WARNING: vocoder checkpoint is not an "
                      "nn.Module or TorchScript module — export the HiFi-GAN "
                      "generator with torch.jit.script/save and retry")
                return None
        voc.to(self.device).eval()
        if hasattr(voc, 'remove_weight_norm'):
            try:
                voc.remove_weight_norm()
            except Exception:
                pass
        print(f"[Pipeline] Vocoder loaded from {path}")
        return voc

    @torch.no_grad()
    def _vocode(self, mel):
        """mel: (1, T, n_mel) → waveform np.float32 at MEL_CONFIG['sr']."""
        wav = self.vocoder(mel.transpose(1, 2).to(self.device))  # (1, n_mel, T) in
        wav = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        return wav

    @torch.no_grad()
    def run(self, keypoints_path, frame_start=None, frame_end=None):
        """Returns (audio or None, text). Audio is None until a vocoder is set."""
        kps, scores, _ = load_npz_keypoints(keypoints_path, frame_start, frame_end)
        if kps is None:
            raise ValueError(f"Failed to load keypoints from {keypoints_path}")

        kps_t = torch.from_numpy(kps).unsqueeze(0).to(self.device)  # (1, T, 282)
        T = kps_t.size(1)
        lengths = torch.tensor([T], device=self.device)

        # 1. Text (Phase 1). NOTE: input_lengths must be a keyword arg —
        # passing it positionally (old code) fed the lengths tensor into
        # max_new_tokens.
        text = self.model.generate(kps_t, input_lengths=lengths)[0]

        # 2. Prosody (Phase 2)
        prosody = None
        if self.prosody_gan is not None:
            pose_emb = self.model.pose_norm(
                self.model.encoder(kps_t, input_lengths=lengths))  # (1, T, 768)
            prosody, _ = self.prosody_gan.generator(pose_emb, lengths)  # (1, T, 2)

        # 3. Mel (Phase 3)
        mel = None
        if self.tts is not None and prosody is not None:
            # Resample prosody from the sign frame rate (video fps) to the
            # mel frame rate so F0/energy align with the acoustic timeline
            # the variance adaptor was trained on.
            mel_rate = MEL_CONFIG['sr'] / MEL_CONFIG['hop_length']  # ≈86 fps
            t_mel = max(int(round(T / self.fps * mel_rate)), 1)
            prosody_mel = F.interpolate(
                prosody.transpose(1, 2), size=t_mel, mode='linear',
                align_corners=False,
            )  # (1, 2, T_mel)

            text_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            mel = self.tts.forward_inference(text_ids, prosody_mel)  # (1, T', n_mel)

        # 4. Audio (vocoder)
        audio = None
        if mel is not None and self.vocoder is not None:
            audio = self._vocode(mel)

        return audio, text

    @torch.no_grad()
    def run_batch(self, manifest_path, keypoints_root, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        results = []
        # Index keypoint files ONCE by stem (the old code re-globbed the whole
        # tree for every manifest line — and then used the same first file for
        # every clip that lacked keypoints_path).
        kp_index = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(keypoints_root, '**', '*.npz'),
                               recursive=True)
        }
        with open(manifest_path) as f:
            for line in tqdm(f, desc="Processing clips"):
                entry = json.loads(line)
                clip_id = entry.get('clip_id', entry.get('id', 'clip'))
                kp = entry.get('keypoints_path') or kp_index.get(clip_id)
                if not kp or not os.path.exists(kp):
                    print(f"[warn] No keypoints for {clip_id}, skipping")
                    continue
                try:
                    audio, text = self.run(
                        kp, entry.get('frame_start'), entry.get('frame_end'))
                    rec = {'clip_id': clip_id, 'text': text}
                    if audio is not None:
                        out_path = os.path.join(output_dir, f'{clip_id}.wav')
                        sf.write(out_path, audio, MEL_CONFIG['sr'])
                        rec['audio_path'] = out_path
                    results.append(rec)
                except Exception as e:
                    print(f"[warn] Failed {clip_id}: {e}")
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Processed {len(results)} clips -> {output_dir}")
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--phase1', required=True, help='encoder checkpoint (phase1_mt5_best.pth)')
    parser.add_argument('--phase2', default=None, help='prosody GAN checkpoint')
    parser.add_argument('--phase3', default=None, help='FastSpeech2 checkpoint')
    parser.add_argument('--keypoints', required=True, help='.npz file or .jsonl manifest')
    parser.add_argument('--output', default='output_audio')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fps', type=float, default=50.0,
                        help='video frame rate of the keypoints (for prosody resampling)')
    parser.add_argument('--vocoder', default=None,
                        help='HiFi-GAN generator checkpoint (TorchScript or nn.Module); '
                             'falls back to config inference.vocoder_path')
    args = parser.parse_args()

    pipe = SignToSpeechPipeline(
        args.config, args.phase1, args.phase2, args.phase3, args.device,
        fps=args.fps, vocoder=args.vocoder)

    if args.keypoints.endswith('.jsonl'):
        kp_root = os.path.dirname(args.keypoints)
        pipe.run_batch(args.keypoints, kp_root, args.output)
    else:
        audio, text = pipe.run(args.keypoints)
        os.makedirs(args.output, exist_ok=True)
        print(f"Text: {text}")
        if audio is not None:
            out_path = os.path.join(args.output, 'output.wav')
            sf.write(out_path, audio, MEL_CONFIG['sr'])
            print(f"Audio saved to {out_path}")
        else:
            print("(No audio: provide --vocoder plus --phase2/--phase3.)")


if __name__ == '__main__':
    main()
