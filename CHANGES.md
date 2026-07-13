# Code review changes — KRSL → Kazakh Speech

# Review pass 3 (2026-07-11): paper-guided improvements

References: CSLRConformer (arXiv:2508.01791), SignBERT+ (via SberDevices
survey habr.com/792660), Uni-Sign (arXiv:2501.15187), S2PFormer
(arXiv:2604.10413 — already the phase-2 basis), STRGCN (arXiv:2505.04167 —
left as a future ablation; no cheap portable piece).

1. **Signer-scale normalization** (CSLRConformer). Offsets were translation-
   invariant but NOT scale-invariant: the same sign closer to the camera
   produced proportionally larger features. All loaders now divide
   coordinates by the per-clip median shoulder width before offset
   conversion (`normalize_signer_scale`).
2. **Detector-spike removal** (CSLRConformer's outlier filtering, simplified
   from DBSCAN to a physics test). A joint > 1.5 shoulder-widths away from
   BOTH temporal neighbours is a glitch → replaced by the neighbour
   midpoint (`remove_keypoint_spikes`). Both steps run in
   `preprocess_keypoints()` in every loader.
3. **Multi-granularity masked-pose** (SignBERT+). The aux-loss mask now
   splits its budget: 50% whole frames, 25% a contiguous span, 25%
   individual joints — matching the three keypoint failure modes (bad pose,
   bad transition, bad landmark). Was frame-only.
4. **Score-aware validity** (Uni-Sign). The asan loader feeds continuous
   per-joint confidence scores into the enriched validity channel
   (`assemble_joint_scores` + `enrich_keypoints(joint_scores=...)`) instead
   of collapsing them to a 0/1 bit. npz-based loaders keep binary validity
   (their npz stores only wholebody scores; extend later if needed).

⚠️ Items 1-2 change the feature definition again — retrain from the
Uni-Sign weights (already required by the 1410-dim change).

# Follow-up (2026-07-11): dual-coords + alignment audit

1. **Dual-coord enriched features (KZ-RU SignFormer item 2, completed).**
   Enriched vector is now 5×282 = **1410**: [offset, absolute, velocity,
   acceleration, validity]. Absolute coordinates are per-clip standardized
   (zero mean/unit std over detected joints) so pixel scale doesn't dominate.
   The encoder projection widens to 5 input channels
   (x_off, y_off, x_abs, y_abs, score); pretrained 3-ch Uni-Sign weights load
   into the matching columns with the absolute columns zero-initialized, so
   the layer computes exactly the pretrained function at init. D=1128
   checkpoints/configs still load (legacy path kept in the mapper).
   ⚠️ New `--use-enriched` runs use 1410 dims — encoders trained at 1128
   are incompatible (retrain, which was already required by pass 2).
2. **Transcript-alignment audit (item 5).** `scripts/audit_alignment.py`
   flags clips whose chars/sec falls outside 5-30 (Kazakh speech ≈ 12-18).
   Supports asan annotations and jsonl manifests, CSV export. Note: the
   legacy InformburoDataset assigns the full-video transcript to every
   segment — inherently misaligned; prefer asan's per-clip annotations.

# Review pass 2 (2026-07-10)

## Data / features (data/utils.py)

1. **Hand skeleton in `to_offset_keypoints` was wrong.** The parent loop attached
   finger bases to the previous finger's tip (`parents[5]=4`, overwriting the
   correct `5→0` wrist edge) and left 8 of 21 nodes (all 3rd/4th finger joints)
   with all-zero offsets. Fixed to the real COCO hand topology: wrist root,
   5 chains of 4.
2. **Body skeleton parents fixed** to anatomical edges matching BODY_IDX
   (eyes←nose, ears←eyes, shoulders←nose, elbows←shoulders, hips←shoulders).
   Previous map connected l_shoulder←l_eye and hips←elbows.
3. **pyin `frame_length=256` could not track F0 below ~125 Hz** at 16 kHz
   (fmin is C2 ≈ 65 Hz). Now 1024. F0 is standardized over voiced frames
   (per-utterance max-normalization erased pitch-range information).
4. Added `resample_prosody()` — see alignment fix below.

⚠️ Items 1–2 change the offset feature definition: encoders trained on the old
features should be retrained (or expect a feature-distribution shift).

## Prosody alignment (khabar/kazsign/prosody datasets)

**Prosody is 100 Hz; keypoints are 25–50 fps. Aligning by min-length truncation
kept only the first ~half of every clip's prosody** (and silently dropped
keypoint frames). All three datasets now linearly resample prosody to the
keypoint frame count. ProsodyDataset also now SKIPS samples without prosody
instead of training the GAN toward all-zero targets, and supports the same
signer-disjoint `split='train'/'val'` as the other datasets.

## Encoder (models/unisign_encoder.py)

1. **Body node layout now matches the Uni-Sign body graph** (edges imply:
   0 neck, 1/2 hips, 3/4 shoulders, 5/6 elbows, 7/8 wrists). The old mapper fed
   hips into wrist slots and fabricated "hips" as `shoulder − 80 px` — a pixel
   constant applied to *offset* features, i.e. pure bias. Wrist nodes repeat
   elbow features (no wrist in BODY_IDX); hand groups carry true wrist motion.
2. Hand fusion now adds the body **wrist** node (7/8) instead of node 1/2
   (which are hips under the graph layout).
3. Face `idx_map` had a duplicate landmark (39 twice); second one → 54
   (right mouth corner).
4. Neck validity score normalized ((v5+v6)/2 — was v5+v6, up to 2.0).

⚠️ Item 1 changes what pretrained body-branch weights see; hands (the fully
compatible group) are unaffected.

## Prosody GAN (models/prosody_gan.py, train/train_prosody.py)

1. **Hinge loss was `relu(1 − mean(scores))`** — collapses once the batch
   average clears the margin. Now the standard `mean(relu(1 ∓ score))`.
2. **Prosody/recon L1 now masked by `input_lengths`** (padding was trained on).
3. Generator's global mean-pool now excludes padded frames; transformer gets
   key-padding masks; positional table no longer crashes for T > 1000.
4. **Phase-2 trainer had a train/val leak in DDP mode** (trained on the full
   dataset, validated on its last 10%) and clip-level random_split otherwise.
   Now signer-disjoint train/val via the dataset split.
5. **DDP was broken**: custom methods (`generator_loss`, …) aren't reachable
   through a DDP wrapper (AttributeError), and `self.gan.generator` at log time
   crashed too. G and D are now wrapped in separate DDP instances and losses
   route forwards through them (real+fake in one D forward, as DDP requires).

## Phase 1 (train/train_encoder_mt5.py)

1. **Masked-pose aux loss never trained the encoder.** It reconstructed from
   *clean* embeddings computed under `torch.no_grad()`, so only the tiny MLP
   head learned. Reconstruction now runs from the masked forward pass inside
   the model (single encoder pass, gradients flow, DDP-synced).
2. **Attention-mask corruption during masked-pose training.** Frame lengths
   were re-derived by counting non-zero frames, so every masked frame cut one
   *trailing* frame out of attention. Collator `input_lengths` are now passed
   through to the model.
3. **LoRA was applied after DDP wrapping** — swapped modules escape DDP's
   bucket registration, silently breaking gradient sync. LoRA now applies
   before the wrap.
4. **Checkpoints now include the MT5 decoder** (full state, or adapter-only
   when using LoRA) + the masked-pose head. Previously only the encoder was
   saved: the fine-tuned decoder was lost, and inference silently ran with
   base pretrained MT5.
5. WER eval capped to the first 25 val batches (beam search over the full val
   set dominated epoch time; subset is fixed so numbers stay comparable).
6. Signer-split shuffles use a local `random.Random(seed)` (global reseeding
   leaked into dataloader workers).

## TTS (data/tts_dataset.py, models/fastspeech2.py)

1. **Critical: mel spectrograms were transposed.** `librosa.melspectrogram`
   returns `(n_mel, T)` but the whole pipeline assumed `(T, n_mel)` — "frame"
   truncation sliced mel bins and every sample appeared to be exactly 80
   frames long. `wav_to_mel` now returns `(T, n_mel)`; cached `.npy` mels are
   orientation-checked.
2. Length regulator no longer drops a sample from the batch when all its
   durations round to 0 (batch desync); FastSpeech2Loss can mask padded mel
   frames via `mel_lengths`.

## Inference (inference/sign2speech.py)

1. **`generate(kps, lengths)` passed the lengths tensor as `max_new_tokens`.**
   Now a keyword argument.
2. Loads fine-tuned MT5 / LoRA weights from the phase-1 checkpoint (warns if
   absent).
3. `run_batch` globbed the entire keypoints tree per manifest line and used
   the same first file for every clip missing `keypoints_path`; now a
   stem-indexed lookup, skipping clips without keypoints.

## Cleanup

- Three copy-pasted collators (Khabar/Informburo/KazSign) unified into
  `data/collators.py::PoseTextCollator` (old names kept as aliases).
- `split` argument validated ('train'/'val') in all datasets.
- Khabar/Prosody `_find_keypoint_file` no longer grabs an arbitrary `.npz`
  from the video dir (could pair the wrong segment with a clip's text); it
  requires an exact clip match or a single candidate.

---

# Review pass 1

Applied by review pass on the current pipeline (Uni-Sign encoder + MT5 → ProsodyGAN → FastSpeech2).
All files compile; FastSpeech2 and the Phase-1 masking logic pass functional smoke tests.

## Phase 1 — train/train_encoder_mt5.py (critical fixes)

The last optimization pass ("speed up, drop the freeze") introduced several bugs.
All are fixed:

1. **Prefix caching crashed on step 2.** Prefix embeddings were cached as autograd
   tensors and reused, so the second `backward()` hit "backward through the graph a
   second time". Also the cached ids were never moved to GPU. Now we cache the prefix
   *token ids* (a `register_buffer`, so it follows `.to(device)`/DDP) and re-embed each
   forward — an embedding lookup on ~10 tokens is free and grad-correct.

2. **Label mask concatenated onto the encoder attention mask.** `attention_mask` was
   `[prefix, pose, labels]` (length P+T+L) for inputs of length P+T. In T5/MT5 the
   encoder mask covers encoder inputs only; the decoder mask is derived from `labels`.
   Fixed to `[prefix, pose]`.

3. **Padded pose frames were attended.** `pose_mask` was all ones. With 2–60 s clips
   this fed a lot of zero-padding into MT5. Now the pose mask is built from
   `input_lengths` and padded embeddings are zeroed.

4. **DDP AttributeError.** Optimizer used `self.model.mt5` after DDP-wrapping. Fixed to
   unwrap `.module` (also in the param-count logs).

5. **Train/val leak under DDP.** `DistributedSampler(dataset)` sampled the whole set
   including val clips. Now an explicit, seed-fixed train/val split is made first and
   only the train subset is sampled. (Split is still clip-level random — see Known
   limitations.)

6. **WER diagnostic.** Validation now prints a few (ref, hyp) pairs, so a WER pinned at
   1.0 while CE falls is immediately explainable (empty generations vs metric bug).

7. Collator no longer emits an unused `label_attn_mask`; `max_text_tokens` now reads
   from config (`max_text_len`) instead of a hardcoded 128.

## Phase 3 — FastSpeech2 (models/fastspeech2.py) + trainer (train/train_tts.py)

- **Decoder used the wrong module.** `Decoder` wrapped `nn.TransformerDecoder`
  (cross-attention, requires a `memory` arg) and called it with one argument — a
  guaranteed crash. FastSpeech2's mel decoder is non-autoregressive self-attention;
  switched to `nn.TransformerEncoder`.
- **No token durations exist in the data (no MFA).** Added `forward_train`, which
  decodes to the *known* target mel length by interpolation-upsampling the
  prosody-adapted text encoding, so predicted/GT mel shapes always match. The duration
  predictor is still trained on a uniform proxy so inference can run without a known
  length.
- **train_tts.py fully rewritten** as a standalone FastSpeech2 trainer (was a stale
  end-to-end script importing the superseded encoder/CTC decoder). Supports single/
  multi-GPU, masked mel L1 + auxiliary duration loss, cosine warmup, checkpointing.

## Inference — inference/sign2speech.py (rewritten)

Was built entirely on the old architecture (old encoder, GlossDecoderCTC, [CLS] slicing,
SentencePiece, 3-dim prosody). Rewritten to load per-phase checkpoints and run
encoder+MT5 → ProsodyGAN(F0,energy) → FastSpeech2 → (HiFi-GAN). Prosody GAN is built
with `d_model=768` to match the encoder output. Degrades gracefully: text-only if
Phase 2/3 or a vocoder aren't provided.

## Config (configs/config.yaml)

- Marked dead CTR-GCN keys (`nhead`, `num_encoder_layers`, `dim_feedforward`) as UNUSED.
- `prosody_dim` 3 → 2 (GAN outputs F0+energy; duration isn't produced by the GAN).
- `warmup_steps` 4000 → 1500 (was ~20% of training; now proportionate).
- `max_text_len` 500 → 128 (single source of truth with the collator).
- phase3 block retargeted from end-to-end to standalone TTS (lr, warmup, lambda_dur).

## Deprecations

`train/train_encoder.py`, `train_encoder_ce.py`, `train_encoder_finetune.py` got
DEPRECATED banners — the old CTR-GCN/CTC path, superseded by `train_encoder_mt5.py`.

## Known limitations (design decisions for you)

Status after review pass 2 follow-up:

- ~~**Val split is clip-level random.**~~ **RESOLVED.** All splits are now
  video-id-disjoint, including TTS training: `TTSDataset` supports
  `split='train'/'val'` (grouped by `video_id` / clip-id prefix / audio-file
  stem) and `train_tts.py` uses it instead of clip-level `randperm`.
- **TTS from scratch vs pretrained KazakhTTS2.** OPEN — swapping to ISSAI's
  KazakhTTS2 means an ESPnet dependency, its own tokenizer, and downloading
  checkpoints; kept as a project decision. Standalone FastSpeech2 remains the
  implementation.
- ~~**Prosody rate vs mel rate.**~~ **RESOLVED.** `sign2speech.py` now
  resamples GAN prosody from the video frame rate (`--fps`, default 50) to the
  mel frame rate (`sr/hop ≈ 86 fps`) before FastSpeech2.
- ~~**HiFi-GAN vocoder** is not included.~~ **RESOLVED (loading path).** The
  pipeline loads a HiFi-GAN generator from `--vocoder` or config
  `inference.vocoder_path` (TorchScript or pickled `nn.Module`; weight-norm
  removed automatically) and writes waveforms at the mel sample rate. The
  checkpoint itself must still be trained/downloaded, and it must expect the
  pipeline's log10-mel features.
