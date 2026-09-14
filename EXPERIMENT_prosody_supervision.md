> **CLOSED 2026-08-25 — negative result. Kept as the record of why.**
>
> Both arms ran 7/10 epochs from an identical checkpoint. The winner flipped
> every epoch, metrics disagreed within the same epoch, and the between-arm
> gaps were the same size as the within-arm noise floor (~0.002 WER /
> ~0.17 BLEU, measured from accidentally duplicated epoch-3 runs). `ProsAux`
> never fell below ~0.82 against a trivial baseline of ~1.0, so the proposed
> mechanism barely engaged. Prosody, and sign-to-speech generally, are no
> longer part of this project; it is keypoints -> Kazakh text only.
>
> Do not restart this without new evidence. The durable findings from it are
> the noise floor above, and that val CE is anti-correlated with generation
> quality here (fixed in `899cf39`).

# Ablation: Prosody as Encoder Supervision for Low-Resource SLT

## Hypothesis

Predicting speech prosody (F0, energy) from sign-encoder embeddings as an
**auxiliary supervision** signal reduces representation collapse and
improves translation quality on a low-resource sign language.

**Mechanism.** `scripts/diagnose_phase1.py` established that our encoder's
embeddings collapse to ~0.98 pairwise cosine across genuinely different
clips — worst in the hand groups (left 0.994, right 0.982) — i.e. the
encoder is not clip-discriminative, which is why generations are fluent
but content-free. Speech prosody contours *do* vary substantially
clip-to-clip. To predict a specific clip's F0/energy contour, the encoder
must encode the temporal sign dynamics it currently discards. The
auxiliary loss therefore applies direct pressure against collapse.

## Positioning (why this is novel)

All prior sign-to-speech work treats prosody as an **output**:
- SignRecGAN / S2PFormer (arXiv:2604.10413) — generates prosody for TTS
- Frontiers 2026 (Kazakh) — predicts prosody to control FastSpeech2
- Our own `models/prosody_gan.py` — generates prosody

We instead use prosody as **input-side supervision on the encoder**, and
never synthesize the prediction. To our knowledge this direction is
unclaimed.

Note the honest framing: this is a **method paper with a controlled
ablation**, not a SOTA claim. Absolute numbers on KRSL are far below
high-resource benchmarks (cf. DeepMind SL2T's 70 BLEURT on FLEURS-ASL);
the contribution is that the technique *helps*, demonstrated under
matched conditions on a language with no existing continuous-SLT
baseline.

## Design

Two arms, identical in **every** respect except the auxiliary loss.

| | Baseline (A) | Treatment (B) |
|---|---|---|
| `--prosody-aux-weight` | `0` | `>0` (sweep) |
| Prosody head built | no | yes |
| Prosody data loaded | **no** | yes |
| Init checkpoint | `ours_enriched_friend_mt5.pth` | same |
| Encoder / decoder / data / seed / epochs / LR schedule | — | identical |

Controls that matter:
- **Same init.** Both resume from the same checkpoint (our enriched
  pose-pretrained encoder + the fine-tuned mT5), so we isolate the effect
  of the aux loss rather than of a different starting point.
- **Head capacity matched.** `build_prosody_aux_head` is deliberately the
  same shape as the existing `build_masked_pose_decoder`, so any effect
  comes from the *signal*, not from added parameters.
- **Baseline never loads prosody**, so the data pipeline is otherwise
  identical and can't introduce a confound.

## Metrics

**Primary (translation quality)** — already implemented in
`utils/metrics.py`, reported every epoch:
WER · BLEU · ROUGE-1/2/L · BERTScore (multilingual)

**Mechanism (the scientific claim)** — this is what makes it a method
paper rather than number-chasing:
- pairwise cross-clip cosine collapse metric, from
  `scripts/diagnose_phase1.py` section (B)
- per-group collapse (body / left / right / face), section (B3) — the
  hypothesis specifically predicts the *hand* groups should improve most,
  since that's where collapse is worst and where motion dynamics live

**Secondary (sanity)** — does the aux task itself learn? Logged per batch
as `ProsAux`. If this doesn't fall, the encoder isn't extracting
prosody-predictive structure and a null result is uninformative rather
than evidence against the hypothesis.

## Predicted outcomes

| Result | Interpretation |
|---|---|
| Collapse ↓ **and** BLEU/WER ↑ | Hypothesis supported — publishable claim |
| Collapse ↓ but quality flat | Mechanism works, doesn't reach the decoder — still a finding, weaker |
| ProsAux flat | Aux task unlearnable; null result is uninformative — debug first |
| Collapse ↑ or quality ↓ | Hypothesis refuted — report honestly |

Pre-committing to these readings guards against post-hoc rationalization.

## Runs

```bash
# Arm A — baseline
CUDA_VISIBLE_DEVICES=1 ... train/train_encoder_mt5.py \
  --resume output/ours_enriched_friend_mt5.pth --use-enriched \
  --epochs 10 --save-dir output/abl_baseline

# Arm B — treatment (weight to be swept: 0.1 / 0.5 / 1.0)
CUDA_VISIBLE_DEVICES=1 ... train/train_encoder_mt5.py \
  --resume output/ours_enriched_friend_mt5.pth --use-enriched \
  --prosody-aux-weight 0.5 --prosody-root data/asan_prosody_v3 \
  --epochs 10 --save-dir output/abl_prosody_w0.5
```

Environment: `ASAN_ROOT=/data/archive/asan-dataset` on the new box.

## Known limitations to state in the paper

- Single language (KRSL) — no claim of generality across sign languages.
- Single seed per arm initially; if the effect is small, seed variance
  must be quantified before claiming it.
- Prosody comes from the *same* clips' audio, so this is not applicable
  to sign data lacking aligned speech.
