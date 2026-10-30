# E17: text-adapted mT5 decoder on the rewritten references

Status: **pre-registered 2026-10-21; queued behind E16b (seeds 3–5).** Results
go at the end.

## Why

E16 (rewritten qazaqstantv references, base mT5) is the best model so far:
chrF 35.6 / 34.8 on dev, 35.8 on test. Its errors (NOTES_e16_error_analysis.md):
it gets the story and common words, almost never rare words (2.6%), and 80% of
its content words are not in the reference (precision 0.20). The decoder is the
remaining lever this project controls.

E7 tried this once: mT5 domain-adapted by span corruption on our train texts
(`~/mt5_kaznews`). It never got a fair test: two seeds killed at epoch 5, one
resumed to epoch 10 that did not take off and collapsed (283/600 distinct,
top-1 share 11%), and it trained on the ORIGINAL references.

## Design

1. **Adapt** (`scripts/adapt_mt5_text.py`, E7's exact settings: span
   corruption 15% / mean span 3, 3 epochs, lr 1e-4, batch 24, seed 0) on the
   train texts of `~/asan_canonical_llmref` → `~/mt5_kaznews_llmref`. Dev
   texts are used only for the perplexity readout.
2. **Train** E16's exact config with one change, `--mt5-path
   ~/mt5_kaznews_llmref`: `train/train_min.py --pretrained-unisign
   checkpoints/unisign/csl_stage1_weight.pth --epochs 10`,
   `ASAN_ROOT=~/asan_canonical_llmref`, seeds 0, 1, 2.

**E17 vs E16 isolates the decoder initialisation.** Baseline: E16's six seeds
(0–5), so the takeoff rate is compared against six runs, not three.

## Measures (dev only; test is spent)

Takeoff (unchanged rule, original-ref selection set). For takeoff seeds:
chrF and content recall; content-word precision and rare-word recall
(`scripts/word_band_recall.py`); coupling Δ; T2 dev on both reference
versions.

Let m = mean chrF of E16's takeoff seeds, s = their max − min, and
p = their mean content-word precision.

## Readings (pre-committed)

| Outcome | Reading |
|---|---|
| E17 takeoff mean chrF ≥ m + max(1.5, 2s) AND precision ≥ p − 0.02 | **The adapted decoder raises quality without more invention.** Adopt it. |
| chrF ≥ m + max(1.5, 2s) but precision < p − 0.02 | More fluent, more invented content. Better scores, less trustworthy output; do not adopt for the user-facing translation without a precision fix. |
| chrF within ±max(1.5, 2s) of m | Text-side adaptation does not raise the ceiling on top of clean references. |
| chrF ≤ m − max(1.5, 2s) | Adaptation hurts. |
| 0/3 take off, with distinct outputs < 500 | Adaptation drives prior collapse, as E7's one seed hinted. Close this direction. |
| Takeoff rate | Reported against E16's k/6; 3 seeds cannot show a change unless it is 0/3 or 3/3. |

## Run

`run_e17.sh`, tmux `e17`. Waits for "E16b DONE", adapts on GPU 0 (~1 h),
trains seeds 0 and 1 in parallel, then 2, then scores each seed on both GPUs
(rescore + pose-shuffled, T2 dev original and rewritten, word bands).
Outputs `output/e17_*`.

## Stopped 2026-10-22 13:28 at the user's request (incomplete)

Adaptation done (`~/mt5_kaznews_llmref`, dev span-corruption perplexity
15.53 → 4.59). Seeds 0 and 1 stopped after epoch 8 of 10 (seed 2 never
started). Checkpoints: `output/e17_seed{0,1}/phase1_mt5_epoch5.pth` only, so
resuming means re-running epochs 6–10 (~2.4 h each, in parallel).

Val CE at epoch 8 (original dev refs): s0 2.3764, s1 2.7489. For reference,
E16 takeoff seeds were ~2.15 at epoch 8 and non-takeoffs ~2.70. A hint only —
the takeoff rule needs the epoch-10 selection score — but neither seed was on
E16's takeoff trajectory. No reading applies.

E16 baseline for the readings (six seeds): takeoff 3/6, m = 34.42, s = 2.65,
so the chrF bar is m ± max(1.5, 2s) = ± 5.3; p from the three takeoffs.
