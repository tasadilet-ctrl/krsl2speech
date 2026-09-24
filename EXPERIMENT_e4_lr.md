# E4: learning-rate balance screen

Status: **pre-registered, launched 2026-09-23.** Results go in a section at the
end; nothing above it is edited after launch.

## Why

Nine runs across E3/E5/E6 show one pattern: training is bimodal. A run either
"takes off" (decoder starts using the pose input; chrF ~29.3–29.8, content recall
~0.12) or it doesn't (chrF 19.8–22.1, recall 0.025–0.040). Nothing has moved
either mode, only which one a run lands in:

| arm | takeoff |
|---|---|
| B, no alignment | 1/3 |
| E5, sequence-level alignment | 0/3 |
| E6, token-level alignment | 1/3 |

Both alignment arms drove their objective far below chance (E5 ~1.9, E6 1.18–1.56
against 5.58) without changing the takeoff rate. The encoder can be made
informative; the decoder still doesn't read it.

**The untested variable is the learning-rate ratio.** mT5 (582M params) trains at
5e-4 while the encoder (5.3M) trains at 5e-5 — a 10× gap that has been fixed since
before any of these experiments, and which the 2026-09-17 audit flagged as a
hypothesis never screened. A decoder that adapts ten times faster than its encoder
is a plausible mechanism for "the decoder fits the language prior before the
encoder is worth reading", which is exactly the failure alignment left untouched.

## Design

Arm-B config (upstream Uni-Sign preprocessing, `--block-padding-mask`, 10 epochs,
canonical root, frozen 600-clip selection manifest, **no alignment**), with the
ratio inverted:

| | encoder LR | mT5 LR | ratio mt5/enc |
|---|---|---|---|
| baseline (B seeds 0/1/2) | 5e-5 | 5e-4 | 10 |
| **E4** | **1e-4** | **5e-5** | **0.5** |

This is the audit's own first screen ("compare current rates with mT5 5e-5 and
encoder 1e-4"). Three seeds, identical otherwise. `--mt5-lr` was added for this;
it sets the decoder group only, so `pose_norm` and any aux heads stay at base_lr
and the screen changes exactly one thing. Verified at launch:
`lrs ['1.00e-04', '5.00e-05', '5.00e-04']` for encoder / mT5 / pose_norm.

**Judged by takeoff rate over three seeds**, against the baseline's 1/3 — not by
one run's chrF, which E3 showed to be uninformative (one config's three seeds
spanned 19.8–29.8 chrF).

## Readings (pre-committed)

| Outcome | Reading |
|---|---|
| 3/3 take off | The LR ratio was the bimodality. Adopt it, re-screen preprocessing and alignment on top, and treat every earlier arm comparison as run under a broken optimizer setting. |
| 2/3 | Promising; the ratio matters but does not fully determine takeoff. Confirm with three more seeds before building on it. |
| 1/3 (= baseline) | The ratio is not the mechanism. Combined with E5/E6 this closes the "make the encoder informative / slow the decoder" family; next look at what the decoder reads (cross-attention input) or at initialization, where the KRSL-tuned mT5 reached chrF 36 versus base mT5's ~29 ceiling. |
| 0/3, CE clearly worse | A decoder at 5e-5 is simply undertrained in 10 epochs; check whether train CE is far above baseline before concluding anything about takeoff. |

A takeoff is scored from the training log the same way as before: dev CE ~2.30
with BLEU > 2 and ROUGE-1 > 0.10 by epoch 10, versus ~2.65–2.75 with BLEU < 0.5
for a run that doesn't. Confirmed afterwards by re-scoring epoch 10 on the
600-clip selection set.

## Caveat carried from E6

Any single result here rests on three seeds. 1/3 versus 1/3 cannot be
distinguished statistically; only a clean 3/3 or 0/3 would be strong evidence at
this sample size.

## Interim results (2026-09-23, seeds 0–1 done; seed 2 at epoch 5)

| run | train CE | dev CE | BLEU | R1 | takeoff |
|---|---|---|---|---|---|
| E4 seed 0 | 3.334 | 2.971 | 0.10 | 0.029 | no |
| E4 seed 1 | 3.361 | 2.992 | 0.08 | 0.028 | no |
| E4 seed 2 | (epoch 5, dev CE 3.05 — same path) | | | | pending |
| baseline B seed 0 (takeoff) | 2.118 | 2.287 | 2.63 | 0.140 | yes |
| baseline B seeds 1/2 (no takeoff) | ~2.50 | 2.691 / 2.718 | 0.42 / 0.27 | 0.036 / 0.030 | no |

**The "0/3 with CE clearly worse" row fires, and it says this is not evidence
about the ratio.** Train CE 3.33–3.36 against the baseline's 2.12 means the
decoder at 5e-5 never finished fitting within 10 epochs; dev CE (2.97–2.99) is
worse than every non-takeoff baseline run. The screen confounds "slower decoder"
with "less decoder training", so it cannot say whether the ratio drives takeoff.

**Why the design was inadequate:** it held epochs fixed while cutting the
decoder's learning rate tenfold. Matched *updates* are not matched *progress*
when one arm's effective step size is a tenth of the other's.

### Follow-up that removes the confound

Raise the encoder rate WITHOUT slowing the decoder: encoder 1e-4, mT5 5e-4
(ratio 5 instead of 10). The decoder then trains exactly as in the baseline, so
train CE should land near 2.1–2.5 and takeoff rate is comparable. That isolates
"the encoder adapts faster" — the half of the hypothesis this run could not test.
The alternative, giving the slow-decoder arm 25–30 epochs to reach matched train
CE, costs ~3× the GPU time and B20 already showed long schedules introduce their
own failure (overfitting the text prior).

---

## E4b: encoder-rate screen (pre-registered, launched 2026-09-23)

E4 could not separate "slower decoder" from "less decoder training". E4b changes
the encoder rate only and leaves the decoder exactly as the baseline has it:

| | encoder LR | mT5 LR | ratio | train CE @ep10 |
|---|---|---|---|---|
| baseline (B seeds 0/1/2) | 5e-5 | 5e-4 | 10 | ~2.12–2.50, takeoff 1/3 |
| E4 (done) | 1e-4 | 5e-5 | 0.5 | 3.33–3.36, **undertrained**, takeoff 0/2 so far |
| **E4b** | **1e-4** | **5e-4** | **5** | expected ~2.1–2.5 |

Three seeds, `--encoder-lr 1e-4 --mt5-lr 5e-4`, otherwise identical to the B
seeds (upstream preprocessing, `--block-padding-mask`, 10 epochs, canonical root,
frozen selection manifest, no alignment). Judged by takeoff rate against the
baseline's 1/3.

**Validity check before reading anything into takeoff:** train CE at epoch 10
must land in the baseline's range (~2.1–2.5). If it is far above, the arm is
undertrained like E4 and the takeoff count says nothing.

| Outcome | Reading |
|---|---|
| 3/3 take off, train CE in range | The encoder was the bottleneck: it adapts too slowly at 5e-5. Adopt 1e-4, then re-screen preprocessing and alignment on top. |
| 2/3, train CE in range | Encoder rate matters but does not determine takeoff. Confirm with three more seeds. |
| 1/3 (= baseline) | Encoder rate is not the mechanism either. Together with E5/E6 this closes the encoder-side family: neither making the encoder informative nor letting it adapt faster changes how often the decoder starts reading it. Next candidates are what the decoder cross-attends to, or initialization (the KRSL-tuned mT5 reached chrF 36 against base mT5's ~29 ceiling). |
| 0/3, train CE in range | A faster encoder actively suppresses takeoff — worth knowing, and the first arm to move the rate in the wrong direction. |
| any count, train CE far above 2.5 | Undertrained like E4; no conclusion about takeoff. |

## E4 final (2026-09-24): 0/3, undertrained — no conclusion about the ratio

Seed 2 confirms seeds 0–1: train CE 3.361, dev CE 2.987, chrF 20.02, recall 0.019.
All three land far above the baseline's train CE (2.12–2.50), so the "undertrained"
row stands: **E4 says nothing about whether the LR ratio drives takeoff.**

## E4b results (2026-09-24): 2/3 takeoff, validity gate passed

| run | train CE | dev CE | chrF | BLEU | recall | takeoff |
|---|---|---|---|---|---|---|
| E4b seed 0 | 2.153 | 2.321 | 29.73 | 2.39 | 0.125 | **yes** |
| E4b seed 1 | 2.069 | 2.239 | **31.47** | **3.04** | **0.145** | **yes** |
| E4b seed 2 | 2.525 | 2.802 | 11.14 | 0.03 | 0.010 | no |
| baseline B (1/3) | 2.12–2.50 | 2.287 / 2.691 / 2.718 | 29.79 / 21.02 / 19.83 | | 0.128 / 0.031 / 0.025 | 1/3 |

**Validity gate passes:** train CE 2.07–2.53 is inside the baseline's 2.12–2.50
range, so unlike E4 the decoder trained normally and the takeoff count is readable.

**Pre-committed reading for 2/3 fires:** "Encoder rate matters but does not
determine takeoff. Confirm with three more seeds." Launched immediately as
seeds 3–5 (`run_e4b.sh`, QUEUE=(3 4 5)).

Two observations that are NOT yet claims:
- **The ceiling may have moved.** E4b seed 1 reaches chrF 31.47 / recall 0.145,
  above every previous takeoff run (29.3–29.8 / 0.12–0.13). n=1; the six-seed
  set will show whether takeoff runs at this rate land systematically higher.
- **The failure got worse when it failed.** E4b seed 2 sits at chrF 11.14, far
  below the usual non-takeoff band (19.8–22.1), with dev CE flat from epoch 5.
  A faster encoder may make the bad mode worse as well as the good mode better.

**Statistics, stated plainly:** 2/3 versus 1/3 is not significant (Fisher exact
p = 1.0). This is a reason to run more seeds, not a result.
