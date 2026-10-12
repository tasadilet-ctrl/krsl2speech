# E7: initialise from a text-adapted mT5

Status: **pre-registered, launched 2026-09-29.** Results go in a section at the
end; nothing above it is edited after launch.

## Why

Three encoder-side interventions have been screened with seeds and none changed
how often the decoder starts reading the encoder:

| arm | takeoff |
|---|---|
| baseline B | 1/3 |
| E5 sequence alignment | 0/3 |
| E6 token alignment | 1/3 |
| E4b encoder LR ×2 | 2/6 |

Overall takeoff is 4/19 (~21%), and every base-mT5 run that does take off lands
at chrF 29.3–31.5. The one materially better number this project has produced
came from a different initialisation: the colleague's KRSL-tuned mT5 reached
chrF 36 / recall 0.207 (E1). Its training history is unrecorded, so it cannot
back a published claim.

E7 reaches for the same effect with clean provenance.

## What was built

`scripts/adapt_mt5_text.py` domain-adapts base mT5 on the **canonical TRAIN
transcripts only** — text, no video, nothing outside our own training split.

- **Objective: T5 span corruption, not language modelling.** Training a decoder
  to emit corpus text unconditionally is the failure mode under investigation,
  so teaching it that first would be self-defeating. Span corruption keeps the
  task conditional and is mT5's own pretraining objective.
- 59,041 transcripts, 3 epochs, lr 1e-4, batch 24, seed 0.
- **Held-out dev span-corruption perplexity: 15.53 → 4.59 (70.4% lower.)** Dev
  text was never trained on; it is used only for this measurement.
- Saved to `~/mt5_kaznews` with `adaptation.json` recording all of the above.

Two implementation details worth recording, both caught before training:
- mT5's sentinel ids must be resolved by **encoding the literal string**:
  `convert_tokens_to_ids('<extra_id_0>')` and `get_vocab()` both return `<unk>`
  (id 2) on this tokenizer version, which walks into negative ids as the span
  index grows. Correct ids are 250099 − k, now asserted at construction.
- Realised masking is 12.6% against the 15% target (span-length clamping);
  left as is, since the goal is domain adaptation rather than exact replication.

## Design

Three seeds of the **baseline B config** — upstream Uni-Sign preprocessing,
`--block-padding-mask`, 10 epochs, canonical root, frozen 600-clip selection
manifest, **baseline learning rates** (encoder 5e-5 / mT5 5e-4), no alignment —
changing one thing: `--mt5-path ~/mt5_kaznews` instead of `google/mt5-base`.

**E7 vs B isolates initialisation.** Judged by takeoff rate against B's 1/3, and
by the ceiling of the runs that do take off.

## Readings (pre-committed)

| Outcome | Reading |
|---|---|
| ≥ 2/3 take off, ceiling above 31.5 | Initialisation is the lever: a decoder already fluent in the domain both takes off more often and reaches further. Adopt it as the base for everything after, and re-screen the encoder-side arms on top. |
| ≥ 2/3 take off, ceiling ~29–31.5 | Adaptation makes takeoff more reliable without raising the ceiling. Still worth adopting; the ceiling is then set by something else. |
| 1/3 (= baseline), ceiling above 31.5 | Adaptation raises the ceiling but not the rate. Worth adopting for quality, and it leaves the bimodality unexplained. |
| 1/3 and ceiling unchanged | Domain-adapting the text side does nothing here. The E1 chrF 36 must then come from the colleague's mT5 having seen the actual video-text task, not merely the domain — which would make it unusable as a clean baseline, and would point at supervised pretraining instead. |
| 0/3 | Adaptation actively interferes; check train CE against baseline's 2.12–2.50 before concluding. |

Sample-size caveat, carried from E4b: three seeds cannot separate 1/3 from 2/3.
A 2/3 here means "run three more", not "confirmed" — E4b's 2/3 became 2/6.

## Run

`run_e7.sh`, tmux `krsl` window `e7`. Trains the three seeds (one per free GPU),
then rescores B and E7 together with one scorer at epoch 10, applies the takeoff
rule (chrF ≥ 26 and content recall ≥ 0.08), and writes
`output/e7_rescore/summary.txt`. Nothing depends on a session staying open.

## Results: none. The run was terminated at epoch 5 of 10 (2026-09-29)

Both seeds were killed at the user's request to free the GPUs; seed 2 never
launched. **No epoch-10 checkpoint exists, so the takeoff rule was never
applied and none of the pre-committed readings above can be claimed.** E7 is
untested, not falsified.

What survives is `phase1_mt5_epoch5.pth` for seeds 0 and 1. Because all three
B seeds also have epoch-5 checkpoints, those were rescored together on the
frozen 600-clip selection set through one metric path (`run_e7_ep5_rescore.sh`,
`output/e7_ep5_rescore/`). Contamination was 0 for all five.

**This comparison is exploratory and off-protocol.** It is at the wrong epoch
for the takeoff rule, and its reference for "on the takeoff trajectory" is a
single run (B seed 0). Read it as a hint, not a result.

| checkpoint | WER | BLEU | chrF | recall | distinct | top-1 share | ep10 verdict |
|---|---|---|---|---|---|---|---|
| B s0 ep5 | **1.0001** | **0.70** | **23.16** | **0.059** | **586/600** | **1.0%** | **took off** (chrF 29.79) |
| B s1 ep5 | 1.0267 | 0.18 | 20.12 | 0.021 | 407/600 | 5.7% | no |
| B s2 ep5 | 1.0202 | 0.09 | 18.11 | 0.017 | 234/600 | 9.7% | no |
| E7 s0 ep5 | 1.0448 | 0.18 | 20.91 | 0.022 | 210/600 | 5.7% | killed |
| E7 s1 ep5 | 1.0247 | 0.13 | 17.41 | 0.019 | 231/600 | **25.8%** | killed |

Three observations.

**Output diversity separates at epoch 5 far more sharply than CE does.** B seed
0 emits 586 distinct outputs of 600 with no sentence above 1%, while every other
checkpoint sits at 210–407 distinct. On validation CE the same contrast is only
2.602 vs 2.75–2.78.

This was followed up across every arm with a known epoch-10 verdict (18 runs,
4 takeoffs) — see `NOTES_ep5_takeoff_predictor.md`. **Epoch-5 distinct-output
count separates the two groups perfectly** (takeoff 577–591, no-takeoff 4–547,
nothing in the gap). Top-1 share, the metric this section originally proposed,
does *not*: 16/18 at best, because takeoff runs span 1.0–3.5% and collide with
non-takeoff runs there. The threshold is fitted in-sample on 4 positives, so it
is a screening heuristic, not a law.

**Neither E7 seed was on the takeoff trajectory when killed.** Both sit with the
non-takeoff B seeds on every column.

**E7 seed 1 is the most collapsed run in the epoch-5 set**: one sentence
("енді әлемді елең еткізген басты жаңалықтарға тоқталсақ") for 155 of 600 clips,
25.8%, against a worst-case 9.7% among the B seeds. This raises a hypothesis the
design did not anticipate: **text adaptation may reinforce the failure mode it
was meant to escape.** Span corruption was chosen over language modelling
precisely to avoid teaching unconditional generation, but span corruption on
this corpus still makes the model fluent in the corpus's stock broadcast
phrases — which is what a decoder ignoring its encoder falls back on. The
adaptation's head start confirms it learned them fast: train CE 4.10/4.15 at
epoch 1 against B's 4.96/4.96/5.44, an advantage fully gone by epoch 5
(2.83–2.89 vs 2.73–2.86).

**If E7 is resumed**, `--epochs 10` counts totals including completed epochs, so
the epoch-5 checkpoints continue rather than restart (~2.4 h for the pair).
Seed 2 would need a full run. Before spending that, the hypothesis above is
worth weighing: it predicts E7 does no better than B, and the cheaper test of it
is to measure top-1 share at epoch 5 across the arms already on disk.


## Outcome: seed 0 carried to epoch 10 (2026-10-01)

Seed 0 was resumed from its epoch-5 checkpoint (`run_e7_resume.sh`; optimizer
state and global_step 9220 restored, so the LR decay continued mid-schedule
rather than re-warming) and completed epoch 10. Rescored with the three B seeds
in one pass: `output/e7_resume_rescore/summary.txt`.

| checkpoint | chrF | recall | BLEU | distinct | top-1 | takeoff |
|---|---|---|---|---|---|---|
| B s0 | 29.79 | 0.128 | 2.63 | 598/600 | 0.3% | **YES** |
| B s1 | 21.02 | 0.031 | 0.42 | 503/600 | 5.3% | no |
| B s2 | 19.83 | 0.025 | 0.27 | 523/600 | 2.3% | no |
| **E7 s0** | **18.85** | **0.018** | 0.18 | 283/600 | 11.0% | **no** |

**Result: 0/1 takeoff.** No pre-committed reading applies -- all of them are
written for three seeds. The nearest is the 0/3 row, whose check is "verify train
CE against baseline's 2.12-2.50 before concluding interference": train CE came in
at **2.4407, inside that band**, so training was healthy and the 0/3 row's
"actively interferes" conclusion is NOT supported.

What the epochs show instead is the text prior being fitted while the pose input
goes unused:

| | epoch 5 | epoch 10 |
|---|---|---|
| Train CE | 2.890 | 2.441 |
| Val CE | 2.782 | 2.788 |

Train CE fell 0.45 while val CE moved +0.006. B seed 0's takeoff reached val CE
2.287. chrF also *declined* from epoch 5 (20.91) to epoch 10 (18.85), ending
below all three B seeds.

**This does not show that adaptation hurts.** B fails 2/3 of the time, so a
single no-takeoff is the single most likely outcome for any B-like arm; 0/1 is
nearly uninformative about E7's rate. The hypothesis recorded above -- that span
corruption on this corpus may make the boilerplate fallback easier to produce --
is consistent with this run and equally consistent with ordinary seed variance.
Separating them needs seeds, not interpretation.

**Status: E7 is 0/1, not closed.** Deciding it either way costs 2 more full runs
(~5.5 h each from scratch; seed 1 could instead be resumed from its epoch-5
checkpoint for ~2.5 h). Given that the encoder-side family is 4/18 and E7 is now
0/1 for a combined 4/19, the prior on any of these arms moving the takeoff rate
is low -- which is an argument for changing what is varied rather than buying
more seeds on this axis.
