# E8: make the text prior insufficient

Status: **design, not launched.** Readings for Step 2 are registered only after
Step 1 reports, for the reason given there. Nothing above the Results section is
edited after launch.

## Diagnosis

Six arms, 19 runs, 4 takeoffs. The rate has not moved:

| arm | takeoff | varied |
|---|---|---|
| B baseline | 1/3 | -- |
| E4 decoder LR /10 | 0/3 | confounded design |
| E4b encoder LR x2 | 2/6 | optimizer |
| E5 sequence alignment | 0/3 | encoder objective |
| E6 token alignment | 1/3 | encoder objective |
| E7 adapted init | 0/1 | decoder initialisation |

E5 and E6 are the informative failures. Both drove their alignment loss far
below chance -- the encoder demonstrably learned to identify its own transcript
-- and neither changed the takeoff rate. E7 made the decoder more fluent in the
domain and did not change it either.

So: making the encoder more informative does not help, and making the decoder
more fluent does not help. What varies between runs is whether the decoder ever
starts *reading* the encoder, and nothing tried so far acts on that directly.

The structural reason is in the objective. Under teacher forcing the decoder
predicts token t from the gold prefix `y_<t` plus cross-attention over the pose.
For broadcast Kazakh the prefix alone is a strong predictor, so there exists a
low-CE solution that ignores cross-attention entirely. That solution is reachable
by gradient descent from the start, while the pose-reading solution requires the
encoder to become useful first. Training is bimodal because the model picks
whichever basin it falls into, and the prior-only basin is always available.

**E8 attacks the availability of that basin** rather than either tower.

## Step 1: build an instrument first (do this before any training)

Takeoff is a binary read on a 4/19 base rate. Detecting a change from 1/3 to 2/3
at that rate needs more seeds than this project can afford. Before spending GPU
on an intervention, replace the binary with a continuous measure of the thing
actually under study.

**Pose-shuffle sensitivity.** Score a checkpoint twice on the frozen 600-clip
selection set: once normally, once with the pose encoder output permuted across
the batch so every clip is decoded against another clip's pose. Define

    delta = chrF(clean) - chrF(pose-shuffled)

A decoder that ignores its encoder scores the same either way (delta ~ 0). A
decoder that reads it loses most of its chrF (delta large). This measures
coupling directly, on a continuous scale, with no retraining.

**Validation (this is the part that decides whether E8 is worth running):**
compute delta for all 19 existing epoch-10 checkpoints, whose takeoff labels are
known. The instrument is usable only if delta separates the 4 takeoff runs from
the 15 others and, more importantly, **varies within the no-takeoff group**. If
every non-takeoff run sits at delta ~ 0, there is no partial coupling to detect
and E8 is back to needing binary takeoff counts.

Cost: ~40 s per checkpoint, 19 checkpoints, about 13 minutes. Clean scores
already exist in `output/all_rescore/` and `output/e4b_rescore/`.

Implement as `--shuffle-pose` in `scripts/rescore_checkpoints.py`, permuting
`pose_emb` along the batch dimension after the encoder and before the mT5 call
(`train/train_encoder_mt5.py:643`). Permute the *encoder output*, not the input
keypoints, so sequence lengths and the pose attention mask stay valid -- a
length mismatch would confound the ablation with a padding artefact. Use a fixed
derangement (no clip gets its own pose) under a fixed seed.

## Step 2: the intervention

**Decoder input corruption.** With probability p, replace each decoder input
token with a random vocabulary token, independently per position per step.
Labels are untouched, so the model is still trained to produce the correct text
-- it simply cannot rely on an intact prefix to do it. Where the prefix is
corrupted, the only remaining source of information about token t is
cross-attention over the pose.

Why this one:

- **One scalar knob**, no new parameters, no new loss term, no second model.
  E5's two optimizer bugs both came from adding trainable heads; this adds none.
- **Inference is unchanged.** chrF, content recall, WER and the takeoff rule stay
  exactly comparable to all 19 existing runs. Nothing about the metric path moves.
- **It removes the prior-only basin instead of penalising it.** At p high enough
  there is no low-CE solution that ignores the encoder, so the model is not being
  asked to prefer coupling -- coupling is the only thing left that works.

Random tokens rather than a mask token: mT5 has no mask token in the decoder
vocabulary, and reusing a sentinel would put the decoder in a span-infilling mode
it was pretrained to answer in a particular way. Random tokens are
distribution-free noise.

Corruption is applied at **training only**. Validation runs with clean teacher
forcing so that validation CE remains comparable across arms and against the 19
runs on record.

Corruption rate is **constant, not annealed**. An anneal adds a schedule shape
to the design and makes "which p mattered" unanswerable; constant p is the
smallest change that tests the hypothesis.

### Choosing p

p too low leaves the prior-only basin reachable; p too high destroys the decoder's
fluency and training fails for an uninteresting reason. Rather than guess, run a
**pilot to epoch 5 only** at p in {0.15, 0.35, 0.55}, one seed each.

Screen the pilot with the epoch-5 distinct-output criterion from
`NOTES_ep5_takeoff_predictor.md` -- **but only to exclude, never to rank.** That
signal is validated as a kill criterion (below ~550 distinct, no run has
recovered in 19 tries). It is *not* validated for choosing among healthy runs:
that would assume distinct count is monotone in eventual quality, and E7 seed 0
is a direct counterexample (210 distinct at epoch 5 rising to 283 at epoch 10
while chrF *fell* 20.91 -> 18.85). Use it to drop any p whose run is already
degenerate, and use Step 1's delta to choose among whatever survives.

Pilot cost: 3 runs x 5 epochs x ~1760 s ~= 2.5 h each, two at a time on the two
GPUs, about 5 h.

### Arms

One p, carried to **three seeds**, 10 epochs, canonical root, frozen 600-clip
selection manifest, baseline LRs, upstream Uni-Sign preprocessing,
`--block-padding-mask`. Baseline is the three B seeds already on record.
**E8 vs B isolates decoder input corruption.**

Main-arm cost: 3 x ~5.7 h, two at a time, about 11.5 h.

## Readings

Step 1's readings are registered now. **Step 2's are deliberately not**, because
the yardstick they would be written against does not exist until Step 1 reports
delta's distribution on the 19 known checkpoints. Pre-committing numeric
thresholds against an unmeasured scale would be theatre. They go in this file
after Step 1 and before E8 launches.

### Step 1 (registered)

| Outcome | Reading |
|---|---|
| delta separates takeoff from no-takeoff AND varies within the no-takeoff group | The instrument works and has resolution below the takeoff threshold. Proceed to Step 2 and judge it primarily on delta, with takeoff rate as a secondary. |
| delta separates the groups but is ~0 for every no-takeoff run | Coupling is all-or-nothing; there is no partial signal to detect. Proceed, but E8 is back to binary takeoff counts and needs the seeds that implies -- reconsider whether it is affordable. |
| delta does not separate the groups | The instrument is invalid, or "takeoff" is not what I think it is. **Stop and investigate that**, because every arm in this project has been judged by a rule that would then not mean what it claims. This outcome is more important than E8. |

## Threats to validity

- **p confounds two things.** Corruption both removes the prior-only basin and
  degrades decoder training generally. A null result at a given p cannot
  distinguish "coupling is not the bottleneck" from "p was wrong". This is why p
  is piloted rather than assumed, and why a null should be reported as "null at
  p=x", not "null".
- **Train CE is not comparable across arms.** Corrupted inputs raise training CE
  mechanically. Compare validation CE (clean) only. The E4 design failed in
  exactly this way by comparing quantities that were not comparable.
- **Three seeds cannot separate 1/3 from 2/3.** E4b's 2/3 became 2/6. If E8 is
  judged on takeoff rate, a 2/3 means "run three more". If Step 1 succeeds, delta
  carries the result instead and this is much less binding -- which is the whole
  reason for Step 1.
- **The pilot's p-selection is one seed per p.** Given the bimodality, a single
  run at the best p can look bad by luck. The exclusion-only rule above limits
  the damage, but if two p values survive the screen and differ only slightly,
  carry the lower one: it is the smaller perturbation to a system whose failure
  modes are already poorly understood.

## Implementation notes

`train/train_encoder_mt5.py:669` passes `labels=label_ids` and lets mT5 build
`decoder_input_ids` internally via `_shift_right`. Corruption requires building
them explicitly:

```python
dec_in = self.mt5._shift_right(label_ids)        # also maps -100 -> pad id
if self.decoder_corrupt > 0.0 and self.training:
    real = dec_in != self.mt5.config.pad_token_id   # never corrupt pads or the
    sel = real & (torch.rand_like(dec_in, dtype=torch.float) < self.decoder_corrupt)
    rnd = torch.randint_like(dec_in, 0, self.mt5.config.vocab_size)
    dec_in = torch.where(sel, rnd, dec_in)
out = self.mt5(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
               decoder_input_ids=dec_in, labels=label_ids, return_dict=True)
```

Position 0 is the start token produced by `_shift_right`; leave it intact so
every sequence still begins correctly. Guard on `self.training` so validation is
clean. Flag: `--decoder-corrupt P`, recorded in `run_args` so
`scripts/rescore_checkpoints.py` keeps refusing mismatched pipelines.

**Verify before launching**, as a smoke test on a handful of batches: that the
realised corruption rate matches p on non-pad positions, that position 0 is never
corrupted, that labels are bit-identical to the uncorrupted run, and that with
p=0 the loss matches the current code path exactly. The E5 optimizer bug and the
mT5 sentinel bug were both caught this way and both would otherwise have produced
a plausible, wrong null.

## Budget

| step | cost |
|---|---|
| 1. instrument + validation on 19 checkpoints | ~15 min |
| 2a. pilot, 3 values of p to epoch 5 | ~5 h |
| 2b. main arm, 3 seeds to epoch 10 | ~11.5 h |

Step 1 is cheap and can invalidate the rest, so it runs first and alone.

## Step 1 results (2026-10-01)

`run_e8_step1.sh`, `output/e8_shuffled/`. All 19 epoch-10 checkpoints with known
takeoff labels, scored with `--shuffle-pose` through the same metric path as
their existing clean scores. Contamination 0 throughout.

Implementation verified before use: `roll(1)` is a derangement (no clip keeps its
own pose), the pose mask is rolled *with* the embedding so lengths stay matched,
`shuffle_pose=False` leaves the existing path bit-identical, and the B>1 guard
stops the tail batch returning a clip its own pose.

### Corpus-level delta separates, but is redundant with chrF

    takeoff    (n=4):  9.41  9.64  10.03  11.32
    no-takeoff (n=15): 0.44 ... 4.12          gap 5.29

Clean separation -- and nearly useless as a new instrument. Over the 15
non-degenerate runs the shuffled score is almost constant (18.49-21.12, stdev
0.61) and **r(delta, clean chrF) = 0.9897**. Corpus delta is clean chrF minus
~19.5. Reporting it as a new measurement would have been self-deception.

Worth keeping from it: that constant ~19.5 is **the text prior's chrF**. Every
non-takeoff run's clean chrF (18.5-24.2) sits at or barely above the score a
takeoff run gets when denied its encoder. Non-takeoff runs perform exactly as
well as a good run with its encoder switched off, which is the premise of E8
stated as a measurement.

### Per-clip paired delta is what provides the power

Scoring sentence-chrF per clip and pairing clean against shuffled gives 600
paired observations per run, so a single run gets a confidence interval:

| arm | mean delta (non-takeoff runs) | takeoff |
|---|---|---|
| E4 decoder LR /10 | **0.52** | 0/3 |
| B baseline | 1.19 | 1/3 |
| E6 token alignment | 1.41 | 1/3 |
| E5 sequence alignment | **1.49** | 0/3 |
| takeoff runs, any arm | 8.91-10.68 | -- |

Within-run 95% CI is about +/-0.5 chrF. Full table in the session record;
`output/e8_shuffled/*.predictions.jsonl` holds every per-clip prediction.

**Three findings the binary takeoff rule hid.**

1. **Coupling is graded.** 11 of 15 non-takeoff runs have a CI excluding zero.
   They do not ignore the encoder; they barely use it.
2. **Alignment did increase coupling.** E5 reaches 1.49 against B's 1.19, and E6
   1.41 -- never enough to cross into takeoff, but in the right direction. E5 was
   recorded as 0/3 and written off; on this measure it moved the quantity it
   targeted.
3. **E4 reduced coupling** to 0.52, the lowest arm, and took off 0/3. The slower
   decoder leaned harder on the prior. Consistent with the E8 hypothesis.

**E4b_s5 validates the scale**: clean chrF 24.24, just under the takeoff bar, and
delta 3.74 -- above every other non-takeoff run, below every takeoff run.
Partial coupling, correctly placed, which is exactly the resolution Step 1
needed to demonstrate.

### Registered reading applied

Outcome 1: delta separates the groups **and** varies within the no-takeoff group
(0.24-3.74 on the per-clip measure, with 11/15 CIs excluding zero). The
instrument has resolution below the takeoff threshold. **Proceed to Step 2.**

With one correction to that reading's wording: judge E8 on the **per-clip paired
delta**, not the corpus delta, which is collinear with chrF and adds nothing.

### Caveats

- Between-seed spread of delta within an arm (~0.2-0.8) is comparable to the
  within-run CI (~+/-0.5). Three seeds per arm are still required; what changes
  is that three seeds now yield a mean with error bars instead of a count out of
  three.
- The per-clip CI is a sampling interval over clips, not over seeds. It states
  how precisely that run's coupling is measured, not how much coupling varies
  between runs of the same config.
- The arm-level differences in point 2 above rest on 2-3 seeds each with
  differences of ~0.3. Suggestive, not established.
- Sentence-level chrF is noisier than corpus chrF and the two are not
  interchangeable; mean per-clip delta is not equal to corpus delta (e.g. B_s0:
  9.52 vs 10.03). Use one or the other consistently, never mixed.

## Step 2 readings (registered 2026-10-01, after Step 1, before launch)

Judged on **mean per-clip paired delta over three seeds**, against the B
baseline's three seeds on the same measure (B: 9.52 / 1.27 / 1.11, mean 3.97;
non-takeoff mean 1.19). Takeoff rate is reported as a secondary.

| Outcome | Reading |
|---|---|
| All three seeds at delta > 4, or any at > 8 | Corrupting the decoder input forces encoder use. The prior-only basin was the mechanism. Adopt, then re-screen the encoder-side arms (E5/E6) on top, since Step 1 shows they move coupling and may now compound. |
| Non-takeoff seeds' mean delta > 2.0, no change in takeoff rate | Coupling increased without crossing into takeoff -- the same partial effect as E5/E6 but larger. Raise p and repeat; the mechanism is right and the dose is low. |
| Mean delta within B's non-takeoff range (0.9-1.5) | Corruption at this p does not change coupling. Report as null **at p=x**, not as a null for the hypothesis, and state the piloted p values. |
| Mean delta below 0.9, or validation CE above B's 2.69-2.72 band | Corruption is damaging the decoder rather than redirecting it. Lower p; if the lowest piloted p also does this, the intervention is too blunt for this model. |

Pre-committed regardless of outcome: report validation CE (clean) alongside, and
do not compare training CE across arms -- corrupted inputs raise it mechanically,
which is the error that invalidated E4.
