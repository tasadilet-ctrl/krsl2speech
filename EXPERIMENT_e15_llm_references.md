# E15: rewrite qazaqstantv references with an LLM

Status: **complete 2026-10-21. Primary (retrieval, original refs): null. Pre-check: ~2/3 of the qazaqstantv retrieval gap was evaluation-reference error. Secondary: qazaqstantv chrF +3.2, specific, mostly from ending under-generation.** Results go at the end.

## Why

E13 (train from scratch without the 4,200 audit-flagged qazaqstantv clips) and
E14 (fine-tune B_s0 without them, with a measured noise floor) are two nulls:
**dropping** misaligned references does not close qazaqstantv's gap to khabar
(T2 dev top-1 ~0.68 vs ~0.96). The flags are real (validated against speech
rate), so the remaining deficit is plausibly the broad share of loosely-matched
references the conservative mutual-mismatch test does not flag. Removal cannot
fix those; rewriting can.

The recollection stores two transcripts per qazaqstantv clip, and they fail in
opposite ways:

| | boundaries | wording |
|---|---|---|
| `text_asr` (Whisper on the broadcast audio) | right: it is what the interpreter signs | noisy, no case or punctuation |
| `text_human` (website article, fuzzy-matched) | shifted: e.g. `kazsign_218286_000`'s text ends early and its tail opens `_001`'s | right |

Training uses `text_human` (89.8%), else `text_asr`.

**Prior attempt, unresolved:** E3 arm E switched the 10% fragment references
to `text_asr` and collapsed (161/600 distinct) — the same seed and config as
B_s0, which took off. E3 concluded the takeoff lottery, not the transcripts,
decided it. E14's fine-tune-from-B_s0 design removes that lottery.

## Step 1: rewrite (no training)

`scripts/llm_realign_refs.py`, Qwen3.6-27B via vLLM, greedy, thinking off.
Input per clip: its ASR, plus the article text (`text_human`) of the clip and
its 2 neighbours on each side in time order. The LLM never sees the current
training reference as such. Instruction: cover exactly the ASR's content, from
its first word to its last, in the article's wording, spelling and numerals
where they match; keep and lightly fix unmatched speech; add nothing.

`scripts/realign_guard.py` accepts a rewrite only if, against the ASR,
word ratio is in [0.67, 1.5] and chrF (lower-cased, punctuation stripped) is at
least 45. A rejected clip keeps its current reference. Guards are set on the
dev pilot, then frozen before the train run.

**Pilot (first 600 dev clips; text-only, no retrieval scores looked at):**
- v1 prompt, 60 clips: 56 accepted. Two flaws: Whisper sometimes emits a
  passage twice and the LLM copied it; the LLM added "..." in 2.
- Prompt v2 adds "write a mistakenly repeated passage once" and "no ellipses".
  Guard v2 compares against the ASR with repeated passages (>= 5 words)
  collapsed, and strips ellipses (still 14/600 despite the instruction).
- v2, 600 clips: **96.8% accepted** (length 1.8%, overlap 0.5%, no speech or
  article 0.8%). Accepted rewrites: chrF vs ASR 72.7 (current) -> 91.8.
- Size of the change: median chrF(rewrite, current) 93; **7.9% below 30**,
  i.e. nearly replaced. Inspected the most-changed: each carries a reference
  from elsewhere in the broadcast (e.g. a speaker caption "Жұманғарин, ҚР
  Премьер-министрінің орынбасары..." on a clip that says "құжат пен келісімге
  қол қойды"); the rewrite says what was spoken.
- Throughput: ~4 clips/s per GPU after ~3 min startup.

Frozen: prompt v2, LO 0.67, HI 1.5, MIN_CHRF 45, Qwen3.6-27B greedy.

## Step 2: pre-check with an independent model (no training)

B_s0 never trained on dev, so it can judge which dev reference fits the
signing. T2 dev within-broadcast retrieval of B_s0 with qazaqstantv dev
references set to (a) original, (b) LLM-rewritten (accepted ones), (c) ASR.
The pool is always the broadcast's own sentences in the same version, so
style is constant within a pool and PMI removes each text's prior.

| Outcome | Reading |
|---|---|
| (b) >= (a) + 0.03 | Rewritten references fit the signing better. Proceed to step 3. |
| (b) within 0.03 of (a) | No evidence the rewrite fits better; step 3 is still run (cheap) but a null is the expected result. |
| (b) < (a) − 0.03 | The rewrite damages references. Stop; inspect failures. |

(c) is context: whether raw ASR, the literal content, is already better.
Also reported: B_s0's mutual-mismatch rate on dev under (a) and (b).

## Step 3: fine-tune (the test)

Exactly E14's recipe: `train/train_min.py --init-from` B_s0 epoch 10,
`--base-lr 1e-4`, 3 epochs, stride 2. Root `~/asan_canonical_llmref`: qazaqstantv
**train** references replaced by accepted rewrites; **dev and test unchanged**
(`scripts/build_text_override_root.py --splits train`).

| arm | training data | seeds |
|---|---|---|
| **llm** | train refs rewritten | 0, 1 |
| full (E14, reused) | original | 0, 1 (full_a 0.692, full_b 0.670 qazaq T2 dev) |

Reusing E14's full arms is valid only if the code path is identical; the only
change since E14 is new scripts. Seeds 0 and 1 match full_a and full_b.

**Primary:** qazaqstantv T2 dev top-1 with ORIGINAL dev references, mean of the
two llm seeds vs mean of the two full seeds. Noise from E14: n = 0.022, so
T = max(0.03, 2n) = **0.044** (conservative for a difference of two-run means).
**Specificity:** khabar, informburo within their E14 T (0.03).
Secondary: T2 dev with rewritten dev references (all four arms); selection-set
chrF and coupling delta; |llm_s0 − llm_s1| as a second noise estimate.

| Outcome | Reading |
|---|---|
| llm − full >= 0.044 on qazaqstantv, others within T | **Reference quality limits qazaqstantv.** Adopt rewritten references for training; offer them to the data owner. |
| llm − full >= 0.044, all sources move together | Not specific to qazaqstantv references. |
| within ±0.044 | Rewritten references do not help at fine-tuning stage. With E13/E14, qazaqstantv's references are unlikely to be its limit at this stage; look elsewhere (signer, domain, pose quality). |
| llm − full <= −0.044 | The rewrite hurts on the original-reference yardstick; check the secondary (rewritten dev refs) before concluding — a style shift would show as a gain there. |
| |llm_s0 − llm_s1| > 0.05 | Noise too large; report and stop. |

Caveat carried from E14: fine-tuning cannot undo 10 epochs on the original
references, so a null is weaker than a positive. A positive here would justify
a from-scratch run.

## Run

`output/e15/`: `raw_*.jsonl` (LLM outputs), `overrides_train.json`,
`overrides_dev.json`, step 2 in `output/e15/precheck/`, step 3 checkpoints
`output/e15_llm_s{0,1}/`, T2 in `output/e15_t2/`.

## Results

### Step 1 (2026-10-21 02:07)

34,768 clips rewritten in 72 min on 2 GPUs. Guard: train 96.5% accepted
(31,129; 29,724 differ from the current reference), dev 97.0% (2,424; 2,321
differ). Same rates as the pilot.

### Step 2: B_s0 pre-check (02:24)

T2 dev, within-broadcast PMI top-1, B_s0 (never trained on dev):

| dev refs | ALL | informburo | khabar | qazaqstantv | qazaq mutual mismatch |
|---|---|---|---|---|---|
| original | 0.851 | 0.925 | 0.967 | **0.670** | 20.4% |
| LLM-rewritten | 0.919 | 0.925 | 0.967 | **0.869** | 6.2% |
| ASR | 0.922 | 0.925 | 0.967 | **0.877** | 6.2% |

**llm − original = +0.198**, far over the pre-set 0.03. khabar and informburo are
identical (their refs are untouched), as they must be. Reading: the rewritten
references fit the signing much better; proceed to step 3.

What this means: the model trained on the old references already retrieves
qazaqstantv at 0.87 once it is judged against references that say what was
signed. **Roughly two thirds of qazaqstantv's measured gap to khabar
(0.670 → 0.967) was reference error in the evaluation set, not model error.**

Caveats:
- Rewritten refs are longer (mean 26.5 words vs 22.6; ASR 29.2), because they
  cover the whole spoken span. Longer candidates carry more evidence, so part
  of the gain could be length rather than fit. **Checked (post-hoc, per
  query, from the same dumps; top-1 by argmax, hence 0.684 not 0.670):**

  | qazaqstantv dev queries | n | original | rewritten |
  |---|---|---|---|
  | rewrite SHORTER (ratio < 0.9) | 61 | 0.705 | 0.803 |
  | same length (0.9–1.1) | 134 | 0.799 | 0.843 |
  | rewrite longer (> 1.1) | 178 | 0.590 | 0.910 |
  | text nearly same (chrF ≥ 85 vs original) | 230 | 0.874 | 0.904 |
  | text changed (30–85) | 104 | 0.481 | 0.798 |
  | text replaced (< 30) | 39 | **0.103** | **0.846** |

  Gains appear where the rewrite is shorter, so length is not the
  explanation. They scale with how much the reference changed: the 39
  references that were nearly replaced went from 10% to 85% retrievable, i.e.
  they were wrong. Pools change together, so per-query numbers are
  descriptive, not independent.
- The rewrite (0.869) is slightly below raw ASR (0.877) — within noise
  (373 queries, SE ~0.017) — so on this measure the LLM's value over ASR is
  cleanliness (case, punctuation, article spelling), not fit.
- The step 3 primary measure (original dev refs) now looks like the weaker
  yardstick; it was pre-registered and stays primary, but the rewritten-ref
  T2 is the more informative secondary.

### Step 3: fine-tunes (done 03:51; scored 04:49)

Val CE (original dev refs), epoch 3: llm_s0 2.2436, llm_s1 2.2399; E14 full
2.2441 / 2.2429. Seeds differ on qazaqstantv T2 by 0.016 (E14: 0.022), so
the noise check passes.

**T2 dev, ORIGINAL references (primary):**

| arm | informburo | khabar | qazaqstantv |
|---|---|---|---|
| llm_s0 | 0.939 | 0.970 | 0.684 |
| llm_s1 | 0.928 | 0.967 | 0.700 |
| E14 full_a / full_b | 0.936 / 0.922 | 0.956 / 0.967 | 0.692 / 0.670 |
| **mean llm − mean full** | +0.005 | +0.007 | **+0.011** |

**Pre-registered reading: "within ±0.044", so the rewritten references do not
help at the fine-tuning stage on this measure.**

**T2 dev, REWRITTEN references (secondary):**

| arm | informburo | khabar | qazaqstantv |
|---|---|---|---|
| llm_s0 / s1 | 0.939 / 0.928 | 0.970 / 0.967 | 0.895 / 0.898 |
| E14 full_a / full_b | 0.936 / 0.922 | 0.956 / 0.967 | 0.882 / 0.866 |
| B_s0 | 0.914 | 0.959 | 0.871 |
| **mean llm − mean full** | +0.005 | +0.007 | **+0.022** |

Positive in the same direction, still under 0.044. (B_s0 here 0.871 vs 0.869
in step 2: different null-pose draws; khabar 0.959 vs 0.967 is the same
effect, so null sampling alone moves a source by ~0.01.)

**Selection set (600 clips, original refs), epoch 3:**

| arm | chrF all | qazaq chrF | qazaq recall | qazaq len ratio | infor chrF | khabar chrF | coupling Δ [95% CI] |
|---|---|---|---|---|---|---|---|
| llm_s0 | 32.37 | **33.34** | 0.177 | **0.97** | 28.32 | 35.42 | 11.29 [10.51, 12.07] |
| llm_s1 | 32.41 | **32.86** | 0.172 | **0.97** | 28.74 | 35.59 | 11.22 [10.44, 12.01] |
| E14 full_a | 31.02 | 30.02 | 0.150 | 0.83 | 27.66 | 35.11 | 10.58 [9.80, 11.39] |
| E14 full_b | 31.34 | 29.74 | 0.146 | 0.84 | 28.54 | 35.41 | 10.64 [9.85, 11.44] |

qazaqstantv chrF +3.2 (the identical full arms differ by 0.28; the llm seeds by
0.48) and content recall +0.027, while informburo (+0.43) and khabar (+0.25) stay
within their arm-to-arm spread: **specific to qazaqstantv.** The mechanism is
visible in the length ratio: 0.84 → 0.97. Models trained on the original
references under-generate on qazaqstantv; trained on references that cover
the whole spoken span, they stop. That is exactly what E3 arm E's
pre-registered reading predicted for the fragment hypothesis ("look for a
rising hypothesis/reference length ratio on qazaqstantv"), which E3 could
never test because the run collapsed. Part of the chrF gain is that length
correction, which chrF's recall side rewards; whether translations also got
more accurate per word is not separated here. Coupling rises 0.65, with
overlapping CIs: not established.

### Reading

1. **The evaluation set was the bigger problem.** About a fifth of qazaqstantv
   dev references do not match the signing (mutual mismatch 20.4% → 6.2% after
   rewriting), and that alone held B_s0's measured retrieval at 0.670 instead of
   0.869. Every qazaqstantv number in this project was measured against those
   references.
2. **Training on rewritten references** gives a real, qazaqstantv-specific
   gain on generation (chrF +3.2, length ratio 0.84 → 0.97), and a small
   positive retrieval change (+0.011 / +0.022) that does not clear the
   pre-registered bar. Retrieval is near its ceiling for an already-coupled
   model, which E14 also showed.
3. Fine-tuning cannot undo 10 epochs on the original references, so the
   training effect is a lower bound.

**Recommendation:**
- Use the rewritten qazaqstantv references for dev/test from now on, keeping
  the originals alongside. Comparisons between old and new numbers then need
  both versions.
- Default training to the rewritten train references (`~/asan_canonical_llmref`).
- Offer the rewrites (`output/e15/overrides_*.json`) to the data owner.
- The from-scratch test (B config, rewritten refs, 3 seeds) is the natural next
  run; it also asks whether cleaner targets change the takeoff rate.
- Test is untouched: no test references have been rewritten or scored.
