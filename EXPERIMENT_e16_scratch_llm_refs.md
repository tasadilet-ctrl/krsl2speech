# E16: train B from scratch on the LLM-rewritten qazaqstantv references

Status: **complete 2026-10-21 20:04. 2/3 take off; both replicate E15 (chrF 35.6 / 34.8 vs B_s0 29.8, gains in every source). Retrieval row does not fire.** Results go at the end.

## Why

E15 fine-tuned the already-coupled B_s0 on rewritten qazaqstantv training
references (`~/asan_canonical_llmref`) and found a qazaqstantv-specific gain
in generation: chrF 29.9 → 33.1, output/reference length ratio 0.84 → 0.97,
khabar and informburo flat. Retrieval moved +0.011 / +0.022, under the bar.
Fine-tuning cannot undo 10 epochs on the old references, so that is a lower
bound. This asks the from-scratch question, and a second one only a scratch
run can: **do cleaner targets change how often training takes off?**

## Design

Exactly E13's B config with the root swapped and nothing excluded:
`train/train_min.py --pretrained-unisign checkpoints/unisign/csl_stage1_weight.pth
--epochs 10`, `ASAN_ROOT=~/asan_canonical_llmref`, seeds 0, 1, 2. Train refs
rewritten (29,724 of 32,268 qazaqstantv clips); dev and test unchanged.

Baselines: B s0/s1/s2 (same config, original refs; B_s0 took off) and
E13 s0/s1/s2 (flagged clips excluded; 0/3, two partial).

## Measures

Takeoff (unchanged rule): chrF ≥ 26 AND content recall ≥ 0.08 on the frozen
600-clip selection set (original refs; 7 of its 200 qazaqstantv refs are
fragments).

For each seed that takes off, against B_s0:
- **qazaqstantv specificity:** qazaq/khabar chrF ratio (B takeoffs
  0.836 / 0.842; E15 fine-tune ~0.93) and qazaqstantv length ratio.
- **T2 dev retrieval** on both reference versions (original: B_s0 0.670;
  rewritten `~/asan_e15_devllm`: B_s0 0.869–0.871).
- Coupling delta (pose-shuffled), as always.

## Readings (pre-committed)

| Outcome | Reading |
|---|---|
| A takeoff seed with qazaq/khabar chrF ratio ≥ 0.90 AND qazaq length ratio ≥ 0.93 | **E15's generation gain replicates from scratch.** Make the rewritten references the training default. |
| Takeoff seeds, ratio within B's range (< 0.87) | The fine-tune gain does not carry to scratch training; treat E15's chrF gain as fragile. |
| T2 qazaqstantv (either ref version) ≥ B_s0 + 0.044 with khabar/informburo within 0.03 | Retrieval gain from scratch, which fine-tuning could not show. |
| 3/3 take off | Suggestive that clean targets raise the takeoff rate (≈4% at the historical ~1/3 rate). Not proof at 3 seeds. |
| 0/3 take off | Takeoff rate not visibly changed; the measures above cannot be applied. Report per-source chrF and length ratio of partial runs against E13/B's partials, as weak evidence. |
| 1–2/3 | Consistent with the historical rate; judge only the takeoff seeds. |

## Run

`run_e16.sh`, tmux `e16`: seeds 0 and 1 in parallel, seed 2 on the first GPU
to free up (~9.5 h of training). Then rescore + pose-shuffled rescore on the
selection set, and T2 dev on both reference versions for all seeds.
Outputs: `output/e16_seed{0,1,2}/`, `output/e16_rescore{,_shuf}/`,
`output/e16_t2/`, `output/e16_t2_llmdev/`.

## Results (2026-10-21; seed 2 added below once it finished)

Selection set, epoch 10 (original refs):

| run | chrF | recall | distinct | informburo | khabar | qazaq | qazaq/khabar | qazaq len | coupling Δ [95% CI] | takeoff |
|---|---|---|---|---|---|---|---|---|---|---|
| **E16_s0** | **35.57** | **0.191** | 599/600 | 32.28 | 38.43 | **35.97** | **0.936** | **0.98** | **14.26** [13.40, 15.17] | yes |
| E16_s1 | 22.70 | 0.034 | 579/600 | 20.85 | 24.07 | 23.15 | (0.962) | (1.01) | 1.50 [1.03, 1.98] | no |
| B_s0 | 29.79 | 0.128 | 598/600 | 27.01 | 33.79 | 28.24 | 0.836 | 0.81 | 9.52 | yes |

**Pre-registered row fires for s0: ratio 0.936 ≥ 0.90 and length 0.98 ≥ 0.93 —
E15's generation gain replicates from scratch.** It is also larger and not
qazaqstantv-only: informburo +5.3, khabar +4.6 chrF over B_s0, coupling
+4.7. No base-mT5 run had exceeded chrF ~31.5; this is 35.6. s0 also took off
earlier (val CE 2.22 at epoch 6 vs B_s0 2.45). One takeoff against one: the
spread between scratch takeoffs is unmeasured. s1 did not take off
(consistent with the ~1/3 historical rate).

Scoring sped up: seeds 0/1 scored on the idle GPU (`output/e16early_*`);
run_e16.sh replaced by `e16_finish.sh` (seed 2 only, both GPUs, `output/e16s2_*`).

### Seed 2 (finished after the table above was written)

`output/e16s2_rescore`, same scorer and selection set: **chrF 34.76, content
recall 0.183, 598/600 distinct, takeoff yes.** So E16 is **2/3**, which is the
count the status line reports, and both takeoffs clear the old base-mT5
ceiling: 35.57 and 34.76 against ~31.5.

The per-source breakdown and coupling delta were not recomputed for s2 — that
run scored the selection set only — so the specificity argument above rests on
s0. Two takeoffs at 35.6 and 34.8 also give the first estimate of the spread
between scratch takeoffs under rewritten references: 0.8 chrF.

T2 dev (PMI top-1), seed 0 vs B_s0:

| refs | run | informburo | khabar | qazaqstantv |
|---|---|---|---|---|
| original | E16_s0 | 0.957 | 0.979 | 0.713 |
| original | B_s0 | 0.925 | 0.967 | 0.670 |
| rewritten | E16_s0 | 0.957 | 0.979 | 0.895 |
| rewritten | B_s0 | 0.914 | 0.959 | 0.871 |

E16_s1 (no takeoff), original refs: 0.487 / 0.589 / 0.464.

**Retrieval row does not fire:** qazaqstantv +0.043 (original refs) and +0.024
(rewritten), both under 0.044, and informburo moves +0.032, outside the 0.03
specificity window. Retrieval improves in every source, in line with the
general chrF gain, rather than specifically on qazaqstantv.

## Final (seed 2 scored 20:04)

| run | chrF | recall | informburo | khabar | qazaq | qazaq/khabar | qazaq len | coupling Δ [95% CI] | T2 qazaq orig / rewritten | takeoff |
|---|---|---|---|---|---|---|---|---|---|---|
| E16_s0 | 35.57 | 0.191 | 32.28 | 38.43 | 35.97 | 0.936 | 0.98 | 14.26 [13.40, 15.17] | 0.713 / 0.895 | yes |
| E16_s1 | 22.70 | 0.034 | 20.85 | 24.07 | 23.15 | — | — | 1.50 [1.03, 1.98] | 0.464 / — | no |
| **E16_s2** | **34.76** | **0.183** | 30.77 | 37.65 | 35.88 | **0.953** | **0.99** | **13.55** [12.71, 14.44] | 0.702 / 0.898 | yes |
| B_s0 | 29.79 | 0.128 | 27.01 | 33.79 | 28.24 | 0.836 | 0.81 | 9.52 | 0.670 / 0.871 | yes |

T2 for E16_s2, original refs: informburo 0.949, khabar 0.967.

### Reading

- **"E15's gain replicates from scratch": fires for both takeoff seeds.**
  qazaq/khabar 0.936 and 0.953 (B 0.836–0.842), length ratio 0.98 and 0.99
  (B_s0 0.81). The two takeoffs agree closely (chrF 35.57 / 34.76, coupling
  14.3 / 13.6), so the gain over B_s0 (+5.0 to +5.8 chrF) is about 7× the
  spread between them.
- **The gain is not qazaqstantv-only.** informburo +3.8 to +5.3, khabar
  +3.9 to +4.6, qazaqstantv +7.6 to +7.7 chrF. Cleaner targets for one
  source improved the whole model; qazaqstantv gained the most and now
  scores above informburo.
- **Retrieval row: does not fire.** Mean qazaqstantv T2 +0.038 (original refs)
  and +0.026 (rewritten) over B_s0, under 0.044; informburo also rises.
  Retrieval is near ceiling for coupled models.
- **Takeoff 2/3:** the "1–2/3, consistent with the historical rate" row. Both
  takeoffs left B's trajectory early (val CE at epoch 6: 2.22 / 2.29 vs
  B_s0 2.45), but 3 seeds cannot show a changed rate.
- Caveat: B_s0 is one baseline takeoff; scratch-takeoff spread under the
  ORIGINAL refs is unmeasured beyond B_s0, E4b_s1 and E9b_s2 (all chrF
  ~29.8–31.5 per E4/E9 notes), which E16's two takeoffs both clear by > 3.

**Recommendation: make `~/asan_canonical_llmref` the training default.** New
best: E16_s0 (`output/e16_seed0/phase1_mt5_epoch10.pth`), chrF 35.57.
Test set is still untouched.

## Test set (2026-10-21, run once; reporting only)

`run_e16_test.sh`. E16_s0 (chosen on dev) with B_s0 as reference. Test
references in two versions: original, and rewritten with the frozen E15
prompt/guards (first time test was rewritten: 97.3% accepted, 2,328 of 2,500
qazaqstantv refs changed, `~/asan_e16_testllm`). Speed-up: B_s0's T1 on
original refs is E12's (same code and manifest, not re-run); on rewritten refs
only qazaqstantv differs, so B_s0 was scored on those 200 clips alone.

**Generation, 600-clip test selection set (chrF / content recall / length ratio):**

| refs | run | ALL | informburo | khabar | qazaqstantv | coupling Δ [95% CI] |
|---|---|---|---|---|---|---|
| original | **E16_s0** | **35.76** / 0.195 | 31.76 / 0.145 | 39.22 / 0.229 | 36.21 / 0.217 / 1.06 | **13.95** [13.10, 14.80] |
| original | B_s0 | 30.27 / 0.135 | 27.31 / 0.095 | 34.32 / 0.171 | 28.70 / 0.141 / 0.85 | 9.39 [8.62, 10.17] |
| rewritten | **E16_s0** | **36.18** / 0.201 | (same) | (same) | 37.46 / 0.232 / 0.89 | |
| rewritten | B_s0 | 30.01 / 0.137 | (same) | (same) | 28.10 / 0.146 / 0.72 | |

**T2 within-broadcast retrieval (top-1):**

| refs | run | ALL | informburo | khabar | qazaqstantv |
|---|---|---|---|---|---|
| original | E16_s0 | 0.863 | 0.943 | 0.983 | 0.688 |
| original | B_s0 | 0.835 | 0.916 | 0.965 | 0.651 |
| rewritten | E16_s0 | **0.912** | 0.943 | 0.983 | **0.826** |
| rewritten | B_s0 | 0.883 | 0.916 | 0.965 | 0.784 |

**T1 200-way retrieval (top-1):**

| refs | run | ALL | informburo | khabar | qazaqstantv |
|---|---|---|---|---|---|
| original | E16_s0 | 0.853 | 0.880 | 0.935 | 0.745 |
| original | B_s0 (E12) | 0.807 | 0.835 | 0.885 | 0.700 |
| rewritten | E16_s0 | **0.900** | 0.880 | 0.935 | **0.885** |
| rewritten | B_s0 | 0.835 | 0.835 | 0.885 | 0.785 |

### Reading

- **Dev results hold on test.** chrF 35.76 vs 30.27 (+5.5; dev +5.8),
  coupling 13.95 vs 9.39 (dev 14.26 vs 9.52), gains in every source.
- **Retrieval improves everywhere on test**: T1 +0.046 overall on original
  refs, +0.065 on rewritten; qazaqstantv T1 0.745 → 0.885 between reference
  versions for E16_s0.
- **The test set has the same reference problem as dev.** B_s0's qazaqstantv
  T2 rises 0.651 → 0.784 and E16_s0's 0.688 → 0.826 just by rewriting the
  references. Report qazaqstantv on both versions.
- One caveat on rewritten-ref generation for qazaqstantv: rewritten refs are
  longer, so length ratios drop (E16_s0 1.06 → 0.89); chrF still rises.
- B_s0's rewritten-ref qazaqstantv T1 used a qazaqstantv-only manifest, so its
  null-pose draws differ from the full run; on dev that moves a source ~0.01.

**Headline (test, original refs): chrF 35.8, content recall 0.195, T1 200-way
top-1 0.853, T2 within-broadcast top-1 0.863.** On rewritten refs: chrF 36.2,
T1 0.900, T2 0.912.

## Seeds 3–5 (E16b, done 2026-10-22 09:09)

| run | chrF | recall | distinct | takeoff | T2 qazaq orig / rewritten |
|---|---|---|---|---|---|
| E16_s3 | 32.92 | 0.160 | 598/600 | yes | 0.705 / 0.906 |
| E16_s4 | 22.77 | 0.036 | 586/600 | no | 0.504 / 0.587 |
| E16_s5 | 17.20 | 0.017 | 320/600 | no (collapse, top-1 19.8%) | 0.300 / 0.354 |

**E16 over six seeds: 3/6 take off** (s0, s2, s3). Takeoff chrF 35.57 / 34.76
/ 32.92: mean 34.42, spread 2.65. All three clear every base-mT5 takeoff on the
original references (≤ 31.5). Against the historical ~4/19 (21%), 3/6 is
higher but not significantly so at this sample size; the takeoff lottery is
not fixed by cleaner targets.
