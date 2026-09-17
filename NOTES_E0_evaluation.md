# E0 complete — canonical splits, frozen selection, metric parity

Done 2026-09-17, addressing the first three defects in
`REVIEW_architecture_research_2026-09-17.md`. Nothing here changes model
behaviour; it changes what we are able to measure.

```bash
export ASAN_ROOT=$HOME/asan_canonical
```

## 1. Canonical, leak-free splits

Built by `scripts/build_canonical_splits.py` (deterministic, salted
SHA-256 assignment — reproducible across machines; `hash()` is process-salted
and must not be used).

| split | clips | videos |
|---|---|---|
| train | 59,008 | 10,398 |
| dev | 8,411 | 2,575 |
| test | 8,360 | 2,534 |

Only qazaqstantv needed rebuilding — khabar and informburo are byte-identical
across dataset versions and already train-disjoint, so they carry over
unchanged.

**Held-out purity.** All 3,610 qazaqstantv videos the initializer's fine-tuning
saw (archive TRAIN) are pinned to canonical TRAIN. Dev/test are drawn only from
the 3,975 videos it never saw. The builder asserts this and exits non-zero on
violation:

```
no split overlap; no eval video seen in archive train  OK
```

**Remaining caveat — not fixable from here.** `ours_enriched_friend_mt5.pth` is
`pose_pretrain_v3 encoder + a colleague's mT5`, and that mT5's training history
is unrecorded. Canonical dev/test exclude everything archive TRAIN held, but
that component cannot be cleared. Runs from this initializer stay
**EXPLORATORY**; final claims need an initializer with documented disjoint
provenance. Recorded in `split_provenance.json` alongside the split.

## 2. Frozen selection manifest

`selection_manifest.json`: **600 clips, 200 per source, over 600 distinct
videos**, stratified across four duration quartiles (30–1000 frames).
One clip per video, so no broadcast can dominate through adjacent clips.

Replaces `validate(max_gen_batches=25)` — the first 25 unshuffled batches,
which were **100% informburo** because sources concatenate in list order.
qazaqstantv is 48.6% of dev and was never generated on.

The trainer now takes `--selection-manifest`, generates over the whole subset,
logs actual coverage, and **refuses to start** if the manifest and `ASAN_ROOT`
disagree rather than silently selecting on a different subset. Without the flag
it warns loudly and falls back to the old behaviour.

Cost: 600 clips vs 200, so ~3× the per-epoch generation. Worth it — the old
number could not see the majority of the data.

## 3. One metric implementation

`utils.metrics.compute_corpus_wer` is now called by both the trainer and
`scripts/evaluate_phase1.py`.

Before, the two entry points disagreed, and the trainer disagreed with
*itself* — raw whitespace WER next to a normalized BLEU. On identical
predictions:

| metric | normalized | raw |
|---|---|---|
| WER | 0.3333 | 0.5333 |
| BLEU | **63.35** | **30.26** |

A 2× BLEU gap from punctuation handling alone.

**Consequence for the record: every BLEU we have quoted is normalized BLEU.**
Published SLT results generally report raw sacrebleu, so our 3.90 is not
comparable to literature numbers — a second, independent reason the earlier
How2Sign comparison was invalid. Both figures are now printed and labelled, and
the evaluator prints its sacrebleu signature.

Corpus WER (pooled edits / pooled words) is what both paths use — note this
differs from `compute_batch_wer_cer`, which averages per-clip WER; the two
diverge whenever clip lengths vary.

## Still open in E0

- `evaluate_phase1.py` still defaults to the first 500 clips and hardcodes LoRA
  settings instead of reading them from checkpoint metadata.
- No explicit seed control in the trainer.
- Partial gradient-accumulation flush is underweighted when fewer than four
  microbatches remain; the scheduler advances after nonfinite-gradient skips.

## Next (E1)

Re-score the existing checkpoints — including `clean_treat/` — on the canonical
dev set through the shared metrics, before spending more GPU time. Then the
decoding sweep (beam 1/4, repetition penalty 1.0/1.1/1.3, trigram blocking
on/off), dev-only.
