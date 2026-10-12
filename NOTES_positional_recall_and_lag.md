# Positional recall and the interpreter-lag hypothesis (2026-10-04)

## Finding: the end of each sentence is half as recoverable from its clip

Pose-attributable content recall (clean minus pose-shuffled, words >= 4 chars)
by position in the reference, mean of the four native-fps takeoff runs plus
E9b_s2:

    bin (start -> end)   1       2       3       4       5
    recall             +0.173  +0.139  +0.132  +0.124  +0.080      2.16x

Identical shape in every run. The shuffled (text-prior) recall is flat at
~0.012 across bins, so this is not the prior favouring early words.

Survives the length confound: restricted to clips whose hypothesis is at least
as long as the reference (156-192 per run), the gradient is 2.06x. So it is not
"the decoder stops early".

## Lag spillover test: null

If interpreters lag speech, the signs for the end of sentence k-1 should spill
into clip k. Built `asan_canonical/consecutive_dev_manifest.json` (903 dev clips
in consecutive runs, 620 adjacent pairs), decoded with E4b_s1 and B_s0
(`output/lag_consec/`), and measured how often END vs START words of ref k-1
appear in a hypothesis, excluding words in that hypothesis's own reference.

| hypothesis read | E4b_s1 END-START | B_s0 END-START |
|---|---|---|
| ADJACENT (clip k) | -0.0015 | +0.0008 |
| DISTANT (clip k+1, same topic) | +0.0022 | -0.0032 |
| FORWARD (clip k-2, lag impossible) | -0.0038 | -0.0035 |

No end-word excess in the adjacent clip (SE of a difference ~0.004 at ~4,000
words per cell, so a spillover of +0.01 would have been ~2.5 sigma). Adjacent
clips share more words overall (0.026-0.033 vs 0.019-0.030), symmetrically
-- topic proximity, not lag.

## Reading

The end-of-sentence content is less recoverable from its own clip AND is not
in the next one. The most plausible explanation is that it was never signed:
simultaneous interpreters into a sign language compress and drop material,
particularly when falling behind fast news speech. Decoder drift late in the
sequence is the alternative and is not excluded.

Either way the speech transcript contains content the signing does not, which
caps achievable recall and rewards the text prior. This is testable directly
only with sign-faithful annotation.
