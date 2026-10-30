# What E16_s0 gets wrong (dev selection set, 600 clips; 2026-10-21)

Script: one-off, scratchpad `err.py` (logic below). Dev only; test is spent.

**Reference-word recall by how often the word occurs in training refs**
(content words of length >= 4, plus numbers; a word counts if it appears anywhere
in the hypothesis):

| word band | n | E16_s0 | B_s0 |
|---|---|---|---|
| common (500+ train occurrences) | 4,603 | 0.323 | 0.237 |
| mid (20–500) | 7,186 | 0.159 | 0.094 |
| rare (< 20) | 2,325 | **0.026** | 0.006 |
| never seen in training | 530 | 0.000 | 0.000 |
| numbers | 477 | **0.532** | 0.382 |

Hypothesis content-word precision (share of its content words found in the
reference): E16_s0 **0.201**, B_s0 0.148.

Per-clip chrF quantiles 10/25/50/75/90: 22.6 / 28.4 / 34.0 / 40.7 / 47.0.
Share of clips at chrF >= 50: informburo 2%, khabar 10%, qazaqstantv 7%.

## Reading

- **The model gets the story, not the specifics.** At the median it names the
  right topic, place and figure ("петропавлдан 50 шақырым", the Наурыз
  programme, Қостанай schools closing for cold) in fluent Kazakh, then fills the
  rest with plausible but wrong detail.
- **Rare and specific words are almost never recovered** (2.6%; 0% for words
  unseen in training). These are mostly names and places: exactly what
  interpreters fingerspell or use name signs for, at hand sizes of 31–51 px.
- **80% of its content words are not in the reference.** Hallucinated specifics
  (e.g. invented village and town names) are the user-facing failure, more than
  bad grammar.
- Numbers are a relative strength (53%).

## What this implies for the next lever

- A stronger decoder prior (E7-style domain-adapted mT5, redone on the
  rewritten refs) should help common/mid words and fluency, but cannot create
  rare-word evidence the pose encoder does not carry; it may also increase
  confident hallucination. Judge it on precision as well as chrF.
- Rare-word recognition is bounded by the input (hand resolution, keypoints
  only). Within the project's constraints, the practical route to specifics is
  the one retrieval already shows works: when candidate text exists (the day's
  articles), choose rather than generate.
