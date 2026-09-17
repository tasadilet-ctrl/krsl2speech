# legacy_prosody — archived Phase 2/3 code

Rescued 2026-09-17 from `../krsl2speech copy/` (a July 10 snapshot) just
before that folder was deleted.

To be precise about what was actually at risk: these files were removed from
the working tree in `3db7e8e` and nine of the ten remain in git history at
`3db7e8e^`. Five of those nine are byte-identical to the history copy and are
therefore redundant here. The other five are not:

| file | status |
|---|---|
| `train/train_prosody_gan.py.bak` | **never tracked — existed nowhere else** |
| `models/prosody_gan.py` | differs from the tracked version (July 10 state) |
| `train/train_prosody.py` | differs |
| `inference/sign2speech.py` | differs |
| `data/tts_dataset.py` | differs |

So the rescue preserved one genuinely unique file and four earlier variants.
The rest is duplication of git history, kept only so the archive reads as a
complete snapshot rather than a partial one.

They are the prosody / speech-synthesis implementation:

| file | role |
|---|---|
| `models/prosody_gan.py` | ProsodyGAN (Phase 2) |
| `models/fastspeech2.py` | FastSpeech2 TTS (Phase 3) |
| `train/train_prosody.py`, `train/train_tts.py` | their trainers |
| `train/train_prosody_gan.py.bak` | an earlier trainer draft |
| `data/prosody_dataset.py`, `data/tts_dataset.py`, `data/extract_prosody.py` | datasets + extraction |
| `inference/sign2speech.py`, `inference/extract_prosody.py` | end-to-end sign→speech inference |

## Status: dead code, kept for reference only

The prosody and speech-synthesis directions were **dropped on 2026-08-25** —
the project is now keypoints → Kazakh text only, focused on translation
quality. Nothing here is imported by the live tree.

Why it was kept rather than deleted: it is real work, it costs ~1 MB, and it
is the only record of the Phase 2/3 approach. If the paper ever needs to
describe what was tried and set aside, this is the source.

Do not wire these back in without an explicit decision to reopen that
direction. See `EXPERIMENT_prosody_supervision.md` for why the
prosody-as-supervision ablation came out null.
