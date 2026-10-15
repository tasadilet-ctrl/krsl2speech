# Input resolution: the hands are 31-51 pixels (2026-10-08)

The stored videos are interpreter crops, and the "raw" videos in the archive
are the same size -- there is no higher-resolution source in this data.

| source | video (W x H) | signer height | hand extent (median) | best-run chrF |
|---|---|---|---|---|
| khabar | 314 x 282 | 240 px | **51 px** | 35-38 |
| informburo | 250 x 370 | 193 px | 34 px | 29-32 |
| qazaqstantv | 204 x 186 | 176 px | **31 px** | 29-34 |

Measured from the keypoints of 40 dev clips per source (hand extent = max
side of the 21-point hand bounding box per frame, median over frames and
clips); resolutions by ffprobe on the processed and raw videos.

## Consequences

- With 21 keypoints across a 31-pixel hand, adjacent finger joints are ~2-3 px
  apart. Handshape -- the main feature distinguishing signs -- is close to
  unrecoverable by any keypoint detector at this size.
- The audit's proposed fix, "crop the interpreter and re-detect hands at
  higher resolution", is **not possible**: the pixels do not exist. It would
  need the original broadcasts (informburo ids are YouTube ids, so their full
  frames may be re-downloadable; the interpreter inset there may be only
  modestly larger).
- RGB hand crops, the usual alternative, hit the same wall at 31 px.
- **The best source by every metric is the one with the largest hands.** The
  frame-rate account of khabar's lead (E9) did not survive its own control;
  resolution is an alternative explanation. Three sources is three data
  points, so this is a hypothesis, not a finding.

The model can still use coarse cues -- arm trajectory, location, movement,
two-handedness, non-manual signals -- which is consistent with partial
information: retrieval works (NOTES_retrieval.md) while generation does not.

## Keypoint quality, checked by eye (2026-10-14)

`scripts/render_keypoint_overlays.py` + `scripts/build_qazaq_viewer.py` render
qazaqstantv clips with keypoints drawn on (served from `~/qazaq_viewer`).

- **The whole-body hands we train on (RTMW, wb_xy[91:133]) beat the dedicated
  sign-language hand model** (`tasmulaev/rtmpose-m-distill`, hand_l_xy /
  hand_r_xy, present in every qazaqstantv .npz). Judged by the project owner
  over 80 dev clips: the dedicated model "messes up the finger keypoints very
  badly". The two sets differ by a median 6.8 px on ~31 px hands. Swapping
  them in is ruled out.
- **Knees and ankles are hallucinated.** The legs are below the interpreter
  crop; RTMW guesses them anew each frame (49-73 px jumps at p90, vs 5-10 px
  for wrists and knuckles) and they land near the resting hands. The model
  never receives them (Uni-Sign body = nose, ears, shoulders, elbows, wrists),
  so this affects only visualisation; the overlay now draws exactly the 69
  model inputs.
- **Guessed hands, measured (2026-10-14, `scripts/measure_guessed_hands.py`,
  200 dev clips per source).** The whole-body model reports both hands in every
  frame and the Uni-Sign threshold (score <= 0.3) masks **0%** of hand-frames,
  so the model cannot tell a seen hand from a guessed one. How often a hand is
  actually not visible:

  | source | hand outside the frame | RTMW score separates it (AUC) |
  |---|---|---|
  | informburo | 0.01% | 1.000 |
  | khabar | 2.08% | 0.991 |
  | qazaqstantv | 1.85% | 0.922 |

  Checked on a contact sheet of flagged frames: out-of-frame hands are real --
  hands resting below the crop, i.e. pauses -- with invented coordinates.
  The dedicated detector's "no hand" frames (2.41% on qazaqstantv) are
  mostly NOT invisibility: both hands are visible but overlapping or crossed,
  and the whole-body points sit on them. Rare frames with no interpreter at
  all (scene transitions) also occur, with hand points scattered on empty
  background; `person_found` never flags them (0 of 81,832 frames).

  **Reading: guessed hands are ~2% of hand-frames on khabar and qazaqstantv,
  ~0% on informburo, concentrated in rest pauses.** The model's own score
  knows (visible median ~5.0-5.6, out-of-frame 0.8-2.5), so a calibrated
  threshold would catch them -- score < 2.0 masks 1.0% of hand-frames. Given
  the small share and that pauses carry little signing, this is a cleanup, not
  a likely lever on translation quality; not worth a seeded arm on its own.
