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
