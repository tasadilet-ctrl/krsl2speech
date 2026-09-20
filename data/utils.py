"""
Shared utilities for loading keypoints and extracting prosody.
"""
import os
import warnings

import numpy as np
import librosa
import torch

# COCO-WholeBody slices (from extract.py)
BODY_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12]  # 11 body points
FACE_IDX = list(range(23, 91))                    # 68 face points
LIPS_IDX = list(range(71, 91))                    # 20 lip points

# Total keypoint dim: body(11×2) + face(68×2) + lips(20×2) + hand_l(21×2) + hand_r(21×2)
# = 22 + 136 + 40 + 42 + 42 = 282
KEYPOINT_DIM = 282

# Sub-pose group definitions (matches encoder)
GROUPS = [
    {'name': 'body',      'num_nodes': 11, 'dim': 22,  'start': 0},
    {'name': 'face',      'num_nodes': 88, 'dim': 176, 'start': 22},
    {'name': 'left_hand', 'num_nodes': 21, 'dim': 42,  'start': 198},
    {'name': 'right_hand','num_nodes': 21, 'dim': 42,  'start': 240},
]


def assemble_keypoints(wb_xy, hand_l_xy, hand_r_xy):
    """
    Assemble keypoints into a single vector per frame.

    Args:
        wb_xy:      (T, 133, 2) — COCO-WholeBody
        hand_l_xy:  (T, 21, 2)  — left hand (RTMPose-m)
        hand_r_xy:  (T, 21, 2)  — right hand (RTMPose-m)

    Returns:
        kps: (T, 282) — concatenated keypoints
    """
    # body + face + lips from wb_xy
    wb_slice = wb_xy[:, BODY_IDX + FACE_IDX + LIPS_IDX, :]  # (T, 99, 2)
    wb_flat = wb_slice.reshape(wb_slice.shape[0], -1)       # (T, 198)

    hl_flat = hand_l_xy.reshape(hand_l_xy.shape[0], -1)     # (T, 42)
    hr_flat = hand_r_xy.reshape(hand_r_xy.shape[0], -1)     # (T, 42)

    kps = np.concatenate([wb_flat, hl_flat, hr_flat], axis=1)  # (T, 282)
    return kps


def to_offset_keypoints(kps):
    """
    Convert absolute keypoint coordinates to offset (skeleton-relative) features.

    This makes the representation translation-invariant (independent of camera
    zoom, signer position, video resolution).

    For each sub-pose group:
      - Compute pairwise offsets between connected joints
      - Each node gets: (x_j - x_i, y_j - y_i, |dx|, |dy|) for each neighbor
      - Root node gets zeros

    Input:  (T, 282) — absolute keypoint coordinates
    Output: (T, 282) — offset features (same dimension)

    Adapted from GloFE (ACL 2023): "GloFE: Gloss-Free End-to-End Sign Language Translation"
    """
    # kps shape: (T, 282)
    T = kps.shape[0]
    offsets = np.zeros_like(kps, dtype=np.float32)

    for g in GROUPS:
        num_nodes = g['num_nodes']
        start = g['start']
        coords = kps[:, start:start + g['dim']].reshape(T, num_nodes, 2)  # (T, N, 2)

        if g['name'] == 'body':
            # Local node order (from BODY_IDX): 0 nose, 1 l_eye, 2 r_eye,
            # 3 l_ear, 4 r_ear, 5 l_shoulder, 6 r_shoulder, 7 l_elbow,
            # 8 r_elbow, 9 l_hip, 10 r_hip. Anatomical parent-child edges;
            # nose(0) is the root and gets zero offsets.
            parents = {
                1: 0, 2: 0,    # eyes ← nose
                3: 1, 4: 2,    # ears ← eyes
                5: 0, 6: 0,    # shoulders ← nose (proxy for neck)
                7: 5, 8: 6,    # elbows ← shoulders
                9: 5, 10: 6,   # hips ← shoulders
            }
        elif g['name'] == 'face':
            # Chain: each node's parent is the previous node
            parents = {i: i - 1 for i in range(1, num_nodes)}
        elif g['name'] in ('left_hand', 'right_hand'):
            # COCO-WholeBody hand: wrist(0) root; fingers are chains of 4:
            # thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20.
            parents = {}
            for base in (1, 5, 9, 13, 17):
                parents[base] = 0  # finger base ← wrist
                for j in range(base + 1, base + 4):
                    parents[j] = j - 1  # along the finger chain

        for child, parent in parents.items():
            if child < num_nodes and parent < num_nodes:
                dx = coords[:, child, 0] - coords[:, parent, 0]
                dy = coords[:, child, 1] - coords[:, parent, 1]
                offsets[:, start + child * 2] = dx
                offsets[:, start + child * 2 + 1] = dy

    return offsets


def load_npz_keypoints(npz_path, frame_start=None, frame_end=None, raw_arrays=False):
    """
    Load keypoints from a .npz file (extract.py format).

    Args:
        npz_path: path to .npz file
        frame_start: optional start frame
        frame_end: optional end frame
        raw_arrays: if True, also return raw wb_xy/hand_l/hand_r (before NaN→0)
                   for validity mask computation

    Returns:
        kps, scores, frame_idx [, wb_xy_raw, hand_l_raw, hand_r_raw]
    """
    if not os.path.exists(npz_path):
        if raw_arrays:
            return None, None, None, None, None, None
        return None, None, None

    try:
        d = np.load(npz_path)
    except Exception as e:
        print(f"[warn] Failed to load {npz_path}: {e}")
        if raw_arrays:
            return None, None, None, None, None, None
        return None, None, None

    # Slice by frame range if specified
    frame_idx = d['frame_idx']
    if frame_start is not None and frame_end is not None:
        mask = (frame_idx >= frame_start) & (frame_idx < frame_end)
        slice_idx = np.where(mask)[0]
    else:
        slice_idx = slice(None)

    # Check if person was detected
    person_found = d['person_found'][slice_idx]
    if not person_found.any():
        if raw_arrays:
            return None, None, None, None, None, None
        return None, None, None

    # Extract keypoints
    wb_xy = d['wb_xy'][slice_idx]         # (T, 133, 2)
    hand_l = d['hand_l_xy'][slice_idx]    # (T, 21, 2)
    hand_r = d['hand_r_xy'][slice_idx]    # (T, 21, 2)

    # Keep raw copies for validity computation
    if raw_arrays:
        wb_xy_raw = wb_xy.copy()
        hand_l_raw = hand_l.copy()
        hand_r_raw = hand_r.copy()

    # Handle NaN (missing hand detection)
    wb_xy = np.nan_to_num(wb_xy, nan=0.0)
    hand_l = np.nan_to_num(hand_l, nan=0.0)
    hand_r = np.nan_to_num(hand_r, nan=0.0)

    kps = assemble_keypoints(wb_xy, hand_l, hand_r)
    scores = np.nan_to_num(d['wb_score'][slice_idx], nan=0.0)

    if raw_arrays:
        return (kps.astype(np.float32), scores.astype(np.float32), frame_idx[slice_idx],
                wb_xy_raw, hand_l_raw, hand_r_raw)
    return kps.astype(np.float32), scores.astype(np.float32), frame_idx[slice_idx]


def extract_prosody_from_audio(audio_path, sr_target=16000, frame_start_sec=0.0, frame_end_sec=None):
    """Extract prosody features (F0, energy, duration) from audio file."""
    if not os.path.exists(audio_path):
        return None

    try:
        audio, sr = librosa.load(audio_path, sr=sr_target)
    except Exception as e:
        print(f"[warn] Failed to load audio {audio_path}: {e}")
        return None

    # Clip to segment
    if frame_start_sec is not None or frame_end_sec is not None:
        start_sample = int(frame_start_sec * sr_target) if frame_start_sec else 0
        end_sample = int(frame_end_sec * sr_target) if frame_end_sec else len(audio)
        segment = audio[start_sample:end_sample]
    else:
        segment = audio

    if len(segment) < sr_target * 0.05:  # at least 50ms
        return None

    # F0 extraction using librosa.
    # frame_length must span >= 2 periods of fmin (C2 ≈ 65 Hz → ~31 ms →
    # ~492 samples at 16 kHz). 1024 samples gives reliable estimates; the
    # previous value of 256 could not track F0 below ~125 Hz at all.
    f0, voiced_flag, voiced_probs = librosa.pyin(
        segment, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr_target, frame_length=1024, hop_length=160,
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    energy = librosa.feature.rms(y=segment, frame_length=1024, hop_length=160).squeeze()
    duration = np.ones_like(energy, dtype=np.float32)

    # Align to same length
    min_len = min(len(f0), len(energy))
    f0 = f0[:min_len]
    energy = energy[:min_len]
    duration = duration[:min_len]

    # Normalize: standardize voiced F0 (keeps intonation contour; per-utterance
    # max-normalization used previously erased pitch range differences),
    # max-normalize energy to [0, 1].
    voiced = f0 > 0
    if voiced.sum() > 1:
        mu, std = f0[voiced].mean(), f0[voiced].std()
        if std > 0:
            f0[voiced] = (f0[voiced] - mu) / std
    if energy.max() > 0: energy = energy / energy.max()

    prosody = np.stack([f0, energy, duration], axis=-1)
    return prosody.astype(np.float32)


# ============================================================
# Richer Pose Features (from KZ-RU SignFormer)
# ============================================================

def compute_velocity(kps, fps=50.0):
    """
    Compute velocity (first temporal derivative) of keypoints.

    Args:
        kps: (T, D) — keypoint coordinates (offset or absolute)
        fps: frames per second (for scaling)

    Returns:
        vel: (T, D) — velocity (centered difference, edge handling)
    """
    vel = np.zeros_like(kps, dtype=np.float32)
    if len(kps) < 2:
        return vel

    # Centered difference for interior frames
    vel[1:-1] = (kps[2:] - kps[:-2]) / 2.0
    # Forward/backward difference for edges
    vel[0] = kps[1] - kps[0]
    vel[-1] = kps[-1] - kps[-2]

    return vel


def compute_acceleration(kps, fps=50.0):
    """
    Compute acceleration (second temporal derivative) of keypoints.

    Args:
        kps: (T, D) — keypoint coordinates

    Returns:
        acc: (T, D) — acceleration
    """
    acc = np.zeros_like(kps, dtype=np.float32)
    if len(kps) < 3:
        return acc

    # Standard second derivative
    acc[1:-1] = kps[2:] - 2 * kps[1:-1] + kps[:-2]
    # Edge handling
    acc[0] = acc[1]
    acc[-1] = acc[-2]

    return acc


def compute_validity(kps_raw, wb_xy=None, hand_l_xy=None, hand_r_xy=None):
    """
    Compute validity mask: 1 where keypoints were detected, 0 where imputed.

    Args:
        kps_raw: (T, 282) — assembled keypoints (may have NaN→0 imputation)
        wb_xy: (T, 133, 2) — raw wholebody keypoints (before NaN→0)
        hand_l_xy: (T, 21, 2) — raw left hand
        hand_r_xy: (T, 21, 2) — raw right hand

    Returns:
        valid: (T, 282) — binary validity mask
    """
    T = kps_raw.shape[0]
    valid = np.ones((T, kps_raw.shape[1]), dtype=np.float32)

    if wb_xy is not None:
        # Check NaN in raw wb_xy
        wb_slice = wb_xy[:, BODY_IDX + FACE_IDX + LIPS_IDX, :]  # (T, 99, 2)
        wb_flat = wb_slice.reshape(T, -1)  # (T, 198)
        wb_valid = ~np.isnan(wb_flat)
        valid[:, :198] = wb_valid.astype(np.float32)

    if hand_l_xy is not None:
        hl_flat = hand_l_xy.reshape(T, -1)
        hl_valid = ~np.isnan(hl_flat)
        valid[:, 198:240] = hl_valid.astype(np.float32)

    if hand_r_xy is not None:
        hr_flat = hand_r_xy.reshape(T, -1)
        hr_valid = ~np.isnan(hr_flat)
        valid[:, 240:282] = hr_valid.astype(np.float32)

    return valid


# ============================================================
# Keypoint preprocessing (CSLRConformer, arXiv:2508.01791:
# spatial normalization + outlier filtering for keypoint CSLR)
# ============================================================

# l_shoulder is body joint 5, r_shoulder is body joint 6 in BODY_IDX order
_L_SHOULDER = slice(10, 12)
_R_SHOULDER = slice(12, 14)


def normalize_signer_scale(kps, eps=1e-6):
    """
    Per-clip scale normalization: divide all coordinates by the median
    shoulder width. Offsets are translation-invariant but NOT
    scale-invariant — the same sign performed closer to the camera (or by
    a broader signer) produced proportionally larger features. After this,
    coordinates are in "shoulder-width" units.

    Input/Output: (T, 282) absolute assembled keypoints.
    """
    ls = kps[:, _L_SHOULDER]
    rs = kps[:, _R_SHOULDER]
    valid = (np.abs(ls).sum(axis=1) > 0) & (np.abs(rs).sum(axis=1) > 0)
    if not valid.any():
        return kps
    widths = np.linalg.norm(ls[valid] - rs[valid], axis=1)
    scale = np.median(widths)
    if scale < eps:
        return kps
    return (kps / scale).astype(np.float32)


def remove_keypoint_spikes(kps, thresh=1.5):
    """
    Outlier filtering: a joint that jumps more than `thresh` (in
    shoulder-width units — apply AFTER normalize_signer_scale) away from
    BOTH temporal neighbours is a detector glitch, not motion. Replace it
    with the midpoint of its neighbours.

    Input/Output: (T, 282)
    """
    T = kps.shape[0]
    if T < 3:
        return kps
    coords = kps.reshape(T, -1, 2)
    prev_c, cur_c, next_c = coords[:-2], coords[1:-1], coords[2:]

    d_prev = np.linalg.norm(cur_c - prev_c, axis=-1)   # (T-2, N)
    d_next = np.linalg.norm(next_c - cur_c, axis=-1)

    # Only consider joints detected in all three frames (imputed zeros
    # would otherwise register as huge "jumps")
    detected = ((np.abs(prev_c).sum(-1) > 0) & (np.abs(cur_c).sum(-1) > 0)
                & (np.abs(next_c).sum(-1) > 0))
    spike = (d_prev > thresh) & (d_next > thresh) & detected  # (T-2, N)

    if spike.any():
        coords = coords.copy()
        mid = (prev_c + next_c) / 2.0
        inner = coords[1:-1]
        inner[spike] = mid[spike]
        coords[1:-1] = inner
        kps = coords.reshape(T, -1).astype(np.float32)
    return kps


def _normalize_group_local(coords, eps=1e-6):
    """
    Frame-wise local normalization of one keypoint group.

    coords: (T, N, 2) -> centred on the group's own per-frame bounding-box
    centre and divided by HALF ITS LARGER SIDE, so the group lands in
    [-1, 1] with aspect ratio preserved (a single isotropic scale, not
    independent x/y scales -- squashing a hand to a square would destroy
    the handshape this is meant to expose).

    Undetected joints arrive imputed to exactly (0, 0) from
    AsanDataset.__getitem__. They are excluded from the box and left at
    (0, 0) on the way out, matching what normalize_signer_scale already
    did (it maps 0 to 0). The validity channel is derived separately from
    the pre-imputation NaN arrays, so nothing is lost by that choice.
    """
    valid = ~((coords[..., 0] == 0) & (coords[..., 1] == 0))    # (T, N)
    masked = np.where(valid[..., None], coords, np.nan)

    # A frame where this whole group went undetected is an all-NaN slice.
    # That is expected on real data (an occluded hand), and is handled by
    # the `usable` mask below -- so silence the warning rather than let it
    # spam training logs once per such frame.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        lo = np.nanmin(masked, axis=1)      # (T, 2)
        hi = np.nanmax(masked, axis=1)      # (T, 2)

        centre = (lo + hi) / 2.0                                # (T, 2)
        half = np.nanmax(hi - lo, axis=1) / 2.0                 # (T,)

    # Frames with no detected joint (all-NaN) or a degenerate box: leave
    # the frame untouched rather than dividing by ~0 and exploding it.
    usable = np.isfinite(half) & (half > eps) & np.isfinite(centre).all(axis=1)
    if not usable.any():
        return coords

    out = coords.copy()
    c = np.where(usable[:, None], centre, 0.0)
    h = np.where(usable, half, 1.0)
    out = (out - c[:, None, :]) / h[:, None, None]

    out[~usable] = coords[~usable]      # untouched frames keep raw values
    out[~valid] = 0.0                   # re-assert the imputation sentinel
    return out


def _body_box(kps, eps):
    """
    SignSpace body box: centre = shoulder midpoint, half-side = 1.5x the
    shoulder distance (so the full side is 3x, per the paper).

    Returns (centre (T,2), half (T,)) or None when no frame has both
    shoulders. Frame-wise as the paper specifies, but a frame missing a
    shoulder falls back to the clip median so one dropped detection cannot
    rescale that frame into a different space from its neighbours.
    """
    ls, rs = kps[:, _L_SHOULDER], kps[:, _R_SHOULDER]
    have = (np.abs(ls).sum(axis=1) > 0) & (np.abs(rs).sum(axis=1) > 0)
    if not have.any():
        return None

    centre = (ls + rs) / 2.0
    shoulder = np.linalg.norm(ls - rs, axis=1)
    median_sh = np.median(shoulder[have])
    if median_sh < eps:
        return None

    shoulder = np.where(have & (shoulder > eps), shoulder, median_sh)
    centre[~have] = centre[have].mean(axis=0)
    return centre, 1.5 * shoulder


def signspace_global(kps, eps=1e-6):
    """
    Body-centred SignSpace box applied to EVERY joint, hands and face
    included -- an isotropic, signer-scale-invariant frame in which a
    hand's POSITION still means something.

    This is the anchor half of the scheme. Use it for the absolute
    coordinate channel, so that normalizing handshape locally (below)
    does not also destroy where in the signing space the hand was.

    Input/Output: (T, 282).
    """
    box = _body_box(kps, eps)
    if box is None:
        return kps
    centre, half = box

    out = kps.copy().astype(np.float32)
    T = out.shape[0]
    for g in GROUPS:
        start, dim, n = g['start'], g['dim'], g['num_nodes']
        coords = out[:, start:start + dim].reshape(T, n, 2)
        valid = ~((coords[..., 0] == 0) & (coords[..., 1] == 0))
        coords = (coords - centre[:, None, :]) / half[:, None, None]
        coords[~valid] = 0.0
        out[:, start:start + dim] = coords.reshape(T, dim)
    return out


def signspace_normalize(kps, eps=1e-6,
                        local_groups=('face', 'left_hand', 'right_hand')):
    """
    SignSpace normalization (Exploring Pose-based SLT, arXiv:2507.01532).

    Two different treatments, which is the whole point:
      * BODY -- global: the body box from _body_box, mapped to [-1, 1].
        Keeps the spatial relationship between body parts, which is
        linguistically meaningful in sign.
      * FACE / LEFT_HAND / RIGHT_HAND -- local and independent, each to its
        own per-frame box (see _normalize_group_local), so the same
        handshape gives the same features regardless of the signer's hand
        size or distance from the camera.

    IMPORTANT -- this output is for the OFFSET/bone channel only. Local
    normalization deliberately discards each hand's position, and our body
    graph cannot give it back (_map_coco_to_unisign_body fills the wrist
    slots with repeated ELBOW values, so the encoder's hand-anchor fusion
    is reading a pseudo-wrist). Pair this with signspace_global() for the
    absolute-coordinate channel, which retains the anchor. Callers that
    use only this function will lose signing location entirely.

    Published effect, How2Sign BLEU-4: none 0.73, frame-wise 1.13,
    SignSpace 2.17 -- their largest single ablation effect. That is
    evidence for testing it here, not a predicted KRSL gain.

    Input/Output: (T, 282).
    """
    out = signspace_global(kps, eps=eps)
    T = out.shape[0]
    for g in GROUPS:
        if g['name'] not in local_groups:
            continue
        start, dim, n = g['start'], g['dim'], g['num_nodes']
        coords = out[:, start:start + dim].reshape(T, n, 2)
        out[:, start:start + dim] = _normalize_group_local(
            coords, eps=eps).reshape(T, dim)
    return out


# Assembled (T, 282) layout slots used by forearm_offsets / real wrists.
_L_ELBOW = slice(14, 16)       # body local node 7 (COCO 7, left elbow)
_R_ELBOW = slice(16, 18)       # body local node 8 (COCO 8, right elbow)
_L_HAND_ROOT = slice(198, 200)  # left-hand group node 0 = left wrist
_R_HAND_ROOT = slice(240, 242)  # right-hand group node 0 = right wrist


def forearm_offsets(kps):
    """
    Real wrist bone vectors: wrist - elbow, per side.

    kps must be GLOBAL-frame coordinates (legacy shoulder-scaled, or
    signspace_global) -- the same frame the body group's other bone offsets
    were computed in. Never pass signspace_normalize output: its hands sit
    in their own local boxes, so wrist - elbow there mixes two spaces.

    Why this exists: to_offset_keypoints leaves every hand ROOT at zero (it
    has no parent inside the hand group), and _map_coco_to_unisign_body had
    no wrist source, so it repeated elbow features into the wrist slots. The
    encoder's hand-anchor fusion was therefore reading a pseudo-wrist. The
    wrist is the hand detector's root joint; this recovers the forearm in
    consistent units so it can be written into the (otherwise always-zero)
    hand-root offset slots.

    Returns (left (T, 2), right (T, 2)); zero where wrist or elbow undetected
    (imputed to exactly (0, 0)), so a missing hand cannot turn into a huge
    bogus "elbow to origin" vector.
    """
    out = []
    for wrist_sl, elbow_sl in ((_L_HAND_ROOT, _L_ELBOW), (_R_HAND_ROOT, _R_ELBOW)):
        w, e = kps[:, wrist_sl], kps[:, elbow_sl]
        ok = (np.abs(w).sum(axis=1) > 0) & (np.abs(e).sum(axis=1) > 0)
        out.append(np.where(ok[:, None], w - e, 0.0).astype(np.float32))
    return out[0], out[1]


def preprocess_keypoints(kps, despike_thresh=1.5, signspace=False):
    """
    Standard preprocessing: signer-scale normalization + spike removal,
    optionally followed by SignSpace normalization.

    Order matters. remove_keypoint_spikes' threshold is expressed in
    shoulder-width units, so it must run while the coordinates are still in
    those units -- i.e. AFTER normalize_signer_scale and BEFORE
    signspace_normalize, which re-derives its own per-group boxes and
    would otherwise silently change what `despike_thresh` means.
    """
    kps = normalize_signer_scale(kps)
    kps = remove_keypoint_spikes(kps, thresh=despike_thresh)
    if signspace:
        kps = signspace_normalize(kps)
    return kps


def standardize_absolute(kps_abs):
    """
    Per-clip standardization of absolute keypoint coordinates.

    Raw pixel coordinates vary with resolution and framing; standardizing
    per clip (zero mean, unit std over detected joints) keeps the GLOBAL
    spatial information the offset features lack — where the hands are
    relative to the body and how they travel over the clip — without
    letting pixel magnitudes dominate the projection layer. Imputed
    (all-zero) joints stay zero.

    Input/Output: (T, 282)
    """
    T = kps_abs.shape[0]
    coords = kps_abs.reshape(T, -1, 2)
    valid = np.abs(coords).sum(axis=-1) > 0  # (T, N)
    out = np.zeros_like(coords, dtype=np.float32)
    if valid.any():
        pts = coords[valid]                      # (M, 2)
        mean = pts.mean(axis=0)
        std = pts.std(axis=0)
        std[std < 1e-6] = 1.0
        out[valid] = (coords[valid] - mean) / std
    return out.reshape(T, -1)


def enrich_keypoints(kps, wb_xy=None, hand_l_xy=None, hand_r_xy=None,
                     kps_abs=None, joint_scores=None):
    """
    Build enriched keypoint features (dual-coords):
    offset + absolute + velocity + acceleration + validity.

    Input:  kps — (T, 282) offset keypoints
            kps_abs — (T, 282) absolute keypoints, pre-offset (optional;
                zeros are used if unavailable so the output dim is stable)
            wb_xy / hand_l_xy / hand_r_xy — raw arrays (NaN = undetected)
                for the binary validity mask
            joint_scores — (T, 282) continuous per-joint confidence in
                [0, 1] (optional). When provided it REPLACES the binary
                validity — score-aware processing per Uni-Sign
                (arXiv:2501.15187) gives the encoder a graded signal for
                how much to trust each joint instead of a 0/1 bit.
    Output: (T, 1410) — [offset, absolute, velocity, acceleration, validity]
            = 5 × 282
    """
    # Offset features (input kps are already offset-encoded)
    offset = kps

    # Absolute coordinates (per-clip standardized)
    if kps_abs is not None:
        absolute = standardize_absolute(kps_abs)
    else:
        absolute = np.zeros_like(offset, dtype=np.float32)

    # Velocity and acceleration (on offset features)
    vel = compute_velocity(kps)
    acc = compute_acceleration(kps)

    # Validity: continuous confidence when available, else binary detection
    if joint_scores is not None:
        valid = np.nan_to_num(joint_scores, nan=0.0).astype(np.float32)
        valid = np.clip(valid, 0.0, 1.0)
    else:
        valid = compute_validity(kps, wb_xy, hand_l_xy, hand_r_xy)

    # Concatenate: (T, 282) × 5 → (T, 1410)
    enriched = np.concatenate([offset, absolute, vel, acc, valid], axis=1)
    return enriched.astype(np.float32)


def assemble_joint_scores(wb_score, hand_l_score=None, hand_r_score=None):
    """
    Expand per-joint confidence scores into the assembled 282-dim layout
    (each joint's score duplicated over its x and y channels).

    Args:
        wb_score: (T, 133) — COCO-WholeBody scores
        hand_l_score / hand_r_score: (T, 21) — dedicated hand-model scores;
            fall back to the wholebody hand slices (91-111 / 112-132)

    Returns: (T, 282)
    """
    wb_slice = wb_score[:, BODY_IDX + FACE_IDX + LIPS_IDX]     # (T, 99)
    hl = hand_l_score if hand_l_score is not None else wb_score[:, 91:112]
    hr = hand_r_score if hand_r_score is not None else wb_score[:, 112:133]
    scores = np.concatenate([
        np.repeat(wb_slice, 2, axis=1),   # (T, 198)
        np.repeat(hl, 2, axis=1),         # (T, 42)
        np.repeat(hr, 2, axis=1),         # (T, 42)
    ], axis=1)
    return scores.astype(np.float32)


def quantile_transform_scores(joint_scores, table, floor=0.05):
    """
    Map raw detector scores (T, 282) through a TRAIN-fitted empirical CDF
    per assembled group (scripts/fit_score_quantiles.py).

    Detected joints land in [floor, 1]; undetected joints (NaN or <= 0) stay at
    exactly 0, so detection remains distinguishable from the weakest detection.
    Replaces clip(score, 0, 1), which pinned 99.6-100% of joints to 1.0.

    Rank-normalized detector response -- NOT a calibrated probability.
    """
    out = np.zeros_like(joint_scores, dtype=np.float32)
    probs = np.asarray(table['_meta']['probs'], dtype=np.float64)
    for g in ('body', 'face', 'hands'):
        a, b = table[g]['slice']
        knots = np.asarray(table[g]['knots'], dtype=np.float64)
        raw = joint_scores[:, a:b]
        det = np.isfinite(raw) & (raw > 0)
        cdf = np.interp(np.where(det, raw, knots[0]), knots, probs)
        out[:, a:b] = np.where(det, floor + (1.0 - floor) * cdf, 0.0)
    return out


# ------------------------------------------------------------------------
# Upstream Uni-Sign pose preprocessing (E3 arm B)
# Port of load_part_kp / crop_scale from the official repository:
# https://github.com/ZechengLi19/Uni-Sign/blob/main/datasets.py
# ------------------------------------------------------------------------
_US_BODY = [0] + list(range(3, 11))          # nose, ears, shoulders, elbows, WRISTS
_US_FACE = list(range(23, 40))[::2] + list(range(83, 91)) + [53]   # 9 jaw + 8 mouth + nose tip
_US_THR = 0.3
UNISIGN_NODES = (('body', 9), ('left', 21), ('right', 21), ('face_all', 18))
UNISIGN_DIM = 69 * 3                          # 207 = (9+21+21+18) nodes x (x, y, score)


def unisign_part_features(wb_xy, wb_score, thr=_US_THR, hand_ratio=None):
    """
    Exact port of Uni-Sign's pose preprocessing, so arm B feeds the pretrained
    weights inputs with the meaning they were trained on.

    Upstream semantics, reproduced deliberately (including the odd parts):
      * body: 9 joints with REAL wrists, absolute coordinates; ONE clip-level
        box over all frames' confident joints (score > thr), side = larger
        extent, mapped to [-1, 1].
      * hands: coordinates relative to that hand's own root (wrist) per frame,
        divided by the body box scale.
      * face_all: 18 points relative to the nose tip, divided by body scale.
      * channels (x, y, score). np.clip(result, -1, 1) runs over the WHOLE
        array, so the score channel is also capped at 1 upstream -- raw
        RTMPose scores are >= 1 for ~100% of joints, so upstream's confidence
        channel is saturated too. Not "fixed" here: this arm reproduces
        upstream, it does not improve it.
      * any joint with score <= thr is zeroed in all three channels.
      * too few confident body joints (< 4) or zero extent => whole clip zero.

    wb_xy (T, 133, 2) may contain NaN for undetected joints; those get score 0.
    Returns (T, 207) float32: body, left, right, face_all, each (x, y, score).

    hand_ratio (E3 arm D, SignSpace-style hand scaling): None reproduces
    upstream exactly. Given a float r -- the TRAIN-fitted median of
    (hand extent / body box scale), see scripts/fit_hand_ratio.py -- each hand
    is rescaled PER FRAME so its extent equals r body-units:
        hand_body_units * (r / this_frame_ratio)
    Wrist-relative coordinates and body-box units are kept, so values stay in
    the range the pretrained Uni-Sign weights expect; only the signer's hand
    size / camera distance is removed. Plain SignSpace (stretch every hand to
    [-1, 1]) was rejected for exactly that reason: E3 arm B showed a mismatch
    with pretrained input semantics costs more than normalization gains.
    Frames with < 2 confident hand joints or zero extent fall back to upstream.
    """
    xy = np.asarray(wb_xy, dtype=np.float64)
    sc = np.asarray(wb_score, dtype=np.float64).copy()
    missing = np.isnan(xy).any(axis=-1) | ~np.isfinite(sc)
    sc[missing] = 0.0
    xy = np.nan_to_num(xy, nan=0.0)
    T = xy.shape[0]

    def part(idx):
        return np.concatenate([xy[:, idx], sc[:, idx, None]], axis=-1)   # (T, N, 3)

    body = part(_US_BODY)
    left = part(list(range(91, 112)));  left[..., :2] -= left[:, :1, :2]
    right = part(list(range(112, 133))); right[..., :2] -= right[:, :1, :2]
    face = part(_US_FACE);              face[..., :2] -= face[:, -1:, :2]

    out = np.zeros((T, 69, 3), dtype=np.float64)
    valid = body[body[..., 2] > thr][:, :2]
    if len(valid) >= 4:
        xmin, xmax = valid[:, 0].min(), valid[:, 0].max()
        ymin, ymax = valid[:, 1].min(), valid[:, 1].max()
        scale = max(xmax - xmin, ymax - ymin)
        if scale > 0:
            xs, ys = (xmin + xmax - scale) / 2, (ymin + ymax - scale) / 2
            body[..., :2] = ((body[..., :2] - [xs, ys]) / scale - 0.5) * 2
            parts = [body]
            for p_ in (left, right, face):
                p_[..., :2] = p_[..., :2] / scale
                parts.append(p_)
            if hand_ratio is not None:
                for p_ in (left, right):
                    _rescale_hand_per_frame(p_, hand_ratio, thr)
            o = 0
            for p_ in parts:
                p_ = np.clip(p_, -1, 1)              # clips score too, as upstream
                p_[p_[..., 2] <= thr] = 0
                out[:, o:o + p_.shape[1]] = p_
                o += p_.shape[1]
    return out.reshape(T, UNISIGN_DIM).astype(np.float32)


def _hand_extent(hand, thr):
    """Per-frame extent (larger side) of confident joints; hand is (T, 21, 3)."""
    ok = hand[..., 2] > thr                                   # (T, 21)
    xy = np.where(ok[..., None], hand[..., :2], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)       # all-NaN frames
        ext = np.nanmax(np.nanmax(xy, axis=1) - np.nanmin(xy, axis=1), axis=-1)
    ext[ok.sum(axis=1) < 2] = np.nan
    return ext                                               # (T,), NaN = unusable


def _rescale_hand_per_frame(hand, ratio, thr):
    """In place: scale each frame so the hand's extent equals `ratio` body-units."""
    ext = _hand_extent(hand, thr)
    usable = np.isfinite(ext) & (ext > 1e-6)
    factor = np.where(usable, ratio / np.where(usable, ext, 1.0), 1.0)
    hand[..., :2] *= factor[:, None, None]


def unpack_unisign_parts(kps):
    """(B, T, 207) torch tensor -> {'body','left','right','face_all'}: (B, T, N, 3)."""
    B, T, _ = kps.shape
    v = kps.reshape(B, T, 69, 3)
    parts, o = {}, 0
    for name, n in UNISIGN_NODES:
        parts[name] = v[:, :, o:o + n]
        o += n
    return parts


def ENRICHED_DIM():
    """Total enriched keypoint dimension: 5 × 282 = 1410 (dual-coords)."""
    return 5 * KEYPOINT_DIM


def resample_prosody(prosody, target_len):
    """
    Linearly resample a prosody sequence to `target_len` frames.

    Prosody is extracted at 100 Hz (10 ms hop) while keypoints are at the
    video frame rate (25–50 fps). Truncating to min length (the old
    behaviour) silently dropped the second half of the audio's prosody;
    resampling keeps the full contour aligned with the keypoints.

    Args:
        prosody: (T_src, C) numpy array
        target_len: desired number of frames

    Returns:
        (target_len, C) numpy array
    """
    prosody = np.asarray(prosody, dtype=np.float32)
    t_src = len(prosody)
    if t_src == 0 or target_len <= 0:
        return np.zeros((max(target_len, 0), prosody.shape[-1] if prosody.ndim > 1 else 1),
                        dtype=np.float32)
    if t_src == target_len:
        return prosody
    src_x = np.linspace(0.0, 1.0, t_src)
    tgt_x = np.linspace(0.0, 1.0, target_len)
    out = np.stack(
        [np.interp(tgt_x, src_x, prosody[:, c]) for c in range(prosody.shape[1])],
        axis=1,
    )
    return out.astype(np.float32)


def _nearest_indices(t_src, target_len):
    """
    Index map for nearest-neighbor resampling: unlike resample_prosody's
    linear interpolation (fine for continuous signals like F0/energy),
    hand-crop images, validity booleans, and reference-point coordinates
    can't be blended between two source frames -- nearest-neighbor is the
    only sound choice.
    """
    if t_src == 0 or target_len <= 0:
        return np.zeros((max(target_len, 0),), dtype=np.int64)
    if t_src == target_len:
        return np.arange(t_src, dtype=np.int64)
    src_x = np.linspace(0.0, t_src - 1, target_len)
    return np.round(src_x).astype(np.int64).clip(0, t_src - 1)


def resample_hand_crops(crops, target_len):
    """
    Nearest-neighbor resample a hand-crop image sequence to `target_len`
    frames. Extraction already downsamples video at the pose frame rate,
    so this is normally a no-op (same early-return-equivalent behavior as
    resample_prosody when lengths already match) -- it only actually
    resamples for the rare max_frames-truncation / independent-decoding
    mismatch case.

    Args:
        crops: (T_src, H, W, 3) uint8 array
        target_len: desired number of frames

    Returns:
        (target_len, H, W, 3) uint8 array
    """
    crops = np.asarray(crops)
    if crops.ndim != 4:
        raise ValueError(f"expected (T,H,W,3), got shape {crops.shape}")
    if len(crops) == 0 or target_len <= 0:
        return np.zeros((max(target_len, 0),) + crops.shape[1:], dtype=crops.dtype)
    idx = _nearest_indices(len(crops), target_len)
    return crops[idx]


def resample_hand_meta(ref, valid, score, target_len):
    """
    Nearest-neighbor resample a hand's per-frame reference points, validity
    mask, and confidence scores to `target_len` frames (same alignment
    convention as resample_hand_crops -- must use the SAME index map so a
    given output frame's crop/ref/valid/score all come from the same
    source frame).

    Args:
        ref: (T_src, 2) float32 -- normalized [-1,1] wrist reference point
        valid: (T_src,) bool -- whether that frame's crop is trustworthy
        score: (T_src,) float32 -- mean keypoint confidence
        target_len: desired number of frames

    Returns:
        (ref, valid, score) each resampled to length target_len
    """
    ref = np.asarray(ref, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    score = np.asarray(score, dtype=np.float32)
    t_src = len(ref)
    if t_src == 0 or target_len <= 0:
        n = max(target_len, 0)
        return (np.zeros((n, 2), dtype=np.float32),
                np.zeros((n,), dtype=bool),
                np.zeros((n,), dtype=np.float32))
    idx = _nearest_indices(t_src, target_len)
    return ref[idx], valid[idx], score[idx]
