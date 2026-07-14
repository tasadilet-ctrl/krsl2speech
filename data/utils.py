"""
Shared utilities for loading keypoints and extracting prosody.
"""
import os
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


def preprocess_keypoints(kps, despike_thresh=1.5):
    """Standard preprocessing: signer-scale normalization + spike removal."""
    kps = normalize_signer_scale(kps)
    kps = remove_keypoint_spikes(kps, thresh=despike_thresh)
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
