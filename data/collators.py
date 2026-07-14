"""
Shared collate function for pose+text datasets.

KhabarKzCollator, InformburoCollator and KazSignCollator were three
copy-pasted implementations of the same logic; they now all alias
PoseTextCollator (kept as subclasses so existing imports keep working).
"""
import torch


class PoseTextCollator:
    """
    Collates variable-length (keypoints, text_ids, prosody) samples.

    Returns None if the whole batch is blank (failed loads).
    """

    def __init__(self, max_text_len=500):
        self.max_text_len = max_text_len

    def __call__(self, batch):
        # Filter out blank samples
        batch = [b for b in batch if b['input_length'] > 0]
        if not batch:
            return None

        # Sort by keypoint length (descending) for efficient padding
        batch.sort(key=lambda x: x['input_length'], reverse=True)

        keypoint_lengths = [b['input_length'] for b in batch]
        max_kp_len = max(keypoint_lengths)

        # Pad keypoints
        kps_padded = []
        for b in batch:
            kps = b['keypoints']
            if kps.shape[0] < max_kp_len:
                pad = torch.zeros(max_kp_len - kps.shape[0], kps.shape[1])
                kps = torch.cat([kps, pad], dim=0)
            kps_padded.append(kps)
        kps_tensor = torch.stack(kps_padded, dim=0)  # (B, T, D)

        # Pad text ids
        text_ids_list = []
        text_lengths = []
        for b in batch:
            tids = b.get('text_ids')
            if tids is not None and len(tids) > 0:
                tid = tids[:self.max_text_len]
                text_ids_list.append(tid.tolist())
                text_lengths.append(len(tid))
            else:
                text_ids_list.append([])
                text_lengths.append(0)

        max_text_len = max(text_lengths) if text_lengths else 1
        text_ids_padded = torch.zeros(len(batch), max(max_text_len, 1), dtype=torch.long)
        for i, tid in enumerate(text_ids_list):
            if tid:
                text_ids_padded[i, :len(tid)] = torch.tensor(tid)

        # Stack prosody (only if every sample in the batch has it)
        prosody_list = [b.get('prosody') for b in batch]
        prosody_tensor = None
        if all(p is not None for p in prosody_list) and prosody_list:
            pros_padded = []
            for p in prosody_list:
                if p.shape[0] < max_kp_len:
                    pad = torch.zeros(max_kp_len - p.shape[0], p.shape[1])
                    p = torch.cat([p, pad], dim=0)
                pros_padded.append(p)
            prosody_tensor = torch.stack(pros_padded, dim=0).transpose(1, 2)  # (B, C, T)

        # Stack RGB features (only if every sample in the batch has them).
        # Kept as (B, T, rgb_dim) -- NOT transposed like prosody -- since
        # this is meant to be fused with pose_emb (B, T, 768) per frame,
        # not consumed as an audio-style (B, C, T) channel stack.
        rgb_list = [b.get('rgb') for b in batch]
        rgb_tensor = None
        if all(r is not None for r in rgb_list) and rgb_list:
            rgb_padded = []
            for r in rgb_list:
                if r.shape[0] < max_kp_len:
                    pad = torch.zeros(max_kp_len - r.shape[0], r.shape[1])
                    r = torch.cat([r, pad], dim=0)
                rgb_padded.append(r)
            rgb_tensor = torch.stack(rgb_padded, dim=0)  # (B, T, rgb_dim)

        return {
            'keypoints': kps_tensor,                          # (B, T, D)
            'input_lengths': torch.tensor(keypoint_lengths),  # (B,)
            'text_ids': text_ids_padded,                      # (B, L)
            'text_lengths': torch.tensor(text_lengths),       # (B,)
            'prosody': prosody_tensor,                        # (B, C, T) or None
            'rgb': rgb_tensor,                                # (B, T, rgb_dim) or None
            'texts': [b['text'] for b in batch],
            'clip_ids': [b['clip_id'] for b in batch],
        }
