"""Batch collation shared by train_encoder_mt5.py, train_min.py and the
scoring scripts. Moved verbatim out of train_encoder_mt5.py (2026-10-08) so
the minimal trainer does not import the legacy one."""
import torch


class SimpleCollator:
    """Collator for MT5 training. Tokenizes text here (not in forward!) for speed."""

    def __init__(self, mt5_tokenizer=None, max_text_tokens=128):
        self.mt5_tokenizer = mt5_tokenizer
        self.max_text_tokens = max_text_tokens

    def __call__(self, batch):
        valid = [b for b in batch if b is not None and b.get('input_length', 0) > 0]
        if not valid:
            return None

        # Filter samples with text
        valid = [b for b in valid if b.get('text') and b['text'].strip()]
        if not valid:
            return None

        # Sort by keypoint length (descending) for efficient padding
        valid.sort(key=lambda x: x['input_length'], reverse=True)

        # Pad keypoints
        kps = [b['keypoints'] for b in valid]
        max_t = max(k.shape[0] for k in kps)
        input_lengths = torch.tensor([k.shape[0] for k in kps], dtype=torch.long)

        kps_padded = torch.zeros(len(valid), max_t, kps[0].shape[1], dtype=torch.float32)
        for i, k in enumerate(kps):
            kps_padded[i, :k.shape[0], :] = k

        # Pad RGB features (only if every sample in the batch has them --
        # AsanDataset.__getitem__ already skips clips missing an rgb file
        # entirely when load_rgb=True, via _blank_sample(), so partial
        # coverage within a kept batch shouldn't normally happen once
        # extraction covers all clips).
        rgb_list = [b.get('rgb') for b in valid]
        rgb_padded = None
        if all(r is not None for r in rgb_list) and rgb_list:
            rgb_padded = torch.zeros(len(valid), max_t, rgb_list[0].shape[1], dtype=torch.float32)
            for i, r in enumerate(rgb_list):
                rgb_padded[i, :r.shape[0], :] = r

        # Prosody ([F0, energy], (T,2)) for the prosody-as-supervision
        # ablation. Same all-or-nothing convention as rgb above:
        # AsanDataset returns a blank sample for clips missing prosody, so a
        # partial batch shouldn't occur once extraction covers the corpus.
        prosody_list = [b.get('prosody') for b in valid]
        prosody_padded = None
        if all(p is not None for p in prosody_list) and prosody_list:
            prosody_padded = torch.zeros(len(valid), max_t,
                                         prosody_list[0].shape[1],
                                         dtype=torch.float32)
            for i, p_ in enumerate(prosody_list):
                prosody_padded[i, :p_.shape[0], :] = p_

        # Pad hand-crop fields for Prior-Guided Fusion (only if every
        # sample in the batch has them -- same "skip clips missing the
        # file entirely via _blank_sample()" convention as rgb above, so
        # partial coverage within a kept batch shouldn't normally happen
        # once extraction covers all clips).
        hand_crops_list = [b.get('hand_crops') for b in valid]
        hand_crops_padded = hand_ref_padded = hand_valid_padded = hand_score_padded = None
        if all(h is not None for h in hand_crops_list) and hand_crops_list:
            def _pad_time(tensors, extra_shape, dtype):
                out = torch.zeros((len(valid), max_t) + extra_shape, dtype=dtype)
                for i, t in enumerate(tensors):
                    out[i, :t.shape[0]] = t
                return out

            hand_crops_padded = _pad_time(hand_crops_list, (2, 112, 112, 3), torch.uint8)
            hand_ref_padded = _pad_time([b['hand_ref'] for b in valid], (2, 2), torch.float32)
            hand_valid_padded = _pad_time([b['hand_valid'] for b in valid], (2,), torch.bool)
            hand_score_padded = _pad_time([b['hand_score'] for b in valid], (2,), torch.float32)

        # Tokenize text ONCE here (not every forward pass)
        texts = [b['text'].strip() for b in valid]
        if self.mt5_tokenizer is not None:
            label_tokens = self.mt5_tokenizer(
                texts, padding="longest", truncation=True,
                max_length=self.max_text_tokens, return_tensors="pt",
            )
            label_ids = label_tokens['input_ids']
            label_ids[label_ids == self.mt5_tokenizer.pad_token_id] = -100
            label_attn = label_tokens['attention_mask']
        else:
            label_ids = None
            label_attn = None

        return {
            'keypoints': kps_padded,
            'input_lengths': input_lengths,
            'label_ids': label_ids,            # (B, L_text) — -100 for pads
            'label_attn_mask': label_attn,     # (B, L_text)
            'rgb': rgb_padded,                  # (B, T, rgb_dim) or None
            'prosody': prosody_padded,          # (B, T, 2) or None
            'hand_crops': hand_crops_padded,    # (B, T, 2, 112, 112, 3) uint8 or None
            'hand_ref': hand_ref_padded,        # (B, T, 2, 2) or None
            'hand_valid': hand_valid_padded,    # (B, T, 2) bool or None
            'hand_score': hand_score_padded,    # (B, T, 2) or None
            'texts': texts,                     # raw strings for generation eval
            # In the SAME (length-sorted) order as every tensor above. Any
            # consumer attaching per-clip metadata must key on these, never on
            # dataset order: this collator sorts by length and drops invalid
            # samples, so batch position != dataset position.
            'clip_ids': [b.get('clip_id', '') for b in valid],
        }
