"""
Proper evaluation of a Phase-1 checkpoint on the asan dev/test split.

Reports:
  - WER (editdistance) and, if sacrebleu is installed, BLEU and chrF2
  - content-word recall: fraction of reference content words (len >= 4)
    that appear in the hypothesis — separates "fluent but wrong story"
    (recall ≈ chance) from genuine partial translation
  - a sample of (REF, HYP) pairs

Usage:
  PYTHONPATH=. python scripts/evaluate_phase1.py \
      --config configs/config.yaml \
      --ckpt output/phase1_v3/phase1_mt5_best.pth \
      --use-enriched --split val --max-clips 500
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.utils.data import DataLoader

from data.asan_dataset import AsanDataset
from data.utils import ENRICHED_DIM, KEYPOINT_DIM
from models.unisign_encoder import KeypointEncoder
from train.train_encoder_mt5 import UniSignMT5, SimpleCollator


def content_word_recall(refs, hyps, min_len=4):
    got, total = 0, 0
    for r, h in zip(refs, hyps):
        h_words = set(h.lower().split())
        r_words = [w for w in r.lower().split() if len(w) >= min_len]
        total += len(r_words)
        got += sum(w in h_words for w in r_words)
    return got / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--use-enriched', action='store_true')
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--max-clips', type=int, default=500)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--num-beams', type=int, default=4)
    ap.add_argument('--show', type=int, default=10)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    from utils.paths import apply_env_overrides
    cfg = apply_env_overrides(cfg)
    acfg = cfg['paths']['asan']

    # ---- Model ----
    input_dim = ENRICHED_DIM() if args.use_enriched else KEYPOINT_DIM
    encoder = KeypointEncoder(hidden_dim=cfg['model']['d_model'],
                              input_dim=input_dim)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model = UniSignMT5(encoder=encoder, lang="Kazakh").to(device)
    model.encoder.load_state_dict(ckpt['encoder'])
    if 'pose_norm' in ckpt:
        model.pose_norm.load_state_dict(ckpt['pose_norm'])
    if 'mt5' in ckpt:
        model.mt5.load_state_dict(ckpt['mt5'])
        print("[eval] loaded fine-tuned MT5")
    elif 'mt5_lora' in ckpt:
        from peft import LoraConfig, get_peft_model, TaskType
        lc = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32,
                        target_modules=["q", "v"], bias="none")
        model.mt5 = get_peft_model(model.mt5, lc).to(device)
        model.mt5.load_state_dict(ckpt['mt5_lora'], strict=False)
        print("[eval] loaded MT5 LoRA adapters")
    else:
        print("[eval] WARNING: no MT5 weights in checkpoint (base model)")
    model.eval()

    # ---- Data ----
    ds = AsanDataset(root=acfg['root'], sources=acfg.get('sources'),
                     lang=acfg.get('lang', 'kz'), split=args.split,
                     downsample_every=acfg.get('downsample_every', 1),
                     use_enriched=args.use_enriched)
    collator = SimpleCollator(mt5_tokenizer=model.mt5_tokenizer,
                              max_text_tokens=128)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collator)

    refs, hyps = [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            kps = batch['keypoints'].to(device)
            lengths = batch['input_lengths'].to(device)
            out = model.generate(kps, input_lengths=lengths,
                                 num_beams=args.num_beams)
            hyps.extend(out)
            refs.extend(batch['texts'])
            print(f"\r{len(refs)} clips", end='', flush=True)
            if len(refs) >= args.max_clips:
                break
    print()

    # ---- Guard: nothing collected ----
    if not refs or not hyps:
        print("\n[eval] ERROR: 0 clips evaluated. Likely causes:")
        print(f"  - split '{args.split}' annotation file missing for one or "
              f"more sources under {acfg['root']} "
              f"(look for 'WARNING: missing ...' lines above)")
        print("  - all pose .pkl files failed to load (look for '[warn] "
              "Failed to load ...' lines above)")
        print(f"  Dataset reported {len(ds)} clips before generation.")
        sys.exit(1)

    # ---- Metrics ----
    try:
        import editdistance
        d = sum(editdistance.eval(r.split(), h.split()) for r, h in zip(refs, hyps))
        n = sum(len(r.split()) for r in refs)
        print(f"WER:   {d / max(n, 1):.4f}")
    except ImportError:
        print("WER:   (pip install editdistance)")

    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        chrf = sacrebleu.corpus_chrf(hyps, [refs])
        print(f"BLEU:  {bleu.score:.2f}")
        print(f"chrF2: {chrf.score:.2f}")
    except ImportError:
        print("BLEU/chrF: (pip install sacrebleu)")

    recall = content_word_recall(refs, hyps)
    # Chance baseline: recall of ref words against SHUFFLED hypotheses
    shuffled = hyps[1:] + hyps[:1]
    chance = content_word_recall(refs, shuffled)
    print(f"content-word recall: {recall:.3f}  (chance≈{chance:.3f}; "
          f"recall >> chance → real visual grounding)")

    print(f"\n--- {args.show} samples ---")
    for r, h in list(zip(refs, hyps))[:args.show]:
        print(f"REF: {r[:100]}")
        print(f"HYP: {h[:100]}\n")


if __name__ == '__main__':
    main()
