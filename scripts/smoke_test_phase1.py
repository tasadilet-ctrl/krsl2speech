"""
CPU smoke test for the Phase-1 (Uni-Sign encoder + mT5) pipeline.

Purpose: catch shape / integration bugs end-to-end BEFORE spending
~10 min/epoch on the HPC. Runs entirely on synthetic data, no dataset,
no HPC, CPU-only. Exercises exactly the code paths train_encoder_mt5.py
uses in one training step + one eval step:

  1. KeypointEncoder forward on enriched 1410-dim input.
  2. build_pose_mask (multi-granularity joint/frame/span masking).
  3. UniSignMT5 forward: CE loss + masked-pose reconstruction aux loss.
  4. loss.backward() and a gradient-flow audit (encoder, pose_norm,
     masked_pose_decoder, mT5 all receive gradient).
  5. generate() (beam search with the anti-loop settings).

Run from the repo root:

    PYTHONPATH=. python scripts/smoke_test_phase1.py                 # standard 282-dim
    PYTHONPATH=. python scripts/smoke_test_phase1.py --enriched      # 1410-dim dual-coords
    PYTHONPATH=. python scripts/smoke_test_phase1.py --enriched --lora

mT5-base is downloaded/loaded once (already cached on the HPC from
training). Everything else is CPU tensors — takes well under a minute.

Exit code 0 = all checks passed; non-zero = a check failed (message tells
you which contract broke).
"""
import argparse
import sys

import torch

# Import the real training modules so this tests the shipping code paths,
# not a reimplementation.
from data.utils import ENRICHED_DIM, KEYPOINT_DIM
from models.unisign_encoder import KeypointEncoder
from train.train_encoder_mt5 import UniSignMT5, SimpleCollator, build_pose_mask, MT5_PATH


def section(msg):
    print(f"\n{'=' * 60}\n{msg}\n{'=' * 60}")


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        raise AssertionError(msg)


def make_synthetic_batch(tokenizer, B=3, D=KEYPOINT_DIM, max_t=20):
    """Mimic what a dataset __getitem__ + SimpleCollator produce."""
    samples = []
    texts = [
        "бүгін ауа райы жайлы болады",
        "президент жаңа заңға қол қойды",
        "спорт жаңалықтары келесі бөлімде",
    ]
    for i in range(B):
        # Variable length so we exercise padding + the length mask.
        t = max_t - i * 3
        samples.append({
            "keypoints": torch.randn(t, D),
            "input_length": t,
            "text": texts[i % len(texts)],
        })
    collate = SimpleCollator(mt5_tokenizer=tokenizer, max_text_tokens=64)
    return collate(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", action="store_true",
                    help="use 1410-dim dual-coord enriched input")
    ap.add_argument("--lora", action="store_true",
                    help="wrap mT5 in LoRA adapters")
    ap.add_argument("--mask-ratio", type=float, default=0.15)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")
    input_dim = ENRICHED_DIM() if args.enriched else KEYPOINT_DIM

    section(f"Config: input_dim={input_dim} "
            f"({'enriched 1410' if args.enriched else 'standard 282'}), "
            f"lora={args.lora}, mask_ratio={args.mask_ratio}")

    # --- Build model exactly as the trainer does ---------------------------
    section("Building encoder + UniSignMT5 (downloads/loads mT5-base once)")
    encoder = KeypointEncoder(hidden_dim=768, input_dim=input_dim)
    model = UniSignMT5(
        encoder,
        mt5_path=MT5_PATH,
        masked_pose_dim=input_dim if args.mask_ratio > 0 else None,
    ).to(device)

    if args.lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(task_type="SEQ_2_SEQ_LM", r=16, lora_alpha=32,
                         target_modules=["q", "v"], lora_dropout=0.1)
        model.mt5 = get_peft_model(model.mt5, cfg)
        print("  mT5 wrapped in LoRA (q, v)")

    model.train()

    # --- Synthetic batch through the real collator -------------------------
    section("Synthetic batch via SimpleCollator")
    batch = make_synthetic_batch(model.mt5_tokenizer, B=3, D=input_dim)
    check(batch is not None, "collator returned a batch")
    kps = batch["keypoints"].to(device)
    label_ids = batch["label_ids"].to(device)
    label_attn = batch["label_attn_mask"].to(device)
    input_lengths = batch["input_lengths"].to(device)
    B, T, D = kps.shape
    check(D == input_dim, f"keypoint feature dim is {input_dim}")
    print(f"  kps={tuple(kps.shape)}, label_ids={tuple(label_ids.shape)}, "
          f"input_lengths={input_lengths.tolist()}")

    # --- Masking + forward (mirrors train_epoch) ---------------------------
    section("Masked-pose masking + forward")
    mask = None
    kps_train = kps
    if args.mask_ratio > 0:
        mask = build_pose_mask(kps, input_lengths, args.mask_ratio)
        check(mask.shape == kps.shape, f"pose mask shape == kps {tuple(kps.shape)}")
        # No sample may have all its valid frames masked.
        valid = (torch.arange(T)[None, :] < input_lengths[:, None])
        fully = mask.all(dim=2)
        check(not ((fully | ~valid).all(dim=1)).any(),
              "every sample keeps >=1 unmasked valid frame")
        kps_train = torch.where(mask, torch.zeros_like(kps), kps)

    loss, mse_loss, ctc_log_probs = model(
        kps_train, label_ids, label_attn,
        input_lengths=input_lengths,
        kps_target=kps if mask is not None else None,
        frame_mask=mask,
    )
    check(torch.isfinite(loss), f"CE loss is finite ({loss.item():.4f})")
    if mask is not None:
        check(mse_loss is not None and torch.isfinite(mse_loss),
              f"masked-pose MSE is finite ({mse_loss.item():.4f})")
        total = loss + 0.1 * mse_loss
    else:
        total = loss

    # --- Backward + gradient-flow audit ------------------------------------
    section("Backward + gradient-flow audit")
    total.backward()

    def grad_norm(module):
        g = [p.grad.norm().item() for p in module.parameters()
             if p.requires_grad and p.grad is not None]
        return sum(g)

    enc_g = grad_norm(model.encoder)
    check(enc_g > 0, f"gradient reaches the encoder (|g|={enc_g:.3e})")
    pn_g = grad_norm(model.pose_norm)
    check(pn_g > 0, f"gradient reaches pose_norm (|g|={pn_g:.3e})")
    if mask is not None:
        md_g = grad_norm(model.masked_pose_decoder)
        check(md_g > 0, f"gradient reaches masked_pose_decoder (|g|={md_g:.3e})")
    mt5_g = grad_norm(model.mt5)
    check(mt5_g > 0, f"gradient reaches mT5 (|g|={mt5_g:.3e})")

    # --- Generation (eval path) --------------------------------------------
    section("generate() — eval path")
    model.eval()
    with torch.no_grad():
        hyps = model.generate(kps, input_lengths=input_lengths,
                              max_new_tokens=32, num_beams=4)
    check(isinstance(hyps, list) and len(hyps) == B,
          f"generate() returned {B} strings")
    for ref, hyp in zip(batch["texts"], hyps):
        print(f"    ref: {ref!r}\n    hyp: {hyp!r}\n")

    section("ALL CHECKS PASSED")
    print("The Phase-1 pipeline is shape- and gradient-consistent end to end.")
    print("(Random weights → hyps are gibberish; that's expected. This test")
    print(" proves the plumbing, not the model quality.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
