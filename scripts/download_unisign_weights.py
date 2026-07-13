"""
Download Uni-Sign pretrained weights from HuggingFace.

Usage:
  python scripts/download_unisign_weights.py
  # Downloads to: checkpoints/unisign/

Available checkpoints:
  csl_stage1_weight.pth          — Pose-only pretraining (1.19 GB)
  csl_stage2_weight.pth          — RGB+pose pretraining (1.2 GB)
  csl_daily_pose_only_slt.pth    — SLT fine-tuned, pose-only (1.19 GB)
  csl_daily_rgb_pose_slt.pth     — SLT fine-tuned, RGB+pose (1.2 GB)

Recommended for KRSL fine-tuning:
  1. csl_stage1_weight.pth — best transfer learning (pretrained, not task-specific)
  2. csl_daily_pose_only_slt.pth — if you want SLT-adapted features

Note: This requires the huggingface_hub package.
  pip install huggingface_hub
"""
import os
import argparse

def download_hf():
    """Download using huggingface_hub."""
    from huggingface_hub import hf_hub_download

    REPO_ID = "ZechengLi19/Uni-Sign"

    CHECKPOINTS = {
        'stage1_pose': {
            'filename': 'csl_stage1_weight.pth',
            'size': '1.19 GB',
            'desc': 'Pose-only pretraining (recommended)',
        },
        'stage2_rgb': {
            'filename': 'csl_stage2_weight.pth',
            'size': '1.2 GB',
            'desc': 'RGB+pose pretraining',
        },
        'slt_pose': {
            'filename': 'csl_daily_pose_only_slt.pth',
            'size': '1.19 GB',
            'desc': 'SLT fine-tuned, pose-only',
        },
        'slt_rgb': {
            'filename': 'csl_daily_rgb_pose_slt.pth',
            'size': '1.2 GB',
            'desc': 'SLT fine-tuned, RGB+pose',
        },
    }

    print("=" * 60)
    print("Uni-Sign Pretrained Weights Downloader")
    print("Repo: https://huggingface.co/ZechengLi19/Uni-Sign")
    print("=" * 60)
    print()

    for key, info in CHECKPOINTS.items():
        print(f"  {key:12s} — {info['filename']:35s} ({info['size']})")
        print(f"             {info['desc']}")
    print()

    # Default: download stage1 (best for transfer learning)
    default_key = 'stage1_pose'
    filename = CHECKPOINTS[default_key]['filename']

    print(f"Downloading {filename}...")
    output_dir = 'checkpoints/unisign'
    os.makedirs(output_dir, exist_ok=True)

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        local_dir=output_dir,
    )

    print(f"\nDownloaded to: {local_path}")
    print(f"\nUsage:")
    print(f"  python train/train_encoder_finetune.py \\")
    print(f"      --tokenizer /path/to/sp_model.model \\")
    print(f"      --pretrained {local_path}")


def download_direct():
    """Download using direct HTTPS (no huggingface_hub needed)."""
    import urllib.request

    REPO_URL = "https://huggingface.co/ZechengLi19/Uni-Sign/resolve/main"

    filename = 'csl_stage1_weight.pth'
    url = f"{REPO_URL}/{filename}"
    output_dir = 'checkpoints/unisign'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    print(f"Downloading {filename} from {url}...")
    print("(This is ~1.19 GB, may take a while)")

    urllib.request.urlretrieve(url, output_path)
    print(f"\nDownloaded to: {output_path}")

    print(f"\nUsage:")
    print(f"  python train/train_encoder_finetune.py \\")
    print(f"      --tokenizer /path/to/sp_model.model \\")
    print(f"      --pretrained {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['hf', 'direct'], default='hf',
                        help='Download method: hf (huggingface_hub) or direct (urllib)')
    parser.add_argument('--checkpoint', default='stage1',
                        help='Which checkpoint: stage1, stage2, slt_pose, slt_rgb')
    args = parser.parse_args()

    try:
        download_hf()
    except ImportError:
        print("[INFO] huggingface_hub not found. Trying direct download...")
        print("[INFO] For faster downloads: pip install huggingface_hub")
        download_direct()
    except Exception as e:
        print(f"[ERROR] HF download failed: {e}")
        print("[INFO] Trying direct download...")
        try:
            download_direct()
        except Exception as e2:
            print(f"[ERROR] Direct download also failed: {e2}")
            print("\nManual download:")
            print("  1. Go to: https://huggingface.co/ZechengLi19/Uni-Sign/tree/main")
            print("  2. Download: csl_stage1_weight.pth")
            print("  3. Place in: checkpoints/unisign/")
            print("  4. Run:")
            print("     python train/train_encoder_finetune.py \\")
            print("         --tokenizer /path/to/sp_model.model \\")
            print("         --pretrained checkpoints/unisign/csl_stage1_weight.pth")


if __name__ == '__main__':
    main()
