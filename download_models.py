#!/usr/bin/env python3
"""
Download required models for dj-flux2 (FLUX.2 Klein 4B)

Downloads are based on FLUX2_MODEL_INFO['flux.2-klein-4b'] requirements from the
flux2 submodule. This provides model-specific configuration including repo IDs,
filenames, and default parameters.

This script downloads:
1. FLUX.2 Klein 4B transformer (~7.4 GB)
   - File: flux-2-klein-4b.safetensors (top-level file in Klein-4B repo)
2. Qwen3-4B-FP8 text encoder (~4.9 GB)
   - Loaded via load_qwen3_embedder(variant="4B") from separate Qwen repo
3. FLUX.2-dev autoencoder (~321 MB)
   - File: ae.safetensors (shared across all FLUX models)
   - Klein models don't have ae.safetensors in their repos - this is expected

Total: ~12.6 GB (+ optional 128 MB for Real-ESRGAN)

Models are downloaded to:
- FLUX models: ~/.cache/huggingface/hub/
- Real-ESRGAN: models/realesrgan/
"""

import sys
from pathlib import Path
import urllib.request

try:
    from huggingface_hub import hf_hub_download, login
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:
    print("Error: huggingface_hub not installed")
    print("Install dependencies first: pip install -r requirements.txt")
    sys.exit(1)


def check_login():
    """Check if user is logged in to Hugging Face"""
    try:
        from huggingface_hub import whoami

        whoami()
        return True
    except Exception:
        return False


def download_model(repo_id: str, filename: str, description: str):
    """Download a single model file"""
    print(f"\n{'=' * 60}")
    print(f"Downloading: {description}")
    print(f"Repository: {repo_id}")
    print(f"File: {filename}")
    print(f"{'=' * 60}")

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
        )
        print(f"✓ Downloaded to: {path}")
        return True
    except HfHubHTTPError as e:
        if "403" in str(e):
            print(f"\n✗ Error: Access denied to {repo_id}")
            print("\nThis model requires accepting license terms:")
            print(f"1. Visit: https://huggingface.co/{repo_id}")
            print("2. Click 'Agree' to accept the license")
            print("3. Run this script again")
            return False
        else:
            print(f"\n✗ Error downloading: {e}")
            return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


def download_realesrgan_model(scale: int, model_dir: Path) -> bool:
    """Download Real-ESRGAN model"""
    model_urls = {
        2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    }

    if scale not in model_urls:
        print(f"Error: Unsupported scale {scale}")
        return False

    url = model_urls[scale]
    filename = f"RealESRGAN_x{scale}plus.pth"
    filepath = model_dir / filename

    if filepath.exists():
        print(f"  ✓ Already exists: {filepath}")
        return True

    print(f"\nDownloading Real-ESRGAN {scale}x model...")
    print(f"  URL: {url}")
    print(f"  Destination: {filepath}")

    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, filepath)
        print(f"  ✓ Downloaded: {filepath}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download models for dj-flux2")
    parser.add_argument(
        "--upscale-models",
        action="store_true",
        help="Also download Real-ESRGAN upscaling models (2x and 4x)",
    )
    parser.add_argument(
        "--upscale-only",
        action="store_true",
        help="Only download Real-ESRGAN models, skip FLUX models",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("dj-flux2 Model Downloader")
    print("=" * 60)
    print("\nThis will download ~12.6 GB of models to:")
    print("  ~/.cache/huggingface/hub/")
    print("\nModels (from FLUX2_MODEL_INFO):")
    print("  • FLUX.2 Klein 4B transformer (7.4 GB)")
    print("  • Qwen3-4B-FP8 text encoder (4.9 GB)")
    print("  • FLUX.2-dev autoencoder - shared (321 MB)")

    if args.upscale_models:
        print("\nOptional upscaling models:")
        print("  • Real-ESRGAN 2x model (64 MB)")
        print("  • Real-ESRGAN 4x model (64 MB)")

    # Check login
    if not check_login():
        print("\n" + "=" * 60)
        print("Hugging Face Login Required")
        print("=" * 60)
        print("\nYou need to login to download gated models.")
        print("\nSteps:")
        print("1. Create account at https://huggingface.co/join")
        print("2. Accept FLUX.2-dev license at:")
        print("   https://huggingface.co/black-forest-labs/FLUX.2-dev")
        print("3. Create a token at https://huggingface.co/settings/tokens")
        print("   - Select: 'Read access to contents of all public gated repos'")
        print("4. Login: huggingface-cli login")
        print("\nThen run this script again.")
        sys.exit(1)

    # Skip FLUX download if only upscale models requested
    if args.upscale_only:
        print("\n" + "=" * 60)
        print("Downloading Real-ESRGAN Models Only")
        print("=" * 60)

        # Use absolute path based on script location
        script_dir = Path(__file__).parent.resolve()
        model_dir = script_dir / "models" / "realesrgan"
        realesrgan_success = 0
        realesrgan_total = 2

        for scale in [2, 4]:
            if download_realesrgan_model(scale, model_dir):
                realesrgan_success += 1

        print(
            f"\n✓ Downloaded {realesrgan_success}/{realesrgan_total} Real-ESRGAN models"
        )

        if realesrgan_success == realesrgan_total:
            print("\nYou can now use AI upscaling:")
            print(
                '  uv run generate_image.py "prompt" --upscale 2 --upscale-method realesrgan'
            )
        return

    print("\n✓ Logged in to Hugging Face")

    input("\nPress Enter to start downloading (Ctrl+C to cancel)...")

    # Model files to download (matches FLUX2_MODEL_INFO requirements)
    # - Klein 4B: flux-2-klein-4b.safetensors (transformer)
    # - FLUX.2-dev: ae.safetensors (shared autoencoder for all models)
    # - Qwen3-4B-FP8: text encoder (loaded separately by load_qwen3_embedder)
    models = [
        {
            "repo_id": "black-forest-labs/FLUX.2-klein-4B",
            "files": [
                ("flux-2-klein-4b.safetensors", "FLUX.2 Klein 4B Transformer"),
            ],
        },
        {
            "repo_id": "black-forest-labs/FLUX.2-dev",
            "files": [
                (
                    "ae.safetensors",
                    "FLUX.2-dev Autoencoder (shared by all FLUX models)",
                ),
            ],
        },
        {
            "repo_id": "Qwen/Qwen3-4B-FP8",
            "files": [
                (
                    "model-00001-of-00002.safetensors",
                    "Qwen3-4B-FP8 Text Encoder (part 1)",
                ),
                (
                    "model-00002-of-00002.safetensors",
                    "Qwen3-4B-FP8 Text Encoder (part 2)",
                ),
            ],
        },
    ]

    success_count = 0
    total_count = sum(len(model["files"]) for model in models)

    for model in models:
        for filename, description in model["files"]:
            if download_model(model["repo_id"], filename, description):
                success_count += 1

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Successfully downloaded: {success_count}/{total_count} files")

    # Download Real-ESRGAN models if requested
    if args.upscale_models:
        print("\n" + "=" * 60)
        print("Downloading Real-ESRGAN Models")
        print("=" * 60)

        # Use absolute path based on script location
        script_dir = Path(__file__).parent.resolve()
        model_dir = script_dir / "models" / "realesrgan"
        realesrgan_success = 0
        realesrgan_total = 2

        for scale in [2, 4]:
            if download_realesrgan_model(scale, model_dir):
                realesrgan_success += 1

        print(f"\nReal-ESRGAN: {realesrgan_success}/{realesrgan_total} models")

    if success_count == total_count:
        print("\n✓ All FLUX models downloaded successfully!")
        print("\nYou can now run:")
        print('  python generate_image.py "a cute cat"')

        if args.upscale_models:
            if realesrgan_success == realesrgan_total:
                print("\n✓ All Real-ESRGAN models downloaded successfully!")
                print("\nFor AI upscaling:")
                print(
                    '  python generate_image.py "prompt" --upscale 2 --upscale-method realesrgan'
                )
    else:
        print(f"\n✗ Failed to download {total_count - success_count} files")
        print("Check the errors above and try again")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user")
        sys.exit(1)
