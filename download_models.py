#!/usr/bin/env python3
"""
Download Real-ESRGAN upscaling models for dj-flux2

FLUX models are downloaded automatically on first use via HuggingFace hub.
This script only handles the Real-ESRGAN weights, which come from GitHub
releases and are stored locally in models/realesrgan/.

Models downloaded:
  • RealESRGAN_x2plus.pth  (64 MB) — 2x upscaling
  • RealESRGAN_x4plus.pth  (64 MB) — 4x upscaling
"""

import sys
import urllib.request
from pathlib import Path


REALESRGAN_URLS = {
    2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}

MODEL_DIR = Path(__file__).parent.resolve() / "models" / "realesrgan"


def download_realesrgan(scale: int) -> bool:
    """Download a single Real-ESRGAN model file.

    Args:
        scale: Upscale factor — 2 or 4.

    Returns:
        True if the file is present (downloaded or already cached).
    """
    url = REALESRGAN_URLS[scale]
    filename = f"RealESRGAN_x{scale}plus.pth"
    filepath = MODEL_DIR / filename

    if filepath.exists():
        print(f"  ✓ Already exists: {filepath}")
        return True

    print(f"\nDownloading Real-ESRGAN {scale}x...")
    print(f"  Source : {url}")
    print(f"  Destination: {filepath}")

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, filepath)
        print(f"  ✓ Saved to: {filepath}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    print("=" * 60)
    print("dj-flux2 — Real-ESRGAN Model Downloader")
    print("=" * 60)
    print()
    print("FLUX models download automatically on first use.")
    print("This script pre-fetches the Real-ESRGAN upscaling weights.")
    print()
    print(f"Destination: {MODEL_DIR}")
    print()

    success = 0
    for scale in [2, 4]:
        if download_realesrgan(scale):
            success += 1

    print()
    print("=" * 60)
    if success == 2:
        print("✓ Both Real-ESRGAN models ready.")
        print()
        print("Use AI upscaling with:")
        print(
            '  uv run generate_image.py "prompt" --upscale 2 --upscale-method realesrgan'
        )
        print(
            "  uv run upscale_image.py -i input.png -o output.png --method realesrgan"
        )
    else:
        print(f"✗ {2 - success} model(s) failed to download. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
