#!/usr/bin/env python
"""AI-powered image upscaling using Real-ESRGAN"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image


def _upscale_fallback(input_path: str, output_path: str, scale: int) -> str:
    """Fallback upscaling using PIL's high-quality Lanczos resampling"""
    print(f"Using Lanczos resampling for {scale}x upscaling...")
    img = Image.open(input_path)
    new_size = (img.width * scale, img.height * scale)
    upscaled = img.resize(new_size, Image.Resampling.LANCZOS)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    upscaled.save(output_path, quality=95)

    print(f"✓ Upscaled image saved: {output_path}")
    print(f"  Final size: {upscaled.size[0]}x{upscaled.size[1]}")
    print(
        "  Note: For AI upscaling, install: uv pip install --no-deps realesrgan basicsr"
    )

    return str(output_path)


def upscale_image(
    input_path: str,
    output_path: str,
    scale: int = 2,
    model_name: str = "RealESRGAN_x2plus",
    tile_size: int = 512,
    half_precision: bool = True,
) -> str:
    """
    Upscale an image using Real-ESRGAN AI model

    Args:
        input_path: Path to input image
        output_path: Path to save upscaled image
        scale: Upscaling factor (2 or 4)
        model_name: Model to use for upscaling
        tile_size: Tile size for processing (avoid OOM)
        half_precision: Use FP16 for faster inference

    Returns:
        Path to upscaled image
    """
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError as e:
        print(f"Error: Real-ESRGAN dependencies not fully installed: {e}")
        print("For now, falling back to high-quality Lanczos upscaling...")
        return _upscale_fallback(input_path, output_path, scale)

    # Model configurations
    models = {
        "RealESRGAN_x2plus": {
            "scale": 2,
            "arch": RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            ),
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        },
        "RealESRGAN_x4plus": {
            "scale": 4,
            "arch": RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            ),
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        },
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(models.keys())}"
        )

    # Create weights directory
    weights_dir = Path(__file__).parent / "weights"
    weights_dir.mkdir(exist_ok=True)
    model_path = weights_dir / f"{model_name}.pth"

    # Download model if needed
    if not model_path.exists():
        print(f"Downloading {model_name} model (~50MB)...")
        import urllib.request

        urllib.request.urlretrieve(models[model_name]["url"], model_path)
        print(f"✓ Model downloaded to {model_path}")

    # Initialize upsampler
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    upsampler = RealESRGANer(
        scale=models[model_name]["scale"],
        model_path=str(model_path),
        model=models[model_name]["arch"],
        tile=tile_size,
        tile_pad=10,
        pre_pad=0,
        half=half_precision and device == "cuda",
        device=device,
    )

    # Load image
    print(f"Loading image: {input_path}")
    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img)

    # Upscale
    print(
        f"Upscaling {img.size[0]}x{img.size[1]} → {img.size[0] * scale}x{img.size[1] * scale}..."
    )
    output_np, _ = upsampler.enhance(img_np, outscale=scale)

    # Save result
    result = Image.fromarray(output_np)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, quality=95)

    print(f"✓ Upscaled image saved: {output_path}")
    print(f"  Final size: {result.size[0]}x{result.size[1]}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Upscale images using Real-ESRGAN AI model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upscale 2x (512x512 → 1024x1024)
  python upscale_image.py -i image.png -o upscaled.png --scale 2

  # Upscale 4x (512x512 → 2048x2048)
  python upscale_image.py -i image.png -o large.png --scale 4

  # Use different model
  python upscale_image.py -i image.png -o output.png --scale 4 --model RealESRGAN_x4plus
""",
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input image path"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output image path"
    )
    parser.add_argument(
        "--scale",
        type=int,
        choices=[2, 4],
        default=2,
        help="Upscaling factor (default: 2)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RealESRGAN_x2plus",
        choices=["RealESRGAN_x2plus", "RealESRGAN_x4plus"],
        help="Upscaling model (default: RealESRGAN_x2plus)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile size for processing (default: 512)",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable FP16 (slower but more compatible)",
    )

    args = parser.parse_args()

    upscale_image(
        input_path=args.input,
        output_path=args.output,
        scale=args.scale,
        model_name=args.model,
        tile_size=args.tile_size,
        half_precision=not args.no_half,
    )


if __name__ == "__main__":
    main()
