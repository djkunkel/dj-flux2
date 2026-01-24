#!/usr/bin/env python
"""High-quality image upscaling using Lanczos resampling"""

import argparse
from pathlib import Path
from PIL import Image


def upscale_image(
    input_path: str,
    output_path: str,
    scale: int = 2,
) -> str:
    """
    Upscale an image using high-quality Lanczos resampling

    Lanczos is a high-quality windowed sinc interpolation algorithm used by
    professional tools like Photoshop and GIMP. It produces excellent results
    for upscaling AI-generated images.

    Args:
        input_path: Path to input image
        output_path: Path to save upscaled image
        scale: Upscaling factor (2 or 4)

    Returns:
        Path to upscaled image
    """
    print(f"Upscaling {scale}x using Lanczos resampling...")

    # Load image
    img = Image.open(input_path).convert("RGB")
    original_size = img.size

    # Calculate new size
    new_size = (img.width * scale, img.height * scale)

    # Upscale using Lanczos (high-quality windowed sinc interpolation)
    upscaled = img.resize(new_size, Image.Resampling.LANCZOS)

    # Save with high quality
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    upscaled.save(output_path, quality=95, optimize=True)

    print(f"✓ Upscaled image saved: {output_path}")
    print(f"  Original: {original_size[0]}x{original_size[1]}")
    print(f"  Upscaled: {new_size[0]}x{new_size[1]}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Upscale images using high-quality Lanczos resampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upscale 2x (512x512 → 1024x1024)
  uv run upscale_image.py -i image.png -o upscaled.png --scale 2

  # Upscale 4x (512x512 → 2048x2048)
  uv run upscale_image.py -i image.png -o large.png --scale 4

About Lanczos:
  Lanczos is a professional-grade upscaling algorithm used by Photoshop,
  GIMP, and other image editing tools. It provides excellent quality for
  AI-generated images and is much faster than neural network approaches.
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

    args = parser.parse_args()

    upscale_image(
        input_path=args.input,
        output_path=args.output,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
