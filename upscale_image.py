#!/usr/bin/env python
"""High-quality image upscaling using Lanczos or AI (Real-ESRGAN)"""

import argparse
from pathlib import Path
from PIL import Image

# Module-level cache: maps (scale, model_path) -> loaded spandrel model.
# Avoids reloading the 64 MB .pth file from disk on every generation call.
_realesrgan_model_cache: dict = {}


def upscale_image(
    input_path: str,
    output_path: str,
    scale: int = 2,
    method: str = "lanczos",
) -> str:
    """
    Upscale an image using Lanczos (fast) or Real-ESRGAN (AI quality)

    Args:
        input_path: Path to input image
        output_path: Path to save upscaled image
        scale: Upscaling factor (2 or 4)
        method: 'lanczos' (fast, CPU) or 'realesrgan' (AI, GPU)

    Returns:
        Path to upscaled image
    """
    if method == "realesrgan":
        return _upscale_realesrgan(input_path, output_path, scale)
    else:
        return _upscale_lanczos(input_path, output_path, scale)


def _upscale_lanczos(input_path: str, output_path: str, scale: int) -> str:
    """
    Upscale using Lanczos resampling (fast, CPU-based)

    Lanczos is a high-quality windowed sinc interpolation algorithm used by
    professional tools like Photoshop and GIMP. It produces excellent results
    for upscaling AI-generated images.
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


def _upscale_realesrgan(input_path: str, output_path: str, scale: int) -> str:
    """
    Upscale using Real-ESRGAN AI model (slower, GPU-based, better quality)

    Real-ESRGAN is an AI-based upscaling method that produces superior results
    for recovering fine details and textures compared to traditional methods.
    """
    import torch
    import numpy as np
    from spandrel import ModelLoader, ImageModelDescriptor

    print(f"Upscaling {scale}x using Real-ESRGAN (AI)...")

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Real-ESRGAN requires a CUDA GPU. Use method='lanczos' for CPU upscaling."
        )

    if scale not in [2, 4]:
        raise ValueError(f"Real-ESRGAN only supports 2x and 4x upscaling, got: {scale}")

    # Model paths (download via download_models.py --upscale-models)
    # Use absolute path based on script location
    script_dir = Path(__file__).parent.resolve()
    model_dir = script_dir / "models" / "realesrgan"
    model_filename = f"RealESRGAN_x{scale}plus.pth"
    model_path = model_dir / model_filename

    if not model_path.exists():
        raise RuntimeError(
            f"Real-ESRGAN model not found: {model_path}\n"
            "Download it first: uv run download_models.py --upscale-models"
        )

    # Load image
    img = Image.open(input_path).convert("RGB")
    original_size = img.size

    # Load model (cached after first call to avoid re-reading 64 MB from disk)
    cache_key = (scale, str(model_path))
    if cache_key not in _realesrgan_model_cache:
        print(f"Loading Real-ESRGAN {scale}x model from {model_path}...")
        model = ModelLoader().load_from_file(str(model_path))
        if not isinstance(model, ImageModelDescriptor):
            raise RuntimeError(
                f"Loaded model is not an image model: {model_path}\n"
                "Make sure the .pth file is a Real-ESRGAN image model."
            )
        model.cuda().eval()
        _realesrgan_model_cache[cache_key] = model
    else:
        print(f"Using cached Real-ESRGAN {scale}x model")
    model = _realesrgan_model_cache[cache_key]

    # Convert image to tensor [1, 3, H, W] in range [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.cuda()

    # Upscale
    print(f"Processing image on GPU...")
    with torch.no_grad():
        output_tensor = model(img_tensor)

    # Convert back to image
    output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    upscaled = Image.fromarray(output_np)

    # Save with high quality
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    upscaled.save(output_path, quality=95, optimize=True)

    new_size = upscaled.size
    print(f"✓ Upscaled image saved: {output_path}")
    print(f"  Original: {original_size[0]}x{original_size[1]}")
    print(f"  Upscaled: {new_size[0]}x{new_size[1]}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Upscale images using Lanczos or AI (Real-ESRGAN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upscale 2x with Lanczos (fast, CPU)
  uv run upscale_image.py -i image.png -o upscaled.png --scale 2

  # Upscale 4x with AI (better quality, GPU required)
  uv run upscale_image.py -i image.png -o large.png --scale 4 --method realesrgan

About methods:
  lanczos     - Professional-grade traditional upscaling (Photoshop/GIMP)
                Fast (0.5s), CPU-based, no model download needed
  
  realesrgan  - AI-based upscaling for superior detail recovery (DEFAULT)
                Slower (2-5s), GPU required, download via:
                uv run download_models.py --upscale-only
                
Note: generate_image.py defaults to realesrgan when using --upscale
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
        "--method",
        type=str,
        choices=["lanczos", "realesrgan"],
        default="lanczos",
        help="Upscaling method (default: lanczos)",
    )

    args = parser.parse_args()

    upscale_image(
        input_path=args.input,
        output_path=args.output,
        scale=args.scale,
        method=args.method,
    )


if __name__ == "__main__":
    main()
