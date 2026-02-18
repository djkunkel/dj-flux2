#!/usr/bin/env python
"""Simple image generation script for FLUX.2 Klein 4B with CUDA support"""

import argparse
import gc
import torch
from einops import rearrange
from PIL import Image, ExifTags
from pathlib import Path
import sys
import os

# Add flux2/src to path (handle both local and installed as tool)
script_dir = Path(__file__).parent.resolve()
flux2_src = script_dir / "flux2" / "src"
sys.path.insert(0, str(flux2_src))

from flux2.util import load_flow_model, load_text_encoder, load_ae, FLUX2_MODEL_INFO
from flux2.sampling import (
    batched_prc_txt,
    batched_prc_img,
    denoise,
    get_schedule,
    scatter_ids,
    encode_image_refs,
)

# Model configuration from FLUX2_MODEL_INFO (flux2 submodule)
# This provides model-specific defaults and configuration
MODEL_NAME = "flux.2-klein-4b"
MODEL_INFO = FLUX2_MODEL_INFO[MODEL_NAME]

# Extract default values for this model
DEFAULT_STEPS = MODEL_INFO["defaults"]["num_steps"]  # 4
DEFAULT_GUIDANCE = MODEL_INFO["defaults"]["guidance"]  # 1.0

# Note: All FLUX models share the flux.2-dev autoencoder
# Klein models don't have ae.safetensors in their repos - this is expected per BFL's design
AE_MODEL_NAME = "flux.2-dev"


class ModelCache:
    """Singleton cache for FLUX models - load once, reuse many times

    This dramatically speeds up subsequent generations by keeping models
    in GPU/RAM memory instead of reloading from disk each time.
    """

    _instance = None
    _models = None
    _model_name = None
    _ae_model_name = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def is_loaded(self) -> bool:
        """Check if models are currently loaded"""
        return self._models is not None

    def load_models(self, model_name: str, ae_model_name: str, device):
        """Load models if not cached, return cached models

        Args:
            model_name: Name of the main FLUX model
            ae_model_name: Name of the autoencoder model
            device: Target device (typically 'cuda')

        Returns:
            Dictionary with 'text_encoder', 'model', and 'ae' keys
        """
        # If already loaded with same model names, return cached
        if self._models and self._model_name == model_name:
            print("Using cached models (fast path)")
            return self._models

        # Different model or first load
        if self._models:
            print("Clearing old models...")
            self.clear()

        print("[1/3] Loading text encoder...")
        text_encoder = load_text_encoder(model_name, device=device)

        print("[2/3] Loading transformer...")
        model = load_flow_model(model_name, device="cpu")

        print("[3/3] Loading autoencoder...")
        ae = load_ae(ae_model_name, device=device)

        ae.eval()
        text_encoder.eval()

        self._models = {
            "text_encoder": text_encoder,
            "model": model,
            "ae": ae,
        }
        self._model_name = model_name
        self._ae_model_name = ae_model_name

        print("✓ Models loaded and cached")
        return self._models

    def clear(self):
        """Unload models and free GPU/RAM"""
        if self._models:
            # Move to CPU first to free VRAM
            for name, model in self._models.items():
                if hasattr(model, "cpu"):
                    model.cpu()

            # Delete references
            del self._models
            self._models = None
            self._model_name = None
            self._ae_model_name = None

            # Force cleanup
            torch.cuda.empty_cache()
            gc.collect()

            print("✓ Models unloaded, memory freed")

    def get_memory_estimate(self) -> str:
        """Return approximate memory usage string"""
        if not self.is_loaded():
            return "0 MB"
        # Rough estimates: text_encoder ~1.2GB, model ~2.8GB, ae ~0.2GB
        return "~4.2 GB"


# Global singleton instance
model_cache = ModelCache()


def generate_image(
    prompt: str,
    output_path: str = "output.png",
    input_image: str = None,
    width: int = 512,
    height: int = 512,
    num_steps: int = DEFAULT_STEPS,
    guidance: float = DEFAULT_GUIDANCE,
    seed: int = None,
) -> int:
    """Generate an image using FLUX.2 Klein 4B

    Args:
        prompt: Text description of the image to generate
        output_path: Where to save the generated image
        input_image: Optional input image path for image-to-image generation
        width: Image width in pixels (must be multiple of 16)
        height: Image height in pixels (must be multiple of 16)
        num_steps: Number of denoising steps (4 is recommended for Klein)
        guidance: Guidance scale (1.0 is recommended for Klein)
        seed: Random seed for reproducibility (None for random)

    Returns:
        The seed used for generation (either provided or randomly generated)
    """
    model_name = MODEL_NAME

    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()

    mode = "Image-to-Image" if input_image else "Text-to-Image"
    print(f"FLUX.2 Klein 4B - {mode}")
    print(f"Prompt: {prompt}")
    if input_image:
        print(f"Input: {input_image}")
    print(
        f"Size: {width}x{height}, Steps: {num_steps}, Guidance: {guidance}, Seed: {seed}"
    )
    print()

    # Clear CUDA cache
    torch.cuda.empty_cache()
    device = torch.device("cuda")

    # Load models (cached if available for fast subsequent generations)
    models = model_cache.load_models(model_name, AE_MODEL_NAME, device)
    text_encoder = models["text_encoder"]
    model = models["model"]
    ae = models["ae"]

    # VRAM management: Move models to optimal devices for this generation
    # Strategy: Never have all 3 models on GPU simultaneously to avoid OOM
    text_encoder = text_encoder.to(device)
    ae = ae.to(device)
    model = model.cpu()  # Keep on CPU until after text encoding
    torch.cuda.empty_cache()

    print(f"\nGenerating {mode.lower()}...")
    with torch.no_grad():
        # Encode prompt
        ctx = text_encoder([prompt]).to(torch.bfloat16)
        ctx, ctx_ids = batched_prc_txt(ctx)

        # Encode reference image if provided
        ref_tokens = None
        ref_ids = None
        if input_image:
            print("  Encoding reference image...")
            img_ctx = [Image.open(input_image)]
            # Optionally match dimensions from input image
            if width is None or height is None:
                width, height = img_ctx[0].size
                print(f"  Using input image dimensions: {width}x{height}")
            ref_tokens, ref_ids = encode_image_refs(ae, img_ctx)

        # Move text encoder to CPU to free VRAM for transformer
        text_encoder = text_encoder.cpu()
        torch.cuda.empty_cache()

        # Move transformer model to GPU (now that text encoder is off GPU)
        model = model.to(device)

        # Create noise
        shape = (1, 128, height // 16, width // 16)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        randn = torch.randn(
            shape, generator=generator, dtype=torch.bfloat16, device="cuda"
        )
        x, x_ids = batched_prc_img(randn)

        # Denoise with optional reference image conditioning
        timesteps = get_schedule(num_steps, x.shape[1])
        x = denoise(
            model,
            x,
            x_ids,
            ctx,
            ctx_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=ref_tokens,
            img_cond_seq_ids=ref_ids,
        )

        # Decode
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        x = ae.decode(x).float()

    # Convert to image
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    # Move transformer back to CPU to free VRAM for next generation
    # This ensures we start each generation with minimal VRAM usage
    model = model.cpu()
    torch.cuda.empty_cache()

    # Save with metadata
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exif_data = Image.Exif()
    model_display_name = MODEL_INFO["repo_id"].split("/")[-1]
    exif_data[ExifTags.Base.Software] = f"FLUX.2 {model_display_name}"
    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
    desc = f"Prompt: {prompt} | Seed: {seed}"
    if input_image:
        desc += f" | Input: {input_image}"
    exif_data[ExifTags.Base.ImageDescription] = desc

    img.save(output_path, exif=exif_data, quality=95, subsampling=0)

    print(f"\n✓ Image saved to: {output_path}")
    print(f"  Seed: {seed} (use this to reproduce)")

    return seed


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with FLUX.2 Klein 4B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text-to-image (default 512x512)
  python generate_image.py "a cute cat"
  
  # Native high resolution (if you have VRAM)
  python generate_image.py "a mountain" -W 1024 -H 1024
  
  # 512x512 + AI upscaling to 1024x1024 (default when using --upscale)
  python generate_image.py "a mountain" --upscale 2
  
  # Use Lanczos for faster CPU-based upscaling
  python generate_image.py "a mountain" --upscale 2 --upscale-method lanczos
  
  # Image-to-image
  python generate_image.py "oil painting" -i photo.jpg -o art.png
  
  # 4x AI upscale to 2048x2048
  python generate_image.py "detailed scene" --upscale 4
""",
    )
    parser.add_argument("prompt", type=str, help="Text prompt describing the image")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.png",
        help="Output path (default: output.png)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Input image for image-to-image transformation",
    )
    parser.add_argument(
        "-W", "--width", type=int, default=512, help="Image width (default: 512)"
    )
    parser.add_argument(
        "-H", "--height", type=int, default=512, help="Image height (default: 512)"
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of steps (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "-g",
        "--guidance",
        type=float,
        default=DEFAULT_GUIDANCE,
        help=f"Guidance scale (default: {DEFAULT_GUIDANCE})",
    )
    parser.add_argument(
        "-S", "--seed", type=int, default=None, help="Random seed (default: random)"
    )
    parser.add_argument(
        "--upscale",
        type=int,
        choices=[2, 4],
        default=None,
        help="Upscale output by 2x or 4x",
    )
    parser.add_argument(
        "--upscale-method",
        type=str,
        choices=["lanczos", "realesrgan"],
        default="realesrgan",
        help="Upscaling method: realesrgan (AI, default) or lanczos (fast, CPU)",
    )

    args = parser.parse_args()

    # If upscaling, generate to a temp file first, then upscale to final output
    if args.upscale:
        import tempfile
        import os

        # Create temp file for intermediate image
        temp_fd, temp_path = tempfile.mkstemp(suffix=".png", prefix="flux_temp_")
        os.close(temp_fd)  # Close file descriptor, we just need the path

        try:
            # Generate to temp file
            generate_image(
                prompt=args.prompt,
                output_path=temp_path,
                input_image=args.input,
                width=args.width,
                height=args.height,
                num_steps=args.steps,
                guidance=args.guidance,
                seed=args.seed,
            )

            # Upscale temp file to final output
            method_name = (
                "Real-ESRGAN (AI)" if args.upscale_method == "realesrgan" else "Lanczos"
            )
            print(f"\nUpscaling {args.upscale}x with {method_name}...")

            from upscale_image import upscale_image

            upscale_image(
                input_path=temp_path,
                output_path=args.output,
                scale=args.upscale,
                method=args.upscale_method,
            )

            print(f"✓ Final upscaled image: {args.output}")

        except Exception as e:
            print(f"Error during upscaling: {e}")
            raise
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        # No upscaling, generate directly to output
        generate_image(
            prompt=args.prompt,
            output_path=args.output,
            input_image=args.input,
            width=args.width,
            height=args.height,
            num_steps=args.steps,
            guidance=args.guidance,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
