#!/usr/bin/env python
"""Simple image generation script for FLUX.2 Klein models with CUDA support"""

import argparse
import gc
import os
import tempfile
import torch
from einops import rearrange
from PIL import Image, ExifTags
from pathlib import Path
import sys

# Add flux2/src to path (handle both local and installed as tool)
script_dir = Path(__file__).parent.resolve()
flux2_src = script_dir / "flux2" / "src"
sys.path.insert(0, str(flux2_src))

from flux2.util import load_flow_model, load_text_encoder, load_ae, FLUX2_MODEL_INFO
from flux2.sampling import (
    batched_prc_txt,
    batched_prc_img,
    denoise,
    denoise_cfg,
    get_schedule,
    scatter_ids,
    encode_image_refs,
)

# The four Klein variants this tool supports.
# flux.2-dev is excluded: it requires the Mistral-Small-3.2-24B text encoder
# (~24 GB VRAM for the encoder alone) — a completely different hardware tier.
SUPPORTED_MODELS = [
    "flux.2-klein-4b",
    "flux.2-klein-9b",
    "flux.2-klein-base-4b",
    "flux.2-klein-base-9b",
]

DEFAULT_MODEL = "flux.2-klein-4b"

# All Klein variants share the flux.2-dev autoencoder weights.
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
            model_name: Name of the main FLUX model (must be in SUPPORTED_MODELS)
            ae_model_name: Name of the autoencoder model
            device: Target device (typically 'cuda')

        Returns:
            Dictionary with 'text_encoder', 'model', and 'ae' keys
        """
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name!r}. "
                f"Choose from: {', '.join(SUPPORTED_MODELS)}"
            )

        # If already loaded with same model names, return cached
        if (
            self._models
            and self._model_name == model_name
            and self._ae_model_name == ae_model_name
        ):
            print("Using cached models (fast path)")
            return self._models

        # Different model or first load — clear first
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
            for name, model in self._models.items():
                if hasattr(model, "cpu"):
                    model.cpu()

            del self._models
            self._models = None
            self._model_name = None
            self._ae_model_name = None

            torch.cuda.empty_cache()
            gc.collect()

            print("✓ Models unloaded, memory freed")

    def get_memory_estimate(self) -> str:
        """Return approximate memory usage based on loaded model parameters"""
        if not self.is_loaded():
            return "0 MB"
        total_bytes = 0
        for model in self._models.values():
            if hasattr(model, "parameters"):
                total_bytes += sum(
                    p.nelement() * p.element_size() for p in model.parameters()
                )
        gb = total_bytes / (1024**3)
        return f"~{gb:.1f} GB"


# Global singleton instance
model_cache = ModelCache()


def generate_image(
    prompt: str,
    output_path: str = "output.png",
    input_image: str = None,
    width: int = 512,
    height: int = 512,
    num_steps: int = None,
    guidance: float = None,
    seed: int = None,
    model_name: str = DEFAULT_MODEL,
) -> int:
    """Generate an image using a FLUX.2 Klein model.

    Args:
        prompt: Text description of the image to generate
        output_path: Where to save the generated image
        input_image: Optional input image path for image-to-image generation
        width: Image width in pixels (must be multiple of 16)
        height: Image height in pixels (must be multiple of 16)
        num_steps: Number of denoising steps (None = model default)
        guidance: Guidance scale (None = model default; ignored for distilled models)
        seed: Random seed for reproducibility (None for random)
        model_name: Which Klein model to use (must be in SUPPORTED_MODELS)

    Returns:
        The seed used for generation (either provided or randomly generated)
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name!r}. "
            f"Choose from: {', '.join(SUPPORTED_MODELS)}"
        )

    model_info = FLUX2_MODEL_INFO[model_name]
    defaults = model_info["defaults"]
    is_distilled = model_info["guidance_distilled"]

    # Apply model defaults for unspecified parameters
    if num_steps is None:
        num_steps = defaults["num_steps"]
    if guidance is None:
        guidance = defaults["guidance"]

    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()

    mode = "Image-to-Image" if input_image else "Text-to-Image"
    print(f"{model_name} - {mode}")
    print(f"Prompt: {prompt}")
    if input_image:
        print(f"Input: {input_image}")
    print(
        f"Size: {width}x{height}, Steps: {num_steps}, "
        f"Guidance: {guidance}{'(fixed)' if is_distilled else ''}, Seed: {seed}"
    )
    print()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU found. FLUX.2 Klein requires a CUDA-capable (NVIDIA) or ROCm-capable (AMD) GPU.\n"
            "NVIDIA: check your driver and CUDA installation.\n"
            "AMD: install the ROCm torch wheel — see README.md for instructions."
        )
    torch.cuda.empty_cache()
    device = torch.device("cuda")

    # Load models (cached if available for fast subsequent generations)
    models = model_cache.load_models(model_name, AE_MODEL_NAME, device)
    text_encoder = models["text_encoder"]
    model = models["model"]
    ae = models["ae"]

    # VRAM management: Move models to optimal devices for this generation.
    # Strategy: Never have all 3 models on GPU simultaneously to avoid OOM.
    text_encoder = text_encoder.to(device)
    ae = ae.to(device)
    model = model.cpu()  # Keep on CPU until after text encoding
    torch.cuda.empty_cache()

    print(f"\nGenerating {mode.lower()}...")
    with torch.no_grad():
        # Encode prompt.
        # Distilled models use a single conditional context.
        # Base models use CFG: concatenate empty + prompt contexts for two-pass denoising.
        if is_distilled:
            ctx = text_encoder([prompt]).to(torch.bfloat16)
        else:
            ctx_empty = text_encoder([""]).to(torch.bfloat16)
            ctx_prompt = text_encoder([prompt]).to(torch.bfloat16)
            ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
        ctx, ctx_ids = batched_prc_txt(ctx)

        # Encode reference image if provided
        ref_tokens = None
        ref_ids = None
        if input_image:
            print("  Encoding reference image...")
            img_ctx = [Image.open(input_image)]
            ref_tokens, ref_ids = encode_image_refs(ae, img_ctx)

        # Move text encoder and ae to CPU to free VRAM for transformer.
        # ae is no longer needed until decode; freeing it gives the transformer
        # the headroom it needs on cards with ~12 GB VRAM.
        text_encoder = text_encoder.cpu()
        ae = ae.cpu()
        torch.cuda.empty_cache()

        # Move transformer to GPU
        model = model.to(device)

        # Create noise
        shape = (1, 128, height // 16, width // 16)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        randn = torch.randn(
            shape, generator=generator, dtype=torch.bfloat16, device="cuda"
        )
        x, x_ids = batched_prc_img(randn)

        timesteps = get_schedule(num_steps, x.shape[1])

        # Distilled models: single-pass denoising with baked-in guidance embedding.
        # Base models: classifier-free guidance (two forward passes per step).
        if is_distilled:
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
        else:
            x = denoise_cfg(
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

        # Move transformer back to CPU, bring ae back for decode.
        # Never have model + ae on GPU simultaneously.
        model = model.cpu()
        ae = ae.to(device)
        torch.cuda.empty_cache()

        # Decode
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        x = ae.decode(x).float()

    # Convert to image
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    # ae stays on GPU (small, ~0.2 GB); VRAM is free for next generation's
    # text_encoder phase. Explicit cache clear to release intermediate tensors.
    torch.cuda.empty_cache()

    # Save with metadata
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exif_data = Image.Exif()
    model_display_name = model_info["repo_id"].split("/")[-1]
    exif_data[ExifTags.Base.Software] = f"FLUX.2 {model_display_name}"
    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
    desc = f"Prompt: {prompt} | Seed: {seed}"
    if input_image:
        desc += f" | Input: {input_image}"
    exif_data[ExifTags.Base.ImageDescription] = desc

    img.save(output_path, exif=exif_data)

    print(f"\n✓ Image saved to: {output_path}")
    print(f"  Seed: {seed} (use this to reproduce)")

    return seed


def main():
    # Build the default steps help string dynamically from model defaults
    default_steps_per_model = {
        m: FLUX2_MODEL_INFO[m]["defaults"]["num_steps"] for m in SUPPORTED_MODELS
    }
    steps_help = "Number of denoising steps. Defaults: " + ", ".join(
        f"{m}={s}" for m, s in default_steps_per_model.items()
    )

    default_guidance_per_model = {
        m: FLUX2_MODEL_INFO[m]["defaults"]["guidance"] for m in SUPPORTED_MODELS
    }
    guidance_help = (
        "Guidance scale. Ignored for distilled models (klein-4b, klein-9b). Defaults: "
        + ", ".join(f"{m}={g}" for m, g in default_guidance_per_model.items())
    )

    parser = argparse.ArgumentParser(
        description="Generate images with FLUX.2 Klein models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text-to-image (default 512x512, Klein 4B)
  python generate_image.py "a cute cat"

  # Use the base model (guidance is meaningful here)
  python generate_image.py "a mountain" -m flux.2-klein-base-4b -g 4.0

  # Native high resolution
  python generate_image.py "a mountain" -W 1024 -H 1024

  # 512x512 + AI upscaling to 1024x1024
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
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=SUPPORTED_MODELS,
        help=f"Model to use (default: {DEFAULT_MODEL})",
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
        default=None,
        help=steps_help,
    )
    parser.add_argument(
        "-g",
        "--guidance",
        type=float,
        default=None,
        help=guidance_help,
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

    generate_kwargs = dict(
        prompt=args.prompt,
        input_image=args.input,
        width=args.width,
        height=args.height,
        num_steps=args.steps,  # None → model default applied inside generate_image()
        guidance=args.guidance,  # None → model default applied inside generate_image()
        seed=args.seed,
        model_name=args.model,
    )

    if args.upscale:
        temp_fd, temp_path = tempfile.mkstemp(suffix=".png", prefix="flux_temp_")
        os.close(temp_fd)

        try:
            generate_image(output_path=temp_path, **generate_kwargs)

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
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        generate_image(output_path=args.output, **generate_kwargs)


if __name__ == "__main__":
    main()
