# Quick Start Guide for dj-flux2

> This is a simplified guide for getting started quickly. See [README.md](README.md) for full documentation.

## Installation (2 commands)

```bash
# Linux / macOS:
uv venv && uv pip install -e .

# Windows (CUDA) — uv tool install resolves independently from the project
# venv, so the PyTorch CUDA index must be passed explicitly:
uv tool install --editable . \
  --index https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  --reinstall-package torch \
  --reinstall-package torchvision \
  --reinstall-package triton-windows
```

For local development on Windows (without global tool install), run `uv sync` after cloning — the project's `pyproject.toml` already configures the CUDA index via `[tool.uv.sources]`, so `uv run` commands work without extra flags.

**AMD GPU (ROCm, Linux only):** After the Linux install, swap in the ROCm torch wheel:
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```
Supported GPUs: RX 6000/7000/9000 series (RDNA 2+). No code changes required. See README.md for details.

## Set Up Hugging Face Access (once)

FLUX models are gated. Before first use:

1. Accept the license for the model(s) you want:
   - https://huggingface.co/black-forest-labs/FLUX.2-klein-4B *(default)*
   - https://huggingface.co/black-forest-labs/FLUX.2-dev *(shared autoencoder — required)*
2. Login:
   ```bash
   hf auth login
   ```

**FLUX models download automatically on first generate** — no separate download step needed.

## Generate Images

```bash
# Basic text-to-image (default 512x512)
uv run generate_image.py "a cute cat"

# Native high resolution (if you have 16GB+ VRAM)
uv run generate_image.py "mountain landscape" -W 1024 -H 1024

# AI upscaling to 1024x1024 (works on 12GB VRAM)
uv run generate_image.py "mountain landscape" --upscale 2

# Image-to-image transformation
uv run generate_image.py "oil painting style" -i photo.jpg -o art.png

# Reproducible (with seed)
uv run generate_image.py "abstract art" -S 42
```

## High-Resolution Images

Two ways to get high-resolution output:

**Native generation** (requires sufficient VRAM):
```bash
uv run generate_image.py "landscape" -W 1024 -H 1024
```

**Lanczos upscaling** (works on any GPU):
```bash
# Generate + upscale in one command
uv run generate_image.py "landscape" --upscale 2

# Result: 1024x1024 image from 512x512 base
```

**Why use upscaling?**
- Works on 12GB VRAM GPUs (native 1024x1024 needs 16GB+)
- Professional-grade Lanczos algorithm (used by Photoshop)
- Fast (~8s total: 7s generation + 0.5s upscaling)

## Why `uv run`?

- **No activation needed** - `uv` automatically finds your `.venv/`
- **Shorter commands** - Just `uv run script.py` instead of activating first
- **Always correct environment** - Never accidentally use wrong Python

## Alternative: Traditional Workflow

If you prefer the traditional approach:

```bash
# Activate once per terminal session
source .venv/bin/activate

# Then use python directly
python generate_image.py "prompt"
python download_models.py
```

## What is Editable Mode?

The `-e` flag in `uv pip install -e .` means **editable mode**:

- Your code changes take effect **immediately**
- No need to reinstall after editing `generate_image.py`
- Perfect for development and learning

## Common Commands

```bash
# Generate image (default: Klein 4B, 512x512)
uv run generate_image.py "your prompt here"

# Use the 9B model for higher quality
uv run generate_image.py "your prompt" -m flux.2-klein-9b

# Use a base model (guidance and steps are meaningful)
uv run generate_image.py "your prompt" -m flux.2-klein-base-4b -g 4.0 -s 50

# Custom output location
uv run generate_image.py "prompt" -o output/my_image.png

# Different size
uv run generate_image.py "prompt" -W 768 -H 512

# See all options
uv run generate_image.py --help
```

## Requirements

- Python 3.10-3.14 (3.12+ recommended)
- NVIDIA GPU with 12+ GB VRAM (CUDA 12.x), **or** AMD GPU with 8+ GB VRAM (RX 6000/7000/9000, ROCm 6.4+)
- ~13 GB disk space for models

## Performance (RTX 4070)

- **Cold start**: ~15 seconds (first run)
- **Warm generation**: ~7 seconds (512x512)
- **VRAM usage**: ~12 GB

## Troubleshooting

**Out of memory?**
```bash
uv run generate_image.py "prompt" -W 512 -H 512
```

**Models not downloading?**
1. Accept the license for the model at `huggingface.co/black-forest-labs`
2. Accept the FLUX.2-dev license (shared autoencoder): https://huggingface.co/black-forest-labs/FLUX.2-dev
3. Check your HF token has "gated repos" read access
4. Re-login: `hf auth login`

The GUI shows a clear error with the exact URL to visit if access hasn't been granted yet.

**GPU not being used?**
```python
import torch
print(torch.__version__, torch.cuda.is_available())  # Should end True
# NVIDIA: 2.x.x  True
# AMD:    2.x.x+rocm6.4  True
```
- **Windows (NVIDIA):** If `False`, you have the CPU-only torch wheel. Reinstall using the Windows command in [Installation](#installation-2-commands) above — `uv tool install` ignores `pyproject.toml` sources, so the CUDA index must be passed explicitly.
- **Linux (AMD):** If `False`, run `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4` to install the ROCm wheel.

## Next Steps

- Read [README.md](README.md) for complete documentation
- Experiment with different prompts and parameters
- Try image-to-image transformations

---

**Built for learning and experimentation!**
