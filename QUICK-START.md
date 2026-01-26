# Quick Start Guide for dj-flux2

> This is a simplified guide for getting started quickly. See [README.md](README.md) for full documentation.

## Installation (2 commands)

```bash
uv venv
uv pip install -e .
```

## Download Models (once)

```bash
# Login to Hugging Face first
huggingface-cli login

# Then download models (~12.6 GB)
uv run download_models.py
```

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
# Generate image
uv run generate_image.py "your prompt here"

# Custom output location
uv run generate_image.py "prompt" -o output/my_image.png

# Different size
uv run generate_image.py "prompt" -W 768 -H 512

# Fewer steps (faster, lower quality)
uv run generate_image.py "prompt" -s 2

# See all options
uv run generate_image.py --help
```

## Requirements

- Python 3.10-3.14 (3.12+ recommended)
- NVIDIA GPU with 12+ GB VRAM
- CUDA 12.x
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
1. Accept FLUX.2-dev license: https://huggingface.co/black-forest-labs/FLUX.2-dev
2. Check your HF token has "gated repos" access
3. Re-login: `huggingface-cli login`

**GPU not being used?**
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

## Next Steps

- Read [README.md](README.md) for complete documentation
- Read [MODS-README.md](MODS-README.md) for technical deep-dive
- Experiment with different prompts and parameters
- Try image-to-image transformations

---

**Built for learning and experimentation!**
