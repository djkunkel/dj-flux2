# dj-flux2

Minimal FLUX.2 Klein image generation with GPU support (NVIDIA/CUDA and AMD/ROCm). Fast, simple, and educational.

## Features

- Text-to-Image and Image-to-Image (single or multi-reference)
- Interactive PySide6 (Qt6) GUI with side-by-side preview
- Multiple models: Klein 4B/9B (distilled) and base variants
- Lanczos and AI (Real-ESRGAN) upscaling
- HTTP API server with WebSocket progress
- NVIDIA (CUDA) and AMD (ROCm) GPU support

## Requirements

- Python 3.10+ (3.12+ recommended)
- NVIDIA GPU with 8+ GB VRAM, **or** AMD GPU with 8+ GB VRAM (RX 6000/7000/9000 series, RDNA 2+)
- CUDA 12.x (NVIDIA) or ROCm 7.2+ (AMD)
- ~13 GB disk space for models

## Quick Start

```bash
git clone --recurse-submodules https://github.com/yourusername/dj-flux2.git
cd dj-flux2

# One-time setup — installs wheels, writes .gpu-backend, adds dj-flux2 to ~/.local/bin
./setup cuda         # NVIDIA GPU (CUDA 12.6)
./setup rocm         # AMD GPU, ROCm 7.2 stable (RDNA 2/3)
./setup rocm-nightly # AMD RDNA4 (RX 9700), ROCm 7.14 nightly (multi-arch)
./setup cpu          # CPU only

# rocm-nightly: override target GPU architecture (default: gfx1201 for RX 9700)
ROCM_GFX=gfx1100 ./setup rocm-nightly   # e.g. RX 7900 XTX

# Then generate
./run generate "a cute cat on a windowsill"
dj-flux2 generate "a cute cat on a windowsill"   # from any directory
```

Supported AMD GPUs: RX 6000 / 7000 / 9000 series (RDNA 2+). APU/iGPU not supported. No code changes required — PyTorch's ROCm backend reuses the `torch.cuda` API identically.

**Using traditional pip (CUDA example):**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[cuda]" --index-url https://download.pytorch.org/whl/cu126
```

### Hugging Face Access

FLUX.2 Klein models are gated. Accept the license for each model you want to use:

1. Create account and token: https://huggingface.co/settings/tokens
   - Enable: "Read access to contents of all public gated repos"
2. Accept the license for each model:
   - https://huggingface.co/black-forest-labs/FLUX.2-dev *(required — shared autoencoder)*
   - https://huggingface.co/black-forest-labs/FLUX.2-klein-4B *(default)*
   - https://huggingface.co/black-forest-labs/FLUX.2-klein-9B *(optional)*
   - https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv *(optional; high VRAM)*
   - https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B *(optional)*
   - https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B *(optional)*
3. Login: `hf auth login`

**FLUX models download automatically on first use.** If generation fails with "Failed to access the model repository", the license gate for that specific model has not been accepted yet.

> **`flux.2-klein-9b-kv` note:** Same size as `flux.2-klein-9b` (~17 GB). Adds KV-cache support that speeds up multi-reference img2img (~2.5x) by caching reference key/value tensors across denoising steps. Peak VRAM ~29 GB for heavy editing (32 GB-class GPUs). No advantage over regular `flux.2-klein-9b` for plain text-to-image.

### Real-ESRGAN Models (optional)

Required for `--upscale-method realesrgan`. Downloads ~128 MB to `models/realesrgan/`:

```bash
dj-flux2 download
```

## Usage

### GUI

```bash
./run gui
dj-flux2 gui   # from any directory
```

The GUI provides txt2img and img2img modes, multi-reference input, side-by-side preview, model selector, upscaling, and persistent model caching (first generation ~15-25s, subsequent ~2-5s on RTX 4070).

### Command Line

`dj-flux2` (installed to `~/.local/bin` by `./setup`) and `./run` are equivalent. All examples below work with either.

```bash
# Text-to-image
dj-flux2 generate "a majestic mountain at sunset"
dj-flux2 generate "a robot" -o my_robot.png -W 768 -H 768 -S 42

# Image-to-image (single or multi-reference)
dj-flux2 generate "oil painting" -i photo.jpg -o painting.png
dj-flux2 generate "combine these styles" -i photo1.jpg -i photo2.jpg -o combined.png

# Upscaling (generate then upscale)
dj-flux2 generate "detailed cityscape" --upscale 2               # AI upscale (default)
dj-flux2 generate "detailed cityscape" --upscale 2 --upscale-method lanczos

# Upscale an existing image
dj-flux2 upscale -i input.png -o output.png --scale 2 --method realesrgan

# API server
dj-flux2 serve                     # 0.0.0.0:8190
dj-flux2 api-generate "prompt" -o out.png   # blocking client

# Other
dj-flux2 download     # download Real-ESRGAN models
dj-flux2 config       # show/set persistent defaults
dj-flux2 skill        # install OpenCode skill to CWD
```

### All Options

```
dj-flux2 generate [options] prompt

  -m, --model         flux.2-klein-4b (default), flux.2-klein-9b,
                      flux.2-klein-base-4b, flux.2-klein-base-9b
  -i, --input         Input image for img2img (repeat for multi-ref: -i a.jpg -i b.jpg)
  -o, --output        Output path (default: output.png)
  -W, --width         Width in pixels (default: 512)
  -H, --height        Height in pixels (default: 512)
  -s, --steps         Denoising steps
  -g, --guidance      Guidance scale (ignored for distilled models)
  -S, --seed          Random seed
  --upscale           Upscale factor: 2 or 4
  --upscale-method    realesrgan (default) or lanczos
```

### High-Resolution Images

Native generation is limited by VRAM (12 GB → ~1024x1024; 24 GB+ → 2048x2048). For larger output, generate at 512x512 and upscale:

```bash
dj-flux2 generate "detailed cityscape" --upscale 2   # → 1024x1024, AI quality
dj-flux2 generate "detailed cityscape" --upscale 4   # → 2048x2048
dj-flux2 generate "detailed cityscape" --upscale 2 --upscale-method lanczos   # fast, CPU
```

| Method | Speed (2x) | Quality | VRAM |
|--------|-----------|---------|------|
| Real-ESRGAN (default) | ~2-5s | Superior | +2 GB |
| Lanczos | ~0.5s | Excellent | None (CPU) |

All dimensions must be multiples of 16.

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| CLI — cold start | ~15s | Loads models from disk |
| CLI — warm | ~7s | Models in HF cache |
| GUI — first generation | ~15-25s | Loads and caches models |
| GUI — subsequent | **2-5s** | Reuses cached models |

Figures for RTX 4070, 512×512. The GUI keeps models in RAM between generations; the "Unload Models" button frees that memory when you're done.

## Architecture

```
Text Prompt → Qwen3 encoder → FLUX.2 Klein transformer → VAE Decoder → Image
                   ↑                      ↑
          4B: Qwen3-4B-FP8       4B: 7.4 GB transformer
          9B: Qwen3-8B-FP8       9B: larger transformer
```

**Distilled models** (klein-4b, klein-9b, klein-9b-kv): guidance is baked into the weights; steps and guidance are fixed. The GUI greys these controls out automatically.

**Base models** (klein-base-4b, klein-base-9b): classifier-free guidance, two forward passes per step. Defaults: guidance 4.0, steps 50.

VRAM choreography keeps at most one large model on GPU at a time, swapping text encoder and autoencoder to CPU around the transformer's denoising pass.

## Configuration

Persistent defaults, stored in `.dj-flux2.conf` (gitignored):

```bash
./run config                        # show current settings
./run config model flux.2-klein-9b
./run config width 768
./run config height 768
./run config steps 4
./run config guidance 1.0
```

CLI flags always override config. The GUI reads config on launch and pre-populates all controls.

## HTTP API Server

```bash
./run serve              # 0.0.0.0:8190
./run serve --port 9000
```

Interactive docs: `http://localhost:8190/docs`

| Method | Path | Description |
|---|---|---|
| `POST` | `/generate` | Submit a job (returns token) |
| `GET` | `/status/{token}` | Job status |
| `GET` | `/result/{token}` | Download completed image |
| `GET` | `/queue` | List all jobs |
| `POST` | `/cancel/{token}` | Cancel a queued job |
| `GET` | `/models` | List supported/loaded models |
| `WS` | `/ws/{token}` | Real-time progress updates |

```bash
# Submit
curl -X POST http://localhost:8190/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a castle on a hill", "width": 512, "height": 512}'
# → {"token": "a3f2b1c4..."}

# Poll and download
curl http://localhost:8190/status/a3f2b1c4...
curl http://localhost:8190/result/a3f2b1c4... -o castle.png

# Multi-reference img2img
curl -X POST http://localhost:8190/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "combine styles", "input_image_base64": ["<b64_1>", "<b64_2>"]}'
```

Jobs are ordered by model name first (to minimise GPU reloads), then FIFO. Queue limit: 50 jobs. Images saved to `output/` with timecoded filenames.

Connect to `/ws/{token}` after submitting to receive JSON progress messages (`queued` → `running` → `complete`/`error`).

### Blocking CLI Client

```bash
dj-flux2 api-generate "a red button icon" -o assets/button.png -W 256 -H 256
dj-flux2 api-generate "oil painting" -i photo.jpg -o art.png
dj-flux2 api-generate "combine styles" -i photo1.jpg -i photo2.jpg -o combined.png
```

Supports all `generate` flags plus `--host`, `--port`, `--timeout`, `--poll-interval`.

### OpenCode Skill

```bash
# From your project directory
dj-flux2 skill
```

Installs `.opencode/skills/generate-image/SKILL.md` so OpenCode agents in that project can generate image assets via `dj-flux2 api-generate`.

## Troubleshooting

**Out of memory:** Use `--upscale 2` instead of a large native resolution, or reduce `-W`/`-H`.

**GPU not detected:**
```bash
./run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# NVIDIA: 2.x.x  True
# AMD:    2.x.x+rocm7.2  True
```
If `False` on AMD, re-run `./setup rocm` or `./setup rocm-nightly`.

**Model download fails:** Accept the license for each model at `huggingface.co/black-forest-labs`, including FLUX.2-dev (shared autoencoder). Re-login with `hf auth login`.

## Project Structure

```
dj-flux2/
├── generate_image.py      # Main inference script + ModelCache
├── gui_generate.py        # GUI application logic (PySide6/Qt6)
├── gui_components.py      # GUI widget layout
├── serve_api.py           # HTTP API server (FastAPI + uvicorn)
├── api_generate.py        # Blocking CLI wrapper for the API server
├── upscale_image.py       # Lanczos and Real-ESRGAN upscaling
├── download_models.py     # Real-ESRGAN model downloader
├── skills/                # OpenCode agent skills
│   └── generate-image/SKILL.md
├── flux2/                 # BFL submodule (do not modify)
├── pyrightconfig.json     # IDE/LSP config
├── pyproject.toml         # Dependencies and entry points
└── uv.lock                # Locked dependency versions
```

## License

This project: MIT License

Black Forest Labs flux2 submodule: See `flux2/LICENSE.md`

FLUX.2 Klein 4B model: Apache 2.0 | FLUX.2-dev model: Non-Commercial License

## Credits

- **Black Forest Labs** for FLUX.2 Klein
- **Hugging Face** for model hosting
