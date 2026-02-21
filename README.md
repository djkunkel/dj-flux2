# dj-flux2

Minimal FLUX.2 Klein image generation with CUDA support. Fast, simple, and educational.

## Features

- 🚀 **Fast**: Sub-second generation on RTX 4070 (4-step distilled models)
- 🎨 **Text-to-Image**: Generate images from text descriptions
- 🖼️ **Image-to-Image**: Transform images with prompts
- 🖥️ **GUI Interface**: Modern PySide6 (Qt6) app for easy experimentation
- 🔍 **Real-time Preview**: See results side-by-side before saving
- 🤖 **Multiple Models**: Klein 4B/9B (distilled) and base variants
- 💾 **Minimal**: Small codebase + BFL submodule
- 🎓 **Educational**: Clear code structure for learning
- 🔧 **CUDA Accelerated**: Runs on NVIDIA GPUs

## Requirements

- Python 3.10+ (3.12+ recommended)
- NVIDIA GPU with 8+ GB VRAM (RTX 3080/4070 or better; 12 GB recommended for 1024x1024)
- CUDA 12.x
- ~13 GB disk space for models

## Quick Start

### Installation Options

**Option 1: Install as a global tool (Recommended for all users):**
```bash
git clone --recurse-submodules https://github.com/yourusername/dj-flux2.git
cd dj-flux2
uv tool install --editable .

# All commands become available globally:
dj-flux2 "your prompt"
dj-flux2-gui              # Launch GUI
dj-flux2-upscale -i input.png -o output.png
dj-flux2-download
```

**Option 2: Local development setup:**
```bash
git clone --recurse-submodules https://github.com/yourusername/dj-flux2.git
cd dj-flux2

# Using uv (recommended)
uv venv
uv pip install -e .

# Using traditional pip
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -e .
```

### Setup Hugging Face Access

FLUX.2 Klein models are gated and require accepting license terms before first use:

1. Create account: https://huggingface.co/join
2. Accept license for each model you want to use:
   - https://huggingface.co/black-forest-labs/FLUX.2-klein-4B *(default)*
   - https://huggingface.co/black-forest-labs/FLUX.2-klein-9B *(optional)*
   - https://huggingface.co/black-forest-labs/FLUX.2-dev *(required for the shared autoencoder)*
3. Create token: https://huggingface.co/settings/tokens
   - Enable: "Read access to contents of all public gated repos"
4. Login:
   ```bash
   huggingface-cli login
   ```

**FLUX models download automatically on first use** — no manual download step required. The first generation will take a few minutes longer while weights are fetched from HuggingFace to `~/.cache/huggingface/hub/`.

### Download Real-ESRGAN Models (optional)

Real-ESRGAN upscaling weights are not on HuggingFace and must be downloaded separately before using `--upscale-method realesrgan`:

```bash
# If installed as a tool:
dj-flux2-download

# If using local development setup:
uv run download_models.py
```

This downloads (~128 MB total) to `models/realesrgan/`:
- Real-ESRGAN 2x model (64 MB)
- Real-ESRGAN 4x model (64 MB)

### Generate Your First Image

```bash
# If installed as a tool:
dj-flux2 "a cute cat sitting on a windowsill"

# If using local development:
uv run generate_image.py "a cute cat sitting on a windowsill"
# Or: python generate_image.py "a cute cat sitting on a windowsill"
```

Output: `output.png`

## Usage

### GUI Interface (Interactive)

For interactive experimentation with real-time preview, use the GUI tool:

```bash
# If installed as a tool (recommended):
dj-flux2-gui

# Or from the repository directory:
uv run gui_generate.py
# Or: python gui_generate.py (with activated venv)
```

**Note:** Install with `uv tool install --editable .` (editable mode) so the tool can access the `flux2/` submodule at runtime.

**The GUI provides:**
- **Two modes**: Text-to-Image and Image-to-Image
- **Side-by-side preview**: See input and output images together (img2img mode)
- **Auto-load workflow**: Switch to img2img and the last generated image loads as input automatically
- **Per-mode prompts**: Separate prompt history for txt2img and img2img — switching modes never clears your work
- **Model selector**: Choose between Klein 4B, 9B, base-4B, and base-9B; guidance/steps grey out automatically for distilled models
- **All parameters**: Prompt, width, height, steps, guidance, seed
- **Upscaling support**: Optional Lanczos or Real-ESRGAN upscaling
- **Model caching**: First generation loads models (~15-25s), subsequent generations are 5-10x faster (2-5s)
- **Auto-download**: Models download automatically on first use with clear error messages if access is not yet granted
- **Memory management**: Smart VRAM handling prevents out-of-memory errors
- **Easy experimentation**: Adjust parameters and regenerate instantly
- **Save when ready**: Only save images you like
- **Seed management**: Copy and reuse seeds for reproducibility
- **Modern Qt6 interface**: Professional, cross-platform GUI framework
- **Unload models**: Free GPU memory when finished generating

**Perfect for:**
- Experimenting with different prompts
- Fine-tuning generation parameters
- Quick iteration on img2img transformations
- Visual comparison of results
- Batch generation workflows (models stay loaded)

### Command Line Options

This project supports multiple ways to run commands:

**Option 1: Installed as a tool (simplest):**
```bash
dj-flux2 "prompt"
dj-flux2-gui                  # Launch GUI
dj-flux2-upscale -i input.png -o output.png
dj-flux2-download
```
- ✅ Available globally, no need to cd into project directory
- ✅ No virtual environment activation needed
- ✅ Clean command names
- ✅ All tools including GUI work globally

**Option 2: Using `uv run` (for development):**
```bash
uv run generate_image.py "prompt"
uv run gui_generate.py  # GUI
uv run upscale_image.py -i input.png -o output.png
```
- ✅ No need to activate virtual environment
- ✅ Good for testing changes during development
- ✅ No installation required

**Option 3: Activated virtual environment**
```bash
source .venv/bin/activate  # Activate once per terminal session
python generate_image.py "prompt"
python gui_generate.py  # GUI
python upscale_image.py -i input.png -o output.png
```
- ✅ Traditional Python workflow
- ✅ Works with any tool (not just uv)

> **Note:** Always install with `uv tool install --editable .` (not plain `uv tool install .`) so the flux2 submodule is accessible at runtime.

### Text-to-Image

```bash
# Basic usage (default 512x512)
dj-flux2 "a majestic mountain landscape at sunset"
# Or: uv run generate_image.py "a majestic mountain landscape at sunset"

# With custom output path
dj-flux2 "a robot" -o my_robot.png

# Native high resolution (requires 16GB+ VRAM)
dj-flux2 "detailed portrait" -W 1024 -H 1024

# Reproducible with seed
dj-flux2 "abstract art" -S 42
```

### High-Resolution Generation

**Three approaches for high-resolution images:**

**1. Native generation** (if you have 16GB+ VRAM):
```bash
dj-flux2 "detailed cityscape" -W 1024 -H 1024
```

**2. Lanczos upscaling** (fast, CPU-based):
```bash
dj-flux2 "detailed cityscape" --upscale 2 --upscale-method lanczos
dj-flux2 "epic landscape" --upscale 4 --upscale-method lanczos
```

**3. AI upscaling with Real-ESRGAN** (best quality, default when using --upscale):
```bash
# First, download the AI upscaling models (~128 MB)
dj-flux2-download --upscale-only

# Generate and AI upscale to 1024x1024 in one step (default)
dj-flux2 "detailed cityscape" -o output.png --upscale 2

# Generate and AI upscale to 2048x2048 in one step
dj-flux2 "epic landscape" -o output.png --upscale 4

# Or explicitly use Lanczos for faster CPU-based upscaling
dj-flux2 "detailed cityscape" -o output.png --upscale 2 --upscale-method lanczos
```

> **Note:** When using `--upscale`, only the final upscaled image is saved to your output path. A temporary intermediate image is used during processing and automatically deleted.

**Upscale existing images:**
```bash
# Lanczos (fast, CPU)
dj-flux2-upscale -i input.png -o output.png --scale 2

# AI upscaling (better quality, GPU)
dj-flux2-upscale -i input.png -o output.png --scale 2 --method realesrgan
```

**Quality comparison:**
- **Real-ESRGAN** (default): AI-based upscaling with superior detail recovery, ~2-5s, best for recovering fine textures and faces
- **Lanczos**: Professional-grade traditional upscaling (Photoshop/GIMP), fast (~0.5s), excellent when speed matters

### Image-to-Image

Transform existing images:

```bash
# Turn photo into oil painting
dj-flux2 "oil painting in impressionist style" -i photo.jpg -o painting.png

# Convert to pencil sketch
dj-flux2 "pencil sketch, detailed line art, black and white" -i portrait.jpg -o sketch.png

# Style transfer
dj-flux2 "watercolor painting with soft colors" -i landscape.jpg -o watercolor.png
```

### All Options

```bash
dj-flux2 --help
dj-flux2-gui              # Launch GUI (no options)
dj-flux2-upscale --help
dj-flux2-download --help
```

```
Options:
  prompt              Text prompt (required)
  -m, --model         Model to use (default: flux.2-klein-4b)
                        flux.2-klein-4b       4B distilled (fast, steps/guidance fixed)
                        flux.2-klein-9b       9B distilled (higher quality)
                        flux.2-klein-base-4b  4B base (guidance meaningful, ~50 steps)
                        flux.2-klein-base-9b  9B base
  -i, --input         Input image for img2img
  -o, --output        Output path (default: output.png)
  -W, --width         Width in pixels (default: 512)
  -H, --height        Height in pixels (default: 512)
  -s, --steps         Denoising steps (default: model default)
  -g, --guidance      Guidance scale (default: model default; ignored for distilled models)
  -S, --seed          Random seed for reproducibility
  --upscale           Upscale output by 2x or 4x (uses AI by default)
  --upscale-method    realesrgan (AI, default) or lanczos (fast, CPU)
```

## Performance

### Generation Speed (RTX 4070, 512x512)

| Operation | Time | Notes |
|-----------|------|-------|
| **CLI - Cold start** | ~15s | Loads models from disk |
| **CLI - Warm generation** | ~7s | Models still in HF cache |
| **GUI - First generation** | ~15-25s | Loads and caches models in RAM |
| **GUI - Subsequent** | **2-5s** | Uses cached models (5-10x faster!) |

### Model Caching (GUI Only)

The GUI uses intelligent model caching for dramatic speedups:

- **First generation**: Loads models from disk (~15-25s)
- **Subsequent generations**: Reuses cached models (~2-5s) 
  - **Memory trade-off**: Keeps ~4 GB of models in RAM for instant access
- **VRAM management**: Automatically shuffles models between CPU/GPU to prevent OOM errors
- **User control**: "Unload Models" button frees memory when finished

**How it works:**
1. Models load once and stay in system RAM
2. During generation, models move to GPU only when needed
3. After generation, large transformer returns to CPU to free VRAM
4. Next generation transfers from RAM → GPU (~500ms vs 15s from disk)

This makes the GUI perfect for iterative workflows where you generate multiple images in one session.

### Memory Usage

- **VRAM**: ~4-5 GB peak during generation (models are shuffled on/off GPU in stages), ~0.2 GB idle
- **RAM**: ~8 GB (includes ~4 GB cached models when GUI has generated images)
- **Disk cache**: ~13 GB (Hugging Face model cache)

### Resolution Limits

**Native generation:**
- **Minimum**: 64x64
- **Default**: 512x512
- **Maximum**: Limited by GPU VRAM
  - 12GB VRAM: Up to 1024x1024
  - 16GB VRAM: Up to 1280x1280
  - 24GB+ VRAM: Up to 1792x1792 (model maximum)

**With upscaling:**

| Method | Speed (2x) | Quality | VRAM | Setup Required |
|--------|-----------|---------|------|----------------|
| **Real-ESRGAN** (default) | ~2-5s | Superior | +2GB | Download models |
| **Lanczos** | ~0.5s | Excellent | None (CPU) | None |

- No VRAM limitations for final output size
- Generate at 512x512, upscale to 1024x1024 or 2048x2048
- Real-ESRGAN provides best results for recovering fine details, textures, and faces

**Note:** All dimensions must be multiples of 16.

## Architecture

```
Text Prompt → Qwen3 encoder → FLUX.2 Klein transformer → VAE Decoder → Image
                   ↑                      ↑
          4B: Qwen3-4B-FP8       4B: 7.4 GB transformer
          9B: Qwen3-8B-FP8       9B: larger transformer
```

**Distilled models** (klein-4b, klein-9b): guidance scale is baked into the weights — the parameter has no effect. Steps and guidance are fixed at 4 and 1.0 respectively. The GUI greys these controls out automatically.

**Base models** (klein-base-4b, klein-base-9b): use classifier-free guidance (two forward passes per step). Guidance and steps are fully meaningful; defaults are 4.0 and 50.

The VRAM choreography keeps at most one large model on GPU at a time: the text encoder and autoencoder are swapped to CPU during the transformer's denoising pass, then the transformer returns to CPU before the autoencoder decodes the result.

## Troubleshooting

### Out of Memory

If native generation causes OOM errors at large resolutions, use upscaling instead:

```bash
# Instead of a very large native resolution:
# Use: --upscale 2 (generates 512x512, upscales to 1024x1024)
uv run generate_image.py "prompt" --upscale 2
```

Or reduce native resolution:
```bash
uv run generate_image.py "prompt" -W 512 -H 512
```

### Slow Generation

Check GPU is being used:
```python
import torch
print(torch.cuda.is_available())
```

### Model Download Fails

Models download automatically on first use. If it fails:

1. Accept the license for the model you're using at `huggingface.co/black-forest-labs`
2. Accept the FLUX.2-dev license (shared autoencoder): https://huggingface.co/black-forest-labs/FLUX.2-dev
3. Check your token has "gated repos" read access
4. Re-login: `huggingface-cli login`

The GUI shows a clear error message with the exact URL to visit if access is not yet granted.

## Learning Resources

- **Black Forest Labs**: https://github.com/black-forest-labs/flux2
- **FLUX.2 Blog**: https://bfl.ai/blog/flux2-klein

## Project Structure

```
dj-flux2/
├── generate_image.py      # Main inference script + ModelCache
├── gui_generate.py        # GUI application logic (PySide6/Qt6)
├── gui_components.py      # GUI widget layout
├── upscale_image.py       # Lanczos and Real-ESRGAN upscaling
├── download_models.py     # Real-ESRGAN model downloader (FLUX models auto-download)
├── flux2/                 # BFL submodule (git, do not modify)
│   └── src/flux2/         # Core FLUX.2 architecture code
├── pyrightconfig.json     # IDE/LSP config (resolves flux2 imports)
├── pyproject.toml         # Dependencies and entry points
├── uv.lock                # Locked dependency versions
└── README.md              # This file
```

## Why This Project?

This is a **minimal, educational** implementation of FLUX.2 Klein:

✅ **Minimal**: Small codebase (+ modern Qt6 GUI!)  
✅ **Fast**: 4-step generation  
✅ **Clear**: Easy to understand  
✅ **Complete**: Text-to-image + image-to-image + interactive GUI  
✅ **Maintainable**: BFL code via submodule  

**Not included** (from full BFL repo):
- ❌ Prompt upsampling (optional feature)
- ❌ API client (not needed for local)
- ❌ Training code (inference only)
- ❌ Watermarking (can be added)

## License

This project: MIT License

Black Forest Labs flux2 submodule: See `flux2/LICENSE.md`

FLUX.2 Klein 4B model: Apache 2.0

FLUX.2-dev model: Non-Commercial License

## Contributing

Improvements welcome! Please:
1. Keep it minimal
2. Add tests for new features
3. Update documentation

## Credits

- **Black Forest Labs** for FLUX.2 Klein
- **Antirez** for flux2.c inspiration
- **Hugging Face** for model hosting

---

Built for learning and experimentation. For production, use the official BFL API.
