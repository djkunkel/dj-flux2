# Agent Guidelines for dj-flux2

> Instructions for AI coding agents working in this repository

## Project Overview

Minimal FLUX.2 Klein 4B inference implementation using official BFL code as a git submodule. Focus: simplicity, clarity, and educational value.

**Key features:**
- Text-to-image and image-to-image generation
- Interactive PySide6 (Qt6) GUI for experimentation
- Traditional (Lanczos) and AI-based (Real-ESRGAN via Spandrel) upscaling
- Minimal dependencies, educational code structure

## Build/Run Commands

### Installation
```bash
# First time setup
uv venv
uv pip install -e .

# Using traditional venv
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Running Scripts
```bash
# Launch interactive GUI
uv run gui_generate.py

# Generate image (text-to-image)
uv run generate_image.py "prompt" -o output.png

# Native high resolution (requires sufficient VRAM)
uv run generate_image.py "prompt" -W 1024 -H 1024 -o output.png

# Lanczos upscaling approach (VRAM efficient)
uv run generate_image.py "prompt" --upscale 2 -o output.png

# Image-to-image transformation
uv run generate_image.py "style prompt" -i input.png -o output.png

# Upscale existing image
uv run upscale_image.py -i input.png -o output.png --scale 2

# Download models
uv run download_models.py

# With activated venv
python gui_generate.py  # GUI
python generate_image.py "prompt"
```

### Testing
```bash
# No formal test suite yet - manual testing only
# Test text-to-image
uv run generate_image.py "test image" -o test.png -S 42

# Test img2img
uv run generate_image.py "pencil sketch" -i test.png -o sketch.png

# Verify imports work
uv run python -c "from flux2.util import load_flow_model; print('✓ OK')"
```

### Linting/Formatting
```bash
# No linters configured - keep code clean manually
# Follow existing code style in generate_image.py and download_models.py
```

### Testing AI Upscaling
```bash
# Download Real-ESRGAN models (only needed once; FLUX models auto-download on first use)
uv run download_models.py

# Test Lanczos (default, fast)
uv run upscale_image.py -i test.png -o test_lanczos_2x.png --scale 2

# Test Real-ESRGAN (AI quality)
uv run upscale_image.py -i test.png -o test_ai_2x.png --scale 2 --method realesrgan

# Test integrated with generate_image.py
uv run generate_image.py "test" --upscale 2 --upscale-method realesrgan
```

## Code Style Guidelines

### Import Order
```python
# 1. Shebang (executable scripts only)
#!/usr/bin/env python

# 2. Module docstring
"""Brief description of what this module does"""

# 3. Standard library imports (alphabetical)
import os
import sys
from pathlib import Path

# 4. Third-party imports (alphabetical)
import torch
from einops import rearrange
from PIL import Image

# 5. flux2 submodule imports
# (sys.path.insert is done at module top in scripts that need it)
from flux2.util import load_flow_model
from flux2.sampling import denoise
```

### Type Hints
Use type hints for function signatures, especially public APIs:
```python
def generate_image(
    prompt: str,
    output_path: str = "output.png",
    width: int = 512,
    seed: int = None,  # Optional types can be None
) -> None:
    """Always include docstrings with Args section"""
```

### Naming Conventions
- **Functions**: `snake_case` (e.g., `generate_image`, `load_model`)
- **Variables**: `snake_case` (e.g., `model_name`, `input_image`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_STEPS = 4`)
- **Classes**: `PascalCase` — several exist: `ModelCache`, `GuiState`, `GenerationWorker`,
  `FluxGUI`, `ImagePreviewPanel`, `LeftConfigPanel`, `RightImagePanel`. Avoid adding more
  unless genuinely necessary.
- **Private methods/helpers**: `_leading_underscore` is used throughout the GUI classes (e.g.,
  `_setup_ui`, `_on_mode_change`, `_upscale_realesrgan`). Keep private helpers private.

### Function Design
- Keep functions under 50 lines where possible
- Single responsibility principle
- Clear, descriptive names (not abbreviated)
- Docstrings with Args/Returns sections

### Error Handling
```python
# Explicit error messages for users
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub not installed")
    print("Install dependencies first: pip install -e .")
    sys.exit(1)

# Handle expected failures gracefully
if seed is None:
    seed = torch.randint(0, 2**32, (1,)).item()
```

### Comments
- Use docstrings for functions, not inline comments for obvious code
- Add comments for non-obvious ML operations (e.g., "Move text encoder to CPU to free VRAM")
- Keep comments up to date or remove them

### File Organization
```
Project root only:
├── gui_generate.py      # GUI business logic: FluxGUI, GuiState, GenerationWorker
├── gui_components.py    # GUI widget classes: ImagePreviewPanel, LeftConfigPanel,
│                        #   RightImagePanel, open_image_file_dialog
├── generate_image.py    # Main inference script + ModelCache singleton
├── upscale_image.py     # Upscaling (Lanczos + Real-ESRGAN via Spandrel)
├── download_models.py   # Model downloader (FLUX + Real-ESRGAN)
├── pyrightconfig.json   # IDE/LSP config pointing at flux2/src
├── pyproject.toml       # Dependencies and entry points
├── CLEANUP.md           # Tracks known bugs and fixes (all items currently done)
└── *.md                 # Documentation

Do NOT add:
- test/ directory (no test framework yet)
- src/ directory (scripts stay in root)
- utils.py (keep code in main scripts)
```

## Project-Specific Rules

### 1. Minimal Dependencies
- Only add dependencies that are absolutely required
- Currently: torch, torchvision, transformers, einops, safetensors, pillow, huggingface-hub, accelerate, spandrel, PySide6, triton-windows (Windows only)
- Do NOT add: fire, click, typer, openai, realesrgan (use spandrel instead), basicsr, gradio, streamlit, PyQt6, tkinterdnd2, or other "nice-to-have" packages
- **GUI note**: Uses PySide6 (Qt6) for professional cross-platform GUI with proper threading support
- **Spandrel note**: Use spandrel for AI upscaling instead of realesrgan package to avoid dependency conflicts

### 1.5. Platform Notes: Linux (primary) vs Windows

**Linux is the primary platform.** All development and testing defaults should target Linux. The Python scripts themselves are cross-platform with no platform conditionals.

#### Why CUDA works out-of-the-box on Linux

On Linux x86_64, the PyPI `torch` wheel declares `nvidia-*-cu12` packages and `triton` as pip dependencies. `uv sync` and `uv tool install` both install them automatically — no special indexes or flags needed. CUDA just works.

#### Why Windows needs extra steps

On Windows, the PyPI `torch` wheel is CPU-only (no CUDA). The CUDA-enabled Windows wheel only exists on PyTorch's own index. Two additional things are required:

1. **`[tool.uv.sources]` in `pyproject.toml`** redirects `torch` and `torchvision` to the PyTorch CUDA index for `uv sync` / `uv run`:
   ```toml
   [tool.uv.sources]
   torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'win32'" }]
   torchvision = [{ index = "pytorch-cu128", marker = "sys_platform == 'win32'" }]

   [[tool.uv.index]]
   name = "pytorch-cu128"
   url = "https://download.pytorch.org/whl/cu128"
   explicit = true
   ```

2. **`triton-windows`** is required on Windows because `transformers`' FP8 quantization (used by the Qwen3-4B-FP8 text encoder) imports `triton` at module level. On Linux, `triton` is a torch dependency and installs automatically. On Windows it doesn't exist, so `triton-windows` (the community port) is added as a Windows-only dependency:
   ```toml
   "triton-windows>=3.6.0; sys_platform == 'win32'",
   ```

#### `uv tool install` on Windows

`uv tool install` ignores `pyproject.toml` sources — it is a user-level command that never reads project config. On Linux this doesn't matter because PyPI torch already pulls CUDA deps. On Windows the CUDA index must be passed explicitly:

```bash
uv tool install --editable . \
  --index https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  --reinstall-package torch \
  --reinstall-package torchvision \
  --reinstall-package triton-windows
```

Without these flags, the installed tool will have CPU-only torch and fail with "CUDA GPU not found".

#### Verifying CUDA is active

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Linux expected:   2.10.0 True
# Windows expected: 2.10.0+cu128 True
# Windows broken:   2.10.0+cpu False  ← wrong wheel installed
```

#### Unicode in print() on Windows

The scripts use `✓` and `✗` characters which may raise `UnicodeEncodeError` on Windows terminals using cp1252 encoding. These are intentional — Linux is the primary platform and the symbols should be preserved. Set `PYTHONIOENCODING=utf-8` when testing on Windows if needed. Do NOT replace the symbols to fix a Windows-only cosmetic issue.

### 1.6. AMD GPU Support (ROCm)

**ROCm is supported on Linux only.** Windows ROCm support for PyTorch is not available via the standard pip index.

#### How ROCm works with this codebase

PyTorch's ROCm backend intentionally mirrors the entire `torch.cuda` API. This means:

- `torch.cuda.is_available()` returns `True` on AMD GPUs when the ROCm wheel is installed
- `torch.device("cuda")` maps to the HIP/ROCm device
- `.cuda()` calls work transparently
- `torch.cuda.OutOfMemoryError` is raised on GPU OOM just like NVIDIA

**No Python code changes are required.** The entire generation pipeline works as-is on ROCm.

#### The one exception: flux2 submodule

`flux2/src/flux2/sampling.py:72` contains a hardcoded `.cuda()` call inside the submodule (which cannot be modified). On ROCm this call is transparently translated by PyTorch's HIP layer, so it works correctly without any workaround.

#### Installing the ROCm torch wheel

The standard Linux PyPI `torch` wheel is CUDA-only. AMD users must replace it after the base install:

```bash
# Step 1: Normal install
uv venv && uv pip install -e .
# OR: uv tool install --editable .

# Step 2: Swap in ROCm-enabled torch (ROCm 6.4 — current stable)
uv pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/rocm6.4
```

ROCm wheels are hosted at `https://download.pytorch.org/whl/rocm6.4` (or `rocm6.3` for the previous stable). Check https://pytorch.org/get-started/locally/ for the latest supported version.

**Important:** Do NOT add a `[tool.uv.sources]` ROCm entry to `pyproject.toml`. There is no pip/uv environment marker that can auto-detect GPU type (NVIDIA vs AMD), so any such entry would apply to all Linux users and break NVIDIA installs. The ROCm wheel must always be installed manually as a post-install step.

#### Verifying ROCm is active

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# NVIDIA expected: 2.x.x  True
# AMD expected:    2.x.x+rocm6.4  True
# AMD broken:      2.x.x  True   ← CUDA wheel, not ROCm; generation may silently fail
# AMD broken:      2.x.x  False  ← ROCm not installed or GPU not supported
```

#### Supported AMD hardware

| GPU family | ROCm support |
|---|---|
| RX 9000 series (RDNA 4) | Yes — ROCm 7.2+ |
| RX 7000 series (RDNA 3) | Yes — ROCm 6.x / 7.x |
| RX 6000 series (RDNA 2) | Yes — ROCm 6.x |
| RX 5000 series (RDNA 1) | Limited / community patches only |
| Vega / older | Not supported |
| Ryzen APU iGPUs (890M, etc.) | **Not supported** — ROCm requires a discrete GPU |

**Minimum VRAM:** 8 GB (same as NVIDIA). 12 GB recommended for 1024×1024 generation.

#### triton on AMD/Linux

On Linux with the ROCm torch wheel, `triton` is included as a dependency of torch (same as the CUDA wheel). No extra packages are needed. The `triton-windows` entry in `pyproject.toml` is irrelevant for AMD/Linux users.

### 2. Submodule Respect
- NEVER modify files in `flux2/` directory (it's a git submodule)
- `pyrightconfig.json` points the LSP at `flux2/src` — do not remove it
- Scripts insert `flux2/src` into `sys.path` at the top for runtime resolution
- Document any BFL API usage in comments

### 3. Documentation First
- Update README.md for user-facing changes
- Update QUICK-START.md for workflow changes
- Keep docs in sync with code changes
- No stale documentation

### 3.5. AI Upscaling (Real-ESRGAN via Spandrel)
- Models are downloaded via `download_models.py --upscale-only`
- Stored in `models/realesrgan/` directory (not in HF cache)
- Use Spandrel's `ModelLoader().load_from_file()` to load models
- Models are cached in memory after first load (`_realesrgan_model_cache` in `upscale_image.py`)
- Default to Lanczos unless user explicitly requests Real-ESRGAN
- Real-ESRGAN requires a GPU — NVIDIA (CUDA) or AMD (ROCm) both work (raises `RuntimeError` if no GPU available — do NOT `sys.exit()` from library functions)
- Test both methods when modifying upscale code
- Model files: `RealESRGAN_x2plus.pth` (64 MB), `RealESRGAN_x4plus.pth` (64 MB)

### 3.6. Qt Widget Lifetime (GUI)
- `RightImagePanel` owns `input_preview` and `output_preview` as persistent widgets that are
  reused across mode switches (txt2img ↔ img2img).
- When clearing `main_layout` during `set_mode()`, **never** call `setParent(None)` on these
  panel widgets — that transfers ownership to Qt and schedules their C++ objects for deletion.
  Any subsequent access raises `RuntimeError: Internal C++ object already deleted`.
- The correct pattern: reparent persistent widgets back to `self` to keep them alive, then call
  `deleteLater()` only on transient containers (e.g., the `QSplitter`), after rescuing the
  panels from them first. See `gui_components.py:set_mode()` for the reference implementation.

### 4. Git Commit Style
```bash
# Format: <type>: <subject>
# Types: feat, fix, docs, test, chore

# Examples:
git commit -m "feat: add batch generation support"
git commit -m "fix: handle missing input image gracefully"
git commit -m "docs: update README with new parameters"
git commit -m "chore: update dependencies in uv.lock"
```

### 5. Configuration
- Use argparse for CLI (already in generate_image.py)
- Default values should be sensible (512x512, 4 steps, guidance 1.0)
- No config files (keep everything explicit)

## What NOT to Do

❌ Add heavyweight frameworks (Django, Flask, FastAPI)  
❌ Create abstractions for 2 scripts (no BaseGenerator class)  
❌ Add type checking (mypy, pyright) without discussion  
❌ Reformat existing code without reason  
❌ Add async/await (synchronous is fine)  
❌ Create "utils.py" for one helper function  
❌ Add logging framework (print() is sufficient)  
❌ Modify flux2/ submodule files  
❌ Commit output/ directory images  

## When Making Changes

1. **Test manually** with actual image generation
2. **Update uv.lock** if dependencies change: `uv lock`
3. **Update docs** in same commit as code changes
4. **Keep it minimal** - when in doubt, don't add it
5. **Ask questions** - prefer discussion over assumption

## Example: Adding a Feature

```python
# Good: Simple, clear, follows existing patterns
def save_image_metadata(img: Image, prompt: str, seed: int) -> None:
    """Save generation metadata to image EXIF data"""
    exif = Image.Exif()
    exif[ExifTags.Base.ImageDescription] = f"Prompt: {prompt} | Seed: {seed}"
    return exif

# Bad: Over-engineered, adds unnecessary complexity
class ImageMetadataManager:
    def __init__(self, config: dict):
        self.config = config
        self._setup_handlers()
    ...
```

## Resources

- **Main Docs**: README.md (user guide)
- **Quick Start**: QUICK-START.md (rapid onboarding)
- **Bug history**: CLEANUP.md (static analysis findings — all resolved)
- **BFL Source**: flux2/src/flux2/ (reference only, do not modify)

---

**Philosophy**: Keep it simple, clear, and educational. This is a learning-focused project, not production infrastructure.
