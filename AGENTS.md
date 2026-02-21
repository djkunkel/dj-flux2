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
# Download AI upscaling models (only needed once)
uv run download_models.py --upscale-only       # only Real-ESRGAN models
uv run download_models.py --upscale-models     # FLUX + Real-ESRGAN models together

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
- Currently: torch, torchvision, transformers, einops, safetensors, pillow, huggingface-hub, accelerate, spandrel, PySide6
- Do NOT add: fire, click, typer, openai, realesrgan (use spandrel instead), basicsr, gradio, streamlit, PyQt6, tkinterdnd2, or other "nice-to-have" packages
- **GUI note**: Uses PySide6 (Qt6) for professional cross-platform GUI with proper threading support
- **Spandrel note**: Use spandrel for AI upscaling instead of realesrgan package to avoid dependency conflicts

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
- Real-ESRGAN requires CUDA (raises `RuntimeError` if not available — do NOT `sys.exit()` from library functions)
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
