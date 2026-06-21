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
# One-time setup per machine — writes .gpu-backend (gitignored) and installs wheels
./setup cuda         # NVIDIA GPU (CUDA 12.6)
./setup rocm         # AMD GPU, ROCm 7.2 stable (RDNA 2/3)
./setup rocm-nightly # AMD RDNA4 (gfx120X / RX 9700), ROCm 7.14 nightly (multi-arch)
./setup cpu          # CPU only — for testing without a GPU

# Using traditional venv (CUDA example)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[cuda]" --index-url https://download.pytorch.org/whl/cu126
```

### Running Scripts

```bash
./run gui                                        # Launch interactive GUI
./run generate "prompt" -o output.png            # Text-to-image
./run generate "prompt" -W 1024 -H 1024          # High resolution
./run generate "prompt" --upscale 2              # With upscaling
./run img2img "style prompt" -i input.png        # Image-to-image
./run upscale -i input.png -o output.png         # Upscale existing image
./run download                                   # Download models
./run config                                     # Show current config
./run config model flux.2-klein-9b               # Set default model
./run config width 768                           # Set default width
./run config height 768                          # Set default height
./run config steps 4                             # Set default steps
./run config guidance 1.0                        # Set default guidance
./run version                                    # Print the dj-flux2 version
```

`./run` reads `.gpu-backend` written by `./setup` to determine the correct
uv extra and env vars. Never run `uv run` or `uv sync` directly — without
`--extra <backend>` uv will replace the GPU-specific torch wheel with the
PyPI default (CUDA on Linux, CPU on Windows/macOS).

Alternatively, activate the venv directly — uv's auto-sync is bypassed entirely:

```bash
source .venv/bin/activate

python gui_generate.py
python generate_image.py "prompt"
```

### Testing
```bash
# No formal test suite yet - manual testing only
# Test text-to-image
./run generate "test image" -o test.png -S 42

# Test img2img
./run img2img "pencil sketch" -i test.png -o sketch.png

# Verify imports work
./run python -c "from flux2.util import load_flow_model; print('✓ OK')"
```

### Linting/Formatting
```bash
# No linters configured - keep code clean manually
# Follow existing code style in generate_image.py and download_models.py
```

### Testing AI Upscaling
```bash
# Download Real-ESRGAN models (only needed once; FLUX models auto-download on first use)
./run download

# Test Lanczos (default, fast)
./run upscale -i test.png -o test_lanczos_2x.png --scale 2

# Test Real-ESRGAN (AI quality)
./run upscale -i test.png -o test_ai_2x.png --scale 2 --method realesrgan

# Test integrated with generate
./run generate "test" --upscale 2 --upscale-method realesrgan
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
├── generate_image.py    # Main inference script + ModelCache singleton + read_config()
├── upscale_image.py     # Upscaling (Lanczos + Real-ESRGAN via Spandrel)
├── download_models.py   # Model downloader (FLUX + Real-ESRGAN)
├── pyrightconfig.json   # IDE/LSP config pointing at flux2/src
├── pyproject.toml       # Dependencies and entry points
├── setup                # One-time install script (writes .gpu-backend + ~/.local/bin/dj-flux2)
├── run                  # Launch script (reads .gpu-backend, dispatches commands)
├── CLEANUP.md           # Tracks known bugs and fixes (all items currently done)
└── *.md                 # Documentation

Gitignored machine-local files (never commit):
├── .gpu-backend         # Written by ./setup — stores the active GPU backend
└── .dj-flux2.conf       # Written by ./run config — stores user defaults

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

#### How the torch wheel is selected: per-extra indexes

The backend is chosen by which optional-dependency extra you install, not by
the host platform. `pyproject.toml` declares three extras (`cpu`, `cuda`,
`rocm`) and routes `torch`/`torchvision` to the matching PyTorch index via
`[tool.uv.sources]` keyed on `extra` (not `marker`):

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cpu",    extra = "cpu"  },
  { index = "pytorch-cu126",  extra = "cuda" },
  { index = "pytorch-rocm72", extra = "rocm" },
]
```

The extras are declared mutually exclusive under `[tool.uv]` `conflicts`, so
uv resolves each independently. `./setup <backend>` runs the correct
`uv sync --extra <backend>`; `./run` reads `.gpu-backend` and passes the same
`--extra` on every invocation. Install exactly one extra — see §1.6 for the
rationale.

Note: PyTorch 2.12 stable dropped CUDA 12.8 (cu128) in favour of CUDA 12.6
(cu126) and 13.0 (cu130). The project uses cu126.

#### Windows specifics

On Windows the PyPI `torch` wheel is CPU-only; the CUDA wheel lives on
PyTorch's own index. The `cuda` extra above already routes there, so
`./setup cuda` (or `pip install -e ".[cuda]" --index-url
https://download.pytorch.org/whl/cu126`) installs the correct wheel — no
platform markers needed.

`triton-windows` is a Windows-only dependency because `transformers`' FP8
quantization (used by the Qwen3-4B-FP8 text encoder) imports `triton` at
module level. On Linux `triton` ships as a torch dependency and installs
automatically; on Windows it does not exist, so the community port is pulled
in via a marker in the base dependencies:

```toml
"triton-windows>=3.6.0; sys_platform == 'win32'",
```

ROCm is not available on Windows — use `cpu` or `cuda` there.

#### Verifying CUDA is active

```bash
./run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Linux expected:   2.12.0 True
# Windows expected: 2.12.0+cu126 True
# Windows broken:   2.12.0+cpu False  ← wrong wheel installed
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

Use the `rocm` extra — uv will pull the correct wheel from the PyTorch ROCm 7.2 index automatically:

```bash
uv sync --extra rocm
```

ROCm wheels are hosted at `https://download.pytorch.org/whl/rocm7.2`. The `rocm` extra in `pyproject.toml` points there explicitly via `[tool.uv.sources]`, so no manual index flags are needed.

**Important:** Install exactly one of `--extra cpu`, `--extra cuda`, or `--extra rocm`. Installing multiple extras simultaneously will cause uv to attempt resolving torch from conflicting indexes.

#### RDNA4 (gfx120X / RX 9700) — ROCm 7.14 nightly (multi-arch)

ROCm 7.2 stable does not support RDNA4 correctly. Use the `rocm-nightly` backend instead:

```bash
./setup rocm-nightly
```

This installs torch from AMD's multi-arch nightly index at
`https://rocm.nightlies.amd.com/whl-multi-arch/` using the device-extras
model. The base torch wheel contains only host code; GPU-specific kernels are
pulled from a separate `amd-torch-device-gfx1201` package automatically. No
manual date pin is needed — pip resolves the latest compatible daily build.

`rocm-nightly` is intentionally **not** a uv extra in `pyproject.toml`. The
nightly index uses pre-release local version specifiers that uv cannot resolve
at lock time. `./setup rocm-nightly` handles it with `uv pip install
--index-url` after first syncing the stable `rocm` extra for all other deps.

`./run` automatically sets the required RDNA4 env vars when `.gpu-backend`
is `rocm-nightly`:

| Variable | Value | Reason |
|---|---|---|
| `LD_LIBRARY_PATH` | `.venv/numa-shim` appended | libnuma.so shim so torch's rocSHMEM dlopen succeeds |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `1` | **No-op on ROCm 7.14** — AOTriton is enabled by default |
| `MIOPEN_FIND_MODE` | `FAST` | **Do not use** — default DYNAMIC_HYBRID builds FindDb cache for repeated resolutions |
| `MIOPEN_FIND_ENFORCE` | `NONE` | **No longer needed** — was paired with FAST to avoid a GPU hang on older MIOpen |
| `PYTORCH_ALLOC_CONF` | `garbage_collection_threshold:0.8,max_split_size_mb:512` | **No-op on ROCm 7.14** — no VRAM fragmentation observed |

**Note:** `MIOPEN_FIND_MODE=FAST` was previously used to avoid a hang on gfx1201 VAE
shapes. On MIOpen 3.5.2 (ROCm 7.14) the default DYNAMIC_HYBRID mode is faster for
repeated resolutions because it builds up the FindDb cache. FAST is only useful
for one-off resolutions you'll never re-run. The vars above are all commented out
in `run` and are listed here for historical context only.

**To update to a newer nightly:** simply re-run `./setup rocm-nightly` — it
always pulls the latest build. Browse available packages at:
`https://rocm.nightlies.amd.com/whl-multi-arch/`

#### Verifying ROCm is active

```bash
./run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# NVIDIA expected: 2.x.x  True
# AMD expected:    2.x.x+rocm7.2  True
# AMD broken:      2.x.x  True   ← CUDA wheel, not ROCm; generation may silently fail
# AMD broken:      2.x.x  False  ← ROCm not installed or GPU not supported
```

#### Supported AMD hardware

| GPU family | ROCm support |
|---|---|
| RX 9000 series (RDNA 4) | Yes — `rocm-nightly` backend (ROCm 7.14 nightly) |
| RX 7000 series (RDNA 3) | Yes — `rocm` backend (ROCm 7.2 stable) |
| RX 6000 series (RDNA 2) | Yes — `rocm` backend (ROCm 7.2 stable) |
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

#### Checking for submodule updates

When working on the project, it's worth checking whether the upstream BFL
`flux2` submodule has new commits, and incorporating them **if they are safe**:

1. Check what's new upstream:
   ```bash
   git -C flux2 fetch
   git -C flux2 log --oneline HEAD..origin/main
   ```
2. Review the incoming changes before pulling them in. Safe to incorporate
   means: no breaking changes to the BFL APIs we call (`load_flow_model`,
   `load_text_encoder`, `load_ae`, `denoise`, `FLUX2_MODEL_INFO`, etc.), no new
   required dependencies we don't already have, and nothing that changes model
   behavior in a way we don't want. When in doubt, skip the update and ask.
3. If safe, bump the pin and commit the pointer (do NOT edit submodule files):
   ```bash
   git -C flux2 checkout origin/main   # or a specific reviewed commit
   git add flux2
   git commit -m "chore: update flux2 submodule to <short-sha> (<summary>)"
   ```
4. Test generation end-to-end after bumping — new submodule revisions can
   change `FLUX2_MODEL_INFO` (e.g. added models) or the sampling path.

Only upgrade when actively working on the project, never as a drive-by change,
and never pull in an update you haven't reviewed.

### 3. Documentation First
- Update README.md for user-facing changes
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

### 4.5. Versioning and Release Tags

The project follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.
The single source of truth is the `version` field in `pyproject.toml`. The first
tagged release is `1.2.0`, cut from `main` once the multi-GPU work is merged.

- **MAJOR** — incompatible/breaking changes to the CLI, config format, or behavior.
- **MINOR** — new functionality added in a backwards-compatible way (e.g. a new
  model, flag, or upscaling method).
- **PATCH** — backwards-compatible bug fixes and documentation-only changes.

**Bump the version and cut a git tag only when we are ready for a new version
to be available to others — not on every commit.** Day-to-day commits land on
feature branches without touching `version`. **Releases are cut only from
`main`**, after the work has been merged. When we decide a set of merged changes
is ready to ship as a release:

1. Make sure you are on `main` with the changes merged in: `git checkout main && git pull`.
2. Bump `version` in `pyproject.toml` to the new number.
3. Commit it on its own: `git commit -m "chore: bump version to X.Y.Z"`.
4. Tag that commit with a `v`-prefixed annotated tag and push both:
   ```bash
   git tag -a vX.Y.Z -m "Release X.Y.Z"
   git push origin main vX.Y.Z
   ```

The tag name (`vX.Y.Z`) must match the `pyproject.toml` `version` (`X.Y.Z`).
Do not create release tags speculatively, from feature branches, or for
work-in-progress.

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
- **Bug history**: CLEANUP.md (static analysis findings — all resolved)
- **BFL Source**: flux2/src/flux2/ (reference only, do not modify)

---

**Philosophy**: Keep it simple, clear, and educational. This is a learning-focused project, not production infrastructure.
