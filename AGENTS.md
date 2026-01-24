# Agent Guidelines for dj-flux2

> Instructions for AI coding agents working in this repository

## Project Overview

Minimal FLUX.2 Klein 4B inference implementation using official BFL code as a git submodule. Focus: simplicity, clarity, and educational value.

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
# Generate image (text-to-image)
uv run generate_image.py "prompt" -o output.png

# Image-to-image transformation
uv run generate_image.py "style prompt" -i input.png -o output.png

# Download models
uv run download_models.py

# With activated venv
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
uv run python -c "import sys; sys.path.insert(0, 'flux2/src'); from flux2.util import load_flow_model; print('✓ OK')"
```

### Linting/Formatting
```bash
# No linters configured - keep code clean manually
# Follow existing code style in generate_image.py and download_models.py
```

## Code Style Guidelines

### Import Order
```python
# 1. Shebang (executable scripts only)
#!/usr/bin/env python

# 2. Module docstring
"""Brief description of what this module does"""

# 3. Standard library imports
import sys
from pathlib import Path

# 4. Third-party imports (alphabetical)
import torch
from einops import rearrange
from PIL import Image

# 5. flux2 submodule imports (after sys.path.insert)
sys.path.insert(0, "flux2/src")
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
- **Classes**: `PascalCase` (none in this project - avoid adding)
- **Private**: `_leading_underscore` (avoid in this minimal project)

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
├── generate_image.py    # Main script (keep under 200 lines)
├── download_models.py   # Model downloader (keep focused)
├── pyproject.toml       # Dependencies only (minimal)
└── *.md                 # Documentation

Do NOT add:
- test/ directory (no test framework yet)
- src/ directory (scripts stay in root)
- utils.py (keep code in main scripts)
```

## Project-Specific Rules

### 1. Minimal Dependencies
- Only add dependencies that are absolutely required
- Currently: torch, transformers, einops, safetensors, pillow, huggingface-hub, accelerate
- Do NOT add: fire, click, typer, openai, or other "nice-to-have" packages

### 2. Submodule Respect
- NEVER modify files in `flux2/` directory (it's a git submodule)
- Always import from flux2 after `sys.path.insert(0, "flux2/src")`
- Document any BFL API usage in comments

### 3. Documentation First
- Update README.md for user-facing changes
- Update QUICK-START.md for workflow changes
- Keep docs in sync with code changes
- No stale documentation

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

- **Main Docs**: README.md (user guide), MODS-README.md (technical)
- **Quick Start**: QUICK-START.md (rapid onboarding)
- **Dependencies**: DEPENDENCIES.md (package management)
- **BFL Source**: flux2/src/flux2/ (reference only, do not modify)

---

**Philosophy**: Keep it simple, clear, and educational. This is a learning-focused project, not production infrastructure.
