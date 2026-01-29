# Migrate flux2 Submodule to Dependency

> Plan to replace git submodule with pip-installable dependency for better IDE support

**Date**: January 25, 2026  
**Status**: Ready to implement  
**Estimated Time**: 1 hour

---

## Problem Statement

**Current Issue**:
- Git submodule requires `sys.path.insert()` hack in all scripts
- IDE can't resolve imports (no autocomplete, no type hints)
- LSP errors for all `flux2.*` imports
- Harder to modify flux2 code (e.g., for AMD GPU support)

**Goal**:
- ✅ IDE autocomplete and type hints work
- ✅ No `sys.path.insert()` hacks needed
- ✅ Can modify flux2 code via fork
- ✅ Clean Python packaging
- ✅ Still get upstream updates from BFL

---

## Solution Overview

**Replace git submodule with pip dependency pointing to your fork**

```
Current:
  dj-flux2 → flux2/ (git submodule) → BFL's repo

New:
  dj-flux2 → flux2 (pip dependency) → your fork → (sync) BFL's repo
```

This gives you:
1. **IDE support** - flux2 installed in venv like any package
2. **Modification ability** - your fork can have device compatibility patches
3. **Upstream sync** - can merge BFL updates into your fork
4. **Clean imports** - no path manipulation needed

---

## Phase 1: Fork flux2 Repository

### Step 1.1: Fork on GitHub (2 minutes)

1. Go to https://github.com/black-forest-labs/flux2
2. Click "Fork" button
3. Create fork in your account: `djkunkel/flux2`

### Step 1.2: Clone Your Fork Locally (1 minute)

```bash
cd ~/repos  # or wherever you keep repos
git clone https://github.com/djkunkel/flux2.git
cd flux2
```

### Step 1.3: Add Upstream Remote (1 minute)

```bash
# Add BFL's repo as upstream for future syncing
git remote add upstream https://github.com/black-forest-labs/flux2.git
git fetch upstream

# Verify remotes
git remote -v
# origin    https://github.com/djkunkel/flux2.git (your fork)
# upstream  https://github.com/black-forest-labs/flux2.git (BFL)
```

---

## Phase 2: Add Device Compatibility Patches

### Step 2.1: Create Device-Compatible Branch (1 minute)

```bash
cd ~/repos/flux2  # Your fork
git checkout -b device-compatible
```

### Step 2.2: Patch sampling.py (5 minutes)

**File**: `src/flux2/sampling.py`

**Find line 72**:
```python
encoded = ae.encode(img[None].cuda())[0]
```

**Replace with**:
```python
# Device-compatible version (works with CUDA and ROCm)
device = next(ae.parameters()).device
encoded = ae.encode(img[None].to(device))[0]
```

**Full context** (lines 69-73):
```python
    # Encode each reference image
    encoded_refs = []
    for img in img_ctx_prep:
        # Device-compatible: infer device from model instead of hardcoding cuda
        device = next(ae.parameters()).device
        encoded = ae.encode(img[None].to(device))[0]
        encoded_refs.append(encoded)
```

### Step 2.3: Patch util.py (Optional, 5 minutes)

If you want to allow CPU fallback in model loaders:

**File**: `src/flux2/util.py`

**Find line 74** (load_flow_model function):
```python
def load_flow_model(model_name: str, debug_mode: bool = False, device: str | torch.device = "cuda") -> Flux2:
```

**Change default**:
```python
def load_flow_model(
    model_name: str, 
    debug_mode: bool = False, 
    device: str | torch.device = None  # Allow auto-detection
) -> Flux2:
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
```

**Repeat for load_text_encoder (line 112) and load_ae (line 117)**

### Step 2.4: Commit and Push (2 minutes)

```bash
git add src/flux2/sampling.py src/flux2/util.py
git commit -m "feat: add device compatibility for AMD GPU/ROCm support

- Replace hardcoded .cuda() with device inference
- Allow auto-detection of CUDA/ROCm/CPU
- Maintains backward compatibility with existing code
"

git push origin device-compatible
```

### Step 2.5: Test Your Fork (5 minutes)

```bash
# Test that your fork works
cd ~/repos/flux2
pip install -e src/

# Quick test in Python
python -c "from flux2.util import FLUX2_MODEL_INFO; print('OK')"
```

---

## Phase 3: Update dj-flux2 to Use Dependency

### Step 3.1: Remove Git Submodule (2 minutes)

```bash
cd /var/mnt/extra/repos/dj-flux2

# Deinitialize and remove submodule
git submodule deinit -f flux2
git rm -f flux2
rm -rf .git/modules/flux2

# Verify removal
git status
# Should show:
#   deleted:    flux2
#   modified:   .gitmodules
```

### Step 3.2: Update pyproject.toml (3 minutes)

**File**: `pyproject.toml`

**Add flux2 as dependency**:
```toml
[project]
name = "dj-flux2"
version = "0.1.0"
description = "Minimal FLUX.2 Klein 4B inference with CUDA support"
readme = "README.md"
requires-python = ">=3.10,<3.15"
authors = [
  { name = "DJ", email = "dj@example.com" }
]
license = "MIT"

dependencies = [
  "torch>=2.8.0",
  "torchvision>=0.23.0",
  "einops>=0.8.1",
  "transformers>=4.56.1",
  "safetensors>=0.4.5",
  "pillow>=10.0.0",
  "huggingface-hub>=0.36.0",
  "accelerate>=1.0.0",
  "spandrel>=0.4.0",
  "PySide6>=6.6.0",
  # flux2 from your fork with device compatibility
  "flux2 @ git+https://github.com/djkunkel/flux2.git@device-compatible#subdirectory=src",
]

[project.scripts]
dj-flux2 = "generate_image:main"
dj-flux2-generate = "generate_image:main"
dj-flux2-upscale = "upscale_image:main"
dj-flux2-download = "download_models:main"
dj-flux2-gui = "gui_generate:main"

[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[tool.uv]
python-preference = "only-managed"

[tool.setuptools]
# Include the main scripts as modules
py-modules = ["generate_image", "upscale_image", "download_models", "gui_generate", "gui_components"]
packages = []
```

**Key changes**:
- Added `flux2 @ git+...` dependency
- Uses your fork: `djkunkel/flux2`
- Uses `device-compatible` branch
- Specifies `subdirectory=src` (where flux2's setup.py lives)

### Step 3.3: Remove sys.path Hacks (10 minutes)

**Update these files**:

#### File 1: `generate_image.py`

**OLD** (lines 12-15):
```python
# Add flux2/src to path (handle both local and installed as tool)
script_dir = Path(__file__).parent.resolve()
flux2_src = script_dir / "flux2" / "src"
sys.path.insert(0, str(flux2_src))

from flux2.util import load_flow_model, load_text_encoder, load_ae
```

**NEW**:
```python
# flux2 is now installed as a dependency - no path manipulation needed
from flux2.util import load_flow_model, load_text_encoder, load_ae
```

**Full updated imports** (lines 1-25):
```python
#!/usr/bin/env python
"""Simple image generation script for FLUX.2 Klein 4B with CUDA support"""

import argparse
import torch
from einops import rearrange
from PIL import Image, ExifTags
from pathlib import Path
import sys
import os

# flux2 is installed as a package dependency
from flux2.util import load_flow_model, load_text_encoder, load_ae
from flux2.sampling import (
    batched_prc_txt,
    batched_prc_img,
    denoise,
    get_schedule,
    scatter_ids,
    encode_image_refs,
)


def generate_image(
    prompt: str,
    output_path: str = "output.png",
```

#### File 2: `gui_generate.py`

**OLD** (lines 11-15):
```python
# Add script directory and flux2/src to path (handle both local and installed as tool)
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))  # For gui_components import
flux2_src = script_dir / "flux2" / "src"
sys.path.insert(0, str(flux2_src))
```

**NEW**:
```python
# gui_components is in the same directory (handled by setuptools)
# flux2 is installed as a package dependency
```

**Full updated imports** (lines 1-34):
```python
#!/usr/bin/env python
"""PySide6 GUI for FLUX.2 Klein 4B image generation"""

import sys
import os
from pathlib import Path
import tempfile
from datetime import datetime
from typing import Optional

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QFileDialog,
    QMessageBox,
    QSplitter,
)
from PySide6.QtCore import Qt, QThread, Signal
from PIL import Image

# Import UI components (from same directory)
from gui_components import LeftConfigPanel, RightImagePanel

# Import existing generation functions
from generate_image import generate_image
from upscale_image import upscale_image
```

#### File 3: `download_models.py` (if it imports flux2)

Check if this file imports from flux2:
```bash
grep -n "flux2" download_models.py
```

If it does, remove the `sys.path.insert()` hack similar to above.

### Step 3.4: Reinstall with New Dependencies (2 minutes)

```bash
cd /var/mnt/extra/repos/dj-flux2

# Reinstall with new dependency
uv pip install -e .

# This will:
# 1. Clone your flux2 fork
# 2. Checkout device-compatible branch
# 3. Install flux2 as package
# 4. Install dj-flux2 in editable mode
```

### Step 3.5: Verify IDE Support (2 minutes)

**Test in your IDE**:
1. Open `generate_image.py`
2. Hover over `load_flow_model` - should show type hints
3. Ctrl+Click (or Cmd+Click) on `load_flow_model` - should jump to definition
4. Type `from flux2.` - should show autocomplete

**Test imports work**:
```bash
uv run python -c "from flux2.util import load_flow_model; print('✓ Import works')"
```

---

## Phase 4: Update Documentation

### Step 4.1: Update README.md (5 minutes)

**Find the installation section** and update:

**OLD**:
```markdown
git clone https://github.com/djkunkel/dj-flux2.git
cd dj-flux2
git submodule update --init --recursive

uv venv
uv pip install -e .
```

**NEW**:
```markdown
git clone https://github.com/djkunkel/dj-flux2.git
cd dj-flux2

# flux2 is installed automatically as a dependency
uv venv
uv pip install -e .
```

**Add note about fork**:
```markdown
### About flux2 Dependency

This project uses a [device-compatible fork of flux2](https://github.com/djkunkel/flux2/tree/device-compatible) 
that enables AMD GPU (ROCm) support alongside NVIDIA CUDA. The fork is automatically 
installed when you run `pip install -e .`

**Changes in fork**:
- Device auto-detection (CUDA/ROCm/CPU)
- Removed hardcoded `.cuda()` calls
- 100% compatible with BFL's official flux2

The fork syncs regularly with [Black Forest Labs' upstream](https://github.com/black-forest-labs/flux2).
```

### Step 4.2: Update AGENTS.md (3 minutes)

**Find "Submodule Respect" section** and update:

**OLD**:
```markdown
### 2. Submodule Respect
- NEVER modify files in `flux2/` directory (it's a git submodule)
- Always import from flux2 after `sys.path.insert(0, "flux2/src")`
- Document any BFL API usage in comments
```

**NEW**:
```markdown
### 2. flux2 Dependency
- We use a fork of BFL's flux2 with device compatibility patches
- Fork: https://github.com/djkunkel/flux2 (branch: device-compatible)
- Installed as pip dependency, not git submodule
- To modify flux2 code:
  1. Make changes in your local flux2 fork
  2. Test changes: `cd ~/repos/flux2 && pip install -e src/`
  3. Commit and push to your fork
  4. dj-flux2 will pick up changes on next install
```

### Step 4.3: Update QUICK-START.md (2 minutes)

**Simplify installation** (remove submodule instructions):

**OLD**:
```markdown
## Installation (2 commands)

```bash
uv venv
uv pip install -e .
```
```

**NEW** (stays mostly the same, but clarify):
```markdown
## Installation (2 commands)

```bash
uv venv
uv pip install -e .
```

This automatically installs all dependencies including our device-compatible 
fork of flux2 (supports both NVIDIA CUDA and AMD ROCm).
```

### Step 4.4: Update AMD-PLAN.md (2 minutes)

**Update status** since device compatibility is now built-in:

Add at the top:
```markdown
**UPDATE (Jan 25, 2026)**: Device compatibility has been implemented via 
forked flux2 dependency. See [SUBMODULE-TO-DEPENDENCY-PLAN.md](SUBMODULE-TO-DEPENDENCY-PLAN.md) 
for details. The changes in this document are now integrated.
```

---

## Phase 5: Test Everything

### Step 5.1: Test Basic Generation (2 minutes)

```bash
cd /var/mnt/extra/repos/dj-flux2

# Test text-to-image
uv run generate_image.py "test image" -o /tmp/test.png -S 42

# Should work without any import errors
```

### Step 5.2: Test GUI (1 minute)

```bash
uv run gui_generate.py

# Should launch without import errors
# Generate a test image in the GUI
```

### Step 5.3: Test as Installed Tool (2 minutes)

```bash
# Reinstall as tool
uv tool install --editable --force .

# Test all commands
dj-flux2 "test" -o /tmp/test2.png
dj-flux2-gui  # Should launch GUI
```

### Step 5.4: Test IDE (2 minutes)

1. **Restart your IDE/LSP server** (important!)
2. Open `generate_image.py`
3. Verify no red squiggles on `from flux2.util import ...`
4. Test autocomplete: type `load_flow_model.` and see methods
5. Test go-to-definition: Ctrl+Click on `denoise`

---

## Phase 6: Commit Changes

### Step 6.1: Review Changes (2 minutes)

```bash
git status
# Should show:
#   deleted: flux2 (submodule)
#   modified: .gitmodules
#   modified: pyproject.toml
#   modified: generate_image.py
#   modified: gui_generate.py
#   modified: README.md
#   modified: AGENTS.md
#   modified: QUICK-START.md
#   modified: AMD-PLAN.md
```

### Step 6.2: Create Commit (2 minutes)

```bash
git add -A

git commit -m "refactor: replace flux2 submodule with pip dependency

Replace git submodule with pip-installable dependency pointing to our
device-compatible fork. This provides:

- IDE autocomplete and type hints (no more sys.path hacks)
- Device compatibility for AMD GPU/ROCm support
- Cleaner Python packaging
- Easier modification via fork

Changes:
- Remove flux2 git submodule
- Add flux2 as pip dependency from djkunkel/flux2@device-compatible
- Remove sys.path.insert() hacks from all scripts
- Update documentation for new approach
- Simplify installation (no more git submodule update)

Breaking change: Users must reinstall
Before: git submodule update --init && uv pip install -e .
After:  uv pip install -e .
"
```

### Step 6.3: Push to GitHub (1 minute)

```bash
git push origin main
```

---

## Maintenance: Syncing Upstream Changes

### When BFL Updates flux2

**Every 1-3 months** (or when you notice BFL has updates):

```bash
cd ~/repos/flux2  # Your fork

# Fetch BFL's latest changes
git fetch upstream

# Switch to main and merge
git checkout main
git merge upstream/main

# Push updated main to your fork
git push origin main

# Rebase your device-compatible branch
git checkout device-compatible
git rebase main

# Resolve any conflicts (usually none)
# Push updated device-compatible
git push origin device-compatible --force-with-lease

# Test in dj-flux2
cd /var/mnt/extra/repos/dj-flux2
uv pip install --upgrade --force-reinstall \
  "flux2 @ git+https://github.com/djkunkel/flux2.git@device-compatible#subdirectory=src"

# Run tests
uv run generate_image.py "test" -o /tmp/test.png
```

**Estimated time**: 10-15 minutes per sync

---

## Rollback Plan (If Something Goes Wrong)

### Option 1: Revert Commit

```bash
git revert HEAD
git push origin main
```

### Option 2: Restore Submodule

```bash
# Restore .gitmodules
git checkout HEAD~1 .gitmodules

# Re-add submodule
git submodule add https://github.com/black-forest-labs/flux2.git flux2
git submodule update --init --recursive

# Restore old code
git checkout HEAD~1 generate_image.py gui_generate.py pyproject.toml
```

---

## Benefits Summary

### Before (Submodule)
- ❌ IDE can't resolve imports
- ❌ No autocomplete or type hints
- ❌ `sys.path.insert()` hacks everywhere
- ❌ Hard to modify flux2 code
- ⚠️ Submodule management overhead

### After (Dependency)
- ✅ Full IDE support (autocomplete, go-to-def, type hints)
- ✅ Clean imports (no path manipulation)
- ✅ Easy to modify via fork
- ✅ Proper Python packaging
- ✅ Device compatibility built-in (AMD GPU ready)
- ✅ Simpler installation

---

## Troubleshooting

### Issue: "Could not find a version that satisfies flux2"

**Solution**: Install git dependency manually first:
```bash
pip install "git+https://github.com/djkunkel/flux2.git@device-compatible#subdirectory=src"
```

### Issue: IDE still shows red squiggles

**Solution**: 
1. Restart LSP server / IDE
2. Verify flux2 installed: `pip show flux2`
3. Check Python interpreter is correct venv
4. Rebuild IDE index

### Issue: Import errors at runtime

**Solution**:
```bash
# Check flux2 is installed
python -c "import flux2; print(flux2.__file__)"

# Reinstall if needed
pip install --force-reinstall \
  "git+https://github.com/djkunkel/flux2.git@device-compatible#subdirectory=src"
```

### Issue: Changes to fork not reflected

**Solution**:
```bash
# Force reinstall from git
pip install --force-reinstall --no-cache-dir \
  "git+https://github.com/djkunkel/flux2.git@device-compatible#subdirectory=src"
```

---

## Checklist

Use this checklist when implementing:

### Phase 1: Fork flux2
- [ ] Fork black-forest-labs/flux2 on GitHub
- [ ] Clone your fork locally
- [ ] Add upstream remote

### Phase 2: Device Patches
- [ ] Create device-compatible branch
- [ ] Patch sampling.py line 72
- [ ] (Optional) Patch util.py loaders
- [ ] Commit and push to fork
- [ ] Test fork installs: `pip install -e src/`

### Phase 3: Update dj-flux2
- [ ] Remove git submodule
- [ ] Update pyproject.toml dependency
- [ ] Remove sys.path hacks from generate_image.py
- [ ] Remove sys.path hacks from gui_generate.py
- [ ] Reinstall: `uv pip install -e .`
- [ ] Test imports work

### Phase 4: Documentation
- [ ] Update README.md installation section
- [ ] Update AGENTS.md submodule section
- [ ] Update QUICK-START.md
- [ ] Add note to AMD-PLAN.md

### Phase 5: Testing
- [ ] Test generate_image.py
- [ ] Test gui_generate.py
- [ ] Test uv tool install
- [ ] Restart IDE and verify autocomplete

### Phase 6: Commit
- [ ] Review all changes with git status
- [ ] Create descriptive commit
- [ ] Push to GitHub

---

## Timeline

| Phase | Time | 
|-------|------|
| Phase 1: Fork flux2 | 5 min |
| Phase 2: Device patches | 15 min |
| Phase 3: Update dj-flux2 | 20 min |
| Phase 4: Documentation | 12 min |
| Phase 5: Testing | 7 min |
| Phase 6: Commit | 5 min |
| **Total** | **~60 minutes** |

---

## Questions?

- **What if BFL changes flux2's structure?** - Your fork is pinned to device-compatible branch, so won't break. Merge upstream changes when ready.
- **Can others contribute to my fork?** - Yes! Your fork is public. Others can submit PRs to your device-compatible branch.
- **Will this work offline?** - After first install, flux2 is cached locally. Subsequent installs from pip cache work offline.
- **Can I still use uv tool install?** - Yes! Just use `uv tool install --editable .` as before.

---

**Ready to start?** Follow Phase 1 when you're ready to implement this plan!
