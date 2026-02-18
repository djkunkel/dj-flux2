# Cleanup Plan: Performance, Memory & Architecture Issues

Findings from a full static analysis of all Python source files. Tasks are ordered by impact —
work top to bottom and the most dangerous bugs are fixed first.

---

## Critical — Fix These First

### 1. ~~`gui_generate.py:266-273` — Race condition: old worker thread not stopped before replacement~~ ✓ DONE

**Problem:** When the user clicks Generate while a previous generation is still running, the old
`GenerationWorker` thread has its signals disconnected and `deleteLater()` called, but is never
`.wait()`-ed. The old thread keeps running in the background and will continue using the GPU
concurrently with the new thread. This can silently corrupt GPU state, cause an OOM crash, or
emit a stale `finished` signal that overwrites the new generation's result.

**Before (`gui_generate.py:265-273`):**
```python
# Clean up previous worker thread
if self.state.worker is not None:
    # Disconnect signals to prevent stale connections
    self.state.worker.progress.disconnect()
    self.state.worker.finished.disconnect()
    self.state.worker.error.disconnect()
    # Schedule for deletion
    self.state.worker.deleteLater()
    self.state.worker = None
```

**After:**
```python
# Clean up previous worker thread
if self.state.worker is not None:
    # Disconnect signals to prevent stale connections
    self.state.worker.progress.disconnect()
    self.state.worker.finished.disconnect()
    self.state.worker.error.disconnect()
    # Request stop and block until the thread actually finishes
    # This prevents concurrent GPU access from two generation threads
    self.state.worker.quit()
    self.state.worker.wait()
    self.state.worker.deleteLater()
    self.state.worker = None
```

Also consider disabling the Generate button while generation is running (it already is —
`generate_btn.setEnabled(False)`) so this code path is only hit programmatically. The
`.quit()` + `.wait()` guard is still necessary as a safety net.

---

### 2. ~~`gui_components.py:77-81` — Dangling buffer: `QImage` holds a raw pointer to a collected bytes object~~ ✓ DONE

**Problem:** `QImage(data, ...)` takes a **raw pointer** to the bytes buffer — it does not copy
the data. The local variable `data` goes out of scope when Python moves on to the next line
after `QPixmap.fromImage(qimage)`. In CPython, reference counting keeps `data` alive as long as
the enclosing function frame is alive, so this usually works in practice. However, Qt's docs
explicitly state the caller must keep the buffer alive for the lifetime of the `QImage`. If the
`QImage` is ever accessed after the function returns (e.g., in a resize event callback before
`fromImage()` completes in a compiled build), this is a use-after-free / memory corruption.

**Before (`gui_components.py:74-81`):**
```python
# Load image and convert to QPixmap
img = Image.open(image_path)
img = img.convert("RGB")
data = img.tobytes("raw", "RGB")
qimage = QImage(
    data, img.width, img.height, img.width * 3, QImage.Format_RGB888
)
self.original_pixmap = QPixmap.fromImage(qimage)
```

**After:**
```python
# Load image and convert to QPixmap
img = Image.open(image_path)
img = img.convert("RGB")
data = img.tobytes("raw", "RGB")
qimage = QImage(
    data, img.width, img.height, img.width * 3, QImage.Format_RGB888
)
# .copy() forces Qt to take ownership of the pixel data,
# so `data` can be safely garbage-collected after this line
self.original_pixmap = QPixmap.fromImage(qimage.copy())
```

---

### 3. ~~`upscale_image.py:84,97,101` — `sys.exit()` called from a library function kills the GUI~~ ✓ DONE

**Problem:** `_upscale_realesrgan()` calls `sys.exit(1)` for three error conditions: no CUDA,
model file missing, and invalid scale. This function is imported and called as a library by both
`gui_generate.py` (via `GenerationWorker.run()`) and `generate_image.py`. Calling `sys.exit()`
from inside a worker thread will terminate the entire GUI process rather than letting the caller
catch the error and show a dialog.

**Before (`upscale_image.py:80-101`):**
```python
# Check CUDA availability
if not torch.cuda.is_available():
    print("Error: Real-ESRGAN requires CUDA GPU")
    print("Use --method lanczos for CPU upscaling")
    sys.exit(1)

# ...

if not model_path.exists():
    print(f"Error: Model not found: {model_path}")
    print("\nPlease download the Real-ESRGAN models first:")
    print("  uv run download_models.py --upscale-models")
    sys.exit(1)

if scale not in [2, 4]:
    print(f"Error: Real-ESRGAN only supports 2x and 4x upscaling")
    sys.exit(1)
```

**After:**
```python
# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError(
        "Real-ESRGAN requires a CUDA GPU. "
        "Use method='lanczos' for CPU upscaling."
    )

# ...

if not model_path.exists():
    raise RuntimeError(
        f"Real-ESRGAN model not found: {model_path}\n"
        "Download it first: uv run download_models.py --upscale-models"
    )

if scale not in [2, 4]:
    raise ValueError(
        f"Real-ESRGAN only supports 2x and 4x upscaling, got: {scale}"
    )
```

Also remove the `import sys` at the top of `_upscale_realesrgan()` since it would no longer
be needed inside that function (verify `sys` isn't used elsewhere in the file first).

---

### 4. ~~`pyproject.toml:41` — `gui_components` missing from `py-modules` breaks non-editable installs~~ ✓ DONE

**Problem:** `gui_components.py` is listed nowhere in `pyproject.toml`. When the package is
installed non-editably (`pip install dj-flux2` or `pip install .`), the file is not included in
the wheel, and `dj-flux2-gui` will immediately crash with `ModuleNotFoundError: No module named
'gui_components'`. This only passes in development because `pip install -e .` leaves the source
directory on `sys.path`.

**Before (`pyproject.toml:41`):**
```toml
py-modules = ["generate_image", "upscale_image", "download_models", "gui_generate"]
```

**After:**
```toml
py-modules = ["generate_image", "upscale_image", "download_models", "gui_generate", "gui_components"]
```

---

### 5. ~~`gui_generate.py:394-417` — Temp file leaked in `_clear()` due to wrong operation order~~ ✓ DONE

**Problem:** `_clear()` calls `self.state.reset_images()` on line 400, which sets
`self.state.generated_image_path = None`. The file-deletion block on lines 406-408 then checks
`if self.state.generated_image_path and os.path.exists(...)`, which is always `False` because
the path was just cleared. The temp PNG for the last generation is never deleted.

**Before (`gui_generate.py:394-417`):**
```python
def _clear(self):
    """Reset form and preview"""
    # Reset left panel
    self.left_panel.reset_to_defaults()

    # Reset state
    self.state.reset_images()          # <-- clears generated_image_path

    # Clear previews
    self.right_panel.clear_previews()

    # Clean up temp files
    if self.state.generated_image_path and os.path.exists(  # <-- always False
        self.state.generated_image_path
    ):
        try:
            os.remove(self.state.generated_image_path)
        except:
            pass

    # Clean up previous temp file too
    if self.state.previous_temp_file:
        self.state.cleanup_temp_file()
        self.state.previous_temp_file = None
    ...
```

**After:**
```python
def _clear(self):
    """Reset form and preview"""
    # Reset left panel
    self.left_panel.reset_to_defaults()

    # Delete temp files BEFORE resetting state (paths are cleared by reset_images)
    if self.state.generated_image_path and os.path.exists(
        self.state.generated_image_path
    ):
        try:
            os.remove(self.state.generated_image_path)
        except Exception:
            pass

    if self.state.previous_temp_file:
        self.state.cleanup_temp_file()
        self.state.previous_temp_file = None

    # Now safe to clear state (paths already used above)
    self.state.reset_images()

    # Clear previews
    self.right_panel.clear_previews()
    ...
```

Also change the bare `except:` on line 411 to `except Exception:` — a bare `except` also
catches `KeyboardInterrupt` and `SystemExit`, which should never be silently swallowed.

---

## Medium Priority

### 6. ~~`upscale_image.py:108-111` — Real-ESRGAN model reloaded from disk on every generation~~ ✓ DONE

**Problem:** Every time upscaling is requested, `_upscale_realesrgan()` calls
`ModelLoader().load_from_file(str(model_path))` which reads 64 MB from disk, deserializes the
model weights, and re-allocates GPU memory. In the GUI, each "Generate + Upscale" click pays
this cost repeatedly. This adds 1-3 seconds of unnecessary overhead per run.

**Fix:** Add a module-level cache dict at the top of `upscale_image.py`, keyed by `(scale,
model_path)`:

**After (`upscale_image.py` — add near top, below imports):**
```python
# Module-level cache: maps scale -> loaded spandrel model
# Avoids reloading the 64 MB .pth file on every call
_realesrgan_model_cache: dict = {}
```

**After (inside `_upscale_realesrgan()`, replace the load block):**
```python
# Load model (cached after first call to avoid re-reading 64 MB from disk)
cache_key = (scale, str(model_path))
if cache_key not in _realesrgan_model_cache:
    print(f"Loading Real-ESRGAN {scale}x model from {model_path}...")
    model = ModelLoader().load_from_file(str(model_path))
    if not isinstance(model, ImageModelDescriptor):
        raise RuntimeError(
            f"Loaded model is not an image model descriptor: {model_path}"
        )
    model.cuda().eval()
    _realesrgan_model_cache[cache_key] = model
else:
    print(f"Using cached Real-ESRGAN {scale}x model")

model = _realesrgan_model_cache[cache_key]
```

This task also fixes issue #7 below (the `assert isinstance` → explicit `if/raise`).

---

### 7. ~~`upscale_image.py:110` — `assert isinstance(model, ImageModelDescriptor)` is fragile~~ ✓ DONE

**Problem:** Python `assert` statements are completely disabled when the interpreter is run with
the `-O` (optimize) flag (`python -O gui_generate.py`). More importantly, an `AssertionError`
has no useful message for end users and is easily confused with a programming error.
`assert isinstance(...)` should only be used for invariants that can never fail in correct code;
here it can legitimately fail if the wrong `.pth` file is pointed at.

**Before:**
```python
model = ModelLoader().load_from_file(str(model_path))
assert isinstance(model, ImageModelDescriptor)
```

**After:**
```python
model = ModelLoader().load_from_file(str(model_path))
if not isinstance(model, ImageModelDescriptor):
    raise RuntimeError(
        f"Loaded model is not an image model: {model_path}\n"
        "Make sure the .pth file is a Real-ESRGAN image model."
    )
```

(This is incorporated into fix #6 above if both tasks are done together.)

---

### 8. ~~`generate_image.py:181` — No CUDA availability check; crashes with opaque error~~ ✓ DONE

**Problem:** `device = torch.device("cuda")` is set unconditionally. On a machine without a
CUDA GPU (or with CUDA unavailable due to a driver issue), the error surfaces deep inside
`model.to(device)` as `AssertionError` or `RuntimeError: No CUDA GPUs are available` —
without any guidance on what to do. A early, explicit check produces a much better user
experience.

**Before (`generate_image.py:181`):**
```python
device = torch.device("cuda")
```

**After:**
```python
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not found. FLUX.2 Klein requires a CUDA-capable GPU.\n"
        "Check your NVIDIA driver and CUDA installation."
    )
device = torch.device("cuda")
```

---

### 9. ~~`generate_image.py:75` — `ae_model_name` is ignored in the cache invalidation check~~ ✓ DONE

**Problem:** `load_models()` returns the cached dict when `self._model_name == model_name`
without checking `ae_model_name`. If the function were ever called with a different AE model
name (e.g., during experimentation), the stale autoencoder would be silently returned and
generation would produce corrupted output.

**Before (`generate_image.py:75`):**
```python
if self._models and self._model_name == model_name:
```

**After:**
```python
if (
    self._models
    and self._model_name == model_name
    and self._ae_model_name == ae_model_name
):
```

---

### 10. ~~`gui_generate.py:411` — Bare `except:` swallows `KeyboardInterrupt` and `SystemExit`~~ ✓ DONE

**Problem:** A bare `except:` (no exception type) is broader than `except Exception:`. It
catches `BaseException` subclasses including `KeyboardInterrupt`, `SystemExit`, and
`GeneratorExit` — signals that are meant to propagate. This makes the application impossible to
interrupt via Ctrl+C while inside the `_clear()` try block.

**Before (`gui_generate.py:411`):**
```python
        except:
            pass
```

**After:**
```python
        except Exception:
            pass
```

---

## Low Priority / Code Quality

### 11. `download_models.py:30` — Dead import: `login` imported but never used

**Before:**
```python
from huggingface_hub import hf_hub_download, login
```

**After:**
```python
from huggingface_hub import hf_hub_download
```

---

### 12. `generate_image.py:360-361` — Deferred imports should be at module top

**Problem:** `import tempfile` and a duplicate `import os` appear inside the `if args.upscale:`
block in `main()`. Both are standard library modules with negligible import cost. Deferring them
buys nothing and violates PEP 8 / the project's own import-order guidelines.

**Before (`generate_image.py:359-361`):**
```python
if args.upscale:
    import tempfile
    import os
```

**After:** Move both to the top of the file with the other stdlib imports. Remove the
conditional `import os` entirely since `os` is already imported on line 11. Add `import
tempfile` to the stdlib block at the top:

```python
import argparse
import gc
import os
import tempfile   # <-- add here
import torch
...
```

---

### 13. `pyproject.toml:26-27` — Duplicate entry points

**Problem:** Both `dj-flux2` and `dj-flux2-generate` map to `generate_image:main`. One of
these provides no additional value and just clutters the installed scripts directory.

**Before:**
```toml
dj-flux2 = "generate_image:main"
dj-flux2-generate = "generate_image:main"
```

**After:** Remove `dj-flux2` (the shorter alias) and keep the more descriptive name, or keep
`dj-flux2` as the canonical short name and remove `dj-flux2-generate`. Either way, delete the
duplicate. Example keeping the short alias:

```toml
dj-flux2 = "generate_image:main"
dj-flux2-upscale = "upscale_image:main"
dj-flux2-download = "download_models:main"
dj-flux2-gui = "gui_generate:main"
```

---

### 14. `generate_image.py:127-132` — `get_memory_estimate()` returns a hardcoded string

**Problem:** The method always returns `"~4.2 GB"` regardless of which models are actually
loaded. It cannot detect if, for example, only the AE was loaded or if a different-sized model
is in the cache.

**After (a simple improvement — compute from actual tensors):**
```python
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
    gb = total_bytes / (1024 ** 3)
    return f"~{gb:.1f} GB"
```

---

### 15. `gui_components.py:119-123` — Dead method `set_placeholder()` with a Qt6-incompatible null check

**Problem:** `set_placeholder()` is defined but never called anywhere in the codebase (confirmed
by grepping all files). Additionally, its implementation uses `if not
self.preview_label.pixmap()` which is unreliable in Qt6/PySide6 — `pixmap()` returns an empty
`QPixmap()` (not `None`) when no pixmap is set, so the truthiness check does not work as
intended. The correct Qt6 check would be `self.preview_label.pixmap().isNull()`.

**Fix:** Remove the method entirely since it is unused. If it is needed in the future, rewrite
with the correct Qt6 null check:

```python
# Remove this entire method (never called):
def set_placeholder(self, text: str):
    """Update placeholder text"""
    self.placeholder_text = text
    if not self.preview_label.pixmap():
        self.preview_label.setText(text)
```

---

### 16. `gui_components.py:444-448` — `QHBoxLayout` created in `set_mode()` is not explicitly deleted

**Problem:** Each time `set_mode(True)` is called (switching to img2img), a new `QHBoxLayout`
(`h_layout`) is created and added to `self.main_layout`. When `set_mode(False)` is later called,
the `while self.main_layout.count()` loop calls `takeAt(0)` to remove items. For widget items,
`item.widget().setParent(None)` triggers deletion. But for layout items (returned when the item
is a sub-layout, not a widget), `item.widget()` returns `None`, so the layout is removed from
the parent but never explicitly deleted or scheduled for deletion.

**After:** Explicitly clean up sub-layouts during the drain loop:

```python
# Clear current layout
while self.main_layout.count():
    item = self.main_layout.takeAt(0)
    if item.widget():
        item.widget().setParent(None)
    elif item.layout():
        # Drain and delete sub-layouts (e.g., h_layout from img2img mode)
        sub = item.layout()
        while sub.count():
            sub_item = sub.takeAt(0)
            if sub_item.widget():
                sub_item.widget().setParent(None)
        sub.deleteLater()
```

---

### 17. `pyproject.toml:6` — `<3.15` Python upper bound will require unnecessary maintenance

**Problem:** `requires-python = ">=3.10,<3.15"` means the package will refuse to install on
Python 3.15 when it's released, even if there are no actual incompatibilities. None of the
dependencies (PySide6, torch, transformers) have known Python 3.15 issues. Upper bounds on
Python versions are an anti-pattern that causes breakage for users who upgrade Python.

**Before:**
```toml
requires-python = ">=3.10,<3.15"
```

**After:**
```toml
requires-python = ">=3.10"
```

---

### 18. `generate_image.py:209-211` — Dead code: `width`/`height` None-check is never reached

**Problem:** Lines 209-211 check `if width is None or height is None` and fall back to the
input image's dimensions. But `width` and `height` have default values of `512` in the function
signature, the CLI `argparse` defaults are also `512`, and the GUI always passes explicit values
from the combo boxes. There is no caller path that passes `None`. This is dead code that adds
noise and could mislead future readers into thinking `None` is a valid input.

**Fix:** Remove the dead block. If `None` support is desired as a future feature, document it
with a `# TODO` comment instead:

```python
# Before (remove these lines):
if width is None or height is None:
    width, height = img_ctx[0].size
    print(f"  Using input image dimensions: {width}x{height}")
```

---

### 19. `gui_components.py:222-224` — Steps spinbox max of 50 has no warning for the 4-step model

**Problem:** The FLUX.2 Klein 4B model is a distilled 4-step model. Setting steps above ~8
produces slower results with no quality improvement (and can actually degrade quality). The
spinbox allows up to 50 with no guidance, which is a UX trap for new users.

**Fix:** Add a tooltip to the steps spinbox:

```python
self.steps_spin = QSpinBox()
self.steps_spin.setRange(1, 50)
self.steps_spin.setValue(4)
self.steps_spin.setToolTip(
    "Recommended: 4 steps (Klein is a distilled model).\n"
    "Values above 8 are slower with no quality benefit."
)
```

---

### 20. `generate_image.py:270` — JPEG-specific kwargs passed to PNG save

**Problem:** `img.save(output_path, exif=exif_data, quality=95, subsampling=0)` passes
`quality=95` and `subsampling=0`, which are JPEG-specific parameters. For PNG output (the
default and recommended format), Pillow silently ignores these. The `exif=exif_data` argument
also behaves differently for PNG vs JPEG — PNG stores EXIF in a separate chunk while JPEG
embeds it in the JFIF/APP1 marker. This is not a bug for the current default-PNG use case, but
it is misleading.

**Fix:** Remove the JPEG-specific parameters from the PNG save call. If JPEG output support is
desired, add a format branch:

```python
# Before:
img.save(output_path, exif=exif_data, quality=95, subsampling=0)

# After (PNG-safe, still embeds EXIF for PNG and JPEG):
img.save(output_path, exif=exif_data)
```

---

## Summary Table

| # | File | Line(s) | Severity | Issue |
|---|------|---------|----------|-------|
| 1 | `gui_generate.py` | 266-273 | **Critical** | Old worker not waited on; concurrent GPU access |
| 2 | `gui_components.py` | 77-81 | **Critical** | `QImage` raw buffer pointer can dangle |
| 3 | `upscale_image.py` | 84, 97, 101 | **Critical** | `sys.exit()` in library kills GUI process |
| 4 | `pyproject.toml` | 41 | **Critical** | `gui_components` missing from `py-modules` |
| 5 | `gui_generate.py` | 400-412 | **Critical** | Temp file leaked: path cleared before deletion |
| 6 | `upscale_image.py` | 108-111 | Medium | ESRGAN model reloaded from disk every call |
| 7 | `upscale_image.py` | 110 | Medium | `assert isinstance` disabled by `-O` flag |
| 8 | `generate_image.py` | 181 | Medium | No CUDA check; crashes with opaque error |
| 9 | `generate_image.py` | 75 | Medium | `ae_model_name` ignored in cache key |
| 10 | `gui_generate.py` | 411 | Medium | Bare `except:` swallows `KeyboardInterrupt` |
| 11 | `download_models.py` | 30 | Low | Dead import: `login` unused |
| 12 | `generate_image.py` | 360-361 | Low | Deferred stdlib imports inside conditional |
| 13 | `pyproject.toml` | 26-27 | Low | Duplicate entry points for `generate_image:main` |
| 14 | `generate_image.py` | 127-132 | Low | `get_memory_estimate()` returns hardcoded string |
| 15 | `gui_components.py` | 119-123 | Low | Dead `set_placeholder()` with broken Qt6 null check |
| 16 | `gui_components.py` | 444-448 | Low | `QHBoxLayout` not deleted on mode switch |
| 17 | `pyproject.toml` | 6 | Low | `<3.15` upper bound will cause unnecessary breakage |
| 18 | `generate_image.py` | 209-211 | Low | Dead code: `width`/`height` None-check unreachable |
| 19 | `gui_components.py` | 222-224 | Low | Steps spinbox max=50 has no tooltip warning |
| 20 | `generate_image.py` | 270 | Low | JPEG `quality`/`subsampling` kwargs silently ignored for PNG |
