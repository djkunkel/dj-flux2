# Plan: Multi-Image Image-to-Image Support

> Branch: `feat/multi-image-img2img`
> Status: Not started — plan only

## Summary

All FLUX.2 models (Klein 4B, 9B, 9B-KV, Base variants, and dev) natively
support multi-reference image editing. The flux2 submodule's
`encode_image_refs()` already accepts `list[Image.Image]` and handles any
number of reference images. The dj-flux2 project currently hardcodes a single
input image everywhere. This plan widens the existing `input_image` parameter
to accept `str | list[str] | None` (Option C — backward-compatible type
widening).

## How Multi-Image Works in the Submodule

The mechanism is fully built into `flux2/src/flux2/sampling.py`:

1. Multiple PIL images are passed as a `list[Image.Image]` to
   `encode_image_refs()`.
2. Each image is VAE-encoded independently via `ae.encode()`.
3. Each image gets a unique temporal offset in the 4D position space
   (t=10, 20, 30...) so the model can distinguish reference sources.
4. All reference tokens are concatenated into a single sequence.
5. During denoising, reference tokens are concatenated with the noisy output
   tokens at every step.
6. The causal attention mechanism ensures reference tokens only self-attend,
   while output tokens attend to everything (text + all references + output).
7. For `klein-9b-kv`, KV caching avoids reprocessing reference tokens after
   step 0, giving up to 2.66x speedup with 4 reference images.

**No submodule changes are required.**

## Approach: Option C (widen existing parameter)

Keep the parameter name `input_image` but accept `str | list[str] | None`:

- `input_image=None` — text-to-image (no change)
- `input_image="photo.jpg"` — single reference (no change)
- `input_image=["a.jpg", "b.jpg"]` — multi-reference (new)

CLI: `-i img1.jpg -i img2.jpg` (argparse `action="append"`). A single
`-i photo.jpg` still works identically — it produces `["photo.jpg"]`.

## Files to Change

| File | Complexity | Summary |
|------|-----------|---------|
| `generate_image.py` | Low | Widen type, `action="append"`, normalize to list |
| `gui_generate.py` | Low-Medium | `input_image_path` → `input_image_paths: list`, wire signals |
| `gui_components.py` | Medium | Multi-image input section + vertical preview column |
| `serve_api.py` | Low | Widen `input_image_base64` type, handle list of temp files |
| `api_generate.py` | Optional | Add `-i` support (follow-up) |
| `run` | None | No changes needed (forwards args) |

---

## Detailed Changes

### 1. `generate_image.py` — Core generation logic

#### a) Function signature (line 185)

```python
# Before:
input_image: str = None,

# After:
input_image: str | list[str] | None = None,
```

#### b) Docstring (line 198)

```python
# Before:
input_image: Optional input image path for image-to-image generation

# After:
input_image: Optional input image path(s) for image-to-image generation.
    Pass a single path string or a list of paths for multi-reference editing.
```

#### c) Mode detection and printing (lines 228-237)

Normalize to a list internally:

```python
# Normalize input_image to a list (or empty list)
input_images: list[str] = []
if isinstance(input_image, str):
    input_images = [input_image]
elif isinstance(input_image, list):
    input_images = list(input_image)

mode = ("Multi-Ref Image-to-Image" if len(input_images) > 1
        else "Image-to-Image" if input_images
        else "Text-to-Image")
print(f"{model_name} - {mode}")
print(f"Prompt: {prompt}")
for i, path in enumerate(input_images):
    label = f"Input[{i}]" if len(input_images) > 1 else "Input"
    print(f"{label}: {path}")
print(
    f"Size: {width}x{height}, Steps: {num_steps}, "
    f"Guidance: {guidance}{'(fixed)' if is_distilled else ''}, Seed: {seed}"
)
```

#### d) Image encoding (lines 274-280)

```python
# Before:
ref_tokens = None
ref_ids = None
if input_image:
    print("  Encoding reference image...")
    img_ctx = [Image.open(input_image)]
    ref_tokens, ref_ids = encode_image_refs(ae, img_ctx)

# After:
ref_tokens = None
ref_ids = None
if input_images:
    n = len(input_images)
    print(f"  Encoding {n} reference image{'s' if n > 1 else ''}...")
    img_ctx = [Image.open(p) for p in input_images]
    ref_tokens, ref_ids = encode_image_refs(ae, img_ctx)
```

#### e) CLI argparse (lines 421-427)

```python
# Before:
parser.add_argument(
    "-i",
    "--input",
    type=str,
    default=None,
    help="Input image for image-to-image transformation",
)

# After:
parser.add_argument(
    "-i",
    "--input",
    type=str,
    action="append",
    default=None,
    help="Input image(s) for img2img. Repeat for multi-ref: -i img1.jpg -i img2.jpg",
)
```

With `action="append"`: omitted → `None`, one `-i` → `["photo.jpg"]`,
multiple → `["a.jpg", "b.jpg"]`. The existing kwarg line
`input_image=args.input` (line 507) passes the list directly.

#### f) Epilog example (line 407)

Add a multi-image example:

```
  # Multi-reference image-to-image
  python generate_image.py "combine styles" -i photo1.jpg -i photo2.jpg -o combined.png
```

### 2. `gui_generate.py` — GUI business logic

#### a) GenerationParams TypedDict (line 50)

```python
# Before:
input_image: Optional[str]

# After:
input_image: Optional[list[str]]
```

#### b) GuiState (line 66)

```python
# Before:
self.input_image_path: Optional[str] = None

# After:
self.input_image_paths: list[str] = []
```

#### c) `reset_images()` (line 89)

```python
# Before:
self.input_image_path = None

# After:
self.input_image_paths = []
```

#### d) `_load_input_image()` → `_add_input_image()` (lines 324-331)

Replace with three methods:

```python
def _add_input_image(self, filepath: str):
    """Append an image to the reference list and update the UI."""
    self.state.input_image_paths.append(filepath)
    idx = len(self.state.input_image_paths) - 1
    self.left_panel.add_input_image(filepath)
    self.right_panel.input_previews.add_image(filepath)

def _remove_input_image(self, index: int):
    """Remove a reference image by index and update the UI."""
    if 0 <= index < len(self.state.input_image_paths):
        self.state.input_image_paths.pop(index)
        self.left_panel.remove_input_image(index)
        self.right_panel.input_previews.remove_image(index)

def _clear_input_images(self):
    """Remove all reference images."""
    self.state.input_image_paths = []
    self.left_panel.clear_input_images()
    self.right_panel.input_previews.clear()
```

#### e) `_browse_input_image()` (lines 333-337)

```python
def _browse_input_image(self):
    """Open file picker and add the selected image to the reference list."""
    filepath = open_image_file_dialog(self, "Select Reference Image")
    if filepath:
        self._add_input_image(filepath)
```

#### f) `_on_mode_change()` (lines 283-311)

Update `state.input_image_path` → `state.input_image_paths` references:

```python
def _on_mode_change(self, is_img2img: bool):
    self.left_panel.set_input_group_visible(is_img2img)
    self.right_panel.set_mode(is_img2img)

    if is_img2img:
        # Auto-populate from last generated image if no references chosen yet
        if self.state.generated_image_path and not self.state.input_image_paths:
            self._add_input_image(self.state.generated_image_path)
        # Restore output preview
        if self.state.generated_image_path:
            self.right_panel.output_preview.display_image(
                self.state.generated_image_path
            )
    else:
        # Switching to txt2img: clear references
        self._clear_input_images()
```

#### g) `_validate_generation_params()` (line 341)

```python
# Before:
if self.left_panel.is_img2img_mode() and not self.state.input_image_path:

# After:
if self.left_panel.is_img2img_mode() and not self.state.input_image_paths:
```

#### h) `_generate_image()` (lines 370-373)

```python
# Before:
params["input_image"] = (
    self.state.input_image_path if self.left_panel.is_img2img_mode() else None
)

# After:
params["input_image"] = (
    list(self.state.input_image_paths) if self.left_panel.is_img2img_mode() else None
)
```

#### i) Signal wiring (around line 272)

Add connections for new signals:

```python
self.left_panel.browse_clicked.connect(self._browse_input_image)
self.left_panel.image_removed.connect(self._remove_input_image)   # new
self.left_panel.images_cleared.connect(self._clear_input_images)  # new
```

### 3. `gui_components.py` — GUI widgets

This is the most substantial change. Two areas: the left-panel input section
and the right-panel input preview.

#### a) `LeftConfigPanel` — new signals (around line 222)

```python
# Existing:
browse_clicked = Signal()

# Add:
image_removed = Signal(int)   # index of removed image
images_cleared = Signal()     # all images cleared
```

#### b) `_build_input_section()` (lines 298-310)

Replace with a multi-image input area:

```python
def _build_input_section(self) -> QWidget:
    """Input image selector with add/remove support for multi-reference."""
    self._input_group = QGroupBox("Reference Images (for img2img)")
    layout = QVBoxLayout()

    # Button row: Add Image + Clear All
    btn_row = QHBoxLayout()
    self._browse_btn = QPushButton("Add Image...")
    self._clear_images_btn = QPushButton("Clear All")
    self._clear_images_btn.setEnabled(False)
    btn_row.addWidget(self._browse_btn)
    btn_row.addWidget(self._clear_images_btn)
    btn_row.addStretch()
    layout.addLayout(btn_row)

    # Scrollable list of image rows
    self._image_list_widget = QWidget()
    self._image_list_layout = QVBoxLayout(self._image_list_widget)
    self._image_list_layout.setContentsMargins(0, 0, 0, 0)
    self._image_list_layout.setSpacing(2)
    self._image_list_layout.addStretch()  # push rows to top

    scroll = QScrollArea()
    scroll.setWidget(self._image_list_widget)
    scroll.setWidgetResizable(True)
    scroll.setMaximumHeight(120)  # compact — don't dominate left panel
    self._image_scroll = scroll

    # Placeholder label (shown when list is empty)
    self._input_label = QLabel("No images selected")
    self._input_label.setStyleSheet("color: #888;")

    layout.addWidget(self._input_label)
    layout.addWidget(scroll)
    scroll.setVisible(False)  # hidden until first image added

    self._input_group.setLayout(layout)
    self._image_rows: list[QWidget] = []  # track rows for removal
    return self._input_group
```

Each image row is a small widget: thumbnail (32x32) + filename + "X" button.

#### c) New `LeftConfigPanel` public methods

```python
def add_input_image(self, filepath: str):
    """Add an image thumbnail row to the input list."""
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(2, 2, 2, 2)

    # Thumbnail (32x32)
    thumb = QLabel()
    pixmap = QPixmap(filepath).scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    thumb.setPixmap(pixmap)
    thumb.setFixedSize(32, 32)

    # Filename
    name = QLabel(Path(filepath).name)
    name.setStyleSheet("font-size: 11px;")

    # Remove button
    idx = len(self._image_rows)
    remove_btn = QPushButton("X")
    remove_btn.setFixedSize(20, 20)
    remove_btn.setStyleSheet("font-size: 10px; padding: 0;")
    remove_btn.clicked.connect(lambda checked, i=idx: self.image_removed.emit(i))

    row_layout.addWidget(thumb)
    row_layout.addWidget(name, 1)
    row_layout.addWidget(remove_btn)

    # Insert before the stretch at the end
    self._image_list_layout.insertWidget(len(self._image_rows), row)
    self._image_rows.append(row)

    # Update visibility
    self._input_label.setVisible(False)
    self._image_scroll.setVisible(True)
    self._clear_images_btn.setEnabled(True)

def remove_input_image(self, index: int):
    """Remove an image row by index and re-index remaining remove buttons."""
    if 0 <= index < len(self._image_rows):
        row = self._image_rows.pop(index)
        self._image_list_layout.removeWidget(row)
        row.deleteLater()

        # Re-index the remaining remove buttons so their signals emit
        # the correct index after removal.
        for i, r in enumerate(self._image_rows):
            btn = r.findChild(QPushButton)
            if btn:
                btn.clicked.disconnect()
                btn.clicked.connect(lambda checked, idx=i: self.image_removed.emit(idx))

        if not self._image_rows:
            self._input_label.setVisible(True)
            self._image_scroll.setVisible(False)
            self._clear_images_btn.setEnabled(False)

def clear_input_images(self):
    """Remove all image rows."""
    for row in self._image_rows:
        self._image_list_layout.removeWidget(row)
        row.deleteLater()
    self._image_rows.clear()
    self._input_label.setVisible(True)
    self._image_scroll.setVisible(False)
    self._clear_images_btn.setEnabled(False)
```

#### d) Signal wiring in `_connect_internal_signals()`

```python
# Existing:
self._browse_btn.clicked.connect(self.browse_clicked)

# Add:
self._clear_images_btn.clicked.connect(self.images_cleared)
```

#### e) Remove old `set_input_label()` method (line 685-687)

No longer needed — replaced by `add_input_image()` / `clear_input_images()`.
However, `reset_to_defaults()` calls `self._input_label.setText(...)` at
line 763. Update that to call `self.clear_input_images()` instead.

#### f) `RightImagePanel` — multi-image input preview

Replace the single `input_preview` (`ImagePreviewPanel`) with a
`MultiImagePreviewPanel` that shows a scrollable vertical column of image
previews.

```python
class MultiImagePreviewPanel(QWidget):
    """Scrollable vertical column of image previews for multi-reference input."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._previews: list[ImagePreviewPanel] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Reference Images")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        title.setFixedHeight(20)
        layout.addWidget(title)

        # Placeholder (shown when empty)
        self._placeholder = QLabel("(img2img mode only)")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet("color: #888;")
        layout.addWidget(self._placeholder)

        # Scrollable container for image previews
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.setSpacing(5)
        self._container_layout.addStretch()

        self._scroll = QScrollArea()
        self._scroll.setWidget(self._container)
        self._scroll.setWidgetResizable(True)
        self._scroll.setVisible(False)
        layout.addWidget(self._scroll, 1)

    def add_image(self, filepath: str):
        """Add an image preview to the vertical column."""
        preview = ImagePreviewPanel(
            f"Reference {len(self._previews) + 1}",
            "",
            min_size=128,
        )
        preview.display_image(filepath)
        self._container_layout.insertWidget(len(self._previews), preview)
        self._previews.append(preview)
        self._placeholder.setVisible(False)
        self._scroll.setVisible(True)

    def remove_image(self, index: int):
        """Remove an image preview by index and re-title remaining ones."""
        if 0 <= index < len(self._previews):
            preview = self._previews.pop(index)
            self._container_layout.removeWidget(preview)
            preview.deleteLater()
            # Re-title remaining previews
            for i, p in enumerate(self._previews):
                # Update title label (first child QLabel)
                # Implementation detail: find the title label and update
                pass  # TODO: implement re-titling
            if not self._previews:
                self._placeholder.setVisible(True)
                self._scroll.setVisible(False)

    def clear(self):
        """Clear all image previews."""
        for preview in self._previews:
            self._container_layout.removeWidget(preview)
            preview.deleteLater()
        self._previews.clear()
        self._placeholder.setVisible(True)
        self._scroll.setVisible(False)
```

#### g) Update `RightImagePanel._setup_ui()` and `set_mode()`

```python
# Replace:
self.input_preview = ImagePreviewPanel(...)

# With:
self.input_previews = MultiImagePreviewPanel()
```

Update `set_mode()` to show/hide `input_previews` instead of `input_preview`.

Update all references in `gui_generate.py` from `right_panel.input_preview`
to `right_panel.input_previews`.

### 4. `serve_api.py` — HTTP API

#### a) `GenerateRequest` model (line 286)

```python
# Before:
input_image_base64: str | None = None

# After:
input_image_base64: str | list[str] | None = None
```

Accepts a single base64 string (backward compat) or a list of them.

#### b) `Job` dataclass (line 76)

```python
# Before:
input_image_path: str | None

# After:
input_image_paths: list[str]  # empty list = no input images
```

#### c) Base64 decode block (lines 395-405)

```python
# Before:
input_image_path = None
if req.input_image_base64:
    try:
        img_bytes = base64.b64decode(req.input_image_base64)
    except Exception:
        raise HTTPException(...)
    fd, input_image_path = tempfile.mkstemp(...)
    ...

# After:
input_image_paths: list[str] = []
if req.input_image_base64:
    items = (req.input_image_base64
             if isinstance(req.input_image_base64, list)
             else [req.input_image_base64])
    for b64 in items:
        try:
            img_bytes = base64.b64decode(b64)
        except Exception:
            # Clean up any already-decoded temp files
            for p in input_image_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
            raise HTTPException(status_code=400, detail="Invalid base64 in input_image_base64")
        fd, path = tempfile.mkstemp(suffix=".png", prefix="flux_api_input_")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(img_bytes)
        input_image_paths.append(path)
```

#### d) Job creation (line 423)

```python
input_image_paths=input_image_paths,
```

#### e) Worker call (line 222)

```python
# Before:
input_image=job.input_image_path,

# After:
input_image=job.input_image_paths or None,
```

#### f) Cleanup (lines 264-269)

```python
# Before:
if job.input_image_path:
    try:
        os.remove(job.input_image_path)
    except OSError:
        pass

# After:
for p in job.input_image_paths:
    try:
        os.remove(p)
    except OSError:
        pass
```

### 5. `api_generate.py` — CLI API client (optional follow-up)

Currently does not support img2img at all. Could add `-i` flag with
`action="append"`, read each file, base64-encode, and send as
`input_image_base64` list. Low priority — can be a separate commit.

### 6. `run` script — No changes

The `img2img` subcommand forwards all args to `generate_image.py`. Since
`-i` with `action="append"` is fully backward compatible, commands like
`./run img2img "prompt" -i a.jpg -i b.jpg` will just work.

### 7. Documentation

Update `AGENTS.md` and `README.md`:
- Add multi-reference examples to CLI usage sections
- Update feature list to mention multi-reference editing
- Update `serve_api.py` docstring with multi-image API example

---

## Testing Plan

```bash
# Single image (backward compat — must work identically to before)
./run img2img "oil painting" -i photo.jpg -o art.png

# Multi-image
./run img2img "combine these styles" -i photo1.jpg -i photo2.jpg -o combined.png

# Text-to-image (no -i flag — must still work)
./run generate "a cat" -o cat.png

# GUI: test add/remove/clear flow in img2img mode
./run gui

# API: single image (backward compat)
curl -X POST http://localhost:8190/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "input_image_base64": "<b64>"}'

# API: multi-image
curl -X POST http://localhost:8190/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "input_image_base64": ["<b64_1>", "<b64_2>"]}'
```

## VRAM Considerations

With multiple reference images, each one adds to the token sequence length
processed by the transformer at every denoising step. Memory usage grows
roughly linearly with the number of references. The `klein-9b-kv` model
mitigates this by caching reference KV after step 0.

The flux2 submodule automatically caps each reference to 1024x1024 pixels
when `len(img_ctx) > 1` (vs 2024x2024 for single). No additional VRAM
management is needed in the calling code.

Consider documenting a practical limit (e.g., 4-5 reference images on a 12 GB
card) once real-world testing establishes the boundary.

## Open Items

- [ ] `api_generate.py` — add `-i` support (follow-up commit)
- [ ] `MultiImagePreviewPanel.remove_image()` — re-titling logic (minor)
- [ ] Real-world VRAM testing with 2-4 reference images
- [ ] Consider multi-select file dialog vs single-select with "Add more"
