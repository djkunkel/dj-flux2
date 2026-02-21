"""UI components for FLUX.2 Klein GUI - separate UI layout from business logic"""

import enum
import sys
from pathlib import Path

# flux2/src must be on sys.path before importing FLUX2_MODEL_INFO
_script_dir = Path(__file__).parent.resolve()
_flux2_src = _script_dir / "flux2" / "src"
if str(_flux2_src) not in sys.path:
    sys.path.insert(0, str(_flux2_src))

from flux2.util import FLUX2_MODEL_INFO
from generate_image import SUPPORTED_MODELS, DEFAULT_MODEL

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QRadioButton,
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QGridLayout,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QFileDialog,
)
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QPixmap, QImage
from PIL import Image


IMAGE_FILE_FILTER = "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.webp);;All Files (*)"


def open_image_file_dialog(parent, title: str = "Select Image") -> str:
    """Open a file dialog with a live image preview panel.

    Uses a non-native QFileDialog so we can attach a preview widget to the
    right side via setLayout on the dialog's existing layout. The preview
    updates as the user navigates files.

    Returns:
        Selected file path, or empty string if cancelled.
    """
    dialog = QFileDialog(parent, title)
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setNameFilter(IMAGE_FILE_FILTER)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.resize(900, 550)

    # Build a preview label and insert it to the right of the dialog layout
    preview = QLabel("No preview")
    preview.setAlignment(Qt.AlignCenter)
    preview.setMinimumSize(QSize(220, 220))
    preview.setMaximumSize(QSize(280, 280))
    preview.setStyleSheet("border: 1px solid #ccc; background: #111; color: #888;")

    layout = dialog.layout()
    # QFileDialog uses a QGridLayout; add preview spanning all rows on the right
    if layout is not None:
        row_count = layout.rowCount()
        layout.addWidget(preview, 0, layout.columnCount(), row_count, 1)

    def update_preview(path: str):
        if not path:
            preview.setText("No preview")
            preview.setPixmap(QPixmap())
            return
        px = QPixmap(path)
        if px.isNull():
            preview.setText("Cannot preview")
            preview.setPixmap(QPixmap())
            return
        preview.setText("")
        preview.setPixmap(
            px.scaled(
                preview.maximumSize(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    dialog.currentChanged.connect(update_preview)

    if dialog.exec() == QFileDialog.Accepted:
        files = dialog.selectedFiles()
        return files[0] if files else ""
    return ""


class ImagePreviewPanel(QWidget):
    """Image preview panel with title and placeholder support"""

    def __init__(
        self, title: str, placeholder_text: str, min_size: int = 512, parent=None
    ):
        super().__init__(parent)
        self.placeholder_text = placeholder_text
        self.min_size = min_size
        self.original_pixmap = None  # Store original for scaling
        self._setup_ui(title)

    def _setup_ui(self, title: str):
        """Build the preview panel UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Title - fixed small height
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        title_label.setFixedHeight(20)

        # Preview label - expands to fill space
        self._preview_label = QLabel(self.placeholder_text)
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setMinimumSize(self.min_size, self.min_size)
        self._preview_label.setStyleSheet(
            "border: 1px solid #ccc; background: #000000; color: #888888;"
        )
        self._preview_label.setScaledContents(False)
        self._preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(title_label)
        layout.addWidget(self._preview_label, 1)  # Stretch factor of 1

    def display_image(self, image_path: str):
        """Load and display image in preview (scaled to fit available space)"""
        try:
            # Release old pixmap before loading new one (prevent memory leak)
            if self.original_pixmap is not None:
                self._preview_label.clear()
                self.original_pixmap = None

            # Load image and convert to QPixmap
            img = Image.open(image_path)
            img = img.convert("RGB")
            data = img.tobytes("raw", "RGB")
            qimage = QImage(
                data, img.width, img.height, img.width * 3, QImage.Format_RGB888
            )
            # .copy() forces Qt to take ownership of the pixel data so that
            # `data` can be safely garbage-collected after this line.
            # Without it, QImage holds a raw pointer to `data`'s buffer and
            # accessing the QImage after `data` is collected is undefined behavior.
            self.original_pixmap = QPixmap.fromImage(qimage.copy())

            # Close PIL Image to release memory immediately
            img.close()

            # Scale to current label size
            self._update_scaled_pixmap()
            self._preview_label.setText("")

        except Exception as e:
            self._preview_label.setText(f"Error loading image:\n{str(e)}")

    def _update_scaled_pixmap(self):
        """Scale the pixmap to fit the current label size"""
        if self.original_pixmap is None:
            return

        label_size = self._preview_label.size()
        scaled_pixmap = self.original_pixmap.scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._preview_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """Handle resize to scale image dynamically"""
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def clear(self):
        """Clear preview and show placeholder"""
        self.original_pixmap = None
        self._preview_label.clear()
        self._preview_label.setText(self.placeholder_text)


class StatusState(enum.Enum):
    """States for the generation status indicator"""

    READY = "ready"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


# CSS styles keyed by StatusState
_STATUS_STYLES: dict[StatusState, str] = {
    StatusState.READY: "color: green; font-weight: bold;",
    StatusState.RUNNING: "color: orange; font-weight: bold;",
    StatusState.SUCCESS: "color: green; font-weight: bold;",
    StatusState.ERROR: "color: red; font-weight: bold;",
}


class LeftConfigPanel(QWidget):
    """Left configuration panel with all generation controls.

    Communicates with the parent window exclusively via Qt signals.
    Internal widget attributes are private (_prefixed); callers use the
    public methods below.

    Each generation mode (txt2img / img2img) keeps its own prompt text.
    Switching modes saves the current prompt and restores the other one,
    so the user never loses work when going back and forth.
    """

    # --- Signals emitted by user actions ---
    mode_changed = Signal(bool)  # True = img2img, False = txt2img
    model_changed = Signal(str)  # new model name
    browse_clicked = Signal()
    generate_clicked = Signal()
    save_clicked = Signal()
    copy_seed_clicked = Signal()
    clear_clicked = Signal()
    unload_models_clicked = Signal()

    # Default prompts shown on first launch for each mode
    _DEFAULT_TXT2IMG_PROMPT = "a cute cat sitting on a windowsill"
    _DEFAULT_IMG2IMG_PROMPT = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Per-mode prompt storage — initialised from defaults
        self._txt2img_prompt: str = self._DEFAULT_TXT2IMG_PROMPT
        self._img2img_prompt: str = self._DEFAULT_IMG2IMG_PROMPT
        self._setup_ui()
        self._connect_internal_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Assemble the left panel from section helpers"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(10)

        title = QLabel("DJ's FLUX.2 Klein Generator")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        layout.addWidget(self._build_mode_section())
        layout.addWidget(self._build_input_section())
        layout.addWidget(self._build_prompt_section(), 1)  # stretch
        layout.addWidget(self._build_params_section())
        layout.addWidget(self._build_upscale_section())
        layout.addLayout(self._build_action_section())
        layout.addLayout(self._build_model_section())
        layout.addStretch()

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        self.setMinimumWidth(450)
        self.setMaximumWidth(600)

    def _build_mode_section(self) -> QWidget:
        """Mode selection radio buttons"""
        group = QGroupBox("Generation Mode")
        layout = QHBoxLayout()

        self._mode_button_group = QButtonGroup()
        self._txt2img_radio = QRadioButton("Text-to-Image")
        self._img2img_radio = QRadioButton("Image-to-Image")
        self._txt2img_radio.setChecked(True)
        self._mode_button_group.addButton(self._txt2img_radio, 0)
        self._mode_button_group.addButton(self._img2img_radio, 1)

        layout.addWidget(self._txt2img_radio)
        layout.addWidget(self._img2img_radio)
        layout.addStretch()
        group.setLayout(layout)
        return group

    def _build_input_section(self) -> QWidget:
        """Input image selector (shown only in img2img mode)"""
        self._input_group = QGroupBox("Input Image (for img2img)")
        layout = QHBoxLayout()

        self._browse_btn = QPushButton("Browse...")
        self._input_label = QLabel("No image selected")

        layout.addWidget(self._browse_btn)
        layout.addWidget(self._input_label)
        layout.addStretch()
        self._input_group.setLayout(layout)
        return self._input_group

    def _build_prompt_section(self) -> QWidget:
        """Prompt text area — content is swapped per mode by _save_and_swap_prompt"""
        group = QGroupBox("Prompt")
        layout = QVBoxLayout()

        self._prompt_text = QTextEdit()
        self._prompt_text.setMinimumHeight(80)
        # Start with the txt2img default (the active mode at launch)
        self._prompt_text.setPlainText(self._txt2img_prompt)
        self._prompt_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(self._prompt_text)
        group.setLayout(layout)
        return group

    def _build_params_section(self) -> QWidget:
        """Model selector, width, height, steps, guidance, seed controls"""
        group = QGroupBox("Generation Parameters")
        layout = QGridLayout()

        # Model selector
        layout.addWidget(QLabel("Model:"), 0, 0)
        self._model_combo = QComboBox()
        for name in SUPPORTED_MODELS:
            # Show a short human-readable label alongside the full model key
            info = FLUX2_MODEL_INFO[name]
            distilled_tag = " (distilled)" if info["guidance_distilled"] else " (base)"
            self._model_combo.addItem(name + distilled_tag, userData=name)
        # Default to DEFAULT_MODEL
        default_idx = SUPPORTED_MODELS.index(DEFAULT_MODEL)
        self._model_combo.setCurrentIndex(default_idx)
        layout.addWidget(self._model_combo, 0, 1, 1, 3)

        # Width
        layout.addWidget(QLabel("Width:"), 1, 0)
        self._width_combo = QComboBox()
        self._width_combo.addItems(["256", "512", "768", "1024", "1280", "1536"])
        self._width_combo.setCurrentText("512")
        layout.addWidget(self._width_combo, 1, 1)

        # Height
        layout.addWidget(QLabel("Height:"), 1, 2)
        self._height_combo = QComboBox()
        self._height_combo.addItems(["256", "512", "768", "1024", "1280", "1536"])
        self._height_combo.setCurrentText("512")
        layout.addWidget(self._height_combo, 1, 3)

        # Steps
        layout.addWidget(QLabel("Steps:"), 2, 0)
        self._steps_spin = QSpinBox()
        self._steps_spin.setRange(1, 200)
        self._steps_spin.setValue(4)
        layout.addWidget(self._steps_spin, 2, 1)

        # Guidance
        layout.addWidget(QLabel("Guidance:"), 2, 2)
        self._guidance_spin = QDoubleSpinBox()
        self._guidance_spin.setRange(0.1, 20.0)
        self._guidance_spin.setSingleStep(0.5)
        self._guidance_spin.setValue(1.0)
        layout.addWidget(self._guidance_spin, 2, 3)

        # Seed
        layout.addWidget(QLabel("Seed:"), 3, 0)
        self._seed_combo = QComboBox()
        self._seed_combo.setEditable(True)
        self._seed_combo.addItem("Random")
        self._seed_combo.setCurrentText("Random")
        layout.addWidget(self._seed_combo, 3, 1, 1, 3)

        group.setLayout(layout)

        # Apply initial fixed-param state for the default model
        self._apply_model_param_constraints(DEFAULT_MODEL)

        return group

    def _build_upscale_section(self) -> QWidget:
        """Upscaling options with sub-controls that enable/disable with the checkbox"""
        group = QGroupBox("Upscaling (Optional)")
        layout = QVBoxLayout()

        self._upscale_check = QCheckBox("Enable Upscaling")
        layout.addWidget(self._upscale_check)

        # Scale radio buttons
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale:"))
        self._scale_button_group = QButtonGroup()
        self._scale_2x = QRadioButton("2x")
        self._scale_4x = QRadioButton("4x")
        self._scale_2x.setChecked(True)
        self._scale_button_group.addButton(self._scale_2x, 2)
        self._scale_button_group.addButton(self._scale_4x, 4)
        scale_layout.addWidget(self._scale_2x)
        scale_layout.addWidget(self._scale_4x)
        scale_layout.addStretch()
        layout.addLayout(scale_layout)

        # Method radio buttons
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self._method_button_group = QButtonGroup()
        self._method_lanczos = QRadioButton("Lanczos (fast, CPU)")
        self._method_realesrgan = QRadioButton("Real-ESRGAN (AI, GPU)")
        self._method_lanczos.setChecked(True)
        self._method_button_group.addButton(self._method_lanczos, 0)
        self._method_button_group.addButton(self._method_realesrgan, 1)
        method_layout.addWidget(self._method_lanczos)
        method_layout.addWidget(self._method_realesrgan)
        method_layout.addStretch()
        layout.addLayout(method_layout)

        group.setLayout(layout)

        # Sub-controls start disabled; they enable only when the checkbox is on
        self._upscale_sub_widgets = [
            self._scale_2x,
            self._scale_4x,
            self._method_lanczos,
            self._method_realesrgan,
        ]
        self._set_upscale_sub_controls_enabled(False)

        return group

    def _build_action_section(self) -> QVBoxLayout:
        """Generate button, status label, and save/copy/clear buttons"""
        layout = QVBoxLayout()

        self._generate_btn = QPushButton("Generate Image")
        self._generate_btn.setMinimumHeight(40)

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet(_STATUS_STYLES[StatusState.READY])
        self._status_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self._generate_btn)
        layout.addWidget(self._status_label)

        button_layout = QHBoxLayout()
        self._save_btn = QPushButton("Save Image...")
        self._save_btn.setEnabled(False)
        self._copy_seed_btn = QPushButton("Copy Seed")
        self._clear_btn = QPushButton("Clear")

        button_layout.addWidget(self._save_btn)
        button_layout.addWidget(self._copy_seed_btn)
        button_layout.addWidget(self._clear_btn)
        layout.addLayout(button_layout)

        return layout

    def _build_model_section(self) -> QHBoxLayout:
        """Model load status indicator and unload button"""
        layout = QHBoxLayout()

        self._model_status_label = QLabel("Models: Not loaded")
        self._model_status_label.setStyleSheet("color: gray; font-size: 10px;")

        self._unload_models_btn = QPushButton("Unload Models")
        self._unload_models_btn.setEnabled(False)
        self._unload_models_btn.setToolTip("Free GPU/RAM by unloading models")

        layout.addWidget(self._model_status_label)
        layout.addWidget(self._unload_models_btn)
        return layout

    # ------------------------------------------------------------------
    # Internal signal wiring
    # ------------------------------------------------------------------

    def _connect_internal_signals(self):
        """Wire internal widget events to outward-facing signals and helpers"""
        # Save the current prompt and restore the other mode's prompt, then
        # emit mode_changed so FluxGUI can react.  Only fires on the newly
        # selected radio (checked=True) to avoid double-emission.
        self._txt2img_radio.toggled.connect(
            lambda checked: self._save_and_swap_prompt(to_img2img=False)
            or self.mode_changed.emit(False)
            if checked
            else None
        )
        self._img2img_radio.toggled.connect(
            lambda checked: self._save_and_swap_prompt(to_img2img=True)
            or self.mode_changed.emit(True)
            if checked
            else None
        )
        self._browse_btn.clicked.connect(self.browse_clicked)
        self._generate_btn.clicked.connect(self.generate_clicked)
        self._save_btn.clicked.connect(self.save_clicked)
        self._copy_seed_btn.clicked.connect(self.copy_seed_clicked)
        self._clear_btn.clicked.connect(self.clear_clicked)
        self._unload_models_btn.clicked.connect(self.unload_models_clicked)

        # Model selector — apply constraints then notify FluxGUI
        self._model_combo.currentIndexChanged.connect(self._on_model_combo_changed)

        # Toggle upscale sub-controls with the checkbox
        self._upscale_check.stateChanged.connect(self._on_upscale_toggled)

    def _on_upscale_toggled(self, state: int):
        """Enable or disable scale/method controls based on the upscale checkbox"""
        self._set_upscale_sub_controls_enabled(bool(state))

    def _set_upscale_sub_controls_enabled(self, enabled: bool):
        for widget in self._upscale_sub_widgets:
            widget.setEnabled(enabled)

    def _on_model_combo_changed(self, index: int):
        """Apply param constraints for the newly selected model, then notify FluxGUI."""
        model_name = self._model_combo.itemData(index)
        self._apply_model_param_constraints(model_name)
        self.model_changed.emit(model_name)

    def _apply_model_param_constraints(self, model_name: str):
        """Lock or unlock steps/guidance based on the model's fixed_params.

        Distilled Klein models (klein-4b, klein-9b) have both 'guidance' and
        'num_steps' in fixed_params — changing them has no effect on output.
        Base models leave fixed_params empty, so both controls are editable.

        Always snaps both spinboxes to the model's documented defaults so the
        user sees the correct starting values when they switch.
        """
        info = FLUX2_MODEL_INFO[model_name]
        fixed = info.get("fixed_params", set())
        defaults = info["defaults"]

        # Steps
        steps_fixed = "num_steps" in fixed
        self._steps_spin.setValue(defaults["num_steps"])
        self._steps_spin.setEnabled(not steps_fixed)
        if steps_fixed:
            self._steps_spin.setToolTip(
                f"Fixed at {defaults['num_steps']} for {model_name} (distilled model)."
            )
        else:
            self._steps_spin.setToolTip("")

        # Guidance
        guidance_fixed = "guidance" in fixed
        self._guidance_spin.setValue(defaults["guidance"])
        self._guidance_spin.setEnabled(not guidance_fixed)
        if guidance_fixed:
            self._guidance_spin.setToolTip(
                f"Not used by {model_name} (distilled — guidance is baked in)."
            )
        else:
            self._guidance_spin.setToolTip("")

    def _save_and_swap_prompt(self, to_img2img: bool):
        """Save the current prompt for the leaving mode and restore the arriving mode's prompt.

        Args:
            to_img2img: True when switching to img2img, False when switching to txt2img.
        """
        current_text = self._prompt_text.toPlainText()
        if to_img2img:
            self._txt2img_prompt = current_text
            self._prompt_text.setPlainText(self._img2img_prompt)
        else:
            self._img2img_prompt = current_text
            self._prompt_text.setPlainText(self._txt2img_prompt)

    # ------------------------------------------------------------------
    # Public API used by FluxGUI
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        """Return the currently selected model key (e.g. 'flux.2-klein-4b')"""
        return self._model_combo.currentData()

    def set_model(self, model_name: str):
        """Select a model by name, triggering constraint updates"""
        for i in range(self._model_combo.count()):
            if self._model_combo.itemData(i) == model_name:
                self._model_combo.setCurrentIndex(i)
                return
        raise ValueError(f"Model not found in selector: {model_name!r}")

    def is_img2img_mode(self) -> bool:
        """Return True if Image-to-Image mode is selected"""
        return self._img2img_radio.isChecked()

    def get_prompt(self) -> str:
        """Return current prompt text"""
        return self._prompt_text.toPlainText()

    def get_seed(self) -> int | None:
        """Return seed value, or None for random"""
        seed_text = self._seed_combo.currentText().strip()
        if seed_text.lower() == "random" or seed_text == "":
            return None
        try:
            return int(seed_text)
        except ValueError:
            return None

    def get_generation_params(self) -> dict:
        """Gather all generation parameters from UI controls"""
        return {
            "model_name": self.get_model_name(),
            "prompt": self.get_prompt(),
            "width": int(self._width_combo.currentText()),
            "height": int(self._height_combo.currentText()),
            "steps": self._steps_spin.value(),
            "guidance": self._guidance_spin.value(),
            "seed": self.get_seed(),
            "do_upscale": self._upscale_check.isChecked(),
            "upscale_scale": self._scale_button_group.checkedId(),
            "upscale_method": (
                "realesrgan" if self._method_realesrgan.isChecked() else "lanczos"
            ),
        }

    def set_input_label(self, text: str):
        """Update the input image filename label"""
        self._input_label.setText(text)

    def set_input_group_visible(self, visible: bool):
        """Show or hide the input image group box"""
        self._input_group.setVisible(visible)

    def set_generate_enabled(self, enabled: bool):
        """Enable or disable the Generate button"""
        self._generate_btn.setEnabled(enabled)

    def set_save_enabled(self, enabled: bool):
        """Enable or disable the Save button"""
        self._save_btn.setEnabled(enabled)

    def set_status(self, message: str, state: StatusState):
        """Update the status label text and colour"""
        self._status_label.setText(message)
        self._status_label.setStyleSheet(_STATUS_STYLES[state])

    def set_seed(self, seed: int | str):
        """Set seed value in the combo box"""
        self._seed_combo.setCurrentText(str(seed))

    def update_model_status(self, is_loaded: bool, memory_str: str = ""):
        """Update the model load status indicator and Unload button state

        Args:
            is_loaded: Whether models are currently loaded in memory
            memory_str: Memory usage string (e.g., "~4.2 GB")
        """
        if is_loaded:
            self._model_status_label.setText(f"Models: Loaded ({memory_str})")
            self._model_status_label.setStyleSheet(
                "color: green; font-size: 10px; font-weight: bold;"
            )
            self._unload_models_btn.setEnabled(True)
        else:
            self._model_status_label.setText("Models: Not loaded")
            self._model_status_label.setStyleSheet("color: gray; font-size: 10px;")
            self._unload_models_btn.setEnabled(False)

    def reset_to_defaults(self):
        """Reset all controls to default values"""
        # Reset stored prompts for both modes
        self._txt2img_prompt = self._DEFAULT_TXT2IMG_PROMPT
        self._img2img_prompt = self._DEFAULT_IMG2IMG_PROMPT
        # Restore the prompt for whichever mode is currently active
        if self._img2img_radio.isChecked():
            self._prompt_text.setPlainText(self._img2img_prompt)
        else:
            self._prompt_text.setPlainText(self._txt2img_prompt)
        # Reset model to default (triggers constraint update via signal)
        self.set_model(DEFAULT_MODEL)
        self._width_combo.setCurrentText("512")
        self._height_combo.setCurrentText("512")
        self._seed_combo.setCurrentText("Random")
        self._upscale_check.setChecked(False)
        self._scale_2x.setChecked(True)
        self._method_lanczos.setChecked(True)
        self._input_label.setText("No image selected")
        self.set_status("Ready", StatusState.READY)
        self._save_btn.setEnabled(False)


class RightImagePanel(QWidget):
    """Right panel for image previews — switches between txt2img and img2img layouts.

    Both ImagePreviewPanel widgets are created once and kept alive for the
    lifetime of this panel.  Mode switching only shows/hides the input panel
    and adjusts the splitter sizes — no widget creation or destruction occurs
    after initialisation.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_mode = None
        self._setup_ui()

    def _setup_ui(self):
        """Create the persistent layout: a QSplitter holding both panels"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        self.input_preview = ImagePreviewPanel(
            "Input Image", "(img2img mode only)", min_size=256
        )
        self.output_preview = ImagePreviewPanel(
            "Generated Image", "Generate an image to see preview", min_size=256
        )

        # Single persistent splitter — never recreated on mode switch
        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.addWidget(self.input_preview)
        self._splitter.addWidget(self.output_preview)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self._splitter)

        # Initialise to txt2img (input panel hidden)
        self.set_mode(False)

    def set_mode(self, is_img2img: bool):
        """Switch between txt2img (single panel) and img2img (side-by-side) layouts.

        No widgets are created or destroyed here — only visibility and splitter
        sizes change.
        """
        if self._current_mode == is_img2img:
            return

        self._current_mode = is_img2img

        if is_img2img:
            self.input_preview.setVisible(True)
            # Equal split
            total = self._splitter.width()
            half = total // 2 if total > 0 else 400
            self._splitter.setSizes([half, half])
        else:
            self.input_preview.setVisible(False)
            # Give all space to output panel
            self._splitter.setSizes([0, self._splitter.width() or 800])

    def clear_previews(self):
        """Clear both preview panels"""
        self.input_preview.clear()
        self.output_preview.clear()
