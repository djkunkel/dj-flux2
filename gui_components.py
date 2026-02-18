"""UI components for FLUX.2 Klein GUI - separate UI layout from business logic"""

from pathlib import Path
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
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PIL import Image


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
        title_label.setFixedHeight(20)  # Small fixed height

        # Preview label - expands to fill space
        self.preview_label = QLabel(self.placeholder_text)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(self.min_size, self.min_size)
        self.preview_label.setStyleSheet(
            "border: 1px solid #ccc; background: #000000; color: #888888;"
        )
        self.preview_label.setScaledContents(False)

        # Make preview expand to fill available space
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(title_label)
        layout.addWidget(self.preview_label, 1)  # Stretch factor of 1

    def display_image(self, image_path: str):
        """Load and display image in preview (scaled to fit available space)"""
        try:
            # Release old pixmap before loading new one (prevent memory leak)
            if self.original_pixmap is not None:
                self.preview_label.clear()
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
            self.preview_label.setText("")

        except Exception as e:
            self.preview_label.setText(f"Error loading image:\n{str(e)}")

    def _update_scaled_pixmap(self):
        """Scale the pixmap to fit the current label size"""
        if self.original_pixmap is None:
            return

        # Get available size (considering minimum size)
        label_size = self.preview_label.size()

        # Scale pixmap to fit while maintaining aspect ratio
        scaled_pixmap = self.original_pixmap.scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        self.preview_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """Handle resize to scale image dynamically"""
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def clear(self):
        """Clear preview and show placeholder"""
        self.original_pixmap = None
        self.preview_label.clear()
        self.preview_label.setText(self.placeholder_text)

    def set_placeholder(self, text: str):
        """Update placeholder text"""
        self.placeholder_text = text
        if not self.preview_label.pixmap():
            self.preview_label.setText(text)


class LeftConfigPanel(QWidget):
    """Left configuration panel with all generation controls"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Build the left panel UI"""
        # Main layout with scroll support
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Scrollable area for controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(10)

        # Title
        title = QLabel("DJ's FLUX.2 Klein Generator")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Mode selection
        self.mode_group_box = QGroupBox("Generation Mode")
        mode_layout = QHBoxLayout()
        self.mode_button_group = QButtonGroup()

        self.txt2img_radio = QRadioButton("Text-to-Image")
        self.img2img_radio = QRadioButton("Image-to-Image")
        self.txt2img_radio.setChecked(True)

        self.mode_button_group.addButton(self.txt2img_radio, 0)
        self.mode_button_group.addButton(self.img2img_radio, 1)

        mode_layout.addWidget(self.txt2img_radio)
        mode_layout.addWidget(self.img2img_radio)
        mode_layout.addStretch()
        self.mode_group_box.setLayout(mode_layout)
        layout.addWidget(self.mode_group_box)

        # Input image section
        self.input_group = QGroupBox("Input Image (for img2img)")
        input_layout = QHBoxLayout()

        self.browse_btn = QPushButton("Browse...")
        self.input_label = QLabel("No image selected")

        input_layout.addWidget(self.browse_btn)
        input_layout.addWidget(self.input_label)
        input_layout.addStretch()
        self.input_group.setLayout(input_layout)
        layout.addWidget(self.input_group)

        # Prompt
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout()

        self.prompt_text = QTextEdit()
        self.prompt_text.setMinimumHeight(80)  # Minimum height for usability
        # No maximum height - will expand to fill available space
        self.prompt_text.setPlainText("a cute cat sitting on a windowsill")
        # Set size policy to allow vertical expansion
        self.prompt_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        prompt_layout.addWidget(self.prompt_text)
        prompt_group.setLayout(prompt_layout)
        # Add with stretch factor to allow resizing
        layout.addWidget(prompt_group, 1)

        # Parameters
        params_group = QGroupBox("Generation Parameters")
        params_layout = QGridLayout()

        # Width
        params_layout.addWidget(QLabel("Width:"), 0, 0)
        self.width_combo = QComboBox()
        self.width_combo.addItems(["256", "512", "768", "1024", "1280", "1536"])
        self.width_combo.setCurrentText("512")
        params_layout.addWidget(self.width_combo, 0, 1)

        # Height
        params_layout.addWidget(QLabel("Height:"), 0, 2)
        self.height_combo = QComboBox()
        self.height_combo.addItems(["256", "512", "768", "1024", "1280", "1536"])
        self.height_combo.setCurrentText("512")
        params_layout.addWidget(self.height_combo, 0, 3)

        # Steps
        params_layout.addWidget(QLabel("Steps:"), 1, 0)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 50)
        self.steps_spin.setValue(4)
        params_layout.addWidget(self.steps_spin, 1, 1)

        # Guidance
        params_layout.addWidget(QLabel("Guidance:"), 1, 2)
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(0.1, 5.0)
        self.guidance_spin.setSingleStep(0.1)
        self.guidance_spin.setValue(1.0)
        params_layout.addWidget(self.guidance_spin, 1, 3)

        # Seed
        params_layout.addWidget(QLabel("Seed:"), 2, 0)
        self.seed_combo = QComboBox()
        self.seed_combo.setEditable(True)
        self.seed_combo.addItem("Random")
        self.seed_combo.setCurrentText("Random")
        params_layout.addWidget(self.seed_combo, 2, 1, 1, 3)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Upscaling options
        upscale_group = QGroupBox("Upscaling (Optional)")
        upscale_layout = QVBoxLayout()

        self.upscale_check = QCheckBox("Enable Upscaling")
        upscale_layout.addWidget(self.upscale_check)

        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale:"))

        self.scale_button_group = QButtonGroup()
        self.scale_2x = QRadioButton("2x")
        self.scale_4x = QRadioButton("4x")
        self.scale_2x.setChecked(True)
        self.scale_button_group.addButton(self.scale_2x, 2)
        self.scale_button_group.addButton(self.scale_4x, 4)

        scale_layout.addWidget(self.scale_2x)
        scale_layout.addWidget(self.scale_4x)
        scale_layout.addStretch()
        upscale_layout.addLayout(scale_layout)

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))

        self.method_button_group = QButtonGroup()
        self.method_lanczos = QRadioButton("Lanczos (fast, CPU)")
        self.method_realesrgan = QRadioButton("Real-ESRGAN (AI, GPU)")
        self.method_lanczos.setChecked(True)
        self.method_button_group.addButton(self.method_lanczos, 0)
        self.method_button_group.addButton(self.method_realesrgan, 1)

        method_layout.addWidget(self.method_lanczos)
        method_layout.addWidget(self.method_realesrgan)
        method_layout.addStretch()
        upscale_layout.addLayout(method_layout)

        upscale_group.setLayout(upscale_layout)
        layout.addWidget(upscale_group)

        # Generate button and status
        control_layout = QVBoxLayout()

        self.generate_btn = QPushButton("Generate Image")
        self.generate_btn.setMinimumHeight(40)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignCenter)

        control_layout.addWidget(self.generate_btn)
        control_layout.addWidget(self.status_label)
        layout.addLayout(control_layout)

        # Action buttons
        button_layout = QHBoxLayout()

        self.save_btn = QPushButton("Save Image...")
        self.save_btn.setEnabled(False)

        self.copy_seed_btn = QPushButton("Copy Seed")

        self.clear_btn = QPushButton("Clear")

        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.copy_seed_btn)
        button_layout.addWidget(self.clear_btn)
        layout.addLayout(button_layout)

        # Model status and control
        model_control_layout = QHBoxLayout()

        self.model_status_label = QLabel("Models: Not loaded")
        self.model_status_label.setStyleSheet("color: gray; font-size: 10px;")

        self.unload_models_btn = QPushButton("Unload Models")
        self.unload_models_btn.setEnabled(False)
        self.unload_models_btn.setToolTip("Free GPU/RAM by unloading models")

        model_control_layout.addWidget(self.model_status_label)
        model_control_layout.addWidget(self.unload_models_btn)
        layout.addLayout(model_control_layout)

        layout.addStretch()

        # Set scroll content
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # Set preferred width
        self.setMinimumWidth(450)
        self.setMaximumWidth(600)

    def is_img2img_mode(self) -> bool:
        """Check if in image-to-image mode"""
        return self.img2img_radio.isChecked()

    def get_prompt(self) -> str:
        """Get current prompt text"""
        return self.prompt_text.toPlainText()

    def get_seed(self) -> int | None:
        """Get seed value (None for random)"""
        seed_text = self.seed_combo.currentText().strip()
        if seed_text.lower() == "random" or seed_text == "":
            return None
        try:
            return int(seed_text)
        except ValueError:
            return None

    def get_generation_params(self) -> dict:
        """Gather all generation parameters from UI controls"""
        return {
            "prompt": self.get_prompt(),
            "width": int(self.width_combo.currentText()),
            "height": int(self.height_combo.currentText()),
            "steps": self.steps_spin.value(),
            "guidance": self.guidance_spin.value(),
            "seed": self.get_seed(),
            "do_upscale": self.upscale_check.isChecked(),
            "upscale_scale": self.scale_button_group.checkedId(),
            "upscale_method": "realesrgan"
            if self.method_realesrgan.isChecked()
            else "lanczos",
        }

    def reset_to_defaults(self):
        """Reset all controls to default values"""
        self.prompt_text.setPlainText("a cute cat sitting on a windowsill")
        self.width_combo.setCurrentText("512")
        self.height_combo.setCurrentText("512")
        self.steps_spin.setValue(4)
        self.guidance_spin.setValue(1.0)
        self.seed_combo.setCurrentText("Random")
        self.upscale_check.setChecked(False)
        self.scale_2x.setChecked(True)
        self.method_lanczos.setChecked(True)
        self.input_label.setText("No image selected")
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        self.save_btn.setEnabled(False)

    def set_seed(self, seed: int | str):
        """Set seed value in combo box"""
        self.seed_combo.setCurrentText(str(seed))

    def update_model_status(self, is_loaded: bool, memory_str: str = ""):
        """Update model status indicator

        Args:
            is_loaded: Whether models are currently loaded in memory
            memory_str: Memory usage string (e.g., "~4.2 GB")
        """
        if is_loaded:
            self.model_status_label.setText(f"Models: Loaded ({memory_str})")
            self.model_status_label.setStyleSheet(
                "color: green; font-size: 10px; font-weight: bold;"
            )
            self.unload_models_btn.setEnabled(True)
        else:
            self.model_status_label.setText("Models: Not loaded")
            self.model_status_label.setStyleSheet("color: gray; font-size: 10px;")
            self.unload_models_btn.setEnabled(False)


class RightImagePanel(QWidget):
    """Right panel for image previews - switches layout based on mode"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_mode = None
        self._setup_ui()

    def _setup_ui(self):
        """Build the right panel UI"""
        # Main layout that will be switched
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        # Create preview panels
        self.input_preview = ImagePreviewPanel(
            "Input Image", "(img2img mode only)", min_size=512
        )
        self.output_preview = ImagePreviewPanel(
            "Generated Image", "Generate an image to see preview", min_size=512
        )

        # Start in text-to-image mode (single preview)
        self.set_mode(False)

    def set_mode(self, is_img2img: bool):
        """Switch between text-to-image and image-to-image layouts"""
        if self._current_mode == is_img2img:
            return  # Already in correct mode

        self._current_mode = is_img2img

        # Clear current layout
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        if is_img2img:
            # Side-by-side layout for img2img
            h_layout = QHBoxLayout()
            h_layout.addWidget(self.input_preview)
            h_layout.addWidget(self.output_preview)
            self.main_layout.addLayout(h_layout)

            # Show input preview
            self.input_preview.setVisible(True)
        else:
            # Single preview for txt2img - fill available space
            self.main_layout.addWidget(self.output_preview)

            # Hide input preview
            self.input_preview.setVisible(False)

    def clear_previews(self):
        """Clear both preview panels"""
        self.input_preview.clear()
        self.output_preview.clear()
