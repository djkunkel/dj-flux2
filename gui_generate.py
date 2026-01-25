#!/usr/bin/env python
"""PySide6 GUI for FLUX.2 Klein 4B image generation"""

import sys
import os
from pathlib import Path
import tempfile
from datetime import datetime

# Add flux2/src to path (handle both local and installed as tool)
script_dir = Path(__file__).parent.resolve()
flux2_src = script_dir / "flux2" / "src"
sys.path.insert(0, str(flux2_src))

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
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
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QGridLayout,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QPixmap, QImage
from PIL import Image

# Import existing generation functions
from generate_image import generate_image
from upscale_image import upscale_image


class GenerationWorker(QThread):
    """Worker thread for image generation"""

    # Signals for thread-safe communication
    progress = Signal(str, str)  # message, color
    finished = Signal(str, object)  # image_path, seed
    error = Signal(str)  # error_message

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        """Run generation in background thread"""
        try:
            # Extract parameters
            prompt = self.params["prompt"]
            width = self.params["width"]
            height = self.params["height"]
            steps = self.params["steps"]
            guidance = self.params["guidance"]
            seed = self.params["seed"]
            input_image = self.params["input_image"]
            do_upscale = self.params["do_upscale"]
            upscale_scale = self.params["upscale_scale"]
            upscale_method = self.params["upscale_method"]

            # Create temp output path
            temp_fd, temp_path = tempfile.mkstemp(suffix=".png", prefix="flux_gui_")
            os.close(temp_fd)

            # Generate image
            self.progress.emit("Generating image...", "orange")
            generate_image(
                prompt=prompt,
                output_path=temp_path,
                input_image=input_image,
                width=width,
                height=height,
                num_steps=steps,
                guidance=guidance,
                seed=seed,
            )

            # Store the seed
            current_seed = seed if seed is not None else "Random (see EXIF)"

            # Optionally upscale
            final_path = temp_path
            if do_upscale:
                self.progress.emit("Upscaling...", "orange")

                upscaled_fd, upscaled_path = tempfile.mkstemp(
                    suffix=".png", prefix="flux_gui_upscaled_"
                )
                os.close(upscaled_fd)

                upscale_image(
                    input_path=temp_path,
                    output_path=upscaled_path,
                    scale=upscale_scale,
                    method=upscale_method,
                )

                # Clean up non-upscaled temp file
                os.remove(temp_path)
                final_path = upscaled_path

            # Emit success
            self.finished.emit(final_path, current_seed)

        except Exception as e:
            self.error.emit(str(e))


class FluxGUI(QMainWindow):
    """Main GUI window for FLUX.2 Klein image generation"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FLUX.2 Klein Image Generator")
        self.setMinimumSize(1000, 900)

        # State
        self.input_image_path = None
        self.generated_image_path = None
        self.current_seed = None
        self.worker = None

        # Build UI
        self._create_ui()

    def _create_ui(self):
        """Build the user interface"""

        # Central widget and main layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)

        # Title
        title = QLabel("FLUX.2 Klein Image Generator")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Mode selection
        mode_group = QGroupBox("Generation Mode")
        mode_layout = QHBoxLayout()
        self.mode_group = QButtonGroup()

        self.txt2img_radio = QRadioButton("Text-to-Image")
        self.img2img_radio = QRadioButton("Image-to-Image")
        self.txt2img_radio.setChecked(True)

        self.mode_group.addButton(self.txt2img_radio, 0)
        self.mode_group.addButton(self.img2img_radio, 1)

        mode_layout.addWidget(self.txt2img_radio)
        mode_layout.addWidget(self.img2img_radio)
        mode_layout.addStretch()
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Connect mode change
        self.txt2img_radio.toggled.connect(self._on_mode_change)

        # Input image section
        self.input_group = QGroupBox("Input Image (for img2img)")
        input_layout = QHBoxLayout()

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_input_image)

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
        self.prompt_text.setMaximumHeight(80)
        self.prompt_text.setPlainText("a cute cat sitting on a windowsill")

        prompt_layout.addWidget(self.prompt_text)
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)

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

        self.scale_group = QButtonGroup()
        self.scale_2x = QRadioButton("2x")
        self.scale_4x = QRadioButton("4x")
        self.scale_2x.setChecked(True)
        self.scale_group.addButton(self.scale_2x, 2)
        self.scale_group.addButton(self.scale_4x, 4)

        scale_layout.addWidget(self.scale_2x)
        scale_layout.addWidget(self.scale_4x)
        scale_layout.addStretch()
        upscale_layout.addLayout(scale_layout)

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))

        self.method_group = QButtonGroup()
        self.method_lanczos = QRadioButton("Lanczos (fast, CPU)")
        self.method_realesrgan = QRadioButton("Real-ESRGAN (AI, GPU)")
        self.method_lanczos.setChecked(True)
        self.method_group.addButton(self.method_lanczos, 0)
        self.method_group.addButton(self.method_realesrgan, 1)

        method_layout.addWidget(self.method_lanczos)
        method_layout.addWidget(self.method_realesrgan)
        method_layout.addStretch()
        upscale_layout.addLayout(method_layout)

        upscale_group.setLayout(upscale_layout)
        layout.addWidget(upscale_group)

        # Generate button and status
        control_layout = QHBoxLayout()

        self.generate_btn = QPushButton("Generate Image")
        self.generate_btn.clicked.connect(self._generate_image)
        self.generate_btn.setMinimumHeight(40)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")

        control_layout.addWidget(self.generate_btn)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()
        layout.addLayout(control_layout)

        # Preview area
        preview_group = QGroupBox("Preview")
        preview_layout = QHBoxLayout()

        # Input preview
        input_preview_layout = QVBoxLayout()
        input_title = QLabel("Input Image")
        input_title.setAlignment(Qt.AlignCenter)
        input_title.setStyleSheet("font-weight: bold;")

        self.input_preview = QLabel("(img2img mode only)")
        self.input_preview.setAlignment(Qt.AlignCenter)
        self.input_preview.setMinimumSize(400, 400)
        self.input_preview.setStyleSheet("border: 1px solid #ccc; background: #f0f0f0;")
        self.input_preview.setScaledContents(False)

        input_preview_layout.addWidget(input_title)
        input_preview_layout.addWidget(self.input_preview)
        preview_layout.addLayout(input_preview_layout)

        # Output preview
        output_preview_layout = QVBoxLayout()
        output_title = QLabel("Generated Image")
        output_title.setAlignment(Qt.AlignCenter)
        output_title.setStyleSheet("font-weight: bold;")

        self.output_preview = QLabel("Generate an image to see preview")
        self.output_preview.setAlignment(Qt.AlignCenter)
        self.output_preview.setMinimumSize(400, 400)
        self.output_preview.setStyleSheet(
            "border: 1px solid #ccc; background: #f0f0f0;"
        )
        self.output_preview.setScaledContents(False)

        output_preview_layout.addWidget(output_title)
        output_preview_layout.addWidget(self.output_preview)
        preview_layout.addLayout(output_preview_layout)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # Bottom buttons
        button_layout = QHBoxLayout()

        self.save_btn = QPushButton("Save Image...")
        self.save_btn.clicked.connect(self._save_image)
        self.save_btn.setEnabled(False)

        self.copy_seed_btn = QPushButton("Copy Seed")
        self.copy_seed_btn.clicked.connect(self._copy_seed)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear)

        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.copy_seed_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Initialize visibility
        self._on_mode_change()

    def _on_mode_change(self):
        """Handle mode switching"""
        is_img2img = self.img2img_radio.isChecked()
        self.input_group.setVisible(is_img2img)

        if not is_img2img:
            self.input_image_path = None
            self.input_label.setText("No image selected")
            self.input_preview.clear()
            self.input_preview.setText("(img2img mode only)")

    def _browse_input_image(self):
        """Open file picker for input image"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.webp);;All Files (*)",
        )

        if filepath:
            self.input_image_path = filepath
            filename = Path(filepath).name
            self.input_label.setText(f"Selected: {filename}")

            # Display preview
            self._display_image(filepath, self.input_preview)

    def _get_seed(self):
        """Get seed value (None for random)"""
        seed_text = self.seed_combo.currentText().strip()
        if seed_text.lower() == "random" or seed_text == "":
            return None
        try:
            return int(seed_text)
        except ValueError:
            return None

    def _generate_image(self):
        """Start image generation in worker thread"""
        # Validate inputs
        if self.img2img_radio.isChecked() and not self.input_image_path:
            QMessageBox.critical(
                self, "Error", "Please select an input image for img2img mode"
            )
            return

        prompt = self.prompt_text.toPlainText().strip()
        if not prompt:
            QMessageBox.critical(self, "Error", "Please enter a prompt")
            return

        # Disable generate button
        self.generate_btn.setEnabled(False)
        self.status_label.setText("Generating...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")

        # Get parameters
        params = {
            "prompt": prompt,
            "width": int(self.width_combo.currentText()),
            "height": int(self.height_combo.currentText()),
            "steps": self.steps_spin.value(),
            "guidance": self.guidance_spin.value(),
            "seed": self._get_seed(),
            "input_image": self.input_image_path
            if self.img2img_radio.isChecked()
            else None,
            "do_upscale": self.upscale_check.isChecked(),
            "upscale_scale": self.scale_group.checkedId(),
            "upscale_method": "realesrgan"
            if self.method_realesrgan.isChecked()
            else "lanczos",
        }

        # Start worker thread
        self.worker = GenerationWorker(params)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_generation_complete)
        self.worker.error.connect(self._on_generation_error)
        self.worker.start()

    def _on_progress(self, message, color):
        """Update status label"""
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _on_generation_complete(self, image_path, seed):
        """Handle successful generation"""
        self.generated_image_path = image_path
        self.current_seed = seed

        # Display image
        self._display_image(image_path, self.output_preview)

        # Update UI
        self.status_label.setText("Generation complete!")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        self.generate_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

    def _on_generation_error(self, error_msg):
        """Handle generation error"""
        self.status_label.setText("Error")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        self.generate_btn.setEnabled(True)

        QMessageBox.critical(
            self, "Generation Error", f"Failed to generate image:\n\n{error_msg}"
        )

    def _display_image(self, image_path, label):
        """Load and display image in label (scaled to fit)"""
        try:
            # Load with PIL and scale
            img = Image.open(image_path)
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)

            # Convert to QPixmap
            img = img.convert("RGB")
            data = img.tobytes("raw", "RGB")
            qimage = QImage(
                data, img.width, img.height, img.width * 3, QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(qimage)

            # Display
            label.setPixmap(pixmap)
            label.setText("")

        except Exception as e:
            label.setText(f"Error loading image:\n{str(e)}")

    def _save_image(self):
        """Save generated image to user-selected location"""
        if not self.generated_image_path:
            QMessageBox.warning(self, "No Image", "No generated image to save")
            return

        # Default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"flux_generated_{timestamp}.png"

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Generated Image",
            default_name,
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)",
        )

        if filepath:
            try:
                img = Image.open(self.generated_image_path)
                img.save(filepath, quality=95, optimize=True)
                QMessageBox.information(self, "Success", f"Image saved to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Save Error", f"Failed to save image:\n{str(e)}"
                )

    def _copy_seed(self):
        """Copy current seed to seed input"""
        if self.current_seed is not None:
            self.seed_combo.setCurrentText(str(self.current_seed))
            QMessageBox.information(
                self, "Seed Copied", f"Seed copied to input: {self.current_seed}"
            )
        else:
            QMessageBox.information(
                self, "No Seed", "Generate an image first to copy its seed"
            )

    def _clear(self):
        """Reset form and preview"""
        # Reset prompt
        self.prompt_text.setPlainText("a cute cat sitting on a windowsill")

        # Reset parameters
        self.width_combo.setCurrentText("512")
        self.height_combo.setCurrentText("512")
        self.steps_spin.setValue(4)
        self.guidance_spin.setValue(1.0)
        self.seed_combo.setCurrentText("Random")
        self.upscale_check.setChecked(False)
        self.scale_2x.setChecked(True)
        self.method_lanczos.setChecked(True)

        # Clear images
        self.input_image_path = None
        self.generated_image_path = None
        self.current_seed = None

        self.input_label.setText("No image selected")
        self.input_preview.clear()
        self.input_preview.setText("(img2img mode only)")
        self.output_preview.clear()
        self.output_preview.setText("Generate an image to see preview")

        # Reset UI state
        self.save_btn.setEnabled(False)
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")

        # Clean up temp files
        if self.generated_image_path and os.path.exists(self.generated_image_path):
            try:
                os.remove(self.generated_image_path)
            except:
                pass


def main():
    """Launch the GUI application"""
    app = QApplication(sys.argv)
    app.setApplicationName("FLUX.2 Klein Generator")

    window = FluxGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
