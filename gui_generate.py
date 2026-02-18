#!/usr/bin/env python
"""PySide6 GUI for FLUX.2 Klein 4B image generation"""

import sys
import os
import gc
from pathlib import Path
import tempfile
from datetime import datetime
from typing import Optional

# Add script directory and flux2/src to path (handle both local and installed as tool)
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))  # For gui_components import
flux2_src = script_dir / "flux2" / "src"
sys.path.insert(0, str(flux2_src))

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

# Import UI components (from script directory)
from gui_components import LeftConfigPanel, RightImagePanel, open_image_file_dialog

# Import existing generation functions
from generate_image import generate_image, model_cache
from upscale_image import upscale_image


class GuiState:
    """Container for GUI state"""

    def __init__(self):
        self.input_image_path: Optional[str] = None
        self.generated_image_path: Optional[str] = None
        self.previous_temp_file: Optional[str] = None
        self.current_seed: Optional[int] = None
        self.worker: Optional[QThread] = None

    def reset_images(self):
        """Reset image-related state"""
        self.input_image_path = None
        self.generated_image_path = None
        self.current_seed = None

    def cleanup_temp_file(self, filepath: Optional[str] = None):
        """Safely delete a temp file"""
        target = filepath if filepath else self.previous_temp_file
        if target and os.path.exists(target):
            try:
                os.remove(target)
            except Exception:
                pass  # Silently ignore cleanup errors


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
            actual_seed = generate_image(
                prompt=prompt,
                output_path=temp_path,
                input_image=input_image,
                width=width,
                height=height,
                num_steps=steps,
                guidance=guidance,
                seed=seed,
            )

            # Store the actual seed used (either provided or generated)
            current_seed = actual_seed

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
        self.setWindowTitle("DJ FLUX.2")
        self.setMinimumSize(900, 725)

        # State
        self.state = GuiState()

        # Build UI
        self._create_ui()
        self._connect_signals()

    def _create_ui(self):
        """Build the user interface"""
        central = QWidget()
        self.setCentralWidget(central)

        # Main horizontal splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel (configuration)
        self.left_panel = LeftConfigPanel(self)

        # Right panel (image previews)
        self.right_panel = RightImagePanel(self)

        # Add to splitter
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 0)  # Left panel fixed-ish
        splitter.setStretchFactor(1, 1)  # Right panel expands
        splitter.setSizes([500, 900])

        # Set as central widget
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # Initialize visibility
        self._on_mode_change()

    def _connect_signals(self):
        """Centralized signal connection for clarity"""
        # Mode switching
        self.left_panel.txt2img_radio.toggled.connect(self._on_mode_change)

        # File operations
        self.left_panel.browse_btn.clicked.connect(self._browse_input_image)

        # Generation
        self.left_panel.generate_btn.clicked.connect(self._generate_image)

        # Actions
        self.left_panel.save_btn.clicked.connect(self._save_image)
        self.left_panel.copy_seed_btn.clicked.connect(self._copy_seed)
        self.left_panel.clear_btn.clicked.connect(self._clear)

        # Model management
        self.left_panel.unload_models_btn.clicked.connect(self._unload_models)

    def _on_mode_change(self):
        """Handle mode switching"""
        is_img2img = self.left_panel.is_img2img_mode()

        # Update left panel visibility
        self.left_panel.input_group.setVisible(is_img2img)

        # Update right panel layout
        self.right_panel.set_mode(is_img2img)

        # Clear input state if switching to txt2img
        if not is_img2img:
            self.state.input_image_path = None
            self.left_panel.input_label.setText("No image selected")
            self.right_panel.input_preview.clear()

        # Clear all previews to release memory when switching modes
        self.right_panel.clear_previews()

        # Force garbage collection to free pixmap memory
        gc.collect()

    def _browse_input_image(self):
        """Open file picker for input image with live preview"""
        filepath = open_image_file_dialog(self, "Select Input Image")

        if filepath:
            self.state.input_image_path = filepath
            filename = Path(filepath).name
            self.left_panel.input_label.setText(f"Selected: {filename}")

            # Display preview
            self.right_panel.input_preview.display_image(filepath)

    def _validate_generation_params(self) -> tuple[bool, str]:
        """Validate parameters before generation"""
        if self.left_panel.is_img2img_mode() and not self.state.input_image_path:
            return False, "Please select an input image for img2img mode"

        prompt = self.left_panel.get_prompt()
        if not prompt.strip():
            return False, "Please enter a prompt"

        return True, ""

    def _generate_image(self):
        """Start image generation in worker thread"""
        # Validate
        valid, error_msg = self._validate_generation_params()
        if not valid:
            QMessageBox.critical(self, "Error", error_msg)
            return

        # Clean up previous generation to prevent memory leaks
        if self.state.previous_temp_file:
            self.state.cleanup_temp_file()
            self.state.previous_temp_file = None

        # Clean up previous worker thread
        if self.state.worker is not None:
            # Disconnect signals to prevent stale connections
            self.state.worker.progress.disconnect()
            self.state.worker.finished.disconnect()
            self.state.worker.error.disconnect()
            # Request stop and block until the thread actually finishes.
            # This prevents concurrent GPU access from two generation threads,
            # which can corrupt GPU state or cause OOM errors.
            self.state.worker.quit()
            self.state.worker.wait()
            self.state.worker.deleteLater()
            self.state.worker = None

        # Force garbage collection to free memory from previous generation
        gc.collect()

        # Update UI state
        self.left_panel.generate_btn.setEnabled(False)
        self.left_panel.status_label.setText("Generating...")
        self.left_panel.status_label.setStyleSheet("color: orange; font-weight: bold;")

        # Get parameters from left panel
        params = self.left_panel.get_generation_params()
        params["input_image"] = (
            self.state.input_image_path if self.left_panel.is_img2img_mode() else None
        )

        # Start worker thread
        self.state.worker = GenerationWorker(params)
        self.state.worker.progress.connect(self._on_progress)
        self.state.worker.finished.connect(self._on_generation_complete)
        self.state.worker.error.connect(self._on_generation_error)
        self.state.worker.start()

    def _on_progress(self, message: str, color: str):
        """Update status label"""
        self.left_panel.status_label.setText(message)
        self.left_panel.status_label.setStyleSheet(
            f"color: {color}; font-weight: bold;"
        )

    def _on_generation_complete(self, image_path: str, seed):
        """Handle successful generation"""
        # Store previous temp file for cleanup on next generation
        self.state.previous_temp_file = self.state.generated_image_path

        self.state.generated_image_path = image_path
        self.state.current_seed = seed

        # Display image
        self.right_panel.output_preview.display_image(image_path)

        # Update UI
        self.left_panel.status_label.setText("Generation complete!")
        self.left_panel.status_label.setStyleSheet("color: green; font-weight: bold;")
        self.left_panel.generate_btn.setEnabled(True)
        self.left_panel.save_btn.setEnabled(True)

        # Update model status indicator (models are now loaded and cached)
        if model_cache.is_loaded():
            mem = model_cache.get_memory_estimate()
            self.left_panel.update_model_status(True, mem)

        # Clean up worker and force garbage collection
        if self.state.worker is not None:
            self.state.worker.deleteLater()
            self.state.worker = None
        gc.collect()

    def _on_generation_error(self, error_msg: str):
        """Handle generation error"""
        self.left_panel.status_label.setText("Error")
        self.left_panel.status_label.setStyleSheet("color: red; font-weight: bold;")
        self.left_panel.generate_btn.setEnabled(True)

        # Clean up worker and force garbage collection even on error
        if self.state.worker is not None:
            self.state.worker.deleteLater()
            self.state.worker = None
        gc.collect()

        QMessageBox.critical(
            self, "Generation Error", f"Failed to generate image:\n\n{error_msg}"
        )

    def _save_image(self):
        """Save generated image to user-selected location"""
        if not self.state.generated_image_path:
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
                img = Image.open(self.state.generated_image_path)
                img.save(filepath, quality=95, optimize=True)
                QMessageBox.information(self, "Success", f"Image saved to:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Save Error", f"Failed to save image:\n{str(e)}"
                )

    def _copy_seed(self):
        """Copy current seed to seed input"""
        if self.state.current_seed is not None:
            self.left_panel.set_seed(self.state.current_seed)
        # If no seed available, do nothing (user will see "Random" in the field)

    def _unload_models(self):
        """Unload models to free GPU/RAM memory"""
        if model_cache.is_loaded():
            model_cache.clear()
            self.left_panel.update_model_status(False)
            QMessageBox.information(
                self,
                "Models Unloaded",
                "Models have been unloaded. Memory has been freed.\n\n"
                "They will be reloaded automatically on next generation.",
            )
        else:
            QMessageBox.information(self, "Info", "Models are not currently loaded")

    def _clear(self):
        """Reset form and preview"""
        # Reset left panel
        self.left_panel.reset_to_defaults()

        # Delete temp files BEFORE resetting state — reset_images() clears the
        # path references, so file deletion must happen first or the files leak.
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

        # Unload models to free GPU/RAM memory
        if model_cache.is_loaded():
            model_cache.clear()
            self.left_panel.update_model_status(False)

        # Force garbage collection to free all memory
        gc.collect()


def main():
    """Launch the GUI application"""
    app = QApplication(sys.argv)
    app.setApplicationName("FLUX.2 Klein Generator")

    window = FluxGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
