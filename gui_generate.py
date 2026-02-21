#!/usr/bin/env python
"""PySide6 GUI for FLUX.2 Klein 4B image generation"""

import sys
import os
import gc
from pathlib import Path
import tempfile
from datetime import datetime
from typing import Optional, TypedDict

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
from gui_components import StatusState

# Import existing generation functions
from generate_image import generate_image, model_cache
from upscale_image import upscale_image


class GenerationParams(TypedDict):
    """All parameters passed from the UI to the GenerationWorker thread"""

    prompt: str
    width: int
    height: int
    steps: int
    guidance: float
    seed: Optional[int]
    input_image: Optional[str]
    do_upscale: bool
    upscale_scale: int
    upscale_method: str


class GuiState:
    """Container for mutable GUI state"""

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

    def __init__(self, params: GenerationParams):
        super().__init__()
        self.params = params

    def run(self):
        """Run generation in background thread"""
        try:
            # Create temp output path
            temp_fd, temp_path = tempfile.mkstemp(suffix=".png", prefix="flux_gui_")
            os.close(temp_fd)

            # Generate image
            self.progress.emit("Generating image...", "orange")
            actual_seed = generate_image(
                prompt=self.params["prompt"],
                output_path=temp_path,
                input_image=self.params["input_image"],
                width=self.params["width"],
                height=self.params["height"],
                num_steps=self.params["steps"],
                guidance=self.params["guidance"],
                seed=self.params["seed"],
            )

            # Optionally upscale
            final_path = temp_path
            if self.params["do_upscale"]:
                self.progress.emit("Upscaling...", "orange")

                upscaled_fd, upscaled_path = tempfile.mkstemp(
                    suffix=".png", prefix="flux_gui_upscaled_"
                )
                os.close(upscaled_fd)

                upscale_image(
                    input_path=temp_path,
                    output_path=upscaled_path,
                    scale=self.params["upscale_scale"],
                    method=self.params["upscale_method"],
                )

                # Clean up non-upscaled temp file
                os.remove(temp_path)
                final_path = upscaled_path

            self.finished.emit(final_path, actual_seed)

        except Exception as e:
            self.error.emit(str(e))


class FluxGUI(QMainWindow):
    """Main GUI window for FLUX.2 Klein image generation"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DJ FLUX.2")
        self.setMinimumSize(900, 725)

        self.state = GuiState()

        self._create_ui()
        self._connect_signals()

    def _create_ui(self):
        """Build the user interface"""
        central = QWidget()
        self.setCentralWidget(central)

        splitter = QSplitter(Qt.Horizontal)

        self.left_panel = LeftConfigPanel(self)
        self.right_panel = RightImagePanel(self)

        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 0)  # Left panel fixed-ish
        splitter.setStretchFactor(1, 1)  # Right panel expands
        splitter.setSizes([500, 900])

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # Initialise panel visibility to match the default mode (txt2img)
        self._on_mode_change(False)

    def _connect_signals(self):
        """Connect LeftConfigPanel signals to FluxGUI handlers"""
        self.left_panel.mode_changed.connect(self._on_mode_change)
        self.left_panel.browse_clicked.connect(self._browse_input_image)
        self.left_panel.generate_clicked.connect(self._generate_image)
        self.left_panel.save_clicked.connect(self._save_image)
        self.left_panel.copy_seed_clicked.connect(self._copy_seed)
        self.left_panel.clear_clicked.connect(self._clear)
        self.left_panel.unload_models_clicked.connect(self._unload_models)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_mode_change(self, is_img2img: bool):
        """Handle mode switching between txt2img and img2img.

        When switching to img2img: if there is already a generated image,
        load it into the input panel automatically so the user can immediately
        start refining without having to browse for the file manually.

        When switching back to txt2img: clear the input state but preserve
        the output panel so the last result stays visible.
        """
        self.left_panel.set_input_group_visible(is_img2img)
        self.right_panel.set_mode(is_img2img)

        if is_img2img:
            # Auto-populate the input from the last generated image if available
            # and no input image has been explicitly chosen yet for this session.
            if self.state.generated_image_path and not self.state.input_image_path:
                self._load_input_image(self.state.generated_image_path)
            # Restore the output preview (mode switch hid nothing on the output
            # side, but if there is a generated image make sure it is displayed)
            if self.state.generated_image_path:
                self.right_panel.output_preview.display_image(
                    self.state.generated_image_path
                )
        else:
            # Switching back to txt2img: drop the input image reference
            self.state.input_image_path = None
            self.left_panel.set_input_label("No image selected")
            self.right_panel.input_preview.clear()
            # Keep the output preview intact so the user can still see the last result

    def _load_input_image(self, filepath: str):
        """Set filepath as the active input image and update the UI.

        Used both by the browse button and the auto-load-on-mode-switch path.
        """
        self.state.input_image_path = filepath
        self.left_panel.set_input_label(f"Selected: {Path(filepath).name}")
        self.right_panel.input_preview.display_image(filepath)

    def _browse_input_image(self):
        """Open file picker for input image with live preview"""
        filepath = open_image_file_dialog(self, "Select Input Image")
        if filepath:
            self._load_input_image(filepath)

    def _validate_generation_params(self) -> tuple[bool, str]:
        """Validate parameters before generation"""
        if self.left_panel.is_img2img_mode() and not self.state.input_image_path:
            return False, "Please select an input image for img2img mode"
        if not self.left_panel.get_prompt().strip():
            return False, "Please enter a prompt"
        return True, ""

    def _generate_image(self):
        """Start image generation in worker thread"""
        valid, error_msg = self._validate_generation_params()
        if not valid:
            QMessageBox.critical(self, "Error", error_msg)
            return

        # Clean up previous generation temp file
        if self.state.previous_temp_file:
            self.state.cleanup_temp_file()
            self.state.previous_temp_file = None

        # Stop any still-running worker to prevent concurrent GPU access
        if self.state.worker is not None:
            self.state.worker.progress.disconnect()
            self.state.worker.finished.disconnect()
            self.state.worker.error.disconnect()
            self.state.worker.quit()
            self.state.worker.wait()
            self.state.worker.deleteLater()
            self.state.worker = None

        gc.collect()

        self.left_panel.set_generate_enabled(False)
        self.left_panel.set_status("Generating...", StatusState.RUNNING)

        params: GenerationParams = self.left_panel.get_generation_params()
        params["input_image"] = (
            self.state.input_image_path if self.left_panel.is_img2img_mode() else None
        )

        self.state.worker = GenerationWorker(params)
        self.state.worker.progress.connect(self._on_progress)
        self.state.worker.finished.connect(self._on_generation_complete)
        self.state.worker.error.connect(self._on_generation_error)
        self.state.worker.start()

    def _on_progress(self, message: str, color: str):
        """Forward worker progress to the status label"""
        # Map the colour string to the nearest StatusState
        state = StatusState.RUNNING if color == "orange" else StatusState.SUCCESS
        self.left_panel.set_status(message, state)

    def _on_generation_complete(self, image_path: str, seed):
        """Handle successful generation"""
        self.state.previous_temp_file = self.state.generated_image_path
        self.state.generated_image_path = image_path
        self.state.current_seed = seed

        self.right_panel.output_preview.display_image(image_path)

        self.left_panel.set_status("Generation complete!", StatusState.SUCCESS)
        self.left_panel.set_generate_enabled(True)
        self.left_panel.set_save_enabled(True)

        if model_cache.is_loaded():
            self.left_panel.update_model_status(True, model_cache.get_memory_estimate())

        if self.state.worker is not None:
            self.state.worker.deleteLater()
            self.state.worker = None
        gc.collect()

    def _on_generation_error(self, error_msg: str):
        """Handle generation error"""
        self.left_panel.set_status("Error", StatusState.ERROR)
        self.left_panel.set_generate_enabled(True)

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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"flux_generated_{timestamp}.png"

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Generated Image",
            default_name,
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)",
        )

        if not filepath:
            return

        try:
            img = Image.open(self.state.generated_image_path)
            ext = Path(filepath).suffix.lower()
            if ext in (".jpg", ".jpeg"):
                img.save(filepath, quality=95, optimize=True)
            else:
                # PNG and everything else: let PIL use format defaults
                img.save(filepath)
            QMessageBox.information(self, "Success", f"Image saved to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save image:\n{str(e)}")

    def _copy_seed(self):
        """Copy the last-used seed back into the seed field"""
        if self.state.current_seed is not None:
            self.left_panel.set_seed(self.state.current_seed)

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
        """Reset the form and previews; temp files are deleted but models stay loaded"""
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

        self.right_panel.clear_previews()
        gc.collect()

    def closeEvent(self, event):
        """Ensure the worker thread is stopped before the window closes"""
        if self.state.worker is not None and self.state.worker.isRunning():
            self.state.worker.quit()
            self.state.worker.wait()
        event.accept()


def main():
    """Launch the GUI application"""
    app = QApplication(sys.argv)
    app.setApplicationName("FLUX.2 Klein Generator")

    window = FluxGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
