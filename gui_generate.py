#!/usr/bin/env python
"""PySide6 GUI for FLUX.2 Klein 4B image generation"""

import sys
import os
import gc
import shutil
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

    model_name: str
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
    """Container for mutable GUI state.

    Temp files are written into a private session directory that lives for the
    lifetime of the application.  Files are never deleted individually during a
    session — the whole directory is removed on explicit Clear or on app exit.
    This prevents the img2img workflow from losing its input file mid-session.
    """

    def __init__(self):
        self.input_image_path: Optional[str] = None
        self.generated_image_path: Optional[str] = None
        self.current_seed: Optional[int] = None
        self.worker: Optional[QThread] = None
        # Private temp directory for this session — created lazily on first use
        self._session_dir: Optional[str] = None

    @property
    def session_dir(self) -> str:
        """Return (creating if needed) the session-scoped temp directory."""
        if self._session_dir is None or not os.path.isdir(self._session_dir):
            self._session_dir = tempfile.mkdtemp(prefix="flux_gui_session_")
        return self._session_dir

    def new_temp_path(self, suffix: str = ".png") -> str:
        """Return a path for a new temp file inside the session directory."""
        fd, path = tempfile.mkstemp(suffix=suffix, dir=self.session_dir)
        os.close(fd)
        return path

    def reset_images(self):
        """Delete all session temp files and reset image-related state."""
        self._purge_session_dir()
        self.input_image_path = None
        self.generated_image_path = None
        self.current_seed = None

    def cleanup(self):
        """Full cleanup on app exit — remove the session directory."""
        self._purge_session_dir()

    def _purge_session_dir(self):
        """Delete the session temp directory and all files inside it."""
        if self._session_dir and os.path.isdir(self._session_dir):
            try:
                shutil.rmtree(self._session_dir)
            except Exception:
                pass
        self._session_dir = None


class GenerationWorker(QThread):
    """Worker thread for image generation"""

    # Signals for thread-safe communication
    progress = Signal(str, str)  # message, color
    finished = Signal(str, object)  # image_path, seed
    error = Signal(str)  # error_message
    oom_occurred = Signal()  # fired when torch.cuda.OutOfMemoryError is raised

    def __init__(self, params: GenerationParams, state: "GuiState"):
        super().__init__()
        self.params = params
        self._state = state

    def run(self):
        """Run generation in background thread.

        Error handling is split into two phases so we can give targeted messages:

        Phase 1 — model loading: catches SystemExit (from sys.exit() calls buried
        inside the flux2 submodule when a repo is inaccessible) and GatedRepoError
        (HuggingFace license-gate) before they can kill the process or produce an
        unhelpful traceback.

        Phase 2 — inference + upscale: catches all other exceptions and surfaces
        them as a generic generation error.

        Temp files are written into the session directory (GuiState.session_dir)
        and are never deleted by the worker.  The main thread owns their lifetime:
        they persist until the user clicks Clear or the app exits.  This ensures
        the img2img source image is never deleted mid-session.
        """
        try:
            # ---- Phase 1: model loading ----
            self.progress.emit("Loading models...", "orange")
            try:
                from huggingface_hub.errors import (
                    GatedRepoError,
                    RepositoryNotFoundError,
                )
                from generate_image import model_cache, AE_MODEL_NAME
                import torch

                device = torch.device("cuda")
                model_cache.load_models(
                    self.params["model_name"], AE_MODEL_NAME, device
                )
            except GatedRepoError:
                model_name = self.params["model_name"]
                repo = f"black-forest-labs/{model_name.replace('flux.2-', 'FLUX.2-')}"
                self.error.emit(
                    f"Access to {model_name!r} is restricted.\n\n"
                    f"You need to accept the license agreement before downloading:\n"
                    f"https://huggingface.co/{repo}\n\n"
                    f"Log in with 'huggingface-cli login' if you haven't already."
                )
                return
            except RepositoryNotFoundError:
                model_name = self.params["model_name"]
                self.error.emit(
                    f"Repository for {model_name!r} was not found.\n\n"
                    f"Check your internet connection and HuggingFace login."
                )
                return
            except SystemExit:
                model_name = self.params["model_name"]
                self.error.emit(
                    f"Could not load {model_name!r}.\n\n"
                    f"This usually means the model repository is inaccessible. "
                    f"Check that you have accepted the license at:\n"
                    f"https://huggingface.co/black-forest-labs\n\n"
                    f"Also verify your internet connection and HuggingFace login "
                    f"('huggingface-cli login')."
                )
                return

            # ---- Phase 2: inference + optional upscale ----
            # All output paths live inside the session directory and are kept
            # alive until Clear / app exit — never deleted here in the worker.
            out_path = self._state.new_temp_path(suffix=".png")

            self.progress.emit("Generating image...", "orange")
            actual_seed = generate_image(
                prompt=self.params["prompt"],
                output_path=out_path,
                input_image=self.params["input_image"],
                width=self.params["width"],
                height=self.params["height"],
                num_steps=self.params["steps"],
                guidance=self.params["guidance"],
                seed=self.params["seed"],
                model_name=self.params["model_name"],
            )

            final_path = out_path
            if self.params["do_upscale"]:
                self.progress.emit("Upscaling...", "orange")
                upscaled_path = self._state.new_temp_path(suffix=".png")
                upscale_image(
                    input_path=out_path,
                    output_path=upscaled_path,
                    scale=self.params["upscale_scale"],
                    method=self.params["upscale_method"],
                )
                final_path = upscaled_path

            self.finished.emit(final_path, actual_seed)

        except SystemExit:
            self.error.emit(
                "The generation process exited unexpectedly.\n\n"
                "A required model file may be inaccessible. "
                "Check your internet connection and HuggingFace login."
            )
        except torch.cuda.OutOfMemoryError as e:
            self.error.emit(
                f"Out of GPU memory: {e}\n\n"
                "Models have been unloaded. Try a smaller resolution, or switch "
                "to a lighter model (e.g. flux.2-klein-4b)."
            )
            self.oom_occurred.emit()
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
        self.left_panel.model_changed.connect(self._on_model_change)
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

    def _on_model_change(self, model_name: str):
        """Unload cached models immediately when the user selects a different model.

        This frees VRAM right away rather than waiting until the next Generate,
        and ensures the status indicator stays accurate.
        """
        if model_cache.is_loaded():
            model_cache.clear()
            self.left_panel.update_model_status(False)

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

        # Stop any still-running worker to prevent concurrent GPU access
        if self.state.worker is not None:
            self.state.worker.progress.disconnect()
            self.state.worker.finished.disconnect()
            self.state.worker.error.disconnect()
            self.state.worker.oom_occurred.disconnect()
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

        self.state.worker = GenerationWorker(params, self.state)
        self.state.worker.progress.connect(self._on_progress)
        self.state.worker.finished.connect(self._on_generation_complete)
        self.state.worker.error.connect(self._on_generation_error)
        self.state.worker.oom_occurred.connect(self._on_oom)
        self.state.worker.start()

    def _on_progress(self, message: str, color: str):
        """Forward worker progress to the status label"""
        # Map the colour string to the nearest StatusState
        state = StatusState.RUNNING if color == "orange" else StatusState.SUCCESS
        self.left_panel.set_status(message, state)

    def _on_generation_complete(self, image_path: str, seed):
        """Handle successful generation"""
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

    def _on_oom(self):
        """Handle GPU out-of-memory: flush cache and reset preview state.

        OOM leaves CUDA in a dirty state — unreleased tensors, fragmented
        allocator, and a model that may be partially on GPU. Clearing the
        model cache here forces a clean reload on the next generation attempt,
        and resetting generated_image_path prevents stale file references from
        reaching display_image() on the next successful run.
        """
        import torch

        # Flush any CUDA allocations left behind by the failed inference pass
        torch.cuda.empty_cache()

        # Force-unload models — their GPU state is unknown after OOM
        if model_cache.is_loaded():
            model_cache.clear()
        self.left_panel.update_model_status(False)

        # Drop the stale path reference so the next success starts fresh
        self.state.generated_image_path = None

        # Clear the output preview — the last displayed image's temp file
        # may have been deleted by the worker's finally block
        self.right_panel.output_preview.clear()

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

    def _clear(self):
        """Reset the form, previews, and session temp files. Models stay loaded."""
        self.left_panel.reset_to_defaults()
        # reset_images() purges the session temp directory and clears path refs
        self.state.reset_images()
        self.right_panel.clear_previews()
        gc.collect()

    def closeEvent(self, event):
        """Stop the worker thread and clean up session temp files on exit."""
        if self.state.worker is not None and self.state.worker.isRunning():
            self.state.worker.quit()
            self.state.worker.wait()
        self.state.cleanup()
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
