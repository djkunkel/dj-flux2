#!/usr/bin/env python
"""HTTP API server for FLUX.2 Klein image generation.

Starts a FastAPI server that queues generation requests and processes them
sequentially.  Jobs are ordered by model name first (to minimise GPU model
reloads) then by submission time (FIFO within the same model).

Each request returns a token that can be used to poll status via REST or
subscribe to real-time progress via WebSocket.

Usage:
    ./run serve                     # default 0.0.0.0:8190
    ./run serve --port 9000         # custom port
    python serve_api.py --host 127.0.0.1 --port 8190
"""

import argparse
import asyncio
import base64
import dataclasses
import gc
import os
import queue
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — same pattern as the other root-level scripts
# ---------------------------------------------------------------------------
script_dir = Path(__file__).parent.resolve()
flux2_src = script_dir / "flux2" / "src"
sys.path.insert(0, str(flux2_src))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from generate_image import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    generate_image,
    model_cache,
    read_config,
)
from upscale_image import upscale_image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_DIR = script_dir / "output"
MAX_QUEUED_JOBS = 50
MAX_TERMINAL_JOBS = 100  # keep last N completed/failed jobs


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Job:
    """Represents a single image generation request."""

    token: str
    status: str  # queued | running | completed | failed | cancelled
    prompt: str
    model: str
    width: int
    height: int
    steps: int | None
    guidance: float | None
    seed: int | None
    input_image_paths: list[str]  # temp files for decoded base64 img2img input(s)
    do_upscale: bool
    upscale_scale: int
    upscale_method: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    output_path: str | None = None
    error: str | None = None
    progress_message: str = "Queued"


# ---------------------------------------------------------------------------
# Job store — thread-safe dict of {token: Job}
# ---------------------------------------------------------------------------
class JobStore:
    """Thread-safe storage for jobs with automatic pruning."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        """Add a job.  Raises ValueError if the queue is full."""
        with self._lock:
            queued = sum(1 for j in self._jobs.values() if j.status == "queued")
            if queued >= MAX_QUEUED_JOBS:
                raise ValueError(
                    f"Queue is full ({MAX_QUEUED_JOBS} jobs). "
                    "Try again later."
                )
            self._jobs[job.token] = job

    def get(self, token: str) -> Job | None:
        with self._lock:
            return self._jobs.get(token)

    def cancel(self, token: str) -> tuple[bool, str]:
        """Cancel a queued job.  Returns (success, reason)."""
        with self._lock:
            job = self._jobs.get(token)
            if job is None:
                return False, "Job not found"
            if job.status != "queued":
                return False, f"Cannot cancel job with status '{job.status}'"
            job.status = "cancelled"
            job.completed_at = datetime.now(timezone.utc)
            return True, "Cancelled"

    def list_all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at)

    def queued_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == "queued")

    def queue_position(self, token: str) -> int | None:
        """Return 1-based queue position, or None if not queued."""
        with self._lock:
            queued = sorted(
                (j for j in self._jobs.values() if j.status == "queued"),
                key=lambda j: j.created_at,
            )
            for i, j in enumerate(queued, 1):
                if j.token == token:
                    return i
            return None

    def prune(self) -> None:
        """Keep only the last MAX_TERMINAL_JOBS completed/failed/cancelled jobs."""
        with self._lock:
            terminal = sorted(
                (
                    j
                    for j in self._jobs.values()
                    if j.status in ("completed", "failed", "cancelled")
                ),
                key=lambda j: j.completed_at or j.created_at,
            )
            to_remove = terminal[:-MAX_TERMINAL_JOBS] if len(terminal) > MAX_TERMINAL_JOBS else []
            for j in to_remove:
                del self._jobs[j.token]


# ---------------------------------------------------------------------------
# Priority queue — (model_name, timestamp, token)
# ---------------------------------------------------------------------------
class ModelPriorityQueue:
    """Priority queue that groups by model name, then FIFO."""

    def __init__(self) -> None:
        self._queue: queue.PriorityQueue[tuple[str, float, str]] = queue.PriorityQueue()

    def put(self, job: Job) -> None:
        self._queue.put((job.model, job.created_at.timestamp(), job.token))

    def get(self, timeout: float = 1.0) -> tuple[str, float, str] | None:
        """Get the next item, or None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self) -> int:
        return self._queue.qsize()


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
store = JobStore()
job_queue = ModelPriorityQueue()
shutdown_event = threading.Event()


# ---------------------------------------------------------------------------
# Worker thread — processes jobs sequentially
# ---------------------------------------------------------------------------
def generation_worker() -> None:
    """Pull jobs from the priority queue and run them one at a time.

    The queue sorts by (model_name, timestamp) so all jobs for the same model
    run consecutively, avoiding unnecessary model reloads.
    """
    import torch  # import here so the module can be loaded without CUDA

    while not shutdown_event.is_set():
        item = job_queue.get(timeout=1.0)
        if item is None:
            continue  # timeout, check shutdown flag and loop

        _model_name, _ts, token = item
        job = store.get(token)
        if job is None or job.status == "cancelled":
            continue  # job was cancelled while queued

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        job.progress_message = "Loading models..."

        try:
            job.progress_message = "Generating image..."
            seed = generate_image(
                prompt=job.prompt,
                output_path=job.output_path,
                input_image=job.input_image_paths or None,
                width=job.width,
                height=job.height,
                num_steps=job.steps,
                guidance=job.guidance,
                seed=job.seed,
                model_name=job.model,
            )

            if job.do_upscale:
                job.progress_message = f"Upscaling {job.upscale_scale}x ({job.upscale_method})..."
                upscale_image(
                    input_path=job.output_path,
                    output_path=job.output_path,
                    scale=job.upscale_scale,
                    method=job.upscale_method,
                )

            job.seed = seed
            job.status = "completed"
            job.completed_at = datetime.now(timezone.utc)
            job.progress_message = "Complete"
            print(f"✓ Job {job.token[:8]} completed: {job.output_path}")

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            model_cache.clear()
            gc.collect()
            job.status = "failed"
            job.completed_at = datetime.now(timezone.utc)
            job.error = "GPU out of memory. Models have been unloaded — try a smaller resolution."
            job.progress_message = "Failed (OOM)"
            print(f"✗ Job {job.token[:8]} OOM")

        except Exception as e:
            job.status = "failed"
            job.completed_at = datetime.now(timezone.utc)
            job.error = str(e)
            job.progress_message = "Failed"
            print(f"✗ Job {job.token[:8]} failed: {e}")

        finally:
            # Clean up temp img2img input file(s)
            for p in job.input_image_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
            store.prune()


# ---------------------------------------------------------------------------
# Pydantic models for request / response
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    """Request body for POST /generate."""

    prompt: str
    model: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None
    seed: int | None = None
    input_image_base64: str | list[str] | None = None
    upscale: int | None = None
    upscale_method: str = "lanczos"


class GenerateResponse(BaseModel):
    token: str


class JobStatusResponse(BaseModel):
    token: str
    status: str
    prompt: str
    model: str
    width: int
    height: int
    seed: int | None
    created_at: str
    started_at: str | None
    completed_at: str | None
    error: str | None
    progress_message: str
    queue_position: int | None


class ModelsResponse(BaseModel):
    supported: list[str]
    loaded: str | None
    vram_usage: str


class CancelResponse(BaseModel):
    success: bool
    message: str


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    _app = FastAPI(
        title="dj-flux2 API",
        description="HTTP API for FLUX.2 Klein image generation",
    )

    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # POST /generate
    # ------------------------------------------------------------------
    @_app.post("/generate", response_model=GenerateResponse, status_code=201)
    def submit_generation(req: GenerateRequest):
        # Validate model
        cfg = read_config()

        model = req.model or cfg.get("model", DEFAULT_MODEL)
        if model not in SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {model!r}. Choose from: {', '.join(SUPPORTED_MODELS)}",
            )

        width = req.width or int(cfg.get("width", 512))
        height = req.height or int(cfg.get("height", 512))

        if width <= 0 or height <= 0:
            raise HTTPException(status_code=400, detail="Width and height must be positive")
        if width > 2048 or height > 2048:
            raise HTTPException(status_code=400, detail="Width and height must be 2048 or under")

        # Steps and guidance: None means "use model default" (handled by generate_image)
        steps = req.steps
        if steps is None and "steps" in cfg:
            try:
                steps = int(cfg["steps"])
            except ValueError:
                steps = None

        guidance = req.guidance
        if guidance is None and "guidance" in cfg:
            try:
                guidance = float(cfg["guidance"])
            except ValueError:
                guidance = None

        # Validate upscale
        do_upscale = req.upscale is not None
        upscale_scale = req.upscale or 2
        if do_upscale and upscale_scale not in (2, 4):
            raise HTTPException(status_code=400, detail="upscale must be 2 or 4")
        if req.upscale_method not in ("lanczos", "realesrgan"):
            raise HTTPException(
                status_code=400,
                detail="upscale_method must be 'lanczos' or 'realesrgan'",
            )

        # Check queue capacity
        if store.queued_count() >= MAX_QUEUED_JOBS:
            raise HTTPException(
                status_code=429,
                detail=f"Queue is full ({MAX_QUEUED_JOBS} jobs). Try again later.",
            )

        # Decode base64 input image(s) for img2img
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

        # Build output path: output/YYYYMMDD_HHMMSS_{token[:8]}.png
        token = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        output_path = str(OUTPUT_DIR / f"{timestamp}_{token[:8]}.png")

        job = Job(
            token=token,
            status="queued",
            prompt=req.prompt,
            model=model,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=req.seed,
            input_image_paths=input_image_paths,
            do_upscale=do_upscale,
            upscale_scale=upscale_scale,
            upscale_method=req.upscale_method,
            created_at=now,
            output_path=output_path,
        )

        store.add(job)
        job_queue.put(job)

        print(f"+ Job {token[:8]} queued: {req.prompt!r} ({model}, {width}x{height})")
        return GenerateResponse(token=token)

    # ------------------------------------------------------------------
    # GET /status/{token}
    # ------------------------------------------------------------------
    @_app.get("/status/{token}", response_model=JobStatusResponse)
    def get_status(token: str):
        job = store.get(token)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _job_to_status(job)

    # ------------------------------------------------------------------
    # GET /result/{token}
    # ------------------------------------------------------------------
    @_app.get("/result/{token}")
    def get_result(token: str):
        job = store.get(token)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "completed":
            raise HTTPException(
                status_code=409,
                detail=f"Job is not completed (status: {job.status})",
            )
        if not job.output_path or not Path(job.output_path).exists():
            raise HTTPException(status_code=404, detail="Output image not found on disk")
        return FileResponse(
            job.output_path,
            media_type="image/png",
            filename=Path(job.output_path).name,
        )

    # ------------------------------------------------------------------
    # GET /queue
    # ------------------------------------------------------------------
    @_app.get("/queue", response_model=list[JobStatusResponse])
    def list_queue():
        jobs = store.list_all()
        return [_job_to_status(j) for j in jobs]

    # ------------------------------------------------------------------
    # POST /cancel/{token}
    # ------------------------------------------------------------------
    @_app.post("/cancel/{token}", response_model=CancelResponse)
    def cancel_job(token: str):
        success, message = store.cancel(token)
        if not success and message == "Job not found":
            raise HTTPException(status_code=404, detail=message)
        if not success:
            raise HTTPException(status_code=409, detail=message)
        print(f"- Job {token[:8]} cancelled")
        return CancelResponse(success=True, message=message)

    # ------------------------------------------------------------------
    # GET /models
    # ------------------------------------------------------------------
    @_app.get("/models", response_model=ModelsResponse)
    def list_models():
        return ModelsResponse(
            supported=list(SUPPORTED_MODELS),
            loaded=model_cache._model_name,
            vram_usage=model_cache.get_memory_estimate(),
        )

    # ------------------------------------------------------------------
    # WebSocket /ws/{token}
    # ------------------------------------------------------------------
    @_app.websocket("/ws/{token}")
    async def job_websocket(websocket: WebSocket, token: str):
        await websocket.accept()

        job = store.get(token)
        if job is None:
            await websocket.send_json({"type": "error", "message": "Job not found"})
            await websocket.close(code=4004)
            return

        try:
            last_msg: dict | None = None
            while job.status in ("queued", "running"):
                msg = {
                    "type": "progress",
                    "status": job.status,
                    "queue_position": store.queue_position(token),
                    "progress": job.progress_message,
                }
                # Only send when state changes
                if msg != last_msg:
                    await websocket.send_json(msg)
                    last_msg = msg
                await asyncio.sleep(0.5)

            # Send final status
            final = {
                "type": "complete" if job.status == "completed" else "error",
                "status": job.status,
                "seed": job.seed,
                "error": job.error,
            }
            await websocket.send_json(final)

        except WebSocketDisconnect:
            pass  # client left — nothing to clean up

    return _app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _job_to_status(job: Job) -> JobStatusResponse:
    """Convert a Job dataclass to the API response model."""
    return JobStatusResponse(
        token=job.token,
        status=job.status,
        prompt=job.prompt,
        model=job.model,
        width=job.width,
        height=job.height,
        seed=job.seed,
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error=job.error,
        progress_message=job.progress_message,
        queue_position=store.queue_position(job.token),
    )


# ---------------------------------------------------------------------------
# FastAPI app instance (used by uvicorn)
# ---------------------------------------------------------------------------
app = create_app()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(
        description="dj-flux2 HTTP API server for image generation",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8190,
        help="Port number (default: 8190)",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Start the worker thread
    worker = threading.Thread(target=generation_worker, daemon=True, name="generation-worker")
    worker.start()

    print(f"dj-flux2 API server starting on http://{args.host}:{args.port}")
    print(f"API docs:   http://{args.host}:{args.port}/docs")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        print("\nShutting down...")
        shutdown_event.set()
        worker.join(timeout=5.0)
        print("✓ Server stopped")


if __name__ == "__main__":
    main()
