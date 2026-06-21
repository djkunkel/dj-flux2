#!/usr/bin/env python
"""Blocking CLI wrapper for the dj-flux2 API server.

Submits a generation request, polls until complete, and downloads the result.
Designed for use by AI agents and scripts that want a simple one-command
interface to image generation.

Usage:
    dj-flux2 api-generate "a red button icon" -o assets/button.png
    dj-flux2 api-generate "forest background" --width 1024 --height 512
    ./run api-generate "pixel art sword" -o sprites/sword.png --width 256 --height 256

Requires the API server to be running (./run serve).
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8190


def api_url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


def post_json(url: str, data: dict) -> dict:
    """POST JSON and return parsed response. Raises on HTTP error."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read()).get("detail", "")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}: {detail or e.reason}") from None


def get_json(url: str) -> dict:
    """GET and return parsed JSON response."""
    try:
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read()).get("detail", "")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}: {detail or e.reason}") from None


def download_file(url: str, output_path: str) -> None:
    """Download a binary file from URL to disk."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        with open(output_path, "wb") as f:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                f.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an image via the dj-flux2 API server (blocking)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  dj-flux2 api-generate "a red button icon" -o assets/button.png
  dj-flux2 api-generate "forest background" --width 1024 --height 512
  dj-flux2 api-generate "pixel art sword" -o sprites/sword.png -W 256 -H 256

The API server must be running: dj-flux2 serve (or ./run serve)
""",
    )
    parser.add_argument("prompt", help="Text prompt describing the image")
    parser.add_argument(
        "-o", "--output", default="output.png", help="Output file path (default: output.png)"
    )
    parser.add_argument("-m", "--model", default=None, help="Model name")
    parser.add_argument("-W", "--width", type=int, default=None, help="Image width")
    parser.add_argument("-H", "--height", type=int, default=None, help="Image height")
    parser.add_argument("-s", "--steps", type=int, default=None, help="Denoising steps")
    parser.add_argument("-g", "--guidance", type=float, default=None, help="Guidance scale")
    parser.add_argument("-S", "--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--upscale", type=int, choices=[2, 4], default=None, help="Upscale factor")
    parser.add_argument(
        "--upscale-method",
        choices=["lanczos", "realesrgan"],
        default="lanczos",
        help="Upscale method (default: lanczos)",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"API server host (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"API server port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between status polls (default: 2.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Maximum wait time in seconds (default: 300)",
    )
    args = parser.parse_args()

    base = api_url(args.host, args.port, "")

    # Verify server is reachable
    try:
        get_json(f"{base}/models")
    except Exception:
        print(f"Error: Cannot reach dj-flux2 API server at {args.host}:{args.port}", file=sys.stderr)
        print("Start it with: dj-flux2 serve (or ./run serve)", file=sys.stderr)
        sys.exit(1)

    # Build request
    request_body: dict = {"prompt": args.prompt}
    if args.model:
        request_body["model"] = args.model
    if args.width:
        request_body["width"] = args.width
    if args.height:
        request_body["height"] = args.height
    if args.steps:
        request_body["steps"] = args.steps
    if args.guidance:
        request_body["guidance"] = args.guidance
    if args.seed is not None:
        request_body["seed"] = args.seed
    if args.upscale:
        request_body["upscale"] = args.upscale
        request_body["upscale_method"] = args.upscale_method

    # Submit
    try:
        result = post_json(f"{base}/generate", request_body)
    except RuntimeError as e:
        print(f"Error submitting job: {e}", file=sys.stderr)
        sys.exit(1)

    token = result["token"]
    print(f"Job submitted: {token[:8]}...")

    # Poll until complete
    start_time = time.time()
    last_progress = ""
    while True:
        elapsed = time.time() - start_time
        if elapsed > args.timeout:
            print(f"Error: Timed out after {args.timeout:.0f}s", file=sys.stderr)
            sys.exit(1)

        try:
            status = get_json(f"{base}/status/{token}")
        except RuntimeError as e:
            print(f"Error checking status: {e}", file=sys.stderr)
            sys.exit(1)

        job_status = status["status"]
        progress = status.get("progress_message", "")

        # Print progress updates
        if progress != last_progress:
            pos = status.get("queue_position")
            if pos:
                print(f"  [{elapsed:.0f}s] {progress} (queue position: {pos})")
            else:
                print(f"  [{elapsed:.0f}s] {progress}")
            last_progress = progress

        if job_status == "completed":
            break
        elif job_status == "failed":
            error = status.get("error", "Unknown error")
            print(f"Error: Generation failed: {error}", file=sys.stderr)
            sys.exit(1)
        elif job_status == "cancelled":
            print("Error: Job was cancelled", file=sys.stderr)
            sys.exit(1)

        time.sleep(args.poll_interval)

    # Download result
    try:
        download_file(f"{base}/result/{token}", args.output)
    except Exception as e:
        print(f"Error downloading result: {e}", file=sys.stderr)
        sys.exit(1)

    seed = status.get("seed", "unknown")
    print(f"Saved: {args.output} (seed: {seed})")


if __name__ == "__main__":
    main()
