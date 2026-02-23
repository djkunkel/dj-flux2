# RTX 3080 (sm_86) Support Plan

**Status:** Ready to implement  
**Goal:** Support NVIDIA compute capability 8.6 (Ampere: RTX 3060/3070/3080/3090, A10, A40) without changing the existing sm_89+ (Ada Lovelace) path.

---

## Root Cause

`flux2/src/flux2/text_encoder.py:436` (read-only submodule) hardcodes FP8 model names:

```python
def load_qwen3_embedder(variant: str, device: str | torch.device = "cuda"):
    return Qwen3Embedder(model_spec=f"Qwen/Qwen3-{variant}-FP8", device=device)
```

`Qwen3-4B-FP8` and `Qwen3-8B-FP8` use `float8_e4m3fn` quantization, which requires native FP8
hardware (NVIDIA sm_89+, i.e. Ada Lovelace, RTX 4000 series). The RTX 3080 is sm_86 (Ampere)
and will raise a runtime error at model load time.

The non-FP8 equivalents exist on HuggingFace and use BF16 (works on sm_80+):
- `Qwen/Qwen3-4B` — used instead of `Qwen/Qwen3-4B-FP8`
- `Qwen/Qwen3-8B` — used instead of `Qwen/Qwen3-8B-FP8`

---

## Approach: Runtime Monkey-Patch

Since the submodule cannot be modified, we monkey-patch `flux2.text_encoder.load_qwen3_embedder`
and `flux2.util.load_qwen3_embedder` at Python runtime, before the first call to
`load_text_encoder()`. Both references must be patched: `util.py` imports the function by name,
binding it locally at import time, so patching only `text_encoder` is not sufficient.

The patch is only applied when `torch.cuda.get_device_capability()` returns a value less than
`(8, 9)`. On sm_89+ hardware, the function returns immediately — the original FP8 path runs
exactly as before, with zero overhead.

---

## Files to Change

### 1. `generate_image.py`

Add a module-level guard flag and a new function `_apply_bf16_text_encoder_patch()` after the
existing imports, and call it once from `generate_image()` before `model_cache.load_models()`.

**New function (add after imports, before `SUPPORTED_MODELS`):**

```python
_bf16_patch_applied = False

def _apply_bf16_text_encoder_patch() -> None:
    """Monkey-patch the FP8 Qwen3 text encoder to BF16 on pre-Ada GPUs.

    FP8 (float8_e4m3fn) requires compute capability >= 8.9 (Ada Lovelace).
    On sm_86 and below (Ampere, RTX 3000 series), we substitute the standard
    BF16 Qwen3 model. Image quality is equivalent; VRAM usage is higher (~8 GB
    vs ~4 GB during the encoding step, but the encoder is moved to CPU afterward).

    This function is a no-op on sm_89+ hardware.
    """
    global _bf16_patch_applied
    if _bf16_patch_applied:
        return

    cap = torch.cuda.get_device_capability()
    if cap >= (8, 9):
        _bf16_patch_applied = True
        return  # FP8 hardware — use original path unchanged

    # sm_86 or below: substitute BF16 models
    import flux2.text_encoder as _te
    import flux2.util as _util
    from flux2.text_encoder import Qwen3Embedder

    def _bf16_load_qwen3_embedder(variant: str, device="cuda"):
        model_spec = f"Qwen/Qwen3-{variant}"
        print(
            f"Note: GPU compute capability {cap[0]}.{cap[1]} detected "
            f"(sm_89+ required for FP8). Loading {model_spec} in BF16 instead. "
            f"Output quality is equivalent; VRAM usage during encoding is ~8 GB."
        )
        return Qwen3Embedder(model_spec=model_spec, device=device)

    _te.load_qwen3_embedder = _bf16_load_qwen3_embedder
    _util.load_qwen3_embedder = _bf16_load_qwen3_embedder
    _bf16_patch_applied = True
```

**Call site** in `generate_image()`, just before `model_cache.load_models()`:

```python
    _apply_bf16_text_encoder_patch()
    models = model_cache.load_models(model_name, AE_MODEL_NAME, device)
```

### 2. `download_models.py`

Apply the same patch before any model download that touches the text encoder, so that sm_86
users download the correct (BF16) model files rather than the FP8 ones.

Import and call `_apply_bf16_text_encoder_patch` from `generate_image` at the start of the
download routine, after confirming CUDA is available.

### 3. `gui_generate.py`

No changes required. The GUI calls `generate_image()` from `generate_image.py`, which already
applies the patch before loading models. The patch is idempotent (guarded by
`_bf16_patch_applied`) so it executes at most once per process regardless of how many
generations are run.

---

## VRAM Constraints on RTX 3080 (10 GB)

The VRAM management strategy in `generate_image.py` already staggers the three models so they
are never all on GPU simultaneously. The text encoder is moved to CPU immediately after encoding
the prompt.

| Phase | sm_89+ (FP8) | sm_86 (BF16) |
|---|---|---|
| Text encoder on GPU (encoding) | ~4 GB | ~8 GB |
| Text encoder on CPU (after encode) | ~0 GB | ~0 GB |
| Transformer on GPU (denoising) | ~8 GB (4B) | ~8 GB (4B) |
| Autoencoder on GPU (decode) | ~0.2 GB | ~0.2 GB |

On a 10 GB card with the BF16 encoder there is roughly 2 GB headroom during the encoding step.
512×512 generation works reliably. 1024×1024 native resolution may OOM during the encoding
phase; in that case use `--upscale 2` to generate at 512×512 and upscale afterward, which
avoids the overlap entirely.

---

## What Does NOT Change

- All sm_89+ behavior is byte-for-byte identical — no conditional imports, no changed defaults,
  no performance overhead on Ada Lovelace hardware.
- The submodule (`flux2/`) is not touched.
- No new pip dependencies are added.
- The monkey-patch executes at most once per process (guarded by `_bf16_patch_applied`).
- The ROCm/AMD path is unaffected in practice: ROCm also lacks FP8 hardware support on current
  consumer GPUs, so `get_device_capability()` on ROCm will return a value below (8, 9) and the
  patch will also apply there, which is correct behaviour.

---

## Testing Checklist

```bash
# Verify imports are still clean
uv run python -c "from flux2.util import load_flow_model; print('imports OK')"

# Text-to-image on sm_86 machine — look for the BF16 note in output
uv run generate_image.py "a cute cat" -o test_sm86.png -S 42

# Image-to-image on sm_86
uv run generate_image.py "pencil sketch" -i test_sm86.png -o sketch_sm86.png -S 42

# Verify sm_89+ machine is unaffected (BF16 note must NOT appear)
uv run generate_image.py "a cute cat" -o test_sm89.png -S 42

# Download models on sm_86 — verify BF16 model files are fetched (~8 GB each)
uv run download_models.py

# Force-test the patch on an sm_89+ machine by temporarily changing the
# capability threshold in _apply_bf16_text_encoder_patch from (8, 9) to (9, 0)
```

---

## Notes

- `Qwen/Qwen3-4B` (BF16) is ~8 GB on disk; `Qwen/Qwen3-4B-FP8` is ~4 GB. Users who previously
  ran on sm_89+ and switch to an sm_86 GPU will need to re-download the text encoder — the
  cached FP8 weights are a different model and will not be reused.
- The `ModelCache` singleton in `generate_image.py` caches by `model_name` string, not by
  encoder dtype. If a user somehow switches GPUs mid-process (not a realistic scenario), the
  cache would serve the wrong encoder. This edge case is out of scope for this project.
- AMD/ROCm: ROCm does not expose FP8 compute capability on any current consumer GPU. The patch
  will activate on ROCm as a side effect, which is the correct and desired behaviour. Verify by
  checking that `torch.cuda.get_device_capability()` returns a sub-(8,9) value on your ROCm
  setup before assuming it works.
