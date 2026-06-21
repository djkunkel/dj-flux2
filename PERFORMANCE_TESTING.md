# Performance Testing Notes

Informal notes on generation performance and environment variable validation.
Not exhaustive — records findings from specific test sessions for future reference.

---

## ROCm 7.14 multi-arch (20260616) — Initial observations

**GPU:** AMD Radeon RX 9700 (gfx1201, RDNA4)
**Backend:** `rocm-nightly` via `https://rocm.nightlies.amd.com/whl-multi-arch/`
**torch:** `2.12.0+rocm7.14.0a20260616`
**MIOpen:** 3.5.2 (new version, fresh cache)
**triton:** `3.7.0+gitb4e20bbe`

### Env var validation

Each var tested by removing it and running `./run generate` at 512×512 and 1024×1024
with `flux.2-klein-4b`. Baseline (all vars set): ~10s at 512×512.

| Variable | Effect observed | Verdict |
|---|---|---|
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | No timing difference at any resolution | **No-op on ROCm 7.14** — safe to remove |
| `MIOPEN_FIND_MODE=FAST` + `MIOPEN_FIND_ENFORCE=NONE` | **Do not use.** Default DYNAMIC_HYBRID is better for repeated resolutions | **Remove** — see §MIOPEN find mode below |
| `PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512` | No effect at 512×512 or 1024×1024 | **No-op on ROCm 7.14** — safe to remove |

### MIOPEN_FIND_MODE behavior (20260620)

MIOpen find modes only matter when there is a **FindDb miss** (shape not yet tuned).

| Mode | On FindDb miss | Effect on FindDb | First run | Subsequent runs |
|---|---|---|---|---|
| `FAST` | AI heuristic fallback immediately | **Never updates** FindDb | ~10.4s at 896×576 | ~10.5s (still heuristic) |
| `DYNAMIC_HYBRID` (default) | Runs find machinery | **Adds entry** to FindDb | ~13.7s at 896×576 | **~10.0s** (cached, fastest) |

Key findings at 896×576 (flux.2-klein-4b, uncached shape):
- DYNAMIC_HYBRID costs **~3.3s extra** on first run vs FAST
- Second run is **~0.5s faster** than FAST because it uses the tuned entry
- For resolutions you run repeatedly (the common case), DYNAMIC_HYBRID wins long-term
- FAST never builds the cache — it perpetually uses heuristics

**Earlier notes claiming "removing MIOPEN vars costs ~3.5s" were wrong** — that cost
was the triton JIT cold-start, not the MIOpen find path. With triton cache warm,
there is no measurable MIOPEN-related overhead with the default mode.

**Conclusion:** `MIOPEN_FIND_MODE=FAST` is not needed. The default DYNAMIC_HYBRID
already provides tuned performance for repeated resolutions. Keep MIOPEN vars
**commented out** in `run`.

### Cold-start penalty on first use after nightly upgrade

**Symptom:** Generation with `flux.2-klein-9b-kv` at 1024×1024 takes ~21s per CLI
invocation, noticeably slower than remembered on ROCm 7.13.

**Root cause: triton JIT compilation cache is cold.**

Each time the ROCm/triton version changes, the `~/.triton/cache/` entries are
invalidated because the cache key includes the triton version string. The
transformer attention blocks use triton kernels that must be recompiled on first
execution. The compiled `.hsaco` files are then cached to disk for subsequent runs.

Key observations:
- In-process warm run (same Python process, second generation): **11s** vs 18s cold
- CLI invocations are always "cold" — each `./run generate` is a fresh process
- The triton cache (`~/.triton/cache/`) accumulates entries as you use the model;
  performance converges after 1-2 full generation runs at each resolution

**This is not MIOpen.** The MIOpen find db (`~/.config/miopen/`) only contains
convolution shapes from the VAE decoder, and `MIOPEN_FIND_MODE=FAST` intentionally
prevents that db from growing further (no benchmarking). The cold-start cost is
entirely in the triton attention kernels.

**Resolution:** Self-healing with normal use. Run a few generations at your typical
resolutions after each nightly upgrade and the cache will warm up. No action needed.

### Timing reference (flux.2-klein-9b-kv, 1024×1024, 4 steps)

| Condition | Wall time |
|---|---|
| Cold triton cache (fresh nightly install) | ~21s |
| Warm triton cache (2+ prior runs at same resolution) | ~11–15s |
| In-process second run (models already in VRAM) | ~11s |

---

## TODO / open questions

- [x] `MIOPEN_FIND_MODE=FAST` — resolved: default DYNAMIC_HYBRID is better for
  repeated resolutions; MIOPEN vars are not needed
- [x] `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` — resolved: no-op on ROCm 7.14
- [x] `PYTORCH_ALLOC_CONF` — resolved: no measurable effect on ROCm 7.14 at 512/1024
- Establish a proper timing baseline once the triton cache is fully warmed on 7.14,
  compare against the recorded 7.13 numbers
- Confirm `PYTORCH_ALLOC_CONF` is safe to remove at very high resolutions
  (1344×768, 1536×1024) where VRAM pressure is higher on the 9b-kv model
- Test img2img with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` removed (low priority)
- Investigate whether `MIOPEN_FIND_ENFORCE=NONE` alone has any effect without
  `MIOPEN_FIND_MODE=FAST` (low priority — both vars are now commented out)
