# AMD GPU (ROCm) Support Plan

> Investigation findings for adding AMD GPU support to dj-flux2

**Date**: January 25, 2026  
**Status**: Deferred - Complexity assessed, implementation pending

---

## Executive Summary

**Complexity Level**: MODERATE ⚠️  
**Estimated Effort**: 1-2 days of development + testing with AMD hardware  
**Recommended Approach**: Full device compatibility layer with forked flux2 submodule

Adding ROCm support is **feasible** but requires:
1. Device detection abstraction (simple)
2. Forking/patching the flux2 submodule (moderate)
3. AMD hardware for testing (blocker without hardware)
4. Updated installation documentation (simple)

---

## Current CUDA Dependencies

### Hardcoded CUDA calls in project code (8 locations):

**`generate_image.py` (5 locations):**
```python
Line 69:  torch.cuda.empty_cache()
Line 70:  device = torch.device("cuda")
Line 105: torch.cuda.empty_cache()
Line 112: torch.Generator(device="cuda").manual_seed(seed)
Line 114: dtype=torch.bfloat16, device="cuda"
```

**`upscale_image.py` (3 locations):**
```python
Line 81:  if not torch.cuda.is_available():
Line 111: model.cuda().eval()
Line 116: img_tensor = img_tensor.cuda()
```

**`flux2/src/flux2/sampling.py` (BFL submodule - cannot modify):**
```python
Line 72: ae.encode(img[None].cuda())[0]
```

---

## Good News ✅

### 1. PyTorch ROCm Compatibility
- **ROCm 6.1/6.2** supports PyTorch 2.8+ (our current version)
- `torch.cuda` API works transparently via HIP compatibility layer
- No need to rewrite PyTorch code - existing calls work on AMD GPUs
- Installation: `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2`

### 2. Minimal Code Changes Required
- Replace hardcoded `"cuda"` strings with dynamic detection
- Add device detection helper:
  ```python
  def get_device():
      if torch.cuda.is_available():
          return torch.device("cuda")
      return torch.device("cpu")
  ```
- Replace `.cuda()` calls with `.to(device)`

### 3. Small Codebase
- Only ~300 lines of main code to modify
- No custom CUDA kernels or C++ extensions
- Simple architecture: text encoder → transformer → VAE

---

## Challenges ⚠️

### 1. BFL flux2 Submodule (BLOCKER)
**Problem**: `flux2/sampling.py` line 72 has hardcoded `.cuda()`  
**Impact**: Cannot modify files in git submodule per project guidelines

**Options**:
1. **Fork & patch** (Recommended)
   - Fork `black-forest-labs/flux2` to your GitHub
   - Apply device compatibility patch
   - Point submodule to your fork
   - ✅ Full control
   - ⚠️ Maintenance overhead (must merge upstream updates)

2. **Monkey patching** (Hacky)
   - Override problematic functions at runtime
   - Keep upstream submodule unchanged
   - ✅ No fork maintenance
   - ⚠️ Fragile, breaks on BFL updates

3. **Upstream PR** (Cleanest but slowest)
   - Submit PR to BFL for device compatibility
   - Wait for review/merge
   - ✅ Benefits entire community
   - ⚠️ Depends on BFL's timeline and acceptance

### 2. Testing Requirements
- Need AMD GPU hardware (RX 7900 XTX, MI300, etc.)
- VRAM requirements likely similar (12GB+)
- Real-ESRGAN (Spandrel) compatibility unknown on ROCm
- Performance characteristics may differ from CUDA

### 3. Installation Complexity
- ROCm installation more complex than CUDA:
  - OS-specific packages (Ubuntu 22.04/24.04, RHEL, SLES)
  - Kernel drivers and GPU permissions
  - Architecture detection (gfx1100, gfx1030, gfx906, etc.)
- Cannot specify both CUDA and ROCm PyTorch in `pyproject.toml`
- Need conditional installation or user choice

### 4. Documentation Burden
- Installation guide needs ROCm section
- Troubleshooting for AMD-specific issues
- Performance benchmarks on AMD hardware
- Compatibility matrix (which AMD GPUs work)

---

## Recommended Implementation Plan

### Phase 1: Device Abstraction (1-2 hours)
```python
# Add to generate_image.py and upscale_image.py

def get_device():
    """Get the best available device (CUDA/ROCm or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        vendor = "NVIDIA" if "nvidia" in torch.cuda.get_device_name(0).lower() else "AMD"
        print(f"Using GPU: {torch.cuda.get_device_name(0)} ({vendor})")
        return device
    print("No GPU detected, using CPU (slow)")
    return torch.device("cpu")

# Replace all hardcoded:
# device = torch.device("cuda")
# WITH:
device = get_device()

# Replace all:
# .cuda()
# WITH:
# .to(device)
```

### Phase 2: Fork & Patch flux2 Submodule (2-3 hours)
1. Fork `https://github.com/black-forest-labs/flux2` to your GitHub
2. Create branch: `device-compatibility`
3. Patch `src/flux2/sampling.py` line 72:
   ```python
   # OLD:
   encoded = ae.encode(img[None].cuda())[0]
   
   # NEW:
   device = next(ae.parameters()).device  # Infer device from model
   encoded = ae.encode(img[None].to(device))[0]
   ```
4. Update `.gitmodules` to point to your fork:
   ```ini
   [submodule "flux2"]
       path = flux2
       url = https://github.com/djkunkel/flux2.git
       branch = device-compatibility
   ```

### Phase 3: Update Documentation (1 hour)
**Add to README.md**:
```markdown
### AMD GPU Support (ROCm)

For AMD Radeon RX 7000 series or MI series GPUs:

1. Install ROCm: https://rocm.docs.amd.com/
2. Install PyTorch for ROCm:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
   ```
3. Run normally:
   ```bash
   dj-flux2 "your prompt"
   ```

**Tested GPUs**: RX 7900 XTX (24GB VRAM)
**Minimum VRAM**: 12GB
```

### Phase 4: Testing (Requires AMD Hardware)
- [ ] Test text-to-image generation
- [ ] Test image-to-image transformation
- [ ] Test Lanczos upscaling (CPU, should work)
- [ ] Test Real-ESRGAN upscaling (GPU, may need Spandrel ROCm support)
- [ ] Benchmark performance vs NVIDIA
- [ ] Test GUI application

---

## Alternative Approaches

### Option A: Documentation-Only (Zero Code Changes)
**Effort**: 30 minutes  
**Approach**: Add `docs/AMD-ROCM.md` with instructions for users to:
1. Install PyTorch ROCm
2. Manually patch the 3 files with device detection
3. Mark as "community-maintained, experimental"

**Pros**: No maintenance burden  
**Cons**: Poor user experience, doesn't align with "minimal and educational" philosophy

### Option B: Basic Compatibility (Device Detection Only)
**Effort**: 2-3 hours  
**Approach**:
1. Add device detection to `generate_image.py` and `upscale_image.py`
2. Document flux2 submodule issue
3. Mark as "experimental - requires manual flux2 patching"

**Pros**: Simple, no submodule fork  
**Cons**: Incomplete solution, users still hit errors

### Option C: Full Support (Recommended)
**Effort**: 4-6 hours + hardware testing  
**Approach**: All phases above  

**Pros**: Professional, complete solution  
**Cons**: Maintenance overhead for forked submodule

---

## Decision Criteria

**Implement ROCm support if**:
- ✅ You have AMD GPU hardware for testing
- ✅ You're willing to maintain a forked flux2 submodule
- ✅ AMD users are a significant part of your audience

**Defer ROCm support if**:
- ❌ No AMD hardware available for testing
- ❌ Limited time for fork maintenance
- ❌ NVIDIA is the primary target audience

---

## ROCm Ecosystem (2026)

### Supported AMD GPUs
**Consumer (RDNA3/4)**:
- RX 7900 XTX (24GB)
- RX 7900 XT (20GB)
- RX 7800 XT (16GB)
- RX 7700 XT (12GB) ⚠️ Minimum VRAM

**Datacenter (CDNA)**:
- MI300X (192GB)
- MI250X (128GB)
- MI210 (64GB)

### Installation Requirements
- **OS**: Ubuntu 22.04/24.04, RHEL 9, SLES 15 (ROCm 6.2)
- **Kernel**: Linux 5.15+ (6.1+ recommended)
- **Driver**: amdgpu kernel module
- **ROCm**: 6.1 or 6.2 (matches PyTorch 2.8)

### PyTorch Compatibility
| PyTorch | ROCm | Status |
|---------|------|--------|
| 2.8.0   | 6.2  | ✅ Stable |
| 2.8.0   | 6.1  | ✅ Stable |
| 2.10.0  | 6.3  | 🔄 Upcoming |

---

## Resources

- **ROCm Docs**: https://rocm.docs.amd.com/
- **PyTorch ROCm**: https://pytorch.org/get-started/locally/ (select ROCm)
- **BFL flux2 Repo**: https://github.com/black-forest-labs/flux2
- **ROCm PyTorch Compatibility**: https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html

---

## Conclusion

AMD GPU support is **technically feasible** but requires:
1. Device abstraction code (simple)
2. Forking flux2 submodule (moderate complexity)
3. AMD hardware for testing (currently blocking)
4. Documentation updates (simple)

**Recommendation**: Defer until AMD hardware is available for testing. When ready, implement **Option C (Full Support)** with forked flux2 submodule for a professional, complete solution that maintains the project's "minimal and educational" philosophy.

---

**Next Steps When Ready**:
1. Acquire AMD GPU (RX 7900 XTX recommended)
2. Install ROCm 6.2 on test system
3. Implement Phase 1 (device abstraction)
4. Fork and patch flux2 (Phase 2)
5. Test all functionality
6. Update documentation
7. Create GitHub issue for community testing
