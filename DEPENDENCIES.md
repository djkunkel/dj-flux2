# Dependency Management

## Overview

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

## Files

- **`pyproject.toml`** - Defines project metadata and dependency requirements (ranges)
- **`uv.lock`** - Locks exact versions of all dependencies (committed to git)

## Why Commit `uv.lock`?

### ✅ Benefits

1. **Reproducibility** - Everyone gets identical dependency versions
2. **Security** - All packages verified with SHA256 hashes
3. **CI/CD** - Deterministic builds across all environments
4. **Debugging** - Easy to see what changed between versions
5. **Speed** - `uv` can install from lock file without resolving

### 📊 What's Locked

The `uv.lock` file contains:
- Exact version of every package (including transitive dependencies)
- SHA256 hashes for security verification
- Source URLs for package downloads
- Python version compatibility markers

Example (49 packages locked):
```
accelerate==1.12.0
certifi==2026.1.4
torch==2.10.0+cu128
transformers==4.57.6
... and 45 more
```

## Installation

### For Users (Recommended)

Install from lock file for exact versions:
```bash
uv venv
uv sync
```

This uses `uv.lock` to install the exact tested versions.

### For Development

Install in editable mode:
```bash
uv venv
uv pip install -e .
```

This installs the package in editable mode while respecting `uv.lock`.

### Updating Dependencies

To update dependencies to latest versions:
```bash
# Update all dependencies
uv lock --upgrade

# Update specific package
uv lock --upgrade-package torch

# Then install updated versions
uv sync
```

After updating, test thoroughly and commit the new `uv.lock`.

## Dependency Requirements

### Core Dependencies

```toml
torch>=2.8.0              # PyTorch with CUDA 12.x
torchvision>=0.23.0       # Vision transforms
einops>=0.8.1             # Tensor operations
transformers>=4.56.1      # Qwen3 text encoder
safetensors>=0.4.5        # Fast model loading
pillow>=10.0.0            # Image I/O
huggingface-hub>=0.36.0   # Model downloads
accelerate>=1.0.0         # Required for device mapping
```

### Why Each Dependency?

- **torch/torchvision**: Core ML framework with CUDA support
- **einops**: Clean tensor reshaping (`rearrange`, `reduce`)
- **transformers**: Provides Qwen3-4B text encoder
- **safetensors**: Fast, secure model file format
- **pillow**: Image loading, saving, EXIF metadata
- **huggingface-hub**: Downloads models from Hugging Face
- **accelerate**: Required by transformers for device_map functionality

## Security

All packages in `uv.lock` include:
- SHA256 hash verification
- Source URL validation
- Exact version pinning

This prevents supply chain attacks and ensures package integrity.

## CI/CD Usage

In CI/CD pipelines, always use the lock file:

```yaml
# GitHub Actions example
- name: Install dependencies
  run: |
    uv venv
    uv sync
```

This ensures CI builds match local development exactly.

## Troubleshooting

### Lock file out of sync

```bash
# Regenerate lock file
uv lock

# Install from updated lock
uv sync
```

### Different Python version

The lock file supports Python 3.10, 3.11, and 3.12:
```toml
requires-python = ">=3.10,<3.13"
```

### Platform-specific dependencies

Some packages (like torch with CUDA) may have platform-specific wheels. The lock file handles this automatically with resolution markers.

## Comparison to Other Tools

| Tool | Lock File | Auto-commit? |
|------|-----------|--------------|
| pip | requirements.txt | ✅ Manual lock |
| pip-tools | requirements.txt | ✅ Yes |
| poetry | poetry.lock | ✅ Yes |
| uv | uv.lock | ✅ Yes |
| pipenv | Pipfile.lock | ✅ Yes |

## Best Practices

1. ✅ **Always commit `uv.lock`** after changes
2. ✅ **Review lock file diffs** in PRs
3. ✅ **Update dependencies regularly** but test thoroughly
4. ✅ **Use `uv sync`** for reproducible installs
5. ✅ **Document major version bumps** in commit messages

## Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [Lock File Format](https://docs.astral.sh/uv/concepts/resolution/)
- [Python Packaging Guide](https://packaging.python.org/)

---

**TL;DR**: Yes, commit `uv.lock` for reproducible builds! 🔒
