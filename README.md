# dj-flux2

Minimal FLUX.2 Klein 4B image generation with CUDA support. Fast, simple, and educational.

## Features

- 🚀 **Fast**: Sub-second generation on RTX 4070 (4-step distilled model)
- 🎨 **Text-to-Image**: Generate images from text descriptions
- 🖼️ **Image-to-Image**: Transform images with prompts
- 💾 **Minimal**: Only ~200 lines of code + BFL submodule
- 🎓 **Educational**: Clear code structure for learning
- 🔧 **CUDA Accelerated**: Runs on NVIDIA GPUs

## Requirements

- Python 3.10+ (3.12 recommended)
- NVIDIA GPU with 12+ GB VRAM (RTX 3090/4070 or better)
- CUDA 12.x
- ~13 GB disk space for models

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/dj-flux2.git
cd dj-flux2

# Initialize the submodule
git submodule update --init --recursive
```

### 2. Install Dependencies

Using uv (recommended):
```bash
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e .
```

Using pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Setup Hugging Face Access

FLUX.2-dev requires accepting license terms:

1. Create account: https://huggingface.co/join
2. Accept license: https://huggingface.co/black-forest-labs/FLUX.2-dev
3. Create token: https://huggingface.co/settings/tokens
   - Enable: "Read access to contents of all public gated repos"
4. Login:
   ```bash
   huggingface-cli login
   ```

### 4. Download Models

```bash
python download_models.py
```

This downloads ~12.6 GB:
- FLUX.2 Klein 4B transformer (7.4 GB)
- Qwen3-4B-FP8 text encoder (4.9 GB)
- FLUX.2-dev autoencoder (321 MB)

### 5. Generate Your First Image

```bash
python generate_image.py "a cute cat sitting on a windowsill"
```

Output: `output.png`

## Usage

### Text-to-Image

```bash
# Basic usage
python generate_image.py "a majestic mountain landscape at sunset"

# With custom output path
python generate_image.py "a robot" -o my_robot.png

# High resolution
python generate_image.py "detailed portrait" -W 1024 -H 1024

# Reproducible with seed
python generate_image.py "abstract art" -S 42
```

### Image-to-Image

Transform existing images:

```bash
# Turn photo into oil painting
python generate_image.py "oil painting in impressionist style" \
  -i photo.jpg -o painting.png

# Convert to pencil sketch
python generate_image.py "pencil sketch, detailed line art, black and white" \
  -i portrait.jpg -o sketch.png

# Style transfer
python generate_image.py "watercolor painting with soft colors" \
  -i landscape.jpg -o watercolor.png
```

### All Options

```bash
python generate_image.py --help
```

```
Options:
  prompt              Text prompt (required)
  -i, --input         Input image for img2img
  -o, --output        Output path (default: output.png)
  -W, --width         Width in pixels (default: 512)
  -H, --height        Height in pixels (default: 512)
  -s, --steps         Denoising steps (default: 4)
  -g, --guidance      Guidance scale (default: 1.0)
  -S, --seed          Random seed for reproducibility
```

## Performance

### Generation Speed (RTX 4070, 512x512)

| Operation | Time |
|-----------|------|
| Cold start | ~15s |
| Warm generation | ~7s |

### Memory Usage

- **VRAM**: ~12 GB
- **RAM**: ~4 GB

### Resolution Limits

- **Minimum**: 64x64
- **Recommended**: 512x512 to 1024x1024
- **Maximum**: 1792x1792

## Architecture

```
Text Prompt → Qwen3-4B → FLUX.2 Klein 4B → VAE Decoder → Image
                ↑              ↑
            (4.9 GB)       (7.4 GB)
```

See [MODS-README.md](MODS-README.md) for detailed technical documentation.

## Troubleshooting

### Out of Memory

Reduce resolution:
```bash
python generate_image.py "prompt" -W 512 -H 512
```

### Slow Generation

Check GPU is being used:
```python
import torch
print(torch.cuda.is_available())
```

### Model Download Fails

1. Accept FLUX.2-dev license
2. Check token has gated repo access
3. Re-login: `huggingface-cli login`

## Learning Resources

- **MODS-README.md**: Detailed technical documentation
- **Black Forest Labs**: https://github.com/black-forest-labs/flux2
- **FLUX.2 Blog**: https://bfl.ai/blog/flux2-klein

## Project Structure

```
dj-flux2/
├── generate_image.py      # Main inference script
├── download_models.py     # Model download automation
├── flux2/                 # BFL submodule (git)
│   └── src/flux2/        # Core FLUX.2 code
├── pyproject.toml        # Dependencies
└── README.md             # This file
```

## Why This Project?

This is a **minimal, educational** implementation of FLUX.2 Klein:

✅ **Minimal**: ~200 lines of code  
✅ **Fast**: 4-step generation  
✅ **Clear**: Easy to understand  
✅ **Complete**: Text-to-image + image-to-image  
✅ **Maintainable**: BFL code via submodule  

**Not included** (from full BFL repo):
- ❌ Prompt upsampling (optional feature)
- ❌ API client (not needed for local)
- ❌ Training code (inference only)
- ❌ Watermarking (can be added)

## License

This project: MIT License

Black Forest Labs flux2 submodule: See `flux2/LICENSE.md`

FLUX.2 Klein 4B model: Apache 2.0

FLUX.2-dev model: Non-Commercial License

## Contributing

Improvements welcome! Please:
1. Keep it minimal
2. Add tests for new features
3. Update documentation

## Credits

- **Black Forest Labs** for FLUX.2 Klein
- **Antirez** for flux2.c inspiration
- **Hugging Face** for model hosting

---

Built for learning and experimentation. For production, use the official BFL API.
