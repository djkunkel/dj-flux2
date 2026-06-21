---
name: generate-image
description: Generate images using the dj-flux2 API server (icons, backgrounds, textures, sprites, UI assets). Requires the server to be running (dj-flux2 serve).
---

## Generate images with dj-flux2

Use the `dj-flux2 api-generate` command to generate images via the running API server. This is a blocking command — it submits a job, waits for completion, and saves the result.

### Prerequisites

The API server must be running in a separate terminal:

```bash
dj-flux2 serve
```

### Basic usage

```bash
dj-flux2 api-generate "description of the image" -o path/to/output.png
```

### Full options

```
dj-flux2 api-generate "prompt" [options]

  -o, --output PATH        Output file path (default: output.png)
  -m, --model MODEL        Model name (default: flux.2-klein-4b)
  -W, --width PIXELS       Image width (default: 512, max: 2048)
  -H, --height PIXELS      Image height (default: 512, max: 2048)
  -S, --seed INT           Random seed for reproducibility
  --upscale {2,4}          Upscale the output by 2x or 4x
  --upscale-method METHOD  lanczos (fast, default) or realesrgan (AI quality)
  --host HOST              API server host (default: localhost)
  --port PORT              API server port (default: 8190)
```

### When to use this skill

Use this whenever you need to generate image assets for a project:

- App icons, button graphics, UI elements
- Background images, textures, patterns
- Sprites, game assets, pixel art
- Placeholder images for development
- Hero images, banners, illustrations

### Prompt guidelines for common asset types

**Icons and UI elements** — use small sizes, describe the style explicitly:
```bash
dj-flux2 api-generate "flat design red notification bell icon, white background, minimal" \
  -o assets/icons/bell.png -W 256 -H 256
```

**Backgrounds and textures** — use larger sizes or upscale:
```bash
dj-flux2 api-generate "seamless dark blue geometric pattern, subtle, professional" \
  -o assets/bg/pattern.png -W 1024 -H 1024
```

**Sprites and game art** — specify art style clearly:
```bash
dj-flux2 api-generate "pixel art treasure chest, 16-bit style, transparent background" \
  -o sprites/chest.png -W 256 -H 256
```

**Photos and illustrations** — use default or larger sizes:
```bash
dj-flux2 api-generate "professional headshot of a friendly robot, studio lighting" \
  -o images/hero.png -W 512 -H 512
```

**Upscaled high-res output** — generate small, then upscale for quality:
```bash
dj-flux2 api-generate "detailed mountain landscape, dramatic lighting" \
  -o images/landscape.png -W 512 -H 512 --upscale 2
```

### Prompt tips

- Be specific about style: "flat design", "photorealistic", "watercolor", "pixel art"
- Mention background: "white background", "transparent background", "dark background"
- For icons, add: "minimal", "simple", "clean lines"
- For textures, add: "seamless", "tileable", "repeating pattern"
- Width and height must be multiples of 16
- The model works best at 512x512; use --upscale for larger final output

### Generating multiple assets

When generating multiple images, run commands sequentially. The API server queues requests automatically:

```bash
dj-flux2 api-generate "red button icon" -o assets/btn-red.png -W 256 -H 256
dj-flux2 api-generate "green button icon" -o assets/btn-green.png -W 256 -H 256
dj-flux2 api-generate "blue button icon" -o assets/btn-blue.png -W 256 -H 256
```

### Reproducibility

Use `--seed` to reproduce a specific image. The seed is printed on completion:

```bash
# Generate and note the seed
dj-flux2 api-generate "a cute fox" -o fox.png
# Output: Saved: fox.png (seed: 2996839204)

# Reproduce the exact same image later
dj-flux2 api-generate "a cute fox" -o fox_copy.png -S 2996839204
```

### Error handling

- If the server is not running, the command exits with a clear error message
- If generation fails (e.g., GPU OOM), the error is reported and the command exits with code 1
- Use `--timeout` to set a maximum wait time (default: 300 seconds)
