# text_remover.py: batch text and watermark removal

Detects text with EasyOCR's CRAFT detector and fills it in with LaMa inpainting.
Originals are never touched; results are written to a separate folder that mirrors the input structure.

## Install (Python 3.12+)

```bash
python3.12 -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate

# 1. PyTorch first. Pick the command for your hardware at https://pytorch.org/get-started/locally/
#    NVIDIA GPU example (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#    CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Everything else
pip install easyocr simple-lama-inpainting opencv-python-headless pillow numpy tqdm
```

If pip reports a version conflict from `simple-lama-inpainting` (it has historically pinned older
Pillow/NumPy versions), install it separately without its pins:

```bash
pip install easyocr opencv-python-headless pillow numpy tqdm fire
pip install --no-deps simple-lama-inpainting
```

First run downloads the CRAFT weights (~80 MB, to `~/.EasyOCR`) and the LaMa weights (~200 MB, to the
torch cache). After that it works offline.

## Example commands

```bash
# One folder, default settings (output: ./photos_cleaned)
python text_remover.py ./photos

# Recursive, explicit output, review images for quality checking
python text_remover.py ./photos -r -o ./out --previews --save-masks

# Check detection first without inpainting (writes masks + before|mask|after previews)
python text_remover.py ./photos -r -o ./out --detect-only

# Hunt harder for faint / small / vertical text (slower, more false positives)
python text_remover.py ./photos -r -o ./out --confidence 0.25 --low-text 0.25 \
    --mag-ratio 1.5 --rotations --mask-mode box --padding 6

# CPU only, no LaMa (fast, lower quality)
python text_remover.py ./photos --device cpu --inpainter opencv

# Apply hand-corrected masks (see below)
python text_remover.py ./photos -r -o ./out --mask-dir ./out/_masks --manual-masks replace
```

Run `python text_remover.py -h` for all options.

## Output layout

```
out/
  <same structure and filenames as input>
  _review/<name>.compare.jpg     original | detected mask in red | result   (--previews)
  _masks/<name>.png              white = will be removed                    (--save-masks)
  run.log
```

## How it works, and why

- **Detection.** CRAFT finds text regions without caring about language, font or position. For accuracy the
  script runs it on overlapping tiles plus a whole-image pass (large images no longer shrink small text below
  detectability), on the original, a CLAHE contrast-boosted and an inverted version (for low-contrast and
  semi-transparent marks), and optionally on 90°/270° rotations. All results are unioned. No location or
  appearance assumptions are made.
- **Mask.** By default, each detected region is narrowed to the pixels that differ from the local background,
  which means less of the picture is repainted. If that segmentation looks unreliable (very faint text, very
  thick strokes) it falls back to the full region. `--mask-mode box` always uses full regions.
- **Inpainting.** LaMa runs on a crop of context around each cluster of masked pixels. Only the masked pixels
  plus a ~1 px feather are written back, so everything else stays bit-identical to the input before saving.
  Output dimensions always match the input.

## Hardware

| Setup | Notes |
|---|---|
| NVIDIA GPU, 6 GB+ VRAM | Recommended. Roughly 1-5 s per image. 4 GB works with a lower `--max-inpaint-side` (e.g. 1024). |
| CPU only | Works, but expect tens of seconds per image (LaMa and CRAFT with 3 variants are both heavy). Use 16 GB RAM for large images. |
| Apple Silicon | Runs on CPU in this script (`--device auto` does not select MPS). |

Speed-ups: `--passes orig`, drop `--rotations`, larger `--tile-size`, or lower `--canvas-size`.

## Limitations (please read)

- **No claim of perfect removal.** Inpainting invents plausible pixels; it does not recover what was underneath.
  Over detailed or structured backgrounds (faces, fine patterns, straight edges, gradients with text across
  them), results can show smearing, repeated texture or ghosting. Judge with the `_review` previews.
- **Missed text is expected in some cases.** CRAFT can miss: very faint or very low-opacity watermarks,
  text smaller than ~8 px, heavily stylised/curved/graffiti lettering, text integrated into the scene (shop
  signs, labels on products, which are arguably content), tiled watermarks with extremely low contrast, and
  text that looks like texture. Lower `--confidence` / `--low-text` and try `--mag-ratio`, `--rotations`.
- **False positives happen too.** Logos, patterns, fences, foliage, or stripes may be treated as text and
  repainted. Lower sensitivity raises precision. The script warns if a mask covers more than 40% of an image.
- **CRAFT gives no per-box confidence score.** `--confidence` is the model's pixel-level text score threshold,
  a proxy for sensitivity rather than a calibrated probability.
- **Large or full-image watermarks** (big semi-transparent overlays) are reconstructed from surrounding
  content only, usually with visible softness. Crops above `--max-inpaint-side` are inpainted at reduced
  resolution inside the masked area.
- **Metadata and format.** EXIF orientation is applied to the pixels and other EXIF data is not copied (the ICC
  profile is). Only the first frame of multi-page TIFF / animated WEBP is processed. 16-bit images are
  converted to 8-bit. Lossy formats (JPEG/WEBP) are re-encoded (quality via `--jpeg-quality`, default 95);
  use PNG/TIFF input if you need a lossless chain.
- **Transparency.** Detection ignores the alpha channel; the original alpha is reattached unchanged for PNG,
  WEBP and TIFF output.
- **Legal/ethical.** Removing watermarks or signatures from images you don't own or have rights to may violate
  copyright or licensing. Use this on your own material or where you have permission.
- This code was written for the stated libraries' documented behaviour; test it on a small sample folder
  (`--detect-only`, then a few images) before running a large batch, since exact package versions vary.

## Correcting masks manually

1. Run with `--save-masks` (or `--detect-only`) to get `out/_masks/<image filename>.png`.
2. Open the mask in any editor (GIMP, Photoshop, Krita, Paint.NET). Paint **white** over text that was
   missed; paint **black** over false positives. Keep the same size and save as PNG.
3. Re-run with `--mask-dir out/_masks --manual-masks replace` to use your edited masks exactly
   (images without a mask file are detected as usual).
4. To only add regions on top of automatic detection, put just your white-on-black additions in a separate
   folder and use `--manual-masks union`.

Mask files are matched by relative path plus the full image filename and `.png`, e.g. `sub/cat.jpg` uses
`sub/cat.jpg.png`. The stem form `sub/cat.png` is also accepted.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower `--max-inpaint-side` (1024 or 768) and `--canvas-size`; close other GPU programs. The script already retries at half size, then falls back to OpenCV for that crop. |
| CUDA requested but not used | Install a CUDA build of PyTorch (see install step). Check `python -c "import torch; print(torch.cuda.is_available())"`. |
| Model download fails | Needs internet on first run (GitHub/torch hub for LaMa, EasyOCR model host). Behind a proxy, set `HTTPS_PROXY`; or download the weights manually into `~/.EasyOCR/model` and the torch hub cache. |
| `ImportError: simple_lama_inpainting` | Install it (see `--no-deps` note above) or run `--inpainter opencv`. |
| Dependency conflicts | Use a fresh virtualenv; install PyTorch first, then the rest. |
| Text remains | Lower `--confidence`/`--low-text`, add `--mag-ratio 1.5 --rotations`, use `--mask-mode box`, increase `--padding`, then fall back to manual masks. |
| Halos/ghost outlines left around letters | Increase `--padding` (6-10) or use `--mask-mode box`. |
| Too much repainted | Raise `--confidence`, use `--mask-mode refine`, lower `--padding`, `--passes orig`, then edit masks manually. |
| Slow | GPU, `--passes orig`, no `--rotations`, larger `--tile-size`. |
| Blurry patch inside a huge mask | Raise `--max-inpaint-side` if VRAM allows. |
