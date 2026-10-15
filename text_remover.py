#!/usr/bin/env python3
"""
text_remover.py - batch removal of text / watermarks from images.

Pipeline
  1. Detection : EasyOCR's CRAFT text detector (detection only, no recognition),
                 run on overlapping tiles + a global pass, on several contrast
                 variants (original / CLAHE / inverted) and optionally rotated
                 90/270 degrees. All detections are unioned into one mask.
  2. Mask      : polygons are refined to the actual text pixels (or kept as
                 full boxes), then dilated by --padding. Optional hand-made
                 masks can be merged in or can replace detection.
  3. Inpainting: LaMa (via simple-lama-inpainting) on context crops around each
                 mask cluster. Only masked pixels (plus a ~1px feather) are
                 written back, so the rest of the image is untouched.
                 OpenCV Telea is available as a lightweight fallback.

Originals are never modified; results go to a separate output folder that
mirrors the input folder structure.

Requires Python 3.12+. See README.md for installation and usage.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
log = logging.getLogger("text_remover")


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Batch text/watermark removal (CRAFT detection + LaMa inpainting).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=Path, help="Input folder (or a single image file)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output folder (default: <input>_cleaned next to the input)")
    p.add_argument("-r", "--recursive", action="store_true", help="Process subfolders too")

    g = p.add_argument_group("detection")
    g.add_argument("--confidence", type=float, default=0.4,
                   help="CRAFT text score threshold (0-1). Lower = more detections "
                        "(better recall, more false positives). EasyOCR default is 0.7")
    g.add_argument("--low-text", type=float, default=0.3,
                   help="CRAFT low-text bound; lower grows regions / catches faint text")
    g.add_argument("--link-threshold", type=float, default=0.4,
                   help="CRAFT affinity threshold for linking characters into words/lines")
    g.add_argument("--min-size", type=int, default=8, help="Ignore boxes smaller than this (px)")
    g.add_argument("--canvas-size", type=int, default=2560, help="CRAFT working resolution")
    g.add_argument("--mag-ratio", type=float, default=1.0,
                   help="Upscale factor before detection; try 1.5-2 for tiny text")
    g.add_argument("--tile-size", type=int, default=1600,
                   help="Tile large images for detection so small text isn't downscaled "
                        "away (0 disables tiling)")
    g.add_argument("--tile-overlap", type=float, default=0.25, help="Tile overlap fraction")
    g.add_argument("--passes", default="orig,clahe,invert",
                   help="Comma list of detection variants: orig, clahe, invert")
    g.add_argument("--rotations", action="store_true",
                   help="Also detect on 90/270 degree rotated copies (vertical text; slower)")
    g.add_argument("--max-box-ratio", type=float, default=0.5,
                   help="Discard a detected box covering more than this fraction of the image")
    g.add_argument("--langs", default="en",
                   help="Comma list of EasyOCR language codes used to init the reader")

    m = p.add_argument_group("mask")
    m.add_argument("--padding", type=int, default=4, help="Mask dilation in pixels")
    m.add_argument("--mask-mode", choices=["refine", "box"], default="refine",
                   help="refine: mask only text-like pixels inside each box (preserves more "
                        "background); box: mask whole detected polygons (safer for faint text)")
    m.add_argument("--mask-dir", type=Path, default=None,
                   help="Folder of manual masks (white = remove) named <image filename>.png, "
                        "mirroring the input structure")
    m.add_argument("--manual-masks", choices=["union", "replace"], default="union",
                   help="How manual masks combine with detection")

    i = p.add_argument_group("inpainting")
    i.add_argument("--inpainter", choices=["lama", "opencv"], default="lama")
    i.add_argument("--max-inpaint-side", type=int, default=2048,
                   help="Crops larger than this are inpainted downscaled (masked area only)")
    i.add_argument("--cluster-gap", type=int, default=48,
                   help="Mask regions closer than this (px) are inpainted together")
    i.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    o = p.add_argument_group("output")
    o.add_argument("--jpeg-quality", type=int, default=95)
    o.add_argument("--webp-lossless", action="store_true")
    o.add_argument("--skip-existing", action="store_true", help="Skip outputs that already exist")
    o.add_argument("--previews", action="store_true",
                   help="Save before | mask overlay | after comparisons to <output>/_review")
    o.add_argument("--save-masks", action="store_true",
                   help="Save final masks to <output>/_masks (editable; feed back via --mask-dir)")
    o.add_argument("--detect-only", action="store_true",
                   help="Only detect: write masks + previews, no inpainting, no output images")
    o.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def collect_images(root: Path, recursive: bool, out_dir: Path):
    if root.is_file():
        return [root], root.parent
    it = root.rglob("*") if recursive else root.glob("*")
    out_res = out_dir.resolve()
    files = []
    for f in it:
        if not f.is_file() or f.suffix.lower() not in EXTS:
            continue
        if out_res in f.resolve().parents:  # never re-process our own output
            continue
        files.append(f)
    return sorted(files), root


def load_image(path: Path):
    """Return (rgb uint8 HxWx3, alpha uint8 or None, icc_profile or None)."""
    im = Image.open(path)
    im.load()
    icc = im.info.get("icc_profile")
    im = ImageOps.exif_transpose(im)  # bake in EXIF orientation
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        arr = np.array(im.convert("RGBA"))
        return arr[..., :3].copy(), arr[..., 3].copy(), icc
    if im.mode.startswith("I") or im.mode == "F":  # 16/32-bit grayscale -> 8-bit
        a = np.array(im, dtype=np.float32)
        a = a / (65535.0 if a.max() > 255 else 255.0) * 255.0
        return np.repeat(np.clip(a, 0, 255).astype(np.uint8)[..., None], 3, axis=2), None, icc
    return np.array(im.convert("RGB")), None, icc


def save_image(path: Path, rgb, alpha, icc, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if alpha is not None and ext in {".png", ".webp", ".tif", ".tiff"}:
        im = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    else:
        im = Image.fromarray(rgb, "RGB")
    kw = {}
    if icc:
        kw["icc_profile"] = icc
    if ext in {".jpg", ".jpeg"}:
        im.save(path, "JPEG", quality=args.jpeg_quality, subsampling=0, optimize=True, **kw)
    elif ext == ".png":
        im.save(path, "PNG", compress_level=6, **kw)
    elif ext == ".webp":
        im.save(path, "WEBP", lossless=args.webp_lossless, quality=args.jpeg_quality, **kw)
    elif ext in {".tif", ".tiff"}:
        im.save(path, "TIFF", compression="tiff_lzw", **kw)
    else:
        im.save(path)


def write_png(path: Path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def axis_spans(n, tile, overlap):
    if tile <= 0 or n <= tile:
        return [(0, n)]
    step = max(1, int(tile * (1 - overlap)))
    starts = list(range(0, n - tile + 1, step))
    if starts[-1] + tile < n:
        starts.append(n - tile)
    return [(s, s + tile) for s in starts]


class TextDetector:
    def __init__(self, langs, gpu, args):
        import easyocr  # heavy import, done once
        self.args = args
        log.info("Loading EasyOCR/CRAFT detector (gpu=%s)...", gpu)
        self.reader = easyocr.Reader(langs, gpu=gpu, verbose=False, recognizer=False)

    def _detect_once(self, img):
        a = self.args
        h_list, f_list = self.reader.detect(
            np.ascontiguousarray(img),
            min_size=a.min_size, text_threshold=a.confidence, low_text=a.low_text,
            link_threshold=a.link_threshold, canvas_size=a.canvas_size,
            mag_ratio=a.mag_ratio, add_margin=0.05,
        )
        polys = []
        for x0, x1, y0, y1 in h_list[0]:
            polys.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32))
        for q in f_list[0]:
            polys.append(np.array(q, np.float32).reshape(-1, 2))
        return polys

    def _variants(self, tile):
        names = [s.strip() for s in self.args.passes.split(",") if s.strip()]
        gray = None
        for n in names:
            if n == "orig":
                yield tile
                continue
            if gray is None:
                g = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
                gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
            if n == "clahe":
                yield np.repeat(gray[..., None], 3, axis=2)
            elif n == "invert":
                yield np.repeat((255 - gray)[..., None], 3, axis=2)

    def _detect_tile(self, tile):
        h, w = tile.shape[:2]
        polys = []
        for v in self._variants(tile):
            polys += self._detect_once(v)
            if self.args.rotations:
                for k in (1, 3):
                    for q in self._detect_once(np.rot90(v, k)):
                        # map rotated coordinates back to the unrotated tile
                        x, y = q[:, 0].copy(), q[:, 1].copy()
                        q = np.stack([w - y, x], 1) if k == 1 else np.stack([y, h - x], 1)
                        polys.append(q.astype(np.float32))
        return polys

    def detect(self, rgb):
        a = self.args
        H, W = rgb.shape[:2]
        regions = [(x0, y0, x1, y1)
                   for (y0, y1) in axis_spans(H, a.tile_size, a.tile_overlap)
                   for (x0, x1) in axis_spans(W, a.tile_size, a.tile_overlap)]
        if len(regions) > 1:
            regions.append((0, 0, W, H))  # global pass catches very large text
        polys = []
        for x0, y0, x1, y1 in regions:
            for q in self._detect_tile(rgb[y0:y1, x0:x1]):
                polys.append(q + np.array([x0, y0], np.float32))
        keep = []
        for q in polys:
            area = cv2.contourArea(q)
            if area < 4 or area > a.max_box_ratio * H * W:
                continue
            keep.append(q)
        return keep


# --------------------------------------------------------------------------- #
# Mask construction
# --------------------------------------------------------------------------- #
def build_mask(rgb, polys, args):
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = np.zeros((H, W), np.uint8)
    for q in polys:
        pi = np.round(q).astype(np.int32)
        if args.mask_mode == "box":
            cv2.fillPoly(mask, [pi], 255)
            continue
        x, y, w, h = cv2.boundingRect(pi)
        mg = 8
        x0, y0 = max(0, x - mg), max(0, y - mg)
        x1, y1 = min(W, x + w + mg), min(H, y + h + mg)
        crop = gray[y0:y1, x0:x1]
        pm = np.zeros(crop.shape, np.uint8)
        cv2.fillPoly(pm, [pi - np.array([x0, y0], np.int32)], 255)
        pm = cv2.dilate(pm, np.ones((3, 3), np.uint8))
        area = int((pm > 0).sum())
        if area == 0:
            continue
        k = int(min(w, h) * 0.6) | 1
        k = max(7, min(k, 151))
        bg = cv2.medianBlur(crop, k)               # local background estimate
        diff = cv2.absdiff(crop, bg)
        vals = diff[pm > 0].reshape(-1, 1)
        thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(thr, 6)
        txt = ((diff > thr) & (pm > 0)).astype(np.uint8) * 255
        edges = cv2.Canny(crop, 50, 150)
        txt |= cv2.bitwise_and(edges, pm)
        cov = (txt > 0).sum() / area
        if cov < 0.08 or cov > 0.85:
            # Segmentation unreliable (faint/transparent text, or huge strokes):
            # fall back to the full detected polygon.
            txt = pm
        else:
            ck = max(3, int(min(w, h) * 0.08) | 1)
            txt = cv2.morphologyEx(txt, cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8))
        mask[y0:y1, x0:x1] |= txt
    if args.padding > 0:
        k = 2 * args.padding + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask


def load_manual_mask(args, rel: Path, shape):
    if not args.mask_dir:
        return None
    for cand in (args.mask_dir / rel.parent / (rel.name + ".png"),
                 args.mask_dir / rel.parent / (rel.stem + ".png")):
        if cand.exists():
            m = np.array(Image.open(cand).convert("L"))
            if m.shape != shape:
                m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            return ((m > 127) * 255).astype(np.uint8)
    return None


# --------------------------------------------------------------------------- #
# Inpainting
# --------------------------------------------------------------------------- #
class Inpainter:
    def __init__(self, kind, device, args):
        self.kind, self.args, self.device = kind, args, device
        self.lama = None
        if kind == "lama":
            from simple_lama_inpainting import SimpleLama  # model loaded once
            log.info("Loading LaMa inpainting model (device=%s)...", device)
            try:
                self.lama = SimpleLama(device=__import__("torch").device(device))
            except TypeError:  # older package versions pick the device themselves
                log.warning("This simple-lama-inpainting version ignores --device")
                self.lama = SimpleLama()

    def _run(self, crop, m):
        if self.kind == "opencv":
            return cv2.inpaint(crop, m, 3, cv2.INPAINT_TELEA)
        out = self.lama(Image.fromarray(crop), Image.fromarray(m))
        out = np.array(out.convert("RGB"))
        if out.shape[:2] != crop.shape[:2]:
            out = cv2.resize(out, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
        return out

    def _run_scaled(self, crop, m, max_side):
        h, w = crop.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s >= 1.0:
            return self._run(crop, m)
        sw, sh = max(8, int(w * s)), max(8, int(h * s))
        small = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)
        ms = (cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8) * 255
        out = self._run(small, ms)
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)

    def _inpaint_crop(self, crop, m):
        side = self.args.max_inpaint_side
        try:
            return self._run_scaled(crop, m, side)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            import torch
            torch.cuda.empty_cache()
            log.warning("GPU out of memory; retrying crop at half resolution")
            try:
                return self._run_scaled(crop, m, max(256, side // 2))
            except RuntimeError:
                torch.cuda.empty_cache()
                log.warning("Retry failed; using OpenCV Telea for this crop (lower quality)")
                return cv2.inpaint(crop, m, 3, cv2.INPAINT_TELEA)

    def __call__(self, rgb, mask):
        if not mask.any():
            return rgb
        H, W = mask.shape
        gap = max(1, self.args.cluster_gap)
        grown = cv2.dilate(mask, np.ones((2 * gap + 1, 2 * gap + 1), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
        mf = mask.astype(np.float32) / 255.0
        soft = np.maximum(mf, cv2.GaussianBlur(mf, (0, 0), 1.2))[..., None]
        result = rgb.copy()
        for i in range(1, n):
            x, y, w, h = stats[i, :4]
            ctx = max(48, int(0.5 * max(w, h)))
            x0, y0 = max(0, x - ctx), max(0, y - ctx)
            x1, y1 = min(W, x + w + ctx), min(H, y + h + ctx)
            crop, mc = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
            if not mc.any():
                continue
            inp = self._inpaint_crop(np.ascontiguousarray(crop), np.ascontiguousarray(mc))
            a = soft[y0:y1, x0:x1]
            blended = crop.astype(np.float32) * (1 - a) + inp.astype(np.float32) * a
            result[y0:y1, x0:x1] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
        return result


# --------------------------------------------------------------------------- #
# Previews
# --------------------------------------------------------------------------- #
def make_preview(before, mask, after, max_h=700):
    overlay = before.copy()
    sel = mask > 0
    overlay[sel] = (0.45 * overlay[sel] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
    s = min(1.0, max_h / before.shape[0])
    panels = []
    for p in (before, overlay, after):
        if s < 1.0:
            p = cv2.resize(p, (int(p.shape[1] * s), int(p.shape[0] * s)), interpolation=cv2.INTER_AREA)
        panels.append(p)
    return np.hstack(panels)


# --------------------------------------------------------------------------- #
# Per-image processing
# --------------------------------------------------------------------------- #
def process_one(src: Path, rel: Path, out_dir: Path, args, detector, inpainter):
    dst = out_dir / rel
    if dst.resolve() == src.resolve():
        raise RuntimeError("output path equals input path; refusing to overwrite original")
    if args.skip_existing and dst.exists() and not args.detect_only:
        return "skipped", 0

    rgb, alpha, icc = load_image(src)
    H, W = rgb.shape[:2]

    manual = load_manual_mask(args, rel, (H, W))
    if manual is not None and args.manual_masks == "replace":
        mask, n_poly = manual, -1
    else:
        polys = detector.detect(rgb)
        mask = build_mask(rgb, polys, args)
        n_poly = len(polys)
        if manual is not None:
            mask = cv2.bitwise_or(mask, manual)

    ratio = (mask > 0).mean()
    if ratio > 0.4:
        log.warning("%s: mask covers %.0f%% of the image - likely false positives; "
                    "check the preview", rel, ratio * 100)

    if args.detect_only:
        after = rgb
    else:
        after = inpainter(rgb, mask)
        assert after.shape == rgb.shape  # dimensions are always preserved
        save_image(dst, after, alpha, icc, args)

    review = out_dir / "_review" / rel.parent
    if args.previews or args.detect_only:
        review.mkdir(parents=True, exist_ok=True)
        pv = make_preview(rgb, mask, after)
        Image.fromarray(pv).save(review / (rel.name + ".compare.jpg"), quality=90)
    if args.save_masks or args.detect_only:
        write_png(out_dir / "_masks" / rel.parent / (rel.name + ".png"), mask)
    return "ok", n_poly


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2
    out_dir = args.output or args.input.parent / (args.input.stem + "_cleaned"
                                                  if args.input.is_file()
                                                  else args.input.name + "_cleaned")
    if out_dir.resolve() == args.input.resolve():
        print("Output folder must differ from input folder (originals are never overwritten).",
              file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "run.log", encoding="utf-8")],
    )

    files, root = collect_images(args.input, args.recursive, out_dir)
    if not files:
        log.error("No supported images found in %s", args.input)
        return 1
    log.info("Found %d image(s); output -> %s", len(files), out_dir)

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU")
        device = "cpu"
    log.info("Using device: %s", device)

    try:
        detector = TextDetector([s.strip() for s in args.langs.split(",")], device == "cuda", args)
    except Exception:
        log.exception("Could not initialise the text detector (pip install easyocr)")
        return 2
    inpainter = None
    if not args.detect_only:
        try:
            inpainter = Inpainter(args.inpainter, device, args)
        except Exception:
            log.exception("Could not initialise the inpainter. Install simple-lama-inpainting "
                          "or use --inpainter opencv")
            return 2

    try:
        from tqdm import tqdm
        iterator = tqdm(files, unit="img")
    except ImportError:
        iterator = files

    ok = skipped = failed = 0
    t0 = time.time()
    for f in iterator:
        rel = f.relative_to(root)
        t = time.time()
        try:
            status, n = process_one(f, rel, out_dir, args, detector, inpainter)
            if status == "skipped":
                skipped += 1
                log.info("SKIP %s (exists)", rel)
            else:
                ok += 1
                log.info("OK   %s (%s regions, %.1fs)", rel, "manual" if n < 0 else n, time.time() - t)
        except Exception:
            failed += 1
            log.exception("FAIL %s", rel)  # continue with the next image
            if device == "cuda":
                torch.cuda.empty_cache()
    log.info("Done in %.1fs: %d ok, %d skipped, %d failed. Log: %s",
             time.time() - t0, ok, skipped, failed, out_dir / "run.log")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
