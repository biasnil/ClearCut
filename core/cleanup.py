"""Clean-up tools: removing leftover specks from masks, and filling in
watermarks / logos / text overlays.

Watermark fill methods (best to simplest):
  "ai"   - LaMa inpainting model (Apache 2.0, ~200 MB, downloaded once).
           Rebuilds texture and patterns so the area really looks untouched.
  "fast" - OpenCV frequency-selective reconstruction (opencv-contrib), no AI.
           Falls back to OpenCV Telea if opencv-contrib isn't installed.

For animations, fills are also smoothed across neighbouring frames so the
patched area doesn't flicker, and film grain / GIF dither noise is matched so
the patch isn't suspiciously smooth.
"""
import os
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

LAMA_URLS = [
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "inpainting_lama/inpainting_lama_2025jan.onnx",
]
LAMA_SIZE = 512

_lama = None
_lama_error = None     # remembered so a failed download isn't retried every frame
_lama_lock = threading.Lock()


# ------------------------------------------------------------------ specks
def remove_specks(alpha: np.ndarray, min_frac: float = 0.03) -> np.ndarray:
    """Drop small floating bits that aren't part of the main subject(s)."""
    solid = (alpha >= 32).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if n <= 2:
        return alpha
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, bool)
    keep[1:] = areas >= areas.max() * min_frac
    return np.where(keep[labels], alpha, 0).astype(np.uint8)


# ------------------------------------------------------------------ regions
def prepare_region(region: np.ndarray, size) -> np.ndarray:
    """Resize a painted region to the image size and grow it slightly so
    anti-aliased text edges and drop shadows are covered too."""
    w, h = size
    if region.shape != (h, w):
        region = np.asarray(Image.fromarray(region).resize((w, h), Image.NEAREST))
    grow = max(2, int(max(w, h) / 250))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1))
    return cv2.dilate((region > 0).astype(np.uint8) * 255, k)


def _patches(region: np.ndarray):
    """Split the region into separate patches (e.g. a logo in each corner),
    each with a crop box that includes surrounding context."""
    h, w = region.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats((region > 0).astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        side = int(max(bw, bh) * 1.6 + 48)
        cw, ch = min(w, max(side, bw + 32)), min(h, max(side, bh + 32))
        cx0 = int(np.clip(x + bw / 2 - cw / 2, 0, w - cw))
        cy0 = int(np.clip(y + bh / 2 - ch / 2, 0, h - ch))
        box = (cx0, cy0, cx0 + cw, cy0 + ch)
        hole = (labels[cy0:cy0 + ch, cx0:cx0 + cw] == i).astype(np.uint8) * 255
        out.append((box, hole))
    return out


# ------------------------------------------------------------------ fill methods
def _clamp_colours(fill: np.ndarray, rgb: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Keep the fill's colours within the range found around the hole, which
    removes the green/purple fringes classic inpainting can invent."""
    ring = _ring(hole, max(4, min(hole.shape) // 6))
    if not ring.any():
        return fill
    lab_src = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
    lab = cv2.cvtColor(fill, cv2.COLOR_RGB2Lab).astype(np.float32)
    for ch in (1, 2):
        lo, hi = np.percentile(lab_src[..., ch][ring], [2, 98])
        lab[..., ch] = np.clip(lab[..., ch], lo, hi)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2RGB)


def _fill_fast(rgb: np.ndarray, hole: np.ndarray) -> np.ndarray:
    if hasattr(cv2, "xphoto"):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dst = np.zeros_like(bgr)
        cv2.xphoto.inpaint(bgr, cv2.bitwise_not(hole), dst, cv2.xphoto.INPAINT_FSR_FAST)
        return _clamp_colours(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), rgb, hole)
    radius = max(3, int(max(rgb.shape[:2]) / 40))
    return cv2.inpaint(rgb, hole, radius, cv2.INPAINT_TELEA)


def lama_path() -> Path:
    home = Path(os.environ.get("U2NET_HOME", Path.home() / ".clearcut"))
    return home / "lama" / "lama_fp32.onnx"


def lama_available() -> bool:
    return lama_path().exists()


def _download_lama(log):
    path = lama_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    last_err = None
    log("  downloading watermark AI model (~200 MB, one time only)…")
    for url in LAMA_URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if tmp.stat().st_size < 10_000_000:
                raise IOError("download incomplete")
            tmp.replace(path)
            return
        except Exception as e:
            last_err = e
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"couldn't download the watermark AI model ({last_err})")


def _get_lama(log):
    global _lama, _lama_error
    with _lama_lock:
        if _lama is not None:
            return _lama
        if _lama_error is not None:
            raise _lama_error
        if not lama_available():
            try:
                _download_lama(log)
            except Exception as e:
                _lama_error = e
                raise
        import onnxruntime as ort
        from . import models
        providers = models._providers()
        if providers[0] == "CUDAExecutionProvider":
            models._prepare_cuda_dlls()
        try:
            sess = ort.InferenceSession(str(lama_path()), providers=providers)
        except Exception:
            sess = ort.InferenceSession(str(lama_path()), providers=["CPUExecutionProvider"])
        ins = sess.get_inputs()
        img_name = next((i.name for i in ins if i.shape[1] == 3), ins[0].name)
        mask_name = next((i.name for i in ins if i.name != img_name), ins[-1].name)
        _lama = (sess, img_name, mask_name)
        return _lama


def _run_lama(rgb, hole, log):
    sess, img_name, mask_name = _get_lama(log)
    h, w = hole.shape
    img = cv2.resize(rgb, (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_AREA
                     if max(h, w) > LAMA_SIZE else cv2.INTER_CUBIC)
    m = cv2.resize(hole, (LAMA_SIZE, LAMA_SIZE), interpolation=cv2.INTER_NEAREST)
    m = (m > 0).astype(np.float32)
    img = img.astype(np.float32) / 255.0 * (1.0 - m[..., None])
    feed = {img_name: img.transpose(2, 0, 1)[None], mask_name: m[None, None]}
    out = sess.run(None, feed)[0][0].transpose(1, 2, 0)
    if out.max() <= 1.5:          # some exports return 0..1
        out = out * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)


_warned = False


def _fill(rgb, hole, method, log):
    global _warned
    if method == "ai":
        try:
            return _run_lama(rgb, hole, log)
        except Exception as e:
            if not _warned:
                _warned = True
                log(f"  watermark AI unavailable, using fast fill ({e})")
    return _fill_fast(rgb, hole)


# ------------------------------------------------------------------ blending
def _ring(hole, width):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width * 2 + 1, width * 2 + 1))
    return (cv2.dilate(hole, k) > 0) & (hole == 0)


def _grain_std(rgb, ring):
    """How noisy the surroundings are (film grain / GIF dither)."""
    if not ring.any():
        return 0.0
    f = rgb.astype(np.float32)
    detail = f - cv2.GaussianBlur(f, (0, 0), 1.2)
    return float(detail[ring].std())


def _paste(orig, fill, hole, grain, rng):
    soft = cv2.GaussianBlur(cv2.dilate(hole, np.ones((3, 3), np.uint8)), (0, 0), 1.2)
    soft = np.maximum(soft, hole).astype(np.float32)[..., None] / 255.0
    fill = fill.astype(np.float32)
    if grain > 1.0:
        fill += rng.normal(0, grain * 0.8, fill.shape[:2])[..., None]
    out = orig.astype(np.float32) * (1 - soft) + fill * soft
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ public API
def inpaint_frames(frames, region, method="ai", progress=None, log=lambda m: None,
                   is_cancelled=lambda: False):
    """Remove the painted region from a list of PIL frames. `region` should be
    prepared with prepare_region(). Returns new RGBA frames."""
    arrs = [np.array(f.convert("RGBA")) for f in frames]
    if not region.any():
        return [Image.fromarray(a, "RGBA") for a in arrs]
    rng = np.random.default_rng(7)
    patches = _patches(region)
    total = len(patches) * len(arrs)
    done = 0

    for (x0, y0, x1, y1), hole in patches:
        crops = [np.ascontiguousarray(a[y0:y1, x0:x1, :3]) for a in arrs]
        fills, fill_cache = [], {}
        for c in crops:
            if is_cancelled():
                from .animation import Cancelled
                raise Cancelled()
            key = hash(c.tobytes())
            if key not in fill_cache:              # repeated frames: fill once
                fill_cache[key] = _fill(c, hole, method, log)
            fills.append(fill_cache[key])
            done += 1
            if progress:
                progress(done / total, f"removing watermark {done}/{total}")

        # Temporal smoothing: average a frame's fill with its neighbours'
        # fills, weighted by how similar the surroundings are. Static areas
        # stop flickering, moving areas keep their own fill.
        if len(crops) > 1:
            ring = _ring(hole, max(4, hole.shape[0] // 12))
            smoothed = []
            for i in range(len(crops)):
                acc = fills[i].astype(np.float32)
                wsum = 1.0
                for j in range(max(0, i - 2), min(len(crops), i + 3)):
                    if j == i:
                        continue
                    diff = np.abs(crops[i][ring].astype(np.float32) -
                                  crops[j][ring].astype(np.float32)).mean()
                    wgt = float(np.exp(-diff / 6.0))
                    acc += fills[j].astype(np.float32) * wgt
                    wsum += wgt
                smoothed.append((acc / wsum).astype(np.uint8))
            fills = smoothed

        grain_ring = _ring(hole, 6)
        for a, c, f in zip(arrs, crops, fills):
            a[y0:y1, x0:x1, :3] = _paste(c, f, hole, _grain_std(c, grain_ring), rng)

    return [Image.fromarray(a, "RGBA") for a in arrs]


def inpaint(img: Image.Image, region: np.ndarray, method="ai", log=lambda m: None) -> Image.Image:
    return inpaint_frames([img], region, method, log=log)[0]