"""Mode 1 (automatic) and Mode 2 (manual boxes) for still images."""
import numpy as np
from PIL import Image

from .models import predict_mask


def compose(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    out = np.dstack([rgb[..., :3], alpha]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def auto_remove(img: Image.Image, quality: str) -> Image.Image:
    rgb = np.asarray(img.convert("RGB"))
    return compose(rgb, predict_mask(img, quality))


def boxes_mask(img: Image.Image, boxes, quality: str,
               context_pad: float = 0.15, clip_pad: float = 0.03) -> np.ndarray:
    """Segment the main subject inside each user box and merge the masks.

    Each box is cropped with extra context around it (helps the model), and
    the mask is clipped to the box plus a small margin, so hair/edges that
    poke out slightly survive but neighbouring subjects don't leak in.
    """
    w, h = img.size
    rgb_img = img.convert("RGB")
    full = np.zeros((h, w), dtype=np.uint8)

    for (x0, y0, x1, y1) in boxes:
        x0, x1 = sorted((int(x0), int(x1)))
        y0, y1 = sorted((int(y0), int(y1)))
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)

        cx0 = max(0, int(x0 - bw * context_pad)); cy0 = max(0, int(y0 - bh * context_pad))
        cx1 = min(w, int(x1 + bw * context_pad)); cy1 = min(h, int(y1 + bh * context_pad))
        if cx1 - cx0 < 8 or cy1 - cy0 < 8:
            continue

        crop_mask = predict_mask(rgb_img.crop((cx0, cy0, cx1, cy1)), quality)

        kx0 = max(cx0, int(x0 - bw * clip_pad)); ky0 = max(cy0, int(y0 - bh * clip_pad))
        kx1 = min(cx1, int(x1 + bw * clip_pad)); ky1 = min(cy1, int(y1 + bh * clip_pad))
        clipped = np.zeros_like(crop_mask)
        sl = (slice(ky0 - cy0, ky1 - cy0), slice(kx0 - cx0, kx1 - cx0))
        clipped[sl] = crop_mask[sl]

        region = full[cy0:cy1, cx0:cx1]
        np.maximum(region, clipped, out=region)

    return full


def boxes_remove(img: Image.Image, boxes, quality: str) -> Image.Image:
    rgb = np.asarray(img.convert("RGB"))
    return compose(rgb, boxes_mask(img, boxes, quality))
