"""Automatic subject detection for manual mode.

Finds every separate subject in an image so the user can click to keep or
remove each one, instead of drawing boxes.

1. If the image has a plain, solid background (logos, icons, product shots),
   the background colour is keyed out exactly. Fast, no AI, pixel-perfect.
2. Otherwise BiRefNet produces a foreground mask.

The foreground is then split into separate subjects. Pieces that sit close
together (e.g. a logo and its text) are grouped into one subject.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .chroma import KeySpec, key_custom
from .image_ops import boxes_mask
from .models import predict_mask


@dataclass
class Subject:
    mask: np.ndarray        # HxW uint8, soft alpha of this subject only
    bbox: tuple             # (x0, y0, x1, y1)
    area: int
    keep: bool = True
    source: str = "auto"    # "auto" or "box"


def solid_border_color(rgb: np.ndarray, max_dist: float = 10.0, min_share: float = 0.9):
    """Return the background colour if the image border is one plain colour."""
    h, w = rgb.shape[:2]
    b = max(1, min(4, h // 20, w // 20))
    px = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                         rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
    med = np.median(px, axis=0)
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    ref = cv2.cvtColor(med.reshape(1, 1, 3).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    dist = np.linalg.norm(lab.reshape(-1, 3) - ref.reshape(3), axis=1)
    if (dist < max_dist).mean() >= min_share:
        return tuple(int(v) for v in med)
    return None


def split_subjects(alpha: np.ndarray, min_area_frac: float = 0.002,
                   group_frac: float = 0.012) -> list:
    """Split a foreground mask into separate subjects."""
    h, w = alpha.shape
    hard = (alpha >= 128).astype(np.uint8)
    if not hard.any():
        return []

    # Grow pieces a little so nearby fragments count as one subject
    k = max(3, int(max(h, w) * group_frac) | 1)
    grouped = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)

    min_area = max(16, int(h * w * min_area_frac))
    subjects = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        region = labels[y:y + bh, x:x + bw] == i
        area = int(hard[y:y + bh, x:x + bw][region].sum())
        if area < min_area:
            continue
        mask = np.zeros_like(alpha)
        mask[y:y + bh, x:x + bw] = np.where(region, alpha[y:y + bh, x:x + bw], 0)
        subjects.append(Subject(mask, (int(x), int(y), int(x + bw), int(y + bh)), area))
    subjects.sort(key=lambda s: s.area, reverse=True)
    return subjects


def detect_subjects(img: Image.Image, quality: str, force_ai: bool = False):
    """Returns (subjects, method, matte_rgb). matte_rgb is the solid
    background colour when one was found (used to clean edges)."""
    rgb = np.ascontiguousarray(np.asarray(img.convert("RGB")))
    color = None if force_ai else solid_border_color(rgb)
    if color is not None:
        _, alpha = key_custom(rgb, KeySpec("custom", rgb=color, tol=22, edges_only=True))
        method = "solid background"
    else:
        alpha = predict_mask(img, quality)
        method = "AI"
    src_alpha = np.asarray(img.convert("RGBA"))[..., 3]
    alpha = np.minimum(alpha, src_alpha)
    return split_subjects(alpha), method, color


def subject_from_box(img: Image.Image, box, quality: str) -> Subject:
    mask = boxes_mask(img, [box], quality)
    ys, xs = np.nonzero(mask >= 128)
    if len(xs):
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    else:
        bbox = tuple(int(v) for v in box)
    return Subject(mask, bbox, int((mask >= 128).sum()), True, "box")


def combine(subjects) -> np.ndarray:
    kept = [s.mask for s in subjects if s.keep]
    if not kept:
        return None
    out = kept[0].copy()
    for m in kept[1:]:
        np.maximum(out, m, out=out)
    return out
