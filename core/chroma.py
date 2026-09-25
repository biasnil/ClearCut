"""Green/blue-screen detection and keying (no AI needed)."""
from dataclasses import dataclass

import cv2
import numpy as np

# OpenCV hue scale is 0-179
KEY_RANGES = {"green": (35, 85), "blue": (95, 130)}


@dataclass
class KeySpec:
    color: str              # "green", "blue", or "custom"
    hue: int = 0            # green/blue: centre hue of the backdrop
    s_min: int = 0
    v_min: int = 0
    tol: int = 14           # green/blue: hue tolerance; custom: colour distance (1-100)
    rgb: tuple = (0, 0, 0)  # custom: the picked colour
    edges_only: bool = True # custom: only remove areas connected to the image border


DEFAULT_GREEN = KeySpec("green", 60, 60, 40)


def _border_pixels(rgb: np.ndarray, border: int = 6) -> np.ndarray:
    b = max(1, min(border, rgb.shape[0] // 4, rgb.shape[1] // 4))
    parts = [rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
             rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)]
    return np.ascontiguousarray(np.concatenate(parts))


def detect_key(frames_rgb, threshold: float = 0.6):
    """Sample the edges of a few frames. If most edge pixels share a green
    or blue hue, return a KeySpec describing that backdrop, else None."""
    if not frames_rgb:
        return None
    idx = np.linspace(0, len(frames_rgb) - 1, min(5, len(frames_rgb))).astype(int)
    px = np.concatenate([_border_pixels(frames_rgb[i]) for i in idx])
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0].astype(int), hsv[:, 1].astype(int), hsv[:, 2].astype(int)
    vivid = (s >= 70) & (v >= 50)

    for color, (lo, hi) in KEY_RANGES.items():
        hit = vivid & (h >= lo) & (h <= hi)
        if hit.mean() >= threshold:
            return KeySpec(
                color=color,
                hue=int(np.median(h[hit])),
                s_min=max(40, int(np.median(s[hit]) * 0.35)),
                v_min=max(30, int(np.median(v[hit]) * 0.30)),
            )
    return None


def key_custom(rgb: np.ndarray, spec: KeySpec):
    """Key out any picked colour (white, black, grey, pink...).

    Uses perceptual (Lab) colour distance with a soft ramp, so edges get
    partial transparency instead of jaggies. With edges_only, matching
    areas enclosed inside the subject (e.g. white eyes on a white
    background) are kept.
    """
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    target = np.array([[spec.rgb]], np.float32) / 255.0
    tgt = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)[0, 0]
    dist = np.linalg.norm(lab - tgt, axis=2)

    outer = float(max(1, spec.tol))
    inner = outer * 0.5
    alpha = np.clip((dist - inner) / (outer - inner), 0.0, 1.0)

    if spec.edges_only:
        region = (dist < outer).astype(np.uint8)
        n, labels = cv2.connectedComponents(region, connectivity=8)
        border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        touching_lut = np.zeros(n, bool)
        touching_lut[np.unique(border)] = True
        touching_lut[0] = False                      # label 0 = non-matching pixels
        alpha = np.where(touching_lut[labels], alpha, 1.0)

    alpha8 = (alpha * 255).astype(np.uint8)
    return decontaminate(rgb, alpha8, spec.rgb), alpha8


def decontaminate(rgb: np.ndarray, alpha8: np.ndarray, color) -> np.ndarray:
    """Remove a known backdrop colour that is blended into semi-transparent
    edge pixels, so cut-outs don't get a white/coloured outline."""
    a = alpha8.astype(np.float32)[..., None] / 255.0
    key = np.array(color, np.float32)
    rgb_f = rgb[..., :3].astype(np.float32)
    semi = (a > 0.02) & (a < 1.0)
    unmixed = (rgb_f - (1.0 - a) * key) / np.maximum(a, 1e-3)
    return np.where(semi, np.clip(unmixed, 0, 255), rgb_f).astype(np.uint8)


def key_frame(rgb: np.ndarray, spec: KeySpec, despill: bool = True):
    """Return (rgb, alpha) with the backdrop keyed out and edges cleaned."""
    rgb = np.ascontiguousarray(rgb[..., :3])
    if spec.color == "custom":
        return key_custom(rgb, spec)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lo = np.array([max(0, spec.hue - spec.tol), spec.s_min, spec.v_min], np.uint8)
    hi = np.array([min(179, spec.hue + spec.tol), 255, 255], np.uint8)
    key = cv2.inRange(hsv, lo, hi)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    key = cv2.morphologyEx(key, cv2.MORPH_OPEN, k3)   # speckles inside the subject
    key = cv2.morphologyEx(key, cv2.MORPH_CLOSE, k3)  # pinholes in the backdrop

    alpha = cv2.bitwise_not(key)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)        # feather edges

    out = rgb.copy()
    if despill:
        # Only correct colour in a band around the key edge, so genuine
        # green/blue inside the subject is left alone.
        band = cv2.dilate(key, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                          iterations=2) > 0
        band &= alpha > 0
        r, g, b = out[..., 0], out[..., 1], out[..., 2]
        if spec.color == "green":
            cap = np.maximum(r, b)
            g[band] = np.minimum(g[band], cap[band])
        else:
            cap = np.maximum(r, g)
            b[band] = np.minimum(b[band], cap[band])
    return out, alpha