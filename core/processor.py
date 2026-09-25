"""Input collection and per-file routing between the three modes."""
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .animation import Cancelled, is_animated, load_animation, process_frames, save_animation
from .chroma import DEFAULT_GREEN, KeySpec, decontaminate, detect_key, key_custom, key_frame
from . import models
from .cleanup import inpaint, inpaint_frames, prepare_region, remove_specks
from .image_ops import auto_remove, boxes_mask, boxes_remove, compose

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".apng"}
ANIM_EXT = {"webp": ".webp", "apng": ".png", "gif": ".gif"}

# Keeps extracted ZIPs alive until the program exits
_temp_dirs = []


@dataclass
class Settings:
    out_dir: Path
    quality: str = "best"          # "best" | "fast"
    background: str = "auto"       # "auto" | "chroma" | "color" | "ai" | "none"
    anim_format: str = "webp"      # "webp" | "apng" | "gif"
    key_rgb: tuple = (0, 255, 0)   # used when background == "color"
    key_tol: int = 25
    key_edges_only: bool = True
    clean_specks: bool = True      # drop small leftover bits after removal
    wm_method: str = "ai"          # watermark fill: "ai" (LaMa) | "fast"


@dataclass
class Job:
    path: Path
    boxes: list = field(default_factory=list)   # Mode 2 boxes (stills only)
    display_name: str = ""
    mask: object = None        # Mode 2: final keep-mask from the subject picker
    matte_rgb: tuple = None    # solid background colour, used to clean edges
    wm_mask: object = None     # painted watermark/logo region to inpaint
    track: dict = None         # animations: {"ref", "seeds", "matte"} subjects to track


def collect_inputs(paths) -> list:
    """Expand files, folders (recursive) and ZIP archives into image files."""
    found = []
    for p in map(Path, paths):
        if p.is_dir():
            found += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        elif p.suffix.lower() == ".zip" and zipfile.is_zipfile(p):
            tmp = tempfile.TemporaryDirectory(prefix="clearcut_")
            _temp_dirs.append(tmp)
            with zipfile.ZipFile(p) as zf:
                zf.extractall(tmp.name)   # zipfile strips ../ and absolute paths
            found += sorted(f for f in Path(tmp.name).rglob("*")
                            if f.suffix.lower() in IMAGE_EXTS and "__MACOSX" not in f.parts)
        elif p.suffix.lower() in IMAGE_EXTS and p.is_file():
            found.append(p)
    seen, unique = set(), []
    for f in found:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def output_path(src: Path, out_dir: Path, ext: str, suffix: str = "_no_bg") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{src.stem}{suffix}"
    candidate = out_dir / f"{base}{ext}"
    n = 1
    while candidate.exists():
        candidate = out_dir / f"{base}_{n}{ext}"
        n += 1
    return candidate


def custom_spec(s: "Settings") -> KeySpec:
    return KeySpec("custom", rgb=tuple(s.key_rgb), tol=s.key_tol,
                   edges_only=s.key_edges_only)


def _resolve_key(frames_rgb, s: "Settings", log):
    """Decide colour key vs AI. Returns a KeySpec or None."""
    background = s.background
    if background == "color":
        r, g, b = s.key_rgb
        log(f"  removing colour #{r:02x}{g:02x}{b:02x} (tolerance {s.key_tol})")
        return custom_spec(s)
    if background == "ai":
        return None
    spec = detect_key(frames_rgb)
    if spec is None and background == "chroma":
        spec = detect_key(frames_rgb, threshold=0.25) or DEFAULT_GREEN
        log(f"  backdrop not clearly detected, keying {spec.color}")
    elif spec is not None:
        log(f"  {spec.color} screen detected, using chroma key")
    return spec


def process_file(job: Job, s: Settings, progress=lambda f, m: None,
                 is_cancelled=lambda: False, log=lambda m: None) -> Path:
    models.set_logger(log)
    try:
        return _process_file(job, s, progress, is_cancelled, log)
    finally:
        models.set_logger(None)


def _process_file(job, s, progress, is_cancelled, log) -> Path:
    src = job.path
    suffix = "_clean" if s.background == "none" else "_no_bg"

    # ---- Mode 3: animations ----
    if is_animated(src):
        progress(0.0, "loading frames")
        anim = load_animation(src)
        if job.wm_mask is not None:
            region = prepare_region(job.wm_mask, anim.frames[0].size)
            anim.frames = inpaint_frames(
                anim.frames, region, s.wm_method, log=log, is_cancelled=is_cancelled,
                progress=lambda f, m: progress(f * 0.4, m))
            log("  watermark removed")
        if s.background == "none":
            out = output_path(src, s.out_dir, ANIM_EXT[s.anim_format], suffix)
            save_animation(anim.frames, anim.durations, anim.loop, out, s.anim_format)
            return out
        if job.track and job.track.get("seeds"):
            frames = _track_animation(anim, job.track, s, progress, is_cancelled, log)
            progress(0.97, "saving")
            out = output_path(src, s.out_dir, ANIM_EXT[s.anim_format], suffix)
            save_animation(frames, anim.durations, anim.loop, out, s.anim_format)
            return out
        rgb_frames = [np.ascontiguousarray(np.asarray(f)[..., :3]) for f in anim.frames]
        spec = _resolve_key(rgb_frames, s, log)
        if spec is None:
            progress(0.0, "loading model")
        frames = process_frames(anim, key_spec=spec, quality=s.quality,
                                progress=lambda f, m: progress(f * 0.95, m),
                                is_cancelled=is_cancelled, clean_specks=s.clean_specks)
        progress(0.97, "saving")
        out = output_path(src, s.out_dir, ANIM_EXT[s.anim_format], suffix)
        save_animation(frames, anim.durations, anim.loop, out, s.anim_format)
        return out

    # ---- Still images ----
    with Image.open(src) as im:
        img = ImageOps.exif_transpose(im).convert("RGBA")
    if is_cancelled():
        raise Cancelled()
    if job.wm_mask is not None:
        progress(0.05, "removing watermark")
        img = inpaint(img, prepare_region(job.wm_mask, img.size), s.wm_method, log)
        log("  watermark removed")

    if s.background == "none":
        out = output_path(src, s.out_dir, ".png", suffix)
        img.save(out, "PNG")
        progress(1.0, "done")
        return out

    if job.mask is not None:                         # Mode 2: picked subjects
        progress(0.3, "applying selection")
        mask = job.mask
        if mask.shape != (img.height, img.width):
            mask = np.asarray(Image.fromarray(mask).resize(img.size, Image.BILINEAR))
        rgb = np.ascontiguousarray(np.asarray(img)[..., :3])
        if job.matte_rgb is not None:
            rgb = decontaminate(rgb, mask, job.matte_rgb)
        result = compose(rgb, mask)
    elif job.boxes:                                  # Mode 2 (boxes)
        progress(0.1, f"segmenting {len(job.boxes)} subject(s)")
        result = boxes_remove(img, job.boxes, s.quality)
    else:
        rgb = np.ascontiguousarray(np.asarray(img)[..., :3])
        spec = _resolve_key([rgb], s, log)
        if spec is not None:                         # chroma still
            progress(0.3, "keying")
            krgb, alpha = key_frame(rgb, spec)
            result = compose(krgb, alpha)
        else:                                        # Mode 1
            progress(0.1, "segmenting")
            result = auto_remove(img, s.quality)
        if s.clean_specks:
            arr = np.asarray(result).copy()
            arr[..., 3] = remove_specks(arr[..., 3])
            result = Image.fromarray(arr, "RGBA")

    # Keep any transparency the source already had
    src_alpha = np.asarray(img)[..., 3]
    arr = np.asarray(result).copy()
    arr[..., 3] = np.minimum(arr[..., 3], src_alpha)
    result = Image.fromarray(arr, "RGBA")

    out = output_path(src, s.out_dir, ".png")
    result.save(out, "PNG", optimize=False)
    progress(1.0, "done")
    return out


def _track_animation(anim, track_info, s, progress, is_cancelled, log):
    """Keep only the subjects picked on the reference frame, tracked through
    every frame."""
    from .tracking import track
    rgbs = [np.ascontiguousarray(np.asarray(f)[..., :3]) for f in anim.frames]
    matte = track_info.get("matte")
    ref = min(int(track_info.get("ref", 0)), len(rgbs) - 1)
    seeds = track_info["seeds"]
    log(f"  tracking {len(seeds)} subject(s) from frame {ref + 1}")

    def segment(i, box):
        x0, y0, x1, y1 = box
        full = np.zeros(rgbs[i].shape[:2], np.uint8)
        if matte is not None:      # plain background: colour key, no AI needed
            _, a = key_custom(np.ascontiguousarray(rgbs[i][y0:y1, x0:x1]),
                              KeySpec("custom", rgb=matte, tol=22, edges_only=False))
            full[y0:y1, x0:x1] = a
            return full
        return boxes_mask(anim.frames[i], [box], s.quality, clip_pad=0.0)

    masks = track(rgbs, ref, seeds, segment,
                  progress=lambda f, m: progress(0.05 + f * 0.9, m),
                  is_cancelled=is_cancelled)
    out = []
    for f, rgb, m in zip(anim.frames, rgbs, masks):
        m = np.minimum(m, np.asarray(f)[..., 3])
        if matte is not None:
            rgb = decontaminate(rgb, m, matte)
        out.append(compose(rgb, m))
    return out