"""Mode 3: animated GIF / WebP / APNG loading, processing and saving."""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

from .chroma import key_frame
from .cleanup import remove_specks
from .models import predict_mask


class Cancelled(Exception):
    pass


@dataclass
class Animation:
    frames: list = field(default_factory=list)      # RGBA PIL images, full frames
    durations: list = field(default_factory=list)   # ms per frame
    loop: int = 0                                   # 0 = forever


def is_animated(path) -> bool:
    try:
        with Image.open(path) as im:
            return getattr(im, "is_animated", False) and getattr(im, "n_frames", 1) > 1
    except Exception:
        return False


def load_animation(path) -> Animation:
    anim = Animation()
    with Image.open(path) as im:
        anim.loop = im.info.get("loop", 0)
        for frame in ImageSequence.Iterator(im):
            # Pillow composites GIF/APNG/WebP frames for us; convert gives a full frame
            anim.frames.append(frame.convert("RGBA"))
            d = frame.info.get("duration", 100) or 100
            anim.durations.append(max(20, int(d)))
    return anim


def process_frames(anim: Animation, *, key_spec=None, quality="best",
                   progress=None, is_cancelled=lambda: False,
                   clean_specks: bool = True) -> list:
    """Remove the background from every frame.

    key_spec set   -> chroma key (fast, no AI)
    key_spec None  -> BiRefNet on each *unique* frame (duplicate frames reuse
                      the result, which is common in GIFs with held poses)
    """
    out, cache = [], {}
    n = len(anim.frames)
    for i, frame in enumerate(anim.frames):
        if is_cancelled():
            raise Cancelled()
        rgba = np.asarray(frame)
        rgb = np.ascontiguousarray(rgba[..., :3])
        src_alpha = rgba[..., 3]

        if key_spec is not None:
            rgb, alpha = key_frame(rgb, key_spec)
        else:
            digest = hashlib.blake2b(rgba.tobytes(), digest_size=16).digest()
            if digest not in cache:
                cache[digest] = predict_mask(frame, quality)
            alpha = cache[digest]

        if clean_specks:
            alpha = remove_specks(alpha)

        # Respect pixels that were already transparent in the source
        alpha = np.minimum(alpha, src_alpha)
        out.append(Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA"))
        if progress:
            progress((i + 1) / n, f"frame {i + 1}/{n}")
    return out


def _to_gif_frame(frame: Image.Image) -> Image.Image:
    """GIF only has on/off transparency: threshold alpha and reserve
    palette index 255 for transparent pixels."""
    alpha = frame.getchannel("A")
    p = frame.convert("RGB").quantize(colors=255, method=Image.Quantize.MEDIANCUT,
                                      dither=Image.Dither.NONE)
    pal = p.getpalette()[:255 * 3]
    pal += [0, 0, 0] * (256 - len(pal) // 3)
    p.putpalette(pal)
    p.paste(255, mask=alpha.point(lambda a: 255 if a < 128 else 0))
    p.info["transparency"] = 255
    return p


def save_animation(frames, durations, loop, path: Path, fmt: str):
    fmt = fmt.lower()
    first, rest = frames[0], frames[1:]
    if fmt == "webp":
        first.save(path, "WEBP", save_all=True, append_images=rest,
                   duration=durations, loop=loop, quality=90, method=4,
                   allow_mixed=False)
    elif fmt == "apng":
        # blend=0 (SOURCE): each frame fully replaces the last, alpha included
        first.save(path, "PNG", save_all=True, append_images=rest,
                   duration=durations, loop=loop, disposal=1, blend=0,
                   default_image=False)
    elif fmt == "gif":
        gframes = [_to_gif_frame(f) for f in frames]
        gframes[0].save(path, "GIF", save_all=True, append_images=gframes[1:],
                        duration=durations, loop=loop, disposal=2,
                        transparency=255, optimize=False)
    else:
        raise ValueError(f"Unknown animation format: {fmt}")