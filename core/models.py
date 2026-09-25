"""BiRefNet model sessions (via rembg + onnxruntime). Loaded lazily and cached.

GPU is used when available. If the GPU can't actually run (missing CUDA /
cuDNN libraries, driver issues...), it falls back to CPU automatically
instead of failing, so the app works on every machine.
"""
import os
import sys
import threading
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

# Only MIT-licensed BiRefNet weights are used. Never fall back to rembg's
# default model (bria-rmbg), which is non-commercial.
MODEL_NAMES = {
    "best": "birefnet-general",       # highest quality, slower
    "fast": "birefnet-general-lite",  # much faster, slightly softer edges
}

_sessions = {}
_lock = threading.Lock()
_dlls_prepared = False
_logger = None
active = "CPU"   # what the loaded model is actually running on


def set_logger(fn):
    """Where GPU/CPU notices go (the log panel). None to disable."""
    global _logger
    _logger = fn


def _notify(msg):
    print(msg)
    if _logger:
        try:
            _logger(msg)
        except Exception:
            pass


def _providers():
    import onnxruntime as ort
    available = ort.get_available_providers()
    preferred = [
        "CUDAExecutionProvider",   # NVIDIA (onnxruntime-gpu)
        "DmlExecutionProvider",    # any DirectX 12 GPU on Windows (onnxruntime-directml)
        "CPUExecutionProvider",
    ]
    chosen = [p for p in preferred if p in available]
    return chosen or ["CPUExecutionProvider"]


def active_device() -> str:
    return active


def _prepare_cuda_dlls():
    """Make CUDA/cuDNN DLLs from the nvidia-* pip packages findable."""
    global _dlls_prepared
    if _dlls_prepared:
        return
    _dlls_prepared = True
    import onnxruntime as ort

    if sys.platform == "win32":
        # pip puts them in site-packages/nvidia/<lib>/bin, which Windows
        # doesn't search by default
        for base in {Path(p) for p in sys.path if p}:
            nv = base / "nvidia"
            if not nv.is_dir():
                continue
            for bin_dir in nv.glob("*/bin"):
                try:
                    os.add_dll_directory(str(bin_dir))
                except OSError:
                    pass
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

    if hasattr(ort, "preload_dlls"):      # onnxruntime >= 1.21
        try:
            ort.preload_dlls()
        except Exception as e:
            print(f"onnxruntime.preload_dlls: {e}")


def _short(e: Exception) -> str:
    text = str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__
    return text[-220:]


def is_downloaded(quality: str) -> bool:
    home = os.environ.get("U2NET_HOME", "")
    return any(Path(home).rglob(f"{MODEL_NAMES[quality]}.onnx")) if home else False


def get_session(quality: str):
    global active
    name = MODEL_NAMES[quality]
    with _lock:
        if name in _sessions:
            return _sessions[name]

        from rembg import new_session, remove
        providers = _providers()
        session = None

        if providers[0] != "CPUExecutionProvider":
            gpu = providers[0].replace("ExecutionProvider", "")
            try:
                if gpu == "CUDA":
                    _prepare_cuda_dlls()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    session = new_session(name, providers=providers)
                used = session.inner_session.get_providers()[0]
                if used == "CPUExecutionProvider":
                    raise RuntimeError(f"{gpu} libraries could not be loaded")
                # Test run: some failures (like missing cuDNN) only show up here
                remove(Image.new("RGB", (64, 64)), session=session, only_mask=True)
                active = gpu
                _notify(f"  model running on GPU ({gpu})")
            except Exception as e:
                session = None
                _notify(f"  GPU not usable, running on CPU instead ({_short(e)})")

        if session is None:
            session = new_session(name, providers=["CPUExecutionProvider"])
            active = "CPU"

        _sessions[name] = session
        return session


def predict_mask(img: Image.Image, quality: str) -> np.ndarray:
    """Return an HxW uint8 alpha mask (255 = subject) for an image."""
    from rembg import remove
    mask = remove(img.convert("RGB"), session=get_session(quality), only_mask=True)
    return np.asarray(mask.convert("L"), dtype=np.uint8)