"""Track chosen subjects through an animation.

The user picks subjects on one reference frame. For every other frame:
  1. optical flow predicts where each subject moved (the previous mask is
     warped to the new frame),
  2. the subject is segmented again, only in a crop around that prediction,
  3. the new mask is limited to the area near the prediction, so other things
     (a blanket, a pillow...) can't join in, and stray pieces are dropped.
Tracking runs forwards and backwards from the reference frame.
"""
import cv2
import numpy as np

FLOW_MAX_SIDE = 480


def _gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _backward_flow(cur_gray, prev_gray):
    """Flow from the current frame to the previous one (for warping prev → cur)."""
    h, w = cur_gray.shape
    s = min(1.0, FLOW_MAX_SIDE / max(h, w))
    if s < 1.0:
        size = (max(8, int(w * s)), max(8, int(h * s)))
        a = cv2.resize(cur_gray, size, interpolation=cv2.INTER_AREA)
        b = cv2.resize(prev_gray, size, interpolation=cv2.INTER_AREA)
    else:
        a, b = cur_gray, prev_gray
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 4, 7, 1.5, 0)
    if s < 1.0:
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / s
    return flow


def _warp(mask, flow):
    h, w = mask.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(mask, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _bbox(mask, thresh=64, pad_frac=0.25):
    ys, xs = np.nonzero(mask >= thresh)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    px = max(16, int((x1 - x0) * pad_frac))
    py = max(16, int((y1 - y0) * pad_frac))
    return (max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py))


def _keep_overlapping(mask, guide):
    """Keep only the pieces of `mask` that overlap the predicted subject."""
    hard = (mask >= 128).astype(np.uint8)
    n, labels = cv2.connectedComponents(hard, connectivity=8)
    if n <= 1:
        return mask
    g = guide >= 128
    keep = np.zeros(n, bool)
    for i in np.unique(labels[g]):
        keep[i] = True
    keep[0] = False
    # soft edge pixels (alpha < 128) next to kept pieces stay too
    kept_hard = keep[labels]
    near = cv2.dilate(kept_hard.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return np.where(kept_hard | (near & ~hard.astype(bool)), mask, 0).astype(np.uint8)


def _limit(seg, warped, reach):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach * 2 + 1, reach * 2 + 1))
    allowed = cv2.dilate((warped >= 64).astype(np.uint8) * 255, k)
    allowed = cv2.GaussianBlur(allowed, (0, 0), max(1.0, reach / 3))
    # Pieces must overlap the subject's core, not just its fringe
    core = cv2.erode((warped >= 128).astype(np.uint8) * 255, k)
    if core.sum() < 0.2 * (warped >= 128).sum() * 255:
        core = warped
    return _keep_overlapping(np.minimum(seg, allowed), core)


class _ColourModel:
    """Colours of the subject vs its surroundings on the reference frame.
    Used to reject newly added areas that look like the background
    (a red blanket next to a white cat, for example)."""
    BINS = 16

    def __init__(self, rgb, seed):
        fg = seed >= 128
        ring = (cv2.dilate(fg.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & ~fg
        bg = ~fg if ring.sum() < 200 else ring | (seed < 16)
        q = self._quant(rgb)
        n = self.BINS ** 3
        self.fg = np.bincount(q[fg], minlength=n).astype(np.float32) + 1
        self.bg = np.bincount(q[bg], minlength=n).astype(np.float32) + 1
        self.fg /= self.fg.sum()
        self.bg /= self.bg.sum()

    def _quant(self, rgb):
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab) // (256 // self.BINS)
        return (lab[..., 0].astype(np.int32) * self.BINS + lab[..., 1]) * self.BINS + lab[..., 2]

    def fg_prob(self, rgb):
        q = self._quant(rgb)
        f, b = self.fg[q], self.bg[q]
        p = f / (f + b)
        return cv2.GaussianBlur(p, (0, 0), 2)


def _step(prev_mask, flow, segment, frame_idx, ref_area, colours=None, rgb=None):
    warped = _warp(prev_mask, flow)
    area = int((warped >= 128).sum())
    if area < 16:
        return np.zeros_like(prev_mask)          # subject has left the frame

    box = _bbox(warped)
    seg = segment(frame_idx, box)

    # How far the new mask may reach beyond the prediction
    reach = max(4, int(0.06 * np.sqrt(area)))
    if colours is not None:
        # Outside the subject's confident core, drop pixels whose colour
        # belongs to the surroundings rather than to the subject
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach * 2 + 1, reach * 2 + 1))
        core = cv2.erode((warped >= 128).astype(np.uint8), k) > 0
        p = colours.fg_prob(rgb)
        seg = np.where(core | (p >= 0.35), seg, 0).astype(np.uint8)
    new = _limit(seg, warped, reach)
    new_area = int((new >= 128).sum())

    # Anti-drift: if the subject suddenly grows, or is much bigger than it
    # was when picked, it's probably merging into something touching it.
    if new_area > 1.12 * area or new_area > 1.3 * ref_area:
        new = _limit(seg, warped, max(2, reach // 4))
        new_area = int((new >= 128).sum())
    if new_area < 0.35 * area:
        # Segmentation lost it (heavy blur, occlusion): trust the motion
        return warped.astype(np.uint8)
    return new


def track(frames_rgb, ref_index, seeds, segment, progress=None, is_cancelled=lambda: False):
    """frames_rgb: list of HxWx3 arrays. seeds: list of masks on the
    reference frame, one per subject. segment(frame_idx, box) -> full-size
    mask. Returns one combined keep-mask per frame."""
    n = len(frames_rgb)
    grays = [_gray(f) for f in frames_rgb]
    per_subject = [[None] * n for _ in seeds]
    ref_areas, colours = [], []
    for s, seed in enumerate(seeds):
        per_subject[s][ref_index] = seed.astype(np.uint8)
        ref_areas.append(max(16, int((seed >= 128).sum())))
        colours.append(_ColourModel(frames_rgb[ref_index], seed))

    order = [(i, i - 1) for i in range(ref_index + 1, n)] + \
            [(i, i + 1) for i in range(ref_index - 1, -1, -1)]
    flows = {}
    done, total = 0, max(1, len(order) * len(seeds))
    for cur, prev in order:
        if is_cancelled():
            from .animation import Cancelled
            raise Cancelled()
        key = (cur, prev)
        if key not in flows:
            flows = {key: _backward_flow(grays[cur], grays[prev])}   # keep memory low
        for s in range(len(seeds)):
            per_subject[s][cur] = _step(per_subject[s][prev], flows[key], segment, cur,
                                        ref_areas[s], colours[s], frames_rgb[cur])
            done += 1
            if progress:
                progress(done / total, f"tracking frame {done // max(1, len(seeds))}/{n - 1}")

    combined = []
    for i in range(n):
        m = per_subject[0][i].copy()
        for s in range(1, len(seeds)):
            np.maximum(m, per_subject[s][i], out=m)
        combined.append(m)
    return combined
