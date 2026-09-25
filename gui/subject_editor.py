"""Mode 2 dialog: subjects are detected automatically and highlighted.
Click a subject to keep/remove it. Drag a box to add one it missed."""
import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QGraphicsRectItem, QGraphicsScene,
                               QGraphicsView, QHBoxLayout, QLabel, QPushButton, QSlider,
                               QVBoxLayout)

from core.subjects import combine, detect_subjects, subject_from_box

from .qt_utils import pil_to_pixmap, pil_to_qimage

KEEP_RGB = (0, 200, 255)      # outline of kept subjects
DROP_RGB = (255, 120, 60)     # outline of removed subjects
DIM_ALPHA = 150               # darkening of everything that will be removed


class _Task(QObject):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class _View(QGraphicsView):
    clicked = Signal(QPointF, bool)     # scene pos, right button
    boxed = Signal(QRectF)

    def __init__(self, scene, bounds: QRectF):
        super().__init__(scene)
        self.bounds = bounds
        self._start = None
        self._rect = None
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(40, 40, 40))
        self.setMouseTracking(True)

    def _clamp(self, p):
        b = self.bounds
        return QPointF(min(max(p.x(), b.left()), b.right() - 1),
                       min(max(p.y(), b.top()), b.bottom() - 1))

    def mousePressEvent(self, e):
        pos = self._clamp(self.mapToScene(e.position().toPoint()))
        if e.button() == Qt.MouseButton.RightButton:
            self.clicked.emit(pos, True)
        elif e.button() == Qt.MouseButton.LeftButton:
            self._start = pos
            self._rect = QGraphicsRectItem(QRectF(pos, pos))
            pen = QPen(QColor(*KEEP_RGB))
            pen.setCosmetic(True)
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            self._rect.setPen(pen)
            self.scene().addItem(self._rect)

    def mouseMoveEvent(self, e):
        if self._rect:
            p = self._clamp(self.mapToScene(e.position().toPoint()))
            self._rect.setRect(QRectF(self._start, p).normalized())

    def mouseReleaseEvent(self, e):
        if not self._rect or e.button() != Qt.MouseButton.LeftButton:
            return
        r = self._rect.rect()
        self.scene().removeItem(self._rect)
        self._rect = None
        # A tiny drag is a click
        view_size = self.mapFromScene(r).boundingRect()
        if view_size.width() < 8 and view_size.height() < 8:
            self.clicked.emit(self._start, False)
        else:
            self.boxed.emit(r)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.fitInView(self.bounds, Qt.AspectRatioMode.KeepAspectRatio)


class SubjectEditor(QDialog):
    def __init__(self, image, quality: str, state: dict = None, parent=None):
        """image: one PIL image, or a list of frames for an animation. For
        animations you pick subjects on one frame and they're tracked."""
        super().__init__(parent)
        self.setWindowTitle("Choose what to keep")
        self.resize(1000, 800)
        self.frames = [f.convert("RGBA") for f in image] if isinstance(image, list) \
            else [image.convert("RGBA")]
        self.ref = int(state.get("ref", 0)) if state else 0
        self.ref = min(self.ref, len(self.frames) - 1)
        self.image = self.frames[self.ref]
        self.quality = quality
        self.subjects, self.method, self.matte = [], "", None
        self._threads = []

        self.scene = QGraphicsScene(self)
        self.base = self.scene.addPixmap(pil_to_pixmap(self.image))
        self.bounds = self.base.boundingRect()
        self.scene.setSceneRect(self.bounds)
        self.overlay = self.scene.addPixmap(pil_to_pixmap(self.image))
        self.overlay.setVisible(False)

        self.view = _View(self.scene, self.bounds)
        self.view.clicked.connect(self.on_click)
        self.view.boxed.connect(self.on_box)

        self.info = QLabel()
        self.info.setWordWrap(True)
        hint = QLabel("Highlighted = kept, darkened = becomes transparent.  "
                      "Click a subject to keep/remove it · drag a box to add something "
                      "it missed · right-click to delete a subject."
                      + ("\nAnimation: pick a clear frame below. What you keep here is "
                         "tracked through every frame. If the subject is joined to "
                         "something else, drag a box around just the subject."
                         if len(self.frames) > 1 else ""))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")

        self.ai_btn = QPushButton("Re-detect with AI")
        self.ai_btn.setToolTip("Use the AI model even if the background is a plain colour")
        self.ai_btn.clicked.connect(lambda: self.detect(force_ai=True))
        keep_all = QPushButton("Keep all")
        keep_all.clicked.connect(lambda: self._set_all(True))
        drop_all = QPushButton("Remove all")
        drop_all.clicked.connect(lambda: self._set_all(False))
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                        QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        row = QHBoxLayout()
        for b in (self.ai_btn, keep_all, drop_all):
            row.addWidget(b)
        row.addStretch()
        row.addWidget(self.buttons)

        lay = QVBoxLayout(self)
        lay.addWidget(self.info)
        lay.addWidget(self.view, 1)
        if len(self.frames) > 1:
            self.frame_slider = QSlider(Qt.Orientation.Horizontal)
            self.frame_slider.setRange(0, len(self.frames) - 1)
            self.frame_slider.setValue(self.ref)
            self.frame_label = QLabel(f"{self.ref + 1}/{len(self.frames)}")
            self.frame_slider.valueChanged.connect(self._preview_frame)
            self.frame_slider.sliderReleased.connect(self._choose_frame)
            fr = QHBoxLayout()
            fr.addWidget(QLabel("Reference frame"))
            fr.addWidget(self.frame_slider, 1)
            fr.addWidget(self.frame_label)
            lay.addLayout(fr)
        lay.addWidget(hint)
        lay.addLayout(row)

        if state and state.get("subjects"):
            self.subjects = state["subjects"]
            self.method = state.get("method", "")
            self.matte = state.get("matte")
            self.redraw()
        else:
            self.detect()

    # ---------------- reference frame (animations)
    def _preview_frame(self, i):
        self.frame_label.setText(f"{i + 1}/{len(self.frames)}")
        if not self.frame_slider.isSliderDown():
            self._choose_frame()
        else:
            self.base.setPixmap(pil_to_pixmap(self.frames[i]))
            self.overlay.setVisible(False)

    def _choose_frame(self):
        i = self.frame_slider.value()
        if i == self.ref and self.subjects:
            self.base.setPixmap(pil_to_pixmap(self.image))
            self.overlay.setVisible(True)
            return
        self.ref = i
        self.image = self.frames[i]
        self.base.setPixmap(pil_to_pixmap(self.image))
        self.subjects = []
        self.overlay.setVisible(False)
        self.detect()

    # ---------------- background work
    def _run(self, fn, on_done, busy_text):
        self._set_busy(True, busy_text)
        th = QThread(self)
        task = _Task(fn)
        task.moveToThread(th)
        th.started.connect(task.run)
        task.done.connect(on_done)
        task.failed.connect(lambda msg: self.info.setText(f"Error: {msg}"))
        for sig in (task.done, task.failed):
            sig.connect(th.quit)
            sig.connect(lambda *_: self._set_busy(False))
        th.finished.connect(task.deleteLater)
        th.finished.connect(lambda: self._threads.remove((th, task)))
        self._threads.append((th, task))
        th.start()

    def _set_busy(self, busy, text=""):
        self.ai_btn.setEnabled(not busy)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(not busy)
        self.view.setEnabled(not busy)
        if hasattr(self, "frame_slider"):
            self.frame_slider.setEnabled(not busy)
        if busy:
            self.info.setText(text)
        else:
            self._update_info()

    def detect(self, force_ai=False):
        text = ("Analysing image with AI… (first use loads the model)" if force_ai
                else "Analysing image…")
        self._run(lambda: detect_subjects(self.image, self.quality, force_ai),
                  self._on_detected, text)

    def _on_detected(self, result):
        self.subjects, self.method, self.matte = result
        self.redraw()

    def on_box(self, r: QRectF):
        box = (r.left(), r.top(), r.right(), r.bottom())
        self._run(lambda: subject_from_box(self.image, box, self.quality),
                  self._on_box_done, "Segmenting the boxed area…")

    def _on_box_done(self, subject):
        if subject.area == 0:
            self.info.setText("Nothing found in that box, try a slightly bigger one.")
            return
        self.subjects.append(subject)
        self.redraw()

    # ---------------- interaction
    def _subject_at(self, pos: QPointF):
        x, y = int(pos.x()), int(pos.y())
        hits = [s for s in self.subjects if s.mask[y, x] >= 64]
        if not hits:   # allow clicking just inside the bounding box of thin shapes
            hits = [s for s in self.subjects
                    if s.bbox[0] <= x < s.bbox[2] and s.bbox[1] <= y < s.bbox[3]]
        return min(hits, key=lambda s: s.area) if hits else None

    def on_click(self, pos: QPointF, right: bool):
        s = self._subject_at(pos)
        if s is None:
            return
        if right:
            self.subjects.remove(s)
        else:
            s.keep = not s.keep
        self.redraw()

    def _set_all(self, keep):
        for s in self.subjects:
            s.keep = keep
        self.redraw()

    # ---------------- drawing
    def redraw(self):
        h, w = self.image.height, self.image.width
        keep = combine(self.subjects)
        keep = np.zeros((h, w), np.uint8) if keep is None else keep
        over = np.zeros((h, w, 4), np.uint8)
        over[..., 3] = (DIM_ALPHA * (1.0 - keep.astype(np.float32) / 255)).astype(np.uint8)

        thick = max(2, int(max(h, w) / 350))
        for s in self.subjects:
            hard = (s.mask >= 128).astype(np.uint8)
            contours, _ = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = KEEP_RGB if s.keep else DROP_RGB
            cv2.drawContours(over, contours, -1, (*color, 255), thick, cv2.LINE_AA)

        from PIL import Image
        from PySide6.QtGui import QPixmap
        self.overlay.setPixmap(QPixmap.fromImage(pil_to_qimage(Image.fromarray(over, "RGBA"))))
        self.overlay.setVisible(True)
        self._update_info()

    def _update_info(self):
        if not self.subjects:
            self.info.setText("No subjects found. Drag a box around what you want to keep.")
            return
        kept = sum(s.keep for s in self.subjects)
        how = f" ({self.method})" if self.method else ""
        self.info.setText(f"{len(self.subjects)} subject(s) found{how}, {kept} kept.")

    # ---------------- results
    def state(self) -> dict:
        return {"subjects": self.subjects, "method": self.method, "matte": self.matte,
                "ref": self.ref}

    def keep_mask(self):
        return combine(self.subjects)

    def done(self, r):
        for th, _ in list(self._threads):
            th.quit()
            th.wait(10000)
        super().done(r)

    def showEvent(self, e):
        super().showEvent(e)
        self.view.fitInView(self.bounds, Qt.AspectRatioMode.KeepAspectRatio)