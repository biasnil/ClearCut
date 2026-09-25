"""Paint over watermarks, logos or text. The painted area is filled in from
its surroundings on every frame (watermarks don't move, so painting once
covers the whole animation)."""
import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QDialogButtonBox,
                               QGraphicsRectItem, QGraphicsScene, QGraphicsView, QHBoxLayout,
                               QLabel, QPushButton, QRadioButton, QSlider, QVBoxLayout)

from core.cleanup import inpaint, lama_available, prepare_region

from .qt_utils import pil_to_qimage

PAINT_RGBA = (255, 40, 80, 130)


class _PaintView(QGraphicsView):
    def __init__(self, editor, scene, bounds):
        super().__init__(scene)
        self.ed = editor
        self.bounds = bounds
        self._last = None
        self._erase = False
        self._box_start = None
        self._box_item = None
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(40, 40, 40))
        self.setMouseTracking(True)

    def _pt(self, e):
        p = self.mapToScene(e.position().toPoint())
        return QPointF(min(max(p.x(), 0), self.bounds.width() - 1),
                       min(max(p.y(), 0), self.bounds.height() - 1))

    def mousePressEvent(self, e):
        if e.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return
        self._erase = e.button() == Qt.MouseButton.RightButton
        p = self._pt(e)
        if self.ed.box_mode():
            self._box_start = p
            self._box_item = QGraphicsRectItem(QRectF(p, p))
            pen = QPen(QColor(255, 255, 255))
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DashLine)
            self._box_item.setPen(pen)
            self.scene().addItem(self._box_item)
        else:
            self._last = p
            self.ed.stroke(p, p, self._erase)

    def mouseMoveEvent(self, e):
        p = self._pt(e)
        if self._box_item:
            self._box_item.setRect(QRectF(self._box_start, p).normalized())
        elif self._last is not None:
            self.ed.stroke(self._last, p, self._erase)
            self._last = p

    def mouseReleaseEvent(self, e):
        if self._box_item:
            r = self._box_item.rect()
            self.scene().removeItem(self._box_item)
            self._box_item = None
            self.ed.fill_box(r, self._erase)
        self._last = None
        self.ed.painting_done()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.fitInView(self.bounds, Qt.AspectRatioMode.KeepAspectRatio)


class WatermarkEditor(QDialog):
    def __init__(self, frames, mask=None, method="ai", parent=None):
        """frames: list of PIL images (a sample of the animation, or one still)."""
        super().__init__(parent)
        self.setWindowTitle("Remove watermarks, logos and text")
        self.resize(900, 780)
        self.frames = [f.convert("RGBA") for f in frames]
        w, h = self.frames[0].size
        self.mask = np.zeros((h, w), np.uint8) if mask is None else mask.copy()
        self.index = 0
        # Preview with the AI only if it's already downloaded (no surprise downloads)
        self.method = "ai" if method == "ai" and lama_available() else "fast"
        self._preview_cache = {}

        self.scene = QGraphicsScene(self)
        self.base = self.scene.addPixmap(QPixmap())
        self.overlay = self.scene.addPixmap(QPixmap())
        self.bounds = QRectF(0, 0, w, h)
        self.scene.setSceneRect(self.bounds)
        self.view = _PaintView(self, self.scene, self.bounds)

        # tools
        self.mode_group = QButtonGroup(self)
        self.brush_rb = QRadioButton("Brush")
        self.box_rb = QRadioButton("Box")
        self.brush_rb.setChecked(True)
        for rb in (self.brush_rb, self.box_rb):
            self.mode_group.addButton(rb)
        self.size = QSlider(Qt.Orientation.Horizontal)
        self.size.setRange(2, max(10, int(max(w, h) / 6)))
        self.size.setValue(max(6, int(max(w, h) / 40)))
        self.size.setFixedWidth(160)
        self.show_result = QCheckBox("Show result" + (" (fast preview)" if
                                     self.method != method else ""))
        self.show_result.toggled.connect(self.refresh)
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear)

        tools = QHBoxLayout()
        tools.addWidget(self.brush_rb)
        tools.addWidget(self.box_rb)
        tools.addWidget(QLabel("  Size"))
        tools.addWidget(self.size)
        tools.addStretch()
        tools.addWidget(self.show_result)
        tools.addWidget(clear)

        # frame scrubber (animations)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, len(self.frames) - 1)
        self.frame_slider.valueChanged.connect(self._set_frame)
        self.frame_label = QLabel()
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame"))
        frame_row.addWidget(self.frame_slider, 1)
        frame_row.addWidget(self.frame_label)
        anim = len(self.frames) > 1

        hint = QLabel("Paint over the watermark with the left button, right button to erase. "
                      "Box mode: drag a rectangle. It's filled in from the surroundings on "
                      + ("every frame. Scrub the frames to check it's covered throughout."
                         if anim else "the image."))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addLayout(tools)
        lay.addWidget(self.view, 1)
        if anim:
            lay.addLayout(frame_row)
        lay.addWidget(hint)
        lay.addWidget(buttons)
        self._set_frame(0)

    # ---------------- painting
    def box_mode(self):
        return self.box_rb.isChecked()

    def stroke(self, a: QPointF, b: QPointF, erase: bool):
        cv2.line(self.mask, (int(a.x()), int(a.y())), (int(b.x()), int(b.y())),
                 0 if erase else 255, self.size.value(), cv2.LINE_AA)
        if self.show_result.isChecked():
            self.show_result.setChecked(False)   # triggers refresh
        else:
            self._draw_overlay()

    def fill_box(self, r: QRectF, erase: bool):
        cv2.rectangle(self.mask, (int(r.left()), int(r.top())),
                      (int(r.right()), int(r.bottom())), 0 if erase else 255, -1)
        self.refresh()

    def painting_done(self):
        pass

    def _clear(self):
        self.mask[:] = 0
        self.refresh()

    # ---------------- display
    def _set_frame(self, i):
        self.index = i
        self.frame_label.setText(f"{i + 1}/{len(self.frames)}")
        self.refresh()

    def refresh(self):
        frame = self.frames[self.index]
        if self.show_result.isChecked() and self.mask.any():
            key = (self.index, self.mask.tobytes().__hash__())
            if key not in self._preview_cache:
                from PySide6.QtWidgets import QApplication
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    self._preview_cache = {key: inpaint(
                        frame, prepare_region(self.mask, frame.size), self.method)}
                finally:
                    QApplication.restoreOverrideCursor()
            shown = self._preview_cache[key]
            self.base.setPixmap(QPixmap.fromImage(pil_to_qimage(shown)))
            self.overlay.setVisible(False)
        else:
            self.base.setPixmap(QPixmap.fromImage(pil_to_qimage(frame)))
            self._draw_overlay()

    def _draw_overlay(self):
        h, w = self.mask.shape
        over = np.zeros((h, w, 4), np.uint8)
        over[self.mask > 0] = PAINT_RGBA
        self.overlay.setPixmap(QPixmap.fromImage(pil_to_qimage(Image.fromarray(over, "RGBA"))))
        self.overlay.setVisible(True)

    def result_mask(self):
        return self.mask.copy() if self.mask.any() else None

    def showEvent(self, e):
        super().showEvent(e)
        self.view.fitInView(self.bounds, Qt.AspectRatioMode.KeepAspectRatio)