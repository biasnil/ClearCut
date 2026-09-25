"""Mode 2 dialog: drag boxes around the subjects you want to keep."""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QGraphicsRectItem,
                               QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout)

from .qt_utils import pil_to_pixmap

BOX_PEN = QPen(QColor(0, 200, 255), 0)      # cosmetic pen: 1px at any zoom
BOX_PEN.setCosmetic(True)
BOX_PEN.setWidth(2)
SEL_PEN = QPen(QColor(255, 170, 0), 0)
SEL_PEN.setCosmetic(True)
SEL_PEN.setWidth(3)
FILL = QColor(0, 200, 255, 40)


class BoxView(QGraphicsView):
    def __init__(self, scene, bounds: QRectF, on_change):
        super().__init__(scene)
        self.bounds = bounds
        self.on_change = on_change
        self._start = None
        self._drawing = None
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(QColor(40, 40, 40))

    def _clamp(self, p: QPointF) -> QPointF:
        b = self.bounds
        return QPointF(min(max(p.x(), b.left()), b.right()),
                       min(max(p.y(), b.top()), b.bottom()))

    def _box_at(self, pos):
        for item in self.items(pos):
            if isinstance(item, QGraphicsRectItem):
                return item
        return None

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            item = self._box_at(e.position().toPoint())
            if item:
                self.scene().removeItem(item)
                self.on_change()
            return
        if e.button() == Qt.MouseButton.LeftButton:
            self._start = self._clamp(self.mapToScene(e.position().toPoint()))
            self._drawing = QGraphicsRectItem(QRectF(self._start, self._start))
            self._drawing.setPen(BOX_PEN)
            self._drawing.setBrush(FILL)
            self.scene().addItem(self._drawing)

    def mouseMoveEvent(self, e):
        if self._drawing:
            p = self._clamp(self.mapToScene(e.position().toPoint()))
            self._drawing.setRect(QRectF(self._start, p).normalized())

    def mouseReleaseEvent(self, e):
        if self._drawing and e.button() == Qt.MouseButton.LeftButton:
            r = self._drawing.rect()
            if r.width() < 8 or r.height() < 8:        # treat tiny drags as clicks
                self.scene().removeItem(self._drawing)
            self._drawing = None
            self.on_change()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.fitInView(self.bounds, Qt.AspectRatioMode.KeepAspectRatio)


class BoxEditor(QDialog):
    def __init__(self, image, boxes=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select subjects to keep")
        self.resize(1000, 720)

        self.scene = QGraphicsScene(self)
        pix = self.scene.addPixmap(pil_to_pixmap(image))
        bounds = pix.boundingRect()
        self.scene.setSceneRect(bounds)

        self.info = QLabel()
        self.view = BoxView(self.scene, bounds, self._update_info)

        for (x0, y0, x1, y1) in boxes or []:
            item = QGraphicsRectItem(QRectF(x0, y0, x1 - x0, y1 - y0))
            item.setPen(BOX_PEN)
            item.setBrush(FILL)
            self.scene.addItem(item)

        hint = QLabel("Drag to draw a box around each subject to keep. "
                      "Right-click a box to remove it. Everything outside the boxes "
                      "becomes transparent. No boxes = automatic mode.")
        hint.setWordWrap(True)

        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self._clear)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        bottom = QHBoxLayout()
        bottom.addWidget(self.info)
        bottom.addStretch()
        bottom.addWidget(clear_btn)
        bottom.addWidget(buttons)

        lay = QVBoxLayout(self)
        lay.addWidget(hint)
        lay.addWidget(self.view, 1)
        lay.addLayout(bottom)
        self._update_info()

    def _rect_items(self):
        return [i for i in self.scene.items() if isinstance(i, QGraphicsRectItem)]

    def _clear(self):
        for i in self._rect_items():
            self.scene.removeItem(i)
        self._update_info()

    def _update_info(self):
        n = len(self._rect_items())
        self.info.setText(f"{n} subject box(es)" if n else "No boxes (automatic mode)")

    def boxes(self):
        out = []
        for i in self._rect_items():
            r = i.rect()
            out.append((int(r.left()), int(r.top()), int(r.right()), int(r.bottom())))
        return out

    def showEvent(self, e):
        super().showEvent(e)
        self.view.fitInView(self.view.bounds, Qt.AspectRatioMode.KeepAspectRatio)
