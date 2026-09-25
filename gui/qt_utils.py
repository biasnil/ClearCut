from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap


def pil_to_qimage(img: Image.Image) -> QImage:
    img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    return QImage(data, img.width, img.height, 4 * img.width,
                  QImage.Format.Format_RGBA8888).copy()


def pil_to_pixmap(img: Image.Image) -> QPixmap:
    return QPixmap.fromImage(pil_to_qimage(img))


def checker_composite(img: Image.Image, max_w: int, max_h: int) -> QPixmap:
    """Scale an image to fit and draw it over a checkerboard so
    transparency is visible."""
    pm = pil_to_pixmap(img).scaled(max(1, max_w), max(1, max_h),
                                   Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
    out = QPixmap(pm.size())
    p = QPainter(out)
    size = 10
    light, dark = QColor(205, 205, 205), QColor(150, 150, 150)
    for y in range(0, pm.height(), size):
        for x in range(0, pm.width(), size):
            p.fillRect(x, y, size, size, light if (x // size + y // size) % 2 == 0 else dark)
    p.drawPixmap(0, 0, pm)
    p.end()
    return out
