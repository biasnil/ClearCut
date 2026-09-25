"""ClearCut main window."""
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import QObject, QPoint, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QCursor
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QColorDialog,
                               QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QRadioButton, QSlider, QSplitter, QVBoxLayout, QWidget)

from core.animation import Cancelled, is_animated, load_animation
from core.chroma import key_frame
from core.subjects import combine
from core.processor import (IMAGE_EXTS, Job, Settings, collect_inputs, custom_spec,
                            process_file)

from .subject_editor import SubjectEditor
from .watermark_editor import WatermarkEditor
from .qt_utils import checker_composite

PATH_ROLE = Qt.ItemDataRole.UserRole
OUTPUT_ROLE = Qt.ItemDataRole.UserRole + 2
ANIM_ROLE = Qt.ItemDataRole.UserRole + 3


# ---------------------------------------------------------------- worker
class Worker(QObject):
    progress = Signal(int)          # 0..1000
    status = Signal(str)
    log = Signal(str)
    file_done = Signal(str, str)    # source, output
    finished = Signal()

    def __init__(self, jobs, settings):
        super().__init__()
        self.jobs, self.settings = jobs, settings
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        n = len(self.jobs)
        ok = 0
        for i, job in enumerate(self.jobs):
            if self._cancel:
                break
            name = job.display_name or job.path.name

            def cb(frac, msg, i=i, name=name):
                self.progress.emit(int((i + frac) / n * 1000))
                self.status.emit(f"[{i + 1}/{n}] {name}: {msg}")

            self.log.emit(f"{name}")
            try:
                out = process_file(job, self.settings, cb, lambda: self._cancel, self.log.emit)
                ok += 1
                self.file_done.emit(str(job.path), str(out))
                self.log.emit(f"  ✓ saved {out.name}")
            except Cancelled:
                self.log.emit("  cancelled")
                break
            except Exception as e:
                self.log.emit(f"  ✗ {type(e).__name__}: {e}")
                print(traceback.format_exc())
        self.progress.emit(1000 if not self._cancel else 0)
        self.status.emit(f"Finished: {ok}/{n} file(s) processed"
                         + (" (cancelled)" if self._cancel else ""))
        self.finished.emit()


# ---------------------------------------------------------------- preview label
class ClickLabel(QLabel):
    clicked = Signal(QPoint)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(e.position().toPoint())
        super().mousePressEvent(e)


# ---------------------------------------------------------------- file list
class DropList(QListWidget):
    def __init__(self, on_drop):
        super().__init__()
        self.on_drop = on_drop
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.on_drop(paths)
        e.acceptProposedAction()


# ---------------------------------------------------------------- window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ClearCut — Offline Background Remover")
        self.resize(1150, 720)
        self.thread = None
        self.worker = None

        # ---- left: file list
        self.list = DropList(self.add_paths)
        self.list.currentItemChanged.connect(lambda *_: self.update_preview())
        self.list.itemDoubleClicked.connect(lambda *_: self.edit_subjects())

        add_files = QPushButton("Add files…")
        add_files.clicked.connect(self.pick_files)
        add_folder = QPushButton("Add folder…")
        add_folder.clicked.connect(self.pick_folder)
        remove = QPushButton("Remove")
        remove.clicked.connect(self.remove_selected)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear_all)

        btns = QHBoxLayout()
        for b in (add_files, add_folder, remove, clear):
            btns.addWidget(b)

        drop_hint = QLabel("Drop images, GIFs, folders or ZIPs here")
        drop_hint.setStyleSheet("color: gray;")
        left = QVBoxLayout()
        left.addWidget(drop_hint)
        left.addWidget(self.list, 1)
        left.addLayout(btns)
        left_w = QWidget()
        left_w.setLayout(left)

        # ---- right: preview + settings
        self.before = ClickLabel("Original")
        self.before.clicked.connect(self.on_preview_click)
        self.after = QLabel("Result")
        self._preview_img = None       # PIL image shown in the Original pane
        self._preview_pm = None        # its scaled pixmap size
        self._picking = False
        self.live_timer = QTimer(self)
        self.live_timer.setSingleShot(True)
        self.live_timer.setInterval(120)
        self.live_timer.timeout.connect(self.update_live_preview)
        for lbl in (self.before, self.after):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumSize(220, 220)
            lbl.setStyleSheet("background:#2b2b2b; color:#aaa; border-radius:6px;")
        prev = QHBoxLayout()
        prev.addWidget(self.before, 1)
        prev.addWidget(self.after, 1)

        self.selections = {}   # path -> SubjectEditor state (manual mode)
        self.subjects_btn = QPushButton("Choose what to keep… (auto-detects subjects)")
        self.subjects_btn.clicked.connect(self.edit_subjects)
        self.watermarks = {}   # path -> painted watermark region
        self.wm_btn = QPushButton("Remove watermark / logo / text…")
        self.wm_btn.clicked.connect(self.edit_watermark)

        # settings
        self.quality = QComboBox()
        self.quality.addItem("Best quality (BiRefNet)", "best")
        self.quality.addItem("Fast (BiRefNet Lite)", "fast")

        self.background = QComboBox()
        self.background.addItem("Auto-detect (green/blue screen → key, else AI)", "auto")
        self.background.addItem("Green / blue screen", "chroma")
        self.background.addItem("Pick a colour…", "color")
        self.background.addItem("AI only", "ai")
        self.background.addItem("Keep background (clean-up only)", "none")
        self.background.currentIndexChanged.connect(self.on_background_changed)

        # ---- colour picker controls (shown only in "Pick a colour" mode)
        self.key_color = QColor(255, 255, 255)
        self.swatch = QPushButton()
        self.swatch.setFixedSize(46, 26)
        self.swatch.setToolTip("Choose colour")
        self.swatch.clicked.connect(self.choose_color)
        self.hex_label = QLabel()
        self.eyedrop = QPushButton("Pick from image")
        self.eyedrop.setCheckable(True)
        self.eyedrop.setToolTip("Then click the background in the Original preview")
        self.eyedrop.toggled.connect(self.set_picking)
        self.tol = QSlider(Qt.Orientation.Horizontal)
        self.tol.setRange(1, 100)
        self.tol.setValue(25)
        self.tol_label = QLabel("25")
        self.tol_label.setFixedWidth(28)
        self.tol.valueChanged.connect(lambda v: (self.tol_label.setText(str(v)),
                                                 self.live_timer.start()))
        self.edges_only = QCheckBox("Keep matching areas inside the subject")
        self.edges_only.setToolTip("Only removes the colour where it connects to the image "
                                   "edges, e.g. keeps white eyes on a white background")
        self.edges_only.setChecked(True)
        self.edges_only.toggled.connect(lambda *_: self.live_timer.start())

        c_row1 = QHBoxLayout()
        c_row1.addWidget(self.swatch)
        c_row1.addWidget(self.hex_label)
        c_row1.addWidget(self.eyedrop)
        c_row1.addStretch()
        c_row2 = QHBoxLayout()
        c_row2.addWidget(QLabel("Tolerance"))
        c_row2.addWidget(self.tol, 1)
        c_row2.addWidget(self.tol_label)
        color_lay = QVBoxLayout()
        color_lay.setContentsMargins(0, 0, 0, 0)
        color_lay.addLayout(c_row1)
        color_lay.addLayout(c_row2)
        color_lay.addWidget(self.edges_only)
        self.color_box = QWidget()
        self.color_box.setLayout(color_lay)
        self._set_key_color(self.key_color)

        self.anim_group = QButtonGroup(self)
        anim_row = QHBoxLayout()
        for label, val, tip in [("WebP", "webp", "Smooth edges, small files (recommended)"),
                                ("APNG", "apng", "Smooth edges, larger files"),
                                ("GIF", "gif", "Max compatibility, hard edges")]:
            rb = QRadioButton(label)
            rb.setToolTip(tip)
            rb.setProperty("fmt", val)
            self.anim_group.addButton(rb)
            anim_row.addWidget(rb)
            if val == "webp":
                rb.setChecked(True)
        anim_row.addStretch()

        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("Default: a 'no_bg' folder next to the first file")
        browse = QPushButton("…")
        browse.setFixedWidth(32)
        browse.clicked.connect(self.pick_out_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_dir)
        out_row.addWidget(browse)

        form = QFormLayout()
        form.addRow("Model:", self.quality)
        form.addRow("Background:", self.background)
        form.addRow("", self.color_box)
        self.specks = QCheckBox("Remove leftover specks")
        self.specks.setToolTip("Drops small floating bits of background that the AI or "
                               "colour key left behind")
        self.specks.setChecked(True)
        form.addRow("Clean-up:", self.specks)
        self.wm_method = QComboBox()
        self.wm_method.addItem("Best (AI, rebuilds texture)", "ai")
        self.wm_method.addItem("Fast (no AI)", "fast")
        form.addRow("Watermark fill:", self.wm_method)
        form.addRow("Animation output:", anim_row)
        form.addRow("Output folder:", out_row)
        settings_box = QGroupBox("Settings")
        settings_box.setLayout(form)

        self.start_btn = QPushButton("Remove backgrounds")
        self.start_btn.setMinimumHeight(38)
        self.start_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        run_row = QHBoxLayout()
        run_row.addWidget(self.start_btn, 1)
        run_row.addWidget(self.cancel_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.status = QLabel("Ready")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)

        right = QVBoxLayout()
        right.addLayout(prev, 3)
        tool_row = QHBoxLayout()
        tool_row.addWidget(self.subjects_btn)
        tool_row.addWidget(self.wm_btn)
        right.addLayout(tool_row)
        right.addWidget(settings_box)
        right.addLayout(run_row)
        right.addWidget(self.progress)
        right.addWidget(self.status)
        right.addWidget(self.log, 1)
        right_w = QWidget()
        right_w.setLayout(right)

        split = QSplitter()
        split.addWidget(left_w)
        split.addWidget(right_w)
        split.setSizes([360, 790])
        self.setCentralWidget(split)
        self.on_background_changed()

    # ---------------- adding files
    def pick_files(self):
        exts = " ".join(f"*{e}" for e in sorted(IMAGE_EXTS)) + " *.zip"
        files, _ = QFileDialog.getOpenFileNames(self, "Add images", "",
                                                f"Images and ZIPs ({exts})")
        if files:
            self.add_paths(files)

    def pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add folder")
        if d:
            self.add_paths([d])

    def pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder")
        if d:
            self.out_dir.setText(d)

    def add_paths(self, paths):
        existing = {self.list.item(i).data(PATH_ROLE) for i in range(self.list.count())}
        added = 0
        for f in collect_inputs(paths):
            key = str(f)
            if key in existing:
                continue
            anim = is_animated(f)
            item = QListWidgetItem()
            item.setData(PATH_ROLE, key)
            item.setData(ANIM_ROLE, anim)
            self.list.addItem(item)
            self._refresh_item(item)
            existing.add(key)
            added += 1
        if not self.out_dir.text() and added:
            first = Path(paths[0])
            base = first if first.is_dir() else first.parent
            self.out_dir.setPlaceholderText(str(base / "no_bg"))
        if self.list.currentItem() is None and self.list.count():
            self.list.setCurrentRow(0)
        self.status.setText(f"{self.list.count()} file(s) queued")

    def remove_selected(self):
        for item in self.list.selectedItems():
            self.selections.pop(item.data(PATH_ROLE), None)
            self.watermarks.pop(item.data(PATH_ROLE), None)
            self.list.takeItem(self.list.row(item))

    def clear_all(self):
        self.selections.clear()
        self.watermarks.clear()
        self.list.clear()

    def _refresh_item(self, item):
        p = Path(item.data(PATH_ROLE))
        tags = []
        if item.data(ANIM_ROLE):
            tags.append("animated")
        sel = self.selections.get(item.data(PATH_ROLE))
        if sel and sel["subjects"]:
            subs = sel["subjects"]
            tags.append(f"{sum(x.keep for x in subs)}/{len(subs)} subjects kept")
        if item.data(PATH_ROLE) in self.watermarks:
            tags.append("watermark")
        if item.data(OUTPUT_ROLE):
            tags.append("✓")
        item.setText(p.name + (f"   [{', '.join(tags)}]" if tags else ""))

    # ---------------- preview
    def _load_preview(self, path):
        with Image.open(path) as im:
            im.seek(0)
            return ImageOps.exif_transpose(im.convert("RGBA")) if not getattr(
                im, "is_animated", False) else im.convert("RGBA")

    def update_preview(self):
        item = self.list.currentItem()
        self.subjects_btn.setEnabled(bool(item))
        self.wm_btn.setEnabled(bool(item))
        if not item:
            self.before.clear()
            self.after.clear()
            self.before.setText("Original")
            self.after.setText("Result")
            return
        try:
            img = self._load_preview(item.data(PATH_ROLE))
            pm = checker_composite(img, self.before.width() - 8, self.before.height() - 8)
            self.before.setPixmap(pm)
            self._preview_img, self._preview_pm = img, pm.size()
        except Exception as e:
            self._preview_img = None
            self.before.setText(f"Can't preview:\n{e}")
        if self.background.currentData() == "color":
            self.update_live_preview()
            return
        out = item.data(OUTPUT_ROLE)
        if out and Path(out).exists():
            try:
                res = self._load_preview(out)
                self.after.setPixmap(checker_composite(res, self.after.width() - 8,
                                                       self.after.height() - 8))
            except Exception as e:
                self.after.setText(f"Can't preview:\n{e}")
        else:
            self.after.clear()
            self.after.setText("Result")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.update_preview()

    # ---------------- colour picking
    def on_background_changed(self, *_):
        is_color = self.background.currentData() == "color"
        self.color_box.setVisible(is_color)
        if not is_color:
            self.eyedrop.setChecked(False)
        self.update_preview()

    def _set_key_color(self, c: QColor):
        self.key_color = QColor(c.red(), c.green(), c.blue())
        self.swatch.setStyleSheet(f"background:{self.key_color.name()};"
                                  "border:1px solid #888; border-radius:4px;")
        self.hex_label.setText(self.key_color.name().upper())
        self.live_timer.start()

    def choose_color(self):
        c = QColorDialog.getColor(self.key_color, self, "Background colour to remove")
        if c.isValid():
            self._set_key_color(c)

    def set_picking(self, on: bool):
        self._picking = on
        self.before.setCursor(QCursor(Qt.CursorShape.CrossCursor if on
                                      else Qt.CursorShape.ArrowCursor))
        if on:
            self.status.setText("Click the background colour in the Original preview")

    def on_preview_click(self, pos: QPoint):
        if not self._picking or self._preview_img is None or self._preview_pm is None:
            return
        pw, ph = self._preview_pm.width(), self._preview_pm.height()
        ox = (self.before.width() - pw) / 2
        oy = (self.before.height() - ph) / 2
        x, y = pos.x() - ox, pos.y() - oy
        if not (0 <= x < pw and 0 <= y < ph):
            return
        img = self._preview_img
        ix = int(x * img.width / pw)
        iy = int(y * img.height / ph)
        # Average a small patch so JPEG noise doesn't skew the pick
        arr = np.asarray(img.convert("RGB"))
        patch = arr[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3].reshape(-1, 3)
        r, g, b = (int(v) for v in patch.mean(axis=0))
        self._set_key_color(QColor(r, g, b))
        self.eyedrop.setChecked(False)
        self.status.setText(f"Picked {self.key_color.name().upper()}")

    def update_live_preview(self):
        """Instant preview of colour keying on the selected file (downscaled)."""
        if self.background.currentData() != "color" or self._preview_img is None:
            return
        img = self._preview_img.copy()
        img.thumbnail((900, 900))
        rgba = np.asarray(img.convert("RGBA"))
        rgb, alpha = key_frame(np.ascontiguousarray(rgba[..., :3]),
                               custom_spec(self._settings()))
        alpha = np.minimum(alpha, rgba[..., 3])
        res = Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA")
        self.after.setPixmap(checker_composite(res, self.after.width() - 8,
                                               self.after.height() - 8))

    # ---------------- manual mode
    def edit_subjects(self):
        item = self.list.currentItem()
        if not item:
            return
        try:
            if item.data(ANIM_ROLE):
                img = load_animation(item.data(PATH_ROLE)).frames   # list → tracking mode
            else:
                img = self._load_preview(item.data(PATH_ROLE))
        except Exception as e:
            QMessageBox.warning(self, "ClearCut", f"Can't open image:\n{e}")
            return
        path = item.data(PATH_ROLE)
        dlg = SubjectEditor(img, self.quality.currentData(), self.selections.get(path), self)
        if dlg.exec():
            if dlg.subjects:
                self.selections[path] = dlg.state()
            else:
                self.selections.pop(path, None)
            self._refresh_item(item)

    def _job_for(self, item) -> Job:
        path = item.data(PATH_ROLE)
        job = Job(Path(path))
        sel = self.selections.get(path)
        if sel and sel["subjects"] and item.data(ANIM_ROLE):
            seeds = [x.mask for x in sel["subjects"] if x.keep]
            job.track = {"ref": sel.get("ref", 0), "seeds": seeds, "matte": sel.get("matte")}
        elif sel and sel["subjects"]:
            mask = combine(sel["subjects"])
            if mask is None:   # everything was removed
                mask = np.zeros_like(sel["subjects"][0].mask)
            job.mask, job.matte_rgb = mask, sel.get("matte")
        job.wm_mask = self.watermarks.get(path)
        return job

    def edit_watermark(self):
        item = self.list.currentItem()
        if not item:
            return
        path = item.data(PATH_ROLE)
        try:
            if item.data(ANIM_ROLE):
                frames = load_animation(path).frames
                step = max(1, len(frames) // 60)      # enough frames to scrub through
                frames = frames[::step]
            else:
                frames = [self._load_preview(path)]
        except Exception as e:
            QMessageBox.warning(self, "ClearCut", f"Can't open file:\n{e}")
            return
        dlg = WatermarkEditor(frames, self.watermarks.get(path),
                              self.wm_method.currentData(), self)
        if dlg.exec():
            m = dlg.result_mask()
            if m is None:
                self.watermarks.pop(path, None)
            else:
                self.watermarks[path] = m
            self._refresh_item(item)

    # ---------------- running
    def _settings(self) -> Settings:
        out = self.out_dir.text().strip() or self.out_dir.placeholderText()
        if not out or out.startswith("Default"):
            out = str(Path.home() / "Pictures" / "no_bg")
        fmt = self.anim_group.checkedButton().property("fmt")
        c = self.key_color
        return Settings(out_dir=Path(out), quality=self.quality.currentData(),
                        background=self.background.currentData(), anim_format=fmt,
                        key_rgb=(c.red(), c.green(), c.blue()), key_tol=self.tol.value(),
                        key_edges_only=self.edges_only.isChecked(),
                        clean_specks=self.specks.isChecked(),
                        wm_method=self.wm_method.currentData())

    def start(self):
        if self.list.count() == 0:
            QMessageBox.information(self, "ClearCut", "Add some images first.")
            return
        selected = self.list.selectedItems()
        items = selected if len(selected) > 1 else \
            [self.list.item(i) for i in range(self.list.count())]
        jobs = [self._job_for(i) for i in items]
        settings = self._settings()

        from core import models
        needs_ai = any(j.mask is None for j in jobs) and settings.background in ("auto", "ai")
        if needs_ai and not models.is_downloaded(settings.quality):
            self.log.appendPlainText("First run: the AI model will be downloaded once "
                                     "(internet needed this one time).")

        self.log.appendPlainText(f"— {len(jobs)} file(s) → {settings.out_dir}")
        self.set_running(True)
        self.thread = QThread(self)
        self.worker = Worker(jobs, settings)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.status.connect(self.status.setText)
        self.worker.log.connect(self.log.appendPlainText)
        self.worker.file_done.connect(self.on_file_done)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(lambda: self.set_running(False))
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.status.setText("Cancelling after current step…")

    def set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.list.setEnabled(not running)
        if running:
            self.subjects_btn.setEnabled(False)
            self.wm_btn.setEnabled(False)
        else:
            self.update_preview()

    def on_file_done(self, src, out):
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(PATH_ROLE) == src:
                item.setData(OUTPUT_ROLE, out)
                self._refresh_item(item)
                if item is self.list.currentItem():
                    self.update_preview()
                break

    def closeEvent(self, e):
        if self.thread and self.thread.isRunning():
            self.worker.cancel()
            self.thread.quit()
            self.thread.wait(5000)
        super().closeEvent(e)