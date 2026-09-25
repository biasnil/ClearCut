"""ClearCut - offline background remover for images and animations.

Run:  python main.py
"""
import os
import sys
from pathlib import Path


def app_dir() -> Path:
    # Works both as a script and as a PyInstaller-frozen exe
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


# Models live in ./models next to the app, so the whole folder can be
# shipped with the models already inside and run fully offline.
# Must be set BEFORE rembg is imported anywhere.
os.environ.setdefault("U2NET_HOME", str(app_dir() / "models"))
(app_dir() / "models").mkdir(exist_ok=True)


def main():
    from PySide6.QtWidgets import QApplication
    from gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("ClearCut")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
