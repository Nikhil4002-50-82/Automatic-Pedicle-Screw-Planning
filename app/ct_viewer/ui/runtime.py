from __future__ import annotations

import sys

from PyQt6.QtCore import qInstallMessageHandler

_INSTALLED = False


def install_qt_message_filter() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    def _handler(_mode, _context, message) -> None:
        text = str(message)
        if text.startswith("QFont::setPointSize: Point size <= 0 (-1)"):
            return
        if text.startswith("QFontDatabase: Cannot find font directory"):
            return
        if text.startswith("js: Canvas2D: Multiple readback operations using getImageData"):
            return
        sys.__stderr__.write(text + "\n")
        sys.__stderr__.flush()

    qInstallMessageHandler(_handler)
    _INSTALLED = True
