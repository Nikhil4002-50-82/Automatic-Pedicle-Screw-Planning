from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow


VIEWER_STYLESHEET = """
QMainWindow { background: #08111d; }
QMenuBar {
    background: #0d1828;
    color: #ecf3fb;
    border-bottom: 1px solid #1c2a3d;
}
QMenuBar::item:selected, QMenu::item:selected { background: #18314d; }
QMenu {
    background: #0d1828;
    color: #ecf3fb;
    border: 1px solid #1f3147;
}
QFrame#sidePanel, QFrame#viewPanel {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0c1522, stop:1 #101c2c);
    border: 1px solid #1f3147;
    border-radius: 18px;
}
QWidget#globalLoadingOverlay {
    background: rgba(7, 12, 20, 0.72);
}
QFrame#loadingCard {
    background: rgba(11, 19, 30, 0.92);
    border: 1px solid rgba(91, 243, 255, 0.18);
    border-radius: 18px;
}
QLabel#loadingOverlayTitle {
    color: #edf2f7;
    font-size: 15px;
    font-weight: 700;
}
QLabel#loadingOverlayText {
    color: #9fb1c4;
    font-size: 12px;
}
QProgressBar {
    background: rgba(20, 31, 45, 0.95);
    border: 1px solid rgba(91, 243, 255, 0.18);
    border-radius: 4px;
}
QProgressBar::chunk {
    background: #5bf3ff;
    border-radius: 4px;
}
QSplitter::handle:vertical {
    background: transparent;
    border: none;
}
QLabel#panelTitle { color: #f7fbff; font-size: 24px; font-weight: 700; }
QLabel#panelSubtitle { color: #9fb1c4; font-size: 13px; }
QLabel#sectionLabel { color: #f6fbff; font-size: 15px; font-weight: 600; margin-top: 4px; }
QLabel#infoBlock {
    color: #d5dfeb;
    background: rgba(16, 29, 45, 0.9);
    border: 1px solid #233448;
    border-radius: 12px;
    padding: 10px 12px;
    font-size: 13px;
}
QLabel#helperText, QLabel#cursorInfo { color: #90a5bb; font-size: 13px; }
QPushButton {
    background: #16314f;
    color: #f4f9ff;
    border: 1px solid #28517d;
    border-radius: 12px;
    min-height: 38px;
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 600;
    text-align: center;
}
QPushButton:hover { background: #1c4067; }
QPushButton:pressed { background: #122a43; }
QPushButton:disabled {
    background: #12263d;
    color: #8aa2ba;
    border: 1px solid #22405f;
}
QPushButton#maskPopupButton {
    color: #f7fbff;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1c446e, stop:1 #17314f);
    border: 1px solid #2c5c8e;
    border-radius: 12px;
    padding: 9px 12px;
    min-height: 42px;
    font-size: 13px;
    font-weight: 700;
}
QPushButton#maskPopupButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #245181, stop:1 #1c4067);
}
QPushButton#maskPopupButton:pressed {
    background: #122a43;
}
QMenu#maskPopupMenu {
    background: rgba(11, 19, 30, 0.98);
    color: #eef5fb;
    border: 1px solid #31526f;
    border-radius: 12px;
    padding: 6px;
}
QMenu#maskPopupMenu::item {
    padding: 8px 14px;
    border-radius: 8px;
    margin: 2px 4px;
}
QMenu#maskPopupMenu::item:selected {
    background: #1c4067;
}
QToolButton#sectionToggle {
    color: #f6fbff;
    background: rgba(19, 34, 55, 0.82);
    border: 1px solid #23405d;
    border-radius: 11px;
    padding: 10px 12px;
    font-size: 13px;
    font-weight: 700;
    text-align: center;
}
QToolButton#sectionToggle:hover { background: rgba(26, 48, 77, 0.92); }
QFrame#collapsibleContent {
    background: transparent;
    border: none;
}
QComboBox, QListWidget, QSlider, QCheckBox { color: #eaf2fb; }
QComboBox, QListWidget {
    background: rgba(13, 24, 40, 0.96);
    border: 1px solid #22354a;
    border-radius: 12px;
    padding: 6px;
}
QComboBox::drop-down { border: none; }
QSlider::groove:horizontal {
    background: #1f2e42;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #59c8ff;
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 18px; height: 18px; }
"""


def apply_styles(viewer: QMainWindow) -> None:
    viewer.setStyleSheet(VIEWER_STYLESHEET)
