import csv
import ast
import json
import os
import sys
import time
from datetime import datetime

import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes

from PyQt6 import QtCore
from PyQt6.QtCore import QObject, Qt, QThread, QSize, QRectF, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QComboBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplashScreen,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QProgressBar,
    QStyle,
)

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except Exception:
    pv = None
    QtInteractor = None

try:
    import qtawesome as qta
except Exception:
    qta = None

QtCore.QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

from geometryV5 import (
    computeDistance,
    computeStableFrame,
    getValidLabels,
    loadNifti,
    optimize,
    pedicleCenters,
)
from mesh_builder import build_vertebra_mesh
from run_totalseg import run_totalseg
from visualizerV7 import visualize_surgical_plan


APP_TITLE = "Automatic Pedicle Screw Planning"
APP_VERSION = "V12.0"
SPLASH_WIDTH = 560
SPLASH_HEIGHT = 280
SPLASH_FILL = "#0b1220"
SPLASH_BORDER = "#2dd4bf"
SPLASH_TEXT = "#ffffff"
SPLASH_TEXT_MUTED = "#e2e8f0"


def loading_panel_stylesheet():
    """In-app loader: dark panel only, no border (workspace stays visible around it)."""
    return (
        f"QFrame#LoadingSplashPanel {{"
        f"  background-color: {SPLASH_FILL};"
        f"  border: none;"
        f"  border-radius: 8px;"
        f"}}"
        f"QLabel#LoadingAppLine {{"
        f"  background: transparent; color: {SPLASH_TEXT};"
        f"  font-size: 11px; font-weight: 800; letter-spacing: 0.4px;"
        f"}}"
        f"QLabel#LoadingTitle {{"
        f"  background: transparent; color: {SPLASH_TEXT};"
        f"  font-size: 20px; font-weight: 900;"
        f"}}"
        f"QLabel#LoadingMessage {{"
        f"  background: transparent; color: {SPLASH_TEXT_MUTED};"
        f"  font-size: 13px; font-weight: 600;"
        f"}}"
        f"QLabel#LoadingHint {{"
        f"  background: transparent; color: {SPLASH_TEXT_MUTED};"
        f"  font-size: 11px; font-weight: 600;"
        f"}}"
        f"QLabel#LoadingPercent {{"
        f"  background: transparent; color: {SPLASH_TEXT};"
        f"  font-size: 22px; font-weight: 900;"
        f"}}"
        f"QProgressBar#LoadingBar {{"
        f"  background: #0f766e;"
        f"  border: none;"
        f"  border-radius: 6px;"
        f"  min-height: 22px;"
        f"  max-height: 22px;"
        f"  color: {SPLASH_TEXT};"
        f"  font-size: 14px;"
        f"  font-weight: 900;"
        f"  text-align: center;"
        f"}}"
        f"QProgressBar#LoadingBar::chunk {{"
        f"  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0f766e, stop:1 {SPLASH_BORDER});"
        f"  border-radius: 6px;"
        f"}}"
    )


class LoadingOverlay(QWidget):
    """Centered loading panel over the workspace; background stays visible outside the panel."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("LoadingOverlayRoot")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setVisible(False)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addStretch(1)

        row = QHBoxLayout()
        row.addStretch(1)

        self.card = QFrame()
        self.card.setObjectName("LoadingSplashPanel")
        self.card.setFixedWidth(SPLASH_WIDTH)
        self.card.setMinimumHeight(SPLASH_HEIGHT)
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(32, 28, 32, 26)
        card_layout.setSpacing(12)

        self.app_line = QLabel(APP_TITLE)
        self.app_line.setObjectName("LoadingAppLine")
        self.app_line.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.loading_title = QLabel("Working...")
        self.loading_title.setObjectName("LoadingTitle")
        self.loading_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.loading_message = QLabel("Please wait.")
        self.loading_message.setObjectName("LoadingMessage")
        self.loading_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_message.setWordWrap(True)

        self.steps_host = QWidget()
        self.steps_host.setStyleSheet("background: transparent;")
        steps_layout = QVBoxLayout(self.steps_host)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(4)
        self.step_labels = []
        self.step_texts = [
            "Prepare data",
            "Segmentation and mesh",
            "Screw optimization",
            "Finalize plan",
        ]
        for text in self.step_texts:
            label = QLabel(f"○ {text}")
            label.setObjectName("LoadingStep")
            label.setStyleSheet(
                f"background: transparent; color: {SPLASH_TEXT_MUTED}; font-size: 12px; font-weight: 700;"
            )
            self.step_labels.append(label)
            steps_layout.addWidget(label)
        self.steps_host.hide()

        self.percent_label = QLabel("")
        self.percent_label.setObjectName("LoadingPercent")
        self.percent_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.percent_label.hide()

        self.loading_bar = QProgressBar()
        self.loading_bar.setObjectName("LoadingBar")
        self.loading_bar.setFixedHeight(22)
        self.loading_bar.setMinimumWidth(SPLASH_WIDTH - 96)
        self.loading_bar.setTextVisible(False)
        self.set_indeterminate()

        self.loading_hint = QLabel("See the Console tab for detailed progress.")
        self.loading_hint.setObjectName("LoadingHint")
        self.loading_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_hint.setWordWrap(True)

        card_layout.addWidget(self.app_line)
        card_layout.addWidget(self.loading_title)
        card_layout.addWidget(self.loading_message)
        card_layout.addWidget(self.steps_host)
        card_layout.addWidget(self.percent_label)
        card_layout.addWidget(self.loading_bar, 0, Qt.AlignmentFlag.AlignHCenter)
        card_layout.addWidget(self.loading_hint)

        row.addWidget(self.card)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)

        self.setStyleSheet("QWidget#LoadingOverlayRoot { background: transparent; }")
        self.card.setStyleSheet(loading_panel_stylesheet())

    def reposition(self):
        if self.parentWidget():
            self.setGeometry(self.parentWidget().rect())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.reposition()

    def set_indeterminate(self):
        self.loading_bar.setRange(0, 0)
        self.percent_label.setText("Loading...")
        self.percent_label.show()

    def set_progress(self, value, message=None):
        self.loading_bar.setRange(0, 100)
        clipped = int(np.clip(value, 0, 100))
        self.loading_bar.setValue(clipped)
        self.percent_label.setText(f"{clipped}%")
        self.percent_label.show()
        if message:
            self.loading_message.setText(message)
        if self.steps_host.isVisible():
            self.update_steps_for_progress(clipped)
        self._resize_card()

    def update_steps_for_progress(self, value):
        if value < 12:
            active = 0
        elif value < 52:
            active = 1
        elif value < 98:
            active = 2
        else:
            active = 3
        for index, label in enumerate(self.step_labels):
            text = self.step_texts[index]
            if index < active:
                label.setText(f"✓ {text}")
                label.setStyleSheet(
                    f"background: transparent; color: {SPLASH_BORDER}; font-size: 12px; font-weight: 800;"
                )
            elif index == active:
                label.setText(f"● {text}")
                label.setStyleSheet(
                    f"background: transparent; color: {SPLASH_TEXT}; font-size: 12px; font-weight: 800;"
                )
            else:
                label.setText(f"○ {text}")
                label.setStyleSheet(
                    f"background: transparent; color: {SPLASH_TEXT_MUTED}; font-size: 12px; font-weight: 700;"
                )

    def _resize_card(self):
        height = SPLASH_HEIGHT
        if self.steps_host.isVisible():
            height += 72
        if self.loading_hint.isVisible():
            height += 8
        self.card.setFixedHeight(height)

    def set_show_steps(self, visible):
        self.steps_host.setVisible(visible)
        if visible:
            self.update_steps_for_progress(self.loading_bar.value() if self.loading_bar.maximum() else 0)
        self._resize_card()

    def present(self, title, message="Please wait.", indeterminate=True, progress=0, show_hint=True, show_steps=False):
        self.app_line.setText(f"{APP_TITLE}  ·  {APP_VERSION}")
        self.loading_title.setText(title)
        self.loading_message.setText(message)
        self.loading_hint.setVisible(show_hint)
        self.set_show_steps(show_steps)
        if indeterminate:
            self.set_indeterminate()
        else:
            self.set_progress(progress, message)
        self._resize_card()
        self.reposition()
        self.raise_()
        self.show()

    def dismiss(self):
        self.hide()


def create_splash_pixmap(
    progress_fraction=0.25,
    status_text="Initializing application...",
    title_text="Starting up",
):
    """Paint startup splash matching the in-app loader (no border, centered white text)."""
    width, height = SPLASH_WIDTH, SPLASH_HEIGHT
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(SPLASH_FILL))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    side = 32
    inner_w = width - 2 * side
    align = int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
    font = painter.font()

    font.setPointSize(11)
    font.setBold(True)
    painter.setPen(QPen(QColor(SPLASH_TEXT)))
    painter.setFont(font)
    painter.drawText(QRectF(side, 22, inner_w, 22), align, f"{APP_TITLE}  ·  {APP_VERSION}")

    font.setPointSize(20)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(QRectF(side, 54, inner_w, 34), align, title_text)

    font.setPointSize(13)
    font.setBold(False)
    painter.setPen(QPen(QColor(SPLASH_TEXT_MUTED)))
    painter.setFont(font)
    painter.drawText(QRectF(side, 92, inner_w, 26), align, status_text)

    percent = int(np.clip(progress_fraction, 0.0, 1.0) * 100)
    font.setPointSize(22)
    font.setBold(True)
    painter.setPen(QPen(QColor(SPLASH_TEXT)))
    painter.setFont(font)
    painter.drawText(QRectF(side, 126, inner_w, 34), align, f"{percent}%")

    track_width = inner_w
    track_height = 22
    track_top = 172
    clipped = float(np.clip(progress_fraction, 0.05, 1.0))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor("#0f766e")))
    painter.drawRoundedRect(side, track_top, track_width, track_height, 6, 6)
    painter.setBrush(QBrush(QColor(SPLASH_BORDER)))
    fill_width = max(32, int(track_width * clipped))
    painter.drawRoundedRect(side, track_top, fill_width, track_height, 6, 6)

    font.setPointSize(11)
    font.setBold(False)
    painter.setPen(QPen(QColor(SPLASH_TEXT_MUTED)))
    painter.setFont(font)
    painter.drawText(QRectF(side, 214, inner_w, 22), align, "Loading interface...")
    painter.end()
    return pixmap


def build_actions_scroll_page(content_widget):
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(content_widget)
    return scroll


AXES = ["Axial", "Coronal", "Sagittal"]
VIEW_HEADER_COLORS = {
    "Axial": "#8b2f2f",
    "Coronal": "#2f7d44",
    "Sagittal": "#8a7428",
    "3D": "#256d7b",
}


def apply_icon(button, icon_name, color="#d9f5f2"):
    if qta is not None:
        try:
            button.setIcon(qta.icon(icon_name, color=color))
            button.setIconSize(QSize(16, 16))
            return
        except Exception:
            pass
    fallback_map = {
        "fa5s.folder-open": QStyle.StandardPixmap.SP_DirOpenIcon,
        "fa5s.file-import": QStyle.StandardPixmap.SP_DialogOpenButton,
        "fa5s.save": QStyle.StandardPixmap.SP_DialogSaveButton,
        "fa5s.play": QStyle.StandardPixmap.SP_MediaPlay,
        "fa5s.undo": QStyle.StandardPixmap.SP_ArrowBack,
        "fa5s.camera": QStyle.StandardPixmap.SP_ComputerIcon,
        "fa5s.expand": QStyle.StandardPixmap.SP_TitleBarMaxButton,
        "fa5s.compress": QStyle.StandardPixmap.SP_TitleBarNormalButton,
        "fa5s.tools": QStyle.StandardPixmap.SP_FileDialogDetailedView,
        "fa5s.notes-medical": QStyle.StandardPixmap.SP_FileDialogInfoView,
        "fa5s.layer-group": QStyle.StandardPixmap.SP_FileDialogListView,
        "fa5s.cube": QStyle.StandardPixmap.SP_DesktopIcon,
        "fa5s.map-marker-alt": QStyle.StandardPixmap.SP_DialogApplyButton,
        "fa5s.file-csv": QStyle.StandardPixmap.SP_FileIcon,
    }
    standard_icon = fallback_map.get(icon_name)
    if standard_icon is not None:
        button.setIcon(button.style().standardIcon(standard_icon))
    button.setIconSize(QSize(16, 16))


def triangles_to_pyvista_faces(faces):
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return faces
    return np.hstack((np.full((faces.shape[0], 1), 3, dtype=np.int64), faces)).ravel()


def polydata_from_triangles(vertices, faces):
    if pv is None:
        return None
    return pv.PolyData(np.asarray(vertices, dtype=float), triangles_to_pyvista_faces(faces))


class LogStream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        if text.strip():
            self.newText.emit(f"[{datetime.now().strftime('%H:%M:%S')}] {text.strip()}")

    def flush(self):
        pass


def normalize_slice(slice_data):
    data = np.asarray(slice_data, dtype=float)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(data.shape, dtype=np.uint8)
    low, high = np.percentile(data[finite], [1, 99])
    if high <= low:
        high = low + 1.0
    return (np.clip((data - low) / (high - low), 0.0, 1.0) * 255).astype(np.uint8)


WINDOW_PRESETS = {
    "Auto": None,
    "Bone": (300, 1500),
    "Soft Tissue": (50, 400),
    "Lung": (-600, 1500),
}


def window_slice(slice_data, preset="Auto", custom_center=300, custom_width=1500):
    data = np.asarray(slice_data, dtype=float)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(data.shape, dtype=np.uint8)
    if preset == "Custom":
        center, width = custom_center, custom_width
    else:
        values = WINDOW_PRESETS.get(preset)
        if values is None:
            low, high = np.percentile(data[finite], [1, 99])
            center = (low + high) / 2.0
            width = max(high - low, 1.0)
        else:
            center, width = values
    width = max(float(width), 1.0)
    low = float(center) - width / 2.0
    high = float(center) + width / 2.0
    return (np.clip((data - low) / max(high - low, 1.0), 0.0, 1.0) * 255).astype(np.uint8)


def label_color(label_value):
    colors = {
        1: np.array([239, 68, 68], dtype=np.uint8),
        2: np.array([245, 158, 11], dtype=np.uint8),
        3: np.array([34, 197, 94], dtype=np.uint8),
        4: np.array([14, 165, 233], dtype=np.uint8),
        5: np.array([168, 85, 247], dtype=np.uint8),
    }
    return colors.get(int(label_value), np.array([236, 72, 153], dtype=np.uint8))


def label_hex(label_value):
    color = label_color(label_value)
    return f"rgb({int(color[0])},{int(color[1])},{int(color[2])})"


def make_rgb_slice(base_slice=None, seg_slice=None, opacity=0.45, preset="Auto", custom_center=300, custom_width=1500):
    if base_slice is None:
        base_slice = seg_slice if seg_slice is not None else np.zeros((64, 64), dtype=float)
    gray = window_slice(base_slice, preset, custom_center, custom_width)
    rgb = np.stack([gray, gray, gray], axis=-1)
    if seg_slice is not None:
        seg = np.asarray(seg_slice)
        mask = seg > 0
        if mask.any():
            overlay = rgb.copy()
            for label in np.unique(seg[mask]):
                overlay[seg == label] = label_color(label)
            rgb = np.where(mask[..., None], (rgb * (1.0 - opacity) + overlay * opacity).astype(np.uint8), rgb)
    return rgb


def slice_for_axis(data, axis, coord):
    if data is None:
        return None
    i, j, k = coord
    if axis == "Axial":
        return np.rot90(data[:, :, k])
    if axis == "Coronal":
        return np.rot90(data[:, j, :])
    return np.rot90(np.flipud(data[i, :, :]))


def axis_dims(shape, axis):
    nx, ny, nz = shape
    if axis == "Axial":
        return nx, ny
    if axis == "Coronal":
        return nx, nz
    return ny, nz


def displayed_dims(shape, axis):
    dim_x, dim_y = axis_dims(shape, axis)
    return dim_x, dim_y


def plane_coords(axis, coord):
    i, j, k = coord
    if axis == "Axial":
        return i, j
    if axis == "Coronal":
        return i, k
    return j, k


def plane_coords_for_shape(axis, coord, shape):
    i, j, k = coord
    dim_x, dim_y = axis_dims(shape, axis)
    if axis == "Axial":
        return i, dim_y - 1 - j
    if axis == "Coronal":
        return i, dim_y - 1 - k
    return dim_x - 1 - j, dim_y - 1 - k


def coord_from_plane(axis, coord, x_value, y_value):
    i, j, k = coord
    if axis == "Axial":
        return [x_value, y_value, k]
    if axis == "Coronal":
        return [x_value, j, y_value]
    return [i, x_value, y_value]


def coord_from_display(axis, coord, x_value, y_value, shape):
    dim_x, dim_y = axis_dims(shape, axis)
    if axis == "Axial":
        return [x_value, dim_y - 1 - y_value, coord[2]]
    if axis == "Coronal":
        return [x_value, coord[1], dim_y - 1 - y_value]
    return [coord[0], dim_x - 1 - x_value, dim_y - 1 - y_value]


def world_to_voxel(point, affine):
    if affine is None:
        return np.asarray(point, dtype=float)
    return nib.affines.apply_affine(np.linalg.inv(affine), point)


def build_label_meshes(seg_data, affine):
    label_meshes = []
    labels = [int(label) for label in np.unique(seg_data) if int(label) != 0]
    for label in labels:
        mask = (seg_data == label).astype(np.uint8)
        if np.sum(mask) < 8:
            continue
        try:
            verts, faces, _, _ = marching_cubes(mask, level=0.5)
        except ValueError:
            continue
        verts_world = nib.affines.apply_affine(affine, verts)
        label_meshes.append({"label": label, "verts": verts_world, "faces": faces})
    return label_meshes


def project_voxel_to_plane(axis, voxel):
    x, y, z = voxel
    if axis == "Axial":
        return x, y, z
    if axis == "Coronal":
        return x, z, y
    return y, z, x


class SliceView(QWidget):
    clicked = pyqtSignal(str, int, int)

    def __init__(self, axis):
        super().__init__()
        self.axis = axis
        self.source_shape = None
        self.pixmap = None
        self.coord = None
        self.annotations = []
        self.plane_bounds = None
        self.fill_mode = True
        self.show_crosshair = False
        self.setMinimumSize(280, 230)
        self.setStyleSheet("background: #000000; color: #e5e7eb; border-radius: 4px;")

    def set_source_shape(self, shape):
        self.source_shape = shape

    def set_fill_mode(self, enabled):
        self.fill_mode = bool(enabled)
        self.update()

    def set_slice(self, rgb, coord, annotations, plane_bounds=None):
        rgb = np.ascontiguousarray(rgb)
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(image.copy())
        self.coord = list(coord)
        self.annotations = list(annotations)
        self.plane_bounds = plane_bounds
        self.update()

    def current_plane_bounds(self):
        if self.source_shape is None:
            return 0, 0, 0, 0
        if self.plane_bounds is not None:
            return self.plane_bounds
        dim_x, dim_y = axis_dims(self.source_shape, self.axis)
        return 0, dim_x - 1, 0, dim_y - 1

    def mousePressEvent(self, event):
        if not self.show_crosshair:
            return
        if self.source_shape is None or self.pixmap is None:
            return
        x_vox, y_vox = self.widget_to_voxel(event.position().x(), event.position().y())
        self.clicked.emit(self.axis, x_vox, y_vox)

    def image_rect(self):
        if self.pixmap is None or self.pixmap.isNull():
            return QRectF(0, 0, self.width(), self.height())
        image_w = max(self.pixmap.width(), 1)
        image_h = max(self.pixmap.height(), 1)
        widget_w = max(self.width(), 1)
        widget_h = max(self.height(), 1)
        scale_x = widget_w / image_w
        scale_y = widget_h / image_h
        scale = max(scale_x, scale_y) if self.fill_mode else min(scale_x, scale_y)
        display_w = image_w * scale
        display_h = image_h * scale
        return QRectF((widget_w - display_w) / 2.0, (widget_h - display_h) / 2.0, display_w, display_h)

    def widget_to_voxel(self, x_pos, y_pos):
        x_min, x_max, y_min, y_max = self.current_plane_bounds()
        rect = self.image_rect()
        px = np.clip((x_pos - rect.left()) / max(rect.width(), 1), 0.0, 1.0)
        py = np.clip((y_pos - rect.top()) / max(rect.height(), 1), 0.0, 1.0)
        x_vox = int(round(x_min + px * max(x_max - x_min, 1)))
        y_vox = int(round(y_min + py * max(y_max - y_min, 1)))
        return x_vox, y_vox

    def voxel_to_widget(self, voxel):
        if self.source_shape is None:
            return 0.0, 0.0
        x_min, x_max, y_min, y_max = self.current_plane_bounds()
        x_plane, y_plane = plane_coords_for_shape(self.axis, voxel, self.source_shape)
        rect = self.image_rect()
        px = (x_plane - x_min) / max(x_max - x_min, 1)
        py = (y_plane - y_min) / max(y_max - y_min, 1)
        return rect.left() + px * rect.width(), rect.top() + py * rect.height()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))
        if self.pixmap is None or self.pixmap.isNull() or self.source_shape is None or self.coord is None:
            painter.setPen(QPen(QColor("#e5e7eb"), 1))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No volume loaded")
            painter.end()
            return

        target = self.image_rect()
        source = QRectF(0, 0, self.pixmap.width(), self.pixmap.height())
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(target, self.pixmap, source)

        if self.show_crosshair:
            x_cross, y_cross = self.voxel_to_widget(self.coord)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(QColor("#38bdf8"), 2))
            painter.drawLine(int(round(x_cross)), 0, int(round(x_cross)), self.height())
            painter.drawLine(0, int(round(y_cross)), self.width(), int(round(y_cross)))
        self.draw_annotations(painter)
        painter.end()

    def draw_annotations(self, painter):
        painter.setPen(QPen(QColor("#facc15"), 3))
        painter.setBrush(QColor(250, 204, 21, 95))
        for annotation in self.annotations:
            voxel = annotation["voxel"]
            selected = {"Axial": self.coord[2], "Coronal": self.coord[1], "Sagittal": self.coord[0]}[self.axis]
            ann_slice = {"Axial": voxel[2], "Coronal": voxel[1], "Sagittal": voxel[0]}[self.axis]
            if ann_slice != selected:
                continue
            ann_x, ann_y = self.voxel_to_widget(voxel)
            painter.drawEllipse(int(ann_x) - 9, int(ann_y) - 9, 18, 18)
            painter.drawLine(int(ann_x) - 14, int(ann_y), int(ann_x) + 14, int(ann_y))
            painter.drawLine(int(ann_x), int(ann_y) - 14, int(ann_x), int(ann_y) + 14)
            painter.drawText(int(ann_x) + 13, int(ann_y) - 11, annotation["label"][:18])
        painter.setBrush(Qt.BrushStyle.NoBrush)


class Mesh3DView(QWidget):
    def __init__(self):
        super().__init__()
        self.label_meshes = []
        self.opacity = 0.35
        self.actors = []
        self.plotter = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if QtInteractor is None:
            missing = QLabel("VTK/PyVista is not available")
            missing.setAlignment(Qt.AlignmentFlag.AlignCenter)
            missing.setStyleSheet("background: #11151b; color: #f59e0b; font-weight: 900;")
            layout.addWidget(missing)
        else:
            self.plotter = QtInteractor(self)
            self.plotter.set_background("#11151b")
            layout.addWidget(self.plotter.interactor)
            self.plotter.add_axes(line_width=1, color="#8da0ae")
            self.plotter.camera_position = "iso"

    def render_empty(self):
        if self.plotter is None:
            return
        for actor in self.actors:
            self.plotter.remove_actor(actor, reset_camera=False)
        self.actors = []
        self.plotter.render()

    def set_scene(self, label_meshes=None, opacity=None):
        self.label_meshes = label_meshes or []
        if opacity is not None:
            self.opacity = opacity
        if not self.label_meshes:
            self.render_empty()
            return
        self.render_scene()

    def render_scene(self):
        if self.plotter is None:
            return
        for actor in self.actors:
            self.plotter.remove_actor(actor, reset_camera=False)
        self.actors = []
        for mesh in self.label_meshes:
            verts = np.asarray(mesh["verts"])
            faces = np.asarray(mesh["faces"])
            label = int(mesh["label"])
            poly = polydata_from_triangles(verts, faces)
            if poly is None:
                continue
            color = label_color(label) / 255.0
            actor = self.plotter.add_mesh(
                poly,
                color=(float(color[0]), float(color[1]), float(color[2])),
                opacity=self.opacity,
                smooth_shading=True,
                specular=0.2,
                name=f"label_{label}",
            )
            self.actors.append(actor)
        self.plotter.reset_camera()
        self.plotter.render()


class FourViewWorkspace(QWidget):
    annotation_added = pyqtSignal(dict)
    show_crosshair_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.ct_data = None
        self.seg_data = None
        self.affine = None
        self.coord = [0, 0, 0]
        self.opacity = 0.45
        self.results = []
        self.label_meshes = []
        self.mesh_opacity = 0.35
        self.window_preset = "Auto"
        self.annotation_mode = False
        self.annotations = []
        self.selected_label = None
        self.label_bounds_cache = {}
        self.label_centroid_cache = {}
        self.show_crosshair = False
        self.views = {}
        self.sliders = {}
        self.view_boxes = {}
        self.view_positions = {}
        self.expand_buttons = {}
        self.expanded_axis = None
        self.init_ui()

    def init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.coord_label = QLabel("Coordinate: none")
        self.coord_label.setStyleSheet("font-weight: 800; color: #d1d5db;")
        self.annotation_btn = QPushButton("Add Annotation")
        apply_icon(self.annotation_btn, "fa5s.map-marker-alt")
        self.annotation_btn.setCheckable(True)
        self.annotation_btn.clicked.connect(self.toggle_annotation_mode)
        top.addWidget(self.coord_label)
        top.addStretch(1)
        top.addWidget(QLabel("Overlay opacity"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(45)
        self.opacity_slider.valueChanged.connect(self.set_opacity)
        self.opacity_value = QLabel("45%")
        top.addWidget(self.opacity_slider)
        top.addWidget(self.opacity_value)
        top.addWidget(QLabel("Mesh opacity"))
        self.mesh_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.mesh_opacity_slider.setRange(5, 100)
        self.mesh_opacity_slider.setValue(35)
        self.mesh_opacity_slider.valueChanged.connect(self.set_mesh_opacity)
        self.mesh_opacity_value = QLabel("35%")
        top.addWidget(self.mesh_opacity_slider)
        top.addWidget(self.mesh_opacity_value)
        self.fill_views_btn = QPushButton("Fill views")
        self.fill_views_btn.setCheckable(True)
        self.fill_views_btn.setChecked(True)
        self.fill_views_btn.setToolTip("Fill each MPR panel by preserving aspect ratio and cropping only when needed")
        self.fill_views_btn.clicked.connect(self.set_fill_views)
        top.addWidget(self.fill_views_btn)
        top.addWidget(self.annotation_btn)
        root.addLayout(top)

        self.close_expand_btn = QPushButton("Close Expanded View")
        apply_icon(self.close_expand_btn, "fa5s.compress")
        self.close_expand_btn.clicked.connect(self.close_expanded_view)
        self.close_expand_btn.hide()
        root.addWidget(self.close_expand_btn)

        self.grid = QGridLayout()
        self.grid.setSpacing(14)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)
        self.grid.setRowStretch(0, 1)
        self.grid.setRowStretch(1, 1)
        for row, col, axis in [(0, 0, "Axial"), (0, 1, "Coronal"), (1, 0, "Sagittal")]:
            box = QGroupBox("")
            box.setObjectName("ViewBox")
            box.setMinimumHeight(300)
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(0, 0, 0, 8)
            box_layout.setSpacing(6)
            header_widget = QWidget()
            header_widget.setStyleSheet(f"background: {VIEW_HEADER_COLORS[axis]}; border-radius: 4px 4px 0 0;")
            header = QHBoxLayout(header_widget)
            header.setContentsMargins(10, 4, 8, 4)
            title = QLabel(axis)
            title.setStyleSheet("font-size: 13px; font-weight: 900; color: #f8fafc; background: transparent;")
            expand_btn = QPushButton("")
            apply_icon(expand_btn, "fa5s.expand")
            expand_btn.setToolTip(f"Expand {axis} view")
            expand_btn.setFixedSize(30, 26)
            expand_btn.clicked.connect(lambda _, a=axis: self.expand_view(a))
            header.addWidget(title)
            header.addStretch(1)
            header.addWidget(expand_btn)
            view = SliceView(axis)
            view.clicked.connect(self.handle_view_click)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.valueChanged.connect(lambda value, a=axis: self.slider_changed(a, value))
            box_layout.addWidget(header_widget)
            box_layout.addWidget(view, 1)
            box_layout.addWidget(slider)
            self.views[axis] = view
            self.sliders[axis] = slider
            self.view_boxes[axis] = box
            self.view_positions[axis] = (row, col)
            self.expand_buttons[axis] = expand_btn
            self.grid.addWidget(box, row, col)
        mesh_box = QGroupBox("")
        mesh_box.setObjectName("ViewBox")
        mesh_box.setMinimumHeight(300)
        mesh_layout = QVBoxLayout(mesh_box)
        mesh_layout.setContentsMargins(0, 0, 0, 8)
        mesh_layout.setSpacing(6)
        mesh_header_widget = QWidget()
        mesh_header_widget.setStyleSheet(f"background: {VIEW_HEADER_COLORS['3D']}; border-radius: 4px 4px 0 0;")
        mesh_header = QHBoxLayout(mesh_header_widget)
        mesh_header.setContentsMargins(10, 4, 8, 4)
        mesh_title = QLabel("3D Segmentation Mesh")
        mesh_title.setStyleSheet("font-size: 13px; font-weight: 900; color: #f8fafc; background: transparent;")
        mesh_expand_btn = QPushButton("")
        apply_icon(mesh_expand_btn, "fa5s.expand")
        mesh_expand_btn.setToolTip("Expand 3D mesh view")
        mesh_expand_btn.setFixedSize(30, 26)
        mesh_expand_btn.clicked.connect(lambda: self.expand_view("3D"))
        mesh_header.addWidget(mesh_title)
        mesh_header.addStretch(1)
        mesh_header.addWidget(mesh_expand_btn)
        self.mesh_view = Mesh3DView()
        mesh_layout.addWidget(mesh_header_widget)
        mesh_layout.addWidget(self.mesh_view, 1)
        mesh_footer = QLabel("")
        mesh_footer.setFixedHeight(22)
        mesh_layout.addWidget(mesh_footer)
        self.view_boxes["3D"] = mesh_box
        self.view_positions["3D"] = (1, 1)
        self.expand_buttons["3D"] = mesh_expand_btn
        self.grid.addWidget(mesh_box, 1, 1)
        root.addLayout(self.grid, 1)

    def toggle_annotation_mode(self):
        self.annotation_mode = self.annotation_btn.isChecked()
        if self.annotation_mode:
            self.set_show_crosshair(True)
        self.annotation_btn.setText("Click MPR Point" if self.annotation_mode else "Add Annotation")

    def set_opacity(self, value):
        self.opacity = value / 100.0
        self.opacity_value.setText(f"{value}%")
        self.update_mpr_views()

    def set_mesh_opacity(self, value):
        self.mesh_opacity = value / 100.0
        self.mesh_opacity_value.setText(f"{value}%")
        self.mesh_view.set_scene(self.label_meshes, self.mesh_opacity)

    def set_window_preset(self, preset):
        self.window_preset = preset if preset in WINDOW_PRESETS or preset == "Custom" else "Auto"
        self.update_mpr_views()

    def set_fill_views(self, checked):
        for view in self.views.values():
            view.set_fill_mode(checked)

    def set_show_crosshair(self, checked):
        checked = bool(checked)
        if self.show_crosshair == checked:
            return
        self.show_crosshair = checked
        self.show_crosshair_changed.emit(checked)
        self.update_mpr_views()

    def set_volumes(self, ct_data=None, seg_data=None, affine=None):
        self.ct_data = ct_data
        self.seg_data = seg_data
        self.affine = affine
        self.selected_label = None
        self.label_bounds_cache = {}
        self.label_centroid_cache = {}
        source = ct_data if ct_data is not None else seg_data
        if source is None:
            return
        shape = source.shape
        self.coord = [shape[0] // 2, shape[1] // 2, shape[2] // 2]
        for view in self.views.values():
            view.set_source_shape(shape)
        self.configure_sliders(shape)
        self.update_mpr_views()

    def set_mesh(self, label_meshes):
        self.label_meshes = label_meshes or []
        self.mesh_view.set_scene(self.label_meshes, self.mesh_opacity)

    def set_results(self, results):
        self.results = results or []
        self.update_mpr_views()
        self.mesh_view.set_scene(self.label_meshes, self.mesh_opacity)

    def expand_view(self, axis):
        if self.expanded_axis == axis:
            return
        self.expanded_axis = axis
        for key, box in self.view_boxes.items():
            box.hide()
        selected = self.view_boxes[axis]
        self.grid.addWidget(selected, 0, 0, 2, 2)
        selected.show()
        self.set_expand_button_mode(axis, close_mode=True)
        self.update_mpr_views()

    def close_expanded_view(self):
        if self.expanded_axis is None:
            return
        expanded_axis = self.expanded_axis
        selected = self.view_boxes[self.expanded_axis]
        self.grid.removeWidget(selected)
        for key, box in self.view_boxes.items():
            row, col = self.view_positions[key]
            self.grid.addWidget(box, row, col)
            box.show()
        self.expanded_axis = None
        self.set_expand_button_mode(expanded_axis, close_mode=False)
        self.update_mpr_views()

    def set_expand_button_mode(self, axis, close_mode=False):
        button = self.expand_buttons[axis]
        try:
            button.clicked.disconnect()
        except TypeError:
            pass
        if close_mode:
            button.setText("")
            apply_icon(button, "fa5s.compress")
            button.setToolTip("Close expanded view")
            button.clicked.connect(self.close_expanded_view)
        else:
            button.setText("")
            apply_icon(button, "fa5s.expand")
            button.setToolTip(f"Expand {axis} view")
            button.clicked.connect(lambda _, a=axis: self.expand_view(a))

    def configure_sliders(self, shape):
        ranges = {"Sagittal": shape[0] - 1, "Coronal": shape[1] - 1, "Axial": shape[2] - 1}
        values = {"Sagittal": self.coord[0], "Coronal": self.coord[1], "Axial": self.coord[2]}
        for axis, slider in self.sliders.items():
            slider.blockSignals(True)
            slider.setRange(0, max(0, ranges[axis]))
            slider.setValue(values[axis])
            slider.blockSignals(False)

    def current_shape(self):
        source = self.ct_data if self.ct_data is not None else self.seg_data
        return None if source is None else source.shape

    def slider_changed(self, axis, value):
        if axis == "Axial":
            self.coord[2] = value
        elif axis == "Coronal":
            self.coord[1] = value
        else:
            self.coord[0] = value
        self.update_sliders()
        self.update_mpr_views()

    def handle_view_click(self, axis, x_value, y_value):
        shape = self.current_shape()
        if shape is None:
            return
        self.coord = coord_from_display(axis, self.coord, x_value, y_value, shape)
        self.coord = [
            int(np.clip(self.coord[0], 0, shape[0] - 1)),
            int(np.clip(self.coord[1], 0, shape[1] - 1)),
            int(np.clip(self.coord[2], 0, shape[2] - 1)),
        ]
        self.select_label_at_coord()
        if self.selected_label is not None:
            self.move_to_selected_label_center()
        self.update_sliders()
        self.update_mpr_views()
        if self.annotation_mode:
            self.create_annotation(axis)

    def jump_to_world_point(self, point):
        shape = self.current_shape()
        if shape is None or self.affine is None:
            return
        voxel = world_to_voxel(point, self.affine)
        self.coord = [
            int(np.clip(round(voxel[0]), 0, shape[0] - 1)),
            int(np.clip(round(voxel[1]), 0, shape[1] - 1)),
            int(np.clip(round(voxel[2]), 0, shape[2] - 1)),
        ]
        self.select_label_at_coord()
        self.update_sliders()
        self.update_mpr_views()

    def jump_to_voxel(self, voxel):
        shape = self.current_shape()
        if shape is None or voxel is None or len(voxel) < 3:
            return
        self.coord = [
            int(np.clip(round(float(voxel[0])), 0, shape[0] - 1)),
            int(np.clip(round(float(voxel[1])), 0, shape[1] - 1)),
            int(np.clip(round(float(voxel[2])), 0, shape[2] - 1)),
        ]
        self.select_label_at_coord()
        self.set_show_crosshair(True)
        self.update_sliders()
        self.update_mpr_views()

    def select_label_at_coord(self):
        if self.seg_data is None:
            self.selected_label = None
            return
        i, j, k = self.coord
        label = int(round(float(self.seg_data[i, j, k])))
        self.selected_label = label if label > 0 else None

    def selected_label_bounds(self):
        if self.seg_data is None or self.selected_label is None:
            return None
        label = int(self.selected_label)
        if label not in self.label_bounds_cache:
            points = np.argwhere(np.asarray(self.seg_data).round().astype(np.int32) == label)
            if points.size == 0:
                self.label_bounds_cache[label] = None
            else:
                self.label_bounds_cache[label] = (points.min(axis=0), points.max(axis=0))
        return self.label_bounds_cache[label]

    def move_to_selected_label_center(self):
        shape = self.current_shape()
        if self.seg_data is None or self.selected_label is None or shape is None:
            return
        label = int(self.selected_label)
        if label not in self.label_centroid_cache:
            points = np.argwhere(np.asarray(self.seg_data).round().astype(np.int32) == label)
            self.label_centroid_cache[label] = None if points.size == 0 else np.floor(points.mean(axis=0) + 0.5).astype(int)
        center = self.label_centroid_cache[label]
        if center is None:
            return
        self.coord = [
            int(np.clip(center[0], 0, shape[0] - 1)),
            int(np.clip(center[1], 0, shape[1] - 1)),
            int(np.clip(center[2], 0, shape[2] - 1)),
        ]

    def axis_plane_bounds(self, axis, shape):
        bounds = self.selected_label_bounds()
        dim_x, dim_y = axis_dims(shape, axis)
        if bounds is None:
            return 0, dim_x - 1, 0, dim_y - 1
        mins, maxs = bounds
        if axis == "Axial":
            x_min, x_max = int(mins[0]), int(maxs[0])
            y_min, y_max = int(mins[1]), int(maxs[1])
        elif axis == "Coronal":
            x_min, x_max = int(mins[0]), int(maxs[0])
            y_min, y_max = int(mins[2]), int(maxs[2])
        else:
            x_min, x_max = int(mins[1]), int(maxs[1])
            y_min, y_max = int(mins[2]), int(maxs[2])

        pad_x = max(10, int(round((x_max - x_min + 1) * 0.35)))
        pad_y = max(10, int(round((y_max - y_min + 1) * 0.35)))
        x_min = max(0, x_min - pad_x)
        x_max = min(dim_x - 1, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(dim_y - 1, y_max + pad_y)
        return x_min, x_max, y_min, y_max

    def create_annotation(self, axis):
        label, ok = QInputDialog.getText(self, "Add Annotation", "Annotation label:")
        if not ok or not label.strip():
            return
        world = None
        if self.affine is not None:
            world = nib.affines.apply_affine(self.affine, self.coord).round(2).tolist()
        annotation = {
            "label": label.strip(),
            "voxel": list(self.coord),
            "world": world,
            "segmentation_label": self.selected_label,
            "view": axis,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        self.annotations.append(annotation)
        self.annotation_added.emit(annotation)
        self.annotation_btn.setChecked(False)
        self.toggle_annotation_mode()
        self.update_mpr_views()

    def update_sliders(self):
        values = {"Sagittal": self.coord[0], "Coronal": self.coord[1], "Axial": self.coord[2]}
        for axis, slider in self.sliders.items():
            slider.blockSignals(True)
            slider.setValue(values[axis])
            slider.blockSignals(False)

    def update_mpr_views(self):
        shape = self.current_shape()
        if shape is None:
            return
        label_text = f" | label={self.selected_label}" if self.selected_label is not None else ""
        self.coord_label.setText(f"Voxel coordinate: i={self.coord[0]}  j={self.coord[1]}  k={self.coord[2]}{label_text}")
        for axis, view in self.views.items():
            base = slice_for_axis(self.ct_data if self.ct_data is not None else self.seg_data, axis, self.coord)
            seg = slice_for_axis(self.seg_data, axis, self.coord) if self.seg_data is not None else None
            rgb = make_rgb_slice(base, seg, self.opacity, self.window_preset)
            self.set_view_pixmap(view, rgb, axis, shape)

    def set_view_pixmap(self, view, rgb, axis, shape):
        view.set_source_shape(shape)
        view.show_crosshair = self.show_crosshair
        view.set_slice(rgb, self.coord, self.annotations)

    def draw_overlays(self, pixmap, axis, shape):
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        dim_x, dim_y = axis_dims(shape, axis)
        x_plane, y_plane = plane_coords(axis, self.coord)
        px = (1.0 - (y_plane / max(dim_y - 1, 1))) * pixmap.width()
        py = (x_plane / max(dim_x - 1, 1)) * pixmap.height()
        painter.setPen(QPen(QColor("#38bdf8"), 1))
        painter.drawLine(int(px), 0, int(px), pixmap.height())
        painter.drawLine(0, int(py), pixmap.width(), int(py))
        self.draw_annotations(painter, pixmap, axis, dim_x, dim_y)
        painter.end()
        return pixmap

    def draw_annotations(self, painter, pixmap, axis, dim_x, dim_y):
        painter.setPen(QPen(QColor("#facc15"), 2))
        for annotation in self.annotations:
            voxel = annotation["voxel"]
            selected = {"Axial": self.coord[2], "Coronal": self.coord[1], "Sagittal": self.coord[0]}[axis]
            ann_slice = {"Axial": voxel[2], "Coronal": voxel[1], "Sagittal": voxel[0]}[axis]
            if ann_slice != selected:
                continue
            ax, ay = plane_coords(axis, voxel)
            ann_x = (1.0 - (ay / max(dim_y - 1, 1))) * pixmap.width()
            ann_y = (ax / max(dim_x - 1, 1)) * pixmap.height()
            painter.drawEllipse(int(ann_x) - 5, int(ann_y) - 5, 10, 10)
            painter.drawText(int(ann_x) + 7, int(ann_y) - 7, annotation["label"][:18])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_mpr_views()


class Worker(QThread):
    screw_found = pyqtSignal(dict)
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, object, list, str)
    failed = pyqtSignal(str)

    def __init__(self, ct_path=None, seg_path=None):
        super().__init__()
        self.ct_path = ct_path
        self.seg_path = seg_path

    def report(self, value, message):
        self.progress.emit(int(np.clip(value, 0, 100)), str(message))

    def build_mesh_from_combined(self, seg_path):
        nii = nib.load(seg_path)
        data = nii.get_fdata()
        mask = (data > 0).astype(np.uint8)
        if np.sum(mask) == 0:
            raise ValueError("Segmented file is empty. No mesh can be created.")
        verts, faces, _, _ = marching_cubes(mask, level=0.5)
        return nib.affines.apply_affine(nii.affine, verts), faces

    def run(self):
        try:
            self.report(0, "Starting planning pipeline...")
            if self.seg_path:
                print("MODE: Using loaded segmentation for mesh and planning.")
                combined_path = self.seg_path
                self.report(8, "Building 3D mesh from segmentation...")
                verts_world, faces = self.build_mesh_from_combined(combined_path)
                self.report(22, "Mesh ready. Preparing screw optimization...")
            elif self.ct_path:
                print("MODE: Running TotalSegmentator from CT scan.")
                self.report(5, "Running TotalSegmentator (this may take several minutes)...")
                seg_data = run_totalseg(self.ct_path)
                self.report(38, "Building vertebra surface mesh...")
                combined_path = seg_data["combined_seg_path"]
                verts_world, faces = build_vertebra_mesh(seg_data["seg_folder"])
                self.report(48, "Segmentation complete. Starting screw optimization...")
            else:
                raise ValueError("Load a CT scan or segmentation before running planning.")

            print("PLANNING: Calculating screw trajectories.")
            results = []
            label_map = {5: "L1", 4: "L2", 3: "L3", 2: "L4", 1: "L5"}
            self.report(52, "Loading segmentation for trajectory analysis...")
            seg, spacing, affine = loadNifti(combined_path)
            valid_segments = list(sorted(getValidLabels(seg), reverse=True))
            planning_steps = max(len(valid_segments) * 2, 1)
            step_index = 0
            plan_start, plan_end = 55, 96

            for label_val, mask in valid_segments:
                name = label_map.get(label_val, str(label_val))
                self.report(
                    plan_start + int((plan_end - plan_start) * step_index / planning_steps),
                    f"Analyzing {name} pedicles...",
                )
                centroid, axes, total_depth = computeStableFrame(mask, affine)
                dist = computeDistance(mask, spacing)
                mask_float = mask.astype(float)
                left_center, right_center = pedicleCenters(mask, dist, centroid, axes, affine)
                for side, center in [("Left", left_center), ("Right", right_center)]:
                    self.report(
                        plan_start + int((plan_end - plan_start) * step_index / planning_steps),
                        f"Optimizing {name} {side} screw trajectory...",
                    )
                    res = optimize(center, axes, side, mask_float, dist, affine, centroid, total_depth, name)
                    step_index += 1
                    if not res:
                        print(f"WARNING: No valid trajectory found for {name} {side}.")
                        continue
                    score, entry, tip, length, min_dt, lr_ang, si_ang, diam = res
                    screw = {
                        "vertebra": name,
                        "side": side,
                        "entry": np.asarray(entry).tolist(),
                        "tip": np.asarray(tip).tolist(),
                        "diameter": float(diam),
                        "length": float(length),
                        "axial": float(lr_ang),
                        "sagittal": float(si_ang),
                        "min_clearance": float(min_dt),
                    }
                    results.append(screw)
                    self.screw_found.emit(screw)
                    print(f"SUCCESS: Trajectory found for {name} {side}.")
            self.report(100, "Planning complete.")
            self.finished.emit(verts_world, faces, results, combined_path)
        except Exception as exc:
            self.failed.emit(str(exc))


class GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ct_path = None
        self.seg_path = None
        self.combined_seg_path = None
        self.ct_data = None
        self.seg_data = None
        self.affine = None
        self.verts = None
        self.faces = None
        self.results = []
        self.ct_records = []
        self.seg_records = []
        self.active_ct_index = None
        self.active_seg_index = None
        self.worker = None
        self.init_ui()
        self.stream = LogStream()
        self.stream.newText.connect(self.update_log)
        sys.stdout = self.stream
        sys.stderr = self.stream

    def init_ui(self):
        self.setWindowTitle("Automatic Pedicle Screw Planning System - V12.0")
        self.resize(1500, 980)
        root = QWidget()
        main = QVBoxLayout(root)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(6)
        self.setCentralWidget(root)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(12)
        self.ct_label = QLabel("CT: not loaded")
        self.seg_label = QLabel("Segmentation: not loaded")
        self.status_label = QLabel("Status: ready")
        self.actions_toggle_btn = QPushButton("Actions")
        apply_icon(self.actions_toggle_btn, "fa5s.tools")
        self.actions_toggle_btn.setCheckable(True)
        self.actions_toggle_btn.clicked.connect(self.toggle_actions)
        self.run_btn = QPushButton("Run Planning")
        apply_icon(self.run_btn, "fa5s.play")
        self.run_btn.clicked.connect(self.run_pipeline)
        self.visual_btn = QPushButton("Launch Manual 3D View")
        apply_icon(self.visual_btn, "fa5s.cube")
        self.visual_btn.clicked.connect(self.visualize)
        self.visual_btn.setEnabled(False)
        header.addWidget(self.ct_label)
        header.addWidget(self.seg_label)
        header.addWidget(self.status_label)
        header.addStretch(1)
        header.addWidget(self.actions_toggle_btn)
        main.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.setMinimumSize(1180, 760)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_area.setWidget(self.tabs)
        main.addWidget(scroll_area, 1)
        self._central_root = root
        self.loading_overlay = LoadingOverlay(root)
        self.loading_overlay.hide()

        workspace_page = QWidget()
        workspace_layout = QHBoxLayout(workspace_page)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        self.workspace = FourViewWorkspace()
        self.workspace.annotation_added.connect(self.add_annotation_row)
        self.workspace.show_crosshair_changed.connect(self.sync_crosshair_toggle)
        workspace_layout.addWidget(self.workspace, 1)
        self.actions_panel = QGroupBox("Actions")
        self.actions_panel.setObjectName("ActionsPanel")
        actions_panel_layout = QVBoxLayout(self.actions_panel)
        actions_panel_layout.setContentsMargins(8, 10, 8, 8)
        actions_panel_layout.setSpacing(6)

        self.actions_tabs = QTabWidget()
        self.actions_tabs.setObjectName("ActionsTabs")
        self.actions_tabs.setDocumentMode(True)

        load_files_host = QWidget()
        load_files_layout = QVBoxLayout(load_files_host)
        load_files_layout.setContentsMargins(4, 8, 4, 8)
        load_files_layout.setSpacing(8)
        load_ct_btn = QPushButton("Load CT")
        apply_icon(load_ct_btn, "fa5s.folder-open")
        load_ct_btn.clicked.connect(self.load_ct)
        load_seg_btn = QPushButton("Load Segmentation")
        apply_icon(load_seg_btn, "fa5s.layer-group")
        load_seg_btn.clicked.connect(self.load_seg)
        load_plan_btn = QPushButton("Load Plan")
        apply_icon(load_plan_btn, "fa5s.file-import")
        load_plan_btn.clicked.connect(self.load_plan)
        self.ct_list = QListWidget()
        self.ct_list.setMinimumHeight(95)
        self.ct_list.itemClicked.connect(self.ct_item_clicked)
        self.seg_list = QListWidget()
        self.seg_list.setMinimumHeight(95)
        self.seg_list.itemClicked.connect(self.seg_item_clicked)
        remove_ct_btn = QPushButton("Remove CT")
        apply_icon(remove_ct_btn, "fa5s.undo")
        remove_ct_btn.clicked.connect(self.remove_selected_ct)
        remove_seg_btn = QPushButton("Remove Seg")
        apply_icon(remove_seg_btn, "fa5s.undo")
        remove_seg_btn.clicked.connect(self.remove_selected_seg)
        load_files_layout.addWidget(load_ct_btn)
        load_files_layout.addWidget(load_seg_btn)
        load_files_layout.addWidget(load_plan_btn)
        load_files_layout.addWidget(QLabel("Loaded CTs"))
        load_files_layout.addWidget(self.ct_list)
        load_files_layout.addWidget(remove_ct_btn)
        load_files_layout.addWidget(QLabel("Loaded Segmentations"))
        load_files_layout.addWidget(self.seg_list)
        load_files_layout.addWidget(remove_seg_btn)
        load_files_layout.addStretch(1)
        self.actions_tabs.addTab(build_actions_scroll_page(load_files_host), "Load Files")

        planning_host = QWidget()
        planning_layout = QVBoxLayout(planning_host)
        planning_layout.setContentsMargins(4, 8, 4, 8)
        planning_layout.setSpacing(8)
        self.crosshair_toggle_btn = QPushButton("Crosshair")
        apply_icon(self.crosshair_toggle_btn, "fa5s.crosshairs")
        self.crosshair_toggle_btn.setCheckable(True)
        self.crosshair_toggle_btn.setChecked(False)
        self.crosshair_toggle_btn.setEnabled(False)
        self.crosshair_toggle_btn.setToolTip("Toggle crosshair visibility")
        self.crosshair_toggle_btn.clicked.connect(self.workspace.set_show_crosshair)
        self.window_preset_combo = QComboBox()
        self.window_preset_combo.addItems(list(WINDOW_PRESETS.keys()))
        self.window_preset_combo.currentTextChanged.connect(self.workspace.set_window_preset)
        planning_layout.addWidget(QLabel("Window preset"))
        planning_layout.addWidget(self.window_preset_combo)
        planning_layout.addWidget(self.crosshair_toggle_btn)
        planning_layout.addWidget(self.run_btn)
        planning_layout.addWidget(self.visual_btn)
        planning_layout.addStretch(1)
        self.actions_tabs.addTab(build_actions_scroll_page(planning_host), "Planning")

        annotation_host = QWidget()
        annotation_layout = QVBoxLayout(annotation_host)
        annotation_layout.setContentsMargins(4, 8, 4, 8)
        annotation_layout.setSpacing(8)
        self.annotation_list = QListWidget()
        self.annotation_list.setMinimumHeight(140)
        self.annotation_list.itemClicked.connect(self.annotation_item_clicked)
        load_ann_btn = QPushButton("Load Annotations JSON")
        apply_icon(load_ann_btn, "fa5s.file-import")
        load_ann_btn.clicked.connect(self.load_annotations)
        save_ann_btn = QPushButton("Save Annotations JSON")
        apply_icon(save_ann_btn, "fa5s.save")
        save_ann_btn.clicked.connect(self.save_annotations)
        annotation_layout.addWidget(self.annotation_list)
        annotation_layout.addWidget(load_ann_btn)
        annotation_layout.addWidget(save_ann_btn)
        annotation_layout.addStretch(1)
        self.actions_tabs.addTab(build_actions_scroll_page(annotation_host), "Annotation")

        export_host = QWidget()
        export_layout = QVBoxLayout(export_host)
        export_layout.setContentsMargins(4, 8, 4, 8)
        export_layout.setSpacing(8)
        save_btn = QPushButton("Save Plan JSON")
        apply_icon(save_btn, "fa5s.save")
        save_btn.clicked.connect(self.save_json)
        report_btn = QPushButton("Export CSV Report")
        apply_icon(report_btn, "fa5s.file-csv")
        report_btn.clicked.connect(self.export_csv)
        export_layout.addWidget(save_btn)
        export_layout.addWidget(report_btn)
        export_layout.addStretch(1)
        self.actions_tabs.addTab(build_actions_scroll_page(export_host), "Export")

        actions_panel_layout.addWidget(self.actions_tabs, 1)
        self.actions_panel.setFixedWidth(340)
        self.actions_panel.hide()
        workspace_layout.addWidget(self.actions_panel)
        self.tabs.addTab(workspace_page, "Planning Workspace")

        results_page = QWidget()
        results_layout = QVBoxLayout(results_page)
        self.table = QTableWidget()
        headers = ["Vertebra", "Side", "Diameter (mm)", "Length (mm)", "Axial", "Sagittal", "Clearance", "Entry Point", "Tip Point"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.cellClicked.connect(self.table_row_clicked)
        results_layout.addWidget(self.table)
        self.tabs.addTab(results_page, "Planning Results")

        console_page = QWidget()
        console_layout = QVBoxLayout(console_page)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        console_layout.addWidget(self.log_box)
        self.tabs.addTab(console_page, "Console")
        self.apply_style()

    def apply_style(self):
        self.setStyleSheet(
            "QWidget { background: #171b22; font-family: Segoe UI; color: #e5e7eb; }"
            "QLabel { color: #d1d5db; }"
            "QPushButton { background: #0f766e; color: white; border: none; padding: 9px 13px; border-radius: 5px; font-weight: 800; }"
            "QPushButton:hover { background: #14b8a6; }"
            "QPushButton:checked { background: #f59e0b; color: #111827; }"
            "QPushButton:disabled { background: #475569; color: #94a3b8; }"
            "QTabWidget::pane { border: 1px solid #343c49; background: #20262f; }"
            "QTabBar::tab { background: #262d37; color: #cbd5e1; padding: 9px 18px; font-weight: 800; border: 1px solid #343c49; }"
            "QTabBar::tab:selected { background: #0f766e; color: white; }"
            "QTableWidget { background: #11151b; alternate-background-color: #1f2937; color: #e5e7eb; gridline-color: #343c49; font-size: 13px; }"
            "QHeaderView::section { background: #20262f; color: #f8fafc; padding: 8px; font-weight: 800; border: 1px solid #343c49; }"
            "QTextEdit { background: #0d1117; color: #d1d5db; font-family: Consolas; font-size: 12px; border-radius: 4px; padding: 8px; border: 1px solid #343c49; }"
            "QGroupBox { border: 1px solid #343c49; border-radius: 5px; margin-top: 8px; padding-top: 10px; font-weight: 800; }"
            "QGroupBox#ViewBox { background: #20262f; border: 1px solid #343c49; border-radius: 5px; margin-top: 0; padding-top: 0; }"
            "QGroupBox#ActionsPanel { background: #20262f; border: 1px solid #3f4a59; border-radius: 6px; }"
            "QTabWidget#ActionsTabs::pane { border: 1px solid #3f4a59; background: #1a212b; border-radius: 4px; top: -1px; }"
            "QTabWidget#ActionsTabs QTabBar::tab { background: #262d37; color: #cbd5e1; padding: 6px 10px; font-weight: 800; border: 1px solid #3f4a59; }"
            "QTabWidget#ActionsTabs QTabBar::tab:selected { background: #0f766e; color: white; }"
            "QLabel#PanelTitle { color: #2dd4bf; font-size: 12px; font-weight: 900; padding-top: 4px; padding-bottom: 2px; }"
            "QListWidget { background: #11151b; color: #e5e7eb; border: 1px solid #343c49; border-radius: 4px; }"
            "QScrollArea { border: none; }"
            "QSlider::groove:horizontal { height: 5px; background: #3f4a59; border-radius: 2px; }"
            "QSlider::handle:horizontal { background: #2dd4bf; width: 14px; margin: -5px 0; border-radius: 7px; }"
        )

    def update_log(self, text):
        self.log_box.append(text)
        self.log_box.ensureCursorVisible()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "loading_overlay"):
            self.loading_overlay.reposition()

    def show_loading(
        self,
        title,
        message="Please wait.",
        indeterminate=True,
        progress=0,
        show_hint=False,
        show_steps=False,
    ):
        self._loading_ui_state = {
            "run": self.run_btn.isEnabled(),
            "visual": self.visual_btn.isEnabled(),
            "actions": self.actions_toggle_btn.isEnabled(),
        }
        self.loading_overlay.present(
            title,
            message,
            indeterminate=indeterminate,
            progress=progress,
            show_hint=show_hint,
            show_steps=show_steps,
        )
        self.tabs.setEnabled(False)
        self.actions_toggle_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.visual_btn.setEnabled(False)
        if hasattr(self, "actions_panel"):
            self.actions_panel.setEnabled(False)
        QApplication.processEvents()

    def hide_loading(self):
        self.loading_overlay.dismiss()
        self.tabs.setEnabled(True)
        previous = getattr(self, "_loading_ui_state", {})
        self.actions_toggle_btn.setEnabled(previous.get("actions", True))
        self.run_btn.setEnabled(previous.get("run", True))
        self.visual_btn.setEnabled(previous.get("visual", False))
        if hasattr(self, "actions_panel"):
            self.actions_panel.setEnabled(True)
        QApplication.processEvents()

    def update_loading_progress(self, value, message):
        if not self.loading_overlay.isVisible():
            return
        self.loading_overlay.set_progress(value, message)
        QApplication.processEvents()

    def add_annotation_row(self, annotation):
        world = annotation.get("world")
        world_text = f" | world {world}" if world else ""
        index = len(self.workspace.annotations) - 1
        self.annotation_list.addItem(f"{annotation['label']} | voxel {annotation['voxel']}{world_text}")
        self.annotation_list.item(self.annotation_list.count() - 1).setData(Qt.ItemDataRole.UserRole, index)

    def annotation_item_clicked(self, item):
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is None or index < 0 or index >= len(self.workspace.annotations):
            return
        annotation = self.workspace.annotations[index]
        self.workspace.jump_to_voxel(annotation.get("voxel"))
        self.tabs.setCurrentIndex(0)

    def sync_crosshair_toggle(self, checked):
        if checked:
            self.crosshair_toggle_btn.setEnabled(True)
        self.crosshair_toggle_btn.blockSignals(True)
        self.crosshair_toggle_btn.setChecked(checked)
        self.crosshair_toggle_btn.blockSignals(False)

    def toggle_actions(self):
        visible = self.actions_toggle_btn.isChecked()
        self.actions_panel.setVisible(visible)

    def refresh_case_lists(self):
        self.ct_list.blockSignals(True)
        self.seg_list.blockSignals(True)
        self.ct_list.clear()
        self.seg_list.clear()
        for index, record in enumerate(self.ct_records):
            prefix = "* " if index == self.active_ct_index else "  "
            self.ct_list.addItem(f"{prefix}{os.path.basename(record['path'])}")
            self.ct_list.item(self.ct_list.count() - 1).setData(Qt.ItemDataRole.UserRole, index)
        for index, record in enumerate(self.seg_records):
            mapped = record.get("ct_index")
            mapped_name = "unmapped"
            if mapped is not None and 0 <= mapped < len(self.ct_records):
                mapped_name = os.path.basename(self.ct_records[mapped]["path"])
            prefix = "* " if index == self.active_seg_index else "  "
            self.seg_list.addItem(f"{prefix}{os.path.basename(record['path'])} -> {mapped_name}")
            self.seg_list.item(self.seg_list.count() - 1).setData(Qt.ItemDataRole.UserRole, index)
        if self.active_ct_index is not None and self.active_ct_index < self.ct_list.count():
            self.ct_list.setCurrentRow(self.active_ct_index)
        if self.active_seg_index is not None and self.active_seg_index < self.seg_list.count():
            self.seg_list.setCurrentRow(self.active_seg_index)
        self.ct_list.blockSignals(False)
        self.seg_list.blockSignals(False)

    def find_record_by_path(self, records, path):
        target = os.path.normcase(os.path.abspath(path))
        for index, record in enumerate(records):
            if os.path.normcase(os.path.abspath(record["path"])) == target:
                return index
        return None

    def load_ct_from_path(self, path):
        self.show_loading("Loading CT", os.path.basename(path))
        try:
            nii = nib.load(path)
            existing = self.find_record_by_path(self.ct_records, path)
            if existing is None:
                self.ct_records.append({"path": path, "data": nii.get_fdata(), "affine": nii.affine})
                existing = len(self.ct_records) - 1
            self.set_active_ct(existing)
            self.crosshair_toggle_btn.setEnabled(True)
            print(f"SYSTEM: Loaded CT {os.path.basename(path)}")
        finally:
            self.hide_loading()

    def load_seg_from_path(self, path):
        self.show_loading("Loading Segmentation", os.path.basename(path))
        try:
            nii = nib.load(path)
            existing = self.find_record_by_path(self.seg_records, path)
            if existing is None:
                self.seg_records.append({
                    "path": path,
                    "data": nii.get_fdata(),
                    "affine": nii.affine,
                    "ct_index": self.active_ct_index,
                })
                existing = len(self.seg_records) - 1
            elif self.seg_records[existing].get("ct_index") is None:
                self.seg_records[existing]["ct_index"] = self.active_ct_index
            self.set_active_seg(existing)
            try:
                self.verts, self.faces = Worker().build_mesh_from_combined(path)
                print("SYSTEM: 3D segmentation mesh loaded.")
            except Exception as exc:
                print(f"WARNING: Could not build 3D mesh yet: {exc}")
            print(f"SYSTEM: Loaded segmentation {os.path.basename(path)}")
            self.crosshair_toggle_btn.setEnabled(True)
            self.workspace.set_show_crosshair(True)
        finally:
            self.hide_loading()

    def set_active_ct(self, index):
        if index is None or index < 0 or index >= len(self.ct_records):
            return
        self.active_ct_index = index
        record = self.ct_records[index]
        self.ct_path = record["path"]
        self.ct_data = record["data"]
        self.affine = record["affine"]
        mapped = [i for i, seg in enumerate(self.seg_records) if seg.get("ct_index") == index]
        self.active_seg_index = mapped[0] if mapped else None
        if mapped:
            combined = np.zeros_like(self.seg_records[mapped[0]]["data"], dtype=float)
            for seg_index in mapped:
                seg_data = np.asarray(self.seg_records[seg_index]["data"])
                combined = np.where(seg_data > 0, seg_data, combined)
            seg_record = self.seg_records[mapped[0]]
            self.seg_path = seg_record["path"]
            self.combined_seg_path = seg_record["path"]
            self.seg_data = combined
            self.workspace.set_mesh(build_label_meshes(self.seg_data, seg_record["affine"]))
            label = os.path.basename(self.seg_path) if len(mapped) == 1 else f"{len(mapped)} mapped segmentations"
            self.seg_label.setText(f"Segmentation: {label}")
        else:
            self.seg_path = None
            self.combined_seg_path = None
            self.seg_data = None
            self.workspace.set_mesh([])
            self.seg_label.setText("Segmentation: not mapped")
        self.ct_label.setText(f"CT: {os.path.basename(self.ct_path)}")
        self.workspace.set_volumes(self.ct_data, self.seg_data, self.affine)
        self.refresh_case_lists()

    def set_active_seg(self, index):
        if index is None or index < 0 or index >= len(self.seg_records):
            return
        self.active_seg_index = index
        record = self.seg_records[index]
        self.seg_path = record["path"]
        self.combined_seg_path = record["path"]
        self.seg_data = record["data"]
        if self.active_ct_index is None and record.get("ct_index") is not None:
            self.active_ct_index = record["ct_index"]
        if record.get("ct_index") is None:
            record["ct_index"] = self.active_ct_index
        if self.affine is None:
            self.affine = record["affine"]
        if self.active_ct_index is not None:
            self.set_active_ct(self.active_ct_index)
            self.active_seg_index = index
        else:
            self.seg_label.setText(f"Segmentation: {os.path.basename(self.seg_path)}")
            self.workspace.set_volumes(self.ct_data, self.seg_data, self.affine)
            self.workspace.set_mesh(build_label_meshes(self.seg_data, record["affine"]))
        self.refresh_case_lists()

    def ct_item_clicked(self, item):
        self.set_active_ct(item.data(Qt.ItemDataRole.UserRole))

    def seg_item_clicked(self, item):
        self.set_active_seg(item.data(Qt.ItemDataRole.UserRole))

    def map_selected_seg_to_ct(self):
        ct_item = self.ct_list.currentItem()
        seg_item = self.seg_list.currentItem()
        if ct_item is None or seg_item is None:
            print("ERROR: Select both a CT and a segmentation to map.")
            return
        ct_index = ct_item.data(Qt.ItemDataRole.UserRole)
        seg_index = seg_item.data(Qt.ItemDataRole.UserRole)
        if seg_index is None or ct_index is None:
            return
        self.seg_records[seg_index]["ct_index"] = ct_index
        self.set_active_ct(ct_index)
        self.set_active_seg(seg_index)
        print("SYSTEM: Mapped segmentation to selected CT.")

    def remove_selected_ct(self):
        item = self.ct_list.currentItem()
        if item is None:
            return
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is None:
            return
        del self.ct_records[index]
        for seg in self.seg_records:
            mapped = seg.get("ct_index")
            if mapped == index:
                seg["ct_index"] = None
            elif mapped is not None and mapped > index:
                seg["ct_index"] = mapped - 1
        self.active_ct_index = 0 if self.ct_records else None
        self.active_seg_index = None
        if self.active_ct_index is not None:
            self.set_active_ct(self.active_ct_index)
        else:
            self.ct_data = None
            self.seg_data = None
            self.ct_label.setText("CT: not loaded")
            self.seg_label.setText("Segmentation: not loaded")
            self.refresh_case_lists()

    def remove_selected_seg(self):
        item = self.seg_list.currentItem()
        if item is None:
            return
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is None:
            return
        del self.seg_records[index]
        self.active_seg_index = None
        if self.active_ct_index is not None:
            self.set_active_ct(self.active_ct_index)
        else:
            self.seg_data = None
            self.seg_label.setText("Segmentation: not loaded")
            self.refresh_case_lists()

    def load_ct(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CT NIfTI", "", "NIfTI Files (*.nii *.nii.gz)")
        if not path:
            return
        self.load_ct_from_path(path)
        self.status_label.setText("Status: CT loaded")

    def load_seg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Segmentation NIfTI", "", "NIfTI Files (*.nii *.nii.gz)")
        if not path:
            return
        self.load_seg_from_path(path)
        self.status_label.setText("Status: segmentation loaded")

    def run_pipeline(self):
        if not self.ct_path and not self.seg_path:
            print("ERROR: Load a CT or segmented file first.")
            return
        self.table.setRowCount(0)
        self.results = []
        self.workspace.set_results([])
        self.run_btn.setEnabled(False)
        self.visual_btn.setEnabled(False)
        self.show_loading(
            "Running Planning",
            "Preparing segmentation and screw trajectories.",
            indeterminate=False,
            progress=0,
            show_hint=True,
            show_steps=True,
        )
        self.status_label.setText("Status: planning running")
        self.tabs.setCurrentIndex(2)
        self.worker = Worker(self.ct_path, self.seg_path)
        self.worker.screw_found.connect(self.add_table_row)
        self.worker.progress.connect(self.update_loading_progress)
        self.worker.finished.connect(self.finish_pipeline)
        self.worker.failed.connect(self.fail_pipeline)
        self.worker.start()

    def add_table_row(self, data):
        row = self.table.rowCount()
        self.table.insertRow(row)
        entry = data["entry"]
        tip = data["tip"]
        values = [
            data["vertebra"],
            data["side"],
            f"{data['diameter']:.1f}",
            f"{data['length']:.1f}",
            f"{data['axial']:.1f}",
            f"{data['sagittal']:.1f}",
            f"{data.get('min_clearance', 0.0):.1f}",
            f"[{entry[0]:.1f}, {entry[1]:.1f}, {entry[2]:.1f}]",
            f"[{tip[0]:.1f}, {tip[1]:.1f}, {tip[2]:.1f}]",
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, col, item)

    def finish_pipeline(self, verts, faces, results, combined_seg_path):
        self.verts = verts
        self.faces = faces
        self.results = results
        self.combined_seg_path = combined_seg_path
        self.hide_loading()
        self.run_btn.setEnabled(True)
        self.visual_btn.setEnabled(bool(results))
        self.status_label.setText(f"Status: complete ({len(results)} screws)")
        print("COMPLETED: Planning pipeline finished.")
        self.refresh_generated_segmentation(combined_seg_path)
        self.workspace.set_results(self.results)
        self.tabs.setCurrentIndex(0)

    def refresh_generated_segmentation(self, combined_seg_path):
        if not combined_seg_path:
            return
        try:
            nii = nib.load(combined_seg_path)
            existing = self.find_record_by_path(self.seg_records, combined_seg_path)
            if existing is None:
                self.seg_records.append({
                    "path": combined_seg_path,
                    "data": nii.get_fdata(),
                    "affine": nii.affine,
                    "ct_index": self.active_ct_index,
                })
                existing = len(self.seg_records) - 1
            self.set_active_seg(existing)
            self.crosshair_toggle_btn.setEnabled(True)
            self.workspace.set_show_crosshair(True)
        except Exception as exc:
            print(f"WARNING: Could not load segmentation into workspace: {exc}")

    def fail_pipeline(self, message):
        self.hide_loading()
        self.run_btn.setEnabled(True)
        self.visual_btn.setEnabled(False)
        self.status_label.setText("Status: failed")
        print(f"ERROR: {message}")

    def table_row_clicked(self, row, _col):
        if row < 0 or row >= len(self.results):
            return
        entry = np.asarray(self.results[row]["entry"], dtype=float)
        tip = np.asarray(self.results[row]["tip"], dtype=float)
        self.workspace.jump_to_world_point(((entry + tip) / 2.0).tolist())
        self.tabs.setCurrentIndex(0)

    def visualize(self):
        if self.verts is None or self.faces is None or not self.results:
            print("ERROR: Run planning before visualization.")
            return
        try:
            combined_seg = self.combined_seg_path or self.seg_path
            seg_folder = os.path.dirname(combined_seg) if combined_seg else None
            fig, show = visualize_surgical_plan(
                self.verts,
                self.faces,
                self.results,
                volume_path=self.ct_path or self.seg_path,
                segmentation_path=combined_seg,
                seg_folder=seg_folder,
            )
            show()
        except Exception as exc:
            print(f"VISUALIZER ERROR: {exc}")

    def resolve_saved_path(self, saved_path, plan_dir):
        if not saved_path:
            return None
        if os.path.exists(saved_path):
            return saved_path
        candidate = os.path.join(plan_dir, saved_path)
        if os.path.exists(candidate):
            return candidate
        return None

    def load_plan(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load saved plan",
            "",
            "Plan Files (*.json *.csv);;JSON Files (*.json);;CSV Files (*.csv)",
        )
        if not path:
            return
        if path.lower().endswith(".json"):
            self.load_plan_json(path)
        elif path.lower().endswith(".csv"):
            self.load_plan_csv(path)
        else:
            print("ERROR: Unsupported plan file type.")

    def load_plan_json(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        plan_dir = os.path.dirname(path)

        ct_path = self.resolve_saved_path(payload.get("ct_path"), plan_dir)
        seg_path = self.resolve_saved_path(payload.get("segmentation_path"), plan_dir)
        if ct_path:
            self.load_ct_from_path(ct_path)
        elif payload.get("ct_path"):
            print(f"WARNING: CT path not found: {payload.get('ct_path')}")

        if seg_path:
            self.load_seg_from_path(seg_path)
        elif payload.get("segmentation_path"):
            print(f"WARNING: Segmentation path not found: {payload.get('segmentation_path')}")

        self.results = payload.get("results", [])
        if payload.get("annotations"):
            try:
                self.workspace.annotations = self.parse_annotations_payload(payload)
            except ValueError as exc:
                print(f"WARNING: Could not load annotations from plan: {exc}")
                self.workspace.annotations = []
        else:
            self.workspace.annotations = []
        self.refresh_annotation_list()
        self.populate_table_from_results()
        self.workspace.set_results(self.results)
        self.visual_btn.setEnabled(bool(self.results and self.verts is not None and self.faces is not None))
        self.status_label.setText(f"Status: loaded plan ({len(self.results)} screws)")
        self.tabs.setCurrentIndex(0)
        print(f"SYSTEM: Loaded plan JSON from {path}")

    def load_plan_csv(self, path):
        results = []
        with open(path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    entry = ast.literal_eval(row.get("Entry", "[]"))
                    tip = ast.literal_eval(row.get("Tip", "[]"))
                    results.append(
                        {
                            "vertebra": row.get("Vertebra", ""),
                            "side": row.get("Side", ""),
                            "diameter": float(row.get("Diameter mm", 0) or 0),
                            "length": float(row.get("Length mm", 0) or 0),
                            "axial": float(row.get("Axial deg", 0) or 0),
                            "sagittal": float(row.get("Sagittal deg", 0) or 0),
                            "min_clearance": float(row.get("Min clearance", 0) or 0),
                            "entry": entry,
                            "tip": tip,
                        }
                    )
                except Exception as exc:
                    print(f"WARNING: Skipped CSV row because it could not be parsed: {exc}")
        self.results = results
        self.populate_table_from_results()
        self.workspace.set_results(self.results)
        self.visual_btn.setEnabled(bool(self.results and self.verts is not None and self.faces is not None))
        self.status_label.setText(f"Status: loaded CSV plan ({len(self.results)} screws)")
        self.tabs.setCurrentIndex(1)
        print(f"SYSTEM: Loaded plan CSV from {path}")

    def refresh_annotation_list(self):
        self.annotation_list.clear()
        for index, annotation in enumerate(self.workspace.annotations):
            world = annotation.get("world")
            world_text = f" | world {world}" if world else ""
            self.annotation_list.addItem(f"{annotation['label']} | voxel {annotation['voxel']}{world_text}")
            self.annotation_list.item(self.annotation_list.count() - 1).setData(Qt.ItemDataRole.UserRole, index)

    def populate_table_from_results(self):
        self.table.setRowCount(0)
        for result in self.results:
            self.add_table_row(result)

    def save_json(self):
        if not self.results:
            print("ERROR: No plan available to save.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save planning JSON", "screw_plan_v12.json", "JSON Files (*.json)")
        if not file_path:
            return
        payload = {
            "ct_path": self.ct_path,
            "segmentation_path": self.combined_seg_path or self.seg_path,
            "results": self.results,
            "annotations": self.workspace.annotations,
        }
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"SYSTEM: Saved plan JSON to {file_path}")

    def export_csv(self):
        if not self.results:
            print("ERROR: No plan available to export.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export planning CSV", "screw_plan_report_v12.csv", "CSV Files (*.csv)")
        if not file_path:
            return
        with open(file_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Vertebra", "Side", "Diameter mm", "Length mm", "Axial deg", "Sagittal deg", "Min clearance", "Entry", "Tip"])
            for item in self.results:
                writer.writerow([
                    item.get("vertebra", ""),
                    item.get("side", ""),
                    item.get("diameter", ""),
                    item.get("length", ""),
                    item.get("axial", ""),
                    item.get("sagittal", ""),
                    item.get("min_clearance", ""),
                    item.get("entry", ""),
                    item.get("tip", ""),
                ])
        print(f"SYSTEM: Exported report CSV to {file_path}")

    def parse_annotations_payload(self, payload):
        if isinstance(payload, list):
            annotations = payload
        elif isinstance(payload, dict):
            annotations = payload.get("annotations", [])
        else:
            raise ValueError("Annotation file must be a JSON list or an object with an 'annotations' field.")
        if not isinstance(annotations, list):
            raise ValueError("Annotations must be stored as a JSON list.")
        parsed = []
        for index, item in enumerate(annotations):
            if not isinstance(item, dict):
                print(f"WARNING: Skipped annotation row {index + 1} (not an object).")
                continue
            voxel = item.get("voxel")
            if voxel is None or len(voxel) < 3:
                print(f"WARNING: Skipped annotation row {index + 1} (missing voxel).")
                continue
            parsed.append({
                "label": str(item.get("label", f"Annotation {index + 1}")).strip() or f"Annotation {index + 1}",
                "voxel": [int(round(float(voxel[0]))), int(round(float(voxel[1]))), int(round(float(voxel[2])))],
                "world": item.get("world"),
                "segmentation_label": item.get("segmentation_label"),
                "view": item.get("view", ""),
                "created": item.get("created", ""),
            })
        return parsed

    def load_annotations(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load annotations",
            "",
            "Annotation Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            annotations = self.parse_annotations_payload(payload)
            if not annotations:
                print("ERROR: No valid annotations found in file.")
                return
            self.workspace.annotations = annotations
            self.refresh_annotation_list()
            self.workspace.update_mpr_views()
            self.tabs.setCurrentIndex(0)
            print(f"SYSTEM: Loaded {len(annotations)} annotation(s) from {path}")
        except Exception as exc:
            print(f"ERROR: Could not load annotations: {exc}")

    def save_annotations(self):
        if not self.workspace.annotations:
            print("ERROR: No annotations to save.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save annotations", "mpr_annotations_v12.json", "JSON Files (*.json)")
        if not file_path:
            return
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(self.workspace.annotations, handle, indent=2)
        print(f"SYSTEM: Saved annotations to {file_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    splash = QSplashScreen(
        create_splash_pixmap(0.25, "Initializing application...", "Starting up"),
        Qt.WindowType.WindowStaysOnTopHint,
    )
    splash.show()
    app.processEvents()
    splash.setPixmap(
        create_splash_pixmap(0.55, "Loading workspace...", APP_TITLE)
    )
    app.processEvents()
    window = GUI()
    app.processEvents()
    splash.setPixmap(create_splash_pixmap(1.0, "Ready", APP_TITLE))
    app.processEvents()
    time.sleep(0.6)
    window.show()
    splash.finish(window)
    sys.exit(app.exec())
