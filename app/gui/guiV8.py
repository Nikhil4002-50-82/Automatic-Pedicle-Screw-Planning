import csv
import json
import os
import sys
import time
from datetime import datetime

import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes

from PyQt6 import QtCore
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSlider,
    QSplashScreen,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

QtCore.QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: F401

from app.geometry.geometryV5 import (
    computeDistance,
    computeStableFrame,
    getValidLabels,
    loadNifti,
    optimize,
    pedicleCenters,
)
from mesh_builder import build_vertebra_mesh
from run_totalseg import run_totalseg
from visualizerV6 import visualize_surgical_plan


AXES = ["Axial", "Coronal", "Sagittal"]


class LogStream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        if text.strip():
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.newText.emit(f"[{timestamp}] {text.strip()}")

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
    data = np.clip((data - low) / (high - low), 0.0, 1.0)
    return (data * 255).astype(np.uint8)


def label_color(label_value):
    colors = {
        1: np.array([239, 68, 68], dtype=np.uint8),
        2: np.array([245, 158, 11], dtype=np.uint8),
        3: np.array([34, 197, 94], dtype=np.uint8),
        4: np.array([14, 165, 233], dtype=np.uint8),
        5: np.array([168, 85, 247], dtype=np.uint8),
    }
    return colors.get(int(label_value), np.array([236, 72, 153], dtype=np.uint8))


def make_rgb_slice(base_slice=None, seg_slice=None, opacity=0.45, mask_only=False):
    if mask_only:
        if seg_slice is None:
            shape = base_slice.shape if base_slice is not None else (64, 64)
            return np.zeros((*shape, 3), dtype=np.uint8)
        seg = np.asarray(seg_slice)
        rgb = np.zeros((*seg.shape, 3), dtype=np.uint8)
        for label in np.unique(seg[seg > 0]):
            rgb[seg == label] = label_color(label)
        return rgb

    if base_slice is None:
        base_slice = seg_slice if seg_slice is not None else np.zeros((64, 64), dtype=float)
    gray = normalize_slice(base_slice)
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
        return data[:, :, k]
    if axis == "Coronal":
        return data[:, j, :]
    return data[i, :, :]


def axis_dims(shape, axis):
    if axis == "Axial":
        return shape[0], shape[1]
    if axis == "Coronal":
        return shape[0], shape[2]
    return shape[1], shape[2]


def plane_coords(axis, coord):
    i, j, k = coord
    if axis == "Axial":
        return i, j
    if axis == "Coronal":
        return i, k
    return j, k


def coord_from_plane(axis, coord, x_value, y_value):
    i, j, k = coord
    if axis == "Axial":
        return [x_value, y_value, k]
    if axis == "Coronal":
        return [x_value, j, y_value]
    return [i, x_value, y_value]


class SliceView(QLabel):
    clicked = pyqtSignal(str, int, int)

    def __init__(self, axis, title):
        super().__init__("No volume loaded")
        self.axis = axis
        self.title = title
        self.source_shape = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(220, 190)
        self.setStyleSheet("background: #111827; color: #e5e7eb; border-radius: 4px;")
        self.setMouseTracking(True)

    def set_source_shape(self, shape):
        self.source_shape = shape

    def mousePressEvent(self, event):
        if self.source_shape is None or self.pixmap() is None:
            return
        x_vox, y_vox = self.widget_to_voxel(event.position().x(), event.position().y())
        self.clicked.emit(self.axis, x_vox, y_vox)

    def widget_to_voxel(self, x_pos, y_pos):
        dim_x, dim_y = axis_dims(self.source_shape, self.axis)
        pixmap = self.pixmap()
        if pixmap is None:
            return 0, 0
        displayed_w = pixmap.width()
        displayed_h = pixmap.height()
        offset_x = (self.width() - displayed_w) / 2.0
        offset_y = (self.height() - displayed_h) / 2.0
        px = np.clip((x_pos - offset_x) / max(displayed_w, 1), 0.0, 1.0)
        py = np.clip((y_pos - offset_y) / max(displayed_h, 1), 0.0, 1.0)

        x_vox = int(round(py * (dim_x - 1)))
        y_vox = int(round((1.0 - px) * (dim_y - 1)))
        return x_vox, y_vox


class SyncedMPRViewer(QWidget):
    annotation_added = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.ct_data = None
        self.seg_data = None
        self.affine = None
        self.coord = [0, 0, 0]
        self.opacity = 0.45
        self.annotation_mode = False
        self.annotations = []
        self.overlay_views = {}
        self.mask_views = {}
        self.sliders = {}
        self.init_ui()

    def init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        coord_row = QHBoxLayout()
        self.coord_label = QLabel("Coordinate: none")
        self.coord_label.setStyleSheet("font-weight: 800; color: #172554;")
        self.annotation_btn = QPushButton("Add Annotation")
        self.annotation_btn.setCheckable(True)
        self.annotation_btn.clicked.connect(self.toggle_annotation_mode)
        coord_row.addWidget(self.coord_label)
        coord_row.addStretch(1)
        coord_row.addWidget(self.annotation_btn)
        root.addLayout(coord_row)

        overlay_box = QGroupBox("CT + Segmentation Overlay")
        overlay_grid = QGridLayout(overlay_box)
        for col, axis in enumerate(AXES):
            view = SliceView(axis, axis)
            view.clicked.connect(self.handle_view_click)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.valueChanged.connect(lambda value, a=axis: self.slider_changed(a, value))
            overlay_grid.addWidget(QLabel(axis), 0, col)
            overlay_grid.addWidget(view, 1, col)
            overlay_grid.addWidget(slider, 2, col)
            self.overlay_views[axis] = view
            self.sliders[axis] = slider
        root.addWidget(overlay_box, 3)

        mask_box = QGroupBox("Segmentation Masks Only")
        mask_grid = QGridLayout(mask_box)
        for col, axis in enumerate(AXES):
            view = SliceView(axis, f"{axis} mask")
            view.clicked.connect(self.handle_view_click)
            mask_grid.addWidget(QLabel(axis), 0, col)
            mask_grid.addWidget(view, 1, col)
            self.mask_views[axis] = view
        root.addWidget(mask_box, 2)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Overlay opacity"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(45)
        self.opacity_slider.valueChanged.connect(self.set_opacity)
        self.opacity_value = QLabel("45%")
        opacity_row.addWidget(self.opacity_slider)
        opacity_row.addWidget(self.opacity_value)
        root.addLayout(opacity_row)

    def toggle_annotation_mode(self):
        self.annotation_mode = self.annotation_btn.isChecked()
        self.annotation_btn.setText("Click MPR Point" if self.annotation_mode else "Add Annotation")

    def set_volumes(self, ct_data=None, seg_data=None, affine=None):
        self.ct_data = ct_data
        self.seg_data = seg_data
        self.affine = affine
        shape_source = ct_data if ct_data is not None else seg_data
        if shape_source is None:
            return
        shape = shape_source.shape
        self.coord = [shape[0] // 2, shape[1] // 2, shape[2] // 2]
        for view in list(self.overlay_views.values()) + list(self.mask_views.values()):
            view.set_source_shape(shape)
        self.configure_sliders(shape)
        self.update_all_views()

    def configure_sliders(self, shape):
        ranges = {"Sagittal": shape[0] - 1, "Coronal": shape[1] - 1, "Axial": shape[2] - 1}
        values = {"Sagittal": self.coord[0], "Coronal": self.coord[1], "Axial": self.coord[2]}
        for axis, slider in self.sliders.items():
            slider.blockSignals(True)
            slider.setRange(0, max(0, ranges[axis]))
            slider.setValue(values[axis])
            slider.blockSignals(False)

    def set_opacity(self, value):
        self.opacity = value / 100.0
        self.opacity_value.setText(f"{value}%")
        self.update_all_views()

    def slider_changed(self, axis, value):
        if axis == "Axial":
            self.coord[2] = value
        elif axis == "Coronal":
            self.coord[1] = value
        else:
            self.coord[0] = value
        self.update_sliders()
        self.update_all_views()

    def handle_view_click(self, axis, x_value, y_value):
        shape = self.current_shape()
        if shape is None:
            return
        self.coord = coord_from_plane(axis, self.coord, x_value, y_value)
        self.coord = [
            int(np.clip(self.coord[0], 0, shape[0] - 1)),
            int(np.clip(self.coord[1], 0, shape[1] - 1)),
            int(np.clip(self.coord[2], 0, shape[2] - 1)),
        ]
        self.update_sliders()
        self.update_all_views()
        if self.annotation_mode:
            self.create_annotation(axis)

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
            "view": axis,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        self.annotations.append(annotation)
        self.annotation_added.emit(annotation)
        self.annotation_btn.setChecked(False)
        self.toggle_annotation_mode()
        self.update_all_views()

    def current_shape(self):
        source = self.ct_data if self.ct_data is not None else self.seg_data
        return None if source is None else source.shape

    def update_sliders(self):
        values = {"Sagittal": self.coord[0], "Coronal": self.coord[1], "Axial": self.coord[2]}
        for axis, slider in self.sliders.items():
            slider.blockSignals(True)
            slider.setValue(values[axis])
            slider.blockSignals(False)

    def update_all_views(self):
        shape = self.current_shape()
        if shape is None:
            return
        self.coord_label.setText(f"Voxel coordinate: i={self.coord[0]}  j={self.coord[1]}  k={self.coord[2]}")
        for axis in AXES:
            base = slice_for_axis(self.ct_data if self.ct_data is not None else self.seg_data, axis, self.coord)
            seg = slice_for_axis(self.seg_data, axis, self.coord) if self.seg_data is not None else None
            overlay_rgb = make_rgb_slice(base, seg, self.opacity, mask_only=False)
            mask_rgb = make_rgb_slice(base, seg, self.opacity, mask_only=True)
            self.set_view_pixmap(self.overlay_views[axis], overlay_rgb, axis, shape)
            self.set_view_pixmap(self.mask_views[axis], mask_rgb, axis, shape)

    def set_view_pixmap(self, view, rgb, axis, shape):
        rgb = np.ascontiguousarray(np.rot90(rgb))
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image.copy())
        pixmap = self.draw_crosshair_and_annotations(pixmap, axis, shape)
        view.setPixmap(
            pixmap.scaled(
                view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def draw_crosshair_and_annotations(self, pixmap, axis, shape):
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        dim_x, dim_y = axis_dims(shape, axis)
        x_plane, y_plane = plane_coords(axis, self.coord)
        px = (1.0 - (y_plane / max(dim_y - 1, 1))) * pixmap.width()
        py = (x_plane / max(dim_x - 1, 1)) * pixmap.height()

        cross_pen = QPen(QColor("#38bdf8"), 1)
        painter.setPen(cross_pen)
        painter.drawLine(int(px), 0, int(px), pixmap.height())
        painter.drawLine(0, int(py), pixmap.width(), int(py))

        ann_pen = QPen(QColor("#facc15"), 2)
        painter.setPen(ann_pen)
        for annotation in self.annotations:
            voxel = annotation["voxel"]
            if not self.annotation_on_slice(axis, voxel):
                continue
            ax, ay = plane_coords(axis, voxel)
            ann_x = (1.0 - (ay / max(dim_y - 1, 1))) * pixmap.width()
            ann_y = (ax / max(dim_x - 1, 1)) * pixmap.height()
            painter.drawEllipse(int(ann_x) - 5, int(ann_y) - 5, 10, 10)
            painter.drawText(int(ann_x) + 7, int(ann_y) - 7, annotation["label"][:18])
        painter.end()
        return pixmap

    def annotation_on_slice(self, axis, voxel):
        if axis == "Axial":
            return abs(voxel[2] - self.coord[2]) <= 0
        if axis == "Coronal":
            return abs(voxel[1] - self.coord[1]) <= 0
        return abs(voxel[0] - self.coord[0]) <= 0

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_all_views()


class Worker(QThread):
    screw_found = pyqtSignal(dict)
    finished = pyqtSignal(object, object, list, str)
    failed = pyqtSignal(str)

    def __init__(self, ct_path=None, seg_path=None):
        super().__init__()
        self.ct_path = ct_path
        self.seg_path = seg_path

    def build_mesh_from_combined(self, seg_path):
        nii = nib.load(seg_path)
        data = nii.get_fdata()
        affine = nii.affine
        mask = (data > 0).astype(np.uint8)
        if np.sum(mask) == 0:
            raise ValueError("Segmented file is empty. No mesh can be created.")
        verts, faces, _, _ = marching_cubes(mask, level=0.5)
        return nib.affines.apply_affine(affine, verts), faces

    def run(self):
        try:
            if self.seg_path:
                print("MODE: Using loaded segmentation for mesh and planning.")
                combined_path = self.seg_path
                verts_world, faces = self.build_mesh_from_combined(combined_path)
            elif self.ct_path:
                print("MODE: Running TotalSegmentator from CT scan.")
                seg_data = run_totalseg(self.ct_path)
                combined_path = seg_data["combined_seg_path"]
                verts_world, faces = build_vertebra_mesh(seg_data["seg_folder"])
            else:
                raise ValueError("Load a CT scan or segmentation before running planning.")

            print("PLANNING: Calculating screw trajectories.")
            results = []
            label_map = {5: "L1", 4: "L2", 3: "L3", 2: "L4", 1: "L5"}
            seg, spacing, affine = loadNifti(combined_path)
            valid_segments = getValidLabels(seg)

            for label_val, mask in sorted(valid_segments, reverse=True):
                name = label_map.get(label_val, str(label_val))
                centroid, axes, total_depth = computeStableFrame(mask, affine)
                dist = computeDistance(mask, spacing)
                mask_float = mask.astype(float)
                left_center, right_center = pedicleCenters(mask, dist, centroid, axes, affine)
                for side, center in [("Left", left_center), ("Right", right_center)]:
                    res = optimize(center, axes, side, mask_float, dist, affine, centroid, total_depth, name)
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
        self.worker = None
        self.init_ui()

        self.stream = LogStream()
        self.stream.newText.connect(self.update_log)
        sys.stdout = self.stream
        sys.stderr = self.stream

    def init_ui(self):
        self.setWindowTitle("Automatic Pedicle Screw Planning System - V8.0")
        self.resize(1480, 960)

        root = QWidget()
        main = QVBoxLayout(root)
        main.setContentsMargins(16, 16, 16, 16)
        main.setSpacing(10)
        self.setCentralWidget(root)

        title = QLabel("AUTOMATIC PEDICLE SCREW PLANNING SYSTEM V8")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: 900; color: #172554;")
        main.addWidget(title)

        toolbar = QHBoxLayout()
        load_ct_btn = QPushButton("Load CT")
        load_ct_btn.clicked.connect(self.load_ct)
        load_seg_btn = QPushButton("Load Segmentation")
        load_seg_btn.clicked.connect(self.load_seg)
        self.run_btn = QPushButton("Run Planning")
        self.run_btn.clicked.connect(self.run_pipeline)
        self.visual_btn = QPushButton("Launch Manual 3D View")
        self.visual_btn.clicked.connect(self.visualize)
        self.visual_btn.setEnabled(False)
        save_btn = QPushButton("Save Plan JSON")
        save_btn.clicked.connect(self.save_json)
        report_btn = QPushButton("Export CSV Report")
        report_btn.clicked.connect(self.export_csv)
        for button in [load_ct_btn, load_seg_btn, self.run_btn, self.visual_btn, save_btn, report_btn]:
            toolbar.addWidget(button)
        toolbar.addStretch(1)
        main.addLayout(toolbar)

        status_row = QHBoxLayout()
        self.ct_label = QLabel("CT: not loaded")
        self.seg_label = QLabel("Segmentation: not loaded")
        self.status_label = QLabel("Status: ready")
        status_row.addWidget(self.ct_label)
        status_row.addWidget(self.seg_label)
        status_row.addStretch(1)
        status_row.addWidget(self.status_label)
        main.addLayout(status_row)

        self.tabs = QTabWidget()
        main.addWidget(self.tabs, 1)

        self.mpr = SyncedMPRViewer()
        self.mpr.annotation_added.connect(self.add_annotation_row)
        mpr_page = QWidget()
        mpr_layout = QHBoxLayout(mpr_page)
        mpr_layout.setContentsMargins(0, 0, 0, 0)
        mpr_layout.addWidget(self.mpr, 1)
        annotation_panel = QGroupBox("Annotations")
        annotation_layout = QVBoxLayout(annotation_panel)
        self.annotation_list = QListWidget()
        save_ann_btn = QPushButton("Save Annotations JSON")
        save_ann_btn.clicked.connect(self.save_annotations)
        annotation_layout.addWidget(self.annotation_list)
        annotation_layout.addWidget(save_ann_btn)
        annotation_panel.setFixedWidth(320)
        mpr_layout.addWidget(annotation_panel)
        self.tabs.addTab(mpr_page, "MPR Review")

        results_page = QWidget()
        results_layout = QVBoxLayout(results_page)
        self.table = QTableWidget()
        headers = ["Vertebra", "Side", "Diameter (mm)", "Length (mm)", "Axial", "Sagittal", "Clearance", "Entry Point", "Tip Point"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        results_layout.addWidget(self.table)
        self.tabs.addTab(results_page, "Planning Results")

        console_page = QWidget()
        console_layout = QVBoxLayout(console_page)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        console_layout.addWidget(self.log_box)
        self.tabs.addTab(console_page, "Console")

        self.setStyleSheet(
            "QWidget { background: #eef2f7; font-family: Segoe UI; color: #0f172a; }"
            "QPushButton { background: #2563eb; color: white; border: none; padding: 10px 14px; border-radius: 5px; font-weight: 800; }"
            "QPushButton:hover { background: #1d4ed8; }"
            "QPushButton:checked { background: #0f766e; }"
            "QPushButton:disabled { background: #94a3b8; }"
            "QTabWidget::pane { border: 1px solid #cbd5e1; background: #f8fafc; }"
            "QTabBar::tab { background: #dbeafe; padding: 10px 18px; font-weight: 800; }"
            "QTabBar::tab:selected { background: #2563eb; color: white; }"
            "QTableWidget { background: white; color: #0f172a; gridline-color: #cbd5e1; font-size: 13px; }"
            "QHeaderView::section { background: #1e293b; color: white; padding: 8px; font-weight: 800; }"
            "QTextEdit { background: #111827; color: #e5e7eb; font-family: Consolas; font-size: 12px; border-radius: 4px; padding: 8px; }"
            "QGroupBox { border: 1px solid #cbd5e1; border-radius: 5px; margin-top: 8px; padding-top: 10px; font-weight: 800; }"
            "QListWidget { background: white; border: 1px solid #cbd5e1; border-radius: 4px; }"
        )

    def update_log(self, text):
        self.log_box.append(text)
        self.log_box.ensureCursorVisible()

    def add_annotation_row(self, annotation):
        world = annotation.get("world")
        world_text = f" | world {world}" if world else ""
        self.annotation_list.addItem(f"{annotation['label']} | voxel {annotation['voxel']}{world_text}")

    def load_ct(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CT NIfTI", "", "NIfTI Files (*.nii *.nii.gz)")
        if not path:
            return
        self.ct_path = path
        nii = nib.load(path)
        self.ct_data = nii.get_fdata()
        self.affine = nii.affine
        self.ct_label.setText(f"CT: {os.path.basename(path)}")
        self.status_label.setText("Status: CT loaded")
        self.mpr.set_volumes(self.ct_data, self.seg_data, self.affine)
        print(f"SYSTEM: Loaded CT {os.path.basename(path)}")

    def load_seg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Segmentation NIfTI", "", "NIfTI Files (*.nii *.nii.gz)")
        if not path:
            return
        self.seg_path = path
        self.combined_seg_path = path
        nii = nib.load(path)
        self.seg_data = nii.get_fdata()
        if self.affine is None:
            self.affine = nii.affine
        self.seg_label.setText(f"Segmentation: {os.path.basename(path)}")
        self.status_label.setText("Status: segmentation loaded")
        self.mpr.set_volumes(self.ct_data, self.seg_data, self.affine)
        print(f"SYSTEM: Loaded segmentation {os.path.basename(path)}")

    def run_pipeline(self):
        if not self.ct_path and not self.seg_path:
            print("ERROR: Load a CT or segmented file first.")
            return
        self.table.setRowCount(0)
        self.results = []
        self.run_btn.setEnabled(False)
        self.visual_btn.setEnabled(False)
        self.status_label.setText("Status: planning running")
        self.tabs.setCurrentIndex(2)
        self.worker = Worker(self.ct_path, self.seg_path)
        self.worker.screw_found.connect(self.add_table_row)
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
        self.run_btn.setEnabled(True)
        self.visual_btn.setEnabled(bool(results))
        self.status_label.setText(f"Status: complete ({len(results)} screws)")
        self.tabs.setCurrentIndex(1)
        print("COMPLETED: Planning pipeline finished.")

        if combined_seg_path and (self.seg_path != combined_seg_path):
            try:
                nii = nib.load(combined_seg_path)
                self.seg_data = nii.get_fdata()
                self.seg_path = combined_seg_path
                if self.affine is None:
                    self.affine = nii.affine
                self.seg_label.setText(f"Segmentation: {os.path.basename(combined_seg_path)}")
                self.mpr.set_volumes(self.ct_data, self.seg_data, self.affine)
            except Exception as exc:
                print(f"WARNING: Could not load generated segmentation into MPR: {exc}")

    def fail_pipeline(self, message):
        self.run_btn.setEnabled(True)
        self.visual_btn.setEnabled(False)
        self.status_label.setText("Status: failed")
        print(f"ERROR: {message}")

    def visualize(self):
        if self.verts is None or self.faces is None or not self.results:
            print("ERROR: Run planning before visualization.")
            return
        try:
            fig, show = visualize_surgical_plan(
                self.verts,
                self.faces,
                self.results,
                volume_path=self.ct_path or self.seg_path,
                segmentation_path=self.combined_seg_path or self.seg_path,
            )
            show()
        except Exception as exc:
            print(f"VISUALIZER ERROR: {exc}")

    def save_json(self):
        if not self.results:
            print("ERROR: No plan available to save.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save planning JSON", "screw_plan_v8.json", "JSON Files (*.json)")
        if not file_path:
            return
        payload = {
            "ct_path": self.ct_path,
            "segmentation_path": self.combined_seg_path or self.seg_path,
            "results": self.results,
            "annotations": self.mpr.annotations,
        }
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"SYSTEM: Saved plan JSON to {file_path}")

    def export_csv(self):
        if not self.results:
            print("ERROR: No plan available to export.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export planning CSV", "screw_plan_report_v8.csv", "CSV Files (*.csv)")
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

    def save_annotations(self):
        if not self.mpr.annotations:
            print("ERROR: No annotations to save.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save annotations", "mpr_annotations.json", "JSON Files (*.json)")
        if not file_path:
            return
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(self.mpr.annotations, handle, indent=2)
        print(f"SYSTEM: Saved annotations to {file_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pix = QPixmap(460, 200)
    pix.fill(QColor("#1e293b"))
    splash = QSplashScreen(pix)
    splash.show()
    window = GUI()
    time.sleep(0.5)
    window.show()
    splash.finish(window)
    sys.exit(app.exec())
