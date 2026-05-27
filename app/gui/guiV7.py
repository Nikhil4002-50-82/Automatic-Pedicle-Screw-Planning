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
from PyQt6.QtGui import QColor, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSplashScreen,
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


def build_overlay_image(base_slice, seg_slice=None, opacity=0.45):
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

    rgb = np.ascontiguousarray(np.rot90(rgb))
    height, width, _ = rgb.shape
    image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class MPRViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.ct_data = None
        self.seg_data = None
        self.opacity = 0.45
        self.labels = {}
        self.sliders = {}
        self.init_ui()

    def init_ui(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        for col, axis in enumerate(["Axial", "Coronal", "Sagittal"]):
            box = QGroupBox(axis)
            box.setStyleSheet("QGroupBox { font-weight: 800; color: #0f172a; }")
            box_layout = QVBoxLayout(box)
            label = QLabel("No volume loaded")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(260, 220)
            label.setStyleSheet("background: #111827; color: #e5e7eb; border-radius: 4px;")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.valueChanged.connect(self.update_views)
            box_layout.addWidget(label)
            box_layout.addWidget(slider)
            self.labels[axis] = label
            self.sliders[axis] = slider
            layout.addWidget(box, 0, col)

        opacity_box = QGroupBox("Segmentation Opacity")
        opacity_layout = QHBoxLayout(opacity_box)
        self.opacity_label = QLabel("45%")
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(45)
        self.opacity_slider.valueChanged.connect(self.set_opacity)
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addWidget(self.opacity_label)
        layout.addWidget(opacity_box, 1, 0, 1, 3)

    def set_volumes(self, ct_data=None, seg_data=None):
        self.ct_data = ct_data
        self.seg_data = seg_data
        shape_source = ct_data if ct_data is not None else seg_data
        if shape_source is None:
            return

        shape = shape_source.shape
        ranges = {
            "Axial": shape[2] - 1,
            "Coronal": shape[1] - 1,
            "Sagittal": shape[0] - 1,
        }
        starts = {
            "Axial": shape[2] // 2,
            "Coronal": shape[1] // 2,
            "Sagittal": shape[0] // 2,
        }
        for axis, slider in self.sliders.items():
            slider.blockSignals(True)
            slider.setRange(0, max(0, ranges[axis]))
            slider.setValue(starts[axis])
            slider.blockSignals(False)
        self.update_views()

    def set_opacity(self, value):
        self.opacity = value / 100.0
        self.opacity_label.setText(f"{value}%")
        self.update_views()

    def get_slice(self, data, axis, index):
        if data is None:
            return None
        if axis == "Axial":
            return data[:, :, index]
        if axis == "Coronal":
            return data[:, index, :]
        return data[index, :, :]

    def update_views(self):
        base_data = self.ct_data if self.ct_data is not None else self.seg_data
        if base_data is None:
            return

        for axis, label in self.labels.items():
            index = self.sliders[axis].value()
            base = self.get_slice(base_data, axis, index)
            seg = self.get_slice(self.seg_data, axis, index) if self.seg_data is not None else None
            pixmap = build_overlay_image(base, seg, self.opacity)
            label.setPixmap(
                pixmap.scaled(
                    label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_views()


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
        verts_world = nib.affines.apply_affine(affine, verts)
        return verts_world, faces

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
        self.setWindowTitle("Automatic Pedicle Screw Planning System - V7.0")
        self.resize(1420, 920)

        root = QWidget()
        main = QVBoxLayout(root)
        main.setContentsMargins(18, 18, 18, 18)
        main.setSpacing(12)
        self.setCentralWidget(root)

        title = QLabel("AUTOMATIC PEDICLE SCREW PLANNING SYSTEM V7")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: 900; color: #172554;")
        main.addWidget(title)

        controls = QHBoxLayout()
        self.ct_label = QLabel("CT: not loaded")
        self.seg_label = QLabel("Segmentation: not loaded")
        self.status_label = QLabel("Status: ready")

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

        for widget in [load_ct_btn, load_seg_btn, self.run_btn, self.visual_btn, save_btn, report_btn]:
            controls.addWidget(widget)
        controls.addStretch(1)
        main.addLayout(controls)

        file_row = QHBoxLayout()
        file_row.addWidget(self.ct_label)
        file_row.addWidget(self.seg_label)
        file_row.addStretch(1)
        file_row.addWidget(self.status_label)
        main.addLayout(file_row)

        self.mpr = MPRViewer()
        main.addWidget(self.mpr, 2)

        self.table = QTableWidget()
        headers = ["Vertebra", "Side", "Diameter (mm)", "Length (mm)", "Axial", "Sagittal", "Clearance", "Entry Point"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        main.addWidget(self.table, 1)

        console_label = QLabel("System Console")
        console_label.setStyleSheet("font-weight: 800;")
        main.addWidget(console_label)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(150)
        main.addWidget(self.log_box)

        self.setStyleSheet(
            "QWidget { background: #eef2f7; font-family: Segoe UI; color: #0f172a; }"
            "QPushButton { background: #2563eb; color: white; border: none; padding: 10px 14px; border-radius: 5px; font-weight: 800; }"
            "QPushButton:hover { background: #1d4ed8; }"
            "QPushButton:disabled { background: #94a3b8; }"
            "QTableWidget { background: white; color: #0f172a; gridline-color: #cbd5e1; }"
            "QHeaderView::section { background: #1e293b; color: white; padding: 8px; font-weight: 800; }"
            "QTextEdit { background: #111827; color: #e5e7eb; font-family: Consolas; font-size: 12px; border-radius: 4px; padding: 8px; }"
            "QGroupBox { border: 1px solid #cbd5e1; border-radius: 5px; margin-top: 8px; padding-top: 10px; }"
        )

    def update_log(self, text):
        self.log_box.append(text)
        self.log_box.ensureCursorVisible()

    def load_ct(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CT NIfTI", "", "NIfTI Files (*.nii *.nii.gz)")
        if not path:
            return
        self.ct_path = path
        nii = nib.load(path)
        self.ct_data = nii.get_fdata()
        self.ct_label.setText(f"CT: {os.path.basename(path)}")
        self.status_label.setText("Status: CT loaded")
        self.mpr.set_volumes(self.ct_data, self.seg_data)
        print(f"SYSTEM: Loaded CT {os.path.basename(path)}")

    def load_seg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Segmentation NIfTI", "", "NIfTI Files (*.nii *.nii.gz)")
        if not path:
            return
        self.seg_path = path
        self.combined_seg_path = path
        nii = nib.load(path)
        self.seg_data = nii.get_fdata()
        self.seg_label.setText(f"Segmentation: {os.path.basename(path)}")
        self.status_label.setText("Status: segmentation loaded")
        self.mpr.set_volumes(self.ct_data, self.seg_data)
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
        self.worker = Worker(self.ct_path, self.seg_path)
        self.worker.screw_found.connect(self.add_table_row)
        self.worker.finished.connect(self.finish_pipeline)
        self.worker.failed.connect(self.fail_pipeline)
        self.worker.start()

    def add_table_row(self, data):
        row = self.table.rowCount()
        self.table.insertRow(row)
        entry = data["entry"]
        values = [
            data["vertebra"],
            data["side"],
            f"{data['diameter']:.1f}",
            f"{data['length']:.1f}",
            f"{data['axial']:.1f}",
            f"{data['sagittal']:.1f}",
            f"{data.get('min_clearance', 0.0):.1f}",
            f"[{entry[0]:.1f}, {entry[1]:.1f}, {entry[2]:.1f}]",
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
        print("COMPLETED: Planning pipeline finished.")

        if combined_seg_path and (self.seg_path != combined_seg_path):
            try:
                nii = nib.load(combined_seg_path)
                self.seg_data = nii.get_fdata()
                self.seg_path = combined_seg_path
                self.seg_label.setText(f"Segmentation: {os.path.basename(combined_seg_path)}")
                self.mpr.set_volumes(self.ct_data, self.seg_data)
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
        file_path, _ = QFileDialog.getSaveFileName(self, "Save planning JSON", "screw_plan_v7.json", "JSON Files (*.json)")
        if not file_path:
            return
        payload = {
            "ct_path": self.ct_path,
            "segmentation_path": self.combined_seg_path or self.seg_path,
            "results": self.results,
        }
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"SYSTEM: Saved plan JSON to {file_path}")

    def export_csv(self):
        if not self.results:
            print("ERROR: No plan available to export.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export planning CSV", "screw_plan_report_v7.csv", "CSV Files (*.csv)")
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pix = QPixmap(440, 200)
    pix.fill(QColor("#1e293b"))
    splash = QSplashScreen(pix)
    splash.show()
    window = GUI()
    time.sleep(0.5)
    window.show()
    splash.finish(window)
    sys.exit(app.exec())
