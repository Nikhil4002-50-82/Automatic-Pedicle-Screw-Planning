import copy
import csv
import json
import sys

import nibabel as nib
import numpy as np

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
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


VIEWER_WINDOWS = []


def icon(name):
    if qta is None:
        return None
    try:
        return qta.icon(name, color="#d9f5f2")
    except Exception:
        return None


def apply_icon(button, name):
    button_icon = icon(name)
    if button_icon is not None:
        try:
            button.setIcon(button_icon)
            button.setIconSize(QSize(16, 16))
            return
        except Exception:
            pass
    fallback_map = {
        "fa5s.undo": QStyle.StandardPixmap.SP_ArrowBack,
        "fa5s.save": QStyle.StandardPixmap.SP_DialogSaveButton,
        "fa5s.file-csv": QStyle.StandardPixmap.SP_FileIcon,
        "fa5s.camera": QStyle.StandardPixmap.SP_ComputerIcon,
    }
    standard_icon = fallback_map.get(name)
    if standard_icon is not None:
        button.setIcon(button.style().standardIcon(standard_icon))
        button.setIconSize(QSize(16, 16))


def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return None
    return vector / norm


def orthonormal_basis(entry, tip):
    axis = normalize(np.asarray(tip, dtype=float) - np.asarray(entry, dtype=float))
    if axis is None:
        return None, None, None

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    n1 = normalize(np.cross(axis, ref))
    if n1 is None:
        return None, None, None
    n2 = normalize(np.cross(axis, n1))
    return axis, n1, n2


def rodrigues_rotate(vector, axis, angle_deg):
    axis = normalize(axis)
    if axis is None:
        return vector
    angle = np.deg2rad(angle_deg)
    vector = np.asarray(vector, dtype=float)
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


def create_cylinder_mesh(entry, tip, diameter, resolution=28):
    entry = np.asarray(entry, dtype=float)
    tip = np.asarray(tip, dtype=float)
    axis, n1, n2 = orthonormal_basis(entry, tip)
    if axis is None or diameter <= 0:
        return None

    radius = diameter / 2.0
    theta = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    ring = np.cos(theta)[:, None] * n1 + np.sin(theta)[:, None] * n2
    vertices = np.vstack((entry + ring * radius, tip + ring * radius, entry, tip))

    idx = np.arange(resolution)
    nxt = (idx + 1) % resolution
    tip_idx = idx + resolution
    tip_nxt = nxt + resolution
    entry_center = 2 * resolution
    tip_center = 2 * resolution + 1

    faces = []
    for a, b, c, d in zip(idx, nxt, tip_nxt, tip_idx):
        faces.append([a, d, c])
        faces.append([a, c, b])
    for a, b in zip(idx, nxt):
        faces.append([entry_center, b, a])
    for a, b in zip(idx, nxt):
        faces.append([tip_center, a + resolution, b + resolution])
    return vertices, np.asarray(faces, dtype=np.int64)


def triangles_to_pyvista_faces(faces):
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return faces
    return np.hstack((np.full((faces.shape[0], 1), 3, dtype=np.int64), faces)).ravel()


def polydata_from_triangles(vertices, faces):
    if pv is None:
        return None
    return pv.PolyData(np.asarray(vertices, dtype=float), triangles_to_pyvista_faces(faces))


def sample_segmentation(seg_data, inv_affine, point):
    voxel = nib.affines.apply_affine(inv_affine, point)
    voxel = np.round(voxel).astype(int)
    if np.any(voxel < 0) or np.any(voxel >= np.asarray(seg_data.shape)):
        return 0
    return seg_data[voxel[0], voxel[1], voxel[2]]


def evaluate_screw_safety(result, seg_data=None, affine=None):
    entry = np.asarray(result["entry"], dtype=float)
    tip = np.asarray(result["tip"], dtype=float)
    length = float(np.linalg.norm(tip - entry))
    diameter = float(result.get("diameter", 0.0) or 0.0)

    if length < 20.0:
        return "Warning", "#f59e0b", "Short screw length"
    if diameter <= 0:
        return "Warning", "#f59e0b", "No valid diameter"

    if seg_data is None or affine is None:
        return "Unchecked", "#38bdf8", "No segmentation loaded for recheck"

    inv_affine = np.linalg.inv(affine)
    sample_count = max(20, int(length / 1.0))
    inside_count = 0
    for t in np.linspace(0.0, 1.0, sample_count):
        point = entry + (tip - entry) * t
        if sample_segmentation(seg_data, inv_affine, point) > 0:
            inside_count += 1

    inside_ratio = inside_count / sample_count
    if inside_ratio > 0.92:
        return "Safe", "#2dd4bf", "Trajectory mostly contained in segmented bone"
    if inside_ratio > 0.75:
        return "Caution", "#f59e0b", "Trajectory is close to segmentation boundary"
    return "Risk", "#ef4444", "Possible cortical breach or wrong level"


def adjusted_result(original, state):
    result = copy.deepcopy(original)
    entry = np.asarray(original["entry"], dtype=float)
    tip = np.asarray(original["tip"], dtype=float)
    axis, lateral, vertical = orthonormal_basis(entry, tip)
    if axis is None:
        return result

    shift = lateral * state["lr_mm"] + vertical * state["ud_mm"]
    new_entry = entry + shift
    direction = tip - entry
    direction = rodrigues_rotate(direction, vertical, state["axial_deg"])
    direction = rodrigues_rotate(direction, lateral, state["sagittal_deg"])
    new_length = max(5.0, np.linalg.norm(direction) + state["length_mm"])
    direction = normalize(direction) * new_length
    new_tip = new_entry + direction

    result["entry"] = new_entry.tolist()
    result["tip"] = new_tip.tolist()
    result["length"] = float(new_length)
    result["adjustments"] = dict(state)
    return result


def load_segmentation(segmentation_path):
    if not segmentation_path:
        return None, None
    nii = nib.load(segmentation_path)
    return nii.get_fdata(), nii.affine


class ManualVisualizerWindow(QMainWindow):
    def __init__(self, verts_world, faces, results, volume_path=None, segmentation_path=None):
        super().__init__()
        self.verts_world = np.asarray(verts_world, dtype=float)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.original_results = copy.deepcopy(results)
        self.volume_path = volume_path
        self.segmentation_path = segmentation_path
        self.seg_data, self.seg_affine = load_segmentation(segmentation_path)
        self.current_index = 0
        self.mesh_actor = None
        self.screw_actor_names = []
        self.plotter = None
        self.states = [
            {"lr_mm": 0.0, "ud_mm": 0.0, "axial_deg": 0.0, "sagittal_deg": 0.0, "length_mm": 0.0}
            for _ in self.original_results
        ]
        self.adjusted_results = copy.deepcopy(self.original_results)
        self.init_ui()
        self.load_static_scene()
        self.refresh_screws()
        self.update_status()

    def init_ui(self):
        self.setWindowTitle("Manual Screw Visualizer V7")
        container = QWidget()
        root = QHBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        if QtInteractor is None:
            missing = QLabel("VTK/PyVista is not available in this Python environment.")
            missing.setAlignment(Qt.AlignmentFlag.AlignCenter)
            missing.setStyleSheet("background: #171b22; color: #f59e0b; font-size: 16px; font-weight: 800;")
            root.addWidget(missing, 1)
        else:
            self.plotter = QtInteractor(container)
            self.plotter.set_background("#151a21")
            root.addWidget(self.plotter.interactor, 1)

        panel = QWidget()
        panel.setFixedWidth(360)
        panel.setStyleSheet(
            "QWidget { background: #20262f; color: #e5e7eb; font-family: Segoe UI; }"
            "QLabel { font-size: 12px; font-weight: 700; color: #d1d5db; }"
            "QPushButton { background: #0f766e; color: white; border: none; padding: 9px 12px; border-radius: 5px; font-weight: 800; }"
            "QPushButton:hover { background: #14b8a6; }"
            "QPushButton:disabled { background: #475569; color: #94a3b8; }"
            "QComboBox { padding: 7px; border: 1px solid #3f4a59; border-radius: 4px; background: #151a21; color: #f8fafc; }"
            "QSlider::groove:horizontal { height: 5px; background: #3f4a59; border-radius: 2px; }"
            "QSlider::handle:horizontal { background: #2dd4bf; width: 14px; margin: -5px 0; border-radius: 7px; }"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Manual Screw Adjustment")
        title.setStyleSheet("font-size: 18px; font-weight: 900; color: #f8fafc;")
        layout.addWidget(title)

        self.screw_combo = QComboBox()
        for result in self.original_results:
            self.screw_combo.addItem(f"{result.get('vertebra', '')} {result.get('side', '')}")
        self.screw_combo.currentIndexChanged.connect(self.select_screw)
        layout.addWidget(self.screw_combo)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("padding: 10px; border-radius: 5px; background: #334155; color: #f8fafc;")
        layout.addWidget(self.status_label)

        self.sliders = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        controls = [
            ("lr_mm", "Left / Right (0.5 mm)", -20, 20),
            ("ud_mm", "Up / Down (0.5 mm)", -20, 20),
            ("axial_deg", "Axial Tilt (0.5 deg)", -40, 40),
            ("sagittal_deg", "Sagittal Tilt (0.5 deg)", -30, 30),
            ("length_mm", "Length Change (0.5 mm)", -30, 30),
        ]
        for row, (key, label, min_val, max_val) in enumerate(controls):
            text = QLabel(label)
            value = QLabel("0.0")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(0)
            slider.valueChanged.connect(lambda _, k=key: self.slider_changed(k))
            minus_btn = QPushButton("-")
            plus_btn = QPushButton("+")
            minus_btn.setFixedWidth(32)
            plus_btn.setFixedWidth(32)
            minus_btn.clicked.connect(lambda _, s=slider: self.nudge_slider(s, -1))
            plus_btn.clicked.connect(lambda _, s=slider: self.nudge_slider(s, 1))
            self.sliders[key] = (slider, value)
            grid.addWidget(text, row * 2, 0)
            grid.addWidget(value, row * 2, 1)
            grid.addWidget(slider, row * 2 + 1, 0, 1, 2)
            grid.addWidget(minus_btn, row * 2 + 1, 2)
            grid.addWidget(plus_btn, row * 2 + 1, 3)
        layout.addLayout(grid)

        reset_btn = QPushButton("Reset Selected Screw")
        apply_icon(reset_btn, "fa5s.undo")
        reset_btn.clicked.connect(self.reset_current)
        layout.addWidget(reset_btn)

        save_json_btn = QPushButton("Save Plan JSON")
        apply_icon(save_json_btn, "fa5s.save")
        save_json_btn.clicked.connect(self.save_json)
        layout.addWidget(save_json_btn)

        export_csv_btn = QPushButton("Export Report CSV")
        apply_icon(export_csv_btn, "fa5s.file-csv")
        export_csv_btn.clicked.connect(self.export_csv)
        layout.addWidget(export_csv_btn)

        export_img_btn = QPushButton("Export Screenshot")
        apply_icon(export_img_btn, "fa5s.camera")
        export_img_btn.clicked.connect(self.export_image)
        layout.addWidget(export_img_btn)

        layout.addStretch(1)
        root.addWidget(panel)
        self.setCentralWidget(container)
        self.resize(1420, 920)

    def load_static_scene(self):
        if self.plotter is None:
            return
        mesh = polydata_from_triangles(self.verts_world, self.faces)
        if mesh is not None:
            self.mesh_actor = self.plotter.add_mesh(
                mesh,
                color="#cbd5e1",
                opacity=0.22,
                smooth_shading=True,
                specular=0.25,
                name="vertebra_mesh",
            )
        self.plotter.add_axes(line_width=1, color="#94a3b8")
        self.plotter.camera_position = "iso"
        self.plotter.reset_camera()

    def select_screw(self, index):
        self.current_index = max(0, index)
        self.load_slider_state()
        self.update_status()

    def load_slider_state(self):
        state = self.states[self.current_index]
        for key, (slider, value_label) in self.sliders.items():
            slider.blockSignals(True)
            slider.setValue(int(round(state[key] * 2.0)))
            slider.blockSignals(False)
            value_label.setText(f"{state[key]:.1f}")

    def slider_changed(self, key):
        slider, value_label = self.sliders[key]
        value = slider.value() / 2.0
        self.states[self.current_index][key] = value
        value_label.setText(f"{value:.1f}")
        self.recompute_results()
        self.update_status()
        self.refresh_screws()

    def nudge_slider(self, slider, delta):
        slider.setValue(int(np.clip(slider.value() + delta, slider.minimum(), slider.maximum())))

    def recompute_results(self):
        self.adjusted_results = [
            adjusted_result(result, state)
            for result, state in zip(self.original_results, self.states)
        ]

    def update_status(self):
        if not self.adjusted_results:
            self.status_label.setText("No screws available.")
            return
        result = self.adjusted_results[self.current_index]
        status, color, message = evaluate_screw_safety(result, self.seg_data, self.seg_affine)
        self.status_label.setText(f"{status}: {message}")
        text_color = "#111827" if color in {"#f59e0b", "#2dd4bf", "#38bdf8"} else "#ffffff"
        self.status_label.setStyleSheet(
            f"padding: 10px; border-radius: 5px; background: {color}; color: {text_color}; font-weight: 900;"
        )

    def refresh_screws(self):
        if self.plotter is None:
            return
        for name in self.screw_actor_names:
            self.plotter.remove_actor(name, reset_camera=False)
        self.screw_actor_names = []

        for index, result in enumerate(self.adjusted_results):
            entry = np.asarray(result["entry"], dtype=float)
            tip = np.asarray(result["tip"], dtype=float)
            status, color, _ = evaluate_screw_safety(result, self.seg_data, self.seg_affine)
            diameter = float(result.get("diameter", 5.5) or 5.5)
            line_name = f"screw_path_{index}"
            entry_name = f"screw_entry_{index}"
            screw_name = f"screw_body_{index}"
            selected = index == self.current_index

            line = pv.Line(entry, tip) if pv is not None else None
            if line is not None:
                self.plotter.add_mesh(line, color=color, line_width=5 if selected else 3, name=line_name)
                self.screw_actor_names.append(line_name)

            marker = pv.Sphere(radius=diameter * 0.42, center=entry) if pv is not None else None
            if marker is not None:
                self.plotter.add_mesh(marker, color="#2dd4bf", name=entry_name)
                self.screw_actor_names.append(entry_name)

            cylinder = create_cylinder_mesh(entry, tip, diameter)
            if cylinder is not None:
                vertices, faces = cylinder
                screw_mesh = polydata_from_triangles(vertices, faces)
                self.plotter.add_mesh(
                    screw_mesh,
                    color=color,
                    opacity=0.85 if selected else 0.55,
                    smooth_shading=True,
                    specular=0.35,
                    name=screw_name,
                )
                self.screw_actor_names.append(screw_name)

        self.plotter.render()

    def reset_current(self):
        self.states[self.current_index] = {"lr_mm": 0.0, "ud_mm": 0.0, "axial_deg": 0.0, "sagittal_deg": 0.0, "length_mm": 0.0}
        self.recompute_results()
        self.load_slider_state()
        self.update_status()
        self.refresh_screws()

    def save_json(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save adjusted plan", "adjusted_screw_plan_v7.json", "JSON Files (*.json)")
        if not file_path:
            return
        payload = {
            "volume_path": self.volume_path,
            "segmentation_path": self.segmentation_path,
            "results": self.adjusted_results,
        }
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def export_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export planning report", "screw_plan_report_v7.csv", "CSV Files (*.csv)")
        if not file_path:
            return
        with open(file_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Vertebra", "Side", "Diameter mm", "Length mm", "Status", "Entry", "Tip", "Adjustments"])
            for result in self.adjusted_results:
                status, _, message = evaluate_screw_safety(result, self.seg_data, self.seg_affine)
                writer.writerow([
                    result.get("vertebra", ""),
                    result.get("side", ""),
                    result.get("diameter", ""),
                    f"{float(result.get('length', 0.0)):.2f}",
                    f"{status}: {message}",
                    np.asarray(result.get("entry", [])).round(2).tolist(),
                    np.asarray(result.get("tip", [])).round(2).tolist(),
                    result.get("adjustments", {}),
                ])

    def export_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save visualization image", "visualization_v7.png", "PNG Files (*.png)")
        if not file_path:
            return
        if self.plotter is not None:
            self.plotter.screenshot(file_path)
        else:
            self.grab().save(file_path)

    def closeEvent(self, event):
        if self.plotter is not None:
            self.plotter.close()
        super().closeEvent(event)


def visualize_surgical_plan(vertsWorld, faces, resultsList, volume_path=None, segmentation_path=None):
    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)

    window = ManualVisualizerWindow(vertsWorld, faces, resultsList, volume_path, segmentation_path)
    VIEWER_WINDOWS.append(window)

    def show_figure():
        window.show()
        if owns_app:
            app.exec()
        return window

    return None, show_figure
