import copy
import csv
import json
import os
import sys

import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QTabWidget,
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
DEFAULT_MESH_OPACITY = 0.22
VERTEBRA_LEVELS = ["L1", "L2", "L3", "L4", "L5"]
VERTEBRA_LABEL_MAP = {"L1": 5, "L2": 4, "L3": 3, "L4": 2, "L5": 1}
LABEL_TO_VERTEBRA = {value: key for key, value in VERTEBRA_LABEL_MAP.items()}
VERTEBRA_SEG_FILES = {name: f"vertebrae_{name}.nii.gz" for name in VERTEBRA_LEVELS}


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


def mesh_from_binary_mask(mask, affine):
    if np.sum(mask) < 8:
        return None, None
    try:
        verts, faces, _, _ = marching_cubes(mask.astype(np.uint8), level=0.5)
    except ValueError:
        return None, None
    return nib.affines.apply_affine(affine, verts), faces


def resolve_seg_folder(segmentation_path=None, seg_folder=None):
    if seg_folder and os.path.isdir(seg_folder):
        return os.path.abspath(seg_folder)
    if segmentation_path:
        path = os.path.abspath(segmentation_path)
        parent = os.path.dirname(path)
        if os.path.isdir(parent):
            if any(
                os.path.exists(os.path.join(parent, VERTEBRA_SEG_FILES[level]))
                for level in VERTEBRA_LEVELS
            ):
                return parent
            if os.path.basename(parent).lower() == "seg_output":
                return parent
            combined = os.path.join(parent, "combined_seg.nii.gz")
            if os.path.exists(combined):
                return parent
    app_dir = os.path.dirname(os.path.abspath(__file__))
    default_folder = os.path.join(app_dir, "seg_output")
    if os.path.isdir(default_folder):
        return default_folder
    return None


def load_vertebra_meshes(segmentation_path=None, seg_folder=None, seg_data=None, seg_affine=None):
    meshes = {}
    folder = resolve_seg_folder(segmentation_path, seg_folder)

    for level in VERTEBRA_LEVELS:
        verts = faces = None
        if folder:
            file_path = os.path.join(folder, VERTEBRA_SEG_FILES[level])
            if os.path.exists(file_path):
                nii = nib.load(file_path)
                mask = nii.get_fdata() > 0
                verts, faces = mesh_from_binary_mask(mask, nii.affine)

        if verts is None and seg_data is not None and seg_affine is not None:
            label_value = VERTEBRA_LABEL_MAP[level]
            mask = np.asarray(seg_data) == label_value
            verts, faces = mesh_from_binary_mask(mask, seg_affine)

        if verts is not None and faces is not None:
            meshes[level] = {
                "verts": np.asarray(verts, dtype=float),
                "faces": np.asarray(faces, dtype=np.int64),
            }
    return meshes


def results_for_vertebra(results, vertebra):
    return [index for index, result in enumerate(results) if result.get("vertebra", "") == vertebra]


def bbox_for_results(results, indices=None, padding=18.0):
    points = []
    indices = indices if indices is not None else range(len(results))
    for index in indices:
        if index < 0 or index >= len(results):
            continue
        result = results[index]
        points.append(np.asarray(result["entry"], dtype=float))
        points.append(np.asarray(result["tip"], dtype=float))
    if not points:
        return None
    stacked = np.vstack(points)
    mins = stacked.min(axis=0) - padding
    maxs = stacked.max(axis=0) + padding
    return mins, maxs


def crop_mesh_by_bbox(verts, faces, bbox):
    if bbox is None or verts is None or faces is None or len(verts) == 0:
        return None, None
    mins, maxs = bbox
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    inside = np.all((verts >= mins) & (verts <= maxs), axis=1)
    if not inside.any():
        return None, None
    keep_faces = inside[faces].all(axis=1)
    if not keep_faces.any():
        return None, None
    faces = faces[keep_faces]
    used = np.unique(faces.reshape(-1))
    remap = -np.ones(verts.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    return verts[used], remap[faces]


def available_vertebra_choices(results, vertebra_meshes):
    choices = []
    for level in VERTEBRA_LEVELS:
        if level in vertebra_meshes or results_for_vertebra(results, level):
            choices.append(level)
    return choices


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
        return "Warning", "#f59e0b", "Short screw length", None
    if diameter <= 0:
        return "Warning", "#f59e0b", "No valid diameter", None

    if seg_data is None or affine is None:
        return "Unchecked", "#38bdf8", "No segmentation loaded for recheck", None

    inv_affine = np.linalg.inv(affine)
    sample_count = max(20, int(length / 1.0))
    inside_count = 0
    for t in np.linspace(0.0, 1.0, sample_count):
        point = entry + (tip - entry) * t
        if sample_segmentation(seg_data, inv_affine, point) > 0:
            inside_count += 1

    inside_ratio = inside_count / sample_count
    inside_pct = inside_ratio * 100.0
    if inside_ratio > 0.92:
        return "Safe", "#2dd4bf", "Trajectory mostly contained in segmented bone", inside_pct
    if inside_ratio > 0.75:
        return "Caution", "#f59e0b", "Trajectory is close to segmentation boundary", inside_pct
    return "Risk", "#ef4444", "Possible cortical breach or wrong level", inside_pct


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


def format_xyz(point):
    point = np.asarray(point, dtype=float).reshape(-1)
    if point.size < 3:
        return "—"
    return f"[{point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f}]"


def build_scroll_tab(content_widget):
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(content_widget)
    return scroll


class ManualVisualizerWindow(QMainWindow):
    def __init__(
        self,
        verts_world,
        faces,
        results,
        volume_path=None,
        segmentation_path=None,
        seg_folder=None,
    ):
        super().__init__()
        self.verts_world = np.asarray(verts_world, dtype=float)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.original_results = copy.deepcopy(results)
        self.volume_path = volume_path
        self.segmentation_path = segmentation_path
        self.seg_folder = resolve_seg_folder(segmentation_path, seg_folder)
        self.seg_data, self.seg_affine = load_segmentation(segmentation_path)
        self.vertebra_meshes = load_vertebra_meshes(
            segmentation_path=segmentation_path,
            seg_folder=self.seg_folder,
            seg_data=self.seg_data,
            seg_affine=self.seg_affine,
        )
        self.vertebra_choices = available_vertebra_choices(self.original_results, self.vertebra_meshes)
        self.selected_vertebra = None
        self.current_index = 0
        self.mesh_actor_names = []
        self.screw_actor_names = []
        self.screw_index_map = list(range(len(self.original_results)))
        self.plotter = None
        self.mesh_opacity = DEFAULT_MESH_OPACITY
        self.value_labels = {}
        self.level_summary_label = None
        self.mesh_opacity_value_label = None
        self._updating_screw_combo = False
        self.states = [
            {"lr_mm": 0.0, "ud_mm": 0.0, "axial_deg": 0.0, "sagittal_deg": 0.0, "length_mm": 0.0}
            for _ in self.original_results
        ]
        self.adjusted_results = copy.deepcopy(self.original_results)
        self.init_ui()
        self.rebuild_screw_combo()
        self.refresh_scene()
        if self.plotter is not None:
            self.plotter.add_axes(line_width=1, color="#94a3b8")
        self.refresh_screws()
        self.update_status()
        self.update_values_panel()

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
        panel.setFixedWidth(400)
        panel.setStyleSheet(
            "QWidget { background: #20262f; color: #e5e7eb; font-family: Segoe UI; }"
            "QLabel { font-size: 12px; font-weight: 700; color: #d1d5db; }"
            "QGroupBox { border: 1px solid #3f4a59; border-radius: 6px; margin-top: 10px; padding-top: 12px; font-weight: 800; color: #94a3b8; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #2dd4bf; }"
            "QPushButton { background: #0f766e; color: white; border: none; padding: 9px 12px; border-radius: 5px; font-weight: 800; }"
            "QPushButton:hover { background: #14b8a6; }"
            "QPushButton:disabled { background: #475569; color: #94a3b8; }"
            "QComboBox { padding: 7px; border: 1px solid #3f4a59; border-radius: 4px; background: #151a21; color: #f8fafc; }"
            "QSlider::groove:horizontal { height: 5px; background: #3f4a59; border-radius: 2px; }"
            "QSlider::handle:horizontal { background: #2dd4bf; width: 14px; margin: -5px 0; border-radius: 7px; }"
            "QTabWidget::pane { border: 1px solid #3f4a59; background: #1a212b; border-radius: 4px; }"
            "QTabBar::tab { background: #262d37; color: #cbd5e1; padding: 7px 14px; font-weight: 800; border: 1px solid #3f4a59; }"
            "QTabBar::tab:selected { background: #0f766e; color: white; }"
        )
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(8)

        title = QLabel("Manual Screw Adjustment")
        title.setStyleSheet("font-size: 18px; font-weight: 900; color: #f8fafc;")
        panel_layout.addWidget(title)

        panel_layout.addWidget(QLabel("Vertebra view"))
        self.vertebra_combo = QComboBox()
        self.vertebra_combo.addItem("All vertebrae", None)
        for level in self.vertebra_choices:
            self.vertebra_combo.addItem(level, level)
        self.vertebra_combo.currentIndexChanged.connect(self.vertebra_changed)
        panel_layout.addWidget(self.vertebra_combo)

        panel_layout.addWidget(QLabel("Screw"))
        self.screw_combo = QComboBox()
        self.screw_combo.currentIndexChanged.connect(self.screw_combo_changed)
        panel_layout.addWidget(self.screw_combo)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("padding: 10px; border-radius: 5px; background: #334155; color: #f8fafc;")
        panel_layout.addWidget(self.status_label)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Mesh opacity"))
        self.mesh_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.mesh_opacity_slider.setRange(5, 60)
        self.mesh_opacity_slider.setValue(int(DEFAULT_MESH_OPACITY * 100))
        self.mesh_opacity_slider.valueChanged.connect(self.mesh_opacity_changed)
        self.mesh_opacity_value_label = QLabel(f"{int(DEFAULT_MESH_OPACITY * 100)}%")
        self.mesh_opacity_value_label.setFixedWidth(40)
        self.mesh_opacity_value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        opacity_row.addWidget(self.mesh_opacity_slider, 1)
        opacity_row.addWidget(self.mesh_opacity_value_label)
        panel_layout.addLayout(opacity_row)

        sidebar_tabs = QTabWidget()
        sidebar_tabs.setDocumentMode(True)

        plan_host = QWidget()
        values_layout = QVBoxLayout(plan_host)
        values_layout.setContentsMargins(4, 8, 4, 8)
        values_layout.setSpacing(6)

        current_box = QGroupBox("Adjusted screw")
        current_form = QFormLayout(current_box)
        current_form.setContentsMargins(10, 14, 10, 10)
        current_form.setSpacing(6)
        for key, label in [
            ("vertebra", "Vertebra"),
            ("side", "Side"),
            ("entry", "Entry (mm)"),
            ("tip", "Tip (mm)"),
            ("length", "Length (mm)"),
            ("diameter", "Diameter (mm)"),
            ("axial", "Axial (deg)"),
            ("sagittal", "Sagittal (deg)"),
            ("clearance", "Min clearance (mm)"),
        ]:
            row = QLabel("—")
            row.setWordWrap(True)
            row.setStyleSheet("color: #f8fafc; font-weight: 600;")
            self.value_labels[f"adj_{key}"] = row
            current_form.addRow(label, row)

        original_box = QGroupBox("Original plan")
        original_form = QFormLayout(original_box)
        original_form.setContentsMargins(10, 14, 10, 10)
        original_form.setSpacing(6)
        for key, label in [
            ("entry", "Entry (mm)"),
            ("tip", "Tip (mm)"),
            ("length", "Length (mm)"),
            ("diameter", "Diameter (mm)"),
        ]:
            row = QLabel("—")
            row.setWordWrap(True)
            row.setStyleSheet("color: #cbd5e1; font-weight: 600;")
            self.value_labels[f"orig_{key}"] = row
            original_form.addRow(label, row)

        delta_box = QGroupBox("Adjustments")
        delta_form = QFormLayout(delta_box)
        delta_form.setContentsMargins(10, 14, 10, 10)
        delta_form.setSpacing(6)
        for key, label in [
            ("lr_mm", "Left / right (mm)"),
            ("ud_mm", "Up / down (mm)"),
            ("axial_deg", "Axial tilt (deg)"),
            ("sagittal_deg", "Sagittal tilt (deg)"),
            ("length_mm", "Length delta (mm)"),
        ]:
            row = QLabel("0.0")
            row.setStyleSheet("color: #f8fafc; font-weight: 600;")
            self.value_labels[f"delta_{key}"] = row
            delta_form.addRow(label, row)

        self.level_summary_label = QLabel("")
        self.level_summary_label.setWordWrap(True)
        self.level_summary_label.setStyleSheet(
            "padding: 8px; border-radius: 5px; background: #151a21; color: #cbd5e1; font-weight: 600;"
        )
        self.level_summary_label.hide()

        values_layout.addWidget(current_box)
        values_layout.addWidget(original_box)
        values_layout.addWidget(delta_box)
        values_layout.addWidget(self.level_summary_label)
        values_layout.addStretch(1)
        sidebar_tabs.addTab(build_scroll_tab(plan_host), "Plan")

        adjust_host = QWidget()
        adjust_layout = QVBoxLayout(adjust_host)
        adjust_layout.setContentsMargins(4, 8, 4, 8)
        adjust_layout.setSpacing(10)

        self.sliders = {}
        slider_box = QGroupBox("Manual controls")
        grid = QGridLayout(slider_box)
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
        adjust_layout.addWidget(slider_box)

        reset_btn = QPushButton("Reset Selected Screw")
        apply_icon(reset_btn, "fa5s.undo")
        reset_btn.clicked.connect(self.reset_current)
        adjust_layout.addWidget(reset_btn)
        adjust_layout.addStretch(1)
        sidebar_tabs.addTab(build_scroll_tab(adjust_host), "Adjust")

        export_host = QWidget()
        export_layout = QVBoxLayout(export_host)
        export_layout.setContentsMargins(4, 8, 4, 8)
        export_layout.setSpacing(10)

        save_json_btn = QPushButton("Save Plan JSON")
        apply_icon(save_json_btn, "fa5s.save")
        save_json_btn.clicked.connect(self.save_json)
        export_layout.addWidget(save_json_btn)

        export_csv_btn = QPushButton("Export Report CSV")
        apply_icon(export_csv_btn, "fa5s.file-csv")
        export_csv_btn.clicked.connect(self.export_csv)
        export_layout.addWidget(export_csv_btn)

        export_img_btn = QPushButton("Export Screenshot")
        apply_icon(export_img_btn, "fa5s.camera")
        export_img_btn.clicked.connect(self.export_image)
        export_layout.addWidget(export_img_btn)
        export_layout.addStretch(1)
        sidebar_tabs.addTab(build_scroll_tab(export_host), "Export")

        panel_layout.addWidget(sidebar_tabs, 1)

        root.addWidget(panel)
        self.setCentralWidget(container)
        self.resize(1480, 920)

    def mesh_opacity_changed(self, value):
        self.mesh_opacity = value / 100.0
        self.mesh_opacity_value_label.setText(f"{value}%")
        self.refresh_scene()

    def visible_screw_indices(self):
        if self.selected_vertebra is None:
            return list(range(len(self.original_results)))
        return results_for_vertebra(self.original_results, self.selected_vertebra)

    def rebuild_screw_combo(self):
        self._updating_screw_combo = True
        previous_index = self.current_index
        self.screw_combo.blockSignals(True)
        self.screw_combo.clear()
        self.screw_index_map = self.visible_screw_indices()
        for index in self.screw_index_map:
            result = self.original_results[index]
            self.screw_combo.addItem(f"{result.get('vertebra', '')} {result.get('side', '')}")
        if previous_index in self.screw_index_map:
            self.screw_combo.setCurrentIndex(self.screw_index_map.index(previous_index))
        elif self.screw_index_map:
            self.current_index = self.screw_index_map[0]
            self.screw_combo.setCurrentIndex(0)
        self.screw_combo.blockSignals(False)
        self._updating_screw_combo = False
        if self.screw_index_map:
            self.current_index = self.screw_index_map[self.screw_combo.currentIndex()]
        self.load_slider_state()

    def vertebra_changed(self):
        self.selected_vertebra = self.vertebra_combo.currentData()
        self.rebuild_screw_combo()
        self.refresh_scene()
        self.refresh_screws()
        self.update_status()
        self.update_values_panel()
        self.focus_camera()

    def screw_combo_changed(self, combo_index):
        if self._updating_screw_combo:
            return
        if combo_index < 0 or combo_index >= len(self.screw_index_map):
            return
        self.current_index = self.screw_index_map[combo_index]
        self.load_slider_state()
        self.update_status()
        self.update_values_panel()
        self.refresh_screws()

    def mesh_for_vertebra(self, level):
        if level in self.vertebra_meshes:
            mesh = self.vertebra_meshes[level]
            return mesh["verts"], mesh["faces"]
        indices = results_for_vertebra(self.original_results, level)
        bbox = bbox_for_results(self.original_results, indices)
        return crop_mesh_by_bbox(self.verts_world, self.faces, bbox)

    def clear_mesh_actors(self):
        if self.plotter is None:
            return
        for name in self.mesh_actor_names:
            self.plotter.remove_actor(name, reset_camera=False)
        self.mesh_actor_names = []

    def refresh_scene(self):
        if self.plotter is None:
            return
        self.clear_mesh_actors()
        if self.selected_vertebra is None:
            mesh = polydata_from_triangles(self.verts_world, self.faces)
            if mesh is not None:
                self.plotter.add_mesh(
                    mesh,
                    color="#cbd5e1",
                    opacity=self.mesh_opacity,
                    smooth_shading=True,
                    specular=0.25,
                    name="vertebra_mesh_all",
                )
                self.mesh_actor_names.append("vertebra_mesh_all")
        else:
            verts, faces = self.mesh_for_vertebra(self.selected_vertebra)
            mesh = polydata_from_triangles(verts, faces) if verts is not None else None
            if mesh is not None:
                self.plotter.add_mesh(
                    mesh,
                    color="#cbd5e1",
                    opacity=self.mesh_opacity,
                    smooth_shading=True,
                    specular=0.3,
                    name=f"vertebra_mesh_{self.selected_vertebra}",
                )
                self.mesh_actor_names.append(f"vertebra_mesh_{self.selected_vertebra}")

        self.focus_camera()
        self.plotter.render()

    def focus_camera(self):
        if self.plotter is None:
            return
        if self.selected_vertebra is None:
            self.plotter.camera_position = "iso"
            self.plotter.reset_camera()
            return

        verts = None
        if self.selected_vertebra in self.vertebra_meshes:
            verts = self.vertebra_meshes[self.selected_vertebra]["verts"]
        else:
            cropped_verts, _ = self.mesh_for_vertebra(self.selected_vertebra)
            verts = cropped_verts

        indices = results_for_vertebra(self.adjusted_results, self.selected_vertebra)
        bbox = bbox_for_results(self.adjusted_results, indices, padding=12.0)
        points = []
        if verts is not None and len(verts) > 0:
            points.append(np.asarray(verts, dtype=float))
        if bbox is not None:
            mins, maxs = bbox
            corners = np.array(
                [
                    [mins[0], mins[1], mins[2]],
                    [maxs[0], maxs[1], maxs[2]],
                ],
                dtype=float,
            )
            points.append(corners)
        if points:
            stacked = np.vstack(points)
            self.plotter.reset_camera(bounds=(
                stacked[:, 0].min(), stacked[:, 0].max(),
                stacked[:, 1].min(), stacked[:, 1].max(),
                stacked[:, 2].min(), stacked[:, 2].max(),
            ))
        else:
            self.plotter.reset_camera()

    def load_slider_state(self):
        if not self.screw_index_map:
            return
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
        self.update_values_panel()
        self.refresh_screws()

    def nudge_slider(self, slider, delta):
        slider.setValue(int(np.clip(slider.value() + delta, slider.minimum(), slider.maximum())))

    def recompute_results(self):
        self.adjusted_results = [
            adjusted_result(result, state)
            for result, state in zip(self.original_results, self.states)
        ]

    def update_status(self):
        if not self.adjusted_results or self.current_index >= len(self.adjusted_results):
            self.status_label.setText("No screws available.")
            return
        result = self.adjusted_results[self.current_index]
        status, color, message, inside_pct = evaluate_screw_safety(result, self.seg_data, self.seg_affine)
        text = f"{status}: {message}"
        if inside_pct is not None:
            text += f"\nInside bone: {inside_pct:.0f}%"
        self.status_label.setText(text)
        text_color = "#111827" if color in {"#f59e0b", "#2dd4bf", "#38bdf8"} else "#ffffff"
        self.status_label.setStyleSheet(
            f"padding: 10px; border-radius: 5px; background: {color}; color: {text_color}; font-weight: 900;"
        )

    def update_values_panel(self):
        if not self.adjusted_results or self.current_index >= len(self.adjusted_results):
            return
        adjusted = self.adjusted_results[self.current_index]
        original = self.original_results[self.current_index]
        state = self.states[self.current_index]

        self.value_labels["adj_vertebra"].setText(str(adjusted.get("vertebra", "—")))
        self.value_labels["adj_side"].setText(str(adjusted.get("side", "—")))
        self.value_labels["adj_entry"].setText(format_xyz(adjusted.get("entry", [])))
        self.value_labels["adj_tip"].setText(format_xyz(adjusted.get("tip", [])))
        self.value_labels["adj_length"].setText(f"{float(adjusted.get('length', 0.0)):.1f}")
        self.value_labels["adj_diameter"].setText(f"{float(adjusted.get('diameter', 0.0)):.1f}")
        self.value_labels["adj_axial"].setText(f"{float(adjusted.get('axial', 0.0)):.1f}")
        self.value_labels["adj_sagittal"].setText(f"{float(adjusted.get('sagittal', 0.0)):.1f}")
        self.value_labels["adj_clearance"].setText(f"{float(adjusted.get('min_clearance', 0.0)):.1f}")

        self.value_labels["orig_entry"].setText(format_xyz(original.get("entry", [])))
        self.value_labels["orig_tip"].setText(format_xyz(original.get("tip", [])))
        self.value_labels["orig_length"].setText(f"{float(original.get('length', 0.0)):.1f}")
        self.value_labels["orig_diameter"].setText(f"{float(original.get('diameter', 0.0)):.1f}")

        for key in ["lr_mm", "ud_mm", "axial_deg", "sagittal_deg", "length_mm"]:
            self.value_labels[f"delta_{key}"].setText(f"{float(state.get(key, 0.0)):.1f}")

        if self.selected_vertebra is not None:
            lines = []
            for side in ["Left", "Right"]:
                indices = [
                    index
                    for index in results_for_vertebra(self.adjusted_results, self.selected_vertebra)
                    if self.adjusted_results[index].get("side") == side
                ]
                if not indices:
                    lines.append(f"{side}: —")
                    continue
                index = indices[0]
                result = self.adjusted_results[index]
                status, _, _, inside_pct = evaluate_screw_safety(result, self.seg_data, self.seg_affine)
                inside_text = f", {inside_pct:.0f}% inside" if inside_pct is not None else ""
                lines.append(
                    f"{side}: L={float(result.get('length', 0.0)):.1f} mm, "
                    f"Ø={float(result.get('diameter', 0.0)):.1f} mm, {status}{inside_text}"
                )
            self.level_summary_label.setText(
                f"<b>{self.selected_vertebra}</b><br>" + "<br>".join(lines)
            )
            self.level_summary_label.show()
        else:
            self.level_summary_label.hide()

    def refresh_screws(self):
        if self.plotter is None:
            return
        for name in self.screw_actor_names:
            self.plotter.remove_actor(name, reset_camera=False)
        self.screw_actor_names = []

        visible = set(self.visible_screw_indices())
        for index, result in enumerate(self.adjusted_results):
            if index not in visible:
                continue
            entry = np.asarray(result["entry"], dtype=float)
            tip = np.asarray(result["tip"], dtype=float)
            status, color, _, _ = evaluate_screw_safety(result, self.seg_data, self.seg_affine)
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
        self.states[self.current_index] = {
            "lr_mm": 0.0,
            "ud_mm": 0.0,
            "axial_deg": 0.0,
            "sagittal_deg": 0.0,
            "length_mm": 0.0,
        }
        self.recompute_results()
        self.load_slider_state()
        self.update_status()
        self.update_values_panel()
        self.refresh_screws()

    def save_json(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save adjusted plan", "adjusted_screw_plan_v7.json", "JSON Files (*.json)"
        )
        if not file_path:
            return
        payload = {
            "volume_path": self.volume_path,
            "segmentation_path": self.segmentation_path,
            "seg_folder": self.seg_folder,
            "results": self.adjusted_results,
        }
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def export_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export planning report", "screw_plan_report_v7.csv", "CSV Files (*.csv)"
        )
        if not file_path:
            return
        with open(file_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["Vertebra", "Side", "Diameter mm", "Length mm", "Status", "Inside %", "Entry", "Tip", "Adjustments"]
            )
            for result in self.adjusted_results:
                status, _, message, inside_pct = evaluate_screw_safety(result, self.seg_data, self.seg_affine)
                status_text = f"{status}: {message}"
                if inside_pct is not None:
                    status_text += f" ({inside_pct:.0f}% inside)"
                writer.writerow([
                    result.get("vertebra", ""),
                    result.get("side", ""),
                    result.get("diameter", ""),
                    f"{float(result.get('length', 0.0)):.2f}",
                    status_text,
                    "" if inside_pct is None else f"{inside_pct:.1f}",
                    np.asarray(result.get("entry", [])).round(2).tolist(),
                    np.asarray(result.get("tip", [])).round(2).tolist(),
                    result.get("adjustments", {}),
                ])

    def export_image(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save visualization image", "visualization_v7.png", "PNG Files (*.png)"
        )
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


def visualize_surgical_plan(
    vertsWorld,
    faces,
    resultsList,
    volume_path=None,
    segmentation_path=None,
    seg_folder=None,
):
    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)

    window = ManualVisualizerWindow(
        vertsWorld,
        faces,
        resultsList,
        volume_path=volume_path,
        segmentation_path=segmentation_path,
        seg_folder=seg_folder,
    )
    VIEWER_WINDOWS.append(window)

    def show_figure():
        window.show()
        if owns_app:
            app.exec()
        return window

    return None, show_figure
