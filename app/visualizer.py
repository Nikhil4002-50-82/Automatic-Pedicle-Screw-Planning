import copy
import csv
import json
import os
import sys

import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QImage, QPainter, QPen, QPixmap, QColor
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
    QProgressBar,
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


def load_and_transform_screw(entry, tip, diameter, length, catalog_type="Generic"):
    """
    Procedurally creates a highly realistic medical pedicle screw featuring:
    1. Polyaxial Tool Head Assembly (Spherical base joint + Slotted Tulip housing)
    2. Varying Core Tapering (Thicker proximal pedicle zone transitioning to a tapered body)
    3. Helical Thread Pitch simulation via close-interval micro-crests
    """
    if pv is None:
        return None

    entry = np.asarray(entry, dtype=float)
    tip = np.asarray(tip, dtype=float)
    axis = normalize(tip - entry)
    if axis is None:
        return None

    # --- 1. HEAD ASSEMBLY (POLYAXIAL TULIP + SPHERE) ---
    if catalog_type == "Stryker":
        thread_pitch = 1.5
        thread_outward_depth = diameter * 0.14
        head_diameter = diameter * 1.5
        tulip_length = 8.5
    elif catalog_type == "Medtronic":
        thread_pitch = 1.75
        thread_outward_depth = diameter * 0.12
        head_diameter = diameter * 1.6
        tulip_length = 10.0
    elif catalog_type == "Globus":
        thread_pitch = 2.0
        thread_outward_depth = diameter * 0.16
        head_diameter = diameter * 1.7
        tulip_length = 11.5
    else:
        thread_pitch = 1.75
        thread_outward_depth = diameter * 0.12
        head_diameter = diameter * 1.6
        tulip_length = 10.0

    sphere_radius = (diameter * 1.3) / 2.0
    
    # Spherical polyaxial base exactly at entry point
    head_sphere = pv.Sphere(radius=sphere_radius, center=entry, theta_resolution=24, phi_resolution=24)
    
    # Slotted Tulip housing sitting behind entry point
    tulip_center = entry - (axis * (tulip_length / 2.0))
    head_tulip = pv.Cylinder(
        center=tulip_center,
        direction=axis,
        radius=head_diameter / 2.0,
        height=tulip_length,
        resolution=32
    )
    combined_hardware = head_sphere.merge(head_tulip)

    # --- 2. TAPERED SHAFT & CORE TIMING ---
    taper_length = diameter * 1.2
    bone_shaft_length = length - taper_length
    if bone_shaft_length < 2.0:
        bone_shaft_length = length * 0.75
        taper_length = length * 0.25

    # Proximal pedicle zone (first 35% of shaft length is slightly thicker)
    pedicle_zone_len = bone_shaft_length * 0.35
    body_zone_len = bone_shaft_length - pedicle_zone_len

    proximal_radius = (diameter * 1.08) / 2.0
    distal_radius = diameter / 2.0

    # Proximal cylindrical core
    prox_center = entry + (axis * (pedicle_zone_len / 2.0))
    prox_cyl = pv.Cylinder(center=prox_center, direction=axis, radius=proximal_radius, height=pedicle_zone_len, resolution=32)
    combined_hardware = combined_hardware.merge(prox_cyl)

    # Distal core (transitioning from proximal to distal base radius)
    dist_center = entry + (axis * (pedicle_zone_len + body_zone_len / 2.0))
    dist_cyl = pv.Cylinder(center=dist_center, direction=axis, radius=distal_radius, height=body_zone_len, resolution=32)
    combined_hardware = combined_hardware.merge(dist_cyl)

    # Sharp self-tapping terminal tip cone
    cone_base_center = entry + (axis * bone_shaft_length)
    tip_cone = pv.Cone(
        center=cone_base_center + (axis * (taper_length / 2.0)),
        direction=axis,
        radius=distal_radius,
        height=taper_length,
        resolution=32
    )
    combined_hardware = combined_hardware.merge(tip_cone)

    # --- 3. PROCEDURAL HELICAL THREAD PITCH ---
    # Stack high-frequency micro-rings along the bone shaft to simulate realistic thread crests
    num_threads = int(bone_shaft_length / thread_pitch)
    thread_crest_thickness = 0.45

    for i in range(num_threads):
        distance_along = pedicle_zone_len * 0.3 + (i * thread_pitch)
        if distance_along >= bone_shaft_length - 1.0:
            break
            
        ring_center = entry + (axis * distance_along)
        
        # Determine appropriate core radius baseline for thread scaling
        current_base_rad = proximal_radius if distance_along <= pedicle_zone_len else distal_radius
        outer_thread_rad = current_base_rad + thread_outward_depth

        # Create a single crisp thread crest ring
        thread_ring = pv.Cylinder(
            center=ring_center,
            direction=axis,
            radius=outer_thread_rad,
            height=thread_crest_thickness,
            resolution=32
        )
        combined_hardware = combined_hardware.merge(thread_ring)

    return combined_hardware


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
    if "diameter" in state:
        result["diameter"] = float(state["diameter"])
        
    result["adjustments"] = {
        "lr_mm": float(state["lr_mm"]),
        "ud_mm": float(state["ud_mm"]),
        "axial_deg": float(state["axial_deg"]),
        "sagittal_deg": float(state["sagittal_deg"]),
        "length_mm": float(state["length_mm"]),
    }
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
    plan_adjusted = pyqtSignal(list)

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

        # Load CT Volume
        self.volume_data = None
        self.volume_affine = None
        if self.volume_path and os.path.exists(self.volume_path):
            try:
                self.volume_nii = nib.load(self.volume_path)
                self.volume_data = self.volume_nii.get_fdata()
                self.volume_affine = self.volume_nii.affine
            except Exception as e:
                print(f"Error loading volume in visualizer: {e}")

        self.states = [
            {
                "lr_mm": 0.0,
                "ud_mm": 0.0,
                "axial_deg": 0.0,
                "sagittal_deg": 0.0,
                "length_mm": 0.0,
                "diameter": float(res.get("diameter", 5.5) or 5.5),
                "catalog_type": "Generic",
            }
            for res in self.original_results
        ]
        self.adjusted_results = copy.deepcopy(self.original_results)
        self.init_ui()
        self.rebuild_screw_combo()
        self.refresh_scene()
        if self.plotter is not None:
            self.plotter.add_axes(line_width=1, color="#94a3b8")
        self.refresh_screws(force_all=True)
        self.update_status()
        self.update_values_panel()

    def init_ui(self):
        self.setWindowTitle("Manual Screw Visualizer V8 (Procedural Realism)")
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
            self.plotter = None
            self._current_plotter_shape = None

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

        # Bone Quality Sidebar Tab (Relocated HUD dashboard)
        bone_quality_host = QWidget()
        bq_tab_layout = QVBoxLayout(bone_quality_host)
        bq_tab_layout.setContentsMargins(4, 8, 4, 8)
        bq_tab_layout.setSpacing(10)

        # 1. Bone Quality Classification Badge
        self.bone_quality_badge = QLabel("CLASSIFYING...")
        self.bone_quality_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bone_quality_badge.setStyleSheet(
            "font-size: 13px; font-weight: 900; color: white; background: #3b4252; padding: 10px; border-radius: 6px;"
        )
        bq_tab_layout.addWidget(self.bone_quality_badge)

        # 2. Hounsfield Units Progress Group
        hu_group = QWidget()
        hu_layout = QVBoxLayout(hu_group)
        hu_layout.setContentsMargins(0, 0, 0, 0)
        hu_layout.setSpacing(4)
        hu_lbl = QLabel("Bone Density (Hounsfield Units)")
        hu_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #cbd5e1;")
        hu_layout.addWidget(hu_lbl)
        self.hu_progress = QProgressBar()
        self.hu_progress.setRange(-200, 1000)
        self.hu_progress.setValue(0)
        self.hu_progress.setFormat("%v HU")
        self.hu_progress.setStyleSheet(
            "QProgressBar { border: 1px solid #3f4a59; border-radius: 4px; text-align: center; color: white; background: #0d1117; height: 24px; font-weight: 800; }"
            "QProgressBar::chunk { background-color: #0f766e; border-radius: 3px; }"
        )
        hu_layout.addWidget(self.hu_progress)
        bq_tab_layout.addWidget(hu_group)

        # 3. Pullout Strength Progress Group
        po_group = QWidget()
        po_layout = QVBoxLayout(po_group)
        po_layout.setContentsMargins(0, 0, 0, 0)
        po_layout.setSpacing(4)
        po_lbl = QLabel("Estimated Pullout Strength")
        po_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #cbd5e1;")
        po_layout.addWidget(po_lbl)
        self.po_progress = QProgressBar()
        self.po_progress.setRange(0, 3000)
        self.po_progress.setValue(0)
        self.po_progress.setFormat("%v N")
        self.po_progress.setStyleSheet(
            "QProgressBar { border: 1px solid #3f4a59; border-radius: 4px; text-align: center; color: white; background: #0d1117; height: 24px; font-weight: 800; }"
            "QProgressBar::chunk { background-color: #f59e0b; border-radius: 3px; }"
        )
        po_layout.addWidget(self.po_progress)
        bq_tab_layout.addWidget(po_group)

        # 4. Clinical Recommendation Alert
        self.recommendation_box = QLabel("Evaluating screw track...")
        self.recommendation_box.setWordWrap(True)
        self.recommendation_box.setStyleSheet(
            "padding: 12px; border-radius: 6px; background: #1e293b; color: #cbd5e1; font-size: 11px; line-height: 1.4; border: 1px solid #334155;"
        )
        bq_tab_layout.addWidget(self.recommendation_box)
        bq_tab_layout.addStretch(1)

        # Plan Sidebar Tab
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
            self.value_labels[f"{key}"] = row
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

        sidebar_tabs.addTab(build_scroll_tab(bone_quality_host), "Bone Quality")
        sidebar_tabs.addTab(build_scroll_tab(plan_host), "Plan")

        adjust_host = QWidget()
        adjust_layout = QVBoxLayout(adjust_host)
        adjust_layout.setContentsMargins(4, 8, 4, 8)
        adjust_layout.setSpacing(10)

        preset_box = QGroupBox("Commercial Presets")
        preset_grid = QGridLayout(preset_box)
        preset_grid.setContentsMargins(10, 14, 10, 10)
        preset_grid.setHorizontalSpacing(8)
        preset_grid.setVerticalSpacing(8)

        preset_grid.addWidget(QLabel("Catalog Preset:"), 0, 0)
        self.preset_catalog_combo = QComboBox()
        self.preset_catalog_combo.addItems(["Generic", "Stryker", "Medtronic", "Globus"])
        self.preset_catalog_combo.currentTextChanged.connect(self.preset_catalog_changed)
        preset_grid.addWidget(self.preset_catalog_combo, 0, 1)

        preset_grid.addWidget(QLabel("Diameter (mm):"), 1, 0)
        self.preset_diam_combo = QComboBox()
        self.preset_diam_combo.addItems(["4.5", "5.0", "5.5", "6.0", "6.5", "7.0", "7.5", "8.0"])
        self.preset_diam_combo.currentTextChanged.connect(self.preset_diameter_changed)
        preset_grid.addWidget(self.preset_diam_combo, 1, 1)

        preset_grid.addWidget(QLabel("Length (mm):"), 2, 0)
        self.preset_len_combo = QComboBox()
        self.preset_len_combo.addItems(["30", "35", "40", "45", "50", "55", "60"])
        self.preset_len_combo.currentTextChanged.connect(self.preset_length_changed)
        preset_grid.addWidget(self.preset_len_combo, 2, 1)

        adjust_layout.addWidget(preset_box)

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
        self.central_widget = container
        self.setCentralWidget(container)
        self.resize(1480, 920)

    @property
    def subplot_indices(self):
        if getattr(self, "_current_plotter_shape", (2, 2)) == (1, 1):
            return [(0, 0)]
        return [(0, 0), (0, 1), (1, 0), (1, 1)]

    def ensure_plotter_shape(self):
        if QtInteractor is None:
            return
        expected_shape = (1, 1) if self.selected_vertebra is None else (2, 2)
        current_shape = getattr(self, "_current_plotter_shape", None)
        if current_shape == expected_shape and self.plotter is not None:
            return
            
        if self.plotter is not None:
            layout = self.central_widget.layout()
            if layout is not None:
                layout.removeWidget(self.plotter.interactor)
            self.plotter.close()
            self.plotter = None
            
        self.plotter = QtInteractor(self.central_widget, shape=expected_shape)
        self._current_plotter_shape = expected_shape
        self.plotter.set_background("#151a21")
        
        layout = self.central_widget.layout()
        if layout is not None:
            layout.insertWidget(0, self.plotter.interactor, 1)
        self.plotter.add_axes(line_width=1, color="#94a3b8")

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
        self.refresh_screws(force_all=True)
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
        self.refresh_screws(force_all=True)

    def set_active_screw(self, index):
        if index < 0 or index >= len(self.original_results):
            return
        result = self.original_results[index]
        if self.selected_vertebra is not None and result.get("vertebra") != self.selected_vertebra:
            self.vertebra_combo.setCurrentIndex(0)
        self.current_index = index
        if index in self.screw_index_map:
            combo_index = self.screw_index_map.index(index)
            self.screw_combo.blockSignals(True)
            self.screw_combo.setCurrentIndex(combo_index)
            self.screw_combo.blockSignals(False)
        self.load_slider_state()
        self.update_status()
        self.update_values_panel()
        self.refresh_screws(force_all=True)

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
            for r, c in self.subplot_indices:
                self.plotter.subplot(r, c)
                self.plotter.remove_actor(name, reset_camera=False)
        self.mesh_actor_names = []

    def refresh_scene(self):
        self.ensure_plotter_shape()
        if self.plotter is None:
            return
        self.clear_mesh_actors()
        if self.selected_vertebra is None:
            mesh = polydata_from_triangles(self.verts_world, self.faces)
            if mesh is not None:
                for r, c in self.subplot_indices:
                    self.plotter.subplot(r, c)
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
                scalars = self.compute_mesh_safety_scalars(verts, self.selected_vertebra)
                for r, c in self.subplot_indices:
                    self.plotter.subplot(r, c)
                    if scalars is not None:
                        mesh.point_data["Safety"] = scalars
                        self.plotter.add_mesh(
                            mesh,
                            scalars="Safety",
                            cmap=["#ef4444", "#f59e0b", "#94a3b8"],
                            clim=[-1.0, 2.0],
                            opacity=self.mesh_opacity,
                            smooth_shading=True,
                            specular=0.3,
                            name=f"vertebra_mesh_{self.selected_vertebra}",
                            show_scalar_bar=False,
                        )
                    else:
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

    def compute_mesh_safety_scalars(self, verts, level):
        if verts is None or len(verts) == 0:
            return None
        
        screws = [
            s for s in self.adjusted_results 
            if s.get("vertebra") == level
        ]
        
        if not screws:
            return np.full(len(verts), 2.0)
            
        N = len(verts)
        min_dist_to_screw = np.full(N, 999.0)
        screw_radii = np.zeros(N)
        
        for screw in screws:
            entry = np.asarray(screw["entry"], dtype=float)
            tip = np.asarray(screw["tip"], dtype=float)
            r = float(screw.get("diameter", 5.5)) / 2.0
            
            u = tip - entry
            u_len_sq = np.dot(u, u)
            if u_len_sq < 1e-8:
                continue
                
            proj = np.dot(verts - entry, u) / u_len_sq
            t = np.clip(proj, 0.0, 1.0)
            closest = entry + t[:, None] * u
            dist = np.linalg.norm(verts - closest, axis=1)
            
            closer_mask = dist < min_dist_to_screw
            min_dist_to_screw[closer_mask] = dist[closer_mask]
            screw_radii[closer_mask] = r

        scalars = np.full(N, 2.0)
        
        breach_mask = min_dist_to_screw < screw_radii
        scalars[breach_mask] = -1.0
        
        caution_mask = (min_dist_to_screw >= screw_radii) & (min_dist_to_screw < screw_radii + 2.0)
        scalars[caution_mask] = (min_dist_to_screw[caution_mask] - screw_radii[caution_mask]) / 2.0
        
        return scalars

    def focus_camera(self):
        if self.plotter is None:
            return
            
        verts = None
        if self.selected_vertebra is not None:
            if self.selected_vertebra in self.vertebra_meshes:
                verts = self.vertebra_meshes[self.selected_vertebra]["verts"]
            else:
                cropped_verts, _ = self.mesh_for_vertebra(self.selected_vertebra)
                verts = cropped_verts

        bounds = None
        if verts is not None and len(verts) > 0:
            bounds = (
                verts[:, 0].min(), verts[:, 0].max(),
                verts[:, 1].min(), verts[:, 1].max(),
                verts[:, 2].min(), verts[:, 2].max(),
            )

        # Loop through all active subplots to set bounding focuses and angles
        for r, c in self.subplot_indices:
            self.plotter.subplot(r, c)
            if bounds is not None:
                self.plotter.reset_camera(bounds=bounds)
            else:
                self.plotter.reset_camera()

        # Apply specific locked surgical projections only in Quad-View
        if len(self.subplot_indices) > 1:
            # 1. Axial / Superior View (Looking down Z)
            self.plotter.subplot(0, 0)
            self.plotter.view_xy()
            
            # 2. Sagittal / Lateral View (Looking down X)
            self.plotter.subplot(0, 1)
            self.plotter.view_zy()
            
            # 3. Coronal / Anterior View (Looking down Y)
            self.plotter.subplot(1, 0)
            self.plotter.view_xz()
            
            # 4. Isometric 3D Interactive View
            self.plotter.subplot(1, 1)
            if bounds is None:
                self.plotter.camera_position = "iso"
        else:
            # Single viewport: set to isometric/3D orbit
            self.plotter.subplot(0, 0)
            if bounds is None:
                self.plotter.camera_position = "iso"

    def load_slider_state(self):
        if not self.screw_index_map:
            return
        state = self.states[self.current_index]
        for key, (slider, value_label) in self.sliders.items():
            if key in state:
                slider.blockSignals(True)
                slider.setValue(int(round(state[key] * 2.0)))
                slider.blockSignals(False)
                value_label.setText(f"{state[key]:.1f}")
        
        if hasattr(self, "preset_catalog_combo"):
            self.preset_catalog_combo.blockSignals(True)
            catalog_val = state.get("catalog_type", "Generic")
            idx = self.preset_catalog_combo.findText(catalog_val)
            if idx != -1:
                self.preset_catalog_combo.setCurrentIndex(idx)
            self.preset_catalog_combo.blockSignals(False)
        
        if hasattr(self, "preset_diam_combo"):
            self.preset_diam_combo.blockSignals(True)
            val_str = f"{state['diameter']:.1f}"
            idx = self.preset_diam_combo.findText(val_str)
            if idx == -1:
                idx = self.preset_diam_combo.findText(f"{int(state['diameter'])}")
            if idx != -1:
                self.preset_diam_combo.setCurrentIndex(idx)
            self.preset_diam_combo.blockSignals(False)

        if hasattr(self, "preset_len_combo"):
            self.preset_len_combo.blockSignals(True)
            current_total_len = float(self.adjusted_results[self.current_index].get("length", 40.0) or 40.0)
            closest_idx = 0
            min_diff = 999.0
            for i in range(self.preset_len_combo.count()):
                try:
                    item_val = float(self.preset_len_combo.itemText(i))
                    diff = abs(item_val - current_total_len)
                    if diff < min_diff:
                        min_diff = diff
                        closest_idx = i
                except ValueError:
                    pass
            self.preset_len_combo.setCurrentIndex(closest_idx)
            self.preset_len_combo.blockSignals(False)

    def slider_changed(self, key):
        slider, value_label = self.sliders[key]
        value = slider.value() / 2.0
        self.states[self.current_index][key] = value
        value_label.setText(f"{value:.1f}")
        self.recompute_results()
        self.update_status()
        self.update_values_panel()
        self.refresh_screws()
        self.refresh_scene()

    def preset_diameter_changed(self, text):
        if self._updating_screw_combo or not self.screw_index_map:
            return
        try:
            val = float(text)
            self.states[self.current_index]["diameter"] = val
            self.recompute_results()
            self.update_status()
            self.update_values_panel()
            self.refresh_screws()
            self.refresh_scene()
        except ValueError:
            pass

    def preset_length_changed(self, text):
        if self._updating_screw_combo or not self.screw_index_map:
            return
        try:
            val = float(text)
            original_len = float(self.original_results[self.current_index].get("length", 40.0) or 40.0)
            delta = val - original_len
            self.states[self.current_index]["length_mm"] = delta
            
            slider, value_label = self.sliders["length_mm"]
            slider.blockSignals(True)
            slider.setValue(int(round(delta * 2.0)))
            slider.blockSignals(False)
            value_label.setText(f"{delta:.1f}")
            
            self.recompute_results()
            self.update_status()
            self.update_values_panel()
            self.refresh_screws()
            self.refresh_scene()
        except ValueError:
            pass

    def preset_catalog_changed(self, text):
        if self._updating_screw_combo or not self.screw_index_map:
            return
        self.states[self.current_index]["catalog_type"] = text
        
        if hasattr(self, "preset_diam_combo") and hasattr(self, "preset_len_combo"):
            self.preset_diam_combo.blockSignals(True)
            self.preset_len_combo.blockSignals(True)
            if text == "Stryker":
                self.preset_diam_combo.setCurrentText("6.5")
                self.preset_len_combo.setCurrentText("45")
                self.states[self.current_index]["diameter"] = 6.5
                original_len = float(self.original_results[self.current_index].get("length", 40.0) or 40.0)
                delta = 45.0 - original_len
                self.states[self.current_index]["length_mm"] = delta
            elif text == "Medtronic":
                self.preset_diam_combo.setCurrentText("5.5")
                self.preset_len_combo.setCurrentText("40")
                self.states[self.current_index]["diameter"] = 5.5
                original_len = float(self.original_results[self.current_index].get("length", 40.0) or 40.0)
                delta = 40.0 - original_len
                self.states[self.current_index]["length_mm"] = delta
            elif text == "Globus":
                self.preset_diam_combo.setCurrentText("7.0")
                self.preset_len_combo.setCurrentText("50")
                self.states[self.current_index]["diameter"] = 7.0
                original_len = float(self.original_results[self.current_index].get("length", 40.0) or 40.0)
                delta = 50.0 - original_len
                self.states[self.current_index]["length_mm"] = delta
            self.preset_diam_combo.blockSignals(False)
            self.preset_len_combo.blockSignals(False)

        delta = self.states[self.current_index]["length_mm"]
        if "length_mm" in self.sliders:
            slider, value_label = self.sliders["length_mm"]
            slider.blockSignals(True)
            slider.setValue(int(round(delta * 2.0)))
            slider.blockSignals(False)
            value_label.setText(f"{delta:.1f}")

        self.recompute_results()
        self.update_status()
        self.update_values_panel()
        self.refresh_screws()
        self.refresh_scene()

    def estimate_pullout_strength(self, entry, tip, diameter, length):
        if self.volume_data is None or self.volume_affine is None:
            return 0.0, 0.0
        
        res = orthonormal_basis(entry, tip)
        if res is None:
            return 0.0, 0.0
        u_dir, n1, n2 = res
        if u_dir is None or n1 is None or n2 is None:
            return 0.0, 0.0
            
        u_vals = np.linspace(0.0, length, 20)
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        r_vals = np.linspace(diameter / 2.0 - 0.5, diameter / 2.0 + 1.5, 3)
        
        sampled_hus = []
        inv_affine = np.linalg.inv(self.volume_affine)
        sh = self.volume_data.shape
        
        for u in u_vals:
            for theta in angles:
                for r in r_vals:
                    pt = entry + u * u_dir + r * (np.cos(theta) * n1 + np.sin(theta) * n2)
                    vpt = (pt @ inv_affine[:3, :3].T) + inv_affine[:3, 3]
                    x = int(np.round(vpt[0]))
                    y = int(np.round(vpt[1]))
                    z = int(np.round(vpt[2]))
                    if 0 <= x < sh[0] and 0 <= y < sh[1] and 0 <= z < sh[2]:
                        sampled_hus.append(self.volume_data[x, y, z])
                        
        if not sampled_hus:
            return 0.0, 0.0
        mean_hu = float(np.mean(sampled_hus))
        strength = max(0.0, 2.5 * mean_hu + 550.0)
        return strength, mean_hu

    def update_reformat_slice(self):
        if not self.adjusted_results or self.current_index >= len(self.adjusted_results):
            self.bone_quality_badge.setText("NO ACTIVE SCREW")
            self.hu_progress.setValue(0)
            self.po_progress.setValue(0)
            self.recommendation_box.setText("Please select a screw.")
            return
            
        result = self.adjusted_results[self.current_index]
        entry = np.asarray(result["entry"], dtype=float)
        tip = np.asarray(result["tip"], dtype=float)
        diameter = float(result.get("diameter", 5.5) or 5.5)
        length = float(result.get("length", 40.0) or 40.0)
        
        strength, mean_hu = self.estimate_pullout_strength(entry, tip, diameter, length)
        
        self.hu_progress.setValue(int(round(mean_hu)))
        self.po_progress.setValue(int(round(strength)))
        
        if mean_hu > 150.0:
            self.bone_quality_badge.setText("🟢 STRONG / HEALTHY BONE")
            self.bone_quality_badge.setStyleSheet(
                "font-size: 13px; font-weight: 900; color: #ffffff; background: #059669; padding: 10px; border-radius: 6px; border: 1px solid #10b981;"
            )
            self.hu_progress.setStyleSheet(
                "QProgressBar { border: 1px solid #3f4a59; border-radius: 4px; text-align: center; color: white; background: #0d1117; height: 24px; font-weight: 800; }"
                "QProgressBar::chunk { background-color: #10b981; border-radius: 3px; }"
            )
            self.recommendation_box.setText(
                "<b>Clinical Analysis:</b><br>"
                "Excellent bone mineral density with optimal screw purchase. Standard titanium screw placement is highly stable. "
                "No augmentation required."
            )
        elif mean_hu >= 100.0:
            self.bone_quality_badge.setText("🟡 OSTEOPENIC BONE")
            self.bone_quality_badge.setStyleSheet(
                "font-size: 13px; font-weight: 900; color: #111827; background: #d97706; padding: 10px; border-radius: 6px; border: 1px solid #f59e0b;"
            )
            self.hu_progress.setStyleSheet(
                "QProgressBar { border: 1px solid #3f4a59; border-radius: 4px; text-align: center; color: #111827; background: #0d1117; height: 24px; font-weight: 800; }"
                "QProgressBar::chunk { background-color: #f59e0b; border-radius: 3px; }"
            )
            self.recommendation_box.setText(
                "<b>Clinical Analysis:</b><br>"
                "Moderately reduced density. Corridors provide solid fixation, but insertion torque should be monitored closely. "
                "Standard titanium pedicle screw purchase is acceptable."
            )
        else:
            self.bone_quality_badge.setText("🔴 OSTEOPOROTIC / WEAK BONE")
            self.bone_quality_badge.setStyleSheet(
                "font-size: 13px; font-weight: 900; color: #ffffff; background: #dc2626; padding: 10px; border-radius: 6px; border: 1px solid #ef4444;"
            )
            self.hu_progress.setStyleSheet(
                "QProgressBar { border: 1px solid #3f4a59; border-radius: 4px; text-align: center; color: white; background: #0d1117; height: 24px; font-weight: 800; }"
                "QProgressBar::chunk { background-color: #ef4444; border-radius: 3px; }"
            )
            self.recommendation_box.setText(
                "<b>WARNING:</b> Low density. High risk of screw loosening. PMMA bone cement augmentation or fenestrated screw insertion is strongly recommended."
            )

    def nudge_slider(self, slider, delta):
        slider.setValue(int(np.clip(slider.value() + delta, slider.minimum(), slider.maximum())))

    def recompute_results(self):
        self.adjusted_results = [
            adjusted_result(result, state)
            for result, state in zip(self.original_results, self.states)
        ]
        self.plan_adjusted.emit(self.adjusted_results)

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
            self.value_labels[f"{key}"].setText(f"{float(state.get(key, 0.0)):.1f}")

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
        self.update_reformat_slice()

    def refresh_screws(self, force_all=False):
        if self.plotter is None:
            return
            
        if force_all:
            for name in self.screw_actor_names:
                for r, c in self.subplot_indices:
                    self.plotter.subplot(r, c)
                    self.plotter.remove_actor(name, reset_camera=False)
            self.screw_actor_names = []
            indices_to_build = self.visible_screw_indices()
        else:
            for r, c in self.subplot_indices:
                self.plotter.subplot(r, c)
                self.plotter.remove_actor(f"screw_path_{self.current_index}", reset_camera=False)
                self.plotter.remove_actor(f"screw_body_{self.current_index}", reset_camera=False)
            indices_to_build = [self.current_index]

        visible = set(self.visible_screw_indices())
        for index in indices_to_build:
            if index not in visible:
                continue
            result = self.adjusted_results[index]
            entry = np.asarray(result["entry"], dtype=float)
            tip = np.asarray(result["tip"], dtype=float)
            status, color, _, _ = evaluate_screw_safety(result, self.seg_data, self.seg_affine)
            diameter = float(result.get("diameter", 5.5) or 5.5)
            line_name = f"screw_path_{index}"
            screw_name = f"screw_body_{index}"
            selected = index == self.current_index

            line = pv.Line(entry, tip) if pv is not None else None
            if line is not None:
                for r, c in self.subplot_indices:
                    self.plotter.subplot(r, c)
                    self.plotter.add_mesh(line, color=color, line_width=5 if selected else 3, name=line_name)
                if line_name not in self.screw_actor_names:
                    self.screw_actor_names.append(line_name)

            # Generate advanced high-fidelity procedural hardware model
            catalog_val = self.states[index].get("catalog_type", "Generic")
            screw_mesh = load_and_transform_screw(
                entry,
                tip,
                diameter,
                float(result.get("length", 40.0)),
                catalog_type=catalog_val,
            )
            if screw_mesh is not None:
                for r, c in self.subplot_indices:
                    self.plotter.subplot(r, c)
                    self.plotter.add_mesh(
                        screw_mesh,
                        color=color,
                        opacity=1.0 if selected else 0.90,
                        smooth_shading=True,
                        specular=0.55,       # High specularity to let simulated threads catch realistic light highlights
                        specular_power=15,
                        name=screw_name,
                    )
                if screw_name not in self.screw_actor_names:
                    self.screw_actor_names.append(screw_name)

        self.plotter.render()

    def reset_current(self):
        self.states[self.current_index] = {
            "lr_mm": 0.0,
            "ud_mm": 0.0,
            "axial_deg": 0.0,
            "sagittal_deg": 0.0,
            "length_mm": 0.0,
            "diameter": float(self.original_results[self.current_index].get("diameter", 5.5) or 5.5),
        }
        self.recompute_results()
        self.load_slider_state()
        self.update_status()
        self.update_values_panel()
        self.refresh_screws()

    def save_json(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save adjusted plan", "adjusted_screw_plan_v8.json", "JSON Files (*.json)"
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
            self, "Export planning report", "screw_plan_report_v8.csv", "CSV Files (*.csv)"
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
            self, "Save visualization image", "visualization_v8.png", "PNG Files (*.png)"
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
        window.raise_()
        window.activateWindow()
        if owns_app:
            app.exec()
        return window

    return None, show_figure