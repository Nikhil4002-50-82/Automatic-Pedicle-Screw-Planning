import copy
import csv
import json
import os
import sys
import tempfile

import nibabel as nib
import numpy as np
import plotly.graph_objects as go

from PyQt6.QtCore import QUrl, Qt
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
)
from PyQt6.QtWebEngineWidgets import QWebEngineView


VIEWER_WINDOWS = []


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

    faces_i = np.concatenate((idx, idx, np.full(resolution, entry_center), np.full(resolution, tip_center)))
    faces_j = np.concatenate((tip_idx, tip_nxt, nxt, tip_idx))
    faces_k = np.concatenate((tip_nxt, nxt, idx, tip_nxt))

    return vertices[:, 0], vertices[:, 1], vertices[:, 2], faces_i, faces_j, faces_k


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
        return "Safe", "#22c55e", "Trajectory mostly contained in segmented bone"
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


def build_figure(verts_world, faces, results, seg_data=None, seg_affine=None, mesh_opacity=0.22):
    fig = go.Figure()
    verts_world = np.asarray(verts_world)
    faces = np.asarray(faces)

    fig.add_trace(
        go.Mesh3d(
            x=verts_world[:, 0],
            y=verts_world[:, 1],
            z=verts_world[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="#d1d5db",
            opacity=mesh_opacity,
            name="Vertebra mesh",
            hoverinfo="skip",
            lighting=dict(ambient=0.55, diffuse=0.75, specular=0.35, roughness=0.55),
        )
    )

    for index, result in enumerate(results):
        entry = np.asarray(result["entry"], dtype=float)
        tip = np.asarray(result["tip"], dtype=float)
        status, color, message = evaluate_screw_safety(result, seg_data, seg_affine)
        name = f"{result.get('vertebra', '')} {result.get('side', '')}"
        diameter = float(result.get("diameter", 5.5) or 5.5)

        fig.add_trace(
            go.Scatter3d(
                x=[entry[0], tip[0]],
                y=[entry[1], tip[1]],
                z=[entry[2], tip[2]],
                mode="lines",
                line=dict(color=color, width=7),
                name=f"{name} path",
                customdata=[index],
                hovertemplate=(
                    f"<b>{name}</b><br>Status: {status}<br>{message}<br>"
                    f"Length: {np.linalg.norm(tip-entry):.1f} mm<br>"
                    f"Diameter: {diameter:.1f} mm<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[entry[0]],
                y=[entry[1]],
                z=[entry[2]],
                mode="markers",
                marker=dict(size=6, color="#0ea5e9"),
                name=f"{name} entry",
                hoverinfo="skip",
            )
        )

        cyl = create_cylinder_mesh(entry, tip, diameter)
        if cyl is not None:
            x, y, z, i, j, k = cyl
            fig.add_trace(
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i,
                    j=j,
                    k=k,
                    color=color,
                    opacity=0.72,
                    name=f"{name} screw",
                    hovertemplate=f"<b>{name}</b><br>{status}: {message}<extra></extra>",
                    lighting=dict(ambient=0.5, diffuse=0.85, specular=0.55, roughness=0.25),
                )
            )

    fig.update_layout(
        title=dict(text="Pedicle Screw Manual Visualizer V6", x=0.5, font=dict(color="#f8fafc")),
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        scene=dict(
            aspectmode="data",
            bgcolor="#0f172a",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, t=42, b=0),
        height=760,
        legend=dict(orientation="h", y=0.02, x=0.02),
        uirevision="visualizer-v6",
    )
    return fig


def load_segmentation(segmentation_path):
    if not segmentation_path:
        return None, None
    nii = nib.load(segmentation_path)
    return nii.get_fdata(), nii.affine


class ManualVisualizerWindow(QMainWindow):
    def __init__(self, verts_world, faces, results, volume_path=None, segmentation_path=None):
        super().__init__()
        self.verts_world = verts_world
        self.faces = faces
        self.original_results = copy.deepcopy(results)
        self.volume_path = volume_path
        self.segmentation_path = segmentation_path
        self.seg_data, self.seg_affine = load_segmentation(segmentation_path)
        self.current_index = 0
        self.states = [
            {"lr_mm": 0.0, "ud_mm": 0.0, "axial_deg": 0.0, "sagittal_deg": 0.0, "length_mm": 0.0}
            for _ in self.original_results
        ]
        self.adjusted_results = copy.deepcopy(self.original_results)
        self.html_path = None
        self.init_ui()
        self.refresh_plot()

    def init_ui(self):
        self.setWindowTitle("Manual Screw Visualizer V6")
        container = QWidget()
        root = QHBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.view = QWebEngineView()
        root.addWidget(self.view, 1)

        panel = QWidget()
        panel.setFixedWidth(360)
        panel.setStyleSheet(
            "QWidget { background: #f8fafc; color: #0f172a; font-family: Segoe UI; }"
            "QLabel { font-size: 12px; font-weight: 600; }"
            "QPushButton { background: #2563eb; color: white; border: none; padding: 9px 12px; border-radius: 5px; font-weight: 700; }"
            "QPushButton:hover { background: #1d4ed8; }"
            "QComboBox { padding: 7px; border: 1px solid #cbd5e1; border-radius: 4px; background: white; }"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Manual Screw Adjustment")
        title.setStyleSheet("font-size: 18px; font-weight: 800;")
        layout.addWidget(title)

        self.screw_combo = QComboBox()
        for result in self.original_results:
            self.screw_combo.addItem(f"{result.get('vertebra', '')} {result.get('side', '')}")
        self.screw_combo.currentIndexChanged.connect(self.select_screw)
        layout.addWidget(self.screw_combo)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("padding: 10px; border-radius: 5px; background: #e2e8f0;")
        layout.addWidget(self.status_label)

        self.sliders = {}
        grid = QGridLayout()
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
        reset_btn.clicked.connect(self.reset_current)
        layout.addWidget(reset_btn)

        save_json_btn = QPushButton("Save Plan JSON")
        save_json_btn.clicked.connect(self.save_json)
        layout.addWidget(save_json_btn)

        export_csv_btn = QPushButton("Export Report CSV")
        export_csv_btn.clicked.connect(self.export_csv)
        layout.addWidget(export_csv_btn)

        export_img_btn = QPushButton("Export Screenshot")
        export_img_btn.clicked.connect(self.export_image)
        layout.addWidget(export_img_btn)

        layout.addStretch(1)
        root.addWidget(panel)
        self.setCentralWidget(container)
        self.resize(1420, 920)

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
        self.refresh_plot()

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
        self.status_label.setStyleSheet(f"padding: 10px; border-radius: 5px; background: {color}; color: white; font-weight: 800;")

    def reset_current(self):
        self.states[self.current_index] = {"lr_mm": 0.0, "ud_mm": 0.0, "axial_deg": 0.0, "sagittal_deg": 0.0, "length_mm": 0.0}
        self.recompute_results()
        self.load_slider_state()
        self.update_status()
        self.refresh_plot()

    def refresh_plot(self):
        fig = build_figure(self.verts_world, self.faces, self.adjusted_results, self.seg_data, self.seg_affine)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            fig.write_html(tmp.name, include_plotlyjs=True)
            next_html = tmp.name

        old_html = self.html_path
        self.html_path = next_html
        self.view.load(QUrl.fromLocalFile(next_html))
        if old_html:
            try:
                os.remove(old_html)
            except OSError:
                pass

    def save_json(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save adjusted plan", "adjusted_screw_plan.json", "JSON Files (*.json)")
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
        file_path, _ = QFileDialog.getSaveFileName(self, "Export planning report", "screw_plan_report.csv", "CSV Files (*.csv)")
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
        file_path, _ = QFileDialog.getSaveFileName(self, "Save visualization image", "visualization_v6.png", "PNG Files (*.png)")
        if file_path:
            self.view.grab().save(file_path)

    def closeEvent(self, event):
        if self.html_path:
            try:
                os.remove(self.html_path)
            except OSError:
                pass
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

    fig = build_figure(vertsWorld, faces, resultsList, window.seg_data, window.seg_affine)
    return fig, show_figure
