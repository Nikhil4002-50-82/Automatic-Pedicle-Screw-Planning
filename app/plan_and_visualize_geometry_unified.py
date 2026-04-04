import argparse
import json
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout

import nibabel as nib
import numpy as np
from PyQt6.QtCore import QCoreApplication, QThread, QTimer, Qt, QUrl, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # pragma: no cover - required for the viewer, but keep the message clear.
    QWebEngineView = None
from skimage.measure import marching_cubes

QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

from geometry import run_planner
from visualizer_unified import (
    _ensure_plotly_imports,
    _ensure_plotlyjs_bundle,
    _figure_without_embedded_controls,
    build_visualization,
)


class PlanningWorker(QThread):
    results_ready = pyqtSignal(list)
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, seg_path):
        super().__init__()
        self._seg_path = seg_path

    def run(self):
        class _ConsoleStream:
            def __init__(self, emit):
                self._emit = emit
                self._buffer = ""

            def write(self, text):
                if not text:
                    return 0
                self._buffer += text
                while "\n" in self._buffer:
                    line, self._buffer = self._buffer.split("\n", 1)
                    self._emit(line)
                return len(text)

            def flush(self):
                if self._buffer:
                    self._emit(self._buffer)
                    self._buffer = ""

        try:
            stream = _ConsoleStream(self.log.emit)
            with redirect_stdout(stream), redirect_stderr(stream):
                results = run_planner(self._seg_path)
            stream.flush()
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.results_ready.emit(results)


class PlanningConsole(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Planning Console")
        self.setModal(False)
        self.resize(900, 420)

        layout = QVBoxLayout(self)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet(
            "QPlainTextEdit {"
            "  background-color: #08111D;"
            "  color: #D7E3F4;"
            "  border: 1px solid rgba(148, 163, 184, 0.28);"
            "  font-family: Consolas, monospace;"
            "  font-size: 12px;"
            "}"
        )
        layout.addWidget(self.output)

    def append_text(self, text):
        if not text:
            return
        self.output.appendPlainText(text.rstrip("\n"))
        self.output.ensureCursorVisible()

    def clear_output(self):
        self.output.clear()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Open a blank unified viewer, load a segmentation mask, and run planning in place."
    )
    parser.add_argument(
        "--preset",
        choices=["classic", "surgical", "cinematic"],
        default="cinematic",
        help="Choose the overall visual treatment for the unified viewer.",
    )
    parser.add_argument(
        "--screw-mode",
        choices=["threaded", "cylinder", "none"],
        default="threaded",
        help="Choose how screws are rendered in the unified viewer.",
    )
    parser.add_argument(
        "--theme",
        choices=["light", "dark"],
        default="dark",
        help="Choose the Plotly theme used by the unified viewer.",
    )
    parser.add_argument(
        "--mesh-opacity",
        type=float,
        default=0.25,
        help="Starting opacity for the vertebra mesh.",
    )
    parser.add_argument(
        "--show-safety-planes",
        action="store_true",
        help="Render the optional tip safety planes from the unified viewer.",
    )
    parser.add_argument(
        "--hide-bounding-box",
        action="store_true",
        help="Disable the anatomy bounding box overlay.",
    )
    parser.add_argument(
        "--hide-trajectory-lines",
        action="store_true",
        help="Disable screw trajectory line overlays.",
    )
    parser.add_argument(
        "--hide-entry-markers",
        action="store_true",
        help="Disable screw entry-point markers.",
    )
    parser.add_argument(
        "--show-tip-markers",
        action="store_true",
        help="Render tip-point markers at the distal end of each trajectory.",
    )
    parser.add_argument(
        "--v2-neon-trajectories",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the bright layered trajectory treatment inspired by visualizerV2.",
    )
    parser.add_argument(
        "--v2-gold-screws",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the shared metallic gold screw palette from visualizerV2.",
    )
    parser.add_argument(
        "--v2-threaded-screws",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable V2-style threaded screw geometry.",
    )
    parser.add_argument(
        "--v2-safety-planes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the V2-style safety planes at the screw tips.",
    )
    parser.add_argument(
        "--fallback-diameter",
        type=float,
        default=None,
        help="Optional visual-only fallback screw diameter when the planner result omits one.",
    )
    return parser


def _build_mask_mesh(seg_path):
    data, _, affine = run_nifti_load(seg_path)
    mask = data > 0
    if not mask.any():
        raise ValueError("The selected segmentation does not contain any labeled voxels.")

    verts, faces, _, _ = marching_cubes(mask.astype("uint8"), level=0.5)
    verts_world = nib.affines.apply_affine(affine, verts)
    return verts_world, faces, mask.shape


def run_nifti_load(seg_path):
    nii = nib.load(seg_path)
    return nii.get_fdata(), nii.header.get_zooms(), nii.affine


def _blank_html(background_color):
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\">
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: {background_color};
    }}
  </style>
</head>
<body></body>
</html>
"""


def _figure_to_html(fig):
    paper_bg = getattr(fig.layout, "paper_bgcolor", None) or "#0B1320"
    plotly_bundle_uri = os.path.abspath(_ensure_plotlyjs_bundle())
    _, pio = _ensure_plotly_imports()
    qt_fig = _figure_without_embedded_controls(fig)

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\">
  <script src=\"{QUrl.fromLocalFile(plotly_bundle_uri).toString()}\"></script>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: {paper_bg};
    }}
    body {{
      font-family: Segoe UI, sans-serif;
    }}
    .plotly-graph-div {{
      width: 100% !important;
      height: 100% !important;
    }}
    .modebar-btn,
    g.updatemenu-button,
    g.updatemenu-button *,
    g.slider *,
    .legendtoggle {{
      cursor: pointer !important;
    }}
  </style>
</head>
<body>
{pio.to_html(qt_fig, full_html=False, include_plotlyjs=False, default_width="100%", default_height="100%", config=dict(displayModeBar=True, displaylogo=False, scrollZoom=True, modeBarButtonsToAdd=["v1hovermode", "toggleSpikelines"]))}
</body>
</html>
"""


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class GeometryPlanningWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self._args = args
        self._current_seg_path = None
        self._current_verts = None
        self._current_faces = None
        self._current_results = []
        self._current_display_mode = "max_diameter"
        self._current_bbox_visible = not args.hide_bounding_box
        self._current_html_path = None
        self._planning_worker = None
        self._planning_console = None
        self._plan_ready = False
        self._current_mask_shape = None
        self._last_directory = os.path.expanduser("~/Downloads")

        self.setWindowTitle("Pedicle Screw Planner Visualization")
        self._build_ui()
        self._render_blank_view()
        self._set_loaded_state(False)

    def _build_ui(self):
        container = QWidget()
        root_layout = QVBoxLayout(container)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        container.setStyleSheet("background-color: #0B1320;")

        if QWebEngineView is None:
            raise RuntimeError("PyQt6.QtWebEngineWidgets is required to display the unified viewer.")

        self.view = QWebEngineView()
        root_layout.addWidget(self.view, 1)

        controls_panel = QWidget()
        controls_panel.setStyleSheet(
            "background-color: rgba(16, 25, 38, 0.96);"
            "border-top: 1px solid rgba(151, 164, 180, 0.22);"
        )
        controls_layout = QHBoxLayout(controls_panel)
        controls_layout.setContentsMargins(18, 10, 18, 10)
        controls_layout.setSpacing(12)

        self.load_button = QPushButton("Load Mask")
        self.load_results_button = QPushButton("Load Results")
        self.run_button = QPushButton("Run Planning")
        self.show_screws_button = QPushButton("Show Max Diameter")
        self.show_traj_button = QPushButton("Show Trajectories")
        self.show_bbox_button = QPushButton("Show Bounding Box")
        self.hide_bbox_button = QPushButton("Hide Bounding Box")
        self.opacity_label = QLabel(f"Mesh Opacity: {float(self._args.mesh_opacity):.2f}")
        self.opacity_label.setStyleSheet("color: #F7FAFC; padding-left: 8px; font-weight: 600;")
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(5, 100)
        self.opacity_slider.setValue(int(round(float(self._args.mesh_opacity) * 100)))
        self.opacity_slider.setFixedWidth(240)
        self.opacity_slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_button = QPushButton("Export Image")

        button_style = (
            "QPushButton {"
            "  background-color: rgba(76, 99, 130, 0.72);"
            "  color: #F7FAFC;"
            "  border: 1px solid rgba(151, 164, 180, 0.18);"
            "  padding: 8px 14px;"
            "  border-radius: 7px;"
            "  font-weight: 600;"
            "}"
            "QPushButton:hover {"
            "  background-color: rgba(96, 125, 163, 0.95);"
            "  border: 1px solid rgba(123, 229, 255, 0.55);"
            "}"
            "QPushButton:pressed {"
            "  background-color: rgba(55, 75, 104, 0.98);"
            "  padding-top: 9px;"
            "  padding-bottom: 7px;"
            "}"
            "QPushButton:checked {"
            "  background-color: rgba(54, 188, 229, 0.92);"
            "  color: #06121D;"
            "  border: 1px solid rgba(186, 248, 255, 0.92);"
            "}"
            "QPushButton:disabled {"
            "  background-color: rgba(51, 65, 85, 0.88);"
            "  color: #94A3B8;"
            "  border: 1px solid rgba(100, 116, 139, 0.2);"
            "}"
        )
        export_style = (
            "QPushButton {"
            "  background-color: rgba(30, 41, 59, 0.92);"
            "  color: #F7FAFC;"
            "  border: 1px solid rgba(148, 163, 184, 0.28);"
            "  padding: 10px 14px;"
            "  border-radius: 7px;"
            "  font-weight: 600;"
            "}"
            "QPushButton:hover {"
            "  background-color: rgba(45, 62, 87, 0.98);"
            "  border: 1px solid rgba(123, 229, 255, 0.45);"
            "}"
            "QPushButton:pressed {"
            "  background-color: rgba(18, 27, 42, 1.0);"
            "  padding-top: 11px;"
            "  padding-bottom: 9px;"
            "}"
            "QPushButton:disabled {"
            "  color: #94A3B8;"
            "  background-color: rgba(30, 41, 59, 0.55);"
            "  border: 1px solid rgba(100, 116, 139, 0.18);"
            "}"
        )
        for button in (
            self.load_button,
            self.load_results_button,
            self.run_button,
            self.show_screws_button,
            self.show_traj_button,
            self.show_bbox_button,
            self.hide_bbox_button,
        ):
            button.setStyleSheet(button_style)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setCheckable(True)

        self.load_button.setCheckable(False)
        self.load_results_button.setCheckable(False)
        self.run_button.setCheckable(False)
        self.export_button.setStyleSheet(export_style)
        self.export_button.setCursor(Qt.CursorShape.PointingHandCursor)

        self.show_screw_buttons = QButtonGroup(self)
        self.show_screw_buttons.setExclusive(True)
        self.show_screw_buttons.addButton(self.show_screws_button)
        self.show_screw_buttons.addButton(self.show_traj_button)

        self.show_bbox_buttons = QButtonGroup(self)
        self.show_bbox_buttons.setExclusive(True)
        self.show_bbox_buttons.addButton(self.show_bbox_button)
        self.show_bbox_buttons.addButton(self.hide_bbox_button)

        self.show_screws_button.setChecked(True)
        self.show_bbox_button.setChecked(self._current_bbox_visible)
        self.hide_bbox_button.setChecked(not self._current_bbox_visible)

        self.opacity_slider.setStyleSheet(
            "QSlider::groove:horizontal {"
            "  height: 8px;"
            "  border-radius: 4px;"
            "  background: rgba(120, 134, 156, 0.32);"
            "}"
            "QSlider::sub-page:horizontal {"
            "  border-radius: 4px;"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4BD3FF, stop:1 #79F2FF);"
            "}"
            "QSlider::add-page:horizontal {"
            "  border-radius: 4px;"
            "  background: rgba(120, 134, 156, 0.24);"
            "}"
            "QSlider::handle:horizontal {"
            "  width: 18px;"
            "  margin: -6px 0;"
            "  border-radius: 9px;"
            "  border: 2px solid rgba(255,255,255,0.88);"
            "  background: #F7FAFC;"
            "}"
            "QSlider::handle:horizontal:hover {"
            "  background: #FFFFFF;"
            "  border: 2px solid #79F2FF;"
            "}"
            "QSlider::handle:horizontal:pressed {"
            "  background: #CFFAFE;"
            "  border: 2px solid #22D3EE;"
            "}"
        )

        controls_layout.addWidget(self.load_button)
        controls_layout.addWidget(self.load_results_button)
        controls_layout.addWidget(self.run_button)
        controls_layout.addWidget(self.show_screws_button)
        controls_layout.addWidget(self.show_traj_button)
        controls_layout.addWidget(self.show_bbox_button)
        controls_layout.addWidget(self.hide_bbox_button)
        controls_layout.addWidget(self.opacity_label)
        controls_layout.addWidget(self.opacity_slider)
        controls_layout.addStretch(1)

        root_layout.addWidget(controls_panel)
        root_layout.setStretchFactor(self.view, 1)

        self.export_button.setMinimumHeight(42)
        root_layout.addWidget(self.export_button)

        self.load_button.clicked.connect(self._load_mask)
        self.load_results_button.clicked.connect(self._load_results)
        self.run_button.clicked.connect(self._run_planning)
        self.show_screws_button.clicked.connect(self._set_max_diameter_mode)
        self.show_traj_button.clicked.connect(self._set_trajectory_mode)
        self.show_bbox_button.clicked.connect(lambda: self._set_bbox_visibility(True))
        self.hide_bbox_button.clicked.connect(lambda: self._set_bbox_visibility(False))
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.export_button.clicked.connect(self._export_image)

        self.setCentralWidget(container)
        self.resize(1320, 940)

    def _set_loaded_state(self, loaded):
        self.run_button.setEnabled(loaded)
        self.load_results_button.setEnabled(loaded)
        self.show_screws_button.setEnabled(loaded)
        self.show_traj_button.setEnabled(loaded)
        self.show_bbox_button.setEnabled(loaded)
        self.hide_bbox_button.setEnabled(loaded)
        self.opacity_slider.setEnabled(loaded)

    def _set_busy_state(self, busy):
        self.load_button.setEnabled(not busy)
        self.load_results_button.setEnabled((not busy) and self._current_verts is not None)
        self.run_button.setEnabled((not busy) and self._current_verts is not None)
        self.show_screws_button.setEnabled((not busy) and self._current_verts is not None)
        self.show_traj_button.setEnabled((not busy) and self._current_verts is not None)
        self.show_bbox_button.setEnabled((not busy) and self._current_verts is not None)
        self.hide_bbox_button.setEnabled((not busy) and self._current_verts is not None)
        self.opacity_slider.setEnabled((not busy) and self._current_verts is not None)

    def _set_max_diameter_mode(self):
        self._current_display_mode = "max_diameter"
        if self._current_verts is not None:
            self._render_scene()

    def _set_trajectory_mode(self):
        self._current_display_mode = "trajectories"
        if self._current_verts is not None:
            self._render_scene()

    def _set_bbox_visibility(self, visible):
        self._current_bbox_visible = visible
        if visible:
            self.show_bbox_button.setChecked(True)
        else:
            self.hide_bbox_button.setChecked(True)
        if self._current_verts is not None:
            self._render_scene()

    def _on_opacity_changed(self, value):
        self.opacity_label.setText(f"Mesh Opacity: {value / 100.0:.2f}")
        if self._current_verts is None:
            return
        self._render_scene_delayed()

    def _render_scene_delayed(self):
        QTimer.singleShot(0, self._render_scene)

    def _build_scene_figure(self):
        show_trajectory_lines = self._current_display_mode == "trajectories"
        screw_mode = "none" if show_trajectory_lines else "cylinder"

        return build_visualization(
            verts_world=self._current_verts,
            faces=self._current_faces,
            results_list=self._current_results,
            volume_path=self._current_seg_path,
            screw_mode=screw_mode,
            theme=self._args.theme,
            visual_preset=self._args.preset,
            mesh_opacity=self.opacity_slider.value() / 100.0,
            show_safety_planes=self._args.show_safety_planes,
            show_bounding_box=self._current_bbox_visible,
            show_trajectory_lines=show_trajectory_lines,
            show_entry_markers=not self._args.hide_entry_markers,
            show_tip_markers=self._args.show_tip_markers,
            v2_neon_trajectories=self._args.v2_neon_trajectories,
            v2_gold_screws=self._args.v2_gold_screws,
            v2_threaded_screws=self._args.v2_threaded_screws,
            v2_safety_planes=self._args.v2_safety_planes,
            fallback_diameter=self._args.fallback_diameter,
        )

    def _load_html(self, html_document):
        if self._current_html_path and os.path.exists(self._current_html_path):
            try:
                os.remove(self._current_html_path)
            except OSError:
                pass

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, encoding="utf-8", mode="w") as tmpfile:
            tmpfile.write(html_document)
            self._current_html_path = tmpfile.name

        self.view.load(QUrl.fromLocalFile(self._current_html_path))

    def _render_blank_view(self):
        self._load_html(_blank_html("#0B1320"))

    def _render_scene(self):
        if self._current_verts is None or self._current_faces is None:
            self._render_blank_view()
            return

        figure = self._build_scene_figure()
        html_document = _figure_to_html(figure)
        self._load_html(html_document)

    def _load_mask(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Segmentation Mask",
            self._last_directory,
            "NIfTI Files (*.nii *.nii.gz);;All Files (*)",
        )
        if not file_path:
            return

        self._last_directory = os.path.dirname(file_path)
        try:
            verts_world, faces, mask_shape = _build_mask_mesh(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Mask Load Failed", str(exc))
            return

        self._current_seg_path = file_path
        self._current_mask_shape = mask_shape
        self._current_verts = verts_world
        self._current_faces = faces
        self._current_results = []
        self._plan_ready = False
        self._current_display_mode = "max_diameter"
        self.show_screws_button.setChecked(True)
        self.show_bbox_button.setChecked(True)
        self._current_bbox_visible = True
        self.run_button.setText("Run Planning")
        self._set_loaded_state(True)
        self._render_scene()

    def _load_results(self):
        if not self._current_seg_path:
            QMessageBox.information(self, "No Mask Loaded", "Load a segmentation mask before loading results.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Planning Results",
            self._last_directory,
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            QMessageBox.critical(self, "Load Results Failed", f"Could not read results file.\n\n{type(exc).__name__}: {exc}")
            return

        source_file = payload.get("source_file")
        if source_file and os.path.abspath(source_file) != os.path.abspath(self._current_seg_path):
            QMessageBox.warning(
                self,
                "Mask Mismatch",
                "This results file was exported from a different mask. Load the matching mask first.",
            )
            return

        results = payload.get("results", [])
        if not isinstance(results, list):
            QMessageBox.critical(self, "Load Results Failed", "The results file is not in the expected format.")
            return

        self._current_results = results
        self._plan_ready = bool(results)
        self._current_display_mode = "max_diameter"
        self.show_screws_button.setChecked(True)
        self.run_button.setText("Export Plan Data" if self._plan_ready else "Run Planning")
        self._render_scene()

        if self._planning_console is not None:
            self._planning_console.clear_output()
            self._planning_console.append_text(f"Loaded planning results from: {file_path}")

    def _run_planning(self):
        if not self._current_seg_path:
            QMessageBox.information(self, "No Mask Loaded", "Load a segmentation mask before running planning.")
            return

        if self._plan_ready:
            self._export_planning_data()
            return

        if self._planning_console is None:
            self._planning_console = PlanningConsole(self)
        console = self._planning_console
        console.clear_output()
        console.append_text(f"Running planning on: {self._current_seg_path}")
        console.show()
        console.raise_()
        console.activateWindow()

        self._set_busy_state(True)
        self.run_button.setText("Running...")
        self._planning_worker = PlanningWorker(self._current_seg_path)
        self._planning_worker.log.connect(console.append_text)
        self._planning_worker.failed.connect(lambda message: console.append_text(f"ERROR: {message}"))
        self._planning_worker.results_ready.connect(self._on_planning_ready)
        self._planning_worker.failed.connect(self._on_planning_failed)
        self._planning_worker.finished.connect(self._on_planning_finished)
        self._planning_worker.start()

    def _on_planning_ready(self, results):
        self._current_results = results
        self._plan_ready = True
        self.run_button.setText("Export Plan Data")
        self._render_scene()

    def _on_planning_failed(self, message):
        QMessageBox.critical(self, "Planning Failed", message)

    def _on_planning_finished(self):
        self._planning_worker = None
        self._set_busy_state(False)
        self._set_loaded_state(self._current_verts is not None)

    def _serialize_plan_data(self):
        return {
            "source_file": self._current_seg_path,
            "mask_shape": list(self._current_mask_shape) if self._current_mask_shape is not None else None,
            "planning_complete": self._plan_ready,
            "parameters": {
                "preset": self._args.preset,
                "screw_mode": self._args.screw_mode,
                "theme": self._args.theme,
                "mesh_opacity": self._args.mesh_opacity,
                "show_safety_planes": self._args.show_safety_planes,
                "hide_bounding_box": self._args.hide_bounding_box,
                "hide_trajectory_lines": self._args.hide_trajectory_lines,
                "hide_entry_markers": self._args.hide_entry_markers,
                "show_tip_markers": self._args.show_tip_markers,
                "v2_neon_trajectories": self._args.v2_neon_trajectories,
                "v2_gold_screws": self._args.v2_gold_screws,
                "v2_threaded_screws": self._args.v2_threaded_screws,
                "v2_safety_planes": self._args.v2_safety_planes,
                "fallback_diameter": self._args.fallback_diameter,
            },
            "results": [
                _json_safe(result)
                for result in self._current_results
            ],
        }

    def _export_planning_data(self):
        if not self._plan_ready or not self._current_results:
            QMessageBox.information(self, "Nothing to Export", "Run planning first to generate exportable data.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Planning Data",
            "planning_data.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_path:
            return

        if not file_path.lower().endswith(".json"):
            file_path += ".json"

        payload = self._serialize_plan_data()
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        if self._planning_console is not None:
            self._planning_console.append_text(f"Exported planning data to: {file_path}")

    def _export_image(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "visualization.png",
            "PNG Files (*.png);;All Files (*)",
        )
        if not file_path:
            return
        pixmap = self.view.grab()
        pixmap.save(file_path)

    def closeEvent(self, event):  # pragma: no cover - UI lifecycle
        if self._current_html_path and os.path.exists(self._current_html_path):
            try:
                os.remove(self._current_html_path)
            except OSError:
                pass
        super().closeEvent(event)


def main():
    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)

    args = build_parser().parse_args()
    window = GeometryPlanningWindow(args)
    window.show()

    if owns_app:
        print("[PyQt6] Starting event loop...")
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
