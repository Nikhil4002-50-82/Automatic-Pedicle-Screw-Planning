import argparse
import json
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import nibabel as nib
import numpy as np
from PyQt6.QtCore import QCoreApplication, QThread, QTimer, Qt, QUrl, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QHeaderView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
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
    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PlanningConsolePanel")
        self.setWindowTitle("Planning Console")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setModal(False)
        self.setMinimumSize(940, 720)
        self.resize(1040, 780)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setSizeGripEnabled(True)

        self._table_headers = [
            "VERTEBRA",
            "SIDE",
            "DIAMETER (mm)",
            "LENGTH (mm)",
            "AXIAL ∠",
            "SAGITTAL ∠",
            "ENTRY POINT",
        ]

        self.setStyleSheet(
            "background-color: rgba(10, 16, 26, 0.98);"
            "border-left: 1px solid rgba(151, 164, 180, 0.22);"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title_label = QLabel("Planning Console")
        title_label.setStyleSheet("color: #F8FAFC; font-size: 14px; font-weight: 700;")
        layout.addWidget(title_label)

        self.tabs = QTabWidget()

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(0, 0, 0, 0)
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
        log_layout.addWidget(self.output)

        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        table_layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, len(self._table_headers))
        self.table.setHorizontalHeaderLabels(self._table_headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setStyleSheet(
            "QTableWidget {"
            "  background-color: #FFFFFF;"
            "  color: #111827;"
            "  gridline-color: #D1D5DB;"
            "}"
            "QHeaderView::section {"
            "  background-color: #1F2937;"
            "  color: #FFFFFF;"
            "  padding: 8px;"
            "  font-weight: 700;"
            "  border: none;"
            "}"
        )
        table_layout.addWidget(self.table)

        self.tabs.addTab(log_tab, "Log")
        self.tabs.addTab(table_tab, "Table")
        layout.addWidget(self.tabs)

    def show_panel(self, tab_index=0):
        if self.isMinimized():
            self.showNormal()
        self.setVisible(True)
        self.raise_()
        self.activateWindow()
        self._position_over_parent()
        if tab_index == 1:
            self.show_table_tab()
        else:
            self.show_log_tab()

    def hide_panel(self):
        self.setVisible(False)

    def toggle_panel(self, tab_index=0):
        if self.isVisible():
            self.hide_panel()
        else:
            self.show_panel(tab_index=tab_index)

    def closeEvent(self, event):  # pragma: no cover - UI lifecycle
        self.hide()
        self.closed.emit()
        event.ignore()

    def _position_over_parent(self):
        parent = self.parentWidget()
        if parent is None:
            return

        parent_top_left = parent.mapToGlobal(parent.rect().topLeft())
        x = parent_top_left.x() + parent.width() - self.width() - 24
        y = parent_top_left.y() + max(24, (parent.height() - self.height()) // 2)
        self.move(x, y)

    def append_text(self, text):
        if not text:
            return
        self.output.appendPlainText(text.rstrip("\n"))
        self.output.ensureCursorVisible()

    def clear_output(self):
        self.output.clear()

    def clear_table(self):
        self.table.setRowCount(0)

    def clear_all(self):
        self.clear_output()
        self.clear_table()

    def show_log_tab(self):
        self.tabs.setCurrentIndex(0)

    def show_table_tab(self):
        self.tabs.setCurrentIndex(1)

    def set_results(self, results):
        self.clear_table()
        for result in results or []:
            self._add_result_row(result)

    def _add_result_row(self, result):
        row = self.table.rowCount()
        self.table.insertRow(row)

        entry = result.get("entry", [])
        if hasattr(entry, "tolist"):
            entry = entry.tolist()
        entry_text = "[" + ", ".join(f"{float(value):.1f}" for value in entry) + "]" if entry else ""

        values = [
            result.get("vertebra", ""),
            result.get("side", ""),
            f"{result.get('diameter', '')} mm",
            f"{float(result.get('length', 0.0)):.1f} mm",
            f"{float(result.get('axial_angle', 0.0)):.1f}°",
            f"{float(result.get('sagittal_angle', 0.0)):.1f}°",
            entry_text,
        ]

        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, column, item)


class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setStyleSheet(
            "background: qradialgradient(cx:0.5, cy:0.5, radius:0.92, fx:0.5, fy:0.5, "
            "stop:0 rgba(0, 0, 0, 0.36), stop:0.58 rgba(0, 0, 0, 0.76), stop:1 rgba(0, 0, 0, 0.94));"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(1)

        panel = QWidget()
        panel.setFixedWidth(340)
        panel.setStyleSheet(
            "QWidget {"
            "  background-color: rgba(7, 12, 19, 0.90);"
            "  border: 1px solid rgba(148, 163, 184, 0.28);"
            "  border-radius: 18px;"
            "}"
        )
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(28, 24, 28, 24)
        panel_layout.setSpacing(8)

        self.title_label = QLabel("Loading")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet(
            "color: #F8FAFC; font-size: 20px; font-weight: 700; letter-spacing: 0.4px;"
        )
        self.detail_label = QLabel("Please wait while the viewer updates.")
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet("color: #CBD5E1; font-size: 12px;")

        panel_layout.addWidget(self.title_label)
        panel_layout.addWidget(self.detail_label)

        center_row = QHBoxLayout()
        center_row.addStretch(1)
        center_row.addWidget(panel)
        center_row.addStretch(1)
        layout.addLayout(center_row)
        layout.addStretch(1)

    def set_message(self, title, detail):
        self.title_label.setText(title)
        self.detail_label.setText(detail)


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
        self._current_display_mode = "trajectories"
        self._current_bbox_visible = False
        self._current_html_path = None
        self._planning_worker = None
        self._planning_console = None
        self._plan_ready = False
        self._current_mask_shape = None
        self._scene_meta = {}
        self._view_ready = False
        self._pending_mesh_opacity = None
        self._last_directory = os.path.expanduser("~/Downloads")

        self.setWindowTitle("Pedicle Screw Planner Visualization")
        self._build_ui()
        self._render_blank_view()
        self._set_loaded_state(False)

    def _build_ui(self):
        container = QWidget()
        root_layout = QHBoxLayout(container)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        container.setStyleSheet("background-color: #0B1320;")

        if QWebEngineView is None:
            raise RuntimeError("PyQt6.QtWebEngineWidgets is required to display the unified viewer.")

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.view = QWebEngineView()
        content_layout.addWidget(self.view, 1)

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
        self.console_button = QPushButton("Log / Table")
        self.console_button.setCheckable(True)
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
            self.console_button,
        ):
            button.setStyleSheet(button_style)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setCheckable(True)

        self.load_button.setCheckable(False)
        self.load_results_button.setCheckable(False)
        self.run_button.setCheckable(False)
        self.console_button.setCheckable(True)
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

        self.show_screws_button.setChecked(False)
        self.show_traj_button.setChecked(True)
        self.show_bbox_button.setChecked(False)
        self.hide_bbox_button.setChecked(True)

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
        controls_layout.addWidget(self.console_button)
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
        self.console_button.clicked.connect(self._toggle_planning_console)
        self.export_button.clicked.connect(self._export_image)

        content_layout.addWidget(controls_panel)
        content_layout.addWidget(self.export_button)

        root_layout.addWidget(content_widget, 1)

        self._planning_console = PlanningConsole(self)
        self._planning_console.closed.connect(self._hide_planning_console)
        self._planning_console.hide_panel()
        self.console_button.setChecked(False)

        self.setCentralWidget(container)
        self._loading_overlay = LoadingOverlay(container)
        self._loading_overlay.hide()
        self._loading_overlay.raise_()
        self.view.loadFinished.connect(self._on_view_load_finished)
        self.resize(1320, 940)

    def _set_loaded_state(self, loaded):
        self.run_button.setEnabled(loaded)
        self.load_results_button.setEnabled(loaded)
        self.show_screws_button.setEnabled(loaded)
        self.show_traj_button.setEnabled(loaded)
        self.show_bbox_button.setEnabled(loaded)
        self.hide_bbox_button.setEnabled(loaded)
        self.opacity_slider.setEnabled(loaded)
        self.console_button.setEnabled(True)

    def _set_busy_state(self, busy):
        self.load_button.setEnabled(not busy)
        self.load_results_button.setEnabled((not busy) and self._current_verts is not None)
        self.run_button.setEnabled((not busy) and self._current_verts is not None)
        self.show_screws_button.setEnabled((not busy) and self._current_verts is not None)
        self.show_traj_button.setEnabled((not busy) and self._current_verts is not None)
        self.show_bbox_button.setEnabled((not busy) and self._current_verts is not None)
        self.hide_bbox_button.setEnabled((not busy) and self._current_verts is not None)
        self.opacity_slider.setEnabled((not busy) and self._current_verts is not None)
        self.console_button.setEnabled(True)

    def _show_planning_console(self, tab_index=0):
        console = self._ensure_planning_console(tab_index=tab_index)
        console.show_panel(tab_index=tab_index)
        self.console_button.setChecked(True)
        return console

    def _hide_planning_console(self):
        if self._planning_console is not None:
            self._planning_console.hide_panel()
        self.console_button.setChecked(False)

    def _toggle_planning_console(self):
        if self._planning_console is not None and self._planning_console.isVisible() and not self._planning_console.isMinimized():
            self._hide_planning_console()
            return

        tab_index = 1 if self._plan_ready else 0
        self._show_planning_console(tab_index=tab_index)

    def _set_max_diameter_mode(self):
        self._current_display_mode = "max_diameter"
        self.show_screws_button.setChecked(True)
        self.show_traj_button.setChecked(False)
        self._apply_scene_visibility_state()

    def _set_trajectory_mode(self):
        self._current_display_mode = "trajectories"
        self.show_screws_button.setChecked(False)
        self.show_traj_button.setChecked(True)
        self._apply_scene_visibility_state()

    def _set_bbox_visibility(self, visible):
        self._current_bbox_visible = visible
        if visible:
            self.show_bbox_button.setChecked(True)
        else:
            self.hide_bbox_button.setChecked(True)
        self._apply_scene_visibility_state()

    def _on_opacity_changed(self, value):
        self.opacity_label.setText(f"Mesh Opacity: {value / 100.0:.2f}")
        self._apply_mesh_opacity(value / 100.0)

    def _render_scene_delayed(self):
        QTimer.singleShot(0, self._render_scene)

    def _set_loading_state(self, title, detail):
        self._loading_overlay.set_message(title, detail)
        self._loading_overlay.setGeometry(self.centralWidget().rect())
        self._loading_overlay.show()
        self._loading_overlay.raise_()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _clear_loading_state(self):
        self._loading_overlay.hide()

    def _ensure_planning_console(self, tab_index=0):
        if self._planning_console is None:
            self._planning_console = PlanningConsole(self)
        return self._planning_console

    def _apply_mesh_opacity(self, opacity_value):
        if self._current_verts is None or self._current_faces is None:
            return

        if not self._view_ready:
            self._pending_mesh_opacity = opacity_value
            return

        self._pending_mesh_opacity = None
        self.view.page().runJavaScript(
            f"""
            (function() {{
                const plot = document.querySelector('.js-plotly-plot');
                if (!plot || !window.Plotly) return;
                Plotly.restyle(plot, {{opacity: [{opacity_value:.3f}]}}, [0]);
            }})();
            """
        )

    def _restyle_plot(self, payload, indices):
        if not self._view_ready:
            return
        self.view.page().runJavaScript(
            f"""
            (function() {{
                const plot = document.querySelector('.js-plotly-plot');
                if (!plot || !window.Plotly) return;
                Plotly.restyle(plot, {json.dumps(payload)}, {json.dumps(indices)});
            }})();
            """
        )

    def _apply_scene_visibility_state(self):
        if not self._scene_meta or self._current_verts is None or self._current_faces is None:
            return

        screw_mode_indices = self._scene_meta.get("screw_mode_indices", [])
        bbox_indices = self._scene_meta.get("bbox_indices", [])
        if self._current_display_mode == "trajectories":
            self._restyle_plot({"visible": self._scene_meta.get("screw_mode_traj_vis", [])}, screw_mode_indices)
        else:
            self._restyle_plot({"visible": self._scene_meta.get("screw_mode_screws_vis", [])}, screw_mode_indices)

        if bbox_indices:
            self._restyle_plot({"visible": [self._current_bbox_visible] * len(bbox_indices)}, bbox_indices)

    def _on_view_load_finished(self, ok):
        self._view_ready = bool(ok)
        if ok:
            QTimer.singleShot(120, self._clear_loading_state)
            if self._current_verts is not None and self._current_faces is not None:
                opacity_value = self._pending_mesh_opacity
                if opacity_value is None:
                    opacity_value = self.opacity_slider.value() / 100.0
                self._apply_mesh_opacity(opacity_value)
                self._apply_scene_visibility_state()
        else:
            self._clear_loading_state()

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

        self._view_ready = False
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

        self._view_ready = False
        figure = self._build_scene_figure()
        scene_meta = getattr(figure.layout, "meta", None) or {}
        if hasattr(scene_meta, "to_plotly_json"):
            scene_meta = scene_meta.to_plotly_json()
        self._scene_meta = scene_meta
        html_document = _figure_to_html(figure)
        self._load_html(html_document)

    def resizeEvent(self, event):  # pragma: no cover - UI lifecycle
        super().resizeEvent(event)
        if hasattr(self, "_loading_overlay"):
            self._loading_overlay.setGeometry(self.centralWidget().rect())

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
        self._set_loading_state("Loading Mask", "Building the anatomical surface from the selected segmentation.")
        try:
            verts_world, faces, mask_shape = _build_mask_mesh(file_path)
        except Exception as exc:
            self._clear_loading_state()
            QMessageBox.critical(self, "Mask Load Failed", str(exc))
            return

        self._current_seg_path = file_path
        self._current_mask_shape = mask_shape
        self._current_verts = verts_world
        self._current_faces = faces
        self._current_results = []
        self._plan_ready = False
        self._current_display_mode = "trajectories"
        self.show_screws_button.setChecked(False)
        self.show_traj_button.setChecked(True)
        self.show_bbox_button.setChecked(False)
        self.hide_bbox_button.setChecked(True)
        self._current_bbox_visible = False
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

        self._set_loading_state("Loading Results", "Restoring planning data onto the currently loaded mask.")
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            self._clear_loading_state()
            QMessageBox.critical(self, "Load Results Failed", f"Could not read results file.\n\n{type(exc).__name__}: {exc}")
            return

        source_file = payload.get("source_file")
        if source_file and os.path.abspath(source_file) != os.path.abspath(self._current_seg_path):
            self._clear_loading_state()
            QMessageBox.warning(
                self,
                "Mask Mismatch",
                "This results file was exported from a different mask. Load the matching mask first.",
            )
            return

        results = payload.get("results", [])
        if not isinstance(results, list):
            self._clear_loading_state()
            QMessageBox.critical(self, "Load Results Failed", "The results file is not in the expected format.")
            return

        self._current_results = results
        self._plan_ready = bool(results)
        self.run_button.setText("Export Plan Data" if self._plan_ready else "Run Planning")
        self._render_scene()

        console = self._show_planning_console(tab_index=1)
        console.clear_output()
        console.set_results(results)
        console.show_table_tab()
        console.append_text(f"Loaded planning results from: {file_path}")

    def _run_planning(self):
        if not self._current_seg_path:
            QMessageBox.information(self, "No Mask Loaded", "Load a segmentation mask before running planning.")
            return

        if self._plan_ready:
            self._export_planning_data()
            return

        console = self._show_planning_console(tab_index=0)
        console.clear_all()
        console.append_text(f"Running planning on: {self._current_seg_path}")

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
        if self._planning_console is not None:
            self._planning_console.set_results(results)
            self._planning_console.show_table_tab()
            self._show_planning_console(tab_index=1)
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

    def _export_stem(self):
        if not self._current_seg_path:
            return "planning_data"

        stem = Path(self._current_seg_path).name
        if stem.endswith(".nii.gz"):
            stem = stem[:-7]
        else:
            stem = Path(stem).stem
        return f"{stem}_planning_data"

    def _export_planning_data(self):
        if not self._plan_ready or not self._current_results:
            QMessageBox.information(self, "Nothing to Export", "Run planning first to generate exportable data.")
            return

        default_name = f"{self._export_stem()}.json"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Planning Data",
            default_name,
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
