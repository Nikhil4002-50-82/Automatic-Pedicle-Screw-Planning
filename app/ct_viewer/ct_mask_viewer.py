from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from PyQt6.QtCore import QObject, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ct_viewer.ui import io as ct_io  # noqa: E402
    from ct_viewer.ui import mask_viz as ct_mask_viz  # noqa: E402
    from ct_viewer.ui import rendering as ct_rendering  # noqa: E402
    from ct_viewer.ui import widgets as ct_widgets  # noqa: E402
else:
    from .ui import io as ct_io  # noqa: E402
    from .ui import mask_viz as ct_mask_viz  # noqa: E402
    from .ui import rendering as ct_rendering  # noqa: E402
    from .ui import widgets as ct_widgets  # noqa: E402

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - optional dependency
    sitk = None


WINDOW_PRESETS = {
    "Auto": None,
    "Bone": (300, 1500),
    "Soft Tissue": (50, 400),
    "Lung": (-600, 1500),
    "Custom": None,
}

MASK_COLORS = [
    (0, 208, 132),
    (255, 140, 92),
    (91, 141, 239),
    (255, 196, 61),
    (188, 118, 255),
    (255, 87, 124),
    (43, 214, 255),
    (171, 235, 82),
]

ORIENTATION_TITLES = {
    "axial": "Axial",
    "coronal": "Coronal",
    "sagittal": "Sagittal",
}

SLICE_AXES = {
    "sagittal": 0,
    "coronal": 1,
    "axial": 2,
}


@dataclass
class MaskLayer:
    name: str
    path: str
    image: nib.spatialimages.SpatialImage
    color: tuple[int, int, int]
    visible: bool = True
    voxel_count: int | None = None


@dataclass
class VolumeSummary:
    minimum: float
    maximum: float
    low_percentile: float
    high_percentile: float
    is_constant: bool


@dataclass
class CTVolume:
    path: str
    image: nib.spatialimages.SpatialImage
    shape: tuple[int, int, int]
    zooms: tuple[float, float, float]
    summary: VolumeSummary


@dataclass
class MaskLoadResult:
    layers: list[MaskLayer]
    warnings: list[str]


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def _is_nifti_path(path: str | Path) -> bool:
    lower = str(path).lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def _squeeze_to_3d(image: nib.spatialimages.SpatialImage) -> nib.spatialimages.SpatialImage:
    squeezed = nib.squeeze_image(image)
    if len(squeezed.shape) != 3:
        if len(squeezed.shape) == 4:
            volumes = nib.funcs.four_to_three(squeezed)
            if volumes:
                return volumes[0]
        raise ValueError(f"Expected a 3D volume, got shape {squeezed.shape}")
    return squeezed


def load_nifti_image(path: str) -> nib.spatialimages.SpatialImage:
    image = nib.load(path)
    canonical = nib.as_closest_canonical(_squeeze_to_3d(image))
    if len(canonical.shape) != 3:
        raise ValueError(f"Expected a 3D volume, got shape {canonical.shape}")
    return canonical


def load_dicom_series(folder: str) -> nib.spatialimages.SpatialImage:
    if sitk is None:
        raise RuntimeError("SimpleITK is required to load DICOM CT series.")

    source_folder = Path(folder)
    if source_folder.is_file():
        source_folder = source_folder.parent
    if not source_folder.exists():
        raise FileNotFoundError(f"DICOM source not found: {source_folder}")

    reader = sitk.ImageSeriesReader()
    series_ids = list(reader.GetGDCMSeriesIDs(str(source_folder)))
    if not series_ids:
        raise ValueError(f"No DICOM series were found in {source_folder}")

    def series_size(series_id: str) -> int:
        return len(reader.GetGDCMSeriesFileNames(str(source_folder), series_id))

    series_id = max(series_ids, key=series_size)
    file_names = reader.GetGDCMSeriesFileNames(str(source_folder), series_id)
    if not file_names:
        raise ValueError(f"No readable DICOM files were found in {source_folder}")

    reader.SetFileNames(file_names)
    image = reader.Execute()

    array = sitk.GetArrayFromImage(image)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D DICOM series, got shape {array.shape}")
    array = np.transpose(array, (2, 1, 0)).astype(np.float32, copy=False)

    spacing = np.asarray(image.GetSpacing(), dtype=np.float32)
    origin = np.asarray(image.GetOrigin(), dtype=np.float32)
    direction = np.asarray(image.GetDirection(), dtype=np.float32).reshape(3, 3)

    affine_lps = np.eye(4, dtype=np.float32)
    affine_lps[:3, :3] = direction @ np.diag(spacing)
    affine_lps[:3, 3] = origin

    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float32)
    affine_ras = lps_to_ras @ affine_lps

    return nib.as_closest_canonical(nib.Nifti1Image(array, affine_ras))


def load_spatial_image(path: str) -> nib.spatialimages.SpatialImage:
    candidate = Path(path)
    if candidate.is_dir():
        return load_dicom_series(str(candidate))
    if _is_nifti_path(candidate):
        return load_nifti_image(str(candidate))

    raise FileNotFoundError(f"Input CT not found: {path}")


def summarize_volume(image: nib.spatialimages.SpatialImage, max_samples_per_axis: int = 96) -> VolumeSummary:
    shape = image.shape[:3]
    slices = tuple(slice(None, None, max(1, int(np.ceil(size / max_samples_per_axis)))) for size in shape)
    sampled = np.asarray(image.dataobj[slices], dtype=np.float32)
    finite = sampled[np.isfinite(sampled)]
    if finite.size == 0:
        return VolumeSummary(0.0, 0.0, 0.0, 1.0, True)

    minimum = float(finite.min())
    maximum = float(finite.max())
    low, high = np.percentile(finite, [1.0, 99.0])
    if float(high) <= float(low):
        high = low + 1.0
    return VolumeSummary(
        minimum=minimum,
        maximum=maximum,
        low_percentile=float(low),
        high_percentile=float(high),
        is_constant=minimum == maximum,
    )


def extract_slice(image: nib.spatialimages.SpatialImage, orientation: str, indices: list[int]) -> np.ndarray:
    if orientation == "axial":
        base = np.asarray(image.dataobj[:, :, indices[2]], dtype=np.float32)
        return np.rot90(base)
    if orientation == "coronal":
        base = np.asarray(image.dataobj[:, indices[1], :], dtype=np.float32)
        return np.rot90(base)
    if orientation == "sagittal":
        base = np.asarray(image.dataobj[indices[0], :, :], dtype=np.float32)
        return np.rot90(np.flipud(base))
    raise ValueError(f"Unsupported orientation: {orientation}")


def crosshair_position(shape: tuple[int, int, int], orientation: str, indices: list[int]) -> tuple[int, int]:
    nx, ny, nz = shape
    if orientation == "axial":
        return indices[0], ny - 1 - indices[1]
    if orientation == "coronal":
        return indices[0], nz - 1 - indices[2]
    if orientation == "sagittal":
        return ny - 1 - indices[1], nz - 1 - indices[2]
    raise ValueError(f"Unsupported orientation: {orientation}")


def orientation_labels(orientation: str) -> tuple[str, str, str, str]:
    if orientation == "axial":
        return "R", "L", "A", "P"
    if orientation == "coronal":
        return "R", "L", "S", "I"
    if orientation == "sagittal":
        return "A", "P", "S", "I"
    raise ValueError(f"Unsupported orientation: {orientation}")


def display_spacing(zooms: tuple[float, float, float], orientation: str) -> tuple[float, float]:
    if orientation == "axial":
        return zooms[0], zooms[1]
    if orientation == "coronal":
        return zooms[0], zooms[2]
    if orientation == "sagittal":
        return zooms[1], zooms[2]
    raise ValueError(f"Unsupported orientation: {orientation}")


def physical_display_size(width: int, height: int, width_spacing: float, height_spacing: float) -> tuple[int, int]:
    base_spacing = max(min(width_spacing, height_spacing), 1e-6)
    scaled_width = max(1, int(round(width * (width_spacing / base_spacing))))
    scaled_height = max(1, int(round(height * (height_spacing / base_spacing))))

    longest_side = max(scaled_width, scaled_height)
    if longest_side > 1800:
        factor = 1800 / float(longest_side)
        scaled_width = max(1, int(round(scaled_width * factor)))
        scaled_height = max(1, int(round(scaled_height * factor)))

    return scaled_width, scaled_height


def grayscale_rgba(ct_slice: np.ndarray, center: int, width: int) -> np.ndarray:
    safe_width = max(1, int(width))
    lower = center - (safe_width / 2.0)
    upper = center + (safe_width / 2.0)
    scaled = np.clip((ct_slice - lower) / max(upper - lower, 1.0), 0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    alpha = np.full_like(gray, 255, dtype=np.uint8)
    return np.ascontiguousarray(np.stack((gray, gray, gray, alpha), axis=-1))


def blend_mask(rgba: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], opacity: float) -> None:
    if not np.any(mask):
        return

    alpha = float(np.clip(opacity, 0.0, 1.0))
    color_array = np.asarray(color, dtype=np.float32)
    base_pixels = rgba[mask, :3].astype(np.float32)
    rgba[mask, :3] = ((1.0 - alpha) * base_pixels + alpha * color_array).astype(np.uint8)

    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    outline = mask & ~eroded
    if np.any(outline):
        rgba[outline, :3] = np.asarray(color, dtype=np.uint8)


def qimage_from_rgba(rgba: np.ndarray) -> QImage:
    height, width, _ = rgba.shape
    bytes_per_line = width * 4
    return QImage(rgba.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888).copy()


class SliceCanvas(QLabel):
    imageClicked = pyqtSignal(str, int, int)
    wheelStepped = pyqtSignal(str, int)

    def __init__(self, orientation: str) -> None:
        super().__init__()
        self.orientation = orientation
        self._base_pixmap = QPixmap()
        self._logical_size = QSize()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(260, 260)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setText("Load a CT to start")
        self.setWordWrap(True)
        self.setStyleSheet(
            """
            QLabel {
                background: #0f1722;
                border: 1px solid #243244;
                border-radius: 14px;
                color: #9fb0c3;
                font-size: 13px;
            }
            """
        )

    def clear_image(self, message: str) -> None:
        self._base_pixmap = QPixmap()
        self._logical_size = QSize()
        self.clear()
        self.setText(message)

    def set_image(self, pixmap: QPixmap, logical_size: tuple[int, int]) -> None:
        self._base_pixmap = pixmap
        self._logical_size = QSize(logical_size[0], logical_size[1])
        self._refresh_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_pixmap()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            coords = self._map_to_image(event.position().x(), event.position().y())
            if coords is not None:
                self.imageClicked.emit(self.orientation, coords[0], coords[1])
        super().mousePressEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta != 0:
            self.wheelStepped.emit(self.orientation, 1 if delta > 0 else -1)
        event.accept()

    def _refresh_pixmap(self) -> None:
        if self._base_pixmap.isNull():
            return

        scaled = self._base_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def _map_to_image(self, x_pos: float, y_pos: float) -> tuple[int, int] | None:
        shown = self.pixmap()
        if shown is None or shown.isNull() or self._logical_size.isEmpty():
            return None

        x_offset = (self.width() - shown.width()) / 2.0
        y_offset = (self.height() - shown.height()) / 2.0
        if x_pos < x_offset or y_pos < y_offset:
            return None
        if x_pos > x_offset + shown.width() or y_pos > y_offset + shown.height():
            return None

        logical_width = self._logical_size.width()
        logical_height = self._logical_size.height()
        image_x = int((x_pos - x_offset) * logical_width / max(shown.width(), 1))
        image_y = int((y_pos - y_offset) * logical_height / max(shown.height(), 1))
        image_x = clamp(image_x, 0, logical_width - 1)
        image_y = clamp(image_y, 0, logical_height - 1)
        return image_x, image_y


class SliceView(QWidget):
    sliceChanged = pyqtSignal(str, int)
    crosshairRequested = pyqtSignal(str, int, int)

    def __init__(self, orientation: str) -> None:
        super().__init__()
        self.orientation = orientation

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QLabel(ORIENTATION_TITLES[orientation])
        self.title_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #edf2f7;")

        self.canvas = SliceCanvas(orientation)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(lambda value: self.sliceChanged.emit(self.orientation, value))

        self.footer_label = QLabel("No volume loaded")
        self.footer_label.setStyleSheet("font-size: 12px; color: #92a4b8;")

        self.canvas.imageClicked.connect(self.crosshairRequested.emit)
        self.canvas.wheelStepped.connect(self._step_from_wheel)

        layout.addWidget(self.title_label)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.slider)
        layout.addWidget(self.footer_label)

    def clear_view(self) -> None:
        self.slider.blockSignals(True)
        self.slider.setRange(0, 0)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.blockSignals(False)
        self.canvas.clear_image("Load a CT to start")
        self.footer_label.setText("No volume loaded")

    def set_slice_bounds(self, max_index: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setRange(0, max_index)
        self.slider.setValue(min(self.slider.value(), max_index))
        self.slider.setEnabled(True)
        self.slider.blockSignals(False)

    def set_slice_index(self, value: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)

    def set_image(self, pixmap: QPixmap, logical_size: tuple[int, int]) -> None:
        self.canvas.set_image(pixmap, logical_size)

    def set_footer(self, text: str) -> None:
        self.footer_label.setText(text)

    def _step_from_wheel(self, orientation: str, step: int) -> None:
        new_value = clamp(self.slider.value() + step, self.slider.minimum(), self.slider.maximum())
        if new_value != self.slider.value():
            self.slider.setValue(new_value)


class CTLoadWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str, str)

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def run(self) -> None:
        try:
            payload = ct_io.load_ct_volume(self.path)
        except Exception as exc:  # pragma: no cover - UI error path
            self.failed.emit("Could not load CT", str(exc))
            return

        self.finished.emit(payload)


class MaskLoadWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str, str)

    def __init__(self, paths: list[str], ct_image: nib.spatialimages.SpatialImage, start_index: int) -> None:
        super().__init__()
        self.paths = paths
        self.ct_image = ct_image
        self.start_index = start_index

    def run(self) -> None:
        try:
            result = ct_io.load_mask_layers(self.paths, self.ct_image, self.start_index)
        except Exception as exc:  # pragma: no cover - UI error path
            self.failed.emit("Could not load masks", str(exc))
            return

        self.finished.emit(result)


class MaskPreviewWorker(QObject):
    finished = pyqtSignal(int, object)
    failed = pyqtSignal(str, str)

    def __init__(self, generation: int, ct_volume: CTVolume, layers: list[MaskLayer]) -> None:
        super().__init__()
        self.generation = generation
        self.ct_volume = ct_volume
        self.layers = layers

    def run(self) -> None:
        try:
            figure = ct_mask_viz.build_mask_preview_figure(self.ct_volume, self.layers)
            payload = figure.to_plotly_json()
        except Exception as exc:  # pragma: no cover - UI error path
            self.failed.emit("Could not build 3D mask preview", str(exc))
            return

        self.finished.emit(self.generation, payload)


class CTMaskViewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ct_volume: CTVolume | None = None
        self.ct_zooms = (1.0, 1.0, 1.0)
        self.mask_layers: list[MaskLayer] = []
        self.current_indices = [0, 0, 0]
        self.auto_window = (300, 1500)
        self.ct_intensity_summary = "Intensity range: unavailable"
        self.ct_slice_cache: dict[tuple[str, int], np.ndarray] = {}
        self.mask_slice_cache: dict[tuple[int, str, int], np.ndarray] = {}
        self._ct_thread: QThread | None = None
        self._mask_thread: QThread | None = None
        self._ct_worker: CTLoadWorker | None = None
        self._mask_worker: MaskLoadWorker | None = None
        self._mask_preview_thread: QThread | None = None
        self._mask_preview_worker: MaskPreviewWorker | None = None
        self._mask_preview_generation = 0
        self._mask_preview_needs_refresh = False
        self._window_control_lock = False

        self.setWindowTitle("CT and Segmentation Viewer")
        self.resize(1680, 980)

        self._build_ui()
        self._build_actions()
        self._apply_styles()
        self.clear_viewer()

    def _build_ui(self) -> None:
        central = QWidget()
        outer_layout = QHBoxLayout(central)
        outer_layout.setContentsMargins(18, 18, 18, 18)
        outer_layout.setSpacing(18)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_view_panel())
        splitter.setSizes([300, 1320])

        outer_layout.addWidget(splitter)
        self.setCentralWidget(central)

        status = QStatusBar()
        status.showMessage("Load a CT file or DICOM folder to begin.")
        self.setStatusBar(status)

    def _build_control_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidePanel")
        panel.setFixedWidth(340)
        panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setFixedWidth(304)
        scroll.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)

        title = QLabel("CT + Mask Viewer")
        title.setObjectName("panelTitle")

        subtitle = QLabel(
            "Fast desktop viewer for CT volumes and accompanying segmentation masks."
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("panelSubtitle")

        self.load_ct_button = QPushButton("Load CT File")
        self.load_dicom_button = QPushButton("Load DICOM Folder")
        self.add_masks_button = QPushButton("Add Mask Files")
        self.add_mask_folder_button = QPushButton("Add Mask Folder")
        self.reset_view_button = QPushButton("Reset Window + Slices")
        self.clear_masks_button = QPushButton("Clear Masks")

        for button in (
            self.load_ct_button,
            self.load_dicom_button,
            self.add_masks_button,
            self.add_mask_folder_button,
            self.reset_view_button,
            self.clear_masks_button,
        ):
            button.setMinimumHeight(42)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.volume_info_label = QLabel("No CT loaded")
        self.volume_info_label.setWordWrap(True)
        self.volume_info_label.setObjectName("infoBlock")
        self.volume_info_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.volume_info_label.setFixedHeight(118)

        self.window_preset_combo = QComboBox()
        self.window_preset_combo.addItems(list(WINDOW_PRESETS.keys()))

        self.window_center_slider = QSlider(Qt.Orientation.Horizontal)
        self.window_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_opacity_slider.setRange(0, 100)
        self.overlay_opacity_slider.setValue(45)

        self.window_center_value = QLabel("300")
        self.window_width_value = QLabel("1500")
        self.overlay_opacity_value = QLabel("45%")

        controls_form = QFormLayout()
        controls_form.setSpacing(10)
        controls_form.addRow("Preset", self.window_preset_combo)
        controls_form.addRow("Center", self._build_slider_row(self.window_center_slider, self.window_center_value))
        controls_form.addRow("Width", self._build_slider_row(self.window_width_slider, self.window_width_value))
        controls_form.addRow("Mask Opacity", self._build_slider_row(self.overlay_opacity_slider, self.overlay_opacity_value))

        self.crosshair_checkbox = QCheckBox("Show crosshair")
        self.crosshair_checkbox.setChecked(True)

        self.mask_list = QListWidget()
        self.mask_list.setMinimumHeight(210)
        self.mask_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        hint = QLabel("Click inside any slice to jump the crosshair. Mouse wheel scrolls that plane.")
        hint.setWordWrap(True)
        hint.setObjectName("helperText")

        import_section = ct_widgets.CollapsibleSection("Import & Actions", expanded=False)
        import_section.content_layout.addWidget(self.load_ct_button)
        import_section.content_layout.addWidget(self.load_dicom_button)
        import_section.content_layout.addWidget(self.add_masks_button)
        import_section.content_layout.addWidget(self.add_mask_folder_button)
        import_section.content_layout.addWidget(self.reset_view_button)
        import_section.content_layout.addWidget(self.clear_masks_button)

        window_section = ct_widgets.CollapsibleSection("Windowing", expanded=False)
        windowing_layout = QVBoxLayout()
        windowing_layout.setSpacing(10)
        windowing_layout.addLayout(controls_form)
        windowing_layout.addWidget(self.crosshair_checkbox)
        window_section.content_layout.addLayout(windowing_layout)

        masks_section = ct_widgets.CollapsibleSection("Masks", expanded=True)
        masks_section.content_layout.addWidget(self.mask_list)
        masks_section.content_layout.addWidget(hint)

        content_layout.addWidget(title)
        content_layout.addWidget(subtitle)
        content_layout.addWidget(self.volume_info_label)
        content_layout.addWidget(import_section)
        content_layout.addWidget(window_section)
        content_layout.addWidget(masks_section, 1)
        content_layout.addStretch(1)

        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        self.load_ct_button.clicked.connect(self.select_ct_file)
        self.load_dicom_button.clicked.connect(self.select_dicom_folder)
        self.add_masks_button.clicked.connect(self.select_mask_files)
        self.add_mask_folder_button.clicked.connect(self.select_mask_folder)
        self.reset_view_button.clicked.connect(self.reset_view_state)
        self.clear_masks_button.clicked.connect(self.clear_masks)
        self.window_preset_combo.currentTextChanged.connect(self.apply_window_preset)
        self.window_center_slider.valueChanged.connect(self.on_window_slider_changed)
        self.window_width_slider.valueChanged.connect(self.on_window_slider_changed)
        self.overlay_opacity_slider.valueChanged.connect(self.on_overlay_opacity_changed)
        self.crosshair_checkbox.toggled.connect(lambda _: self.render_all_views())
        self.mask_list.itemChanged.connect(self.on_mask_item_changed)

        return panel

    def _build_view_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("viewPanel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        self.cursor_info_label = QLabel("Voxel: -, -, -    HU: -")
        self.cursor_info_label.setObjectName("cursorInfo")

        self.views = {
            "axial": ct_widgets.SliceView("axial"),
            "coronal": ct_widgets.SliceView("coronal"),
            "sagittal": ct_widgets.SliceView("sagittal"),
        }
        self.mask_viz = ct_mask_viz.MaskVisualizationPane()

        self.views["axial"].sliceChanged.connect(self.on_slice_changed)
        self.views["axial"].crosshairRequested.connect(self.on_view_clicked)

        axial_column = QSplitter(Qt.Orientation.Vertical)
        axial_column.setObjectName("maskAxialSplit")
        axial_column.setChildrenCollapsible(False)
        axial_column.setHandleWidth(0)
        axial_column.addWidget(self.mask_viz)
        axial_column.addWidget(self.views["axial"])
        axial_column.setSizes([560, 440])

        views_layout = QHBoxLayout()
        views_layout.setSpacing(14)
        views_layout.addWidget(axial_column, 1)
        for orientation in ("coronal", "sagittal"):
            view = self.views[orientation]
            view.sliceChanged.connect(self.on_slice_changed)
            view.crosshairRequested.connect(self.on_view_clicked)
            views_layout.addWidget(view, 1)

        layout.addWidget(self.cursor_info_label)
        layout.addLayout(views_layout, 1)
        return panel

    def _build_slider_row(self, slider: QSlider, value_label: QLabel) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        slider.setSingleStep(1)
        value_label.setMinimumWidth(56)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        value_label.setStyleSheet("color: #e4edf5;")

        row.addWidget(slider, 1)
        row.addWidget(value_label)
        return container

    def _build_actions(self) -> None:
        load_ct_action = QAction("Load CT File", self)
        load_ct_action.setShortcut(QKeySequence("Ctrl+O"))
        load_ct_action.triggered.connect(self.select_ct_file)

        load_dicom_action = QAction("Load DICOM Folder", self)
        load_dicom_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        load_dicom_action.triggered.connect(self.select_dicom_folder)

        add_mask_action = QAction("Add Mask Files", self)
        add_mask_action.setShortcut(QKeySequence("Ctrl+M"))
        add_mask_action.triggered.connect(self.select_mask_files)

        add_folder_action = QAction("Add Mask Folder", self)
        add_folder_action.setShortcut(QKeySequence("Ctrl+Shift+M"))
        add_folder_action.triggered.connect(self.select_mask_folder)

        reset_action = QAction("Reset Window + Slices", self)
        reset_action.setShortcut(QKeySequence("R"))
        reset_action.triggered.connect(self.reset_view_state)

        clear_masks_action = QAction("Clear Masks", self)
        clear_masks_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        clear_masks_action.triggered.connect(self.clear_masks)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(load_ct_action)
        file_menu.addAction(load_dicom_action)
        file_menu.addAction(add_mask_action)
        file_menu.addAction(add_folder_action)
        file_menu.addSeparator()
        file_menu.addAction(clear_masks_action)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(reset_action)

        self.addAction(load_ct_action)
        self.addAction(load_dicom_action)
        self.addAction(add_mask_action)
        self.addAction(add_folder_action)
        self.addAction(reset_action)
        self.addAction(clear_masks_action)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
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
        )

    def clear_viewer(self) -> None:
        self.ct_volume = None
        self.mask_layers = []
        self.ct_slice_cache.clear()
        self.mask_slice_cache.clear()
        self.volume_info_label.setText("No CT loaded")
        self.cursor_info_label.setText("Voxel: -, -, -    HU: -")
        for view in self.views.values():
            view.clear_view()
        if hasattr(self, "mask_viz"):
            self.mask_viz.clear_view()
        self._mask_preview_needs_refresh = False
        self.mask_list.clear()
        self.window_center_slider.setEnabled(False)
        self.window_width_slider.setEnabled(False)
        self.overlay_opacity_slider.setEnabled(False)
        self.crosshair_checkbox.setEnabled(False)
        self.window_preset_combo.setEnabled(False)
        self._set_loading_state("Load a CT file or DICOM folder to begin.", busy=False)

    def select_ct_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CT Volume",
            "",
            "CT Volumes (*.nii *.nii.gz *.dcm);;All Files (*.*)",
        )
        if path:
            self.load_ct_async(path)

    def select_dicom_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if folder:
            self.load_ct_async(folder)

    def select_mask_files(self) -> None:
        if not self._require_ct_loaded():
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Segmentation Masks",
            "",
            "NIfTI Files (*.nii *.nii.gz)",
        )
        if paths:
            self.load_masks_async(paths)

    def select_mask_folder(self) -> None:
        if not self._require_ct_loaded():
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Mask Folder")
        if not folder:
            return

        folder_path = Path(folder)
        mask_paths = sorted(str(path) for path in folder_path.glob("*.nii"))
        mask_paths.extend(sorted(str(path) for path in folder_path.glob("*.nii.gz")))
        if not mask_paths:
            QMessageBox.information(self, "No Masks Found", "The selected folder does not contain any NIfTI masks.")
            return

        self.load_masks_async(mask_paths)

    def load_ct_async(self, path: str) -> None:
        if self._ct_thread is not None:
            return

        self._set_loading_state("Loading CT...", busy=True)
        thread = QThread(self)
        worker = CTLoadWorker(path)
        self._ct_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_ct_loaded)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_ct_thread)
        self._ct_thread = thread
        thread.start()

    def load_masks_async(self, paths: list[str]) -> None:
        if self.ct_volume is None or self._mask_thread is not None:
            return

        seen_paths = {os.path.normcase(os.path.abspath(layer.path)) for layer in self.mask_layers}
        unique_paths = [path for path in paths if os.path.normcase(os.path.abspath(path)) not in seen_paths]
        if not unique_paths:
            self.statusBar().showMessage("All selected masks are already loaded.", 4000)
            return

        self._set_loading_state("Loading masks...", busy=True)
        thread = QThread(self)
        worker = MaskLoadWorker(unique_paths, self.ct_volume.image, len(self.mask_layers))
        self._mask_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_masks_loaded)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_mask_thread)
        self._mask_thread = thread
        thread.start()

    def clear_masks(self) -> None:
        if not self.mask_layers:
            return

        self.mask_layers = []
        self.mask_slice_cache.clear()
        self.refresh_mask_list()
        self.update_volume_info()
        self.render_all_views()
        self.refresh_mask_visualization()
        self.statusBar().showMessage("Cleared all masks.", 4000)

    def _on_ct_loaded(self, volume: CTVolume) -> None:
        self.ct_volume = volume
        self.ct_zooms = volume.zooms
        self.mask_layers = []
        self.ct_slice_cache.clear()
        self.mask_slice_cache.clear()
        self.current_indices = [dimension // 2 for dimension in volume.shape]
        self.ct_intensity_summary = (
            f"Intensity range: {volume.summary.minimum:.1f} to {volume.summary.maximum:.1f}"
        )

        self._configure_window_controls()
        self._configure_slice_views()
        self.refresh_mask_list()
        self.update_volume_info()
        self.render_all_views()
        self.refresh_mask_visualization()
        self._set_loading_state(f"Loaded CT: {Path(volume.path).name}", busy=False)

        if volume.summary.is_constant:
            QMessageBox.warning(
                self,
                "Volume Appears Empty",
                (
                    f"{Path(volume.path).name} has no visible intensity variation.\n\n"
                    "The loaded volume is constant, so the slice views will look blank.\n"
                    "If this is a segmentation export, load the original CT instead and use this file as a mask."
                ),
            )

    def _on_masks_loaded(self, result: MaskLoadResult) -> None:
        if result.layers:
            self.mask_layers.extend(result.layers)
            self.mask_slice_cache.clear()
            self.refresh_mask_list()
            self.update_volume_info()
            self.render_all_views()
            self.refresh_mask_visualization()
            self._set_loading_state(f"Loaded {len(result.layers)} mask(s).", busy=False)
        else:
            self._set_loading_state("No masks were loaded.", busy=False)

        if result.warnings:
            QMessageBox.information(self, "Mask Load Notes", "\n".join(result.warnings))

    def _on_worker_failed(self, title: str, message: str) -> None:
        self._set_loading_state("Load failed.", busy=False)
        self._show_error(title, message)

    def _clear_ct_thread(self) -> None:
        self._ct_thread = None
        self._ct_worker = None
        self._set_loading_state(self.statusBar().currentMessage(), busy=self._mask_thread is not None)

    def _clear_mask_thread(self) -> None:
        self._mask_thread = None
        self._mask_worker = None
        self._set_loading_state(self.statusBar().currentMessage(), busy=self._ct_thread is not None)

    def _set_loading_state(self, message: str, busy: bool) -> None:
        self.load_ct_button.setEnabled(not busy and self._ct_thread is None)
        self.load_dicom_button.setEnabled(not busy and self._ct_thread is None)
        self.add_masks_button.setEnabled(not busy and self.ct_volume is not None and self._mask_thread is None)
        self.add_mask_folder_button.setEnabled(not busy and self.ct_volume is not None and self._mask_thread is None)
        self.reset_view_button.setEnabled(not busy and self.ct_volume is not None)
        self.clear_masks_button.setEnabled(not busy and bool(self.mask_layers))
        self.statusBar().showMessage(message)

    def reset_view_state(self) -> None:
        if self.ct_volume is None:
            return

        self.current_indices = [dimension // 2 for dimension in self.ct_volume.shape]
        self.ct_slice_cache.clear()
        self.mask_slice_cache.clear()
        self.apply_window_preset("Auto")
        self.render_all_views()
        self.statusBar().showMessage("Reset slices and windowing.", 4000)

    def _configure_window_controls(self) -> None:
        if self.ct_volume is None:
            return

        summary = self.ct_volume.summary
        data_min = int(np.floor(summary.minimum))
        data_max = int(np.ceil(summary.maximum))
        slider_center_min = min(data_min, -1500)
        slider_center_max = max(data_max, 3000)
        slider_width_max = max(slider_center_max - slider_center_min, 4000)

        self.auto_window = (
            int(round((summary.low_percentile + summary.high_percentile) / 2.0)),
            int(round(max(summary.high_percentile - summary.low_percentile, 1.0))),
        )

        self.window_center_slider.setRange(slider_center_min, slider_center_max)
        self.window_width_slider.setRange(1, slider_width_max)
        self.window_center_slider.setEnabled(True)
        self.window_width_slider.setEnabled(True)
        self.overlay_opacity_slider.setEnabled(True)
        self.crosshair_checkbox.setEnabled(True)
        self.window_preset_combo.setEnabled(True)
        self.apply_window_preset("Auto")

    def _configure_slice_views(self) -> None:
        if self.ct_volume is None:
            return

        for orientation, view in self.views.items():
            axis = SLICE_AXES[orientation]
            view.set_slice_bounds(self.ct_volume.shape[axis] - 1)
            view.set_slice_index(self.current_indices[axis])

    def update_volume_info(self) -> None:
        if self.ct_volume is None:
            self.volume_info_label.setText("No CT loaded")
            return

        shape_text = " x ".join(str(value) for value in self.ct_volume.shape)
        spacing_text = " x ".join(f"{value:.2f} mm" for value in self.ct_zooms)
        mask_count = len(self.mask_layers)
        filename = Path(self.ct_volume.path).name
        metrics = QFontMetrics(self.volume_info_label.font())
        elided_name = metrics.elidedText(filename, Qt.TextElideMode.ElideMiddle, 240)

        self.volume_info_label.setText(
            "\n".join(
                [
                    f"CT: {elided_name}",
                    f"Shape: {shape_text}",
                    f"Spacing: {spacing_text}",
                    f"{self.ct_intensity_summary} | Masks: {mask_count} | Orientation: canonical RAS+",
                ]
            )
        )

    def refresh_mask_list(self) -> None:
        self.mask_list.blockSignals(True)
        self.mask_list.clear()

        for layer in self.mask_layers:
            item = QListWidgetItem(layer.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
            item.setCheckState(Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
            item.setForeground(QColor(*layer.color))
            tooltip = layer.path
            if layer.voxel_count is not None:
                tooltip = f"{tooltip}\nVoxels: {layer.voxel_count:,}"
            item.setToolTip(tooltip)
            self.mask_list.addItem(item)

        self.mask_list.blockSignals(False)

    def on_mask_item_changed(self, item: QListWidgetItem) -> None:
        row = self.mask_list.row(item)
        if row < 0 or row >= len(self.mask_layers):
            return

        self.mask_layers[row].visible = item.checkState() == Qt.CheckState.Checked
        self.mask_slice_cache.clear()
        self.render_all_views()
        self.refresh_mask_visualization()

    def apply_window_preset(self, preset_name: str) -> None:
        if self.ct_volume is None or self._window_control_lock:
            return
        if preset_name == "Custom":
            return

        if preset_name == "Auto":
            center, width = self.auto_window
        else:
            preset = WINDOW_PRESETS.get(preset_name)
            if preset is None:
                return
            center, width = preset

        self._window_control_lock = True
        if self.window_preset_combo.currentText() != preset_name:
            self.window_preset_combo.setCurrentText(preset_name)
        self.window_center_slider.setValue(clamp(center, self.window_center_slider.minimum(), self.window_center_slider.maximum()))
        self.window_width_slider.setValue(clamp(width, self.window_width_slider.minimum(), self.window_width_slider.maximum()))
        self.window_center_value.setText(str(self.window_center_slider.value()))
        self.window_width_value.setText(str(self.window_width_slider.value()))
        self._window_control_lock = False
        self.render_all_views()

    def on_window_slider_changed(self) -> None:
        if self.ct_volume is None:
            return

        self.window_center_value.setText(str(self.window_center_slider.value()))
        self.window_width_value.setText(str(self.window_width_slider.value()))

        if not self._window_control_lock and self.window_preset_combo.currentText() != "Custom":
            self._window_control_lock = True
            self.window_preset_combo.setCurrentText("Custom")
            self._window_control_lock = False

        self.render_all_views()

    def on_overlay_opacity_changed(self, value: int) -> None:
        self.overlay_opacity_value.setText(f"{value}%")
        self.render_all_views()

    def on_slice_changed(self, orientation: str, index: int) -> None:
        if self.ct_volume is None:
            return

        self.current_indices[SLICE_AXES[orientation]] = index
        self.render_all_views()

    def on_view_clicked(self, orientation: str, image_x: int, image_y: int) -> None:
        if self.ct_volume is None:
            return

        nx, ny, nz = self.ct_volume.shape
        if orientation == "axial":
            self.current_indices[0] = clamp(image_x, 0, nx - 1)
            self.current_indices[1] = clamp(ny - 1 - image_y, 0, ny - 1)
        elif orientation == "coronal":
            self.current_indices[0] = clamp(image_x, 0, nx - 1)
            self.current_indices[2] = clamp(nz - 1 - image_y, 0, nz - 1)
        elif orientation == "sagittal":
            self.current_indices[1] = clamp(ny - 1 - image_x, 0, ny - 1)
            self.current_indices[2] = clamp(nz - 1 - image_y, 0, nz - 1)

        self.render_all_views()

    def render_all_views(self) -> None:
        if self.ct_volume is None:
            return

        center = self.window_center_slider.value()
        width = self.window_width_slider.value()
        opacity = self.overlay_opacity_slider.value() / 100.0

        for orientation, view in self.views.items():
            pixmap, logical_size = self._render_view(orientation, center, width, opacity)
            axis = SLICE_AXES[orientation]
            slice_index = self.current_indices[axis]
            view.set_slice_index(slice_index)
            view.set_image(pixmap, logical_size)
            view.set_footer(f"Slice {slice_index + 1} / {self.ct_volume.shape[axis]}")

        self._update_cursor_info()

    def refresh_mask_visualization(self) -> None:
        if not hasattr(self, "mask_viz"):
            return

        if self.ct_volume is None:
            self.mask_viz.clear_view()
            return

        visible_layers = [layer for layer in self.mask_layers if layer.visible]
        if not visible_layers:
            self.mask_viz.clear_view()
            return

        if self._mask_preview_thread is not None:
            self._mask_preview_needs_refresh = True
            return

        self._mask_preview_generation += 1
        generation = self._mask_preview_generation
        self.mask_viz.set_busy(f"Rendering {len(visible_layers)} visible mask(s)...")

        thread = QThread(self)
        worker = MaskPreviewWorker(generation, self.ct_volume, visible_layers)
        self._mask_preview_thread = thread
        self._mask_preview_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_mask_preview_ready)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_mask_preview_thread)
        thread.start()

    def _on_mask_preview_ready(self, generation: int, figure_json: object) -> None:
        if generation != self._mask_preview_generation or not hasattr(self, "mask_viz"):
            return

        self.mask_viz.set_data_json(figure_json if isinstance(figure_json, dict) else None, len([layer for layer in self.mask_layers if layer.visible]))
        self._mask_preview_needs_refresh = False

    def _clear_mask_preview_thread(self) -> None:
        self._mask_preview_thread = None
        self._mask_preview_worker = None
        if self._mask_preview_needs_refresh and self.ct_volume is not None and any(layer.visible for layer in self.mask_layers):
            self._mask_preview_needs_refresh = False
            QTimer.singleShot(0, self.refresh_mask_visualization)

    def _render_view(self, orientation: str, center: int, width: int, opacity: float) -> tuple[QPixmap, tuple[int, int]]:
        ct_slice = self._get_ct_slice(orientation)
        rgba = ct_rendering.grayscale_rgba(ct_slice, center, width)

        for layer_index, layer in enumerate(self.mask_layers):
            if not layer.visible:
                continue
            mask_slice = self._get_mask_slice(layer_index, layer, orientation)
            ct_rendering.blend_mask(rgba, mask_slice, layer.color, opacity)

        image = ct_rendering.qimage_from_rgba(rgba)
        if self.crosshair_checkbox.isChecked():
            cross_x, cross_y = ct_rendering.crosshair_position(self.ct_volume.shape, orientation, self.current_indices)
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            pen = QPen(QColor("#44d7ff"))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawLine(cross_x, 0, cross_x, image.height() - 1)
            painter.drawLine(0, cross_y, image.width() - 1, cross_y)
            left_label, right_label, top_label, bottom_label = ct_rendering.orientation_labels(orientation)
            painter.setPen(QPen(QColor("#f7fbff")))
            painter.drawText(12, image.height() // 2, left_label)
            painter.drawText(image.width() - 22, image.height() // 2, right_label)
            painter.drawText(image.width() // 2 - 6, 20, top_label)
            painter.drawText(image.width() // 2 - 6, image.height() - 12, bottom_label)
            painter.end()

        pixmap = QPixmap.fromImage(image)
        width_spacing, height_spacing = ct_rendering.display_spacing(self.ct_zooms, orientation)
        target_width, target_height = ct_rendering.physical_display_size(image.width(), image.height(), width_spacing, height_spacing)
        if target_width != image.width() or target_height != image.height():
            pixmap = pixmap.scaled(
                target_width,
                target_height,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )

        return pixmap, (image.width(), image.height())

    def _get_ct_slice(self, orientation: str) -> np.ndarray:
        axis = SLICE_AXES[orientation]
        index = self.current_indices[axis]
        cache_key = (orientation, index)
        cached = self.ct_slice_cache.get(cache_key)
        if cached is not None:
            return cached

        ct_slice = ct_rendering.extract_slice(self.ct_volume.image, orientation, self.current_indices)
        if not np.isfinite(ct_slice).all():
            ct_slice = np.nan_to_num(ct_slice, copy=False)

        self.ct_slice_cache = {cache_key: ct_slice}
        return ct_slice

    def _get_mask_slice(self, layer_index: int, layer: MaskLayer, orientation: str) -> np.ndarray:
        axis = SLICE_AXES[orientation]
        index = self.current_indices[axis]
        cache_key = (layer_index, orientation, index)
        cached = self.mask_slice_cache.get(cache_key)
        if cached is not None:
            return cached

        mask_slice = ct_rendering.extract_slice(layer.image, orientation, self.current_indices) != 0
        if len(self.mask_slice_cache) > 18:
            self.mask_slice_cache.clear()
        self.mask_slice_cache[cache_key] = mask_slice
        return mask_slice

    def _update_cursor_info(self) -> None:
        if self.ct_volume is None:
            self.cursor_info_label.setText("Voxel: -, -, -    HU: -")
            return

        x_idx, y_idx, z_idx = self.current_indices
        hu_value = float(np.asarray(self.ct_volume.image.dataobj[x_idx, y_idx, z_idx], dtype=np.float32))
        active_masks = []
        for layer_index, layer in enumerate(self.mask_layers):
            if not layer.visible:
                continue
            voxel_value = np.asarray(layer.image.dataobj[x_idx, y_idx, z_idx])
            if np.any(voxel_value != 0):
                active_masks.append(layer.name)
        mask_text = ", ".join(active_masks) if active_masks else "None"
        self.cursor_info_label.setText(
            f"Voxel: x={x_idx}  y={y_idx}  z={z_idx}    HU: {hu_value:.1f}    Masks here: {mask_text}"
        )

    def _require_ct_loaded(self) -> bool:
        if self.ct_volume is not None:
            return True
        QMessageBox.information(self, "Load CT First", "Load a CT volume before adding segmentation masks.")
        return False

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast CT and segmentation viewer")
    parser.add_argument(
        "source",
        nargs="?",
        help="Optional CT input path. Accepts a NIfTI file or a DICOM folder.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args(sys.argv[1:])
    app = QApplication([sys.argv[0]])
    app.setApplicationName("CT and Segmentation Viewer")
    viewer = CTMaskViewer()
    viewer.show()

    if args.source:
        QTimer.singleShot(0, lambda: viewer.load_ct_async(args.source))

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
