from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from . import io as ct_io
from . import rendering as ct_rendering
from . import mask_viz as ct_mask_viz
from .models import CTVolume, MaskLayer, MaskLoadResult, SLICE_AXES


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

    def __init__(self, paths: list[str], ct_image, start_index: int) -> None:
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
    finished = pyqtSignal(int, object, object, int)
    failed = pyqtSignal(str, str)

    def __init__(
        self,
        generation: int,
        ct_volume: CTVolume | None,
        layers: list[MaskLayer],
        signature: tuple[str, ...],
        visible_mask_count: int,
    ) -> None:
        super().__init__()
        self.generation = generation
        self.ct_volume = ct_volume
        self.layers = layers
        self.signature = signature
        self.visible_mask_count = visible_mask_count

    def run(self) -> None:
        try:
            figure = ct_mask_viz.build_mask_preview_figure(self.ct_volume, self.layers)
            payload = figure.to_plotly_json()
        except Exception as exc:  # pragma: no cover - UI error path
            self.failed.emit("Could not build 3D mask preview", str(exc))
            return

        self.finished.emit(self.generation, payload, self.signature, self.visible_mask_count)


class StudyRenderWorker(QObject):
    finished = pyqtSignal(int, object)
    failed = pyqtSignal(str, str)

    def __init__(
        self,
        generation: int,
        ct_volume: CTVolume,
        layers: list[MaskLayer],
        current_indices: list[int],
        center: int,
        width: int,
        opacity: float,
    ) -> None:
        super().__init__()
        self.generation = generation
        self.ct_volume = ct_volume
        self.layers = layers
        self.current_indices = list(current_indices)
        self.center = center
        self.width = width
        self.opacity = opacity

    def run(self) -> None:
        try:
            payload: dict[str, dict[str, object]] = {}
            for orientation in ("axial", "coronal", "sagittal"):
                ct_slice = ct_rendering.extract_slice(self.ct_volume.image, orientation, self.current_indices)
                if not np.isfinite(ct_slice).all():
                    ct_slice = np.nan_to_num(ct_slice, copy=False)
                rgba = ct_rendering.grayscale_rgba(ct_slice, self.center, self.width)

                for layer_index, layer in enumerate(self.layers):
                    if not layer.visible:
                        continue
                    mask_slice = ct_rendering.extract_slice(layer.image, orientation, self.current_indices) != 0
                    ct_rendering.blend_mask(rgba, mask_slice, layer.color, self.opacity)

                width_spacing, height_spacing = ct_rendering.display_spacing(self.ct_volume.zooms, orientation)
                logical_size = (rgba.shape[1], rgba.shape[0])
                target_size = ct_rendering.physical_display_size(
                    rgba.shape[1],
                    rgba.shape[0],
                    width_spacing,
                    height_spacing,
                )
                axis = SLICE_AXES[orientation]
                slice_index = self.current_indices[axis]
                payload[orientation] = {
                    "rgba": rgba,
                    "logical_size": logical_size,
                    "target_size": target_size,
                    "footer": f"Slice {slice_index + 1} / {self.ct_volume.shape[axis]}",
                    "slice_index": slice_index,
                    "indices": tuple(self.current_indices),
                }
        except Exception as exc:  # pragma: no cover - UI error path
            self.failed.emit("Could not render CT views", str(exc))
            return

        self.finished.emit(self.generation, payload)
