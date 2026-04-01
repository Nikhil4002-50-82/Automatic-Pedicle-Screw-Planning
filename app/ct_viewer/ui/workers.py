from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from . import io as ct_io
from . import mask_viz as ct_mask_viz
from .models import CTVolume, MaskLayer, MaskLoadResult


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
