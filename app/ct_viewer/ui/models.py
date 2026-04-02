from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib

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


@dataclass
class ViewerStudy:
    key: str
    label: str
    ct_volume: CTVolume | None = None
    ct_zooms: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mask_layers: list[MaskLayer] = field(default_factory=list)
    current_indices: list[int] = field(default_factory=lambda: [0, 0, 0])
    auto_window: tuple[int, int] = (300, 1500)
    window_center: int = 300
    window_width: int = 1500
    window_preset: str = "Auto"
    ct_intensity_summary: str = "Intensity range: unavailable"
    ct_slice_cache: dict[tuple[object, ...], object] = field(default_factory=dict)
    mask_slice_cache: dict[tuple[object, ...], object] = field(default_factory=dict)

    @property
    def display_label(self) -> str:
        if self.label:
            return self.label
        if self.ct_volume is not None:
            return Path(self.ct_volume.path).name
        return "Mask-only study"

    @property
    def mask_count(self) -> int:
        return len(self.mask_layers)

    @property
    def has_ct(self) -> bool:
        return self.ct_volume is not None


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))
