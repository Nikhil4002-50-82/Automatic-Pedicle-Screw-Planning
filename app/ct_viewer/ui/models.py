from __future__ import annotations

from dataclasses import dataclass

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


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))
