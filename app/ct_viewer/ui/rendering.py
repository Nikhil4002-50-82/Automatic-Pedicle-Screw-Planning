from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QImage
from scipy import ndimage

from .models import clamp


def extract_slice(image, orientation: str, indices: list[int]) -> np.ndarray:
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
