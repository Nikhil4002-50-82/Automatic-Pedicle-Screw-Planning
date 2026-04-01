from __future__ import annotations

import numpy as np
from scipy import ndimage

from .models import MaskLayer, clamp


def focus_indices_from_masks(shape: tuple[int, int, int], layers: list[MaskLayer]) -> list[int] | None:
    if not layers:
        return None

    mask_union = np.zeros(shape, dtype=bool)
    for layer in layers:
        mask_union |= np.asarray(layer.image.dataobj) != 0

    if not np.any(mask_union):
        return None

    center = ndimage.center_of_mass(mask_union.astype(np.uint8, copy=False))
    target = [
        clamp(int(round(center[0])), 0, shape[0] - 1),
        clamp(int(round(center[1])), 0, shape[1] - 1),
        clamp(int(round(center[2])), 0, shape[2] - 1),
    ]

    if not mask_union[tuple(target)]:
        coords = np.argwhere(mask_union)
        if coords.size == 0:
            return None
        deltas = coords.astype(np.float32, copy=False) - np.asarray(center, dtype=np.float32)
        target = coords[int(np.argmin(np.sum(deltas * deltas, axis=1)))].tolist()

    return [int(value) for value in target]
