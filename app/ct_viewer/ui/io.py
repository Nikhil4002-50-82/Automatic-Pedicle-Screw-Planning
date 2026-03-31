from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from .models import CTVolume, MASK_COLORS, MaskLayer, MaskLoadResult, VolumeSummary

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - optional dependency
    sitk = None


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


def load_ct_volume(path: str) -> CTVolume:
    image = load_spatial_image(path)
    summary = summarize_volume(image)
    return CTVolume(
        path=path,
        image=image,
        shape=tuple(int(value) for value in image.shape[:3]),
        zooms=tuple(float(value) for value in image.header.get_zooms()[:3]),
        summary=summary,
    )


def load_mask_layers(paths: list[str], ct_image: nib.spatialimages.SpatialImage, start_index: int) -> MaskLoadResult:
    layers: list[MaskLayer] = []
    warnings: list[str] = []

    for index, path in enumerate(paths, start=start_index):
        try:
            mask_image = load_nifti_image(path)
            if mask_image.shape != ct_image.shape or not np.allclose(
                mask_image.affine,
                ct_image.affine,
                atol=1e-3,
            ):
                mask_image = resample_from_to(mask_image, ct_image, order=0)
                warnings.append(f"{Path(path).name}: resampled to match CT geometry.")

            layers.append(
                MaskLayer(
                    name=Path(path).name,
                    path=path,
                    image=mask_image,
                    color=MASK_COLORS[index % len(MASK_COLORS)],
                    visible=True,
                    voxel_count=None,
                )
            )
        except Exception as exc:  # pragma: no cover - UI error path
            warnings.append(f"{Path(path).name}: {exc}")

    return MaskLoadResult(layers=layers, warnings=warnings)
