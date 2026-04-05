import gc
import os
import time
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, label as cc_label, map_coordinates
from sklearn.decomposition import PCA

# --- Global Constraints ---
voxelThreshold = 5000
stepMM = 0.5
minLengthMM = 20
maxDepthRatio = 0.8

# Standard Lumbar Hardware (mm)
globalDiameters = [8.5, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0]

# --- Clinical Diameter Standards ---
CLINICAL_LIMITS = {
    "L1": (4.5, 6.0),
    "L2": (4.5, 6.0),
    "L3": (5.0, 6.5),
    "L4": (6.0, 7.5),
    "L5": (6.5, 8.5),
}

# --- Clinical Length Limits (mm) ---
LENGTH_LIMITS = {
    "L1": 45,
    "L2": 50,
    "L3": 50,
    "L4": 55,
    "L5": 60,
}

# --- FINAL BALANCED WEIGHTS ---
wDiam = 15.0
wConv = 20.0
wDT = 5.0
wLen = 1.0
wTilt = 10.0
wMidline = 50.0

labelMap = {5: "L1", 4: "L2", 3: "L3", 2: "L4", 1: "L5"}


def _log(message, log_queue=None):
    if log_queue is not None:
        try:
            log_queue.put(str(message))
            return
        except Exception:
            pass
    print(message, flush=True)


def loadNifti(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii.header.get_zooms(), nii.affine


def getValidLabels(seg):
    valid = []
    uniqueLabels = np.unique(seg)
    uniqueLabels = uniqueLabels[uniqueLabels != 0]
    for labelVal in uniqueLabels:
        mask = (seg == labelVal)
        labeled, _ = cc_label(mask)
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        if len(sizes) == 0:
            continue
        largest = np.argmax(sizes)
        component = (labeled == largest)
        if np.sum(component) > voxelThreshold:
            valid.append((int(labelVal), component))
    return valid


def computeStableFrame(mask, affine):
    coords = np.argwhere(mask)
    coordsWorld = nib.affines.apply_affine(affine, coords)

    centroid = coordsWorld.mean(axis=0)

    pca = PCA(n_components=3)
    pca.fit(coordsWorld - centroid)
    axes = pca.components_

    worldZ = np.array([0, 0, 1])

    siAxis = axes[np.argmax(np.abs(axes @ worldZ))]
    if np.dot(siAxis, worldZ) < 0:
        siAxis = -siAxis

    tempAxis = axes[np.argmin(np.abs(axes @ worldZ))]

    lrAxis = tempAxis - np.dot(tempAxis, siAxis) * siAxis
    lrAxis /= np.linalg.norm(lrAxis)

    apAxis = np.cross(siAxis, lrAxis)
    apAxis /= np.linalg.norm(apAxis)

    relAP = (coordsWorld - centroid) @ apAxis
    totalDepth = np.percentile(relAP, 95) - np.percentile(relAP, 5)

    return centroid, np.vstack([siAxis, lrAxis, apAxis]), totalDepth


def computeDistance(mask, spacing):
    return distance_transform_edt(mask, sampling=spacing)


def pedicleCenters(mask, dist, centroid, axes, affine):
    coords = np.argwhere(mask)
    coordsWorld = nib.affines.apply_affine(affine, coords)

    siAxis, lrAxis, apAxis = axes
    rel = coordsWorld - centroid

    midMask = np.abs(rel @ siAxis) < np.percentile(np.abs(rel @ siAxis), 35)
    posteriorMask = (rel @ apAxis) < np.percentile(rel @ apAxis, 40)

    lCoords = coords[midMask & posteriorMask & ((rel @ lrAxis) < 0)]
    rCoords = coords[midMask & posteriorMask & ((rel @ lrAxis) > 0)]

    if len(lCoords) < 20 or len(rCoords) < 20:
        return None, None

    lCenter = nib.affines.apply_affine(
        affine,
        lCoords[np.argmax(dist[lCoords[:, 0], lCoords[:, 1], lCoords[:, 2]])]
    )

    rCenter = nib.affines.apply_affine(
        affine,
        rCoords[np.argmax(dist[rCoords[:, 0], rCoords[:, 1], rCoords[:, 2]])]
    )

    return lCenter, rCenter


def findEntry(center, axes, maskFloat, affine, side):
    siAxis, lrAxis, apAxis = axes

    lateralShift = -1.5 if side == "Left" else 1.5
    startP = center + (lrAxis * lateralShift)

    direction = -apAxis

    invAff = np.linalg.inv(affine)

    p = startP.copy()

    for _ in range(100):
        vox = nib.affines.apply_affine(invAff, p)

        if any(v < 0 or v >= s - 1 for v, s in zip(vox, maskFloat.shape)):
            break

        if map_coordinates(maskFloat, [[vox[0]], [vox[1]], [vox[2]]], order=1)[0] < 0.5:
            break

        p += direction * stepMM

    return p + apAxis * 1.5


def evaluate(entry, direction, side, maskFloat, dist, affine, axes, centroid, maxAllowedLen, v_name):
    d = direction / np.linalg.norm(direction)

    siAxis, lrAxis, apAxis = axes
    invAff = np.linalg.inv(affine)

    if np.dot(d, siAxis) > 0:
        return None

    t = 0
    minDT = 999
    midlineViolation = 0

    lr_comp = np.dot(d, lrAxis)
    conv = lr_comp if side == "Left" else -lr_comp

    t_exit = None

    while t < maxAllowedLen:
        p = entry + d * t
        vox = nib.affines.apply_affine(invAff, p)

        if any(v < 0 or v >= s - 1 for v, s in zip(vox, maskFloat.shape)):
            t_exit = t
            break

        inside = map_coordinates(
            maskFloat,
            [[vox[0]], [vox[1]], [vox[2]]],
            order=1
        )[0]

        if inside < 0.5:
            t_exit = t
            break

        relP = p - centroid
        lrPos = np.dot(relP, lrAxis)

        if (side == "Left" and lrPos > 1.5) or (side == "Right" and lrPos < -1.5):
            midlineViolation += 1

        dtVal = map_coordinates(dist, [[vox[0]], [vox[1]], [vox[2]]], order=1)[0]

        if 5.0 < t < 30.0:
            minDT = min(minDT, dtVal)

        t += stepMM

    if t_exit is None:
        t_exit = t

    safetyMargin = 3.0
    t_safe = t_exit - safetyMargin

    if t_safe < minLengthMM:
        return None

    max_safe = (minDT * 2) - 0.5
    min_limit, max_limit = CLINICAL_LIMITS.get(v_name, (4.0, 8.5))

    diam = 0.0
    for d_val in globalDiameters:
        if d_val <= max_safe and d_val <= max_limit:
            diam = d_val
            break

    tilt = abs(np.dot(d, siAxis))

    score = (
        (wDiam * diam)
        + (wConv * conv * 15)
        + (wDT * minDT)
        - (wTilt * tilt)
        - (wMidline * midlineViolation)
    )

    safe_tip = entry + d * t_safe

    return score, t_safe, minDT, safe_tip, diam


def optimize(center, axes, side, maskFloat, dist, affine, centroid, totalDepth, v_name):
    if center is None:
        return None

    entry = findEntry(center, axes, maskFloat, affine, side)

    geomLimit = totalDepth * maxDepthRatio
    clinicalLimit = LENGTH_LIMITS.get(v_name, 50)
    maxAllowedLen = min(geomLimit, clinicalLimit)

    best = None

    angles = np.linspace(5, 40, 25) if side == "Left" else np.linspace(-40, -5, 25)

    for lrAng in angles:
        for siAng in np.linspace(-10, 0, 11):
            direction = (
                axes[2]
                + np.tan(np.deg2rad(lrAng)) * axes[1]
                + np.tan(np.deg2rad(siAng)) * axes[0]
            )

            res = evaluate(
                entry,
                direction,
                side,
                maskFloat,
                dist,
                affine,
                axes,
                centroid,
                maxAllowedLen,
                v_name
            )

            if res and (best is None or res[0] > best[0]):
                best = (
                    res[0],
                    entry,
                    res[3],
                    res[1],
                    res[2],
                    lrAng,
                    siAng,
                    res[4]
                )

    return best


def process_single_vertebra(labelVal, mask, name, affine, spacing, log_queue=None):
    t_start = time.perf_counter()
    _log(f"\n[TIMING] Starting {name}...", log_queue)

    t = time.perf_counter()
    centroid, axes, totalDepth = computeStableFrame(mask, affine)
    _log(f"[TIMING] [{name}] computeStableFrame: {time.perf_counter() - t:.3f}s", log_queue)

    t = time.perf_counter()
    dist = computeDistance(mask, spacing)
    _log(f"[TIMING] [{name}] computeDistance: {time.perf_counter() - t:.3f}s", log_queue)

    maskFloat = mask.astype(np.float32)

    t = time.perf_counter()
    lCenter, rCenter = pedicleCenters(mask, dist, centroid, axes, affine)
    _log(f"[TIMING] [{name}] pedicleCenters: {time.perf_counter() - t:.3f}s", log_queue)

    del mask
    gc.collect()

    t = time.perf_counter()
    local_results = []
    for side, center in [("Left", lCenter), ("Right", rCenter)]:
        res = optimize(center, axes, side, maskFloat, dist, affine, centroid, totalDepth, name)
        if res:
            score, entry, tip, length, minDT, lrAng, siAng, diam = res
            local_results.append({
                "vertebra": name,
                "side": side,
                "entry": entry,
                "tip": tip,
                "diameter": diam,
                "length": length,
                "axial_angle": lrAng,
                "sagittal_angle": siAng,
            })
            _log(
                f"{name} {side}: Diam {diam}mm, Axial {round(lrAng,1)}°, Sagittal {round(siAng,1)}°",
                log_queue
            )

    _log(f"[TIMING] [{name}] optimize (both sides): {time.perf_counter() - t:.3f}s", log_queue)

    del dist, maskFloat, centroid, axes, lCenter, rCenter
    gc.collect()

    _log(f"[TIMING] [{name}] TOTAL: {time.perf_counter() - t_start:.3f}s", log_queue)
    return local_results


def process_single_vertebra_worker(args):
    if len(args) == 5:
        labelVal, mask, name, affine, spacing = args
        log_queue = None
    else:
        labelVal, mask, name, affine, spacing, log_queue = args
    return process_single_vertebra(labelVal, mask, name, affine, spacing, log_queue=log_queue)


def run_planner(segPath, log_queue=None):
    resultsList = []
    _total_start = time.perf_counter()
    _log("[TIMING] Starting lumbar screw planning pipeline...", log_queue)
    _log(f"[INFO] System cores: {os.cpu_count()}", log_queue)

    t = time.perf_counter()
    seg, spacing, affine = loadNifti(segPath)
    _log(f"[TIMING] loadNifti: {time.perf_counter() - t:.3f}s (shape: {seg.shape})", log_queue)

    t = time.perf_counter()
    validSegments = getValidLabels(seg)
    _log(f"[TIMING] getValidLabels: {time.perf_counter() - t:.3f}s ({len(validSegments)} segments)", log_queue)

    del seg
    gc.collect()

    worker_count = min(2, os.cpu_count() or 1)
    worker_count = max(1, min(worker_count, len(validSegments) or 1))
    _log(f"\n[INFO] Processing vertebrae in parallel ({worker_count} workers)...", log_queue)

    vertebra_args = []
    for labelVal, mask in sorted(validSegments, reverse=True):
        name = labelMap.get(labelVal, str(labelVal))
        if log_queue is None:
            vertebra_args.append((labelVal, mask, name, affine, spacing))
        else:
            vertebra_args.append((labelVal, mask, name, affine, spacing, log_queue))

    t = time.perf_counter()
    if worker_count == 1 or len(vertebra_args) <= 1:
        for args in vertebra_args:
            resultsList.extend(process_single_vertebra_worker(args))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for res in executor.map(process_single_vertebra_worker, vertebra_args):
                resultsList.extend(res)

    _log(f"\n[TIMING] All vertebrae processing: {time.perf_counter() - t:.3f}s", log_queue)
    _log(f"[TIMING] TOTAL pipeline execution: {time.perf_counter() - _total_start:.3f}s", log_queue)

    return resultsList


# if __name__ == "__main__":
#     results = run_planner(r"C:\Users\heman\Downloads\sub-verse823_dir-iso_ct_segmented_vertebrae_L5_vertebrae_L1.nii.gz")
#     print("\n[RESULTS]")
#     for r in results:
#         print(r)
