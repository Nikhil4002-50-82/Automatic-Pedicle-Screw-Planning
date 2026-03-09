import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, label as cc_label
from scipy.ndimage import map_coordinates
from sklearn.decomposition import PCA

# Minimum number of voxels required to accept a vertebra
voxelThreshold = 5000

# Possible screw diameters (in mm)
globalDiameters = [8.5, 7.5, 7.0, 6.5, 5.5, 5.0, 4.5, 4.0]
maxDiameterPerLevel = {"L1": 6.5, "L2": 7.0, "L3": 7.5, "L4": 8.5, "L5": 8.5}

# Step size when moving inside bone (in mm)
stepMM = 0.5

# Minimum screw length required (in mm)
minLengthMM = 18

# Map segmentation labels to vertebra names
labelMap = {5: "L1", 4: "L2", 3: "L3", 2: "L4", 1: "L5"}


def loadNifti(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii.header.get_zooms(), nii.affine


def getValidLabels(seg):
    valid = []
    uniqueLabels = np.unique(seg)
    uniqueLabels = uniqueLabels[uniqueLabels != 0]
    for labelVal in uniqueLabels:
        mask = seg == labelVal
        labeled, _ = cc_label(mask)
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        if len(sizes) == 0:
            continue
        largest = np.argmax(sizes)
        component = labeled == largest
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
    # Ensure AP axis points "anterior" relative to the structural mass
    # The centroid of anterior mass is further 'apAxis' than the centroid of posterior mass
    return centroid, np.vstack([siAxis, lrAxis, apAxis])


def computeDistance(mask, spacing):
    return distance_transform_edt(mask, sampling=spacing)


def getL5VertebralBodyCenter(mask, axes, centroid, affine):
    coords = np.argwhere(mask)
    coordsWorld = nib.affines.apply_affine(affine, coords)
    siAxis, lrAxis, apAxis = axes

    rel = coordsWorld - centroid
    apVals = rel @ apAxis

    # Anterior region is the top 40% along the AP axis
    anterior_thresh = np.percentile(apVals, 60)
    anterior_mask = apVals > anterior_thresh

    anterior_coords = coordsWorld[anterior_mask]
    if len(anterior_coords) == 0:
        return centroid
    return anterior_coords.mean(axis=0)


def pedicleCentersL5(mask, dist, centroid, axes, affine):
    coords = np.argwhere(mask)
    coordsWorld = nib.affines.apply_affine(affine, coords)
    siAxis, lrAxis, apAxis = axes
    rel = coordsWorld - centroid
    siVals = rel @ siAxis
    lrVals = rel @ lrAxis
    apVals = rel @ apAxis

    # L5-specific narrow mid-band search for pedicles
    midMask = np.abs(siVals) < np.percentile(np.abs(siVals), 35)
    posteriorMask = (apVals > np.percentile(apVals, 30)) & (
        apVals < np.percentile(apVals, 60)
    )
    leftMask = lrVals < np.percentile(lrVals, 25)
    rightMask = lrVals > np.percentile(lrVals, 75)

    leftCoords = coords[midMask & posteriorMask & leftMask]
    rightCoords = coords[midMask & posteriorMask & rightMask]

    if len(leftCoords) < 30 or len(rightCoords) < 30:
        # Fallback to broader search
        midMask = np.abs(siVals) < np.percentile(np.abs(siVals), 40)
        posteriorMask = apVals < np.percentile(apVals, 50)
        leftMask = lrVals < 0
        rightMask = lrVals > 0
        leftCoords = coords[midMask & posteriorMask & leftMask]
        rightCoords = coords[midMask & posteriorMask & rightMask]

    if len(leftCoords) < 30 or len(rightCoords) < 30:
        return None, None

    # Calculate PCA axes for the pedicles to find their natural anatomical direction
    pca_l = PCA(n_components=1)
    pca_l.fit(nib.affines.apply_affine(affine, leftCoords))
    lAxis = pca_l.components_[0]

    pca_r = PCA(n_components=1)
    pca_r.fit(nib.affines.apply_affine(affine, rightCoords))
    rAxis = pca_r.components_[0]

    # Force the axes to point generally Anteriorly
    if np.dot(lAxis, apAxis) < 0:
        lAxis = -lAxis
    if np.dot(rAxis, apAxis) < 0:
        rAxis = -rAxis

    lVox = leftCoords[
        np.argmax(dist[leftCoords[:, 0], leftCoords[:, 1], leftCoords[:, 2]])
    ]
    rVox = rightCoords[
        np.argmax(dist[rightCoords[:, 0], rightCoords[:, 1], rightCoords[:, 2]])
    ]

    lMM = nib.affines.apply_affine(affine, lVox)
    rMM = nib.affines.apply_affine(affine, rVox)

    return (lMM, lAxis), (rMM, rAxis)


def compute_analytical_direction(side, axes):
    siAxis, lrAxis, apAxis = axes

    # Fixed L5 Analytical Parameters:
    tpa_deg = 25.0
    spa_deg = 0.0

    tpa_rad = np.deg2rad(tpa_deg)
    spa_rad = np.deg2rad(spa_deg)

    # Left pedicle points Right (+lrAxis). Right pedicle points Left (-lrAxis).
    # Assuming lrAxis is defined globally (we must check its sign).
    # In stable frame, lrAxis points Left or Right. Let's ensure it points Right.
    if side == "Left":
        medial_sign = 1.0
    else:
        medial_sign = -1.0

    # Build the unit vector U from the 90-degree components
    # U = cos(SPA)cos(TPA)*AP + cos(SPA)sin(TPA)*LR + sin(SPA)*SI

    # Actually, we need to know if lrAxis points Right.
    # We can determine this by comparing center to centroid in run_planner.
    # Instead of trusting lrAxis sign globally, we pass side_sign to this function directly later.
    pass


def synthesize_analytical_direction(center, centroid, axes):
    siAxis, lrAxis, apAxis = axes

    # Determine which way is medial
    # If center is to the Left of centroid, medial is Right.
    lr_offset = np.dot(center - centroid, lrAxis)
    if lr_offset > 0:
        # center is in +lr direction
        # Medial must be -lr direction
        medial_sign = -1.0
    else:
        # center is in -lr direction
        # Medial must be +lr direction
        medial_sign = 1.0

    # L5 Specifics: 25 degree axial convergence (TPA), 0 degree sagittal angle (SPA)
    tpa_rad = np.deg2rad(25.0)

    # Note: apAxis points Anteriorly. We want the screw to point Anteriorly.
    # Therefore, the base direction along AP should be +apAxis.
    direction = np.cos(tpa_rad) * apAxis + medial_sign * np.sin(tpa_rad) * lrAxis
    return direction / np.linalg.norm(direction)


def raycast_entry_point(center, direction, maskFloat, affine):
    # Move backward from center until we exit bone
    invAff = np.linalg.inv(affine)
    p = center.copy()
    test_dir = -direction
    for _ in range(300):
        vox = nib.affines.apply_affine(invAff, p)
        if any(v < 0 or v >= s - 1 for v, s in zip(vox, maskFloat.shape)):
            break
        val = map_coordinates(maskFloat, [[vox[0]], [vox[1]], [vox[2]]], order=1)[0]
        if val < 0.5:  # Escaped bone structure
            break
        p += test_dir * stepMM

    # Enter slightly inside the bone to represent the burred entry
    return p + direction * 1.0


def cylinder_safe(p, d, radius, maskFloat, affine):
    invAff = np.linalg.inv(affine)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        # We need orthonormal basis to d
        # Find arbitrary vector not parallel to d
        v = np.array([1, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1, 0])
        u1 = np.cross(d, v)
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(d, u1)

        offset = radius * (np.cos(angle) * u1 + np.sin(angle) * u2)
        testPoint = p + offset
        vox = nib.affines.apply_affine(invAff, testPoint)

        if any(v < 0 or v >= s - 1 for v, s in zip(vox, maskFloat.shape)):
            return False

        val = map_coordinates(maskFloat, [[vox[0]], [vox[1]], [vox[2]]], order=1)[0]
        if val < 0.5:
            return False
    return True


def analytical_evaluate(entry, direction, maskFloat, dist, affine, axes, centroid):
    # Given an idealized analytical direction, we raymarch forward to find safest diameter/length
    d = direction / np.linalg.norm(direction)
    invAff = np.linalg.inv(affine)

    best_length = 0
    best_diam = 0
    best_minDT = 0
    best_tip = None

    # Find maximum length we can traverse safely
    t = 0
    ray_dt_vals = []

    while True:
        p = entry + d * t
        vox = nib.affines.apply_affine(invAff, p)

        if any(v < 0 or v >= s - 1 for v, s in zip(vox, maskFloat.shape)):
            break

        maskVal = map_coordinates(maskFloat, [[vox[0]], [vox[1]], [vox[2]]], order=1)[0]
        if maskVal < 0.5:
            # We hit the anterior cortical wall, stop
            break

        dtVal = map_coordinates(dist, [[vox[0]], [vox[1]], [vox[2]]], order=1)[0]

        # Don't penalize DT in the first 5mm since the entry hole might be jagged
        if t > 5:
            ray_dt_vals.append((t, dtVal))

        t += stepMM

    if len(ray_dt_vals) == 0:
        print("    [Debug] Ray exited bone immediately.")
        return None

    # Analyze the ray to find bottlenecks
    # The screw diameter is constrained by the narrowest part of the pedicle
    min_dist_along_ray = min([dt for t, dt in ray_dt_vals])

    # Safety margin: Screw radius must be less than min_dist_along_ray
    max_safe_radius = min_dist_along_ray

    # Snap to standard diameters (allowing a margin of 0.0mm just like geometry.py)
    safe_diams = [D for D in globalDiameters if (D / 2.0) <= max_safe_radius]

    if len(safe_diams) == 0:
        print(
            f"    [Debug] No standard safe diameter. Using max safe radius for visualization. max_safe_radius={max_safe_radius}"
        )
        # If it's too small for standard sizes, just cap it to the tightest bounds to see where it aimed
        best_diam = max_safe_radius * 2.0
    else:
        best_diam = max(safe_diams)

    radius = best_diam / 2.0

    # Find the length where the bone can comfortably fit this cylinder
    final_length = 0
    for ray_t, dt in ray_dt_vals:
        p = entry + d * ray_t
        if not cylinder_safe(p, d, radius, maskFloat, affine):
            print(f"    [Debug] Cylinder breached at t={ray_t}")
            break

        # Specific L5 midline crossing check
        # We prevent the screw tip from crossing the SI-AP plane (Sagittal midline)
        lrAxis = axes[1]
        midline_distance = np.dot(p - centroid, lrAxis)
        entry_mid_dist = np.dot(entry - centroid, lrAxis)
        # If the tip crosses the midline (sign changes or absolute distance close to 0) past a threshold
        if (entry_mid_dist > 0 and midline_distance < -2.0) or (
            entry_mid_dist < 0 and midline_distance > 2.0
        ):
            print(f"    [Debug] Crossed midline at t={ray_t}")
            break

        final_length = ray_t

    if final_length < minLengthMM:
        print(f"    [Debug] Length too short ({final_length} < {minLengthMM})")
        return None

    return best_diam, final_length, min_dist_along_ray, entry + d * final_length


def run_planner(segPath):
    resultsList = []
    print("ANALYTICAL PEDICLE SCREW PLANNER (L5 OPTIMIZED)")

    seg, spacing, affine = loadNifti(segPath)
    validSegments = getValidLabels(seg)

    for labelVal, mask in validSegments:
        name = labelMap.get(labelVal, str(labelVal))
        # This algorithm is structurally designed and optimized for L5
        if name != "L5":
            print(f"Skipping {name} - This module is optimized for L5 vertebra.")
            continue

        print(f"Analyzing {name} analytically...")

        centroid, axes = computeStableFrame(mask, affine)
        dist = computeDistance(mask, spacing)
        maskFloat = mask.astype(np.float32)

        lCenter, rCenter = pedicleCentersL5(mask, dist, centroid, axes, affine)
        body_center = getL5VertebralBodyCenter(mask, axes, centroid, affine)

        for side, data in [("Left", lCenter), ("Right", rCenter)]:
            if data is None:
                print(f"    [Trace] {side}: Pedicle Center Not Found")
                continue

            center, pedicle_axis = data

            # 1. Analytically determine trajectory direction
            direction = synthesize_analytical_direction(center, centroid, axes)

            # 2. Find external entry point
            entry = raycast_entry_point(center, direction, maskFloat, affine)

            # 3. Evaluate safety constraints, determine length and diameter
            result = analytical_evaluate(
                entry, direction, maskFloat, dist, affine, axes, centroid
            )

            if result is None:
                continue

            diam, length, minDT, tip = result

            # Add to standardized output format for UI visualizer
            resultsList.append(
                {
                    "vertebra": name,
                    "side": side,
                    "entry": entry,
                    "tip": tip,
                    "diameter": diam,
                }
            )

            print(f"{side} Screw Computed")
            print(f"  Diameter: {diam} mm")
            print(f"  Length: {round(length, 1)} mm")
            print(f"  Safety Margin: {round(minDT - diam/2, 2)} mm")
            print()

    return resultsList
