import concurrent.futures
from pathlib import Path

# Top-level function for multiprocessing
def angle_eval(args):
    lrAng, siAng, true_center, centroid, lrAxis, siAxis, apAxis, maskFloat, affine, mask = args
    # Import here to avoid issues with multiprocessing on Windows
    import numpy as np
    from scipy.ndimage import map_coordinates
    from nibabel.affines import apply_affine
    # Use the same logic as before
    lateral_offset = np.dot(true_center - centroid, lrAxis)
    medial_dir = lrAxis if lateral_offset < 0 else -lrAxis
    lr_rad = np.deg2rad(lrAng)
    si_rad = np.deg2rad(siAng)
    test_dir = apAxis + np.tan(lr_rad) * medial_dir + np.tan(si_rad) * siAxis
    test_dir = test_dir / np.linalg.norm(test_dir)
    # Robust raycast entry point
    invAff = np.linalg.inv(affine)
    stepMM = 0.5
    p = true_center - test_dir * 60.0
    entry = None
    for i in range(200):
        p = p + test_dir * stepMM
        vox = apply_affine(invAff, p)
        if any(v < 0 or v >= s - 1 for v, s in zip(vox, maskFloat.shape)):
            continue
        val = map_coordinates(maskFloat, [[vox[0]], [vox[1]], [vox[2]]], order=1)[0]
        if val >= 0.5:
            entry = p + test_dir * 1.0
            break
    if entry is None:
        entry = true_center
    # Measure bone path length
    max_steps = 200
    found_exit = False
    depth = 0.0
    for i in range(4, max_steps):
        p2 = entry + test_dir * (i * stepMM)
        vox = apply_affine(invAff, p2)
        if any(v < 0 or v >= s - 1 for v, s in zip(vox, mask.shape)):
            depth = i * stepMM
            found_exit = True
            break
        val = map_coordinates(mask.astype(np.float32), [[vox[0]], [vox[1]], [vox[2]]], order=1)[0]
        if val < 0.5:
            depth = i * stepMM
            found_exit = True
            break
    if not found_exit:
        depth = max_steps * stepMM
    path_len = depth
    step = 2.0
    ts = np.arange(4.0, path_len, step)
    if ts.size == 0:
        print(f"[AngleEval] (LR={lrAng:.1f}, SI={siAng:.1f}) - No valid steps.")
        return None
    points = entry + np.outer(ts, test_dir)
    # Batched cylinder safety check (copied from main function)
    v = np.array([1, 0, 0]) if abs(test_dir[0]) < 0.9 else np.array([0, 1, 0])
    u1 = np.cross(test_dir, v)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(test_dir, u1)
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    offsets = np.array([2.25 * (np.cos(a) * u1 + np.sin(a) * u2) for a in angles])
    all_points = points[:, None, :] + offsets[None, :, :]
    all_points_flat = all_points.reshape(-1, 3)
    vox_cyl = apply_affine(invAff, all_points_flat)
    in_bounds = np.all((vox_cyl >= 0) & (vox_cyl < (np.array(maskFloat.shape) - 1)), axis=1)
    vals = np.zeros(vox_cyl.shape[0], dtype=np.float32)
    valid_idx = np.where(in_bounds)[0]
    if valid_idx.size > 0:
        vals[valid_idx] = map_coordinates(maskFloat, [vox_cyl[valid_idx,0], vox_cyl[valid_idx,1], vox_cyl[valid_idx,2]], order=1)
    vals = vals.reshape(points.shape[0], 8)
    safe_mask = np.all(vals >= 0.5, axis=1)
    p_lateral = np.dot(points - centroid, lrAxis)
    if lateral_offset < 0:
        midline_mask = p_lateral <= 0
    else:
        midline_mask = p_lateral >= 0
    valid_mask = safe_mask & midline_mask
    if np.any(valid_mask):
        safe_len = ts[valid_mask][-1]
    else:
        safe_len = 0.0
    print(f"[AngleEval] (LR={lrAng:.1f}, SI={siAng:.1f}) - Safe length: {safe_len:.2f} mm")
    return (safe_len, test_dir, entry, path_len)
"""
Standalone runner for L5 deterministic trajectory planner + 3D visualizer.

Compatible with the existing ``visualizer.py`` module.
Reads the segmented L5 NIfTI path from ``data/source.txt``, builds a
marching-cubes mesh, uses the proven segmentation-based pedicle localization
from ``analytical_geometry.py``, plans both pedicle screws, and opens the
Plotly interactive 3D visualization.

Usage
-----
    cd app/
    python plan_and_visualize_l5.py

    OR from project root:
    python app/plan_and_visualize_l5.py
"""

import os
import sys
import numpy as np
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# ---------------------------------------------------------------------------
#  Imports from the existing codebase (NOT modified)
# ---------------------------------------------------------------------------
from app.visualizer import visualize_surgical_plan

# Reuse the battle-tested pedicle localization utilities from the codebase.
# These use distance transforms, PCA, and anatomical filtering — far more
# accurate than naive mesh-centroid heuristics.
from app.analytical_geometry import (
    loadNifti,
    getValidLabels,
    computeStableFrame,
    computeDistance,
    pedicleCentersL5,
    raycast_entry_point,
    synthesize_analytical_direction,
    labelMap,
)


def build_mesh_from_single_vertebra(segmented_file=None, data=None, affine=None, mask_name="vertebra"):
    """
    Build a mesh from a single vertebra .nii file (same as
    plan_and_visualize_geometry.py — duplicated here to stay self-contained
    without modifying existing files).
    """
    import nibabel as nib
    from skimage.measure import marching_cubes

    if data is None:
        if segmented_file is None:
            raise ValueError("Either segmented_file or data/affine must be provided.")
        print(f"[Mesh] Loading segmentation: {segmented_file}")
        nii = nib.load(segmented_file)
        data = nii.get_fdata()
        affine = nii.affine

    if affine is None:
        raise ValueError("Affine is required when mesh data is provided directly.")

    binary_mask = np.asarray(data) > 0
    voxel_count = int(np.count_nonzero(binary_mask))
    print(f"[Mesh] Volume shape: {binary_mask.shape}, voxel count: {voxel_count}")

    if voxel_count == 0:
        raise ValueError(
            f"[Mesh] Cannot build a mesh for {mask_name}: the segmentation mask is empty."
        )

    verts, faces, _, _ = marching_cubes(binary_mask.astype(np.uint8), level=0.5)
    vertsWorld = nib.affines.apply_affine(affine, verts)
    print(f"[Mesh] Mesh built: {len(verts)} vertices, {len(faces)} faces")
    return vertsWorld, faces


def _full_spine_candidate_paths(segmented_file):
    """Return likely companion L1-L5 segmentation files for an L5-only input."""
    source_path = Path(segmented_file)
    candidates = []

    filename = source_path.name
    if "_segmented_vertebrae_L5" in filename:
        candidates.append(source_path.with_name(filename.replace(
            "_segmented_vertebrae_L5",
            "_segmented_vertebrae_L1_vertebrae_L5",
        )))

    prefix, marker, _ = filename.partition("_segmented_vertebrae_")
    if marker:
        candidates.extend(
            sorted(
                source_path.parent.glob(f"{prefix}_segmented_vertebrae_L1_vertebrae_L5.nii*")
            )
        )

    unique_candidates = []
    seen = set()
    source_resolved = source_path.resolve()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved == source_resolved or resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(candidate)

    return unique_candidates


def _resolve_visualization_mask(segmented_file, seg, affine):
    """
    Choose the visualization mask source.
    Planning always uses the original L5 segmentation; visualization prefers
    the full L1-L5 mask when it is already present or when a companion file exists.
    """
    nonzero_labels = np.unique(seg[seg > 0])
    if nonzero_labels.size > 1:
        print("[Runner] Visualization source: current file already contains multiple vertebra labels.")
        return segmented_file, seg > 0, affine

    for candidate in _full_spine_candidate_paths(segmented_file):
        if not candidate.exists():
            continue
        print(f"[Runner] Visualization source: using companion full-spine mask: {candidate}")
        vis_seg, _, vis_affine = loadNifti(str(candidate))
        return str(candidate), vis_seg > 0, vis_affine

    print("[Runner] Visualization source: no companion full-spine mask found. Falling back to L5-only mask.")
    return segmented_file, seg > 0, affine


def _crop_planning_mask(mask, affine, spacing, padding_mm=80.0, min_padding_voxels=8):
    """
    Crop the L5 planning mask to a padded bounding box.

    The mask is reduced to speed up distance transform / PCA / raycasting,
    while the affine is shifted so world coordinates remain unchanged.
    """
    import nibabel as nib

    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return mask, affine

    coords = np.argwhere(mask)
    min_idx = coords.min(axis=0)
    max_idx = coords.max(axis=0)

    spacing_xyz = np.asarray(spacing[:3], dtype=float)
    spacing_xyz = np.where(spacing_xyz > 0, spacing_xyz, 1.0)
    padding_vox = np.maximum(
        np.ceil(float(padding_mm) / spacing_xyz).astype(int),
        int(min_padding_voxels),
    )

    start = np.maximum(min_idx - padding_vox, 0)
    stop = np.minimum(max_idx + padding_vox + 1, np.asarray(mask.shape, dtype=int))

    slices = tuple(slice(int(start[i]), int(stop[i])) for i in range(3))
    cropped_mask = mask[slices]

    cropped_affine = np.array(affine, dtype=float, copy=True)
    cropped_affine[:3, 3] = nib.affines.apply_affine(affine, start)

    print(
        "[Runner] Planning crop: "
        f"shape {tuple(mask.shape)} -> {tuple(cropped_mask.shape)}, "
        f"pad={padding_mm:.1f}mm"
    )
    return cropped_mask, cropped_affine


def measure_pedicle_dimensions(center, axes, dist, mask, affine):
    """
    Estimate pedicle width (mediolateral) and height (craniocaudal) at
    the pedicle center using the distance transform.

    The distance transform value at the center gives the radius of the
    largest inscribed sphere.  We probe along LR and SI axes to measure
    actual cortical-wall clearance.

    Returns (pedicle_width_mm, pedicle_height_mm).
    """
    import nibabel as nib
    from scipy.ndimage import map_coordinates

    siAxis, lrAxis, apAxis = axes
    invAff = np.linalg.inv(affine)
    stepMM = 0.3

    def probe_extent(origin, direction, max_steps=200):
        """Walk from origin in direction until we leave the bone mask. Return distance in mm."""
        p = origin.copy()
        for i in range(max_steps):
            p = p + direction * stepMM
            vox = nib.affines.apply_affine(invAff, p)
            if any(v < 0 or v >= s - 1 for v, s in zip(vox, mask.shape)):
                return i * stepMM
            val = map_coordinates(
                mask.astype(np.float32), [[vox[0]], [vox[1]], [vox[2]]], order=1
            )[0]
            if val < 0.5:
                return i * stepMM
        return max_steps * stepMM

    # Mediolateral (width) = extent in +LR and −LR
    w_pos = probe_extent(center, +lrAxis)
    w_neg = probe_extent(center, -lrAxis)
    width = w_pos + w_neg

    # Craniocaudal (height) = extent in +SI and −SI
    h_pos = probe_extent(center, +siAxis)
    h_neg = probe_extent(center, -siAxis)
    height = h_pos + h_neg

    print(f"  [Pedicle Dim] Width (LR): +{w_pos:.1f} / −{w_neg:.1f} = {width:.1f} mm")
    print(f"  [Pedicle Dim] Height (SI): +{h_pos:.1f} / −{h_neg:.1f} = {height:.1f} mm")

    # Sanity clamp — never report less than 5 mm or more than 30 mm
    width = max(5.0, min(width, 30.0))
    height = max(5.0, min(height, 30.0))
    return width, height


def robust_raycast_entry_point(center, direction, mask, affine):
    """
    Find the posterior entry point by starting well outside the bone (posteriorly)
    and raymarching FORWARD along the trajectory until we hit the bone surface.
    This guarantees we find the true posterior cortical surface (superior articular process / lamina)
    rather than accidentally breaking out of the lateral pedicle wall from the inside.
    """
    import nibabel as nib
    from scipy.ndimage import map_coordinates

    invAff = np.linalg.inv(affine)
    stepMM = 0.5

    # Start 60mm backward along the trajectory
    p = center - direction * 60.0

    # March forward until we hit the bone mask (val >= 0.5)
    for i in range(200):
        p = p + direction * stepMM
        vox = nib.affines.apply_affine(invAff, p)

        # Check bounds
        if any(v < 0 or v >= s - 1 for v, s in zip(vox, mask.shape)):
            continue

        val = map_coordinates(
            mask.astype(np.float32), [[vox[0]], [vox[1]], [vox[2]]], order=1
        )[0]

        if val >= 0.5:
            # We hit the exterior posterior surface!
            return p + direction * 1.0  # Embed 1mm for the screw head

    # Fallback to center if something failed
    print(
        "  [Raycast] WARNING: Failed to hit bone from outside, falling back to pedicle center."
    )
    return center


def measure_bone_path_length(entry_point, direction, mask, affine):
    """
    Measure the distance from the entry point through the bone mask along
    the specified trajectory direction before exiting the anterior cortex.

    Returns the available bone length in mm.
    """
    import nibabel as nib
    from scipy.ndimage import map_coordinates

    invAff = np.linalg.inv(affine)
    stepMM = 0.5

    # Start slightly inside to ensure we are in the mask
    p = entry_point + direction * 2.0

    max_steps = 200  # Up to 100mm
    found_exit = False
    depth = 0.0

    for i in range(4, max_steps):
        p = entry_point + direction * (i * stepMM)
        vox = nib.affines.apply_affine(invAff, p)

        # Out of bounds
        if any(v < 0 or v >= s - 1 for v, s in zip(vox, mask.shape)):
            depth = i * stepMM
            found_exit = True
            break

        val = map_coordinates(
            mask.astype(np.float32), [[vox[0]], [vox[1]], [vox[2]]], order=1
        )[0]

        # Exited bone
        if val < 0.5:
            depth = i * stepMM
            found_exit = True
            break

    if not found_exit:
        depth = max_steps * stepMM

    print(f"  [Bone Path] Anterior cortex at {depth:.1f} mm from entry point")
    return depth


def plan_and_visualize_l5():
    """
    Main orchestration: read source, build mesh, locate pedicles from the
    segmentation, plan screws, visualize.
    """
    import time
    global start_time
    if 'start_time' not in globals():
        start_time = time.time()
    from l5_trajectory_planner import (
        plan_l5_pedicle_screw,
        compute_trajectory,
        compute_world_trajectory,
    )

    # --- Read segmentation path from data/source.txt ---
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    source_txt = os.path.join(data_dir, "source.txt")

    if not os.path.exists(source_txt):
        print(f"[Runner] ERROR: source file not found at {source_txt}")
        sys.exit(1)

    with open(source_txt, "r") as f:
        segmented_file = f.readline().strip().strip('"').strip("'")

    print(f"[Runner] Segmented file: {segmented_file}")

    if not os.path.exists(segmented_file):
        print(f"[Runner] ERROR: segmented file does not exist: {segmented_file}")
        sys.exit(1)

    # --- Load segmentation for anatomical analysis ---
    print("[Runner] Loading segmentation for pedicle localization...")
    t_start = time.perf_counter()
    seg, spacing, affine = loadNifti(segmented_file)
    elapsed_load = time.perf_counter() - t_start
    print(f"[Timer] Segmentation load completed in {elapsed_load:.2f}s")
    
    # OPTIMIZATION: For L1-L5 files, extract ONLY L5 (label=1) to avoid
    # expensive connected component analysis on 5 vertebrae when we only need 1.
    # This can speed up loading from ~60+ seconds to ~5-20 seconds!
    L5_LABEL = 1  # According to labelMap: {5: "L1", 4: "L2", 3: "L3", 2: "L4", 1: "L5"}
    if L5_LABEL in np.unique(seg):
        print(f"[Runner] L5-specific extraction: Isolating L5 (label={L5_LABEL}) from multi-level segmentation")
        t_start = time.perf_counter()
        mask_l5_only = seg == L5_LABEL
        from scipy.ndimage import label as cc_label
        labeled, _ = cc_label(mask_l5_only)
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        if len(sizes) > 0:
            largest = np.argmax(sizes)
            mask = labeled == largest
            labelVal = L5_LABEL
            elapsed_extract = time.perf_counter() - t_start
            print(f"[Timer] L5 extraction completed in {elapsed_extract:.2f}s")
            print(f"[Runner] Extracted L5: {np.count_nonzero(mask)} voxels")
        else:
            # Fallback if extraction fails
            validSegments = getValidLabels(seg)
            if not validSegments:
                print("[Runner] ERROR: No valid L5 segment found!")
                sys.exit(1)
            labelVal, mask = validSegments[0]
    else:
        # No L5 label found, use old method (single L5 file)
        print("[Runner] L5-only file detected. Processing single segment...")
        validSegments = getValidLabels(seg)
        if not validSegments:
            nonzero_voxels = int(np.count_nonzero(seg))
            print(
                "[Runner] ERROR: No valid vertebra segments found in the segmentation "
                f"(non-zero voxels: {nonzero_voxels})."
            )
            sys.exit(1)
        labelVal, mask = validSegments[0]
    
    name = labelMap.get(labelVal, str(labelVal))
    print(f"[Runner] Found segment: label={labelVal} → {name}")

    # --- Build visualization mesh ---
    # Planning stays L5-only; visualization can use the full L1-L5 mask when available.
    print("[Runner] Resolving visualization mask...")
    vis_volume_path, vis_mask, vis_affine = _resolve_visualization_mask(
        segmented_file,
        seg,
        affine,
    )
    print("[Runner] Building visualization mesh...")
    try:
        vertsWorld, faces = build_mesh_from_single_vertebra(
            segmented_file=vis_volume_path,
            data=vis_mask,
            affine=vis_affine,
            mask_name=f"VisualizationMask-{name}",
        )
    except ValueError as exc:
        print(f"[Runner] ERROR: {exc}")
        sys.exit(1)

    # --- Crop the planning volume only ---
    # This keeps all analytical steps on a much smaller grid without changing world coordinates.
    mask, affine = _crop_planning_mask(mask, affine, spacing, padding_mm=80.0)

    # --- Compute robust L5 anatomical frame ---
    from app.analytical_geometry import computeStableFrameL5
    import time
    print("[Runner] Computing distance transform... (time-consuming on large volumes)")
    t_start = time.perf_counter()
    dist = computeDistance(mask, spacing)
    elapsed_dist = time.perf_counter() - t_start
    print(f"[Timer] Distance transform completed in {elapsed_dist:.2f}s")
    
    t_start = time.perf_counter()
    centroid, axes = computeStableFrameL5(mask, affine, dist)
    elapsed_frame = time.perf_counter() - t_start
    print(f"[Timer] Stable frame computation completed in {elapsed_frame:.2f}s")
    siAxis, lrAxis, apAxis = axes
    print(f"[Runner] Centroid (L5 robust) = {np.round(centroid, 2)}")
    print(f"[Runner] SI axis  = {np.round(siAxis, 4)}")
    print(f"[Runner] LR axis  = {np.round(lrAxis, 4)}")
    print(f"[Runner] AP axis  = {np.round(apAxis, 4)}")
    maskFloat = mask.astype(np.float32)

    # --- Find the true robust Anterior Target (Vertebral Body Center) ---
    from app.analytical_geometry import getL5VertebralBodyCenter
    anterior_center = getL5VertebralBodyCenter(mask, axes, centroid, affine)
    print(f"[Runner] Anterior body center (L5 robust) = {np.round(anterior_center, 2)}")

    # --- Find anatomically correct pedicle centers ---
    print("[Runner] Locating pedicle centers (L5-specific filters)...")
    t_start = time.perf_counter()
    lData, rData = pedicleCentersL5(mask, dist, centroid, axes, affine)
    elapsed_pedicles = time.perf_counter() - t_start
    print(f"[Timer] Pedicle center location completed in {elapsed_pedicles:.2f}s")

    if lData is None or rData is None:
        print("[Runner] ERROR: Could not locate one or both pedicle centers!")
        sys.exit(1)

    lCenter, lPedAxis = lData
    rCenter, rPedAxis = rData
    print(f"[Runner] Left  pedicle center = {np.round(lCenter, 2)}")
    print(f"[Runner] Right pedicle center = {np.round(rCenter, 2)}")

    # --- Measure actual pedicle dimensions from the segmentation ---
    print("[Runner] Measuring left pedicle dimensions...")
    t_start = time.perf_counter()
    lWidth, lHeight = measure_pedicle_dimensions(lCenter, axes, dist, mask, affine)
    elapsed_lmeas = time.perf_counter() - t_start
    print(f"[Timer] Left pedicle measurement completed in {elapsed_lmeas:.2f}s")
    
    print("[Runner] Measuring right pedicle dimensions...")
    t_start = time.perf_counter()
    rWidth, rHeight = measure_pedicle_dimensions(rCenter, axes, dist, mask, affine)
    elapsed_rmeas = time.perf_counter() - t_start
    print(f"[Timer] Right pedicle measurement completed in {elapsed_rmeas:.2f}s")

    # --- Plan screws using the deterministic trajectory planner ---
    print("[Runner] Starting trajectory planning for both pedicles...")
    t_start = time.perf_counter()
    resultsList = []

    # Calculate lateral distances to determine true pedicle vs. transverse process
    left_dist = np.abs(np.dot(lCenter - centroid, lrAxis))
    right_dist = np.abs(np.dot(rCenter - centroid, lrAxis))
    
    print(f"[Runner] Mediolateral distance from centroid: Left={left_dist:.1f}mm, Right={right_dist:.1f}mm")

    # The point closer to the midline is the true pedicle
    if left_dist < right_dist:
        print("[Runner] Left point is closer to midline. Symmetrizing Right to match Left's lateral offset...")
        # Mirror the left lateral offset to the right side
        final_lCenter = lCenter
        
        # Project lCenter's LR offset to the right side
        lateral_offset = np.dot(lCenter - centroid, lrAxis)
        # Assuming lrAxis points Right (positive), so left is negative
        # Mirror: if left is at -X, right should be at +X (-offset)
        final_rCenter = lCenter - 2 * lateral_offset * lrAxis
    else:
        print("[Runner] Right point is closer to midline. Symmetrizing Left to match Right's lateral offset...")
        final_rCenter = rCenter
        
        lateral_offset = np.dot(rCenter - centroid, lrAxis)
        final_lCenter = rCenter - 2 * lateral_offset * lrAxis

    print(f"[Runner] Final Left center  = {np.round(final_lCenter, 2)}")
    print(f"[Runner] Final Right center = {np.round(final_rCenter, 2)}")

    from app.analytical_geometry import cylinder_safe

    for side, true_center, p_width, p_height in [
        ("left", final_lCenter, lWidth, lHeight),
        ("right", final_rCenter, rWidth, rHeight),
    ]:
        print(f"\n--- Constructing {side.capitalize()} Trajectory ---")
        
        # 1. Deterministic trajectory calculation 
        tpa = 25.0
        # The user requested to LOWER the posterior entry point. 
        # By setting a positive Sagittal Pedicle Angle (SPA), we tilt the anterior tip SUPERIORLY.
        # Pivoting around the pedicle center, this mathematically drops the posterior entry point INFERIORLY.
        spa = 10.0  

        # Calculate base trajectory direction
        test_dir = compute_world_trajectory(
            center=true_center,
            anterior_target=anterior_center,
            anatomical_axes=axes,
            transverse_pedicle_angle=tpa,
            sagittal_pedicle_angle=spa,
            side=side
        )

        # 2. Find posterior ENTRY POINT mathematically
        # DO NOT assume PIP is the entry point. Raycast BACKWARD to find true surface entry.
        best_entry = robust_raycast_entry_point(true_center, test_dir, maskFloat, affine)
        best_bone_length = measure_bone_path_length(best_entry, test_dir, maskFloat, affine)
        
        # 3. Let our trajectory planner execute to evaluate optimal length & diameter
        plan = plan_l5_pedicle_screw(
            entry_point=best_entry,
            pedicle_width=p_width,
            pedicle_height=p_height,
            vertebral_body_depth=best_bone_length,
            transverse_pedicle_angle=tpa,
            sagittal_pedicle_angle=spa,
            safety_margin=2.0,
            side=side,
            anatomical_axes=axes,
            pedicle_center=true_center,
            anterior_target=anterior_center
        )

        if plan is None:
            print(f"[Runner] ERROR: Trajectory formulation totally failed for {side} pedicle!")
            continue

        best_traj = plan["trajectory_vector"]
        screw_length = plan["recommended_screw_length"]
        screw_diam = plan["recommended_screw_diameter"]
        tip = plan["tip_point"]
        
        resultsList.append({
            "vertebra": "L5",
            "side": side.capitalize(),
            "entry": best_entry,
            "tip": tip,
            "diameter": screw_diam,
            "length": screw_length,
        })
        
        print(f"[Runner] {side.capitalize()} Entry: {np.round(best_entry, 2)}")
        print(f"[Runner] {side.capitalize()} Trajectory Tip: {np.round(tip, 2)}")
        print(f"[Runner] {side.capitalize()} Bone Length: {best_bone_length:.2f} mm (actual), {screw_length:.2f} mm (used)")

    elapsed_traj = time.perf_counter() - t_start
    print(f"[Timer] Trajectory planning completed in {elapsed_traj:.2f}s")

    # --- Visualize anterior center as a marker for debug ---
    import plotly.graph_objects as go
    anterior_marker = go.Scatter3d(
        x=[anterior_center[0]],
        y=[anterior_center[1]],
        z=[anterior_center[2]],
        mode='markers',
        marker=dict(size=7, color='red'),
        name='Anterior Center (L5)'
    )

    # --- Visualize ---
    import time
    elapsed = time.time() - start_time
    print(f"[Timer] Total runtime (before visualization): {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    fig, show_figure = visualize_surgical_plan(vertsWorld, faces, resultsList, volume_path=segmented_file)
    # Add anterior marker to the figure if possible
    try:
        fig.add_trace(anterior_marker)
    except Exception:
        pass
    show_figure(fig)
    print("[Runner] L5 entry and trajectory planning and visualization completed.")


if __name__ == "__main__":
    import time
    start_time = time.time()
    plan_and_visualize_l5()
    elapsed = time.time() - start_time
    print(f"[Timer] Total runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
