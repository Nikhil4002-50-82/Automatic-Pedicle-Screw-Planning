import concurrent.futures

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

# ---------------------------------------------------------------------------
#  Imports from the existing codebase (NOT modified)
# ---------------------------------------------------------------------------
from visualizer import visualize_surgical_plan

# Reuse the battle-tested pedicle localization utilities from the codebase.
# These use distance transforms, PCA, and anatomical filtering — far more
# accurate than naive mesh-centroid heuristics.
from analytical_geometry import (
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


def _select_l5_segment(valid_segments):
    for label_val, mask in valid_segments:
        if labelMap.get(label_val, str(label_val)) == "L5":
            return label_val, mask
    return None, None


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
        segmented_file = f.readline().strip()

    print(f"[Runner] Segmented file: {segmented_file}")

    if not os.path.exists(segmented_file):
        print(f"[Runner] ERROR: segmented file does not exist: {segmented_file}")
        sys.exit(1)

    # --- Load segmentation for anatomical analysis ---
    print("[Runner] Loading segmentation for pedicle localization...")
    seg, spacing, affine = loadNifti(segmented_file)
    validSegments = getValidLabels(seg)

    if len(validSegments) == 0:
        nonzero_voxels = int(np.count_nonzero(seg))
        print(
            "[Runner] ERROR: No valid vertebra segments found in the segmentation "
            f"(non-zero voxels: {nonzero_voxels})."
        )
        sys.exit(1)

    labelVal, mask = _select_l5_segment(validSegments)
    if mask is None:
        found_levels = [labelMap.get(value, str(value)) for value, _ in validSegments]
        print(
            "[Runner] ERROR: No L5 segment was found in the loaded segmentation. "
            f"Available levels: {', '.join(found_levels)}"
        )
        sys.exit(1)
    name = labelMap.get(labelVal, str(labelVal))
    print(f"[Runner] Found segment: label={labelVal} → {name}")

    # Keep the rendered anatomy aligned with the loaded mask, even when the
    # planning algorithm is restricted to the L5 component.
    try:
        vertsWorld, faces = build_mesh_from_single_vertebra(
            segmented_file=segmented_file,
            mask_name="loaded segmentation",
        )
    except ValueError as exc:
        print(f"[Runner] ERROR: {exc}")
        sys.exit(1)

    if len(validSegments) > 1:
        print("[Runner] Visualization uses the full loaded mask; planning remains restricted to L5.")

    # --- Compute robust L5 anatomical frame ---
    from analytical_geometry import computeStableFrameL5
    dist = computeDistance(mask, spacing)
    centroid, axes = computeStableFrameL5(mask, affine, dist)
    siAxis, lrAxis, apAxis = axes
    print(f"[Runner] Centroid (L5 robust) = {np.round(centroid, 2)}")
    print(f"[Runner] SI axis  = {np.round(siAxis, 4)}")
    print(f"[Runner] LR axis  = {np.round(lrAxis, 4)}")
    print(f"[Runner] AP axis  = {np.round(apAxis, 4)}")
    maskFloat = mask.astype(np.float32)

    # --- Find the true robust Anterior Target (Vertebral Body Center) ---
    from analytical_geometry import getL5VertebralBodyCenter
    anterior_center = getL5VertebralBodyCenter(mask, axes, centroid, affine)
    print(f"[Runner] Anterior body center (L5 robust) = {np.round(anterior_center, 2)}")

    # --- Find anatomically correct pedicle centers ---
    print("[Runner] Locating pedicle centers (L5-specific filters)...")
    lData, rData = pedicleCentersL5(mask, dist, centroid, axes, affine)

    if lData is None or rData is None:
        print("[Runner] ERROR: Could not locate one or both pedicle centers!")
        sys.exit(1)

    lCenter, lPedAxis = lData
    rCenter, rPedAxis = rData
    print(f"[Runner] Left  pedicle center = {np.round(lCenter, 2)}")
    print(f"[Runner] Right pedicle center = {np.round(rCenter, 2)}")

    # --- Measure actual pedicle dimensions from the segmentation ---
    print("[Runner] Measuring left pedicle dimensions...")
    lWidth, lHeight = measure_pedicle_dimensions(lCenter, axes, dist, mask, affine)
    print("[Runner] Measuring right pedicle dimensions...")
    rWidth, rHeight = measure_pedicle_dimensions(rCenter, axes, dist, mask, affine)

    # --- Plan screws using the deterministic trajectory planner ---
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

    from analytical_geometry import cylinder_safe

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
