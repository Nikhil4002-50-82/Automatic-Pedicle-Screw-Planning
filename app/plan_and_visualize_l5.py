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


def build_mesh_from_single_vertebra(segmented_file):
    """
    Build a mesh from a single vertebra .nii file (same as
    plan_and_visualize_geometry.py — duplicated here to stay self-contained
    without modifying existing files).
    """
    import nibabel as nib
    from skimage.measure import marching_cubes

    print(f"[Mesh] Loading segmentation: {segmented_file}")
    nii = nib.load(segmented_file)
    data = nii.get_fdata()
    affine = nii.affine
    print(f"[Mesh] Volume shape: {data.shape}, voxel count: {int(np.sum(data > 0))}")

    verts, faces, _, _ = marching_cubes(data, level=0.5)
    vertsWorld = nib.affines.apply_affine(affine, verts)
    print(f"[Mesh] Mesh built: {len(verts)} vertices, {len(faces)} faces")
    return vertsWorld, faces


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

    # --- Build mesh for visualization ---
    vertsWorld, faces = build_mesh_from_single_vertebra(segmented_file)

    # --- Load segmentation for anatomical analysis ---
    print("[Runner] Loading segmentation for pedicle localization...")
    seg, spacing, affine = loadNifti(segmented_file)
    validSegments = getValidLabels(seg)

    if len(validSegments) == 0:
        print("[Runner] ERROR: No valid vertebra segments found!")
        sys.exit(1)

    # Use the first (and usually only) segment
    labelVal, mask = validSegments[0]
    name = labelMap.get(labelVal, str(labelVal))
    print(f"[Runner] Found segment: label={labelVal} → {name}")

    # --- Compute robust L5 anatomical frame ---
    from geometry_2 import computeStableFrameL5
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
    
    for side, true_center in [
        ("left", final_lCenter),
        ("right", final_rCenter),
    ]:
        print(f"\n--- Constructing {side.capitalize()} Trajectory ---")
        print(f"[Runner] Running Generalized Volumetric Grid Search for {side} pedicle...")
        
        # Determine medial direction (+lrAxis or -lrAxis)
        lateral_offset = np.dot(true_center - centroid, lrAxis)
        medial_dir = lrAxis if lateral_offset < 0 else -lrAxis
        
        best_score = -1.0
        best_traj = None
        best_entry = None
        best_bone_length = 0.0
        
        # Grid sweep of clinical angles: 
        # Medial convergence: 10 to 40 degrees
        # Sagittal tilt: -10 to +20 degrees 
        for lrAng in np.linspace(10, 40, 10):
            for siAng in np.linspace(-10, 20, 8):
                lr_rad = np.deg2rad(lrAng)
                si_rad = np.deg2rad(siAng)
                
                # Construct test trajectory pointing anteriorly (apAxis) 
                # plus defined medial and sagittal components.
                test_dir = apAxis + np.tan(lr_rad) * medial_dir + np.tan(si_rad) * siAxis
                test_dir = test_dir / np.linalg.norm(test_dir)
                
                # Check posterior entry point along this vector from the pedicle axis center
                entry = robust_raycast_entry_point(true_center, test_dir, maskFloat, affine)
                path_len = measure_bone_path_length(entry, test_dir, mask, affine)
                
                # Verify safety cylinder (2.25mm radius = 4.5mm screw)
                safe_len = 0.0
                step = 2.0
                
                # Test the cylinder's safety along the path
                for t in np.arange(4.0, path_len, step):
                    p = entry + test_dir * t
                    
                    if not cylinder_safe(p, test_dir, 2.25, maskFloat, affine):
                        break
                        
                    # Stop if it crosses the sagittal midline (spinal canal blowout protection)
                    p_lateral = np.dot(p - centroid, lrAxis)
                    if (lateral_offset < 0 and p_lateral > 0) or (lateral_offset > 0 and p_lateral < 0):
                        break
                    
                    safe_len = t
                    
                # Maximize completely safe bone path
                if safe_len > best_score:
                    best_score = safe_len
                    best_traj = test_dir
                    best_entry = entry
                    best_bone_length = path_len
                    
        if best_traj is None:
            print(f"[Runner] ERROR: Could not find any safe valid trajectory for {side} pedicle!")
            continue
            
        print(f"[Runner] Optimal {side.capitalize()} Trajectory: Safe inner core length = {best_score:.1f} mm")
        print(f"[Runner] Trajectory Vector: {np.round(best_traj, 4)}")
        
        # 3. Sink the screw
        # Use a safe clinical maximum
        screw_length = min(best_bone_length, 45.0)
        
        tip = best_entry + best_traj * screw_length
        exit_point = best_entry + best_traj * best_bone_length
        
        resultsList.append({
            "vertebra": "L5",
            "side": side.capitalize(),
            "entry": best_entry,
            "tip": tip
        })
        
        print(f"[Runner] {side.capitalize()} Entry: {np.round(best_entry, 2)}")
        print(f"[Runner] {side.capitalize()} Trajectory Tip: {np.round(tip, 2)}")
        print(f"[Runner] {side.capitalize()} Bone Length: {best_bone_length:.2f} mm (actual), {screw_length:.2f} mm (used)")


        # Add exit point marker for visualization
        if 'exit_markers' not in locals():
            exit_markers = []
        import plotly.graph_objects as go
        exit_markers.append(go.Scatter3d(
            x=[exit_point[0]],
            y=[exit_point[1]],
            z=[exit_point[2]],
            mode='markers',
            marker=dict(size=6, color='orange'),
            name=f'{side.capitalize()} Exit Point'
        ))

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
    fig = visualize_surgical_plan(vertsWorld, faces, resultsList, volume_path=segmented_file)
    # Add anterior marker and exit markers to the figure if possible
    try:
        fig.add_trace(anterior_marker)
        for marker in exit_markers:
            fig.add_trace(marker)
    except Exception:
        pass
    print("[Runner] L5 entry and trajectory planning and visualization completed.")


if __name__ == "__main__":
    plan_and_visualize_l5()
