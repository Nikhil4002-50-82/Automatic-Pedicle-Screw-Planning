"""
L5 Pedicle Screw Trajectory Planner — Deterministic Analytical Algorithm
=========================================================================

Computes a safe, anatomically valid pedicle screw trajectory for the L5
vertebra using closed-form geometry (rotation matrices) rather than
brute-force search.

Anatomical Reference Frame
--------------------------
    X axis → mediolateral  (left/right)
    Y axis → anteroposterior  (posterior → anterior)
    Z axis → craniocaudal  (inferior → superior)

The posterior vertebral surface is the reference plane.  The baseline
trajectory vector v₀ = (0, 1, 0) is perpendicular to this surface,
pointing anteriorly.

Trajectory Computation
----------------------
1. Start with v₀ = (0, 1, 0)
2. Rotate about Z-axis by ±transverse_pedicle_angle  (axial convergence)
3. Rotate about X-axis by  sagittal_pedicle_angle    (sup/inf tilt)
4. Normalize → final unit trajectory direction

Screw Sizing
-------------
Diameter: max_diam = min(pedicle_width, pedicle_height) − 2 × safety_margin,
          rounded down to the nearest standard size.

Length:   vertebral_body_depth − safety_margin,
          clamped to the clinical range [25, 55] mm.

Dependencies: numpy only.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

# Standard pedicle screw diameters available clinically (mm, descending)
STANDARD_DIAMETERS = [8.5, 7.5, 7.0, 6.5, 5.5, 5.0, 4.5, 4.0]

# Standard pedicle screw lengths available clinically (mm)
STANDARD_LENGTHS = [25, 30, 35, 40, 45, 50, 55]

# Clinical limits for L5 pedicle angles (degrees)
L5_TPA_RANGE = (15.0, 45.0)  # transverse pedicle angle
L5_SPA_RANGE = (-10.0, 10.0)  # sagittal pedicle angle

# Minimum/maximum clinically acceptable screw diameter (mm)
MIN_SCREW_DIAMETER = 4.0
MAX_SCREW_DIAMETER = 8.5

# ---------------------------------------------------------------------------
#  Rotation helpers
# ---------------------------------------------------------------------------


def _rotation_matrix_z(angle_deg: float) -> np.ndarray:
    """
    3×3 rotation matrix about the Z-axis (craniocaudal).
    Positive angle → medial rotation for a LEFT pedicle (toward midline).

    Parameters
    ----------
    angle_deg : float
        Rotation angle in degrees.

    Returns
    -------
    np.ndarray, shape (3, 3)
    """
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rotation_matrix_x(angle_deg: float) -> np.ndarray:
    """
    3×3 rotation matrix about the X-axis (mediolateral).
    Positive angle → superior tilt (cephalad angulation).

    Parameters
    ----------
    angle_deg : float
        Rotation angle in degrees.

    Returns
    -------
    np.ndarray, shape (3, 3)
    """
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# ---------------------------------------------------------------------------
#  Core trajectory computation
# ---------------------------------------------------------------------------


def compute_world_trajectory(
    center: np.ndarray,
    anterior_target: np.ndarray,
    anatomical_axes: np.ndarray,
    transverse_pedicle_angle: float,
    sagittal_pedicle_angle: float,
    side: str = "left",
) -> np.ndarray:
    """
    Compute the trajectory vector directly in world (scanner) coordinates,
    ensuring it always points anteriorly and converges medially regardless
    of arbitrary PCA axis inversions.

    Parameters
    ----------
    center : np.ndarray, shape (3,)
        Pedicle center in world coordinates.
    anterior_target : np.ndarray, shape (3,)
        A reliable point that is strictly anterior to the pedicles (e.g. vertebral body center).
    anatomical_axes : np.ndarray, shape (3, 3)
        Rows are [siAxis, lrAxis, apAxis] from computeStableFrame.
    transverse_pedicle_angle : float
        Medial angulation in the axial plane (degrees).
    sagittal_pedicle_angle : float
        Sup/inf tilt in the sagittal plane (degrees).
    side : str
        ``"left"`` or ``"right"`` pedicle.

    Returns
    -------
    np.ndarray, shape (3,)
        Trajectory direction in world coordinates (normalized).
    """
    siAxis = anatomical_axes[0]
    lrAxis = anatomical_axes[1]
    apAxis = anatomical_axes[2]

    # 1. Determine robust Anterior direction
    # We want apAxis to point from Posterior (pedicle) to Anterior (anterior_target).
    pa_vector = anterior_target - center
    if np.dot(pa_vector, apAxis) < 0:
        ap_dir = -apAxis
    else:
        ap_dir = apAxis

    # 2. Determine robust Medial direction
    # The anterior_target (center of body) is medial to the pedicles.
    if np.dot(pa_vector, lrAxis) < 0:
        medial_dir = -lrAxis
    else:
        medial_dir = lrAxis

    # 3. Determine robust Cranial direction
    # (Assuming we want positive SPA to point superiorly). We won't strictly
    # enforce this one against centroid because Y isn't as easily checked,
    # but normally computeStableFrame returns SI pointing superiorly.
    cranial_dir = siAxis

    # 4. Construct trajectory from angles
    tpa_rad = np.deg2rad(transverse_pedicle_angle)
    spa_rad = np.deg2rad(sagittal_pedicle_angle)

    # Base AP projection
    v_world = np.cos(tpa_rad) * ap_dir
    # Add medial convergence
    v_world += np.sin(tpa_rad) * medial_dir
    # Add sagittal tilt (simplified check)
    v_world += np.sin(spa_rad) * cranial_dir

    norm = np.linalg.norm(v_world)
    if norm < 1e-12:
        return ap_dir

    v_world = v_world / norm

    print(f"  [WorldTransform] AP   dir (world) = {np.round(ap_dir, 4)}")
    print(f"  [WorldTransform] Medial dir (world) = {np.round(medial_dir, 4)}")
    print(f"  [WorldTransform] World direction  = {np.round(v_world, 6)}")

    return v_world


def compute_trajectory(
    transverse_pedicle_angle: float,
    sagittal_pedicle_angle: float,
    side: str = "left",
) -> np.ndarray:
    """
    Compute the unit trajectory vector for an L5 pedicle screw.

    Steps
    -----
    1. v₀ = (0, 1, 0)  → perpendicular to posterior surface, pointing anteriorly.
    2. Rotate v₀ about Z by ±TPA  (sign depends on side).
    3. Rotate result about X by SPA.
    4. Normalize.

    Parameters
    ----------
    transverse_pedicle_angle : float
        Medial angulation in the axial plane (degrees, positive = medial).
    sagittal_pedicle_angle : float
        Sup/inf tilt in the sagittal plane (degrees, positive = cephalad).
    side : str
        ``"left"`` or ``"right"`` pedicle.

    Returns
    -------
    np.ndarray, shape (3,)
        Normalized trajectory direction.
    """
    side_lower = side.strip().lower()

    # Step 1 — baseline vector: anterior direction
    v0 = np.array([0.0, 1.0, 0.0])
    print(f"  [Trajectory] Baseline vector v0 = {v0}")

    # Step 2 — axial convergence (transverse pedicle angle about Z)
    # Left pedicle: rotate +TPA (medially = toward +X)
    # Right pedicle: rotate −TPA (medially = toward −X)
    if side_lower == "left":
        tpa_signed = +transverse_pedicle_angle
    else:
        tpa_signed = -transverse_pedicle_angle

    Rz = _rotation_matrix_z(tpa_signed)
    v1 = Rz @ v0
    print(
        f"  [Trajectory] After Z-rotation by {tpa_signed:+.1f}°  →  v1 = {np.round(v1, 6)}"
    )

    # Step 3 — sagittal tilt (about X)
    Rx = _rotation_matrix_x(sagittal_pedicle_angle)
    v2 = Rx @ v1
    print(
        f"  [Trajectory] After X-rotation by {sagittal_pedicle_angle:+.1f}°  →  v2 = {np.round(v2, 6)}"
    )

    # Step 4 — normalize
    norm = np.linalg.norm(v2)
    if norm < 1e-12:
        print("  [Trajectory] WARNING: Degenerate trajectory vector — returning v0")
        return v0
    trajectory = v2 / norm
    print(f"  [Trajectory] Final unit vector = {np.round(trajectory, 6)}")
    return trajectory


# ---------------------------------------------------------------------------
#  Screw diameter selection
# ---------------------------------------------------------------------------


def compute_screw_diameter(
    pedicle_width: float,
    pedicle_height: float,
    safety_margin: float,
) -> Tuple[float, List[str]]:
    """
    Select the largest standard screw diameter that fits within the pedicle
    with the specified safety margin.

    max_diameter = min(pedicle_width, pedicle_height) − 2 × safety_margin

    Parameters
    ----------
    pedicle_width : float   Mediolateral diameter of pedicle (mm).
    pedicle_height : float  Superior-inferior diameter of pedicle (mm).
    safety_margin : float   Minimum clearance from cortical walls (mm).

    Returns
    -------
    diameter : float        Recommended screw diameter (mm).
    warnings : list[str]    Any constraint warnings.
    """
    warnings: List[str] = []

    limiting_dim = min(pedicle_width, pedicle_height)
    max_diam = limiting_dim - 2.0 * safety_margin

    print(f"  [Diameter] Pedicle width={pedicle_width} mm, height={pedicle_height} mm")
    print(f"  [Diameter] Limiting dimension = {limiting_dim} mm")
    print(
        f"  [Diameter] Max safe diameter  = {limiting_dim} − 2×{safety_margin} = {max_diam:.2f} mm"
    )

    if max_diam < MIN_SCREW_DIAMETER:
        warnings.append(
            f"Pedicle too narrow: max safe diameter {max_diam:.1f} mm < "
            f"minimum screw size {MIN_SCREW_DIAMETER} mm"
        )
        print(f"  [Diameter] WARNING: {warnings[-1]}")
        return max_diam, warnings

    # Snap down to nearest standard size
    candidates = [d for d in STANDARD_DIAMETERS if d <= max_diam]
    if not candidates:
        warnings.append(f"No standard diameter fits (max_diam={max_diam:.1f} mm)")
        print(f"  [Diameter] WARNING: {warnings[-1]}")
        return max_diam, warnings

    diameter = max(candidates)
    print(f"  [Diameter] Selected standard diameter = {diameter} mm")
    return diameter, warnings


# ---------------------------------------------------------------------------
#  Screw length computation
# ---------------------------------------------------------------------------


def compute_screw_length(
    vertebral_body_depth: float,
    safety_margin: float,
) -> Tuple[float, List[str]]:
    """
    Compute the recommended screw length.

    length = vertebral_body_depth − safety_margin, clamped to [25, 55] mm.

    Parameters
    ----------
    vertebral_body_depth : float  Posterior-to-anterior cortex distance (mm).
    safety_margin : float         Clearance from anterior cortex (mm).

    Returns
    -------
    length : float        Recommended screw length (mm).
    warnings : list[str]  Any constraint warnings.
    """
    warnings: List[str] = []

    raw_length = vertebral_body_depth - safety_margin
    print(f"  [Length] Vertebral body depth = {vertebral_body_depth} mm")
    print(
        f"  [Length] Raw length = {vertebral_body_depth} − {safety_margin} = {raw_length:.2f} mm"
    )

    if raw_length > max(STANDARD_LENGTHS):
        warnings.append(
            f"Computed length {raw_length:.1f} mm exceeds max standard "
            f"{max(STANDARD_LENGTHS)} mm — clamped"
        )
        print(f"  [Length] WARNING: {warnings[-1]}")

    if raw_length < min(STANDARD_LENGTHS):
        warnings.append(
            f"Computed length {raw_length:.1f} mm below min standard "
            f"{min(STANDARD_LENGTHS)} mm — clamped"
        )
        print(f"  [Length] WARNING: {warnings[-1]}")

    # Snap down to nearest standard length
    candidates = [l for l in STANDARD_LENGTHS if l <= raw_length]

    if not candidates:
        # Too short for any standard — use the minimum
        length = min(STANDARD_LENGTHS)
        warnings.append(f"Using minimum standard length {length} mm")
        print(f"  [Length] WARNING: {warnings[-1]}")
    else:
        length = max(candidates)

    print(f"  [Length] Selected standard length = {length} mm")
    return length, warnings


# ---------------------------------------------------------------------------
#  Safety validation
# ---------------------------------------------------------------------------


def validate_plan(
    screw_diameter: float,
    screw_length: float,
    pedicle_width: float,
    pedicle_height: float,
    vertebral_body_depth: float,
    transverse_pedicle_angle: float,
    sagittal_pedicle_angle: float,
) -> List[str]:
    """
    Validate the planned screw against anatomical constraints.

    Returns
    -------
    warnings : list[str]
        Empty if all checks pass.
    """
    warnings: List[str] = []
    print("  [Validate] Running safety checks...")

    # 1. Diameter < pedicle width
    if screw_diameter >= pedicle_width:
        w = f"Screw diameter ({screw_diameter} mm) >= pedicle width ({pedicle_width} mm)"
        warnings.append(w)
        print(f"  [Validate] FAIL: {w}")
    else:
        print(
            f"  [Validate] PASS: diameter {screw_diameter} < pedicle width {pedicle_width}"
        )

    # 2. Diameter < pedicle height
    if screw_diameter >= pedicle_height:
        w = f"Screw diameter ({screw_diameter} mm) >= pedicle height ({pedicle_height} mm)"
        warnings.append(w)
        print(f"  [Validate] FAIL: {w}")
    else:
        print(
            f"  [Validate] PASS: diameter {screw_diameter} < pedicle height {pedicle_height}"
        )

    # 3. Length within vertebral body
    if screw_length > vertebral_body_depth:
        w = f"Screw length ({screw_length} mm) > vertebral body depth ({vertebral_body_depth} mm)"
        warnings.append(w)
        print(f"  [Validate] FAIL: {w}")
    else:
        print(
            f"  [Validate] PASS: length {screw_length} <= body depth {vertebral_body_depth}"
        )

    # 4. TPA within anatomical range
    if not (L5_TPA_RANGE[0] <= transverse_pedicle_angle <= L5_TPA_RANGE[1]):
        w = (
            f"Transverse pedicle angle ({transverse_pedicle_angle}°) outside "
            f"L5 range {L5_TPA_RANGE}"
        )
        warnings.append(w)
        print(f"  [Validate] WARN: {w}")
    else:
        print(
            f"  [Validate] PASS: TPA {transverse_pedicle_angle}° within {L5_TPA_RANGE}"
        )

    # 5. SPA within anatomical range
    if not (L5_SPA_RANGE[0] <= sagittal_pedicle_angle <= L5_SPA_RANGE[1]):
        w = (
            f"Sagittal pedicle angle ({sagittal_pedicle_angle}°) outside "
            f"L5 range {L5_SPA_RANGE}"
        )
        warnings.append(w)
        print(f"  [Validate] WARN: {w}")
    else:
        print(f"  [Validate] PASS: SPA {sagittal_pedicle_angle}° within {L5_SPA_RANGE}")

    if not warnings:
        print("  [Validate] All safety checks PASSED")
    else:
        print(f"  [Validate] {len(warnings)} warning(s) raised")

    return warnings


# ---------------------------------------------------------------------------
#  Tip-point computation
# ---------------------------------------------------------------------------


def compute_tip_point(
    entry_point: np.ndarray,
    trajectory_vector: np.ndarray,
    screw_length: float,
) -> np.ndarray:
    """
    Compute the screw tip location.

    tip = entry + trajectory × length

    Parameters
    ----------
    entry_point : array-like, shape (3,)
    trajectory_vector : array-like, shape (3,)   Must be a unit vector.
    screw_length : float (mm)

    Returns
    -------
    np.ndarray, shape (3,)
    """
    entry = np.asarray(entry_point, dtype=float)
    direction = np.asarray(trajectory_vector, dtype=float)
    tip = entry + direction * screw_length
    print(
        f"  [Tip] entry={np.round(entry, 2)}  +  dir×{screw_length} mm  =  tip={np.round(tip, 2)}"
    )
    return tip


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------


def plan_l5_pedicle_screw(
    entry_point: np.ndarray,
    pedicle_width: float,
    pedicle_height: float,
    vertebral_body_depth: float,
    transverse_pedicle_angle: float,
    sagittal_pedicle_angle: float,
    safety_margin: float = 2.0,
    side: str = "left",
    anatomical_axes: Optional[np.ndarray] = None,
    pedicle_center: Optional[np.ndarray] = None,
    anterior_target: Optional[np.ndarray] = None,
    forced_trajectory_world: Optional[np.ndarray] = None,
) -> Dict:
    """
    Plan a single L5 pedicle screw trajectory.

    Parameters
    ----------
    entry_point : array-like, shape (3,)
        (x, y, z) coordinates on the posterior L5 surface.
    pedicle_width : float
        Mediolateral diameter of the pedicle (mm).
    pedicle_height : float
        Superior-inferior diameter of the pedicle (mm).
    vertebral_body_depth : float
        Posterior cortex to anterior cortex (mm).
    transverse_pedicle_angle : float
        Medial angulation in axial plane (degrees).
    sagittal_pedicle_angle : float
        Sup/inf inclination in sagittal plane (degrees).
    safety_margin : float
        Minimum clearance from cortical walls (mm).
    side : str
        ``"left"`` or ``"right"``.
    anatomical_axes : np.ndarray, shape (3, 3), optional
        Rows are [siAxis, lrAxis, apAxis] in world coordinates.
    pedicle_center : array-like, shape (3,), optional
        Pedicle center in world coordinates (required if anatomical_axes is set).
    anterior_target : array-like, shape (3,), optional
        A point strictly anterior to the pedicles (required if anatomical_axes is set).
        When provided, the trajectory is computed directly in world coordinates,
        guaranteeing correct anterior/medial projection.
        When None, the trajectory stays in a local abstract frame (suitable for
        standalone/unit-test usage).
    forced_trajectory_world : array-like, shape (3,), optional
        If provided, completely bypasses angle computation and forces this exact trajectory vector.

    Returns
    -------
    dict with keys:
        entry_point, trajectory_vector, recommended_screw_diameter,
        recommended_screw_length, tip_point, warnings, side
    """
    entry = np.asarray(entry_point, dtype=float)

    print("=" * 65)
    print(f"  L5 PEDICLE SCREW PLAN — {side.upper()} SIDE")
    print("=" * 65)
    print(f"  Input Parameters:")
    print(f"    Entry point            = {np.round(entry, 2)}")
    print(f"    Pedicle width          = {pedicle_width} mm")
    print(f"    Pedicle height         = {pedicle_height} mm")
    print(f"    Vertebral body depth   = {vertebral_body_depth} mm")
    print(f"    Transverse angle (TPA) = {transverse_pedicle_angle}°")
    print(f"    Sagittal angle  (SPA)  = {sagittal_pedicle_angle}°")
    print(f"    Safety margin          = {safety_margin} mm")
    print()

    # 1. Trajectory direction (in local anatomical frame)
    print("  — Step 1: Compute trajectory direction —")
    trajectory_local = compute_trajectory(
        transverse_pedicle_angle, sagittal_pedicle_angle, side
    )

    # 1b. Compute robust world coordinates if axes are provided
    if forced_trajectory_world is not None:
        print("  — Step 1b: Using forced world trajectory (bypassing synthetic angles) —")
        trajectory_world = np.asarray(forced_trajectory_world, dtype=float)
        # normalize just in case
        trajectory_world = trajectory_world / np.linalg.norm(trajectory_world)
        print(f"  [WorldTransform] Forced direction = {np.round(trajectory_world, 6)}")
        print()
    elif (
        anatomical_axes is not None
        and pedicle_center is not None
        and anterior_target is not None
    ):
        print("  — Step 1b: Calculate world trajectory dynamically —")
        trajectory_world = compute_world_trajectory(
            pedicle_center,
            anterior_target,
            anatomical_axes,
            transverse_pedicle_angle,
            sagittal_pedicle_angle,
            side,
        )
        print()
    else:
        print(
            "  [WorldTransform] No anatomical information provided — using local frame directly"
        )
        trajectory_world = trajectory_local
        print()

    # 2. Screw diameter
    print("  — Step 2: Select screw diameter —")
    diameter, diam_warnings = compute_screw_diameter(
        pedicle_width, pedicle_height, safety_margin
    )
    print()

    # 3. Screw length
    print("  — Step 3: Compute screw length —")
    length, len_warnings = compute_screw_length(vertebral_body_depth, safety_margin)
    print()

    # 4. Tip point (uses world-space direction for correct placement)
    print("  — Step 4: Compute tip point —")
    tip = compute_tip_point(entry, trajectory_world, length)
    print()

    # 5. Safety validation
    print("  — Step 5: Safety validation —")
    val_warnings = validate_plan(
        diameter,
        length,
        pedicle_width,
        pedicle_height,
        vertebral_body_depth,
        transverse_pedicle_angle,
        sagittal_pedicle_angle,
    )

    all_warnings = diam_warnings + len_warnings + val_warnings

    print()
    print("  — Result Summary —")
    print(f"    Trajectory (local)     = {np.round(trajectory_local, 6)}")
    print(f"    Trajectory (world)     = {np.round(trajectory_world, 6)}")
    print(f"    Screw diameter         = {diameter} mm")
    print(f"    Screw length           = {length} mm")
    print(f"    Entry point            = {np.round(entry, 2)}")
    print(f"    Tip point              = {np.round(tip, 2)}")
    if all_warnings:
        print(f"    ⚠ Warnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"      • {w}")
    else:
        print("    ✓ No warnings — plan is safe")
    print("=" * 65)
    print()

    return {
        "entry_point": entry,
        "trajectory_vector": trajectory_world,
        "trajectory_vector_local": trajectory_local,
        "recommended_screw_diameter": diameter,
        "recommended_screw_length": length,
        "tip_point": tip,
        "warnings": all_warnings,
        "side": side,
    }


# ---------------------------------------------------------------------------
#  Standalone demo with published L5 averages
# ---------------------------------------------------------------------------


def run_planner_standalone() -> List[Dict]:
    """
    Run the planner for both pedicles using published average L5 anatomy.

    Literature references for default values:
        - Pedicle width  ~18 mm  (Zindrick et al., Spine 1987)
        - Pedicle height ~15 mm
        - Body depth     ~35 mm
        - TPA            ~25°    (Weinstein et al., Spine 1992)
        - SPA            ~ 0°
    """
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  L5 DETERMINISTIC TRAJECTORY PLANNER — STANDALONE DEMO     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Default anatomical measurements (published L5 averages)
    pedicle_width = 18.0  # mm
    pedicle_height = 15.0  # mm
    vertebral_body_depth = 35.0  # mm
    transverse_pedicle_angle = 25.0  # degrees
    sagittal_pedicle_angle = 0.0  # degrees
    safety_margin = 2.0  # mm

    # Symmetric entry points on the posterior L5 surface
    entry_left = np.array([-14.0, 0.0, 0.0])  # left pedicle
    entry_right = np.array([14.0, 0.0, 0.0])  # right pedicle

    results = []

    for side, entry in [("left", entry_left), ("right", entry_right)]:
        result = plan_l5_pedicle_screw(
            entry_point=entry,
            pedicle_width=pedicle_width,
            pedicle_height=pedicle_height,
            vertebral_body_depth=vertebral_body_depth,
            transverse_pedicle_angle=transverse_pedicle_angle,
            sagittal_pedicle_angle=sagittal_pedicle_angle,
            safety_margin=safety_margin,
            side=side,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
#  CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_planner_standalone()
    print("\nFinal output (both pedicles):")
    for r in results:
        print(
            {
                "side": r["side"],
                "entry_point": tuple(np.round(r["entry_point"], 2)),
                "trajectory_vector": tuple(np.round(r["trajectory_vector"], 6)),
                "recommended_screw_diameter": r["recommended_screw_diameter"],
                "recommended_screw_length": r["recommended_screw_length"],
                "tip_point": tuple(np.round(r["tip_point"], 2)),
                "warnings": r["warnings"],
            }
        )
