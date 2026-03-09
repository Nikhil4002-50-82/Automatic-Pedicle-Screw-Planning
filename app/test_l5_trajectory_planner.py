"""
Tests for l5_trajectory_planner.py
===================================

Run with:
    cd app/
    python -m pytest test_l5_trajectory_planner.py -v

Or without pytest:
    python test_l5_trajectory_planner.py
"""

import numpy as np
import sys
import os

# Ensure the app directory is on the import path
sys.path.insert(0, os.path.dirname(__file__))

from l5_trajectory_planner import (
    _rotation_matrix_z,
    _rotation_matrix_x,
    compute_trajectory,
    compute_screw_diameter,
    compute_screw_length,
    validate_plan,
    compute_tip_point,
    plan_l5_pedicle_screw,
    STANDARD_DIAMETERS,
    STANDARD_LENGTHS,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def assert_close(a, b, tol=1e-6, msg=""):
    """Assert two arrays or scalars are close within tolerance."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = np.max(np.abs(a - b))
    assert diff < tol, f"{msg}  max diff={diff}  a={a}  b={b}"


def assert_unit(v, tol=1e-9):
    """Assert vector is unit length."""
    norm = np.linalg.norm(v)
    assert abs(norm - 1.0) < tol, f"Expected unit vector, got norm={norm}"


# ── Rotation matrix tests ───────────────────────────────────────────────────


def test_rotation_z_identity():
    """Zero rotation should be identity."""
    R = _rotation_matrix_z(0)
    assert_close(R, np.eye(3), msg="Rz(0)")


def test_rotation_z_90():
    """90° about Z should rotate X→Y."""
    R = _rotation_matrix_z(90)
    v = R @ np.array([1, 0, 0])
    assert_close(v, [0, 1, 0], tol=1e-9, msg="Rz(90) @ [1,0,0]")


def test_rotation_x_identity():
    """Zero rotation should be identity."""
    R = _rotation_matrix_x(0)
    assert_close(R, np.eye(3), msg="Rx(0)")


def test_rotation_x_90():
    """90° about X should rotate Y→Z."""
    R = _rotation_matrix_x(90)
    v = R @ np.array([0, 1, 0])
    assert_close(v, [0, 0, 1], tol=1e-9, msg="Rx(90) @ [0,1,0]")


# ── Trajectory tests ────────────────────────────────────────────────────────


def test_trajectory_zero_angles():
    """Zero TPA and SPA → should return baseline v0 = (0, 1, 0)."""
    traj = compute_trajectory(0, 0, side="left")
    assert_unit(traj)
    assert_close(traj, [0, 1, 0], tol=1e-9, msg="zero angles")


def test_trajectory_left_right_symmetry():
    """Left and right trajectories should be mirror images about X=0."""
    tl = compute_trajectory(25, 0, side="left")
    tr = compute_trajectory(25, 0, side="right")
    assert_unit(tl)
    assert_unit(tr)
    # X should be opposite sign
    assert_close(tl[0], -tr[0], tol=1e-9, msg="mirror X")
    # Y should be same
    assert_close(tl[1], tr[1], tol=1e-9, msg="same Y")
    # Z should be same
    assert_close(tl[2], tr[2], tol=1e-9, msg="same Z")


def test_trajectory_25deg_left():
    """25° TPA, 0° SPA, left side → known analytical result."""
    tpa = 25.0
    traj = compute_trajectory(tpa, 0, side="left")
    assert_unit(traj)
    # Expected: Rz(+25) @ (0,1,0) = (-sin25, cos25, 0)
    expected = np.array([np.sin(np.deg2rad(25)), np.cos(np.deg2rad(25)), 0])
    # Note: Rz(+25) @ [0,1,0] = [-sin(25), cos(25), 0]
    # But our convention: left pedicle uses +TPA, Rz(+25)@(0,1,0) = (-sin25, cos25, 0)
    # However, our Rz: [c,-s; s,c], so Rz(25)@(0,1,0) = (-s, c, 0) = (-sin25, cos25, 0)
    expected_correct = np.array([-np.sin(np.deg2rad(25)), np.cos(np.deg2rad(25)), 0])
    assert_close(traj, expected_correct, tol=1e-9, msg="25° TPA left")


def test_trajectory_with_sagittal():
    """Combined TPA and SPA should still produce a unit vector."""
    traj = compute_trajectory(25, 5, side="left")
    assert_unit(traj)
    # Z component should be non-zero (sagittal tilt)
    assert abs(traj[2]) > 0.01, f"Expected nonzero Z, got {traj[2]}"


# ── Diameter tests ───────────────────────────────────────────────────────────


def test_diameter_normal():
    """Normal L5 pedicle (18mm wide, 15mm high, 2mm margin)."""
    diam, warns = compute_screw_diameter(18.0, 15.0, 2.0)
    # max = min(18,15) - 4 = 11 → largest standard ≤ 11 is 8.5
    assert diam == 8.5
    assert len(warns) == 0


def test_diameter_narrow_pedicle():
    """Narrow pedicle where max safe < smallest standard."""
    diam, warns = compute_screw_diameter(6.0, 5.0, 1.5)
    # max = min(6,5) - 3 = 2 → below MIN_SCREW_DIAMETER (4.0)
    assert diam < 4.0
    assert len(warns) > 0


def test_diameter_exact_fit():
    """Pedicle that exactly fits 7.0 mm screw."""
    diam, warns = compute_screw_diameter(11.0, 12.0, 2.0)
    # max = min(11,12) - 4 = 7.0  → standard 7.0 fits exactly
    assert diam == 7.0
    assert len(warns) == 0


# ── Length tests ─────────────────────────────────────────────────────────────


def test_length_normal():
    """Normal vertebral body (35mm deep, 2mm margin)."""
    length, warns = compute_screw_length(35.0, 2.0)
    # raw = 33 → nearest standard ≤ 33 is 30
    assert length == 30
    assert len(warns) == 0


def test_length_deep_body():
    """Very deep body (60mm) → clamped to max standard."""
    length, warns = compute_screw_length(60.0, 2.0)
    # raw = 58 → max standard = 55
    assert length == 55
    assert len(warns) > 0  # Should warn about clamping


def test_length_shallow_body():
    """Very shallow body (20mm) → raw = 18 < min standard 25."""
    length, warns = compute_screw_length(20.0, 2.0)
    # raw = 18 → no standard ≤ 18, use minimum 25
    assert length == 25
    assert len(warns) > 0


# ── Validation tests ────────────────────────────────────────────────────────


def test_validate_all_pass():
    """All parameters within safe limits."""
    warns = validate_plan(
        screw_diameter=6.5,
        screw_length=30,
        pedicle_width=18.0,
        pedicle_height=15.0,
        vertebral_body_depth=35.0,
        transverse_pedicle_angle=25.0,
        sagittal_pedicle_angle=0.0,
    )
    assert len(warns) == 0


def test_validate_diameter_too_wide():
    """Screw wider than pedicle."""
    warns = validate_plan(
        screw_diameter=20.0,
        screw_length=30,
        pedicle_width=18.0,
        pedicle_height=15.0,
        vertebral_body_depth=35.0,
        transverse_pedicle_angle=25.0,
        sagittal_pedicle_angle=0.0,
    )
    assert any("pedicle width" in w for w in warns)


def test_validate_angle_out_of_range():
    """TPA outside anatomical bounds."""
    warns = validate_plan(
        screw_diameter=6.5,
        screw_length=30,
        pedicle_width=18.0,
        pedicle_height=15.0,
        vertebral_body_depth=35.0,
        transverse_pedicle_angle=50.0,  # Too large
        sagittal_pedicle_angle=0.0,
    )
    assert any("Transverse" in w for w in warns)


# ── Tip point test ──────────────────────────────────────────────────────────


def test_tip_point():
    """Tip = entry + direction × length."""
    entry = np.array([10.0, 0.0, 5.0])
    direction = np.array([0.0, 1.0, 0.0])
    tip = compute_tip_point(entry, direction, 30.0)
    assert_close(tip, [10.0, 30.0, 5.0], msg="tip point")


# ── Full plan test ──────────────────────────────────────────────────────────


def test_full_plan_realistic_l5():
    """Full plan with realistic L5 parameters — should produce valid output."""
    result = plan_l5_pedicle_screw(
        entry_point=np.array([-14.0, 0.0, 0.0]),
        pedicle_width=18.0,
        pedicle_height=15.0,
        vertebral_body_depth=35.0,
        transverse_pedicle_angle=25.0,
        sagittal_pedicle_angle=0.0,
        safety_margin=2.0,
        side="left",
    )

    expected_keys = {
        "entry_point",
        "trajectory_vector",
        "trajectory_vector_local",
        "recommended_screw_diameter",
        "recommended_screw_length",
        "tip_point",
        "warnings",
        "side",
    }
    assert (
        set(result.keys()) == expected_keys
    ), f"Keys mismatch: expected {expected_keys}, got {set(result.keys())}"

    # Trajectory should be a unit vector
    assert_unit(result["trajectory_vector"])

    # Diameter should be a positive standard size
    assert result["recommended_screw_diameter"] > 0
    assert result["recommended_screw_diameter"] in STANDARD_DIAMETERS

    # Length should be a positive standard size
    assert result["recommended_screw_length"] > 0
    assert result["recommended_screw_length"] in STANDARD_LENGTHS

    # No warnings for this healthy anatomy
    assert len(result["warnings"]) == 0, f"Unexpected warnings: {result['warnings']}"

    # Side preserved
    assert result["side"] == "left"

    print("\n✓ Full plan test PASSED with realistic L5 values")


def test_full_plan_visualizer_compatible():
    """Verify the output can be converted to visualizer format trivially."""
    result = plan_l5_pedicle_screw(
        entry_point=np.array([14.0, 0.0, 0.0]),
        pedicle_width=18.0,
        pedicle_height=15.0,
        vertebral_body_depth=35.0,
        transverse_pedicle_angle=25.0,
        sagittal_pedicle_angle=0.0,
        safety_margin=2.0,
        side="right",
    )

    # Build visualizer dict
    viz = {
        "vertebra": "L5",
        "side": result["side"].capitalize(),
        "entry": result["entry_point"],
        "tip": result["tip_point"],
        "diameter": result["recommended_screw_diameter"],
    }

    # Verify types that the visualizer expects
    assert isinstance(viz["entry"], np.ndarray)
    assert isinstance(viz["tip"], np.ndarray)
    assert isinstance(viz["diameter"], (int, float))
    assert len(viz["entry"]) == 3
    assert len(viz["tip"]) == 3

    print("\n✓ Visualizer compatibility test PASSED")


# ── Runner for non-pytest ────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_rotation_z_identity,
        test_rotation_z_90,
        test_rotation_x_identity,
        test_rotation_x_90,
        test_trajectory_zero_angles,
        test_trajectory_left_right_symmetry,
        test_trajectory_25deg_left,
        test_trajectory_with_sagittal,
        test_diameter_normal,
        test_diameter_narrow_pedicle,
        test_diameter_exact_fit,
        test_length_normal,
        test_length_deep_body,
        test_length_shallow_body,
        test_validate_all_pass,
        test_validate_diameter_too_wide,
        test_validate_angle_out_of_range,
        test_tip_point,
        test_full_plan_realistic_l5,
        test_full_plan_visualizer_compatible,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS: {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {t.__name__} — {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {t.__name__} — {e}")

    print(f"\n{'=' * 50}")
    print(f"  {passed} passed,  {failed} failed,  {len(tests)} total")
    print(f"{'=' * 50}")
    sys.exit(1 if failed else 0)
