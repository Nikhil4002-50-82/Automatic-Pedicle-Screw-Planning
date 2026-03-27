# Visualizer Unified Module - Bug Analysis

## Overview
The `visualizer_unified.py` module attempts to combine the best features of `visualizer.py` and `visualizerV2.py`. After thorough analysis, several bugs and issues have been identified that need to be fixed.

## Identified Bugs

### 1. **CRITICAL: Return Value Inconsistency in `visualize_surgical_plan`**
- **Location**: Line 800
- **Issue**: The function returns `fig, show_visualization` where `show_visualization` is a function object, not the result of calling it.
- **Expected Behavior**: Should return `fig, show_visualization(fig, renderer=...)` or similar
- **Impact**: The adapter functions in `plan_and_visualize_geometry_unified.py` (line 121) and `plan_and_visualize_l5_unified.py` (line 125) expect to call `show_figure(fig, renderer=args.renderer)`, but they receive a function object instead of a callable result.
- **Current Code**:
  ```python
  return fig, show_visualization  # Returns function object
  ```
- **Fix**: Should be:
  ```python
  return fig, lambda fig_obj=None: show_visualization(fig if fig_obj is None else fig_obj, renderer=...)
  ```

### 2. **CRITICAL: Missing Parameters in `visualize_surgical_plan` Signature**
- **Location**: Lines 759-778
- **Issue**: The function signature is missing `neon_trajectories`, `gold_screws`, and `threaded_screws` parameters that are passed to `build_visualization` (lines 794-797).
- **Impact**: Calling `visualize_surgical_plan` with these parameters will raise a `TypeError`.
- **Current Signature**:
  ```python
  def visualize_surgical_plan(
      vertsWorld,
      faces,
      resultsList,
      volume_path=None,
      screw_mode="threaded",
      theme="dark",
      mesh_opacity=None,
      visual_preset="cinematic",
      show_safety_planes=False,
      show_bounding_box=True,
      show_trajectory_lines=True,
      show_entry_markers=True,
      show_tip_markers=False,
      v2_neon_trajectories=None,
      v2_gold_screws=None,
      v2_threaded_screws=None,
      v2_safety_planes=None,
      fallback_diameter=None,
  ):
  ```
- **Fix**: Add missing parameters to the signature.

### 3. **BUG: Trajectory Visibility Logic is Inverted**
- **Location**: Line 455
- **Issue**: The trajectory line visibility is set to `visible=show_trajectory_lines and eff_screw_mode == "none"`.
- **Expected Behavior**: Trajectory lines should be visible when screws are NOT rendered (i.e., when `eff_screw_mode == "none"`), OR when explicitly enabled.
- **Current Logic**: Trajectory lines are only visible when `show_trajectory_lines=True` AND `eff_screw_mode == "none"`.
- **Problem**: This means trajectory lines are hidden when screws are rendered, which is the opposite of what users expect.
- **Fix**: Change to `visible=show_trajectory_lines` or `visible=show_trajectory_lines and eff_screw_mode != "none"`.

### 4. **BUG: Missing `name` Parameter in Safety Plane Trace**
- **Location**: Line 517
- **Issue**: The safety plane trace doesn't have a `name` parameter.
- **Impact**: This could cause issues with the legend and hover behavior.
- **Current Code**:
  ```python
  safety_plane_traces.append(
      go.Surface(
          x=plane_x,
          y=plane_y,
          z=plane_z,
          opacity=0.3,
          showscale=False,
          surfacecolor=np.ones_like(plane_x),
          colorscale=[[0.0, style["safety_plane"]], [1.0, style["safety_plane"]]],
          cmin=0,
          cmax=1,
          hovertemplate="80% Safety Limit<extra></extra>",
      )
  )
  ```
- **Fix**: Add `name="80% Safety Limit"` parameter.

### 5. **BUG: Missing `showlegend` for Safety Plane Traces**
- **Location**: Line 517
- **Issue**: Safety plane traces don't have `showlegend` set.
- **Impact**: This could cause inconsistent legend behavior.
- **Fix**: Add `showlegend=False` parameter.

### 6. **BUG: `_is_trace_visible` Function Doesn't Handle Non-Boolean Values**
- **Location**: Lines 149-152
- **Issue**: The function checks `trace.visible` but doesn't handle the case where `trace.visible` might be a list or other non-boolean value.
- **Current Code**:
  ```python
  def _is_trace_visible(trace):
      if trace.visible is None:
          return True
      return bool(trace.visible)
  ```
- **Problem**: Plotly traces can have `visible` set to `"legendonly"` or other string values.
- **Fix**: Handle string values like `"legendonly"` properly.

### 7. **Minor Issue: Inconsistent Parameter Naming**
- **Location**: Lines 759-778
- **Issue**: The function uses `vertsWorld` (camelCase) while `build_visualization` uses `verts_world` (snake_case).
- **Impact**: Not a bug, but inconsistent with Python naming conventions.
- **Recommendation**: Consider standardizing to snake_case for consistency.

## Summary of Critical Issues
1. **Return value inconsistency** - Will cause runtime errors when adapters try to call the returned function
2. **Missing parameters** - Will cause TypeError when calling with v2_* parameters
3. **Inverted trajectory visibility** - Confusing user experience

## Recommended Fix Priority
1. Fix return value inconsistency (Critical)
2. Add missing parameters to signature (Critical)
3. Fix trajectory visibility logic (High)
4. Add name and showlegend to safety plane traces (Medium)
5. Improve _is_trace_visible function (Low)


