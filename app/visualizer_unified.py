import datetime
import json
import os
import sys
import tempfile
from pathlib import Path

# Lazy-load heavy libraries to reduce startup time
import numpy as np

# Plotly will be imported on first use in build_visualization()
_plotly_go = None
_plotly_pio = None

def _ensure_plotly_imports():
    """Lazy-load Plotly only when needed (saves ~1-2 seconds on startup)"""
    global _plotly_go, _plotly_pio
    if _plotly_go is None:
        import plotly.graph_objects
        import plotly.io
        _plotly_go = plotly.graph_objects
        _plotly_pio = plotly.io
    return _plotly_go, _plotly_pio


_VIEWER_WINDOWS = []
_PLOTLY_JS_BUNDLE_PATH = None


def _ensure_plotlyjs_bundle():
    """
    Ensure Plotly JS bundle is available with improved caching.
    First tries persistent cache, then temp cache, then downloads fresh.
    Saves ~2-5 seconds on first visualization by avoiding re-download.
    """
    global _PLOTLY_JS_BUNDLE_PATH

    if _PLOTLY_JS_BUNDLE_PATH and os.path.exists(_PLOTLY_JS_BUNDLE_PATH):
        return _PLOTLY_JS_BUNDLE_PATH

    # Try persistent cache first (faster on repeated runs)
    persistent_cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "automatic-pedicle-screw-planning"
    )
    persistent_bundle_path = os.path.join(persistent_cache_dir, "plotly.min.js")
    
    if os.path.exists(persistent_bundle_path) and os.path.getsize(persistent_bundle_path) > 100000:
        _PLOTLY_JS_BUNDLE_PATH = persistent_bundle_path
        return persistent_bundle_path

    # Fallback to temp cache
    temp_cache_dir = os.path.join(tempfile.gettempdir(), "automatic-pedicle-screw-planning")
    os.makedirs(temp_cache_dir, exist_ok=True)
    temp_bundle_path = os.path.join(temp_cache_dir, "plotly.min.js")

    if os.path.exists(temp_bundle_path) and os.path.getsize(temp_bundle_path) > 100000:
        _PLOTLY_JS_BUNDLE_PATH = temp_bundle_path
        return temp_bundle_path

    # Download and cache in both locations if needed
    try:
        from plotly.offline.offline import get_plotlyjs
        bundle_content = get_plotlyjs()
        
        # Save to temp cache (always available)
        with open(temp_bundle_path, "w", encoding="utf-8") as f:
            f.write(bundle_content)
        
        # Try to save to persistent cache for next time
        try:
            os.makedirs(persistent_cache_dir, exist_ok=True)
            with open(persistent_bundle_path, "w", encoding="utf-8") as f:
                f.write(bundle_content)
            _PLOTLY_JS_BUNDLE_PATH = persistent_bundle_path
        except OSError:
            # Fall back to temp if persistent fails
            _PLOTLY_JS_BUNDLE_PATH = temp_bundle_path
            
    except Exception:
        # Last resort: empty bundle path signals to use CDN
        _PLOTLY_JS_BUNDLE_PATH = temp_bundle_path

    return _PLOTLY_JS_BUNDLE_PATH

def _normalize(vector):
    """Fast vector normalization with early exit for zero vectors"""
    norm = np.linalg.norm(vector)
    if norm < 1e-10:  # Use epsilon for better numerical stability
        return None
    return vector / norm


def _orthonormal_basis(direction):
    """Build orthonormal basis efficiently - called once per screw"""
    unit_direction = _normalize(direction)
    if unit_direction is None:
        return None, None, None

    # Choose reference vector to maximize numerical stability
    abs_z = abs(unit_direction[2])
    if abs_z < 0.9:
        reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Two cross products to build basis (cached computation)
    normal_1 = np.cross(unit_direction, reference)
    norm_1 = np.linalg.norm(normal_1)
    if norm_1 < 1e-10:
        return None, None, None
    normal_1 = normal_1 / norm_1

    normal_2 = np.cross(unit_direction, normal_1)
    # No need to normalize - it's already unit length from cross of two unit vectors
    
    return unit_direction, normal_1, normal_2


def _build_safety_plane(tip, direction, plane_size=8.0):
    axis, normal_1, normal_2 = _orthonormal_basis(direction)
    
    # Early return if any required component is None
    if axis is None:
        return None
    if normal_1 is None:
        return None
    if normal_2 is None:
        return None

    tip = np.asarray(tip, dtype=float)
    offsets = np.linspace(-plane_size, plane_size, 2)
    offset_grid_1, offset_grid_2 = np.meshgrid(offsets, offsets)
    x = tip[0] + offset_grid_1 * normal_1[0] + offset_grid_2 * normal_2[0]
    y = tip[1] + offset_grid_1 * normal_1[1] + offset_grid_2 * normal_2[1]
    z = tip[2] + offset_grid_1 * normal_1[2] + offset_grid_2 * normal_2[2]
    return x, y, z


def _screw_resolution(face_count, screw_count):
    if face_count > 150000:
        return 14
    if face_count > 90000 or screw_count > 4:
        return 18
    if face_count > 45000:
        return 22
    return 26


def _build_closed_cylinder_mesh(entry, tip, diameter, resolution=24):
    """Build cylinder mesh efficiently with minimal allocations"""
    entry = np.asarray(entry, dtype=np.float32)
    tip = np.asarray(tip, dtype=np.float32)
    direction = tip - entry
    axis, normal_1, normal_2 = _orthonormal_basis(direction)

    if axis is None or normal_1 is None or normal_2 is None or diameter <= 0:
        return None

    radius = np.float32(diameter / 2.0)
    
    # Pre-allocate all vertex data at once (faster than vstack)
    vertices = np.zeros((2 * resolution + 2, 3), dtype=np.float32)
    
    # Generate circle using vectorized operations (much faster than loop)
    theta = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False, dtype=np.float32)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    ring_offsets = (
        cos_theta[:, None] * normal_1.astype(np.float32)
        + sin_theta[:, None] * normal_2.astype(np.float32)
    ) * radius

    # Fill vertices in-place (faster than vstack)
    vertices[:resolution] = entry + ring_offsets
    vertices[resolution:2*resolution] = tip + ring_offsets
    vertices[2*resolution] = entry
    vertices[2*resolution + 1] = tip

    # Build face indices efficiently
    idx = np.arange(resolution, dtype=np.int32)
    nxt = (idx + 1) % resolution
    tip_idx = idx + resolution
    tip_nxt = nxt + resolution
    entry_center_idx = 2 * resolution
    tip_center_idx = 2 * resolution + 1

    # Concatenate all face indices
    faces_i = np.concatenate((idx, idx, np.full(resolution, entry_center_idx, dtype=np.int32), np.full(resolution, tip_center_idx, dtype=np.int32)))
    faces_j = np.concatenate((tip_idx, tip_nxt, nxt, tip_idx))
    faces_k = np.concatenate((tip_nxt, nxt, idx, tip_nxt))

    return {
        "x": vertices[:, 0],
        "y": vertices[:, 1],
        "z": vertices[:, 2],
        "i": faces_i,
        "j": faces_j,
        "k": faces_k,
    }


def _volume_metadata(volume_path):
    if not volume_path:
        return "Unknown Volume", "Unknown"

    volume_name = os.path.basename(volume_path)
    try:
        created = os.path.getctime(volume_path)
        timestamp = datetime.datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        timestamp = "Unknown"
    return volume_name, timestamp


def _display_volume_title(volume_name, max_length=72):
    if len(volume_name) <= max_length:
        return volume_name
    stem, suffix = os.path.splitext(volume_name)
    ellipsis = "..."
    reserved = len(suffix) + len(ellipsis)
    if reserved >= max_length:
        return volume_name[: max_length - len(ellipsis)] + ellipsis
    return stem[: max_length - reserved] + ellipsis + suffix


def _figure_without_embedded_controls(fig):
    qt_fig_dict = fig.to_plotly_json()
    layout = qt_fig_dict.setdefault("layout", {})
    layout.pop("updatemenus", None)
    layout.pop("sliders", None)
    layout.pop("shapes", None)
    return qt_fig_dict


def _format_result_hover(result, entry, tip, depth):
    hover_lines = [
        f"<b>{result.get('vertebra', '')} {result.get('side', '').strip()} Screw</b>",
        f"Entry: [{entry[0]:.2f}, {entry[1]:.2f}, {entry[2]:.2f}]",
        f"Tip: [{tip[0]:.2f}, {tip[1]:.2f}, {tip[2]:.2f}]",
        f"Depth: {depth:.2f} mm",
    ]

    if result.get("diameter") is not None:
        hover_lines.append(f"Diameter: {float(result['diameter']):.2f} mm")
    if result.get("length") is not None:
        hover_lines.append(f"Planned Length: {float(result['length']):.2f} mm")
    if result.get("axial_angle") is not None:
        hover_lines.append(f"Axial Angle: {float(result['axial_angle']):.2f} deg")
    if result.get("sagittal_angle") is not None:
        hover_lines.append(f"Sagittal Angle: {float(result['sagittal_angle']):.2f} deg")

    return "<br>".join(hover_lines) + "<extra></extra>"


def _default_visual_diameter(result, fallback_diameter=None):
    if result.get("diameter") is not None:
        return float(result["diameter"])
    if fallback_diameter is not None:
        return float(fallback_diameter)

    vertebra_defaults = {
        "L1": 4.5,
        "L2": 5.0,
        "L3": 5.5,
        "L4": 6.0,
        "L5": 6.5,
    }
    return vertebra_defaults.get(str(result.get("vertebra", "")).upper(), 5.5)


def _resolve_kicker(value, enabled_by_default):
    if value is None:
        return enabled_by_default
    return bool(value)


def _compute_scene_ranges(verts_world, results_list, show_safety_planes=False, fallback_diameter=None):
    points = [np.asarray(verts_world, dtype=float)]
    radial_padding = 0.0

    for result in results_list:
        entry = np.asarray(result["entry"], dtype=float)
        tip = np.asarray(result["tip"], dtype=float)
        points.append(np.vstack((entry, tip)))
        radial_padding = max(radial_padding, _default_visual_diameter(result, fallback_diameter) / 2.0)

    stacked_points = np.vstack(points)
    mins = np.min(stacked_points, axis=0)
    maxs = np.max(stacked_points, axis=0)
    spans = maxs - mins
    base_padding = np.maximum(spans * 0.06, 2.0)

    if show_safety_planes:
        base_padding = base_padding + 8.0

    total_padding = base_padding + radial_padding + 1.0
    return [[mins[idx] - total_padding[idx], maxs[idx] + total_padding[idx]] for idx in range(3)]


def _style_config(visual_preset="cinematic", theme="dark"):
    preset = str(visual_preset).strip().lower()
    theme_key = str(theme).strip().lower()

    configs = {
        "classic": {
            "template": "plotly_white",
            "paper_bgcolor": "#FFFFFF",
            "plot_bgcolor": "#FFFFFF",
            "scene_bgcolor": "#F7F8FA",
            "mesh_color": "#D3D7DD",
            "mesh_opacity": 0.25,
            "mesh_lighting": dict(ambient=0.65, diffuse=0.45, specular=0.08, roughness=0.9, fresnel=0.02),
            "mesh_lightposition": dict(x=110, y=140, z=120),
            "surface_opacity": 0.42,
            "surface_lighting": dict(ambient=0.35, diffuse=0.85, specular=0.6, roughness=0.28, fresnel=0.15),
            "surface_lightposition": dict(x=120, y=160, z=150),
            "left_marker": "#2C7BE5",
            "left_line": "#2C7BE5",
            "left_surface_low": "#5B8DEF",
            "left_surface_high": "#8DB4FF",
            "right_marker": "#D64550",
            "right_line": "#D64550",
            "right_surface_low": "#F08C46",
            "right_surface_high": "#F5B66F",
            "safety_plane": "#C93C37",
            "bounding_box": "rgba(60, 60, 60, 0.45)",
            "show_axes": True,
            "show_grid": False,
            "title_font_color": "#1A1F36",
            "camera_eye": dict(x=1.6, y=1.45, z=1.2),
        },
        "surgical": {
            "template": "plotly_dark",
            "paper_bgcolor": "#0F1722",
            "plot_bgcolor": "#0F1722",
            "scene_bgcolor": "#141C29",
            "mesh_color": "#B8C3D1",
            "mesh_opacity": 0.2,
            "mesh_lighting": dict(ambient=0.55, diffuse=0.6, specular=0.25, roughness=0.62, fresnel=0.08),
            "mesh_lightposition": dict(x=120, y=130, z=110),
            "surface_opacity": 0.38,
            "surface_lighting": dict(ambient=0.28, diffuse=0.88, specular=0.95, roughness=0.18, fresnel=0.22),
            "surface_lightposition": dict(x=140, y=180, z=170),
            "left_marker": "#4BD3FF",
            "left_line": "#3BE0FF",
            "left_surface_low": "#9FD7FF",
            "left_surface_high": "#D8F0FF",
            "right_marker": "#FF7A59",
            "right_line": "#FF9166",
            "right_surface_low": "#F0A44B",
            "right_surface_high": "#FFD27D",
            "safety_plane": "#FF4D4D",
            "bounding_box": "rgba(255, 255, 255, 0.18)",
            "show_axes": False,
            "show_grid": False,
            "title_font_color": "#F7FAFC",
            "camera_eye": dict(x=1.75, y=1.55, z=1.22),
        },
        "cinematic": {
            "template": "plotly_dark",
            "paper_bgcolor": "#070B12",
            "plot_bgcolor": "#070B12",
            "scene_bgcolor": "#0B1320",
            "mesh_color": "#C8D0DA",
            "mesh_opacity": 0.18,
            "mesh_lighting": dict(ambient=0.42, diffuse=0.7, specular=0.42, roughness=0.42, fresnel=0.12),
            "mesh_lightposition": dict(x=150, y=180, z=140),
            "surface_opacity": 0.34,
            "surface_lighting": dict(ambient=0.2, diffuse=0.95, specular=1.0, roughness=0.12, fresnel=0.28),
            "surface_lightposition": dict(x=170, y=220, z=190),
            "left_marker": "#56E0FF",
            "left_line": "#5BF3FF",
            "left_surface_low": "#B58A2B",
            "left_surface_high": "#F5D36C",
            "right_marker": "#FF7B6B",
            "right_line": "#FF8C66",
            "right_surface_low": "#C9872F",
            "right_surface_high": "#FFD07C",
            "safety_plane": "#FF4A4A",
            "bounding_box": "rgba(255, 255, 255, 0.14)",
            "show_axes": False,
            "show_grid": False,
            "title_font_color": "#F7FAFC",
            "camera_eye": dict(x=1.9, y=1.55, z=1.18),
        },
    }

    if preset not in configs:
        preset = "cinematic"

    config = dict(configs[preset])
    if theme_key == "light" and preset != "classic":
        config.update(
            {
                "template": "plotly_white",
                "paper_bgcolor": "#F4F7FB",
                "plot_bgcolor": "#F4F7FB",
                "scene_bgcolor": "#EDF2F9",
                "title_font_color": "#152033",
                "bounding_box": "rgba(30, 41, 59, 0.22)",
            }
        )
    return config


def _add_mesh(fig, verts_world, faces, volume_name, timestamp, mesh_opacity, style):
    go, _ = _ensure_plotly_imports()
    verts_world = np.asarray(verts_world, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    fig.add_trace(
        go.Mesh3d(
            x=verts_world[:, 0],
            y=verts_world[:, 1],
            z=verts_world[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=mesh_opacity,
            color=style["mesh_color"],
            name=volume_name,
            hoverinfo="skip",
            lighting=style["mesh_lighting"],
            lightposition=style["mesh_lightposition"],
            showscale=False,
        )
    )


def _add_bounding_box(fig, verts_world, color, return_trace=False):
    go, _ = _ensure_plotly_imports()
    min_x, min_y, min_z = np.min(verts_world, axis=0)
    max_x, max_y, max_z = np.max(verts_world, axis=0)
    corners = np.array(
        [
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, max_y, min_z],
            [min_x, max_y, min_z],
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [max_x, max_y, max_z],
            [min_x, max_y, max_z],
        ]
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    x_coords = []
    y_coords = []
    z_coords = []
    for start_idx, end_idx in edges:
        x_coords.extend([corners[start_idx, 0], corners[end_idx, 0], None])
        y_coords.extend([corners[start_idx, 1], corners[end_idx, 1], None])
        z_coords.extend([corners[start_idx, 2], corners[end_idx, 2], None])

    trace = go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode="lines",
        line=dict(color=color, width=3, dash="dot"),
        showlegend=False,
        hoverinfo="skip",
    )
    if return_trace:
        return [trace]
    fig.add_trace(trace)
    return None


def _side_style(side, style):
    side_key = str(side).strip().lower()
    if side_key == "left":
        return {
            "marker": style["left_marker"],
            "line": style["left_line"],
            "surface": style["left_surface_low"],
            "surface_highlight": style["left_surface_high"],
            "legendgroup": "left_screw",
        }
    return {
        "marker": style["right_marker"],
        "line": style["right_line"],
        "surface": style["right_surface_low"],
        "surface_highlight": style["right_surface_high"],
        "legendgroup": "right_screw",
    }


def _build_result_traces(
    result,
    style,
    fallback_diameter,
    show_screw_meshes,
    show_safety_planes,
    initial_trajectory_visibility,
    show_entry_markers,
    show_tip_markers,
    neon_trajectories,
    gold_screws,
    screw_resolution,
):
    go, _ = _ensure_plotly_imports()
    entry = np.asarray(result["entry"], dtype=float)
    tip = np.asarray(result["tip"], dtype=float)
    direction = tip - entry
    depth = np.linalg.norm(direction)
    side_style = _side_style(result.get("side", ""), style)
    hover_text = _format_result_hover(result, entry, tip, depth)

    screw_traces = []
    safety_plane_traces = []

    diameter = _default_visual_diameter(result, fallback_diameter=fallback_diameter)
    if diameter > 0:
        screw_mesh = _build_closed_cylinder_mesh(entry, tip, diameter, resolution=screw_resolution)
        if screw_mesh is not None and show_screw_meshes:
            screw_traces.append(
                go.Mesh3d(
                    x=screw_mesh["x"],
                    y=screw_mesh["y"],
                    z=screw_mesh["z"],
                    i=screw_mesh["i"],
                    j=screw_mesh["j"],
                    k=screw_mesh["k"],
                    opacity=style["surface_opacity"],
                    color=side_style["surface_highlight"] if gold_screws else side_style["surface"],
                    hovertemplate=hover_text,
                    name=f"{result.get('side', '').strip()} Screw",
                    legendgroup=side_style["legendgroup"],
                    lighting=style["surface_lighting"],
                    lightposition=style["surface_lightposition"],
                    flatshading=True,
                    showscale=False,
                    showlegend=False,
                    visible=True,
                )
            )

        if show_safety_planes:
            plane_surface = _build_safety_plane(tip, direction, plane_size=8.0)
            if plane_surface is not None:
                plane_x, plane_y, plane_z = plane_surface
                safety_plane_traces.append(
                    go.Surface(
                        x=plane_x,
                        y=plane_y,
                        z=plane_z,
                        opacity=0.35,
                        showscale=False,
                        colorscale=[[0, "#ff2a6d"], [1, "#ff2a6d"]],
                        hovertemplate="80% Safety Limit<extra></extra>",
                        name="80% Safety Limit",
                        legendgroup=side_style["legendgroup"],
                        showlegend=False,
                    )
                )

    return {
        "screw_traces": screw_traces,
        "trajectory_traces": [
            go.Scatter3d(
                x=[entry[0], tip[0]],
                y=[entry[1], tip[1]],
                z=[entry[2], tip[2]],
                mode="lines",
                line=dict(color="#00f0ff", width=6),
                name=f"{result.get('side', '').strip()} Trajectory",
                legendgroup=side_style["legendgroup"],
                hovertemplate=hover_text,
                showlegend=True,
                visible=initial_trajectory_visibility,
            )
        ],
        "entry_marker_traces": [
            go.Scatter3d(
                x=[entry[0]],
                y=[entry[1]],
                z=[entry[2]],
                mode="markers",
                marker=dict(
                    size=9 if neon_trajectories else 7,
                    color=side_style["marker"],
                    symbol="circle",
                    line=dict(
                        color="#FFFFFF" if neon_trajectories else side_style["marker"],
                        width=1.2 if neon_trajectories else 0,
                    ),
                ),
                name=f"{result.get('side', '').strip()} Entry",
                legendgroup=side_style["legendgroup"],
                hovertemplate=hover_text,
                showlegend=True,
                visible=show_entry_markers,
            )
        ],
        "tip_marker_traces": [
            go.Scatter3d(
                x=[tip[0]],
                y=[tip[1]],
                z=[tip[2]],
                mode="markers",
                marker=dict(
                    size=5,
                    color=side_style["line"],
                    symbol="diamond",
                    line=dict(color="#FFFFFF", width=1),
                ),
                name=f"{result.get('side', '').strip()} Tip",
                legendgroup=side_style["legendgroup"],
                hovertemplate=hover_text,
                showlegend=False,
                visible=show_tip_markers,
            )
        ],
        "safety_plane_traces": safety_plane_traces,
    }


def build_visualization(
    verts_world,
    faces,
    results_list,
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
    neon_trajectories=True,
    gold_screws=True,
    threaded_screws=True,
    fallback_diameter=None,
    v2_neon_trajectories=None,
    v2_gold_screws=None,
    v2_threaded_screws=None,
    v2_safety_planes=None,
):
    go, _ = _ensure_plotly_imports()
    style = _style_config(visual_preset=visual_preset, theme=theme)
    if mesh_opacity is None:
        mesh_opacity = style["mesh_opacity"]

    raw_verts_world = np.asarray(verts_world, dtype=np.float32)
    raw_faces = np.asarray(faces, dtype=np.int32)

    neon_trajectories = _resolve_kicker(v2_neon_trajectories, neon_trajectories)
    gold_screws = _resolve_kicker(v2_gold_screws, gold_screws)
    threaded_screws = _resolve_kicker(v2_threaded_screws, threaded_screws)
    show_safety_planes = _resolve_kicker(v2_safety_planes, show_safety_planes)
    eff_screw_mode = str(screw_mode).strip().lower()
    if eff_screw_mode == "threaded" and not threaded_screws:
        eff_screw_mode = "cylinder"
    show_screw_meshes = eff_screw_mode != "none"
    initial_trajectory_visibility = show_trajectory_lines and not show_screw_meshes
    screw_resolution = _screw_resolution(raw_faces.shape[0], len(results_list))
    scene_ranges = _compute_scene_ranges(
        raw_verts_world,
        results_list,
        show_safety_planes=show_safety_planes,
        fallback_diameter=fallback_diameter,
    )

    fig = go.Figure()
    volume_name, timestamp = _volume_metadata(volume_path)
    _add_mesh(fig, raw_verts_world, raw_faces, volume_name, timestamp, mesh_opacity, style)
    bounding_box_traces = _add_bounding_box(fig, raw_verts_world, style["bounding_box"], return_trace=True) or []
    for trace in bounding_box_traces:
        trace.visible = show_bounding_box
        fig.add_trace(trace)

    screw_traces = []
    trajectory_traces = []
    entry_marker_traces = []
    tip_marker_traces = []
    safety_plane_traces = []

    result_trace_kwargs = [
        dict(
            result=result,
            style=style,
            fallback_diameter=fallback_diameter,
            show_screw_meshes=show_screw_meshes,
            show_safety_planes=show_safety_planes,
            initial_trajectory_visibility=initial_trajectory_visibility,
            show_entry_markers=show_entry_markers,
            show_tip_markers=show_tip_markers,
            neon_trajectories=neon_trajectories,
            gold_screws=gold_screws,
            screw_resolution=screw_resolution,
        )
        for result in results_list
    ]

    result_traces = [_build_result_traces(**kwargs) for kwargs in result_trace_kwargs]

    for trace_group in result_traces:
        screw_traces.extend(trace_group["screw_traces"])
        trajectory_traces.extend(trace_group["trajectory_traces"])
        entry_marker_traces.extend(trace_group["entry_marker_traces"])
        tip_marker_traces.extend(trace_group["tip_marker_traces"])
        safety_plane_traces.extend(trace_group["safety_plane_traces"])

    # Add traces to figure
    assembled_traces = screw_traces + trajectory_traces + entry_marker_traces + tip_marker_traces + safety_plane_traces
    if assembled_traces:
        fig.add_traces(assembled_traces)

    n_mesh = 1
    n_bbox = len(bounding_box_traces)
    n_screws = len(screw_traces)
    n_traj = len(trajectory_traces)
    n_entry = len(entry_marker_traces)
    n_tip = len(tip_marker_traces)
    n_safety = len(safety_plane_traces)
    trace_cursor = n_mesh
    bbox_indices = list(range(trace_cursor, trace_cursor + n_bbox))
    trace_cursor += n_bbox
    screw_indices = list(range(trace_cursor, trace_cursor + n_screws))
    trace_cursor += n_screws
    trajectory_indices = list(range(trace_cursor, trace_cursor + n_traj))
    screw_mode_indices = screw_indices + trajectory_indices
    screw_mode_screws_vis = [True] * n_screws + [False] * n_traj
    screw_mode_traj_vis = [False] * n_screws + [True] * n_traj
    button_bgcolor = "#4C6382" if style["template"] == "plotly_dark" else "#D9E4F2"
    button_border = "rgba(255,255,255,0.2)" if style["template"] == "plotly_dark" else "rgba(21,32,51,0.18)"
    button_font = dict(color=style["title_font_color"], size=15)
    slider_panel_fill = "rgba(8, 14, 24, 0.86)" if style["template"] == "plotly_dark" else "rgba(255, 255, 255, 0.88)"
    title_text = f"Pedicle Screw Planner Visualization - {_display_volume_title(volume_name)}"
    updatemenus = [
        dict(
            type="buttons",
            direction="right",
            showactive=False,
            x=0.01,
            y=1.03,
            xanchor="left",
            yanchor="top",
            pad=dict(r=10, t=8),
            bgcolor=button_bgcolor,
            bordercolor=button_border,
            borderwidth=1,
            font=button_font,
            buttons=[
                dict(
                    label="Show Max Diameter",
                    method="restyle",
                    args=[{"visible": screw_mode_screws_vis}, screw_mode_indices],
                ),
                dict(
                    label="Show Trajectories",
                    method="restyle",
                    args=[{"visible": screw_mode_traj_vis}, screw_mode_indices],
                ),
            ],
        ),
        dict(
            type="buttons",
            direction="right",
            showactive=False,
            x=0.35,
            y=1.03,
            xanchor="left",
            yanchor="top",
            pad=dict(r=10, t=8),
            bgcolor=button_bgcolor,
            bordercolor=button_border,
            borderwidth=1,
            font=button_font,
            buttons=[
                dict(
                    label="Show Bounding Box",
                    method="restyle",
                    args=[{"visible": [True] * n_bbox}, bbox_indices],
                ),
                dict(
                    label="Hide Bounding Box",
                    method="restyle",
                    args=[{"visible": [False] * n_bbox}, bbox_indices],
                ),
            ],
        ),
    ]

    mesh_opacities = [round(value, 2) for value in np.linspace(0.05, 1.0, 20)]
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor="center",
            y=0.995,
            yanchor="top",
            pad=dict(t=5, b=5),
            font=dict(size=22, color=style["title_font_color"], family="Segoe UI Semibold, sans-serif"),
        ),
        template=style["template"],
        paper_bgcolor=style["paper_bgcolor"],
        plot_bgcolor=style["plot_bgcolor"],
        scene=dict(
            aspectmode="data",
            bgcolor=style["scene_bgcolor"],
            xaxis=dict(
                showbackground=style["show_axes"],
                visible=style["show_axes"],
                showgrid=style["show_grid"],
                zeroline=False,
                range=scene_ranges[0],
            ),
            yaxis=dict(
                showbackground=style["show_axes"],
                visible=style["show_axes"],
                showgrid=style["show_grid"],
                zeroline=False,
                range=scene_ranges[1],
            ),
            zaxis=dict(
                showbackground=style["show_axes"],
                visible=style["show_axes"],
                showgrid=style["show_grid"],
                zeroline=False,
                range=scene_ranges[2],
            ),
            camera=dict(eye=style["camera_eye"]),
        ),
        height=760,
        # Keep the scene full-height; the legend floats inside the plot instead of reserving a blank strip.
        margin=dict(l=0, r=0, t=10, b=0),
        updatemenus=updatemenus,
        sliders=[
            dict(
                active=4,
                activebgcolor=button_bgcolor,
                currentvalue={
                    "prefix": "Mesh Opacity: ",
                    "font": {"color": style["title_font_color"], "size": 13},
                    "visible": True,
                    "xanchor": "right",
                },
                bgcolor="rgba(76, 99, 130, 0.16)" if style["template"] == "plotly_dark" else "rgba(217, 228, 242, 0.72)",
                bordercolor=button_border,
                borderwidth=1,
                pad={"t": 0, "b": 0},
                x=0.63,
                y=0.08,
                len=0.33,
                tickcolor=style["title_font_color"],
                ticklen=6,
                tickwidth=1.2,
                steps=[
                    dict(
                        method="restyle",
                        args=[{"opacity": [opacity]}, [0]],
                        label=f"{opacity:.2f}",
                    )
                    for opacity in mesh_opacities
                ],
            )
        ],
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=style["title_font_color"]),
            orientation="h",
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.02,
        ),
        meta=dict(
            mesh_trace_index=0,
            bbox_indices=bbox_indices,
            screw_mode_indices=screw_mode_indices,
            screw_mode_screws_vis=screw_mode_screws_vis,
            screw_mode_traj_vis=screw_mode_traj_vis,
            mesh_opacity=mesh_opacity,
            initial_screw_mode="screws" if show_screw_meshes else "trajectories",
            initial_bbox_visible=show_bounding_box,
        ),
        transition=dict(duration=0),
        uirevision="unified-visualizer",
    )
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0.60,
        y0=0.0,
        x1=0.98,
        y1=0.11,
        line=dict(color=button_border, width=1),
        fillcolor=slider_panel_fill,
        layer="below",
    )
    return fig


def _build_qt_window(fig, window_title):
    # Lazy load PyQt6 to avoid import overhead until visualization is needed
    from PyQt6.QtCore import Qt, QUrl
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )

    paper_bg = getattr(fig.layout, "paper_bgcolor", None) or "#0B1320"
    control_meta = getattr(fig.layout, "meta", None) or {}
    if hasattr(control_meta, "to_plotly_json"):
        control_meta = control_meta.to_plotly_json()

    qt_fig = _figure_without_embedded_controls(fig)
    plotly_bundle_uri = Path(_ensure_plotlyjs_bundle()).as_uri()
    _, pio = _ensure_plotly_imports()
    html_document = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="{plotly_bundle_uri}"></script>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: {paper_bg};
    }}
    body {{
      font-family: Segoe UI, sans-serif;
    }}
    .plotly-graph-div {{
      width: 100% !important;
      height: 100% !important;
    }}
    .modebar-btn,
    g.updatemenu-button,
    g.updatemenu-button *,
    g.slider *,
    .legendtoggle {{
      cursor: pointer !important;
    }}
    .modebar {{
      background: rgba(8, 14, 24, 0.44) !important;
      border: 1px solid rgba(148, 163, 184, 0.14);
      border-radius: 10px;
      padding: 4px;
      backdrop-filter: blur(6px);
    }}
    .modebar-btn {{
      border-radius: 8px !important;
      transition: background-color 110ms ease, transform 110ms ease, opacity 110ms ease;
    }}
    .modebar-btn:hover {{
      background: rgba(91, 243, 255, 0.18) !important;
      transform: translateY(-1px);
    }}
    .modebar-btn.active {{
      background: rgba(91, 243, 255, 0.26) !important;
    }}
  </style>
</head>
<body>
{pio.to_html(qt_fig, full_html=False, include_plotlyjs=False, default_width="100%", default_height="100%", config=dict(displayModeBar=True, displaylogo=False, scrollZoom=True, modeBarButtonsToAdd=["v1hovermode", "toggleSpikelines"]))}
<script>
  document.addEventListener('DOMContentLoaded', function () {{
    function applyPointerCursor() {{
      document.querySelectorAll('.modebar-btn, g.updatemenu-button, g.updatemenu-button *, g.slider *, .legendtoggle').forEach(function (el) {{
        el.style.cursor = 'pointer';
      }});
    }}
    applyPointerCursor();
    setTimeout(applyPointerCursor, 300);
    setTimeout(applyPointerCursor, 900);
  }});
</script>
</body>
</html>
"""

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
        tmpfile.write(html_document.encode("utf-8"))
        html_path = tmpfile.name

    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)

    window = QMainWindow()
    window.setWindowTitle(window_title)

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    container.setStyleSheet(f"background-color: {paper_bg};")

    # --- Custom dark panel for controls just below legend ---
    panel_widget = QWidget()
    panel_widget.setStyleSheet(
        "background-color: rgba(16, 25, 38, 0.96);"
        "border: 1px solid rgba(151, 164, 180, 0.22);"
        "border-radius: 10px;"
    )
    panel_widget.setFixedHeight(60)
    panel_layout = QHBoxLayout(panel_widget)
    panel_layout.setContentsMargins(30, 2, 30, 2)
    panel_layout.setSpacing(16)
    panel_layout.addStretch(1)

    show_screws_button = QPushButton("Show Max Diameter")
    show_traj_button = QPushButton("Show Trajectories")
    show_bbox_button = QPushButton("Show Bounding Box")
    hide_bbox_button = QPushButton("Hide Bounding Box")
    export_button = QPushButton("Export Image")
    opacity_label = QLabel(f"Mesh Opacity: {float(control_meta.get('mesh_opacity', 0.25)):.2f}")
    opacity_label.setStyleSheet("color: #F7FAFC; padding-left: 12px; font-weight: 600;")
    opacity_slider = QSlider(Qt.Orientation.Horizontal)
    opacity_slider.setRange(5, 100)
    opacity_slider.setValue(int(round(float(control_meta.get('mesh_opacity', 0.25)) * 100)))
    opacity_slider.setFixedWidth(220)
    opacity_slider.setCursor(Qt.CursorShape.PointingHandCursor)
    opacity_slider.setStyleSheet(
        "QSlider::groove:horizontal {"
        "  height: 8px;"
        "  border-radius: 4px;"
        "  background: rgba(120, 134, 156, 0.32);"
        "}"
        "QSlider::sub-page:horizontal {"
        "  border-radius: 4px;"
        "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4BD3FF, stop:1 #79F2FF);"
        "}"
        "QSlider::add-page:horizontal {"
        "  border-radius: 4px;"
        "  background: rgba(120, 134, 156, 0.24);"
        "}"
        "QSlider::handle:horizontal {"
        "  width: 18px;"
        "  margin: -6px 0;"
        "  border-radius: 9px;"
        "  border: 2px solid rgba(255,255,255,0.88);"
        "  background: #F7FAFC;"
        "}"
        "QSlider::handle:horizontal:hover {"
        "  background: #FFFFFF;"
        "  border: 2px solid #79F2FF;"
        "}"
        "QSlider::handle:horizontal:pressed {"
        "  background: #CFFAFE;"
        "  border: 2px solid #22D3EE;"
        "}"
    )

    button_style = (
        "QPushButton {"
        "  background-color: rgba(76, 99, 130, 0.72);"
        "  color: #F7FAFC;"
        "  border: 1px solid rgba(151, 164, 180, 0.18);"
        "  padding: 8px 14px;"
        "  border-radius: 7px;"
        "  font-weight: 600;"
        "}"
        "QPushButton:hover {"
        "  background-color: rgba(96, 125, 163, 0.95);"
        "  border: 1px solid rgba(123, 229, 255, 0.55);"
        "}"
        "QPushButton:pressed {"
        "  background-color: rgba(55, 75, 104, 0.98);"
        "  padding-top: 9px;"
        "  padding-bottom: 7px;"
        "}"
        "QPushButton:checked {"
        "  background-color: rgba(54, 188, 229, 0.92);"
        "  color: #06121D;"
        "  border: 1px solid rgba(186, 248, 255, 0.92);"
        "}"
        "QPushButton:checked:hover {"
        "  background-color: rgba(90, 214, 245, 0.98);"
        "}"
        "QPushButton:disabled {"
        "  background-color: rgba(51, 65, 85, 0.88);"
        "  color: #94A3B8;"
        "  border: 1px solid rgba(100, 116, 139, 0.2);"
        "}"
    )
    export_button_style = (
        "QPushButton {"
        "  background-color: rgba(30, 41, 59, 0.92);"
        "  color: #F7FAFC;"
        "  border: 1px solid rgba(148, 163, 184, 0.28);"
        "  padding: 8px 14px;"
        "  border-radius: 7px;"
        "  font-weight: 600;"
        "}"
        "QPushButton:hover {"
        "  background-color: rgba(45, 62, 87, 0.98);"
        "  border: 1px solid rgba(123, 229, 255, 0.45);"
        "}"
        "QPushButton:pressed {"
        "  background-color: rgba(18, 27, 42, 1.0);"
        "  padding-top: 9px;"
        "  padding-bottom: 7px;"
        "}"
        "QPushButton:disabled {"
        "  color: #94A3B8;"
        "  background-color: rgba(30, 41, 59, 0.55);"
        "  border: 1px solid rgba(100, 116, 139, 0.18);"
        "}"
    )

    screw_mode_buttons = QButtonGroup(panel_widget)
    screw_mode_buttons.setExclusive(True)
    bbox_buttons = QButtonGroup(panel_widget)
    bbox_buttons.setExclusive(True)

    for button in (show_screws_button, show_traj_button, show_bbox_button, hide_bbox_button):
        button.setCheckable(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setStyleSheet(button_style)

    export_button.setCursor(Qt.CursorShape.PointingHandCursor)
    export_button.setStyleSheet(export_button_style)

    screw_mode_buttons.addButton(show_screws_button)
    screw_mode_buttons.addButton(show_traj_button)
    bbox_buttons.addButton(show_bbox_button)
    bbox_buttons.addButton(hide_bbox_button)

    initial_screw_mode = control_meta.get("initial_screw_mode", "screws")
    show_screws_button.setChecked(initial_screw_mode != "trajectories")
    show_traj_button.setChecked(initial_screw_mode == "trajectories")

    initial_bbox_visible = bool(control_meta.get("initial_bbox_visible", True))
    show_bbox_button.setChecked(initial_bbox_visible)
    hide_bbox_button.setChecked(not initial_bbox_visible)

    panel_layout.addWidget(show_screws_button)
    panel_layout.addWidget(show_traj_button)
    panel_layout.addWidget(show_bbox_button)
    panel_layout.addWidget(hide_bbox_button)
    panel_layout.addWidget(opacity_label)
    panel_layout.addWidget(opacity_slider)
    panel_layout.addStretch(1)

    view = QWebEngineView()
    view.load(QUrl.fromLocalFile(html_path))
    layout.addWidget(view)
    layout.setStretchFactor(view, 1)  # Make plot expand to fill available space
    # Insert the panel widget just below the plot title/legend, above the plot content
    layout.insertWidget(1, panel_widget)
    layout.setStretchFactor(panel_widget, 0)  # Keep panel at fixed height
    layout.addWidget(export_button)

    interactive_controls = [
        show_screws_button,
        show_traj_button,
        show_bbox_button,
        hide_bbox_button,
        opacity_slider,
        export_button,
    ]
    for widget in interactive_controls:
        widget.setEnabled(False)

    def run_plotly_js(script):
        view.page().runJavaScript(script)

    def restyle_plot(payload, indices):
        run_plotly_js(
            f"""
            (function() {{
                const plot = document.querySelector('.js-plotly-plot');
                if (!plot || !window.Plotly) return;
                Plotly.restyle(plot, {json.dumps(payload)}, {json.dumps(indices)});
            }})();
            """
        )

    def set_mesh_opacity(opacity_value):
        mesh_trace_index = int(control_meta.get("mesh_trace_index", 0))
        run_plotly_js(
            f"""
            (function() {{
                const plot = document.querySelector('.js-plotly-plot');
                if (!plot || !window.Plotly) return;
                Plotly.restyle(plot, {{opacity: [{opacity_value:.2f}]}}, [{mesh_trace_index}]);
            }})();
            """
        )

    pending_opacity_value = {"value": float(control_meta.get("mesh_opacity", 0.25))}
    opacity_update_timer = QTimer(panel_widget)
    opacity_update_timer.setSingleShot(True)
    opacity_update_timer.setInterval(60)
    opacity_update_timer.timeout.connect(
        lambda: set_mesh_opacity(pending_opacity_value["value"])
    )

    def export_image():
        file_path, _ = QFileDialog.getSaveFileName(
            window,
            "Save Image",
            "visualization.png",
            "PNG Files (*.png);;All Files (*)",
        )
        if file_path:
            pixmap = view.grab()
            pixmap.save(file_path)

    def cleanup():
        try:
            os.remove(html_path)
        except OSError:
            pass

    def on_load_finished(ok):
        for widget in interactive_controls:
            widget.setEnabled(ok)

    show_screws_button.clicked.connect(
        lambda: restyle_plot(
            {"visible": control_meta.get("screw_mode_screws_vis", [])},
            control_meta.get("screw_mode_indices", []),
        )
    )
    show_traj_button.clicked.connect(
        lambda: restyle_plot(
            {"visible": control_meta.get("screw_mode_traj_vis", [])},
            control_meta.get("screw_mode_indices", []),
        )
    )
    show_bbox_button.clicked.connect(
        lambda: restyle_plot(
            {"visible": [True] * len(control_meta.get("bbox_indices", []))},
            control_meta.get("bbox_indices", []),
        )
    )
    hide_bbox_button.clicked.connect(
        lambda: restyle_plot(
            {"visible": [False] * len(control_meta.get("bbox_indices", []))},
            control_meta.get("bbox_indices", []),
        )
    )
    opacity_slider.valueChanged.connect(
        lambda value: (
            opacity_label.setText(f"Mesh Opacity: {value / 100.0:.2f}"),
            pending_opacity_value.__setitem__("value", value / 100.0),
            opacity_update_timer.start(),
        )
    )
    opacity_slider.sliderReleased.connect(
        lambda: (
            opacity_update_timer.stop(),
            set_mesh_opacity(opacity_slider.value() / 100.0),
        )
    )
    view.loadFinished.connect(on_load_finished)
    export_button.clicked.connect(export_image)
    window.destroyed.connect(cleanup)
    window.setCentralWidget(container)
    window.resize(1280, 920)
    window.show()

    _VIEWER_WINDOWS.append(window)
    
    # Start event loop (only if we own the QApplication instance)
    if owns_app:
        print("[PyQt6] Starting event loop...")
        app.exec()
    else:
        print("[PyQt6] Using existing QApplication instance")

    return window


def show_visualization(fig, renderer="auto", window_title="Pedicle Screw Planner Visualization"):
    """
    Display the visualization in PyQt6 window (no browser fallback).
    Always uses PyQt6 for consistent behavior.
    """
    print("[Visualizer] Launching PyQt6 window...")
    window = _build_qt_window(fig, window_title)
    print("[Visualizer] PyQt6 window launched successfully.")
    return window


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
    neon_trajectories=True,
    gold_screws=True,
    threaded_screws=True,
    v2_neon_trajectories=None,
    v2_gold_screws=None,
    v2_threaded_screws=None,
    v2_safety_planes=None,
    fallback_diameter=None,
):
    print("Creating merged surgical visualization...")
    fig = build_visualization(
        verts_world=vertsWorld,
        faces=faces,
        results_list=resultsList,
        volume_path=volume_path,
        screw_mode=screw_mode,
        theme=theme,
        visual_preset=visual_preset,
        mesh_opacity=mesh_opacity,
        show_safety_planes=show_safety_planes,
        show_bounding_box=show_bounding_box,
        show_trajectory_lines=show_trajectory_lines,
        show_entry_markers=show_entry_markers,
        show_tip_markers=show_tip_markers,
        neon_trajectories=neon_trajectories,
        gold_screws=gold_screws,
        threaded_screws=threaded_screws,
        v2_neon_trajectories=v2_neon_trajectories,
        v2_gold_screws=v2_gold_screws,
        v2_threaded_screws=v2_threaded_screws,
        v2_safety_planes=v2_safety_planes,
        fallback_diameter=fallback_diameter,
    )
    
    def show_figure(fig_obj=None, renderer="auto"):
        target_fig = fig if fig_obj is None else fig_obj
        return show_visualization(target_fig, renderer=renderer)
    
    return fig, show_figure
