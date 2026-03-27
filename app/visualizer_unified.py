import datetime
import os
import sys
import tempfile

import numpy as np
import plotly.graph_objects as go


_VIEWER_WINDOWS = []


def _normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return None
    return vector / norm


def _orthonormal_basis(direction):
    unit_direction = _normalize(direction)
    if unit_direction is None:
        return None, None, None

    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(unit_direction, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    normal_1 = np.cross(unit_direction, reference)
    normal_1 = _normalize(normal_1)
    if normal_1 is None:
        return None, None, None

    normal_2 = np.cross(unit_direction, normal_1)
    normal_2 = _normalize(normal_2)
    return unit_direction, normal_1, normal_2


def _build_screw_surface(entry, tip, diameter, screw_mode="threaded", resolution=36, axial_steps=48):
    entry = np.asarray(entry, dtype=float)
    tip = np.asarray(tip, dtype=float)
    direction = tip - entry
    length = np.linalg.norm(direction)
    axis, normal_1, normal_2 = _orthonormal_basis(direction)
    
    # Early return if any required component is None or invalid
    if axis is None:
        return None
    if normal_1 is None:
        return None
    if normal_2 is None:
        return None
    if diameter <= 0:
        return None
    if screw_mode == "none":
        return None

    theta = np.linspace(0.0, 2.0 * np.pi, resolution)
    t = np.linspace(0.0, 1.0, axial_steps)
    theta_grid, t_grid = np.meshgrid(theta, t, indexing="ij")

    base_radius = diameter / 2.0
    if screw_mode == "threaded":
        thread_depth = min(0.35, max(0.12, base_radius * 0.12))
        thread_turns = max(8, int(length / 2.4))
        radius = base_radius + thread_depth * np.sin(t_grid * thread_turns * 2.0 * np.pi)
    else:
        radius = np.full_like(t_grid, base_radius)

    axial_distance = length * t_grid
    x = (
        entry[0]
        + axis[0] * axial_distance
        + radius * (np.cos(theta_grid) * normal_1[0] + np.sin(theta_grid) * normal_2[0])
    )
    y = (
        entry[1]
        + axis[1] * axial_distance
        + radius * (np.cos(theta_grid) * normal_1[1] + np.sin(theta_grid) * normal_2[1])
    )
    z = (
        entry[2]
        + axis[2] * axial_distance
        + radius * (np.cos(theta_grid) * normal_1[2] + np.sin(theta_grid) * normal_2[2])
    )
    return x, y, z


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


def _is_trace_visible(trace):
    if trace.visible is None:
        return True
    if isinstance(trace.visible, str):
        return trace.visible != "legendonly"
    return bool(trace.visible)


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
            "surface_opacity": 0.92,
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
            "surface_opacity": 0.95,
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
            "surface_opacity": 0.97,
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
            hovertemplate=(
                f"<b>{volume_name}</b><br>Created: {timestamp}"
                "<br>X: %{x:.2f} Y: %{y:.2f} Z: %{z:.2f}"
                "<extra></extra>"
            ),
            lighting=style["mesh_lighting"],
            lightposition=style["mesh_lightposition"],
            showscale=False,
        )
    )


def _add_bounding_box(fig, verts_world, color, return_trace=False):
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
    traces = []
    for start_idx, end_idx in edges:
        trace = go.Scatter3d(
            x=[corners[start_idx, 0], corners[end_idx, 0]],
            y=[corners[start_idx, 1], corners[end_idx, 1]],
            z=[corners[start_idx, 2], corners[end_idx, 2]],
            mode="lines",
            line=dict(color=color, width=3, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        )
        traces.append(trace)
        if not return_trace:
            fig.add_trace(trace)
    if return_trace:
        return traces


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
    style = _style_config(visual_preset=visual_preset, theme=theme)
    if mesh_opacity is None:
        mesh_opacity = style["mesh_opacity"]

    neon_trajectories = _resolve_kicker(v2_neon_trajectories, neon_trajectories)
    gold_screws = _resolve_kicker(v2_gold_screws, gold_screws)
    threaded_screws = _resolve_kicker(v2_threaded_screws, threaded_screws)
    show_safety_planes = _resolve_kicker(v2_safety_planes, show_safety_planes)

    fig = go.Figure()
    volume_name, timestamp = _volume_metadata(volume_path)
    _add_mesh(fig, verts_world, faces, volume_name, timestamp, mesh_opacity, style)
    bounding_box_traces = _add_bounding_box(fig, verts_world, style["bounding_box"], return_trace=True) or []
    for trace in bounding_box_traces:
        trace.visible = show_bounding_box
        fig.add_trace(trace)

    screw_traces = []
    trajectory_traces = []
    entry_marker_traces = []
    tip_marker_traces = []
    safety_plane_traces = []

    for result in results_list:
        entry = np.asarray(result["entry"], dtype=float)
        tip = np.asarray(result["tip"], dtype=float)
        direction = tip - entry
        depth = np.linalg.norm(direction)
        side_style = _side_style(result.get("side", ""), style)
        hover_text = _format_result_hover(result, entry, tip, depth)

        diameter = _default_visual_diameter(result, fallback_diameter=fallback_diameter)
        if diameter > 0:
            # Compute unit vector and orthogonal vectors for the screw
            direction = tip - entry
            mag = np.linalg.norm(direction)
            if mag > 0:
                v_unit = direction / mag
            else:
                v_unit = np.array([0., 0., 1.])

            not_v = np.array([1, 0, 0]) if abs(v_unit[0]) < 0.8 else np.array([0, 1, 0])
            n1 = np.cross(v_unit, not_v)
            n1 = n1 / np.linalg.norm(n1)
            n2 = np.cross(v_unit, n1)

            t = np.linspace(0, 1, 60)
            theta = np.linspace(0, 2*np.pi, 36)
            t_grid, theta_grid = np.meshgrid(t, theta)

            radius = diameter / 2.0

            x = entry[0] + direction[0]*t_grid + radius*(np.cos(theta_grid)*n1[0] + np.sin(theta_grid)*n2[0])
            y = entry[1] + direction[1]*t_grid + radius*(np.cos(theta_grid)*n1[1] + np.sin(theta_grid)*n2[1])
            z = entry[2] + direction[2]*t_grid + radius*(np.cos(theta_grid)*n1[2] + np.sin(theta_grid)*n2[2])

            screw_traces.append(
                go.Surface(
                    x=x, y=y, z=z,
                    opacity=0.75,
                    showscale=False,
                    colorscale=[[0, '#39ff14'], [1, '#0aff9d']],
                    lighting=dict(ambient=0.6, diffuse=0.8, specular=0.5, roughness=0.4),
                    hovertemplate=hover_text,
                    name=f"{result.get('side', '').strip()} Screw",
                    legendgroup=side_style["legendgroup"],
                    showlegend=False,
                )
            )

            if show_safety_planes:
                plane_size = 8
                p_t = np.linspace(-plane_size, plane_size, 2)
                p_s = np.linspace(-plane_size, plane_size, 2)
                pt_grid, ps_grid = np.meshgrid(p_t, p_s)

                plane_x = tip[0] + pt_grid*n1[0] + ps_grid*n2[0]
                plane_y = tip[1] + pt_grid*n1[1] + ps_grid*n2[1]
                plane_z = tip[2] + pt_grid*n1[2] + ps_grid*n2[2]

                safety_plane_traces.append(
                    go.Surface(
                        x=plane_x, y=plane_y, z=plane_z,
                        opacity=0.35,
                        showscale=False,
                        colorscale=[[0, '#ff2a6d'], [1, '#ff2a6d']],
                        name="80% Safety Limit"
                    )
                )

        trajectory_traces.append(
            go.Scatter3d(
                x=[entry[0], tip[0]],
                y=[entry[1], tip[1]],
                z=[entry[2], tip[2]],
                mode="lines",
                line=dict(color='#00f0ff', width=6),
                name=f"{result.get('side', '').strip()} Trajectory",
                legendgroup=side_style["legendgroup"],
                hovertemplate=hover_text,
                showlegend=True,
                visible=show_trajectory_lines,
            )
        )

        entry_marker_traces.append(
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
        )

        tip_marker_traces.append(
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
        )

    # Add traces to figure
    for trace in screw_traces:
        fig.add_trace(trace)
    for trace in trajectory_traces:
        fig.add_trace(trace)
    for trace in entry_marker_traces:
        fig.add_trace(trace)
    for trace in tip_marker_traces:
        fig.add_trace(trace)
    for trace in safety_plane_traces:
        fig.add_trace(trace)

    n_mesh = 1
    n_bbox = len(bounding_box_traces)
    n_screws = len(screw_traces)
    n_traj = len(trajectory_traces)
    n_entry = len(entry_marker_traces)
    n_tip = len(tip_marker_traces)
    n_safety = len(safety_plane_traces)
    fat_screws_vis = (
        [True] * n_mesh
        + [show_bounding_box] * n_bbox
        + [True] * n_screws
        + [False] * n_traj
        + [show_entry_markers] * n_entry
        + [show_tip_markers] * n_tip
        + [show_safety_planes] * n_safety
    )
    thin_traj_vis = (
        [True] * n_mesh
        + [show_bounding_box] * n_bbox
        + [False] * n_screws
        + [show_trajectory_lines] * n_traj
        + [show_entry_markers] * n_entry
        + [show_tip_markers] * n_tip
        + [show_safety_planes] * n_safety
    )
    bbox_on_vis = [True] + [True] * n_bbox + [_is_trace_visible(trace) for trace in fig.data[(n_mesh + n_bbox):]]
    bbox_off_vis = [True] + [False] * n_bbox + [_is_trace_visible(trace) for trace in fig.data[(n_mesh + n_bbox):]]
    updatemenus = [
        dict(
            type="buttons",
            direction="right",
            showactive=True,
            x=0.01,
            y=1.05,
            xanchor="left",
            yanchor="top",
            buttons=[
                dict(
                    label="Show Screws",
                    method="update",
                    args=[{"visible": fat_screws_vis}],
                ),
                dict(
                    label="Show Trajectories",
                    method="update",
                    args=[{"visible": thin_traj_vis}],
                ),
            ],
        ),
        dict(
            type="buttons",
            direction="right",
            showactive=True,
            x=0.35,
            y=1.05,
            xanchor="left",
            yanchor="top",
            buttons=[
                dict(
                    label="Show Bounding Box",
                    method="update",
                    args=[{"visible": bbox_on_vis}],
                ),
                dict(
                    label="Hide Bounding Box",
                    method="update",
                    args=[{"visible": bbox_off_vis}],
                ),
            ],
        ),
    ]

    mesh_opacities = [round(value, 2) for value in np.linspace(0.05, 1.0, 20)]
    fig.update_layout(
        title=f"Pedicle Screw Planner Visualization - {volume_name}",
        template=style["template"],
        paper_bgcolor=style["paper_bgcolor"],
        plot_bgcolor=style["plot_bgcolor"],
        title_font=dict(size=20, color=style["title_font_color"]),
        scene=dict(
            aspectmode="data",
            bgcolor=style["scene_bgcolor"],
            xaxis=dict(
                showbackground=style["show_axes"],
                visible=style["show_axes"],
                showgrid=style["show_grid"],
                zeroline=False,
            ),
            yaxis=dict(
                showbackground=style["show_axes"],
                visible=style["show_axes"],
                showgrid=style["show_grid"],
                zeroline=False,
            ),
            zaxis=dict(
                showbackground=style["show_axes"],
                visible=style["show_axes"],
                showgrid=style["show_grid"],
                zeroline=False,
            ),
            camera=dict(eye=style["camera_eye"]),
        ),
        height=780,
        margin=dict(l=0, r=0, t=60, b=20),
        updatemenus=updatemenus,
        sliders=[
            dict(
                active=4,
                currentvalue={"prefix": "Mesh Opacity: "},
                pad={"t": 10, "b": 0},
                x=0.1,
                y=-0.05,
                len=0.8,
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
    )
    fig.add_annotation(
        x=0.99,
        y=0.98,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="top",
        showarrow=False,
        text=f"{visual_preset.title()} View",
        font=dict(size=11, color=style["title_font_color"]),
        bgcolor="rgba(0,0,0,0.18)" if style["template"] == "plotly_dark" else "rgba(255,255,255,0.76)",
        borderpad=6,
    )
    return fig


def _should_use_browser(renderer):
    if renderer == "browser":
        return True
    if renderer != "auto":
        return False
    return "PyQt5" in sys.modules or "PyQt5.QtWidgets" in sys.modules


def _build_qt_window(fig, window_title):
    from PyQt6.QtCore import QUrl
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow, QPushButton, QVBoxLayout, QWidget

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
        fig.write_html(tmpfile.name)
        html_path = tmpfile.name

    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)

    window = QMainWindow()
    window.setWindowTitle(window_title)

    container = QWidget()
    layout = QVBoxLayout(container)

    view = QWebEngineView()
    view.load(QUrl.fromLocalFile(html_path))
    layout.addWidget(view)

    export_button = QPushButton("Export Image")
    layout.addWidget(export_button)

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

    export_button.clicked.connect(export_image)
    window.destroyed.connect(cleanup)
    window.setCentralWidget(container)
    window.resize(1280, 920)
    window.show()

    _VIEWER_WINDOWS.append(window)
    if owns_app:
        app.exec()

    return window


def show_visualization(fig, renderer="auto", window_title="Pedicle Screw Planner Visualization"):
    if _should_use_browser(renderer):
        fig.show(renderer="browser")
        return None

    try:
        return _build_qt_window(fig, window_title)
    except Exception:
        fig.show(renderer="browser")
        return None


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
