import plotly.graph_objects as go
import numpy as np

# PyQt6 imports for custom window visualization
import sys
import tempfile
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QUrl
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except ImportError:
    QWebEngineView = None  # fallback if not available

def createCylinder(entry, tip, diameter, resolution=40):
    """
    Builds a cylindrical screw surface between entry and tip.
    """

    entry = np.array(entry)
    tip = np.array(tip)

    direction = tip - entry
    length = np.linalg.norm(direction)

    if length == 0:
        return None, None, None

    direction = direction / length

    # Find perpendicular vectors
    if abs(direction[2]) < 0.9:
        v = np.cross(direction, [0,0,1])
    else:
        v = np.cross(direction, [0,1,0])

    v = v / np.linalg.norm(v)
    w = np.cross(direction, v)

    radius = diameter / 2

    theta = np.linspace(0, 2*np.pi, resolution)
    z = np.linspace(0, length, 30)

    theta, z = np.meshgrid(theta, z)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    X = entry[0] + direction[0]*z + v[0]*x + w[0]*y
    Y = entry[1] + direction[1]*z + v[1]*x + w[1]*y
    Z = entry[2] + direction[2]*z + v[2]*x + w[2]*y

    return X, Y, Z


import os
import datetime

def visualize_surgical_plan(vertsWorld, faces, resultsList, volume_path=None):
    """
    Creates interactive 3D visualization of vertebra mesh and screws.
    If volume_path is provided, includes the filename in the title and shows file creation timestamp on hover.
    """
    print("Creating Surgical-Grade Visualization...")
    fig = go.Figure()

    # Prepare volume name and timestamp for display
    if volume_path is not None:
        volume_name = os.path.basename(volume_path)
        try:
            ctime = os.path.getctime(volume_path)
            dt = datetime.datetime.fromtimestamp(ctime)
            timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            timestamp_str = "Unknown"
    else:
        volume_name = "Unknown Volume"
        timestamp_str = "Unknown"

    # Vertebra surface with hover info (show coordinates)
    fig.add_trace(go.Mesh3d(
        x=vertsWorld[:,0],
        y=vertsWorld[:,1],
        z=vertsWorld[:,2],
        i=faces[:,0],
        j=faces[:,1],
        k=faces[:,2],
        opacity=0.25,
        color='lightgray',
        name=volume_name,
        hovertemplate=(
            f"<b>{volume_name}</b><br>Created: {timestamp_str}" 
            "<br>X: %{x:.2f} Y: %{y:.2f} Z: %{z:.2f}"
            "<extra></extra>"
        )
    ))

    # Screws with depth info on hover
    for r in resultsList:
        diameter = r.get("diameter", 0.2)
        entry = np.array(r["entry"])
        tip = np.array(r["tip"])
        depth = np.linalg.norm(tip - entry)
        X, Y, Z = createCylinder(entry, tip, diameter)
        if X is None:
            continue
        # Custom hover text for the screw surface
        hovertext = (
            f"<b>{r.get('side', '')} Screw</b>"
            f"<br>Entry: [{entry[0]:.2f}, {entry[1]:.2f}, {entry[2]:.2f}]"
            f"<br>Tip: [{tip[0]:.2f}, {tip[1]:.2f}, {tip[2]:.2f}]"
            f"<br>Depth: {depth:.2f} mm"
            "<extra></extra>"
        )
        side = r.get("side", "").lower()
        if side == "left":
            screw_colorscale = [[0, '#000000'], [1, '#000000']]
        else:
            screw_colorscale = [[0, 'red'], [1, 'red']]
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=Z,
            showscale=False,
            opacity=1,
            surfacecolor=np.ones_like(X),
            colorscale=screw_colorscale,
            cmin=0,
            cmax=1,
            hovertemplate=hovertext
        ))

    # Entry markers and trajectory lines with legend coords and color coding
    for r in resultsList:
        entry = np.array(r["entry"])
        tip = np.array(r["tip"])
        side = r.get("side", "")
        if side.lower() == "left":
            entry_color = 'blue'
            # Use explicit RGB hex for black, and set legendgroup to avoid color inheritance
            fig.add_trace(go.Scatter3d(
                x=[entry[0]],
                y=[entry[1]],
                z=[entry[2]],
                mode='markers',
                marker=dict(size=8, color=entry_color, symbol='circle'),
                name=f"{side} Entry: [{entry[0]:.2f}, {entry[1]:.2f}, {entry[2]:.2f}]",
                showlegend=True,
                legendgroup='left_traj',
            ))
            fig.add_trace(go.Scatter3d(
                x=[entry[0], tip[0]],
                y=[entry[1], tip[1]],
                z=[entry[2], tip[2]],
                mode='lines',
                line=dict(color='#000000', width=6),
                name=f"{side} Trajectory",
                showlegend=True,
                legendgroup='left_traj',
            ))
        else:
            entry_color = 'green'
            fig.add_trace(go.Scatter3d(
                x=[entry[0]],
                y=[entry[1]],
                z=[entry[2]],
                mode='markers',
                marker=dict(size=8, color=entry_color, symbol='circle'),
                name=f"{side} Entry: [{entry[0]:.2f}, {entry[1]:.2f}, {entry[2]:.2f}]",
                showlegend=True,
                legendgroup='right_traj',
            ))
            fig.add_trace(go.Scatter3d(
                x=[entry[0], tip[0]],
                y=[entry[1], tip[1]],
                z=[entry[2], tip[2]],
                mode='lines',
                line=dict(color='red', width=6),
                name=f"{side} Trajectory",
                showlegend=True,
                legendgroup='right_traj',
            ))

    fig.update_layout(
        title=f"Pedicle Screw Planner Visualization — {volume_name}",
        scene=dict(aspectmode='data'),
        height=900
    )

    def show_figure(fig_obj=None):
        # Show Plotly figure in a PyQt6 window using QWebEngineView
        if QWebEngineView is not None:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
                fig.write_html(tmpfile.name)
                html_path = tmpfile.name
            app = QApplication.instance()
            if not app:
                app = QApplication(sys.argv)
            window = QMainWindow()
            window.setWindowTitle("Pedicle Screw Planner Visualization")
            view = QWebEngineView()
            view.load(QUrl.fromLocalFile(html_path))
            window.setCentralWidget(view)
            window.resize(1200, 900)
            window.show()
            app.exec()
        else:
            fig.show(renderer="browser")
    return fig, show_figure