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

    # Vertebra surface with hover info
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
        hovertemplate=f"<b>{volume_name}</b><br>Created: {timestamp_str}<extra></extra>"
    ))

    # Screws
    for r in resultsList:
        # Use the thinnest possible diameter for screws (e.g., 0.2 mm)
        diameter = r.get("diameter", 0.2)
        X, Y, Z = createCylinder(
            r["entry"],
            r["tip"],
            diameter
        )
        if X is None:
            continue
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=Z,
            showscale=False,
            opacity=1,
            surfacecolor=np.ones_like(X),
            colorscale=[[0, 'red'], [1, 'red']],
            cmin=0,
            cmax=1
        ))

    # Entry markers
    for r in resultsList:

        entry = r["entry"]

        fig.add_trace(go.Scatter3d(
            x=[entry[0]],
            y=[entry[1]],
            z=[entry[2]],
            mode='markers',
            marker=dict(size=5, color='green'),
            showlegend=False
        ))

    fig.update_layout(
        title=f"Pedicle Screw Planner Visualization — {volume_name}",
        scene=dict(aspectmode='data'),
        height=900
    )

    # Show Plotly figure in a PyQt6 window using QWebEngineView
    if QWebEngineView is not None:
        # Save the figure to a temporary HTML file
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
        # Fallback: open in browser if QWebEngineView is not available
        fig.show(renderer="browser")