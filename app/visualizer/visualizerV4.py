import plotly.graph_objects as go
import numpy as np
import os
import datetime

# PyQt6 (STRICT: required)
import sys
import tempfile
from PyQt6.QtWidgets import QApplication, QMainWindow

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except ImportError:
    QWebEngineView = None


# --------------------------------------------------
# Cylinder Generator (same style as your V3 math)
# --------------------------------------------------
def createCylinder(entry, tip, diameter, resolution=40):
    entry = np.array(entry)
    tip = np.array(tip)

    v = tip - entry
    mag = np.linalg.norm(v)

    if mag == 0:
        return None, None, None

    v_unit = v / mag

    # Orthogonal vectors (same logic as your V3)
    not_v = np.array([1, 0, 0]) if abs(v_unit[0]) < 0.8 else np.array([0, 1, 0])
    n1 = np.cross(v_unit, not_v)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v_unit, n1)

    radius = diameter / 2.0

    t = np.linspace(0, mag, 30)
    theta = np.linspace(0, 2 * np.pi, resolution)

    t_grid, theta_grid = np.meshgrid(t, theta)

    x = entry[0] + v_unit[0]*t_grid + radius*(np.cos(theta_grid)*n1[0] + np.sin(theta_grid)*n2[0])
    y = entry[1] + v_unit[1]*t_grid + radius*(np.cos(theta_grid)*n1[1] + np.sin(theta_grid)*n2[1])
    z = entry[2] + v_unit[2]*t_grid + radius*(np.cos(theta_grid)*n1[2] + np.sin(theta_grid)*n2[2])

    return x, y, z


# --------------------------------------------------
# MAIN VISUALIZER FUNCTION
# --------------------------------------------------
def visualize_surgical_plan(vertsWorld, faces, resultsList, volume_path=None):

    print("Creating Enhanced Surgical Visualization (V4)...")

    fig = go.Figure()

    # -------------------------------
    # Volume Metadata
    # -------------------------------
    if volume_path:
        volume_name = os.path.basename(volume_path)
        try:
            ctime = os.path.getctime(volume_path)
            timestamp = datetime.datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S")
        except:
            timestamp = "Unknown"
    else:
        volume_name = "Unknown Volume"
        timestamp = "Unknown"

    # -------------------------------
    # 1. Vertebra Mesh
    # -------------------------------
    fig.add_trace(go.Mesh3d(
        x=vertsWorld[:, 0],
        y=vertsWorld[:, 1],
        z=vertsWorld[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=0.25,
        color='lightgray',
        name=volume_name,
        hovertemplate=(
            f"<b>{volume_name}</b><br>"
            f"Created: {timestamp}<br>"
            "X: %{x:.2f} Y: %{y:.2f} Z: %{z:.2f}<extra></extra>"
        )
    ))

    # -------------------------------
    # 2. Bounding Box
    # -------------------------------
    min_xyz = np.min(vertsWorld, axis=0)
    max_xyz = np.max(vertsWorld, axis=0)

    corners = np.array([
        [min_xyz[0], min_xyz[1], min_xyz[2]],
        [max_xyz[0], min_xyz[1], min_xyz[2]],
        [max_xyz[0], max_xyz[1], min_xyz[2]],
        [min_xyz[0], max_xyz[1], min_xyz[2]],
        [min_xyz[0], min_xyz[1], max_xyz[2]],
        [max_xyz[0], min_xyz[1], max_xyz[2]],
        [max_xyz[0], max_xyz[1], max_xyz[2]],
        [min_xyz[0], max_xyz[1], max_xyz[2]],
    ])

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[corners[i,0], corners[j,0]],
            y=[corners[i,1], corners[j,1]],
            z=[corners[i,2], corners[j,2]],
            mode='lines',
            line=dict(color='#00f0ff',width=6),
            showlegend=False,
            hoverinfo='skip'
        ))

    # -------------------------------
    # 3. Screws + Trajectories
    # -------------------------------
    for r in resultsList:
        entry = np.array(r["entry"])
        tip = np.array(r["tip"])
        diam = r.get("diameter", 0.2)
        side = r.get("side", "")

        # --- Trajectory Line
        fig.add_trace(go.Scatter3d(
            x=[entry[0], tip[0]],
            y=[entry[1], tip[1]],
            z=[entry[2], tip[2]],
            mode='lines',
            line=dict(
                color='red' if side.lower() != "left" else '#000000',
                width=6
            ),
            name=f"{side} Trajectory"
        ))

        # --- Entry Point
        fig.add_trace(go.Scatter3d(
            x=[entry[0]],
            y=[entry[1]],
            z=[entry[2]],
            mode='markers',
            marker=dict(
                size=8,
                color='#00f0ff'
            ),
            name=f"{side} Entry"
        ))

        # --- Screw Cylinder
        if diam > 0:
            X, Y, Z = createCylinder(entry, tip, diam)
            if X is None:
                continue

            depth = np.linalg.norm(tip - entry)

            fig.add_trace(go.Surface(
                x=X,
                y=Y,
                z=Z,
                opacity=0.75,
                showscale=False,
                colorscale=[[0, '#39ff14'], [1, '#0aff9d']],  # neon green gradient
                hovertemplate=(
                    f"<b>{side} Screw</b><br>"
                    f"Depth: {depth:.2f} mm<extra></extra>"
                ),
                lighting=dict(
                    ambient=0.6,
                    diffuse=0.8,
                    specular=0.5,
                    roughness=0.4
                )
            ))

    # -------------------------------
    # 4. Opacity Slider
    # -------------------------------
    opacities = np.linspace(0.05, 1.0, 20)

    steps = [
        dict(
            method="restyle",
            args=[{"opacity": [op]}, [0]],
            label=f"{op:.2f}"
        )
        for op in opacities
    ]

    fig.update_layout(
        title=f"Pedicle Screw Planner Visualization — {volume_name}",
        scene=dict(
            aspectmode='data',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='#0a1f3d'   # deep blue background
        ),
        paper_bgcolor='#0a1f3d',
        plot_bgcolor='#0a1f3d',
        template="plotly_dark",
        height=900,
        title_font=dict(size=20, color="#00f0ff")  # neon title
    )

    # -------------------------------
    # 5. Layout
    # -------------------------------
    fig.update_layout(
        title=f"Pedicle Screw Planner Visualization — {volume_name}",
        scene=dict(aspectmode='data'),
        template="plotly_dark",
        height=900
    )

    # -------------------------------
    # 6. SHOW (STRICT PyQt ONLY)
    # -------------------------------
    def show_figure():

        if QWebEngineView is None:
            raise RuntimeError(
                "PyQt6 WebEngine is REQUIRED.\n"
                "Install using: pip install PyQt6-WebEngine"
            )

        from PyQt6.QtWidgets import QPushButton, QFileDialog, QVBoxLayout, QWidget
        from PyQt6.QtCore import QUrl

        # Save temporary HTML
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            fig.write_html(tmp.name)
            html_path = tmp.name

        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)

        window = QMainWindow()
        window.setWindowTitle("Surgical Visualization (In-App)")

        # Layout
        main_widget = QWidget()
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # Plot view
        view = QWebEngineView()
        view.load(QUrl.fromLocalFile(html_path))
        layout.addWidget(view)

        # Export button
        export_btn = QPushButton("Export Image")
        layout.addWidget(export_btn)

        def export_image():
            img = view.grab()
            file_path, _ = QFileDialog.getSaveFileName(
                window,
                "Save Image",
                "visualization.png",
                "PNG Files (*.png)"
            )
            if file_path:
                img.save(file_path)

        export_btn.clicked.connect(export_image)

        window.setCentralWidget(main_widget)
        window.resize(1200, 900)
        window.show()

        app.exec()

    return fig, show_figure