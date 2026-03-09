import plotly.graph_objects as go
import numpy as np


def visualize_surgical_plan(vertsWorld, faces, resultsList):
    """
    Temporary visualization showing:
    - Vertebra mesh
    - Entry points
    - Pedicle trajectory lines
    """

    print("Creating Trajectory Visualization...")

    fig = go.Figure()

    # -------------------------
    # Vertebra mesh
    # -------------------------
    fig.add_trace(go.Mesh3d(
        x=vertsWorld[:,0],
        y=vertsWorld[:,1],
        z=vertsWorld[:,2],
        i=faces[:,0],
        j=faces[:,1],
        k=faces[:,2],
        opacity=0.25,
        color='lightgray',
        name="Lumbar Vertebrae"
    ))

    # -------------------------
    # Trajectory lines
    # -------------------------
    for r in resultsList:

        entry = np.array(r["entry"])
        tip = np.array(r["tip"])

        fig.add_trace(go.Scatter3d(
            x=[entry[0], tip[0]],
            y=[entry[1], tip[1]],
            z=[entry[2], tip[2]],
            mode='lines',
            line=dict(
                color='red',
                width=6
            ),
            name=f"{r['vertebra']} {r['side']}"
        ))

    # -------------------------
    # Entry markers
    # -------------------------
    for r in resultsList:

        entry = r["entry"]

        fig.add_trace(go.Scatter3d(
            x=[entry[0]],
            y=[entry[1]],
            z=[entry[2]],
            mode='markers',
            marker=dict(
                size=6,
                color='green'
            ),
            showlegend=False
        ))

    fig.update_layout(
        title="Pedicle Screw Trajectory Planner",
        scene=dict(aspectmode='data'),
        height=900
    )

    fig.show(renderer="browser")