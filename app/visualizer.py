# import plotly.graph_objects as go
# import numpy as np


# def createCylinder(entry, tip, diameter, resolution=40):
#     """
#     Builds a cylindrical screw surface between entry and tip.
#     """

#     entry = np.array(entry)
#     tip = np.array(tip)

#     direction = tip - entry
#     length = np.linalg.norm(direction)

#     if length == 0:
#         return None, None, None

#     direction = direction / length

#     # Find perpendicular vectors
#     if abs(direction[2]) < 0.9:
#         v = np.cross(direction, [0,0,1])
#     else:
#         v = np.cross(direction, [0,1,0])

#     v = v / np.linalg.norm(v)
#     w = np.cross(direction, v)

#     radius = diameter / 2

#     theta = np.linspace(0, 2*np.pi, resolution)
#     z = np.linspace(0, length, 30)

#     theta, z = np.meshgrid(theta, z)

#     x = radius * np.cos(theta)
#     y = radius * np.sin(theta)

#     X = entry[0] + direction[0]*z + v[0]*x + w[0]*y
#     Y = entry[1] + direction[1]*z + v[1]*x + w[1]*y
#     Z = entry[2] + direction[2]*z + v[2]*x + w[2]*y

#     return X, Y, Z


# def visualize_surgical_plan(vertsWorld, faces, resultsList):
#     """
#     Creates interactive 3D visualization of vertebra mesh and screws.
#     """

#     print("Creating Surgical-Grade Visualization...")

#     fig = go.Figure()

#     # Vertebra surface
#     fig.add_trace(go.Mesh3d(
#         x=vertsWorld[:,0],
#         y=vertsWorld[:,1],
#         z=vertsWorld[:,2],
#         i=faces[:,0],
#         j=faces[:,1],
#         k=faces[:,2],
#         opacity=0.25,
#         color='lightgray',
#         name="Lumbar Vertebrae"
#     ))

#     # Screws
#     for r in resultsList:

#         X, Y, Z = createCylinder(
#             r["entry"],
#             r["tip"],
#             r["diameter"]
#         )

#         if X is None:
#             continue

#         fig.add_trace(go.Surface(
#             x=X,
#             y=Y,
#             z=Z,
#             showscale=False,
#             opacity=1
#         ))

#     # Entry markers
#     for r in resultsList:

#         entry = r["entry"]

#         fig.add_trace(go.Scatter3d(
#             x=[entry[0]],
#             y=[entry[1]],
#             z=[entry[2]],
#             mode='markers',
#             marker=dict(size=5, color='green'),
#             showlegend=False
#         ))

#     fig.update_layout(
#         title="Pedicle Screw Planner Visualization",
#         scene=dict(aspectmode='data'),
#         height=900
#     )

#     # Always open in browser (stable local behavior)
#     fig.show(renderer="browser")


import plotly.graph_objects as go
import numpy as np


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


def visualize_surgical_plan(vertsWorld, faces, resultsList):
    """
    Creates interactive 3D visualization of vertebra mesh and screws.
    """

    print("Creating Surgical-Grade Visualization...")

    fig = go.Figure()

    # Vertebra surface
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

    # Screws
    for r in resultsList:

        X, Y, Z = createCylinder(
            r["entry"],
            r["tip"],
            r["diameter"]
        )

        if X is None:
            continue

        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=Z,
            showscale=False,
            opacity=1
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
        title="Pedicle Screw Planner Visualization",
        scene=dict(aspectmode='data'),
        height=900
    )

    # Always open in browser (stable local behavior)
    fig.show(renderer="browser")