import plotly.graph_objects as go
import numpy as np

def visualize_surgical_plan(vertsWorld, faces, resultsList):
    print("Creating Enhanced Surgical Visualization...")
    fig = go.Figure()

    # --- 1. Vertebra Mesh ---
    fig.add_trace(go.Mesh3d(
        x=vertsWorld[:,0], y=vertsWorld[:,1], z=vertsWorld[:,2],
        i=faces[:,0], j=faces[:,1], k=faces[:,2],
        opacity=0.3, color='lightgray', name="Anatomy"
    ))

    for r in resultsList:
        entry = np.array(r["entry"])
        tip = np.array(r["tip"])
        diam = r.get("diameter", 0)
        v = tip - entry
        mag = np.linalg.norm(v)
        v_unit = v / mag if mag > 0 else np.array([0., 0., 1.])

        # --- 2. Darker Neon Cyan Trajectory ---
        fig.add_trace(go.Scatter3d(
            x=[entry[0], tip[0]], y=[entry[1], tip[1]], z=[entry[2], tip[2]],
            mode='lines', 
            line=dict(color='#00b8d4', width=6),   # Darker cyan for better balance
            name=f"{r['vertebra']} Path"
        ))

        # --- 3. Smooth Transparent Neon Green Screw ---
        if diam > 0:
            # Orthogonal vectors for circular cross-section
            not_v = np.array([1, 0, 0]) if abs(v_unit[0]) < 0.8 else np.array([0, 1, 0])
            n1 = np.cross(v_unit, not_v); n1 /= np.linalg.norm(n1)
            n2 = np.cross(v_unit, n1)
            
            t = np.linspace(0, 1, 60)
            theta = np.linspace(0, 2*np.pi, 36)
            t_grid, theta_grid = np.meshgrid(t, theta)
            
            # Clean cylinder - no thread oscillation
            radius = diam / 2.0
            
            x = entry[0] + v[0]*t_grid + radius*(np.cos(theta_grid)*n1[0] + np.sin(theta_grid)*n2[0])
            y = entry[1] + v[1]*t_grid + radius*(np.cos(theta_grid)*n1[1] + np.sin(theta_grid)*n2[1])
            z = entry[2] + v[2]*t_grid + radius*(np.cos(theta_grid)*n1[2] + np.sin(theta_grid)*n2[2])
            
            fig.add_trace(go.Surface(
                x=x, y=y, z=z, 
                opacity=0.55,          # More transparent as requested
                colorscale=[[0, '#39ff14'], [1, '#7fff4d']],  # Slightly lighter neon green
                showscale=False, 
                name=f"{diam}mm Screw",
                lighting=dict(ambient=0.6, diffuse=0.8, specular=0.5, roughness=0.4)
            ))

            # --- 4. 80% Safety Plane ---
            plane_size = 8
            p_t = np.linspace(-plane_size, plane_size, 2)
            p_s = np.linspace(-plane_size, plane_size, 2)
            pt_grid, ps_grid = np.meshgrid(p_t, p_s)
            
            plane_x = tip[0] + pt_grid*n1[0] + ps_grid*n2[0]
            plane_y = tip[1] + pt_grid*n1[1] + ps_grid*n2[1]
            plane_z = tip[2] + pt_grid*n1[2] + ps_grid*n2[2]

            fig.add_trace(go.Surface(
                x=plane_x, y=plane_y, z=plane_z,
                opacity=0.35, 
                colorscale=[[0, '#ff2a6d'], [1, '#ff2a6d']],
                showscale=False, 
                name="80% Safety Limit"
            ))

    fig.update_layout(
        title="Geometry Based Pedicle Screw Navigation Planner",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            bgcolor='#0a1f3d'
        ),
        paper_bgcolor='#0a1f3d',
        plot_bgcolor='#0a1f3d',
        template="plotly_dark",
        height=1000,
        title_font=dict(size=20, color="#00f0ff")
    )
    fig.show(renderer="browser")