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
        v_unit = v / mag

        # --- 2. Bright Neon Trajectory ---
        fig.add_trace(go.Scatter3d(
            x=[entry[0], tip[0]], y=[entry[1], tip[1]], z=[entry[2], tip[2]],
            mode='lines', 
            line=dict(color='cyan', width=5), # Bright cyan for visibility
            name=f"{r['vertebra']} Path"
        ))

        # --- 3. Realistic Threaded Screw ---
        if diam > 0:
            # Orthogonal vectors for circular cross-section
            not_v = np.array([1, 0, 0]) if abs(v_unit[0]) < 0.8 else np.array([0, 1, 0])
            n1 = np.cross(v_unit, not_v); n1 /= np.linalg.norm(n1)
            n2 = np.cross(v_unit, n1)
            
            t = np.linspace(0, 1, 50) # More points for thread detail
            theta = np.linspace(0, 2*np.pi, 30)
            t_grid, theta_grid = np.meshgrid(t, theta)
            
            # Thread Logic: Oscillate the radius based on length
            thread_depth = 0.4 
            thread_freq = 15 # Number of threads along the length
            radius = (diam / 2.0) + thread_depth * np.sin(t_grid * thread_freq * 2 * np.pi)
            
            x = entry[0] + v[0]*t_grid + radius*(np.cos(theta_grid)*n1[0] + np.sin(theta_grid)*n2[0])
            y = entry[1] + v[1]*t_grid + radius*(np.cos(theta_grid)*n1[1] + np.sin(theta_grid)*n2[1])
            z = entry[2] + v[2]*t_grid + radius*(np.cos(theta_grid)*n1[2] + np.sin(theta_grid)*n2[2])
            
            fig.add_trace(go.Surface(
                x=x, y=y, z=z, opacity=0.9, 
                colorscale=[[0, 'gold'], [1, 'orange']], 
                showscale=False, name=f"{diam}mm Screw"
            ))

            # --- 4. 80% Safety Plane ---
            # Create a small plane at the tip orthogonal to the screw
            plane_size = 8
            p_t = np.linspace(-plane_size, plane_size, 2)
            p_s = np.linspace(-plane_size, plane_size, 2)
            pt_grid, ps_grid = np.meshgrid(p_t, p_s)
            
            plane_x = tip[0] + pt_grid*n1[0] + ps_grid*n2[0]
            plane_y = tip[1] + pt_grid*n1[1] + ps_grid*n2[1]
            plane_z = tip[2] + pt_grid*n1[2] + ps_grid*n2[2]

            fig.add_trace(go.Surface(
                x=plane_x, y=plane_y, z=plane_z,
                opacity=0.4, colorscale=[[0, 'red'], [1, 'red']],
                showscale=False, name="80% Safety Limit"
            ))

    fig.update_layout(
        title="Geometry Based Pedicle Screw Navigation Planner",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        template="plotly_dark", # Dark mode makes cyan/gold look amazing
        height=1000
    )
    fig.show(renderer="browser")