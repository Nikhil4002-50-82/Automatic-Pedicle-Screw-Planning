import os
from geometry import run_planner
from visualizer import visualize_surgical_plan

def plan_and_visualize_geometry():
    """
    Run planning and visualization using geometry_2.py (L5-optimized geometry planner) on a segmented file specified in data/source.txt.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    source_txt = os.path.join(data_dir, "source.txt")
    with open(source_txt, "r") as f:
        segmented_file = f.readline().strip().strip('"').strip("'")
    print(f"Running geometry_2.py (L5-optimized) planning and visualization on: {segmented_file}")
    vertsWorld, faces = build_mesh_from_single_vertebra(segmented_file)
    results = run_planner(segmented_file)
    fig, show_figure = visualize_surgical_plan(vertsWorld, faces, results, volume_path=segmented_file)
    show_figure(fig)
    print("geometry_2.py (L5-optimized) planning and visualization completed.")

def build_mesh_from_single_vertebra(segmented_file):
    """
    Build a mesh from a single vertebra .nii file (e.g., L5 only).
    Returns vertsWorld, faces.
    """
    import nibabel as nib
    import numpy as np
    from skimage.measure import marching_cubes
    nii = nib.load(segmented_file)
    data = nii.get_fdata()
    affine = nii.affine
    verts, faces, _, _ = marching_cubes(data, level=0.5)
    vertsWorld = nib.affines.apply_affine(affine, verts)
    return vertsWorld, faces

if __name__ == "__main__":
    plan_and_visualize_geometry()
