import os
from mesh_builder import build_vertebra_mesh
from geometryV2 import run_planner
from visualizerV2 import visualize_surgical_plan

def main():

    # Base directory
    baseDir = os.path.dirname(os.path.abspath(__file__))

    print("PEDICLE SCREW PLANNING PIPELINE STARTED")

    # -----------------------------
    # Use existing segmentation
    # -----------------------------
    segFolder = os.path.join(baseDir, "seg_output")
    combinedSegPath = os.path.join(baseDir, "seg_output", "combined_seg.nii.gz")

    # Step 1: Build 3D mesh
    vertsWorld, faces = build_vertebra_mesh(segFolder)

    # Step 2: Plan screws
    resultsList = run_planner(combinedSegPath)

    # Step 3: Visualize
    visualize_surgical_plan(vertsWorld, faces, resultsList)

    print("PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()