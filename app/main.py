# import os

# from run_totalseg import run_totalseg
# from mesh_builder import build_vertebra_mesh
# from geometry import run_planner
# from visualizer import visualize_surgical_plan

# baseDir = os.path.dirname(os.path.abspath(__file__))
# inputCT = os.path.join(baseDir, "data", "case_0000.nii")

# # 1️⃣ Run segmentation
# segFolder = run_totalseg(inputCT)

# # 2️⃣ Build mesh
# vertsWorld, faces = build_vertebra_mesh(
#     segmentation_path=segFolder,
#     single_file=False
# )

# # 3️⃣ Plan screws
# segPath = os.path.join(baseDir, "data", "spine_2-label.nii")
# resultsList = run_planner(segPath)

# # 4️⃣ Visualize
# visualize_surgical_plan(
#     vertsWorld,
#     faces,
#     resultsList
# )

import os

from run_totalseg import run_totalseg
from mesh_builder import build_vertebra_mesh
from geometry import run_planner
from visualizer import visualize_surgical_plan


def main():

    baseDir = os.path.dirname(os.path.abspath(__file__))
    inputCT = os.path.join(baseDir, "data", "case_0000.nii")

    print("\n=== PEDICLE SCREW PLANNING PIPELINE STARTED ===\n")

    # 1️⃣ Run segmentation
    segData = run_totalseg(inputCT)

    segFolder = segData["seg_folder"]
    combinedSegPath = segData["combined_seg_path"]

    # 2️⃣ Build 3D mesh
    vertsWorld, faces = build_vertebra_mesh(segFolder)

    # 3️⃣ Run screw planner (uses unified segmentation)
    resultsList = run_planner(combinedSegPath)

    # 4️⃣ Visualize results
    visualize_surgical_plan(
        vertsWorld,
        faces,
        resultsList
    )

    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===\n")


if __name__ == "__main__":
    main()