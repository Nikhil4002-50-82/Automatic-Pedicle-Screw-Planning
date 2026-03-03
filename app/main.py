import os
from run_totalseg import run_totalseg
from mesh_builder import build_vertebra_mesh
from geometry import run_planner
from visualizer import visualize_surgical_plan
def main():
    # Get current folder
    baseDir=os.path.dirname(os.path.abspath(__file__))
    # Path to CT scan
    inputCT=os.path.join(baseDir,"data","case_0000.nii")
    print("PEDICLE SCREW PLANNING PIPELINE STARTED")
    # Step 1: Run segmentation
    segData=run_totalseg(inputCT)
    segFolder=segData["seg_folder"]
    combinedSegPath=segData["combined_seg_path"]
    # Step 2: Build 3D mesh
    vertsWorld,faces=build_vertebra_mesh(segFolder)
    # Step 3: Plan screws
    resultsList=run_planner(combinedSegPath)
    # Step 4: Visualize result
    visualize_surgical_plan(vertsWorld,faces,resultsList)
    print("PIPELINE COMPLETED SUCCESSFULLY")
if __name__=="__main__":
    # Run main function
    main()