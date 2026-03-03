import os
import shutil
import subprocess
import nibabel as nib
import numpy as np
def run_totalseg(inputCT):
    # Run TotalSegmentator and prepare output
    print("Running TotalSegmentator...")
    baseDir=os.path.dirname(os.path.abspath(__file__))
    segFolder=os.path.join(baseDir,"seg_output")
    # Delete old results
    if os.path.exists(segFolder):
        shutil.rmtree(segFolder)
    os.makedirs(segFolder,exist_ok=True)
    # Set license if not already set
    os.environ["TOTALSEG_LICENSE"]=os.getenv("TOTALSEG_LICENSE","aca_V4E9OLA5VBGIQO")
    # Command to segment L1-L5
    command=[
        "TotalSegmentator",
        "-i",inputCT,
        "-o",segFolder,
        "-ta","total",
        "-rs",
        "vertebrae_L1",
        "vertebrae_L2",
        "vertebrae_L3",
        "vertebrae_L4",
        "vertebrae_L5"
    ]
    subprocess.run(command,check=True)
    print("Segmentation Completed")
    combinedSegPath=build_combined_segmentation(segFolder)
    return {"seg_folder":segFolder,"combined_seg_path":combinedSegPath}
def build_combined_segmentation(segFolder):
    # Combine L1-L5 into one labeled file
    print("Building unified segmentation...")
    combinedMask=None
    affine=None
    vertebraFiles=[
        "vertebrae_L1.nii.gz",
        "vertebrae_L2.nii.gz",
        "vertebrae_L3.nii.gz",
        "vertebrae_L4.nii.gz",
        "vertebrae_L5.nii.gz"
    ]
    labelValue=5
    for fileName in vertebraFiles:
        filePath=os.path.join(segFolder,fileName)
        # Skip missing files
        if not os.path.exists(filePath):
            labelValue-=1
            continue
        nii=nib.load(filePath)
        data=nii.get_fdata()
        # Convert to binary mask
        binaryMask=(data>0).astype(np.int16)
        if combinedMask is None:
            combinedMask=binaryMask*labelValue
            affine=nii.affine
        else:
            combinedMask+=binaryMask*labelValue
        labelValue-=1
    combinedPath=os.path.join(segFolder,"combined_seg.nii.gz")
    nib.save(nib.Nifti1Image(combinedMask,affine),combinedPath)
    print("Unified segmentation saved.")
    return combinedPath