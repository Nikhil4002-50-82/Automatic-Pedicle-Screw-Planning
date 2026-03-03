# import os
# import shutil
# import subprocess

# def run_totalseg(inputCT):

#     print("Activating TotalSegmentator Student License...")
#     os.environ["TOTALSEG_LICENSE"] = "aca_V4E9OLA5VBGIQO"
#     print("License Activated")

#     baseDir = os.path.dirname(os.path.abspath(__file__))
#     outputFolder = os.path.join(baseDir, "totalseg_out")

#     if os.path.exists(outputFolder):
#         shutil.rmtree(outputFolder)

#     os.makedirs(outputFolder, exist_ok=True)

#     print("Running TotalSegmentator...")

#     command = [
#         "TotalSegmentator",
#         "-i", inputCT,
#         "-o", outputFolder,
#         "-ta", "total",
#         "-rs",
#         "vertebrae_L1",
#         "vertebrae_L2",
#         "vertebrae_L3",
#         "vertebrae_L4",
#         "vertebrae_L5"
#     ]

#     subprocess.run(command, check=True)

#     print("Segmentation Completed")

#     return outputFolder


import os
import shutil
import subprocess
import nibabel as nib
import numpy as np


def run_totalseg(inputCT):
    """
    Runs TotalSegmentator and prepares standardized segmentation output.

    Returns:
        {
            "seg_folder": folder containing vertebra files,
            "combined_seg_path": unified segmentation file
        }
    """

    print("Running TotalSegmentator...")

    baseDir = os.path.dirname(os.path.abspath(__file__))
    segFolder = os.path.join(baseDir, "seg_output")

    # Remove old results
    if os.path.exists(segFolder):
        shutil.rmtree(segFolder)

    os.makedirs(segFolder, exist_ok=True)

    # Optional: activate license if needed
    os.environ["TOTALSEG_LICENSE"] = os.getenv("TOTALSEG_LICENSE", "aca_V4E9OLA5VBGIQO")

    command = [
        "TotalSegmentator",
        "-i", inputCT,
        "-o", segFolder,
        "-ta", "total",
        "-rs",
        "vertebrae_L1",
        "vertebrae_L2",
        "vertebrae_L3",
        "vertebrae_L4",
        "vertebrae_L5"
    ]

    subprocess.run(command, check=True)

    print("Segmentation Completed")

    combinedSegPath = build_combined_segmentation(segFolder)

    return {
        "seg_folder": segFolder,
        "combined_seg_path": combinedSegPath
    }


def build_combined_segmentation(segFolder):
    """
    Combines vertebrae_L1-L5 into a single multi-label mask.
    """

    print("Building unified segmentation...")

    combinedMask = None
    affine = None

    vertebraFiles = [
        "vertebrae_L1.nii.gz",
        "vertebrae_L2.nii.gz",
        "vertebrae_L3.nii.gz",
        "vertebrae_L4.nii.gz",
        "vertebrae_L5.nii.gz"
    ]

    labelValue = 5  # L1=5 ... L5=1 (to match geometry labelMap)

    for fileName in vertebraFiles:

        filePath = os.path.join(segFolder, fileName)

        if not os.path.exists(filePath):
            labelValue -= 1
            continue

        nii = nib.load(filePath)
        data = nii.get_fdata()

        binaryMask = (data > 0).astype(np.int16)

        if combinedMask is None:
            combinedMask = binaryMask * labelValue
            affine = nii.affine
        else:
            combinedMask += binaryMask * labelValue

        labelValue -= 1

    combinedPath = os.path.join(segFolder, "combined_seg.nii.gz")

    nib.save(nib.Nifti1Image(combinedMask, affine), combinedPath)

    print("Unified segmentation saved.")

    return combinedPath