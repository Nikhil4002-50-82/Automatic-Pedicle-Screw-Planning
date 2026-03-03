import os
import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes
def build_vertebra_mesh(segmentation_path):
    # Create smooth 3D mesh from vertebra files
    print("Building Smooth Vertebra Surface Mesh...")
    vertebraFiles=[
        "vertebrae_L1.nii.gz",
        "vertebrae_L2.nii.gz",
        "vertebrae_L3.nii.gz",
        "vertebrae_L4.nii.gz",
        "vertebrae_L5.nii.gz"
    ]
    combinedMask=None
    affine=None
    # Load and combine vertebra masks
    for fileName in vertebraFiles:
        filePath=os.path.join(segmentation_path,fileName)
        if not os.path.exists(filePath):
            continue
        nii=nib.load(filePath)
        data=nii.get_fdata()
        if combinedMask is None:
            combinedMask=data
            affine=nii.affine
        else:
            combinedMask+=data
    # Stop if no files found
    if combinedMask is None:
        raise ValueError("No vertebra segmentation files found.")
    # Create surface mesh
    verts,faces,_,_=marching_cubes(combinedMask,level=0.5)
    # Convert to real-world coordinates
    vertsWorld=nib.affines.apply_affine(affine,verts)
    print("Mesh Ready")
    return vertsWorld,faces