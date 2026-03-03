# import os
# import nibabel as nib
# import numpy as np
# from skimage.measure import marching_cubes


# # Build a smooth 3D surface of the vertebrae
# def build_vertebra_mesh(segmentation_path,
#                         vertebra_names=None,
#                         single_file=False):
#     """
#     Builds a smooth 3D mesh from vertebra segmentation.

#     Parameters:
#         segmentation_path : folder path OR file path
#         vertebra_names : list of filenames (if folder mode)
#         single_file : True if segmentation is one multi-label file

#     Returns:
#         vertsWorld, faces
#     """

#     print("Building Smooth Vertebra Surface Mesh...")

#     if single_file:
#         # If friend gives one multi-label segmentation
#         nii = nib.load(segmentation_path)
#         mask = nii.get_fdata()
#         affine = nii.affine

#     else:
#         # Folder mode (TotalSegmentator style)
#         if vertebra_names is None:
#             vertebra_names = [
#                 "vertebrae_L1.nii.gz",
#                 "vertebrae_L2.nii.gz",
#                 "vertebrae_L3.nii.gz",
#                 "vertebrae_L4.nii.gz",
#                 "vertebrae_L5.nii.gz"
#             ]

#         mask = None
#         affine = None

#         for name in vertebra_names:
#             file_path = os.path.join(segmentation_path, name)

#             if not os.path.exists(file_path):
#                 continue

#             nii = nib.load(file_path)
#             data = nii.get_fdata()

#             if mask is None:
#                 mask = data
#                 affine = nii.affine
#             else:
#                 mask += data

#         if mask is None:
#             raise ValueError("No segmentation files found.")

#     # Create a smooth surface mesh from the mask
#     verts, faces, _, _ = marching_cubes(mask, level=0.5)

#     # Convert mesh points from voxel space to real world coordinates
#     vertsWorld = nib.affines.apply_affine(affine, verts)

#     print("Mesh Ready")

#     return vertsWorld, faces

import os
import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes


def build_vertebra_mesh(segmentation_path):
    """
    Builds a smooth 3D surface mesh from vertebra segmentation folder.

    Parameters:
        segmentation_path : folder containing vertebrae_L1-L5 files

    Returns:
        vertsWorld, faces
    """

    print("Building Smooth Vertebra Surface Mesh...")

    vertebraFiles = [
        "vertebrae_L1.nii.gz",
        "vertebrae_L2.nii.gz",
        "vertebrae_L3.nii.gz",
        "vertebrae_L4.nii.gz",
        "vertebrae_L5.nii.gz"
    ]

    combinedMask = None
    affine = None

    for fileName in vertebraFiles:

        filePath = os.path.join(segmentation_path, fileName)

        if not os.path.exists(filePath):
            continue

        nii = nib.load(filePath)
        data = nii.get_fdata()

        if combinedMask is None:
            combinedMask = data
            affine = nii.affine
        else:
            combinedMask += data

    if combinedMask is None:
        raise ValueError("No vertebra segmentation files found.")

    # Create surface mesh
    verts, faces, _, _ = marching_cubes(combinedMask, level=0.5)

    # Convert to real world coordinates
    vertsWorld = nib.affines.apply_affine(affine, verts)

    print("Mesh Ready")

    return vertsWorld, faces