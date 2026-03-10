import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
from kaggle_data_loader import get_l5_bounding_box_centers, _load_data
import os

def compute_local_pedicle_pca(mask, affine, center_world, radius_mm=15.0, global_ap_axis=None):
    """
    Isolates the pedicle using a spherical bounding box and runs PCA strictly 
    on the local bone voxels to determine the true anatomical trajectory.

    Args:
        mask (np.ndarray): 3D boolean/binary mask of the L5 vertebra.
        affine (np.ndarray): 4x4 affine matrix of the NIfTI volume.
        center_world (np.ndarray): The (x, y, z) target in WORLD coordinates.
        radius_mm (float): The radius of the spherical crop in millimeters.
        global_ap_axis (np.ndarray): The general anterior-posterior axis to ensure 
                                     the final vector points forward.

    Returns:
        np.ndarray: The normalized primary axis (screw trajectory) vector.
    """
    print(f"  [Local PCA] Cropping {radius_mm}mm sphere around {np.round(center_world, 1)}...")
    
    # 1. Extract all bone voxel indices
    # nibabel loads data as (X, Y, Z), so np.where returns arrays in that order
    voxel_coords = np.column_stack(np.where(mask > 0.5))

    # 2. Convert bone voxels to physical World Coordinates
    world_coords = nib.affines.apply_affine(affine, voxel_coords)

    # 3. Calculate Euclidean distance from every bone voxel to our target center
    distances = np.linalg.norm(world_coords - center_world, axis=1)

    # 4. The Divide: Keep only bone voxels inside the 15mm sphere
    local_bone_coords = world_coords[distances <= radius_mm]
    
    print(f"  [Local PCA] Isolated {len(local_bone_coords)} voxels for pedicle analysis.")

    if len(local_bone_coords) < 10:
        print("  [Local PCA] ERROR: Bounding sphere is empty! Check coordinate mapping.")
        # Fallback to a straight AP vector if the crop fails
        return global_ap_axis if global_ap_axis is not None else np.array([0, 1, 0])

    # 5. The Conquer: Run PCA on the isolated pedicle cylinder
    pca = PCA(n_components=3)
    pca.fit(local_bone_coords)
    
    # The primary eigenvector defines the longitudinal axis of the isolated pedicle
    primary_axis = pca.components_[0]

    # 6. Vector Alignment: Ensure it points Anteriorly (toward the vertebral body)
    if global_ap_axis is not None:
        if np.dot(primary_axis, global_ap_axis) < 0:
            primary_axis = -primary_axis

    print(f"  [Local PCA] Computed Trajectory: {np.round(primary_axis, 4)}")
    return primary_axis

# ----------------- Integration Bridge ----------------- #

def run_pedicle_pipeline(nifti_path, study_id, csv_path='archive/data/coords_rsna_improved.csv'):
    """
    End-to-end pipeline mapping Kaggle CSV bounding box targets to NIfTI 
    to extract ideal PCA pedicle trajectories.
    """
    # 1. Get targets from Kaggle dataloader
    kaggle_targets = get_l5_bounding_box_centers(study_id, csv_path=csv_path)

    # Load the NIfTI segmentation mask and spatial properties
    if not os.path.exists(nifti_path):
        raise FileNotFoundError(f"NIfTI mask file not found at: {nifti_path}")
        
    img = nib.load(nifti_path)
    # Using get_fdata() loads the data into a float array
    mask = img.get_fdata()
    affine = img.affine
    
    results = {}
    
    for side_name, dicom_target in zip(["Left", "Right"], kaggle_targets):
        print(f"\n--- Processing {side_name} Pedicle ---")
        
        # 1. Adjust the Z-axis (instance_number is 1-indexed in DICOM, numpy array mapping is usually 0-indexed)
        voxel_target = np.array([
            dicom_target[0],           # x pixel
            dicom_target[1],           # y pixel
            dicom_target[2] - 1        # z slice index
        ])
        
        # 2. Transform the Voxel Array Target to Physical millimeters (World Target)
        center_world = nib.affines.apply_affine(affine, voxel_target)
        
        print(f"  Kaggle Voxel Indices: {np.round(voxel_target, 1)}")
        print(f"  NIfTI World Coord:    {np.round(center_world, 1)}")
        
        # 3. Define an assumed global AP axis (Y-axis typically corresponds to AP in NIfTI)
        global_ap_axis = np.array([0, 1, 0]) 
        
        # 4. Invoke PCA using mm coordinates against the voxel mask 
        trajectory = compute_local_pedicle_pca(
            mask=mask, 
            affine=affine, 
            center_world=center_world, 
            radius_mm=15.0, 
            global_ap_axis=global_ap_axis
        )
        
        print(f"  Final Trajectory Vector for {side_name}: {np.round(trajectory, 4)}")
        results[side_name] = trajectory
        
    return results

if __name__ == "__main__":
    # Example usage (update with actual correct paths to test)
    # study_id_test = "100200854"  # Replace with a valid ID
    # nifti_path_test = f"archive/data/segmentations/{study_id_test}_mask.nii.gz"
    # if os.path.exists(nifti_path_test):
    #     run_pedicle_pipeline(nifti_path_test, study_id_test)
    pass
