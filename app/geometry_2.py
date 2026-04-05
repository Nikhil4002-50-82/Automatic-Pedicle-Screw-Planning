import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, label as cc_label
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import gc
import time
import os

# --- Global Constraints (EXACTLY as original) ---
voxelThreshold = 5000
stepMM = 0.5
minLengthMM = 20
maxDepthRatio = 0.8
globalDiameters = [8.5, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0]

CLINICAL_LIMITS = {
    "L1": (4.5, 6.0), "L2": (4.5, 6.0), "L3": (5.0, 6.5),
    "L4": (6.0, 7.5), "L5": (6.5, 8.5)
}
LENGTH_LIMITS = {"L1": 45, "L2": 50, "L3": 50, "L4": 55, "L5": 60}

wDiam, wConv, wDT, wLen, wTilt, wMidline = 15.0, 20.0, 5.0, 1.0, 10.0, 50.0
labelMap = {5: "L1", 4: "L2", 3: "L3", 2: "L4", 1: "L5"}

def loadNifti(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii.header.get_zooms(), nii.affine

def getValidLabels(seg):
    valid = []
    uniqueLabels = np.unique(seg)
    uniqueLabels = uniqueLabels[uniqueLabels != 0]
    for labelVal in uniqueLabels:
        mask = (seg == labelVal)
        labeled, _ = cc_label(mask)
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        if len(sizes) == 0: continue
        largest = np.argmax(sizes)
        component = (labeled == largest)
        if np.sum(component) > voxelThreshold:
            valid.append((int(labelVal), component))
    return valid

def computeStableFrame(mask, affine):
    coords = np.argwhere(mask)
    coordsWorld = nib.affines.apply_affine(affine, coords)
    centroid = coordsWorld.mean(axis=0)
    pca = PCA(n_components=3)
    pca.fit(coordsWorld - centroid)
    axes = pca.components_
    worldZ = np.array([0,0,1])
    siAxis = axes[np.argmax(np.abs(axes @ worldZ))]
    if np.dot(siAxis, worldZ) < 0: siAxis = -siAxis
    tempAxis = axes[np.argmin(np.abs(axes @ worldZ))]
    lrAxis = tempAxis - np.dot(tempAxis, siAxis) * siAxis
    lrAxis /= np.linalg.norm(lrAxis)
    apAxis = np.cross(siAxis, lrAxis)
    apAxis /= np.linalg.norm(apAxis)
    relAP = (coordsWorld - centroid) @ apAxis
    totalDepth = np.percentile(relAP, 95) - np.percentile(relAP, 5)
    return centroid, np.vstack([siAxis, lrAxis, apAxis]), totalDepth

def computeDistance(mask, spacing):
    return distance_transform_edt(mask, sampling=spacing)

def pedicleCenters(mask, dist, centroid, axes, affine):
    coords = np.argwhere(mask)
    coordsWorld = nib.affines.apply_affine(affine, coords)
    siAxis, lrAxis, apAxis = axes
    rel = coordsWorld - centroid
    midMask = np.abs(rel @ siAxis) < np.percentile(np.abs(rel @ siAxis), 35)
    posteriorMask = (rel @ apAxis) < np.percentile(rel @ apAxis, 40)
    lCoords = coords[midMask & posteriorMask & ((rel @ lrAxis) < 0)]
    rCoords = coords[midMask & posteriorMask & ((rel @ lrAxis) > 0)]
    if len(lCoords) < 20 or len(rCoords) < 20: return None, None
    lCenter = nib.affines.apply_affine(affine, lCoords[np.argmax(dist[lCoords[:,0], lCoords[:,1], lCoords[:,2]])])
    rCenter = nib.affines.apply_affine(affine, rCoords[np.argmax(dist[rCoords[:,0], rCoords[:,1], rCoords[:,2]])])
    return lCenter, rCenter

# --- GPU ACCELERATED CORE (BATCH DIMENSION FIXED) ---
def optimize_gpu(center, axes, side, maskFloat, dist, affine, centroid, totalDepth, v_name, device):
    if center is None:
        return None

    # 1. Entry point (CPU, unchanged)
    siAxis, lrAxis, apAxis = axes
    lateralShift = -1.5 if side == "Left" else 1.5
    startP = center + (lrAxis * lateralShift)
    direction = -apAxis
    invAff = np.linalg.inv(affine)
    p = startP.copy()
    for _ in range(100):
        vox = nib.affines.apply_affine(invAff, p)
        if any(v < 0 or v >= s - 1 for v, s in zip(vox, maskFloat.shape)):
            break
        p += direction * stepMM
    entry = p + apAxis * 1.5

    # 2. Limits & Ray Generation
    geomLimit = totalDepth * maxDepthRatio
    clinicalLimit = LENGTH_LIMITS.get(v_name, 50)
    maxAllowedLen = min(geomLimit, clinicalLimit)
    t_vals = np.arange(0, maxAllowedLen, stepMM)
    n_steps = len(t_vals)

    angles = np.linspace(5, 40, 25) if side == "Left" else np.linspace(-40, -5, 25)
    siAngles = np.linspace(-10, 0, 11)
    lrGrid, siGrid = np.meshgrid(angles, siAngles, indexing='ij')
    lrGrid, siGrid = lrGrid.ravel(), siGrid.ravel()

    dirs = (axes[2][np.newaxis, :] +
            np.tan(np.deg2rad(lrGrid))[:, np.newaxis] * axes[1][np.newaxis, :] +
            np.tan(np.deg2rad(siGrid))[:, np.newaxis] * axes[0][np.newaxis, :])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    n_rays = len(dirs)

    # Points in world coordinates
    points = entry[np.newaxis, np.newaxis, :] + dirs[:, np.newaxis, :] * t_vals[np.newaxis, :, np.newaxis]
    voxels = nib.affines.apply_affine(invAff, points.reshape(-1, 3)).reshape(n_rays, n_steps, 3)

    # 3. Move everything to GPU (correct data types)
    t_vals_t = torch.tensor(t_vals, dtype=torch.float32, device=device)
    voxels_t = torch.tensor(voxels, dtype=torch.float32, device=device)
    dirs_t = torch.tensor(dirs, dtype=torch.float32, device=device)
    centroid_t = torch.tensor(centroid, dtype=torch.float32, device=device)
    axes_t = torch.tensor(axes, dtype=torch.float32, device=device)
    points_t = torch.tensor(points, dtype=torch.float32, device=device)

    # --- FIX: Convert shape to GPU tensor before division ---
    shape = np.array(maskFloat.shape, dtype=np.float32)
    shape_t = torch.tensor(shape, dtype=torch.float32, device=device)

    # Grid for F.grid_sample: (1, n_rays, 1, n_steps, 3)
    grid_t = ((voxels_t / (shape_t - 1)) * 2.0 - 1.0).unsqueeze(0).unsqueeze(2)

    # Volumes: (1, 1, D, H, W)
    mask_t = torch.from_numpy(maskFloat).unsqueeze(0).unsqueeze(0).to(device)
    dist_t = torch.from_numpy(dist).unsqueeze(0).unsqueeze(0).to(device)

    # 4. Single sampling call
    mask_vals = F.grid_sample(mask_t, grid_t, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze()
    dist_vals = F.grid_sample(dist_t, grid_t, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze()

    # 5. Vectorized scoring (unchanged logic, now all tensors on GPU)
    valid_bounds = torch.all((voxels_t >= 0) & (voxels_t < shape_t - 1), dim=2)
    inside = mask_vals >= 0.5
    valid = valid_bounds & inside

    t_exit = torch.full((n_rays,), t_vals[-1], dtype=torch.float32, device=device)
    first_invalid = (~valid).int().argmax(dim=1)
    has_invalid = (~valid).any(dim=1)
    t_exit[has_invalid] = t_vals_t[first_invalid[has_invalid]]

    before_exit = t_vals_t.unsqueeze(0) < t_exit.unsqueeze(1)

    relP = points_t - centroid_t
    lrPos = relP @ axes_t[1]
    if side == "Left":
        violations = lrPos > 1.5
    else:
        violations = lrPos < -1.5
    midlineViolation = (violations & valid & before_exit).sum(dim=1).float()

    dt_range = (t_vals_t.unsqueeze(0) > 5.0) & (t_vals_t.unsqueeze(0) < 30.0)
    valid_dt = dt_range & valid & before_exit
    minDT = torch.full((n_rays,), 999.0, device=device)
    has_dt = valid_dt.any(dim=1)
    if has_dt.any():
        masked_dist = dist_vals.clone()
        masked_dist[~valid_dt] = 999.0
        minDT[has_dt] = masked_dist[has_dt].min(dim=1).values

    t_safe = t_exit - 3.0
    valid_len = t_safe >= minLengthMM

    max_safe = (minDT * 2) - 0.5
    min_limit, max_limit = CLINICAL_LIMITS.get(v_name, (4.0, 8.5))

    diam = torch.zeros(n_rays, device=device)
    for d_val in globalDiameters:
        diam[(d_val <= max_safe) & (d_val <= max_limit) & (diam == 0)] = d_val

    lr_comp = dirs_t @ axes_t[1]
    conv = lr_comp if side == "Left" else -lr_comp
    tilt = torch.abs(dirs_t @ axes_t[0])

    score = (wDiam * diam) + (wConv * conv * 15) + (wDT * minDT) - (wTilt * tilt) - (wMidline * midlineViolation)
    score[~valid_len] = -torch.inf

    best_idx = torch.argmax(score)
    if score[best_idx] == -torch.inf:
        return None

    return (score[best_idx].item(), entry,
            entry + dirs[best_idx].cpu().numpy() * t_safe[best_idx].item(),
            t_safe[best_idx].item(), minDT[best_idx].item(),
            angles[best_idx // 11], siAngles[best_idx % 11], diam[best_idx].item())

if __name__ == "__main__":
    # Example:
    results = run_planner_gpu(r"C:\Users\heman\Downloads\sub-verse823_dir-iso_ct_segmented_vertebrae_L5_vertebrae_L1.nii.gz")