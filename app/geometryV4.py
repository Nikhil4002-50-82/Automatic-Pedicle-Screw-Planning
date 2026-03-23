import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, label as cc_label, map_coordinates
from sklearn.decomposition import PCA

# --- Global Constraints ---
voxelThreshold = 5000
stepMM = 0.5
minLengthMM = 20 
maxDepthRatio = 0.8 

# Standard Lumbar Hardware (mm)
globalDiameters = [8.5, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0]

# --- Clinical Diameter Standards ---
CLINICAL_LIMITS = {
    "L1": (4.5, 6.0),
    "L2": (4.5, 6.0),
    "L3": (5.0, 6.5),
    "L4": (6.0, 7.5),
    "L5": (6.5, 8.5)
}

# --- FINAL BALANCED WEIGHTS ---
wDiam = 15.0
wConv = 20.0
wDT = 5.0
wLen = 1.0
wTilt = 10.0
wMidline = 50.0

labelMap = {5:"L1", 4:"L2", 3:"L3", 2:"L4", 1:"L5"}

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

        if len(sizes) == 0:
            continue

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

    if np.dot(siAxis, worldZ) < 0:
        siAxis = -siAxis

    tempAxis = axes[np.argmin(np.abs(axes @ worldZ))]

    lrAxis = tempAxis - np.dot(tempAxis, siAxis) * siAxis
    lrAxis /= np.linalg.norm(lrAxis)

    apAxis = np.cross(siAxis, lrAxis)
    apAxis /= np.linalg.norm(apAxis)

    relAP = (coordsWorld - centroid) @ apAxis

    totalDepth = np.percentile(relAP,95) - np.percentile(relAP,5)

    return centroid, np.vstack([siAxis, lrAxis, apAxis]), totalDepth


def computeDistance(mask, spacing):

    return distance_transform_edt(mask, sampling=spacing)


def pedicleCenters(mask, dist, centroid, axes, affine):

    coords = np.argwhere(mask)

    coordsWorld = nib.affines.apply_affine(affine, coords)

    siAxis, lrAxis, apAxis = axes

    rel = coordsWorld - centroid

    midMask = np.abs(rel @ siAxis) < np.percentile(np.abs(rel @ siAxis),35)

    posteriorMask = (rel @ apAxis) < np.percentile(rel @ apAxis,40)

    lCoords = coords[midMask & posteriorMask & ((rel @ lrAxis) < 0)]

    rCoords = coords[midMask & posteriorMask & ((rel @ lrAxis) > 0)]

    if len(lCoords) < 20 or len(rCoords) < 20:
        return None, None

    lCenter = nib.affines.apply_affine(
        affine,
        lCoords[np.argmax(dist[lCoords[:,0],lCoords[:,1],lCoords[:,2]])]
    )

    rCenter = nib.affines.apply_affine(
        affine,
        rCoords[np.argmax(dist[rCoords[:,0],rCoords[:,1],rCoords[:,2]])]
    )

    return lCenter, rCenter


def findEntry(center, axes, maskFloat, affine, side):

    siAxis, lrAxis, apAxis = axes

    lateralShift = -1.5 if side == "Left" else 1.5

    startP = center + (lrAxis * lateralShift)

    direction = -apAxis

    invAff = np.linalg.inv(affine)

    p = startP.copy()

    for _ in range(100):

        vox = nib.affines.apply_affine(invAff, p)

        if any(v<0 or v>=s-1 for v,s in zip(vox,maskFloat.shape)):
            break

        if map_coordinates(maskFloat,[[vox[0]],[vox[1]],[vox[2]]],order=1)[0] < 0.5:
            break

        p += direction * stepMM

    return p + apAxis * 1.5


def evaluate(entry, direction, side, maskFloat, dist, affine, axes, centroid, maxAllowedLen, v_name):

    d = direction / np.linalg.norm(direction)

    siAxis, lrAxis, apAxis = axes

    invAff = np.linalg.inv(affine)

    if np.dot(d, siAxis) > 0:
        return None

    t = 0
    minDT = 999
    midlineViolation = 0

    lr_comp = np.dot(d, lrAxis)

    conv = lr_comp if side == "Left" else -lr_comp

    while t < maxAllowedLen:

        p = entry + d*t

        vox = nib.affines.apply_affine(invAff,p)

        if any(v<0 or v>=s-1 for v,s in zip(vox,maskFloat.shape)):
            break

        if map_coordinates(maskFloat,[[vox[0]],[vox[1]],[vox[2]]],order=1)[0] < 0.5:
            break

        relP = p - centroid

        lrPos = np.dot(relP, lrAxis)

        if (side=="Left" and lrPos>1.5) or (side=="Right" and lrPos<-1.5):
            midlineViolation += 1

        dtVal = map_coordinates(dist,[[vox[0]],[vox[1]],[vox[2]]],order=1)[0]

        if 5.0 < t < 30.0:
            minDT = min(minDT,dtVal)

        t += stepMM

    if t < minLengthMM:
        return None

    max_safe = (minDT*2) - 0.5

    min_limit, max_limit = CLINICAL_LIMITS.get(v_name,(4.0,8.5))

    diam = 0.0

    for d_val in globalDiameters:
        if d_val <= max_safe and d_val <= max_limit:
            diam = d_val
            break

    tilt = abs(np.dot(d,siAxis))

    score = (
        (wDiam*diam)
        + (wConv*conv*15)
        + (wDT*minDT)
        - (wTilt*tilt)
        - (wMidline*midlineViolation)
    )

    return score,t,minDT,p,diam


def optimize(center, axes, side, maskFloat, dist, affine, centroid, totalDepth, v_name):

    if center is None:
        return None

    entry = findEntry(center, axes, maskFloat, affine, side)

    maxAllowedLen = totalDepth * maxDepthRatio

    best = None

    angles = np.linspace(5,40,25) if side=="Left" else np.linspace(-40,-5,25)

    # ---- NEW: FORCE TRAJECTORY THROUGH PEDICLE CENTER ----
    baseDir = center - entry
    baseDir = baseDir / np.linalg.norm(baseDir)

    for lrAng in angles:

        for siAng in np.linspace(-10,0,11):

            direction = (
                baseDir
                + np.tan(np.deg2rad(lrAng))*axes[1]
                + np.tan(np.deg2rad(siAng))*axes[0]
            )

            res = evaluate(
                entry,
                direction,
                side,
                maskFloat,
                dist,
                affine,
                axes,
                centroid,
                maxAllowedLen,
                v_name
            )

            if res and (best is None or res[0] > best[0]):

                best = (
                    res[0],
                    entry,
                    res[3],
                    res[1],
                    res[2],
                    lrAng,
                    siAng,
                    res[4]
                )

    return best


def run_planner(segPath):

    resultsList = []

    seg,spacing,affine = loadNifti(segPath)

    validSegments = getValidLabels(seg)

    for labelVal,mask in sorted(validSegments,reverse=True):

        name = labelMap.get(labelVal,str(labelVal))

        centroid,axes,totalDepth = computeStableFrame(mask,affine)

        dist = computeDistance(mask,spacing)

        maskFloat = mask.astype(np.float32)

        lCenter,rCenter = pedicleCenters(mask,dist,centroid,axes,affine)

        for side,center in [("Left",lCenter),("Right",rCenter)]:

            res = optimize(
                center,
                axes,
                side,
                maskFloat,
                dist,
                affine,
                centroid,
                totalDepth,
                name
            )

            if res:

                score,entry,tip,length,minDT,lrAng,siAng,diam = res

                resultsList.append({
                    "vertebra":name,
                    "side":side,
                    "entry":entry,
                    "tip":tip,
                    "diameter":diam,
                    "length":length,
                    "axial_angle":lrAng,
                    "sagittal_angle":siAng
                })

                print(
                    f"{name} {side}: Diam {diam}mm, "
                    f"Axial {round(lrAng,1)}°, "
                    f"Sagittal {round(siAng,1)}°"
                )

    return resultsList