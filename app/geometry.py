# import numpy as np
# import nibabel as nib
# from scipy.ndimage import distance_transform_edt, label as cc_label
# from scipy.ndimage import map_coordinates
# from sklearn.decomposition import PCA
# import os

# # Paths to CT scan and segmentation
# baseDir = os.path.dirname(os.path.abspath(__file__))
# ctPath = os.path.join(baseDir,"data","case_0000.nii")
# segPath = os.path.join(baseDir,"data","spine_2-label.nii")

# # Minimum voxels needed to accept a vertebra
# voxelThreshold = 5000

# # Possible screw diameters (mm)
# globalDiameters = [8.5,7.5,7.0,6.5,5.5,5.0,4.5,4.0]

# # Maximum screw size allowed per vertebra
# maxDiameterPerLevel = {
# "L1":6.5,
# "L2":7.0,
# "L3":7.5,
# "L4":8.5,
# "L5":8.5
# }

# # Movement step along screw path
# stepMM = 0.5

# # Minimum screw length
# minLengthMM = 18

# # Allowed direction variation
# directionConeDegLR = 35
# directionConeDegSI = 20

# # Number of directions tested
# directionSamplesLR = 13
# directionSamplesSI = 9

# # Screw scoring weights
# wDT = 5.0
# wLen = 1.5
# wTilt = 0.5

# # Segmentation label names
# labelMap = {5:"L1",4:"L2",3:"L3",2:"L4",1:"L5"}

# # Load CT or segmentation file
# def loadNifti(path):
#     nii = nib.load(path)
#     return nii.get_fdata(), nii.header.get_zooms(), nii.affine

# # Keep only real vertebra components
# def getValidLabels(seg):
#     valid = []
#     uniqueLabels = np.unique(seg)
#     uniqueLabels = uniqueLabels[uniqueLabels != 0]

#     for labelVal in uniqueLabels:
#         mask = (seg == labelVal)
#         if np.sum(mask) == 0:
#             continue
#         labeled, nc = cc_label(mask)
#         sizes = np.bincount(labeled.ravel())
#         sizes[0] = 0
#         if len(sizes) == 0:
#             continue
#         largest = np.argmax(sizes)
#         component = (labeled == largest)
#         if np.sum(component) > voxelThreshold:
#             valid.append((int(labelVal), component))
#     return valid

# # Find vertebra orientation (Up-Down, Left-Right, Front-Back)
# def computeStableFrame(mask, affine):
#     coords = np.argwhere(mask)
#     coordsWorld = nib.affines.apply_affine(affine, coords)
#     centroid = coordsWorld.mean(axis=0)
#     pca = PCA(n_components=3)
#     pca.fit(coordsWorld-centroid)
#     axes = pca.components_
#     worldZ = np.array([0,0,1])
#     siAxis = axes[np.argmax(np.abs(axes@worldZ))]
#     if np.dot(siAxis,worldZ) < 0:
#         siAxis = -siAxis
#     tempAxis = axes[np.argmin(np.abs(axes@worldZ))]
#     lrAxis = tempAxis - np.dot(tempAxis,siAxis)*siAxis
#     lrAxis /= np.linalg.norm(lrAxis)
#     apAxis = np.cross(siAxis,lrAxis)
#     apAxis /= np.linalg.norm(apAxis)
#     return centroid, np.vstack([siAxis,lrAxis,apAxis])

# # Distance from bone boundary
# # Big value = thicker bone = safer
# def computeDistance(mask, spacing):
#     return distance_transform_edt(mask, sampling=spacing)

# # Find pedicle centers on left and right
# def pedicleCenters(mask, dist, centroid, axes, affine):
#     coords = np.argwhere(mask)
#     coordsWorld = nib.affines.apply_affine(affine, coords)
#     siAxis,lrAxis,apAxis = axes
#     rel = coordsWorld-centroid
#     siVals = rel@siAxis
#     lrVals = rel@lrAxis
#     apVals = rel@apAxis
#     midMask = np.abs(siVals)<np.percentile(np.abs(siVals),35)
#     posteriorMask = apVals<np.percentile(apVals,40)
#     leftMask = lrVals<0
#     rightMask = lrVals>0
#     leftCoords = coords[midMask & posteriorMask & leftMask]
#     rightCoords = coords[midMask & posteriorMask & rightMask]
#     if len(leftCoords)<30 or len(rightCoords)<30:
#         return None,None
#     lVox = leftCoords[np.argmax(dist[leftCoords[:,0],leftCoords[:,1],leftCoords[:,2]])]
#     rVox = rightCoords[np.argmax(dist[rightCoords[:,0],rightCoords[:,1],rightCoords[:,2]])]
#     lMM = nib.affines.apply_affine(affine,lVox)
#     rMM = nib.affines.apply_affine(affine,rVox)
#     return lMM,rMM

# # Find entry point on back surface
# def findEntry(center,axes,maskFloat,affine):
#     siAxis,lrAxis,apAxis = axes
#     direction = -apAxis
#     invAff = np.linalg.inv(affine)
#     p = center.copy()
#     for _ in range(300):
#         vox = nib.affines.apply_affine(invAff,p)
#         if any(v<0 or v>=s-1 for v,s in zip(vox,maskFloat.shape)):
#             break
#         val = map_coordinates(maskFloat,[[vox[0]],[vox[1]],[vox[2]]],order=1)[0]
#         if val<0.5:
#             break
#         p += direction*stepMM
#     return p+apAxis*1.0

# # Check if screw cylinder stays inside bone
# def cylinderSafe(p,d,radius,maskFloat,affine):
#     invAff = np.linalg.inv(affine)
#     for angle in np.linspace(0,2*np.pi,8,endpoint=False):
#         offset = radius*(np.cos(angle)*np.cross(d,[0,0,1])+np.sin(angle)*np.cross(d,[1,0,0]))
#         testPoint = p+offset
#         vox = nib.affines.apply_affine(invAff,testPoint)
#         if any(v<0 or v>=s-1 for v,s in zip(vox,maskFloat.shape)):
#             return False
#         val = map_coordinates(maskFloat,[[vox[0]],[vox[1]],[vox[2]]],order=1)[0]
#         if val<0.5:
#             return False
#     return True

# # Test one screw path
# def evaluate(entry,direction,maskFloat,dist,affine,radius,axes):
#     d = direction/np.linalg.norm(direction)
#     invAff = np.linalg.inv(affine)
#     t = 0
#     minDT = 999
#     while True:
#         if t<5:
#             t+=stepMM
#             continue
#         p = entry+d*t
#         vox = nib.affines.apply_affine(invAff,p)
#         if any(v<0 or v>=s-1 for v,s in zip(vox,maskFloat.shape)):
#             break
#         maskVal = map_coordinates(maskFloat,[[vox[0]],[vox[1]],[vox[2]]],order=1)[0]
#         if maskVal<0.5:
#             break
#         dtVal = map_coordinates(dist,[[vox[0]],[vox[1]],[vox[2]]],order=1)[0]
#         minDT = min(minDT,dtVal)
#         if not cylinderSafe(p,d,radius,maskFloat,affine):
#             break
#         t+=stepMM
#     if t<minLengthMM:
#         return None
#     if minDT-radius<=0:
#         return None
#     siAxis = axes[0]
#     tilt = abs(np.dot(d,siAxis))
#     score = wDT*minDT + wLen*(t/10) - wTilt*tilt
#     return score,t,minDT,p,d

# # Try many diameters and directions
# # Keep best safe screw
# def optimize(center,axes,maskFloat,dist,affine,diameters):
#     if center is None:
#         return None
#     siAxis,lrAxis,apAxis = axes
#     entry = findEntry(center,axes,maskFloat,affine)
#     best=None
#     for diam in diameters:
#         radius=diam/2
#         for lrAng in np.linspace(-directionConeDegLR,directionConeDegLR,directionSamplesLR):
#             for siAng in np.linspace(-directionConeDegSI,directionConeDegSI,directionSamplesSI):
#                 direction=apAxis+np.tan(np.deg2rad(lrAng))*lrAxis+np.tan(np.deg2rad(siAng))*siAxis
#                 r=evaluate(entry,direction,maskFloat,dist,affine,radius,axes)
#                 if r is None:
#                     continue
#                 score,length,minDT,tip,d=r
#                 if best is None or score>best[0]:
#                     best=(score,entry,tip,length,minDT,diam)
#     return best


# # Store results for later use

# def run_planner(segPath):
#     resultsList=[]
#     print("\nFINAL ROBUST PEDICLE SCREW PLANNER:\n")
#     ct,spacingCT,affineCT=loadNifti(ctPath)
#     seg,spacing,affine=loadNifti(segPath)
#     validSegments=getValidLabels(seg)
#     for labelVal,mask in validSegments:
#         name=labelMap.get(labelVal,str(labelVal))
#         maxDiam=maxDiameterPerLevel.get(name,max(globalDiameters))
#         diameters=[d for d in globalDiameters if d<=maxDiam]

#         print("==============")
#         print(name)
#         print("==============")
#         print("Tested Diameters:",diameters)

#         centroid,axes=computeStableFrame(mask,affine)
#         dist=computeDistance(mask,spacing)
#         maskFloat=mask.astype(np.float32)
#         lCenter,rCenter=pedicleCenters(mask,dist,centroid,axes,affine)

#         for side,center in [("Left",lCenter),("Right",rCenter)]:
#             result=optimize(center,axes,maskFloat,dist,affine,diameters)

#             if result is None:
#                 print(f"{side}: NO SAFE PATH\n")
#                 continue

#             score,entry,tip,length,minDT,diam=result

#             resultsList.append({
#                 "vertebra":name,
#                 "side":side,
#                 "entry":entry,
#                 "tip":tip,
#                 "diameter":diam
#             })

#             print(f"{side} Screw Found")
#             print("Diameter:",diam,"mm")
#             print("Length:",round(length,1),"mm")
#             print("Safety Margin:",round(minDT-diam/2,2),"mm")
#             print()

#     return resultsList



import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, label as cc_label
from scipy.ndimage import map_coordinates
from sklearn.decomposition import PCA


# ==============================
# CONFIGURATION
# ==============================

voxelThreshold = 5000

globalDiameters = [8.5,7.5,7.0,6.5,5.5,5.0,4.5,4.0]

maxDiameterPerLevel = {
    "L1":6.5,
    "L2":7.0,
    "L3":7.5,
    "L4":8.5,
    "L5":8.5
}

stepMM = 0.5
minLengthMM = 18

directionConeDegLR = 35
directionConeDegSI = 20

directionSamplesLR = 13
directionSamplesSI = 9

wDT = 5.0
wLen = 1.5
wTilt = 0.5

labelMap = {5:"L1",4:"L2",3:"L3",2:"L4",1:"L5"}


# ==============================
# UTILITIES
# ==============================

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

    lrAxis = tempAxis - np.dot(tempAxis, siAxis)*siAxis
    lrAxis /= np.linalg.norm(lrAxis)

    apAxis = np.cross(siAxis, lrAxis)
    apAxis /= np.linalg.norm(apAxis)

    return centroid, np.vstack([siAxis, lrAxis, apAxis])


def computeDistance(mask, spacing):
    return distance_transform_edt(mask, sampling=spacing)


def pedicleCenters(mask, dist, centroid, axes, affine):

    coords = np.argwhere(mask)
    coordsWorld = nib.affines.apply_affine(affine, coords)

    siAxis, lrAxis, apAxis = axes
    rel = coordsWorld - centroid

    siVals = rel @ siAxis
    lrVals = rel @ lrAxis
    apVals = rel @ apAxis

    midMask = np.abs(siVals) < np.percentile(np.abs(siVals),35)
    posteriorMask = apVals < np.percentile(apVals,40)

    leftMask = lrVals < 0
    rightMask = lrVals > 0

    leftCoords = coords[midMask & posteriorMask & leftMask]
    rightCoords = coords[midMask & posteriorMask & rightMask]

    if len(leftCoords) < 30 or len(rightCoords) < 30:
        return None, None

    lVox = leftCoords[np.argmax(dist[leftCoords[:,0],leftCoords[:,1],leftCoords[:,2]])]
    rVox = rightCoords[np.argmax(dist[rightCoords[:,0],rightCoords[:,1],rightCoords[:,2]])]

    lMM = nib.affines.apply_affine(affine, lVox)
    rMM = nib.affines.apply_affine(affine, rVox)

    return lMM, rMM


def findEntry(center, axes, maskFloat, affine):

    siAxis, lrAxis, apAxis = axes
    direction = -apAxis

    invAff = np.linalg.inv(affine)
    p = center.copy()

    for _ in range(300):

        vox = nib.affines.apply_affine(invAff, p)

        if any(v < 0 or v >= s-1 for v,s in zip(vox, maskFloat.shape)):
            break

        val = map_coordinates(maskFloat, [[vox[0]],[vox[1]],[vox[2]]], order=1)[0]

        if val < 0.5:
            break

        p += direction * stepMM

    return p + apAxis*1.0


def cylinderSafe(p, d, radius, maskFloat, affine):

    invAff = np.linalg.inv(affine)

    for angle in np.linspace(0,2*np.pi,8,endpoint=False):

        offset = radius * (
            np.cos(angle)*np.cross(d,[0,0,1]) +
            np.sin(angle)*np.cross(d,[1,0,0])
        )

        testPoint = p + offset
        vox = nib.affines.apply_affine(invAff, testPoint)

        if any(v < 0 or v >= s-1 for v,s in zip(vox, maskFloat.shape)):
            return False

        val = map_coordinates(maskFloat, [[vox[0]],[vox[1]],[vox[2]]], order=1)[0]

        if val < 0.5:
            return False

    return True


def evaluate(entry, direction, maskFloat, dist, affine, radius, axes):

    d = direction / np.linalg.norm(direction)
    invAff = np.linalg.inv(affine)

    t = 0
    minDT = 999

    while True:

        if t < 5:
            t += stepMM
            continue

        p = entry + d*t
        vox = nib.affines.apply_affine(invAff, p)

        if any(v < 0 or v >= s-1 for v,s in zip(vox, maskFloat.shape)):
            break

        maskVal = map_coordinates(maskFloat, [[vox[0]],[vox[1]],[vox[2]]], order=1)[0]
        if maskVal < 0.5:
            break

        dtVal = map_coordinates(dist, [[vox[0]],[vox[1]],[vox[2]]], order=1)[0]
        minDT = min(minDT, dtVal)

        if not cylinderSafe(p, d, radius, maskFloat, affine):
            break

        t += stepMM

    if t < minLengthMM:
        return None

    if minDT - radius <= 0:
        return None

    siAxis = axes[0]
    tilt = abs(np.dot(d, siAxis))

    score = wDT*minDT + wLen*(t/10) - wTilt*tilt

    return score, t, minDT, p


def optimize(center, axes, maskFloat, dist, affine, diameters):

    if center is None:
        return None

    siAxis, lrAxis, apAxis = axes
    entry = findEntry(center, axes, maskFloat, affine)

    best = None

    for diam in diameters:

        radius = diam / 2

        for lrAng in np.linspace(-directionConeDegLR, directionConeDegLR, directionSamplesLR):
            for siAng in np.linspace(-directionConeDegSI, directionConeDegSI, directionSamplesSI):

                direction = (
                    apAxis
                    + np.tan(np.deg2rad(lrAng)) * lrAxis
                    + np.tan(np.deg2rad(siAng)) * siAxis
                )

                r = evaluate(entry, direction, maskFloat, dist, affine, radius, axes)

                if r is None:
                    continue

                score, length, minDT, tip = r

                if best is None or score > best[0]:
                    best = (score, entry, tip, length, minDT, diam)

    return best


# ==============================
# MAIN PLANNER
# ==============================

def run_planner(segPath):

    resultsList = []

    print("\nGEOMETRY BASED PEDICLE SCREW PLANNER:\n")

    seg, spacing, affine = loadNifti(segPath)

    validSegments = getValidLabels(seg)

    for labelVal, mask in validSegments:

        name = labelMap.get(labelVal, str(labelVal))

        maxDiam = maxDiameterPerLevel.get(name, max(globalDiameters))
        diameters = [d for d in globalDiameters if d <= maxDiam]

        print("==============")
        print(name)
        print("==============")
        print("Tested Diameters:", diameters)

        centroid, axes = computeStableFrame(mask, affine)
        dist = computeDistance(mask, spacing)
        maskFloat = mask.astype(np.float32)

        lCenter, rCenter = pedicleCenters(mask, dist, centroid, axes, affine)

        for side, center in [("Left",lCenter), ("Right",rCenter)]:

            if center is None:
                print(f"{side}: NO SAFE PATH\n")
                continue

            result = optimize(center, axes, maskFloat, dist, affine, diameters)

            if result is None:
                print(f"{side}: NO SAFE PATH\n")
                continue

            score, entry, tip, length, minDT, diam = result

            resultsList.append({
                "vertebra": name,
                "side": side,
                "entry": entry,
                "tip": tip,
                "diameter": diam
            })

            print(f"{side} Screw Found")
            print("Diameter:", diam, "mm")
            print("Length:", round(length,1), "mm")
            print("Safety Margin:", round(minDT - diam/2,2), "mm")
            print()

    return resultsList