import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, label as cc_label
from scipy.ndimage import map_coordinates
from sklearn.decomposition import PCA

# Minimum number of voxels required to accept a vertebra
voxelThreshold = 5000

# Possible screw diameters (in mm)
globalDiameters = [8.5,7.5,7.0,6.5,5.5,5.0,4.5,4.0]

# Maximum allowed screw size for each vertebra
maxDiameterPerLevel = {
    "L1":6.5,"L2":7.0,"L3":7.5,"L4":8.5,"L5":8.5
}

# Step size when moving inside bone (in mm)
stepMM = 0.5

# Minimum screw length required (in mm)
minLengthMM = 18

# Allowed angle variation (degrees)
directionConeDegLR = 35
directionConeDegSI = 20

# Number of angle samples to test
directionSamplesLR = 13
directionSamplesSI = 9

# Weights used to calculate best screw score
wDT = 5.0
wLen = 1.5
wTilt = 0.5

# Map segmentation labels to vertebra names
labelMap = {5:"L1",4:"L2",3:"L3",2:"L4",1:"L5"}

# L5-specific tuning on top of the geometry.py baseline
L5MedialConeDeg = (4,30)
L5DirectionRangeSI = (-5,15)
L5MaxMidlineCrossMM = 2.0

def loadNifti(path):
    # Load 3D medical image
    nii = nib.load(path)
    return nii.get_fdata(), nii.header.get_zooms(), nii.affine

def getValidLabels(seg):
    # Find all vertebra labels except background
    valid = []
    uniqueLabels = np.unique(seg)
    uniqueLabels = uniqueLabels[uniqueLabels != 0]
    for labelVal in uniqueLabels:
        mask = (seg == labelVal)
        labeled,_ = cc_label(mask)
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        if len(sizes) == 0:
            continue
        largest = np.argmax(sizes)
        component = (labeled == largest)
        # Accept only large enough vertebra parts
        if np.sum(component) > voxelThreshold:
            valid.append((int(labelVal),component))
    return valid

def computeStableFrame(mask,affine):
    # Get 3D coordinates of vertebra
    coords = np.argwhere(mask)
    coordsWorld = nib.affines.apply_affine(affine,coords)
    centroid = coordsWorld.mean(axis=0)
    # Use PCA to find main directions of bone
    pca = PCA(n_components=3)
    pca.fit(coordsWorld-centroid)
    axes = pca.components_
    worldZ = np.array([0,0,1])
    # Find vertical axis (head to toe)
    siAxis = axes[np.argmax(np.abs(axes@worldZ))]
    if np.dot(siAxis,worldZ) < 0:
        siAxis = -siAxis
    tempAxis = axes[np.argmin(np.abs(axes@worldZ))]
    # Find left-right axis
    lrAxis = tempAxis - np.dot(tempAxis,siAxis)*siAxis
    lrAxis /= np.linalg.norm(lrAxis)
    # Find front-back axis
    apAxis = np.cross(siAxis,lrAxis)
    apAxis /= np.linalg.norm(apAxis)
    return centroid,np.vstack([siAxis,lrAxis,apAxis])



def computeStableFrameL5(mask,affine,dist):
    # L5 has large transverse processes that can corrupt whole-bone PCA.
    # Anchor the frame in the vertebral body and run PCA on a local body region.
    coords = np.argwhere(mask)
    coordsWorld = nib.affines.apply_affine(affine,coords)
    max_idx = np.unravel_index(np.argmax(dist), dist.shape)
    centroid = nib.affines.apply_affine(affine,max_idx)
    bodyRadiusMM = 22.0
    bodyMask = np.linalg.norm(coordsWorld-centroid, axis=1) < bodyRadiusMM
    bodyCoordsWorld = coordsWorld[bodyMask]
    if len(bodyCoordsWorld) < 100:
        bodyCoordsWorld = coordsWorld
    pca = PCA(n_components=3)
    pca.fit(bodyCoordsWorld-centroid)
    axes = pca.components_
    worldZ = np.array([0,0,1])
    siAxis = axes[np.argmax(np.abs(axes@worldZ))]
    if np.dot(siAxis,worldZ) < 0:
        siAxis = -siAxis
    tempAxis = axes[np.argmin(np.abs(axes@worldZ))]
    lrAxis = tempAxis - np.dot(tempAxis,siAxis)*siAxis
    lrAxis /= np.linalg.norm(lrAxis)
    apAxis = np.cross(siAxis,lrAxis)
    apAxis /= np.linalg.norm(apAxis)
    bodyRel = bodyCoordsWorld-centroid
    bodyAp = bodyRel@apAxis
    bodyLr = bodyRel@lrAxis
    anteriorMask = bodyAp > np.percentile(bodyAp,75)
    posteriorMask = bodyAp < np.percentile(bodyAp,25)
    if np.any(anteriorMask) and np.any(posteriorMask):
        anteriorWidth = np.mean(np.abs(bodyLr[anteriorMask]))
        posteriorWidth = np.mean(np.abs(bodyLr[posteriorMask]))
        if anteriorWidth < posteriorWidth:
            apAxis = -apAxis
    # Whole-vertebra COM is pulled posterior by the posterior elements.
    # Enforce AP so it points from the posterior-heavy whole COM toward the body center.
    wholeCentroid = coordsWorld.mean(axis=0)
    if np.dot(apAxis, centroid - wholeCentroid) < 0:
        apAxis = -apAxis
    return centroid,np.vstack([siAxis,lrAxis,apAxis])

def computeDistance(mask,spacing):
    # Compute distance from bone boundary
    return distance_transform_edt(mask,sampling=spacing)

def pedicleCenters(mask,dist,centroid,axes,affine):
    # Find best center point inside left and right pedicles
    coords = np.argwhere(mask)
    coordsWorld = nib.affines.apply_affine(affine,coords)
    siAxis,lrAxis,apAxis = axes
    rel = coordsWorld-centroid
    siVals = rel@siAxis
    lrVals = rel@lrAxis
    apVals = rel@apAxis
    midMask = np.abs(siVals)<np.percentile(np.abs(siVals),35)
    posteriorMask = apVals<np.percentile(apVals,40)
    leftMask = lrVals<0
    rightMask = lrVals>0
    leftCoords = coords[midMask&posteriorMask&leftMask]
    rightCoords = coords[midMask&posteriorMask&rightMask]
    if len(leftCoords)<30 or len(rightCoords)<30:
        return None,None
    # Choose point with maximum thickness
    lVox = leftCoords[np.argmax(dist[leftCoords[:,0],leftCoords[:,1],leftCoords[:,2]])]
    rVox = rightCoords[np.argmax(dist[rightCoords[:,0],rightCoords[:,1],rightCoords[:,2]])]
    lMM = nib.affines.apply_affine(affine,lVox)
    rMM = nib.affines.apply_affine(affine,rVox)
    return lMM,rMM

def pedicleCentersL5(mask,dist,centroid,axes,affine):
    # For isolated L5, the pedicle center should sit near the pedicle-body junction,
    # not at the extreme posterior cortex or transverse process.
    coords = np.argwhere(mask)
    coordsWorld = nib.affines.apply_affine(affine,coords)
    siAxis,lrAxis,apAxis = axes
    rel = coordsWorld-centroid
    siVals = rel@siAxis
    lrVals = rel@lrAxis
    apVals = rel@apAxis

    siMask = np.abs(siVals) < np.percentile(np.abs(siVals),35)
    posteriorBand = apVals < np.percentile(apVals,45)
    anteriorBand = apVals > np.percentile(apVals,20)
    leftMask = lrVals < 0
    rightMask = lrVals > 0

    targetAp = np.percentile(apVals,35)
    targetAbsLr = np.percentile(np.abs(lrVals),55)

    def choose(sideMask, sideSign):
        sideCoords = coords[siMask & posteriorBand & anteriorBand & sideMask]
        if len(sideCoords) < 30:
            return None
        sideWorld = nib.affines.apply_affine(affine,sideCoords)
        sideRel = sideWorld - centroid
        sideDist = dist[sideCoords[:,0],sideCoords[:,1],sideCoords[:,2]]
        sideSi = np.abs(sideRel@siAxis)
        sideLr = sideRel@lrAxis
        sideAp = sideRel@apAxis
        score = (
            1.0 * sideDist
            - 0.05 * np.abs(sideAp - targetAp)
            - 0.04 * np.abs(np.abs(sideLr) - targetAbsLr)
            - 0.02 * sideSi
        )
        chosen = sideCoords[np.argmax(score)]
        return nib.affines.apply_affine(affine,chosen)

    lMM = choose(leftMask, -1.0)
    rMM = choose(rightMask, 1.0)
    if lMM is None or rMM is None:
        return pedicleCenters(mask,dist,centroid,axes,affine)
    return lMM,rMM

def findEntry(center,axes,maskFloat,affine):
    # Move from center toward back to find entry point
    siAxis,lrAxis,apAxis = axes
    direction = -apAxis
    invAff = np.linalg.inv(affine)
    p = center.copy()
    for _ in range(300):
        vox = nib.affines.apply_affine(invAff,p)
        if any(v<0 or v>=s-1 for v,s in zip(vox,maskFloat.shape)):
            break
        val = map_coordinates(maskFloat,[[vox[0]],[vox[1]],[vox[2]]],order=1)[0]
        if val<0.5:
            break
        p += direction*stepMM
    return p+apAxis*1.0

def cylinderSafe(p,d,radius,maskFloat,affine):
    # Check if screw cylinder stays inside bone
    invAff = np.linalg.inv(affine)
    for angle in np.linspace(0,2*np.pi,8,endpoint=False):
        offset = radius*(np.cos(angle)*np.cross(d,[0,0,1])+
                         np.sin(angle)*np.cross(d,[1,0,0]))
        testPoint = p+offset
        vox = nib.affines.apply_affine(invAff,testPoint)
        if any(v<0 or v>=s-1 for v,s in zip(vox,maskFloat.shape)):
            return False
        val = map_coordinates(maskFloat,[[vox[0]],[vox[1]],[vox[2]]],order=1)[0]
        if val<0.5:
            return False
    return True

def evaluate(entry,direction,maskFloat,dist,affine,radius,axes):
    # Check how long and safe a screw path is
    d = direction/np.linalg.norm(direction)
    invAff = np.linalg.inv(affine)
    t = 0
    minDT = 999
    while True:
        if t<5:
            t += stepMM
            continue
        p = entry+d*t
        vox = nib.affines.apply_affine(invAff,p)
        if any(v<0 or v>=s-1 for v,s in zip(vox,maskFloat.shape)):
            break
        maskVal = map_coordinates(maskFloat,[[vox[0]],[vox[1]],[vox[2]]],order=1)[0]
        if maskVal<0.5:
            break
        dtVal = map_coordinates(dist,[[vox[0]],[vox[1]],[vox[2]]],order=1)[0]
        minDT = min(minDT,dtVal)
        if not cylinderSafe(p,d,radius,maskFloat,affine):
            break
        t += stepMM
    # Reject if too short or unsafe
    if t<minLengthMM:
        return None
    if minDT-radius<=0:
        return None
    siAxis = axes[0]
    tilt = abs(np.dot(d,siAxis))
    # Final score calculation
    score = wDT*minDT + wLen*(t/10) - wTilt*tilt
    return score,t,minDT,p

def optimize(center,axes,maskFloat,dist,affine,diameters,lrRange=None,siRange=None):
    # Try all diameters and angles to find best screw
    if center is None:
        return None
    siAxis,lrAxis,apAxis = axes
    entry = findEntry(center,axes,maskFloat,affine)
    best = None
    if lrRange is None:
        lrRange = (-directionConeDegLR,directionConeDegLR)
    if siRange is None:
        siRange = (-directionConeDegSI,directionConeDegSI)
    for diam in diameters:
        radius = diam/2
        for lrAng in np.linspace(lrRange[0],lrRange[1],directionSamplesLR):
            for siAng in np.linspace(siRange[0],siRange[1],directionSamplesSI):
                direction = (apAxis +
                             np.tan(np.deg2rad(lrAng))*lrAxis +
                             np.tan(np.deg2rad(siAng))*siAxis)
                r = evaluate(entry,direction,maskFloat,dist,affine,radius,axes)
                if r is None:
                    continue
                score,length,minDT,tip = r
                if best is None or score>best[0]:
                    best = (score,entry,tip,length,minDT,diam)
    return best

def optimizeL5(center,centroid,axes,maskFloat,dist,affine,diameters):
    # Use the same entry-point calculation as geometry.py and derive mediality from the center itself.
    lrAxis = axes[1]
    centerLr = np.dot(center-centroid,lrAxis)
    if centerLr == 0:
        return None,None
    if centerLr < 0:
        lrRange = L5MedialConeDeg
        sideSign = -1.0
    else:
        lrRange = (-L5MedialConeDeg[1],-L5MedialConeDeg[0])
        sideSign = 1.0
    result = optimize(
        center,
        axes,
        maskFloat,
        dist,
        affine,
        diameters,
        lrRange=lrRange,
        siRange=L5DirectionRangeSI
    )
    if result is None:
        return None,None
    score,entry,tip,length,minDT,diam = result
    entryLr = np.dot(entry-centroid,lrAxis)
    tipLr = np.dot(tip-centroid,lrAxis)
    rejectReason = None
    if sideSign*entryLr < 0:
        rejectReason = "entry crossed to wrong side"
    elif sideSign*tipLr < -L5MaxMidlineCrossMM:
        rejectReason = "tip crossed too far past midline"
    if rejectReason is not None:
        return None,(result,rejectReason)
    return result,None

def debug_l5_geometry(side, centroid, axes, center, entry, tip, prefix="Accepted"):
    siAxis,lrAxis,apAxis = axes
    centerRel = center-centroid
    entryRel = entry-centroid
    tipRel = tip-centroid
    print(f"{side} {prefix} Debug:")
    print("  Center:", np.round(center,2))
    print("  Entry :", np.round(entry,2))
    print("  Tip   :", np.round(tip,2))
    print("  Center(AP,LR,SI):", round(np.dot(centerRel,apAxis),2), round(np.dot(centerRel,lrAxis),2), round(np.dot(centerRel,siAxis),2))
    print("  Entry (AP,LR,SI):", round(np.dot(entryRel,apAxis),2), round(np.dot(entryRel,lrAxis),2), round(np.dot(entryRel,siAxis),2))
    print("  Tip   (AP,LR,SI):", round(np.dot(tipRel,apAxis),2), round(np.dot(tipRel,lrAxis),2), round(np.dot(tipRel,siAxis),2))
    print("  Dir   (AP,LR,SI):", round(np.dot(tip-entry,apAxis),2), round(np.dot(tip-entry,lrAxis),2), round(np.dot(tip-entry,siAxis),2))
    print()

def run_planner(segPath):
    # Main function that plans screws for all vertebrae
    resultsList = []
    print("GEOMETRY BASED PEDICLE SCREW PLANNER (L5 OPTIMIZED)")
    seg,spacing,affine = loadNifti(segPath)
    validSegments = getValidLabels(seg)
    for labelVal,mask in validSegments:
        name = labelMap.get(labelVal,str(labelVal))
        maxDiam = maxDiameterPerLevel.get(name,max(globalDiameters))
        diameters = [d for d in globalDiameters if d<=maxDiam]
        print(name)
        print("Tested Diameters:",diameters)
        dist = computeDistance(mask,spacing)
        if name == "L5":
            centroid,axes = computeStableFrameL5(mask,affine,dist)
        else:
            centroid,axes = computeStableFrame(mask,affine)
        maskFloat = mask.astype(np.float32)
        if name == "L5":
            lCenter,rCenter = pedicleCentersL5(mask,dist,centroid,axes,affine)
            print("L5 Frame Debug:")
            print("  Centroid:", np.round(centroid,2))
            print("  SI Axis :", np.round(axes[0],4))
            print("  LR Axis :", np.round(axes[1],4))
            print("  AP Axis :", np.round(axes[2],4))
            print("  Left Center :", None if lCenter is None else np.round(lCenter,2))
            print("  Right Center:", None if rCenter is None else np.round(rCenter,2))
            print()
        else:
            lCenter,rCenter = pedicleCenters(mask,dist,centroid,axes,affine)
        for side,center in [("Left",lCenter),("Right",rCenter)]:
            if center is None:
                print(side+": NO SAFE PATH")
                continue
            if name == "L5":
                result,rejected = optimizeL5(center,centroid,axes,maskFloat,dist,affine,diameters)
                if rejected is not None:
                    rejectedResult,rejectReason = rejected
                    _,rejEntry,rejTip,rejLength,rejMinDT,rejDiam = rejectedResult
                    debug_l5_geometry(side,centroid,axes,center,rejEntry,rejTip,prefix="Rejected")
                    print(f"  Reject Reason: {rejectReason}")
                    print(f"  Candidate Diameter: {rejDiam} mm, Length: {round(rejLength,1)} mm, Safety Margin: {round(rejMinDT-rejDiam/2,2)} mm")
                    print()
            else:
                result = optimize(center,axes,maskFloat,dist,affine,diameters)
                rejected = None
            if result is None:
                print(side+": NO SAFE PATH")
                continue
            score,entry,tip,length,minDT,diam = result
            if name == "L5":
                debug_l5_geometry(side,centroid,axes,center,entry,tip)
            resultsList.append({
                "vertebra":name,
                "side":side,
                "entry":entry,
                "tip":tip,
                "diameter":diam
            })
            print(side,"Screw Found")
            print("Diameter:",diam,"mm")
            print("Length:",round(length,1),"mm")
            print("Safety Margin:",round(minDT-diam/2,2),"mm")
            print()
    return resultsList
