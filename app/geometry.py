import numpy as np
import nibabel as nib

# distance_transform_edt:
# Computes distance from each voxel inside the bone
# to the nearest outer boundary.
#
# Bigger value = thicker / safer bone region.
from scipy.ndimage import (
    distance_transform_edt,
    label as cc_label,
    map_coordinates
)

# PCA helps find the main anatomical directions
# of the vertebra automatically.
from sklearn.decomposition import PCA


# ============================================================
# GLOBAL CONSTRAINTS
# ============================================================

# Ignore very tiny noisy vertebra regions
# Smaller components are usually segmentation noise
voxelThreshold = 5000

# Step size while marching through bone (in mm)
stepMM = 0.5

# Minimum acceptable screw length
minLengthMM = 20

# Maximum allowed screw depth ratio
# Example:
# if vertebral depth = 50 mm
# max screw depth = 0.8 * 50 = 40 mm
maxDepthRatio = 0.8


# ============================================================
# AVAILABLE STANDARD SCREW DIAMETERS (mm)
# ============================================================

# Ordered from largest to smallest.
# We try larger screws first because they provide
# better mechanical fixation.
globalDiameters = [
    8.5, 7.5, 7.0, 6.5,
    6.0, 5.5, 5.0, 4.5, 4.0
]


# ============================================================
# CLINICAL DIAMETER LIMITS
# ============================================================

# Different vertebrae can safely hold
# different screw sizes.
CLINICAL_LIMITS = {

    "L1": (4.5, 6.0),
    "L2": (4.5, 6.0),
    "L3": (5.0, 6.5),
    "L4": (6.0, 7.5),
    "L5": (6.5, 8.5)
}


# ============================================================
# MAXIMUM CLINICAL SCREW LENGTHS
# ============================================================

LENGTH_LIMITS = {

    "L1": 45,
    "L2": 50,
    "L3": 50,
    "L4": 55,
    "L5": 60
}


# ============================================================
# OPTIMIZATION WEIGHTS
# ============================================================

# These weights decide what the planner prefers.
#
# Higher wDiam:
#     prefers thicker screws
#
# Higher wConv:
#     prefers inward convergence
#
# Higher wDT:
#     prefers safer bone corridors
#
# Higher wTilt:
#     penalizes excessive tilt
#
# Higher wMidline:
#     strongly penalizes crossing vertebral midline

wDiam = 15.0
wConv = 20.0
wDT = 5.0
wLen = 1.0
wTilt = 10.0
wMidline = 50.0


# ============================================================
# LABEL → VERTEBRA NAME MAPPING
# ============================================================

labelMap = {

    5: "L1",
    4: "L2",
    3: "L3",
    2: "L4",
    1: "L5"
}


# ============================================================
# LOAD NIFTI FILE
# ============================================================

def loadNifti(path):

    """
    Load segmentation volume from NIfTI file.

    Returns:
        segmentation volume
        voxel spacing
        affine matrix
    """

    nii = nib.load(path)

    return (
        nii.get_fdata(),
        nii.header.get_zooms(),
        nii.affine
    )


# ============================================================
# REMOVE SMALL NOISY COMPONENTS
# ============================================================

def getValidLabels(seg):

    """
    Keep only large valid vertebra regions.

    Why?
    Small disconnected regions are usually
    segmentation noise and can break geometry calculations.

    Steps:
    1. Extract each vertebra label
    2. Find connected components
    3. Keep largest connected region
    4. Remove tiny components
    """

    valid = []

    # Get all labels present in segmentation
    uniqueLabels = np.unique(seg)

    # Remove background label (0)
    uniqueLabels = uniqueLabels[uniqueLabels != 0]

    for labelVal in uniqueLabels:

        # Binary mask for current vertebra
        mask = (seg == labelVal)

        # Connected component labeling
        # Each disconnected region gets unique ID
        labeled, _ = cc_label(mask)

        # Count voxel count of every region
        sizes = np.bincount(labeled.ravel())

        # Ignore background
        sizes[0] = 0

        if len(sizes) == 0:
            continue

        # Largest connected component
        largest = np.argmax(sizes)

        # Keep only largest component
        component = (labeled == largest)

        # Ignore tiny noisy regions
        if np.sum(component) > voxelThreshold:

            valid.append(
                (int(labelVal), component)
            )

    return valid


# ============================================================
# BUILD STABLE ANATOMICAL COORDINATE SYSTEM
# ============================================================

def computeStableFrame(mask, affine):

    """
    Build patient-specific anatomical coordinate system.

    PCA is used to estimate:
        SI axis → Superior-Inferior
        LR axis → Left-Right
        AP axis → Anterior-Posterior

    This makes the planner independent
    of CT scan orientation.
    """

    # Get all voxel coordinates belonging to vertebra
    coords = np.argwhere(mask)

    # Convert voxel coordinates into world coordinates
    coordsWorld = nib.affines.apply_affine(
        affine,
        coords
    )

    # Compute vertebral center
    centroid = coordsWorld.mean(axis=0)

    # PCA finds main geometric directions
    pca = PCA(n_components=3)

    pca.fit(coordsWorld - centroid)

    # PCA axes
    axes = pca.components_

    # Global vertical direction
    worldZ = np.array([0,0,1])

    # Choose PCA axis most aligned with vertical direction
    siAxis = axes[
        np.argmax(np.abs(axes @ worldZ))
    ]

    # Make sure SI axis points upward
    if np.dot(siAxis, worldZ) < 0:
        siAxis = -siAxis

    # Temporary axis least aligned with vertical
    tempAxis = axes[
        np.argmin(np.abs(axes @ worldZ))
    ]

    # Remove SI component from temporary axis
    # to obtain clean left-right axis
    lrAxis = (
        tempAxis
        - np.dot(tempAxis, siAxis) * siAxis
    )

    lrAxis /= np.linalg.norm(lrAxis)

    # Cross product creates AP axis
    # perpendicular to SI and LR
    apAxis = np.cross(siAxis, lrAxis)

    apAxis /= np.linalg.norm(apAxis)

    # Estimate vertebral AP depth
    relAP = (
        (coordsWorld - centroid)
        @ apAxis
    )

    totalDepth = (
        np.percentile(relAP,95)
        - np.percentile(relAP,5)
    )

    return (
        centroid,
        np.vstack([siAxis, lrAxis, apAxis]),
        totalDepth
    )


# ============================================================
# DISTANCE TRANSFORM
# ============================================================

def computeDistance(mask, spacing):

    """
    Compute distance transform.

    For every voxel inside bone:
        value =
        distance to nearest boundary

    Larger values mean:
        thicker bone
        safer screw corridor
    """

    return distance_transform_edt(
        mask,
        sampling=spacing
    )


# ============================================================
# FIND PEDICLE CENTERS
# ============================================================

def pedicleCenters(mask, dist, centroid, axes, affine):

    """
    Estimate left and right pedicle centers.

    Strategy:
    - focus near vertebral middle
    - focus in posterior region
    - split into left/right halves
    - choose thickest bone region
    """

    # Vertebral voxel coordinates
    coords = np.argwhere(mask)

    # Convert to world coordinates
    coordsWorld = nib.affines.apply_affine(
        affine,
        coords
    )

    # Anatomical axes
    siAxis, lrAxis, apAxis = axes

    # Coordinates relative to vertebral center
    rel = coordsWorld - centroid

    # Focus near middle vertebral height
    midMask = (
        np.abs(rel @ siAxis)
        <
        np.percentile(
            np.abs(rel @ siAxis),
            35
        )
    )

    # Focus on posterior region
    posteriorMask = (
        (rel @ apAxis)
        <
        np.percentile(rel @ apAxis,40)
    )

    # Left pedicle candidate voxels
    lCoords = coords[
        midMask
        & posteriorMask
        & ((rel @ lrAxis) < 0)
    ]

    # Right pedicle candidate voxels
    rCoords = coords[
        midMask
        & posteriorMask
        & ((rel @ lrAxis) > 0)
    ]

    # If too few voxels exist,
    # pedicle detection failed
    if len(lCoords) < 20 or len(rCoords) < 20:
        return None, None

    # Choose voxel with largest DT value
    # Larger DT = thicker / safer bone region
    lCenter = nib.affines.apply_affine(

        affine,

        lCoords[
            np.argmax(
                dist[
                    lCoords[:,0],
                    lCoords[:,1],
                    lCoords[:,2]
                ]
            )
        ]
    )

    rCenter = nib.affines.apply_affine(

        affine,

        rCoords[
            np.argmax(
                dist[
                    rCoords[:,0],
                    rCoords[:,1],
                    rCoords[:,2]
                ]
            )
        ]
    )

    return lCenter, rCenter


# ============================================================
# ESTIMATE POSTERIOR ENTRY POINT
# ============================================================

def findEntry(center, axes, maskFloat, affine, side):

    """
    Estimate posterior screw entry point.

    Steps:
    1. Start near pedicle center
    2. Shift slightly sideways
    3. March backward toward posterior cortex
    4. Stop when exiting bone
    5. Move slightly back inside

    Final result:
        approximate screw entry point
    """

    siAxis, lrAxis, apAxis = axes

    # Small lateral shift
    lateralShift = -1.5 if side == "Left" else 1.5

    startP = center + (lrAxis * lateralShift)

    # March posteriorly
    direction = -apAxis

    # Convert world → voxel coordinates
    invAff = np.linalg.inv(affine)

    p = startP.copy()

    # March step-by-step
    for _ in range(100):

        vox = nib.affines.apply_affine(invAff, p)

        # Stop if outside image
        if any(v<0 or v>=s-1 for v,s in zip(vox,maskFloat.shape)):
            break

        # Interpolate segmentation value
        # <0.5 means outside bone
        if map_coordinates(
            maskFloat,
            [[vox[0]],[vox[1]],[vox[2]]],
            order=1
        )[0] < 0.5:

            break

        # Continue marching
        p += direction * stepMM

    # Move slightly back inside bone
    return p + apAxis * 1.5


# ============================================================
# EVALUATE SCREW TRAJECTORY
# ============================================================

def evaluate(
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
):

    """
    Evaluate candidate screw trajectory.

    Checks:
    - cortical containment
    - bone clearance
    - medial wall safety
    - screw diameter feasibility
    - anatomical convergence

    Returns:
        trajectory score
        safe length
        minimum clearance
        safe tip point
        screw diameter
    """

    # Normalize direction vector
    d = direction / np.linalg.norm(direction)

    siAxis, lrAxis, apAxis = axes

    invAff = np.linalg.inv(affine)

    # Reject strongly upward trajectories
    if np.dot(d, siAxis) > 0:
        return None

    t = 0

    # Minimum distance transform value
    minDT = 999

    # Counts medial crossing violations
    midlineViolation = 0

    # LR direction component
    lr_comp = np.dot(d, lrAxis)

    # Convergence direction
    conv = lr_comp if side == "Left" else -lr_comp

    t_exit = None

    # March along trajectory
    while t < maxAllowedLen:

        # Current world point
        p = entry + d*t

        # Convert to voxel coordinates
        vox = nib.affines.apply_affine(invAff,p)

        # Stop if outside image
        if any(v<0 or v>=s-1 for v,s in zip(vox,maskFloat.shape)):

            t_exit = t
            break

        # Interpolated inside/outside check
        inside = map_coordinates(

            maskFloat,

            [[vox[0]],[vox[1]],[vox[2]]],

            order=1

        )[0]

        # Stop when exiting bone
        if inside < 0.5:

            t_exit = t
            break

        # Relative vertebral position
        relP = p - centroid

        lrPos = np.dot(relP, lrAxis)

        # Detect crossing toward opposite side
        if (side=="Left" and lrPos>1.5) or (
            side=="Right" and lrPos<-1.5
        ):

            midlineViolation += 1

        # Local bone thickness
        dtVal = map_coordinates(

            dist,

            [[vox[0]],[vox[1]],[vox[2]]],

            order=1

        )[0]

        # Only evaluate main screw body region
        if 5.0 < t < 30.0:

            minDT = min(minDT,dtVal)

        t += stepMM

    if t_exit is None:
        t_exit = t

    # Safety margin before cortical exit
    safetyMargin = 3.0

    t_safe = t_exit - safetyMargin

    # Reject too-short screws
    if t_safe < minLengthMM:
        return None

    # Estimate maximum safe diameter
    max_safe = (minDT*2) - 0.5

    # Clinical vertebra-specific limits
    min_limit, max_limit = CLINICAL_LIMITS.get(
        v_name,
        (4.0,8.5)
    )

    diam = 0.0

    # Choose largest safe screw diameter
    for d_val in globalDiameters:

        if d_val <= max_safe and d_val <= max_limit:

            diam = d_val
            break

    # Penalize excessive tilt
    tilt = abs(np.dot(d,siAxis))

    # Final optimization score
    score = (

        (wDiam*diam)

        + (wConv*conv*15)

        + (wDT*minDT)

        - (wTilt*tilt)

        - (wMidline*midlineViolation)
    )

    # Final safe screw tip
    safe_tip = entry + d * t_safe

    return (
        score,
        t_safe,
        minDT,
        safe_tip,
        diam
    )


# ============================================================
# SEARCH FOR BEST TRAJECTORY
# ============================================================

def optimize(
    center,
    axes,
    side,
    maskFloat,
    dist,
    affine,
    centroid,
    totalDepth,
    v_name
):

    """
    Search for best screw trajectory.

    Strategy:
    - generate many candidate angles
    - evaluate each trajectory
    - keep highest scoring trajectory
    """

    if center is None:
        return None

    # Estimate entry point
    entry = findEntry(
        center,
        axes,
        maskFloat,
        affine,
        side
    )

    # Geometry-based depth limit
    geomLimit = totalDepth * maxDepthRatio

    # Clinical depth limit
    clinicalLimit = LENGTH_LIMITS.get(v_name, 50)

    # Final allowed screw length
    maxAllowedLen = min(
        geomLimit,
        clinicalLimit
    )

    best = None

    # Search convergence angles
    angles = (
        np.linspace(5,40,25)
        if side=="Left"
        else np.linspace(-40,-5,25)
    )

    # Search angle combinations
    for lrAng in angles:

        for siAng in np.linspace(-10,0,11):

            # Build trajectory direction vector
            direction = (

                axes[2]

                + np.tan(np.deg2rad(lrAng))*axes[1]

                + np.tan(np.deg2rad(siAng))*axes[0]
            )

            # Evaluate candidate trajectory
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

            # Keep highest scoring solution
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


# ============================================================
# MAIN PLANNING PIPELINE
# ============================================================

def run_planner(segPath):

    """
    Complete automatic pedicle screw planning pipeline.

    Steps:
    1. Load segmentation
    2. Remove noisy components
    3. Build anatomical coordinate system
    4. Compute distance transform
    5. Detect pedicle centers
    6. Optimize screw trajectories
    7. Return final screw plans
    """

    resultsList = []

    # Load segmentation volume
    seg,spacing,affine = loadNifti(segPath)

    # Extract valid vertebrae
    validSegments = getValidLabels(seg)

    # Process each vertebra
    for labelVal,mask in sorted(validSegments,reverse=True):

        name = labelMap.get(labelVal,str(labelVal))

        print(f"\nPROCESSING: {name}")

        # Build vertebral coordinate system
        centroid,axes,totalDepth = computeStableFrame(
            mask,
            affine
        )

        # Compute bone thickness map
        dist = computeDistance(mask,spacing)

        # Float version needed for interpolation
        maskFloat = mask.astype(np.float32)

        # Detect pedicle centers
        lCenter,rCenter = pedicleCenters(
            mask,
            dist,
            centroid,
            axes,
            affine
        )

        # Process left and right pedicles
        for side,center in [
            ("Left",lCenter),
            ("Right",rCenter)
        ]:

            print(f"  Searching {side} trajectory...")

            # Find optimal trajectory
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

            # Skip failed trajectories
            if res:

                (
                    score,
                    entry,
                    tip,
                    length,
                    minDT,
                    lrAng,
                    siAng,
                    diam
                ) = res

                # Store final screw result
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

                    f"  SUCCESS: "

                    f"{side} | "

                    f"Diameter={diam}mm | "

                    f"Length={round(length,1)}mm | "

                    f"Axial={round(lrAng,1)}° | "

                    f"Sagittal={round(siAng,1)}°"
                )

            else:

                print(f"  FAILED: {side}")

    print("\nPLANNING COMPLETE")

    return resultsList