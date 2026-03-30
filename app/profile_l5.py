#!/usr/bin/env python
"""
Profiling script to identify performance bottlenecks
when loading L1-L5 vs L5-only segmentations.
"""

import time
import sys
import os
import numpy as np
from pathlib import Path

# Add app directory to path
sys.path.insert(0, os.path.dirname(__file__))

def profile_step(name, func, *args, **kwargs):
    """Wrap a function call with timing."""
    print(f"\n[PROFILE] Starting: {name}")
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"[PROFILE] Completed: {name} in {elapsed:.2f}s")
    return result, elapsed

def profile_l5_pipeline(segmented_file):
    """Profile each step of the L5 planning pipeline."""
    from analytical_geometry import (
        loadNifti,
        getValidLabels,
        computeStableFrameL5,
        computeDistance,
        pedicleCentersL5,
    )
    
    timings = {}
    
    # Step 1: Load NIfTI
    (seg, spacing, affine), timings['load_nifti'] = profile_step(
        "Load NIfTI file",
        loadNifti,
        segmented_file
    )
    print(f"  - Volume shape: {seg.shape}, dtype: {seg.dtype}")
    print(f"  - Unique labels: {np.unique(seg)}")
    
    # Step 2: Get valid labels
    validSegments, timings['get_valid_labels'] = profile_step(
        "Get valid labels + connected components",
        getValidLabels,
        seg
    )
    print(f"  - Found {len(validSegments)} valid segments")
    for label_val, mask in validSegments:
        voxel_count = np.count_nonzero(mask)
        print(f"    - Label {label_val}: {voxel_count} voxels")
    
    # Step 3: Extract first segment
    labelVal, mask = validSegments[0]
    print(f"\n[PROFILE] Using segment: label={labelVal}")
    print(f"  - Mask shape: {mask.shape}, voxel count: {np.count_nonzero(mask)}")
    
    # Step 4: Compute distance transform
    dist, timings['distance_transform'] = profile_step(
        "Distance transform (EDT)",
        computeDistance,
        mask,
        spacing
    )
    print(f"  - Distance transform computed, min/max: {np.min(dist):.2f}/{np.max(dist):.2f}")
    
    # Step 5: Compute stable frame
    (centroid, axes), timings['stable_frame'] = profile_step(
        "Compute stable L5 frame (PCA)",
        computeStableFrameL5,
        mask,
        affine,
        dist
    )
    print(f"  - Centroid: {centroid}")
    
    # Step 6: Locate pedicle centers
    result, timings['pedicle_centers'] = profile_step(
        "Locate pedicle centers",
        pedicleCentersL5,
        mask,
        dist,
        centroid,
        axes,
        affine
    )
    print(f"  - Pedicle localization result: {result is not None}")
    
    return timings

def main():
    # Find test data
    data_dir = Path(__file__).parent / "data"
    
    # Look for both L5-only and L1-L5 files
    l5_only = None
    l1_l5 = None
    
    for nii_file in data_dir.glob("*.nii"):
        name = nii_file.name.lower()
        if "l5" in name:
            l5_only = nii_file
        if "l1" in name or "multi" in name:
            l1_l5 = nii_file
    
    if l5_only is None:
        print("ERROR: No L5 test file found in app/data/")
        print("Available files:")
        for f in data_dir.glob("*.nii"):
            print(f"  - {f.name}")
        sys.exit(1)
    
    print("=" * 70)
    print("L5-ONLY SEGMENTATION PROFILE")
    print("=" * 70)
    
    timings_l5 = profile_l5_pipeline(str(l5_only))
    
    if l1_l5:
        print("\n" + "=" * 70)
        print("L1-L5 SEGMENTATION PROFILE")
        print("=" * 70)
        
        timings_multi = profile_l5_pipeline(str(l1_l5))
        
        # Compare
        print("\n" + "=" * 70)
        print("PERFORMANCE COMPARISON")
        print("=" * 70)
        print(f"{'Step':<30} {'L5-only':<12} {'L1-L5':<12} {'Ratio':<8}")
        print("-" * 62)
        
        total_l5 = 0
        total_multi = 0
        
        for key in sorted(timings_l5.keys()):
            t_l5 = timings_l5[key]
            t_multi = timings_multi.get(key, 0)
            ratio = t_multi / t_l5 if t_l5 > 0 else 0
            
            total_l5 += t_l5
            total_multi += t_multi
            
            print(f"{key:<30} {t_l5:<12.3f}s {t_multi:<12.3f}s {ratio:<8.1f}x")
        
        print("-" * 62)
        print(f"{'TOTAL':<30} {total_l5:<12.3f}s {total_multi:<12.3f}s {total_multi/total_l5:<8.1f}x")
        
        if total_multi > total_l5 * 2:
            print("\n⚠️  L1-L5 processing is >2x slower!")
            print("   Likely culprit: getValidLabels (connected component analysis)")
            print("   Recommendation: Extract L5 only from L1-L5 files before processing")


if __name__ == "__main__":
    main()
