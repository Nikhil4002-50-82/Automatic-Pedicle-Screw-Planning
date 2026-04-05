#!/usr/bin/env python
"""
Test script to measure distance_transform_edt performance
on full volume vs cropped bounding box.
"""

import time
import numpy as np
from scipy.ndimage import distance_transform_edt

# Simulate L5-only file (small volume)
print("=" * 70)
print("SCENARIO 1: L5-only file (small volume)")
print("=" * 70)
small_shape = (256, 256, 300)  # L5 only bounding box
small_mask = np.random.rand(*small_shape) > 0.95  # Sparse mask
small_mask_count = np.count_nonzero(small_mask)
print(f"Volume shape: {small_shape}")
print(f"Voxel count: {small_mask_count}")

start = time.perf_counter()
small_dist = distance_transform_edt(small_mask)
elapsed_small = time.perf_counter() - start
print(f"distance_transform_edt time: {elapsed_small:.2f} seconds\n")

# Simulate L1-L5 file (FULL volume, NOT cropped)
print("=" * 70)
print("SCENARIO 2: L1-L5 file (FULL volume - current bottleneck)")
print("=" * 70)
full_shape = (512, 512, 1147)  # L1-L5 full bounding box
full_mask = np.zeros(full_shape, dtype=bool)
# Create L5-sized region in the lower part
full_mask[128:384, 128:384, 850:1150] = np.random.rand(256, 256, 300) > 0.95
full_mask_count = np.count_nonzero(full_mask)
print(f"Volume shape: {full_shape}")
print(f"Voxel count: {full_mask_count} (same L5 voxels, larger volume)")

start = time.perf_counter()
full_dist = distance_transform_edt(full_mask)
elapsed_full = time.perf_counter() - start
print(f"distance_transform_edt time: {elapsed_full:.2f} seconds\n")

# Simulate OPTIMIZED approach (crop to bounding box)
print("=" * 70)
print("SCENARIO 3: L1-L5 file (CROPPED bounding box - optimized)")
print("=" * 70)
# Find crop bounds
where = np.argwhere(full_mask)
if len(where) > 0:
    min_coords = where.min(axis=0)
    max_coords = where.max(axis=0)
    cropped_shape = tuple(max_coords - min_coords + 1)
    cropped_mask = full_mask[
        min_coords[0]:max_coords[0]+1,
        min_coords[1]:max_coords[1]+1,
        min_coords[2]:max_coords[2]+1
    ]
    cropped_count = np.count_nonzero(cropped_mask)
    print(f"Volume shape: {cropped_shape} (cropped from {full_shape})")
    print(f"Voxel count: {cropped_count} (same)")
    
    start = time.perf_counter()
    cropped_dist = distance_transform_edt(cropped_mask)
    elapsed_cropped = time.perf_counter() - start
    print(f"distance_transform_edt time: {elapsed_cropped:.2f} seconds\n")

# Summary
print("=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)
print(f"L5-only (baseline):        {elapsed_small:.2f}s")
print(f"L1-L5 full volume:         {elapsed_full:.2f}s ({elapsed_full/elapsed_small:.1f}x slower)")
print(f"L1-L5 cropped (optimized): {elapsed_cropped:.2f}s ({elapsed_cropped/elapsed_small:.1f}x)")
print(f"\nSpeedup from cropping: {elapsed_full/elapsed_cropped:.1f}x")
print(f"Estimated wait time saved: {(elapsed_full - elapsed_cropped)/60:.1f} minutes")
