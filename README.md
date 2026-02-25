# Automatic Pedicle Screw Planning (CT-Based)

## Overview

This project implements a geometry-driven pipeline for automatic pedicle screw planning from CT scans. The system operates on segmented vertebrae and computes safe screw trajectories using PCA-based anatomical alignment and Euclidean distance transform (EDT)–based clearance analysis.

The method is fully deterministic and does not require training.

## Input

* CT volume (NIfTI)
* Vertebra segmentation mask (e.g., generated using TotalSegmentator)

## Pipeline Overview

### 1. Vertebra Segmentation

A target vertebra (e.g., L1–L5) is extracted from the segmentation mask.

The segmentation defines the bone region used for all geometric computations.

### 2. PCA-Based Anatomical Frame Estimation

Principal Component Analysis (PCA) is applied to the vertebral voxel coordinates.

This produces:

* PC1: Largest variance direction
* PC2: Second largest variance direction
* PC3: Smallest variance direction

These principal components define an orthogonal vertebra-centered coordinate system.

The Left–Right (LR) axis is identified from the PCA frame and used to split the vertebra into left and right halves.

Purpose:

* Establish a consistent anatomical reference frame
* Enable side-specific trajectory planning

### 3. Euclidean Distance Transform (EDT)

A 3D Euclidean Distance Transform is computed on the vertebral mask.

For each voxel inside the bone, EDT provides:

Distance to the nearest cortical boundary.

This distance field is used to:

* Identify interior safe regions
* Perform cortical breach checking
* Score trajectory clearance

The distance transform is computed once per vertebra and reused for all trajectory evaluations.

### 4. Candidate Interior Voxel Selection

Candidate seed voxels are selected based on:

* High EDT values (deep interior regions)
* Spatial constraints within the pedicle region

These candidates represent potential screw corridor centers.

### 5. Trajectory Optimization

For each candidate voxel:

1. Generate yaw and pitch angle variations.
2. Construct a line representing a potential screw trajectory.
3. Sample points along the trajectory.
4. Evaluate EDT values along the path.

A trajectory is considered valid if:

EDT(sampled_point) > screw_radius
for all sampled points.

This ensures the screw remains fully inside bone.

### 6. Trajectory Selection

Among all valid trajectories:

* Compute a clearance score (e.g., minimum EDT along path).
* Select the trajectory that maximizes clearance.
* Reject candidates that result in cortical breach.

### 7. Entry Point Determination

Once the optimal trajectory is selected:

* Trace the trajectory posteriorly.
* Identify the intersection with the posterior cortical surface.
* Define this intersection as the screw entry point.

## Output

For each vertebra and side:

* Entry point (3D coordinates)
* Screw axis direction vector
* Screw length
* Screw diameter
* Clearance metrics
* Optional JSON / VTK export for visualization

## Key Design Principles

* Geometry-driven (no neural network dependency)
* Deterministic and reproducible
* Anatomically aligned via PCA
* Physically validated via Euclidean distance transform
* Efficient (single EDT computation per vertebra)

## Current Limitations

* PCA axes may not perfectly align with pedicle axis in upper lumbar levels.
* Severe deformity or segmentation artifacts can affect candidate generation.
* Narrow pedicles may result in no_safe_path failures.

Future improvements may include:

* Atlas-based trajectory priors
* Multi-candidate aggregation
* AI-based axis refinement


