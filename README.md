# Pedicle Screw Planning using CT Scans

## Overview

This project implements a **geometry-based pedicle screw planning system** for lumbar vertebrae using CT scan data.

The system automatically:

* Segments lumbar vertebrae (L1–L5)
* Detects pedicle centers
* Computes safe screw trajectories
* Selects optimal screw diameter and length
* Calculates safety margins
* Generates a 3D surgical-style visualization

The algorithm is fully automatic and works directly on **3D CT scan data (NIfTI format).**

This project is intended for **research and educational purposes.**

## Features

### Automatic Vertebra Segmentation

Lumbar vertebrae (L1–L5) are segmented using:

* **TotalSegmentator**

Each vertebra is extracted separately and used for planning.

### Automatic Pedicle Detection

The system automatically finds:

* Left pedicle center
* Right pedicle center

It selects the **thickest bone region** inside each pedicle.

### Automatic Screw Planning

For each vertebra (L1–L5), the algorithm:

* Tests multiple screw diameters
* Tests multiple directions
* Computes screw length
* Checks if screw stays inside bone
* Calculates safety margin

Then selects the **best safe screw trajectory.**

### Safety Verification

The planner ensures:

* Screw cylinder remains inside bone
* Minimum screw length requirement
* Bone thickness is sufficient
* Cortical breach is avoided

Safety margin is calculated as:

Safety Margin = Bone Thickness − Screw Radius

### 3D Visualization

The system produces a 3D surgical-style visualization showing:

* Lumbar vertebrae surface
* Pedicle screws
* Entry points

Visualization is generated using:

* Plotly 3D rendering

## Technologies Used

### Programming Language

* Python 3.10+

### Libraries

#### Medical Imaging

* nibabel
* TotalSegmentator

#### Scientific Computing

* numpy
* scipy
* scikit-learn

#### 3D Processing

* scikit-image

#### Visualization

* plotly

## Project Pipeline

### Step 1 — Input CT Scan

Input must be a CT scan in **NIfTI format (.nii or .nii.gz)**.

Example:

```
case_0000.nii
```

### Step 2 — Vertebra Segmentation

TotalSegmentator extracts:

* L1
* L2
* L3
* L4
* L5

Output example:

```
vertebrae_L1.nii.gz
vertebrae_L2.nii.gz
vertebrae_L3.nii.gz
vertebrae_L4.nii.gz
vertebrae_L5.nii.gz
```

### Step 3 — Coordinate System Estimation

The system automatically estimates vertebra orientation:

* Superior–Inferior axis
* Left–Right axis
* Anterior–Posterior axis

This allows the planner to work on rotated spines.

### Step 4 — Pedicle Center Detection

The algorithm:

1. Finds middle vertebra region
2. Selects posterior region
3. Splits left and right
4. Chooses thickest bone region

### Step 5 — Screw Optimization

The system:

* Tests many directions
* Tests many diameters
* Computes safety margin
* Computes screw length

Best screw is selected automatically.

### Step 6 — Surface Mesh Generation

Marching Cubes algorithm is used to generate a smooth 3D mesh of vertebrae.

### Step 7 — Visualization

Final output includes:

* Vertebra surface
* Screw cylinders
* Entry points

## How to Run Locally

### Step 1 — Install Python

Install Python 3.10 or newer.

Check version:

```
python --version
```

### Step 2 — Create Virtual Environment

Create environment:

```
python -m venv pedicle_env
```

Activate environment:

#### Windows

```
pedicle_env\Scripts\activate
```

#### Linux / Mac

```
source pedicle_env/bin/activate
```

### Step 3 — Install Required Packages

Run:

```
pip install numpy
pip install nibabel
pip install scipy
pip install scikit-learn
pip install scikit-image
pip install plotly
pip install totalsegmentator
```

Or install everything at once:

```
pip install numpy nibabel scipy scikit-learn scikit-image plotly totalsegmentator
```

### Step 4 — Activate TotalSegmentator License

Before running segmentation:

```
export TOTALSEG_LICENSE=""
```

Windows:

```
set TOTALSEG_LICENSE=
```

### Step 5 — Prepare Dataset

Place CT scan:

```
SpineData/
 └── case_002/
      └── case_0000.nii
```

### Step 6 — Run the Program

Run:

```
python main.py
```

The program will:

1. Segment vertebrae
2. Plan screws
3. Generate visualization

## Input Requirements

### CT Scan Format

* NIfTI (.nii or .nii.gz)

### Required Region

Lumbar spine:

* L1
* L2
* L3
* L4
* L5

## Output

### Console Output

Example:

```
L4
Left Screw Found
Diameter: 6.5 mm
Length: 33 mm
Safety Margin: 1.2 mm
```

### Visualization Output

Interactive 3D model showing:

* Vertebrae
* Screws
* Entry points

## Project Structure Example

```
project/
│
├── main.py
├── README.md
│
└── SpineData/
    └── case_002/
         └── case_0000.nii
```

## Important Notes

* Not intended for clinical use yet
* Results depend on segmentation quality
* CT scans must include lumbar vertebrae
* If TotalSegmentator fails with **CUDA out of memory**, force CPU mode via `PEDICLE_TOTALSEG_DEVICE=cpu` (Windows: `set PEDICLE_TOTALSEG_DEVICE=cpu`, Linux/Mac: `export PEDICLE_TOTALSEG_DEVICE=cpu`)
