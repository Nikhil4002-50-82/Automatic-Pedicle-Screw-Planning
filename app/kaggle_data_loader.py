import pandas as pd
import numpy as np
import os

_DATA_CACHE = None

def _load_data(csv_path='archive/data/coords_rsna_improved.csv'):
    """
    Loads and preprocesses the coordinate CSV file.
    Filters by the relevant levels and conditions, and caches the result.
    """
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Coordinate CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate that necessary columns exist
    required_cols = ['study_id', 'level', 'condition', 'x', 'y', 'instance_number']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the CSV.")

    # Filter for L5 associated levels (e.g., L4/L5, L5/S1)
    valid_levels = ['L4/L5', 'L5/S1']
    df = df[df['level'].isin(valid_levels)].copy()

    if df.empty:
        raise ValueError("No rows found for valid L5 levels ('L4/L5', 'L5/S1').")

    # Filter for conditions that flank the pedicle
    valid_conditions = [
        'Left Subarticular Stenosis',
        'Right Subarticular Stenosis',
        'Left Neural Foraminal Narrowing',
        'Right Neural Foraminal Narrowing'
    ]
    df = df[df['condition'].isin(valid_conditions)].copy()

    if df.empty:
        raise ValueError("No rows found matching the required Subarticular or Neural Foraminal conditions.")

    # Ensure coordinate types are numeric
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['instance_number'] = pd.to_numeric(df['instance_number'], errors='coerce')  # maps to Z-axis

    # Drop any rows with NaN in critical coordinate columns
    df = df.dropna(subset=['x', 'y', 'instance_number', 'study_id'])

    _DATA_CACHE = df
    return _DATA_CACHE

def get_l5_bounding_box_centers(study_id, csv_path='archive/data/coords_rsna_improved.csv'):
    """
    Takes a study_id and returns the estimated [left_target, right_target] 
    3D voxel coordinates based on the averaged positions of the Subarticular 
    and Neural Foraminal points for that side.
    
    These targets can act as center points for robust 3D spherical cropping.
    
    Args:
        study_id: The identifier for a patient's study volume.
        csv_path (str): The path to the RSNA Kaggle coordinate CSV dataset.
        
    Returns:
        list of np.ndarray: A list containing two NumPy arrays for the left 
                            and right targets respectively in (x, y, z) format.
    """
    df = _load_data(csv_path)

    # Filter dataframe for the requested study_id
    study_df = df[df['study_id'] == study_id]
    
    if study_df.empty:
        raise ValueError(f"study_id '{study_id}' not found or lacks relevant L5 landmarks in the dataset.")

    # Define the conditions for both sides
    left_conditions = ['Left Subarticular Stenosis', 'Left Neural Foraminal Narrowing']
    right_conditions = ['Right Subarticular Stenosis', 'Right Neural Foraminal Narrowing']

    left_df = study_df[study_df['condition'].isin(left_conditions)]
    right_df = study_df[study_df['condition'].isin(right_conditions)]

    # Validate that we have points for both sides to compute the bounding box centers safely
    if left_df.empty:
        raise ValueError(f"No valid left-side L5 landmarks found for study_id '{study_id}'.")
    if right_df.empty:
        raise ValueError(f"No valid right-side L5 landmarks found for study_id '{study_id}'.")

    # Compute averages of the coordinates for each side to find the uncorruptible center points
    left_target = np.array([
        left_df['x'].mean(),
        left_df['y'].mean(),
        left_df['instance_number'].mean()
    ])

    right_target = np.array([
        right_df['x'].mean(),
        right_df['y'].mean(),
        right_df['instance_number'].mean()
    ])

    return [left_target, right_target]
