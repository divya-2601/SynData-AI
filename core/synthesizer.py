"""
SynData AI - Synthesizer Engine
Connects to real CTGAN and TVAE generative models.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import torch

try:
    from ctgan import CTGAN, TVAE
except ImportError:
    CTGAN = None
    TVAE = None

def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifies numerical and discrete/categorical columns for generative modeling.
    """
    discrete_columns = []
    numerical_columns = []
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category' or df[col].dtype == 'bool':
            discrete_columns.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # If numeric has very few unique values (< 5 and integer-like), consider whether discrete
            if df[col].nunique() <= 3 and pd.api.types.is_integer_dtype(df[col]):
                discrete_columns.append(col)
            else:
                numerical_columns.append(col)
        else:
            discrete_columns.append(col)
            
    return discrete_columns, numerical_columns

def preprocess_for_training(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handles missing values and prepares data types for CTGAN/TVAE.
    """
    clean_df = df.copy()
    meta = {"original_dtypes": {col: str(df[col].dtype) for col in df.columns}}
    
    for col in clean_df.columns:
        if clean_df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(clean_df[col]):
                clean_df[col] = clean_df[col].fillna(clean_df[col].median())
            else:
                mode_val = clean_df[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else "Missing"
                clean_df[col] = clean_df[col].fillna(fill_val)
                
    return clean_df, meta

def train_and_generate(
    df: pd.DataFrame,
    model_type: str = "CTGAN",
    epochs: int = 30,
    batch_size: int = 200,
    num_samples: int = 500,
    random_seed: int = 42,
    progress_callback=None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trains a real CTGAN or TVAE model and generates synthetic tabular records.
    """
    if CTGAN is None or TVAE is None:
        raise ImportError("CTGAN library is not installed.")

    start_time = time.time()
    
    # Set seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    clean_df, meta = preprocess_for_training(df)
    discrete_cols, num_cols = detect_column_types(clean_df)
    
    # Adjust batch size: CTGAN requires batch_size % 10 == 0 and <= len(df)
    n_rows = len(clean_df)
    safe_batch = min(batch_size, n_rows)
    safe_batch = max(10, (safe_batch // 10) * 10)
    if safe_batch == 0:
        safe_batch = 10

    if progress_callback:
        progress_callback(0.15, f"Initializing {model_type} architecture...")
        
    if model_type.upper() == "CTGAN":
        model = CTGAN(
            epochs=epochs,
            batch_size=safe_batch,
            verbose=False
        )
    elif model_type.upper() == "TVAE":
        model = TVAE(
            epochs=epochs,
            batch_size=safe_batch
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Expected 'CTGAN' or 'TVAE'.")

    if progress_callback:
        progress_callback(0.35, f"Fitting generative model ({epochs} epochs)...")

    # Fit real model
    model.fit(clean_df, discrete_columns=discrete_cols)
    
    if progress_callback:
        progress_callback(0.80, f"Generating {num_samples} synthetic records...")
        
    synthetic_df = model.sample(num_samples)
    
    # Post-process types to match original types
    for col in num_cols:
        if col in synthetic_df.columns and pd.api.types.is_integer_dtype(df[col]):
            synthetic_df[col] = np.round(synthetic_df[col]).astype(int)

    train_duration = time.time() - start_time
    
    if progress_callback:
        progress_callback(1.0, "Generation complete!")

    execution_info = {
        "model_type": model_type.upper(),
        "epochs": epochs,
        "batch_size": safe_batch,
        "num_samples": num_samples,
        "training_time_seconds": round(train_duration, 2),
        "discrete_columns": discrete_cols,
        "numerical_columns": num_cols,
        "random_seed": random_seed
    }
    
    return synthetic_df, execution_info
