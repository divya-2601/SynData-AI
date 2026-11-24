"""
SynData AI - Privacy Risk Analysis Engine
Evaluates memorization risks, nearest neighbor distances, and proximity metrics.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Any, Tuple

def encode_and_scale_data(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Standardizes numerical features and one-hot encodes categorical features
    consistently across real and synthetic datasets for Euclidean distance calculation.
    """
    common_cols = [c for c in real_df.columns if c in syn_df.columns]
    
    # Fill missing values
    r_clean = real_df[common_cols].copy()
    s_clean = syn_df[common_cols].copy()
    
    for c in common_cols:
        if pd.api.types.is_numeric_dtype(r_clean[c]):
            med = r_clean[c].median()
            r_clean[c] = r_clean[c].fillna(med)
            s_clean[c] = s_clean[c].fillna(med)
        else:
            mode = r_clean[c].mode()[0] if not r_clean[c].mode().empty else "Unknown"
            r_clean[c] = r_clean[c].fillna(mode).astype(str)
            s_clean[c] = s_clean[c].fillna(mode).astype(str)
            
    # Combine for unified one-hot encoding
    combined = pd.concat([r_clean.assign(is_syn=0), s_clean.assign(is_syn=1)], ignore_index=True)
    combined_encoded = pd.get_dummies(combined.drop(columns=['is_syn']), drop_first=True)
    
    # Scale all features to [0, 1]
    scaler = MinMaxScaler()
    scaled_matrix = scaler.fit_transform(combined_encoded)
    
    real_scaled = scaled_matrix[combined['is_syn'] == 0]
    syn_scaled = scaled_matrix[combined['is_syn'] == 1]
    
    return real_scaled, syn_scaled, list(combined_encoded.columns)

def compute_privacy_metrics(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    suspicious_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Calculates Nearest Neighbor Distances (NND) between synthetic and real records.
    """
    real_scaled, syn_scaled, feature_names = encode_and_scale_data(real_df, syn_df)
    
    # Fit NearestNeighbors on real data
    # Normalizing Euclidean distance by sqrt(number of features) so distance is scaled between [0, 1]
    n_features = max(1, real_scaled.shape[1])
    norm_factor = np.sqrt(n_features)
    
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
    nn.fit(real_scaled)
    
    distances, indices = nn.kneighbors(syn_scaled)
    # Normalize distances
    normalized_distances = (distances.flatten() / norm_factor)
    
    min_nnd = float(np.min(normalized_distances))
    avg_nnd = float(np.mean(normalized_distances))
    med_nnd = float(np.median(normalized_distances))
    
    # Suspiciously close records: distance <= threshold (or exact match if 0)
    exact_matches = int(np.sum(normalized_distances < 1e-4))
    suspicious_count = int(np.sum(normalized_distances <= suspicious_threshold))
    total_syn = len(syn_df)
    suspicious_pct = (suspicious_count / total_syn) * 100.0 if total_syn > 0 else 0.0
    
    # Assess Privacy Risk Level
    if exact_matches > 0 or suspicious_pct > 5.0 or min_nnd < 0.02:
        risk_level = "High Privacy Risk"
        risk_color = "#DC4545"
        risk_badge = "badge-error"
        risk_summary = (
            "Model shows potential signs of memorization. Some synthetic records are "
            "virtually identical or extremely close to real individuals in feature space."
        )
    elif suspicious_pct > 1.0 or min_nnd < 0.05:
        risk_level = "Moderate Privacy Risk"
        risk_color = "#E5A12B"
        risk_badge = "badge-warning"
        risk_summary = (
            "Moderate proximity detected. Most records are safely perturbed, but a small subset "
            "falls within close proximity to real data training points."
        )
    else:
        risk_level = "Low Privacy Risk"
        risk_color = "#16A36A"
        risk_badge = "badge-success"
        risk_summary = (
            "Generative model demonstrates high generalization with low memorization risk. "
            "Synthetic records occupy plausible data manifolds without replicating real records."
        )
        
    return {
        "min_nnd": round(min_nnd, 4),
        "avg_nnd": round(avg_nnd, 4),
        "median_nnd": round(med_nnd, 4),
        "exact_matches": exact_matches,
        "suspicious_count": suspicious_count,
        "suspicious_pct": round(suspicious_pct, 2),
        "total_records": total_syn,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_badge": risk_badge,
        "risk_summary": risk_summary,
        "distance_distribution": normalized_distances,
        "threshold_used": suspicious_threshold
    }
