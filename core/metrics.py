"""
SynData AI - Statistical Metrics & Comparison Engine
Calculates real statistical similarity, Kolmogorov-Smirnov tests, and correlation analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Tuple

def compute_numerical_comparison(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Mean, Median, Std, Min, Max for real vs synthetic datasets.
    """
    num_cols = [c for c in real_df.columns if pd.api.types.is_numeric_dtype(real_df[c]) and c in syn_df.columns]
    records = []
    
    for col in num_cols:
        r_series = real_df[col].dropna()
        s_series = syn_df[col].dropna()
        
        if len(r_series) == 0 or len(s_series) == 0:
            continue
            
        r_mean, s_mean = r_series.mean(), s_series.mean()
        r_std, s_std = r_series.std(), s_series.std()
        r_med, s_med = r_series.median(), s_series.median()
        r_min, s_min = r_series.min(), s_series.min()
        r_max, s_max = r_series.max(), s_series.max()
        
        mean_diff = abs(r_mean - s_mean)
        std_diff = abs(r_std - s_std) if not np.isnan(r_std) and not np.isnan(s_std) else 0.0
        
        records.append({
            "Feature": col,
            "Real Mean": round(float(r_mean), 2),
            "Syn Mean": round(float(s_mean), 2),
            "Mean Diff": round(float(mean_diff), 2),
            "Real Median": round(float(r_med), 2),
            "Syn Median": round(float(s_med), 2),
            "Real Std": round(float(r_std), 2) if not np.isnan(r_std) else 0.0,
            "Syn Std": round(float(s_std), 2) if not np.isnan(s_std) else 0.0,
            "Real Min": round(float(r_min), 2),
            "Syn Min": round(float(s_min), 2),
            "Real Max": round(float(r_max), 2),
            "Syn Max": round(float(s_max), 2),
        })
        
    return pd.DataFrame(records)

def compute_ks_tests(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs Two-Sample Kolmogorov-Smirnov Test for every numerical feature.
    """
    num_cols = [c for c in real_df.columns if pd.api.types.is_numeric_dtype(real_df[c]) and c in syn_df.columns]
    results = []
    
    for col in num_cols:
        r_clean = real_df[col].dropna()
        s_clean = syn_df[col].dropna()
        
        if len(r_clean) < 2 or len(s_clean) < 2:
            continue
            
        ks_stat, p_val = stats.ks_2samp(r_clean, s_clean)
        
        # KS Statistic ranges from 0 to 1 (lower is better, 0 = identical distributions)
        # Similarity = (1 - KS Stat) * 100
        similarity = max(0.0, (1.0 - ks_stat) * 100.0)
        
        if ks_stat < 0.15:
            assessment = "Excellent Match"
        elif ks_stat < 0.30:
            assessment = "Good Match"
        elif ks_stat < 0.50:
            assessment = "Moderate Match"
        else:
            assessment = "Divergent"
            
        results.append({
            "Feature": col,
            "KS Statistic": round(float(ks_stat), 4),
            "P-Value": f"{p_val:.4e}" if p_val < 0.0001 else f"{round(float(p_val), 4)}",
            "Similarity Score (%)": round(similarity, 1),
            "Distribution Fit": assessment
        })
        
    return pd.DataFrame(results)

def compute_correlation_matrices(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Computes Pearson correlation for numerical columns and calculates Mean Absolute Difference.
    """
    num_cols = [c for c in real_df.columns if pd.api.types.is_numeric_dtype(real_df[c]) and c in syn_df.columns]
    
    if len(num_cols) < 2:
        return pd.DataFrame(), pd.DataFrame(), 0.0
        
    r_corr = real_df[num_cols].corr().fillna(0)
    s_corr = syn_df[num_cols].corr().fillna(0)
    
    # Calculate Mean Absolute Error across correlations
    diff_matrix = (r_corr - s_corr).abs()
    # Exclude diagonal (which is always 0)
    n = len(num_cols)
    if n > 1:
        triu_indices = np.triu_indices(n, k=1)
        mae = float(np.mean(diff_matrix.values[triu_indices]))
    else:
        mae = 0.0
        
    return r_corr, s_corr, round(mae, 4)

def compute_categorical_comparison(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Compares frequency distribution for categorical / discrete columns.
    """
    cat_cols = [c for c in real_df.columns if not pd.api.types.is_numeric_dtype(real_df[c]) and c in syn_df.columns]
    results = []
    
    for col in cat_cols:
        r_counts = real_df[col].value_counts(normalize=True).to_dict()
        s_counts = syn_df[col].value_counts(normalize=True).to_dict()
        
        all_cats = list(set(list(r_counts.keys()) + list(s_counts.keys())))
        cat_data = []
        tv_dist = 0.0
        
        for cat in all_cats:
            r_freq = r_counts.get(cat, 0.0)
            s_freq = s_counts.get(cat, 0.0)
            tv_dist += abs(r_freq - s_freq)
            cat_data.append({
                "Category": str(cat),
                "Real Proportion (%)": round(r_freq * 100, 2),
                "Synthetic Proportion (%)": round(s_freq * 100, 2),
            })
            
        # Total Variation Distance (TVD) = 0.5 * sum(|p - q|)
        tvd = 0.5 * tv_dist
        sim_score = max(0.0, (1.0 - tvd) * 100.0)
        
        results.append({
            "column": col,
            "data": pd.DataFrame(cat_data),
            "tvd": round(tvd, 4),
            "similarity_pct": round(sim_score, 1)
        })
        
    return results

def compute_overall_quality_score(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Aggregates statistical tests into an empirical quality score.
    """
    ks_df = compute_ks_tests(real_df, syn_df)
    cat_res = compute_categorical_comparison(real_df, syn_df)
    _, _, corr_mae = compute_correlation_matrices(real_df, syn_df)
    
    scores = []
    if not ks_df.empty:
        scores.append(ks_df["Similarity Score (%)"].mean())
        
    if cat_res:
        cat_mean = np.mean([item["similarity_pct"] for item in cat_res])
        scores.append(cat_mean)
        
    if len(scores) > 0:
        base_score = float(np.mean(scores))
        # Penalty for correlation discrepancy
        corr_penalty = min(20.0, corr_mae * 25.0)
        final_score = max(10.0, min(99.0, base_score - corr_penalty))
    else:
        final_score = 75.0
        
    if final_score >= 80:
        rating = "High Fidelity"
        badge_type = "badge-success"
    elif final_score >= 65:
        rating = "Good Fidelity"
        badge_type = "badge-purple"
    elif final_score >= 50:
        rating = "Moderate Fidelity"
        badge_type = "badge-warning"
    else:
        rating = "Low Fidelity"
        badge_type = "badge-error"
        
    return {
        "overall_score": round(final_score, 1),
        "fidelity_rating": rating,
        "badge_class": badge_type,
        "ks_average_similarity": round(ks_df["Similarity Score (%)"].mean(), 1) if not ks_df.empty else 0.0,
        "correlation_mae": corr_mae,
        "num_features_evaluated": len(ks_df) + len(cat_res)
    }
