"""
SynData AI - Machine Learning Utility Evaluation Engine
Implements Train on Synthetic, Test on Real (TSTR) vs Train on Real, Test on Real (TRTR).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
from typing import Dict, Any, Tuple, Optional, List

def identify_candidate_targets(df: pd.DataFrame) -> List[str]:
    """
    Identifies sensible candidate target columns for ML utility testing.
    Prioritizes low-cardinality discrete columns or columns named 'target', 'churn', etc.
    """
    candidates = []
    # Priority keyword search
    priority_names = ['churn', 'target', 'label', 'outcome', 'class', 'status', 'approved', 'survived']
    
    for col in df.columns:
        if any(p in col.lower() for p in priority_names):
            candidates.append(col)
            
    # Then columns with 2 to 10 unique values
    for col in df.columns:
        if col not in candidates:
            n_unique = df[col].nunique()
            if 2 <= n_unique <= 10:
                candidates.append(col)
                
    # If still none, add all columns
    if not candidates:
        candidates = list(df.columns)
        
    return candidates

def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_syn: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encodes and scales feature matrices consistently to prevent data leakage.
    Fitting is strictly done on the training references.
    """
    # Align categorical columns
    cols = list(X_train.columns)
    
    # Simple median imputation for numerical, mode for categorical
    X_tr = X_train.copy()
    X_te = X_test.copy()
    X_sy = X_syn.copy()
    
    for c in cols:
        if pd.api.types.is_numeric_dtype(X_tr[c]):
            med = X_tr[c].median()
            X_tr[c] = X_tr[c].fillna(med)
            X_te[c] = X_te[c].fillna(med)
            X_sy[c] = X_sy[c].fillna(med)
        else:
            mode = X_tr[c].mode()[0] if not X_tr[c].mode().empty else "Unknown"
            X_tr[c] = X_tr[c].fillna(mode).astype(str)
            X_te[c] = X_te[c].fillna(mode).astype(str)
            X_sy[c] = X_sy[c].fillna(mode).astype(str)
            
    # One-hot encoding with dummy alignment
    all_dfs = pd.concat([X_tr.assign(split=0), X_te.assign(split=1), X_sy.assign(split=2)], ignore_index=True)
    dummies = pd.get_dummies(all_dfs.drop(columns=['split']), drop_first=True)
    
    tr_mask = (all_dfs['split'] == 0)
    te_mask = (all_dfs['split'] == 1)
    sy_mask = (all_dfs['split'] == 2)
    
    scaler = StandardScaler()
    # Fit scaler strictly on real training set
    tr_scaled = scaler.fit_transform(dummies[tr_mask])
    te_scaled = scaler.transform(dummies[te_mask])
    sy_scaled = scaler.transform(dummies[sy_mask])
    
    return tr_scaled, te_scaled, sy_scaled

def evaluate_ml_utility(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.30,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Evaluates ML utility by training models on real vs synthetic datasets
    and evaluating both on the exact same unseen real test partition.
    """
    if target_column not in real_df.columns or target_column not in syn_df.columns:
        raise ValueError(f"Target column '{target_column}' must be present in both datasets.")
        
    # Check task type: Classification vs Regression
    is_classification = (
        real_df[target_column].dtype == 'object' or
        real_df[target_column].dtype.name == 'category' or
        real_df[target_column].nunique() <= 10
    )
    
    # Filter valid rows
    r_valid = real_df.dropna(subset=[target_column]).copy()
    s_valid = syn_df.dropna(subset=[target_column]).copy()
    
    X_real = r_valid.drop(columns=[target_column])
    y_real = r_valid[target_column]
    
    X_syn = s_valid.drop(columns=[target_column])
    y_syn = s_valid[target_column]
    
    # Align labels for classification
    if is_classification:
        y_real = y_real.astype(str)
        y_syn = y_syn.astype(str)
        strat = y_real if y_real.value_counts().min() >= 2 else None
        
        X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
            X_real, y_real, test_size=test_size, random_state=random_state, stratify=strat
        )
        
        # Preprocess features
        X_tr, X_te, X_sy = preprocess_features(X_real_train, X_real_test, X_syn)
        
        # Model 1: Train on Real Train, Test on Real Test (TRTR)
        clf_real = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=8)
        clf_real.fit(X_tr, y_real_train)
        y_pred_real = clf_real.predict(X_te)
        
        # Model 2: Train on Synthetic, Test on Real Test (TSTR)
        clf_syn = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=8)
        # Ensure synthetic training labels only contain classes seen or handled
        clf_syn.fit(X_sy, y_syn)
        y_pred_syn = clf_syn.predict(X_te)
        
        # Calculate metrics
        acc_real = accuracy_score(y_real_test, y_pred_real)
        acc_syn = accuracy_score(y_real_test, y_pred_syn)
        
        prec_real = precision_score(y_real_test, y_pred_real, average='weighted', zero_division=0)
        prec_syn = precision_score(y_real_test, y_pred_syn, average='weighted', zero_division=0)
        
        rec_real = recall_score(y_real_test, y_pred_real, average='weighted', zero_division=0)
        rec_syn = recall_score(y_real_test, y_pred_syn, average='weighted', zero_division=0)
        
        f1_real = f1_score(y_real_test, y_pred_real, average='weighted', zero_division=0)
        f1_syn = f1_score(y_real_test, y_pred_syn, average='weighted', zero_division=0)
        
        # Calculate Utility Retention Ratio
        utility_retention = (f1_syn / f1_real * 100.0) if f1_real > 0 else 0.0
        utility_retention = max(0.0, min(100.0, utility_retention))
        
        metrics_table = pd.DataFrame([
            {"Metric": "Accuracy", "Real Model (TRTR)": round(acc_real, 4), "Synthetic Model (TSTR)": round(acc_syn, 4), "Difference": round(acc_syn - acc_real, 4)},
            {"Metric": "Precision", "Real Model (TRTR)": round(prec_real, 4), "Synthetic Model (TSTR)": round(prec_syn, 4), "Difference": round(prec_syn - prec_real, 4)},
            {"Metric": "Recall", "Real Model (TRTR)": round(rec_real, 4), "Synthetic Model (TSTR)": round(rec_syn, 4), "Difference": round(rec_syn - rec_real, 4)},
            {"Metric": "F1-Score", "Real Model (TRTR)": round(f1_real, 4), "Synthetic Model (TSTR)": round(f1_syn, 4), "Difference": round(f1_syn - f1_real, 4)},
        ])
        
        return {
            "task_type": "Classification",
            "target_column": target_column,
            "classes": sorted(list(y_real.unique())),
            "trtr_accuracy": round(acc_real, 4),
            "tstr_accuracy": round(acc_syn, 4),
            "trtr_f1": round(f1_real, 4),
            "tstr_f1": round(f1_syn, 4),
            "utility_retention_pct": round(utility_retention, 2),
            "metrics_table": metrics_table,
            "train_size": len(X_real_train),
            "test_size": len(X_real_test),
            "synthetic_train_size": len(X_syn)
        }
    else:
        # Regression case
        X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
            X_real, y_real, test_size=test_size, random_state=random_state
        )
        X_tr, X_te, X_sy = preprocess_features(X_real_train, X_real_test, X_syn)
        
        reg_real = RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=8)
        reg_real.fit(X_tr, y_real_train)
        y_pred_real = reg_real.predict(X_te)
        
        reg_syn = RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=8)
        reg_syn.fit(X_sy, y_syn)
        y_pred_syn = reg_syn.predict(X_te)
        
        r2_real = r2_score(y_real_test, y_pred_real)
        r2_syn = r2_score(y_real_test, y_pred_syn)
        
        mse_real = mean_squared_error(y_real_test, y_pred_real)
        mse_syn = mean_squared_error(y_real_test, y_pred_syn)
        
        utility_retention = max(0.0, min(100.0, (r2_syn / r2_real * 100.0) if r2_real > 0 else 0.0))
        
        metrics_table = pd.DataFrame([
            {"Metric": "R2 Score", "Real Model (TRTR)": round(r2_real, 4), "Synthetic Model (TSTR)": round(r2_syn, 4), "Difference": round(r2_syn - r2_real, 4)},
            {"Metric": "MSE", "Real Model (TRTR)": round(mse_real, 4), "Synthetic Model (TSTR)": round(mse_syn, 4), "Difference": round(mse_syn - mse_real, 4)},
        ])
        
        return {
            "task_type": "Regression",
            "target_column": target_column,
            "trtr_accuracy": round(r2_real, 4),
            "tstr_accuracy": round(r2_syn, 4),
            "trtr_f1": round(r2_real, 4),
            "tstr_f1": round(r2_syn, 4),
            "utility_retention_pct": round(utility_retention, 2),
            "metrics_table": metrics_table,
            "train_size": len(X_real_train),
            "test_size": len(X_real_test),
            "synthetic_train_size": len(X_syn)
        }
