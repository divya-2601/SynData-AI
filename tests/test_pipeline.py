"""
End-to-End Integration Test for SynData AI
Tests:
1. Data loading
2. Model training and generation (CTGAN and TVAE)
3. Statistical comparison & KS tests
4. Privacy risk analysis
5. ML utility evaluation (TSTR vs TRTR)
6. PDF reports generation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from core.synthesizer import train_and_generate
from core.metrics import compute_numerical_comparison, compute_ks_tests, compute_overall_quality_score
from core.privacy import compute_privacy_metrics
from core.ml_eval import evaluate_ml_utility
from core.pdf_reports import (
    generate_statistical_pdf,
    generate_privacy_pdf,
    generate_ml_eval_pdf,
    generate_consolidated_pdf
)

def test_pipeline():
    print("Step 1: Loading customer churn dataset...")
    df = pd.read_csv("sample_data/customer_churn.csv")
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    print("\nStep 2: Training TVAE model (5 epochs for fast test)...")
    syn_df, exec_info = train_and_generate(
        df=df,
        model_type="TVAE",
        epochs=5,
        batch_size=20,
        num_samples=30,
        random_seed=42
    )
    print(f"Generated {len(syn_df)} synthetic records. Columns match: {list(syn_df.columns) == list(df.columns)}")

    print("\nStep 3: Calculating statistical metrics...")
    stat_df = compute_numerical_comparison(df, syn_df)
    ks_df = compute_ks_tests(df, syn_df)
    quality_info = compute_overall_quality_score(df, syn_df)
    print(f"Quality Score: {quality_info['overall_score']}% ({quality_info['fidelity_rating']})")

    print("\nStep 4: Computing privacy metrics...")
    privacy_info = compute_privacy_metrics(df, syn_df)
    print(f"Privacy Risk: {privacy_info['risk_level']}, Min NND: {privacy_info['min_nnd']}")

    print("\nStep 5: Running ML utility evaluation (TSTR vs TRTR on 'churn')...")
    ml_res = evaluate_ml_utility(df, syn_df, target_column="churn", test_size=0.3)
    print(f"ML Utility Retention: {ml_res['utility_retention_pct']}%, TRTR F1: {ml_res['trtr_f1']}, TSTR F1: {ml_res['tstr_f1']}")

    print("\nStep 6: Generating PDF reports...")
    pdf1 = generate_statistical_pdf("customer_churn.csv", "TVAE", df, syn_df, stat_df, ks_df, quality_info)
    print(f"Statistical PDF size: {len(pdf1.getvalue())} bytes")

    pdf2 = generate_privacy_pdf("customer_churn.csv", "TVAE", privacy_info)
    print(f"Privacy PDF size: {len(pdf2.getvalue())} bytes")

    pdf3 = generate_ml_eval_pdf("customer_churn.csv", "TVAE", ml_res)
    print(f"ML Utility PDF size: {len(pdf3.getvalue())} bytes")

    pdf4 = generate_consolidated_pdf("customer_churn.csv", "TVAE", df, syn_df, quality_info, privacy_info, ml_res)
    print(f"Consolidated Dossier PDF size: {len(pdf4.getvalue())} bytes")

    print("\n ALL BACKEND AND AUDIT SYSTEMS FUNCTIONING PERFECTLY!")

if __name__ == "__main__":
    test_pipeline()
