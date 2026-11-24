"""
SynData AI - Data Comparison Page
Clean, empirical statistical and distributional comparison.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import config
from ui.components import render_top_bar, render_step_header, render_empty_box, apply_plotly_theme
from core.metrics import (
    compute_numerical_comparison,
    compute_ks_tests,
    compute_correlation_matrices,
    compute_categorical_comparison,
    compute_overall_quality_score
)
from core.synthesizer import detect_column_types

def render_comparison_view():
    render_top_bar("Data Comparison")

    has_real = st.session_state.get("real_df") is not None
    has_syn = st.session_state.get("syn_df") is not None
    real_df = st.session_state.get("real_df")
    syn_df = st.session_state.get("syn_df")
    model_name = st.session_state.get("model_name", "AI Model")

    render_step_header(
        step_tag="STEP 03 / DATA QUALITY",
        title="Compare the patterns",
        subtitle="Explore how closely synthetic data reflects your original dataset."
    )

    # Top Filter Bar with Dropdowns and Legend
    c_f1, c_f2, c_leg = st.columns([1.5, 1.5, 2])
    with c_f1:
        orig_opts = [st.session_state.get("dataset_name")] if has_real else ["No dataset uploaded"]
        st.selectbox("Original", options=orig_opts, index=0, disabled=not has_real)
    with c_f2:
        syn_opts = [f"Synthetic ({model_name})"] if has_syn else ["No dataset generated"]
        st.selectbox("Synthetic", options=syn_opts, index=0, disabled=not has_syn)
    with c_leg:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        badge_state = "Datasets active" if (has_real and has_syn) else "Awaiting datasets"
        st.markdown(
            f"""<div style="display:flex;align-items:center;justify-content:flex-end;gap:14px;">
<div style="font-size:0.8rem;color:#6D667F;"><span style="color:#6C3BDB;">●</span> Original</div>
<div style="font-size:0.8rem;color:#6D667F;"><span style="color:#388BFD;">●</span> Synthetic</div>
<div style="background:#F4F2F8;color:#6D667F;padding:3px 10px;border-radius:12px;font-size:0.75rem;font-weight:600;">{badge_state}</div>
</div>""",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    stat_df = None
    ks_df = None
    r_corr, s_corr, corr_mae = pd.DataFrame(), pd.DataFrame(), 0.0
    cat_comparisons = []
    num_cols = []

    if has_real and has_syn:
        stat_df = compute_numerical_comparison(real_df, syn_df)
        st.session_state.stat_df = stat_df
        ks_df = compute_ks_tests(real_df, syn_df)
        st.session_state.ks_df = ks_df
        quality_info = compute_overall_quality_score(real_df, syn_df)
        st.session_state.quality_info = quality_info
        r_corr, s_corr, corr_mae = compute_correlation_matrices(real_df, syn_df)
        cat_comparisons = compute_categorical_comparison(real_df, syn_df)
        _, num_cols = detect_column_types(real_df)

    # Middle Row: Numerical & Categorical Distributions
    col_dist1, col_dist2 = st.columns(2)

    with col_dist1:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;height:100%;">
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;margin-bottom:2px;">Numerical distributions</div>
<div style="font-size:0.8rem;color:#6D667F;margin-bottom:12px;">Histogram comparison for each numerical column</div>""",
            unsafe_allow_html=True
        )
        if has_real and has_syn and num_cols:
            sel_num = st.selectbox("Feature", num_cols, key="comp_num_key", label_visibility="collapsed")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=real_df[sel_num].dropna(),
                name="Original",
                opacity=0.6,
                marker_color=config.COLOR_PRIMARY_PURPLE
            ))
            fig_hist.add_trace(go.Histogram(
                x=syn_df[sel_num].dropna(),
                name="Synthetic",
                opacity=0.6,
                marker_color="#388BFD"
            ))
            fig_hist.update_layout(barmode='overlay')
            apply_plotly_theme(fig_hist, height=260)
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            render_empty_box("📊", "Generate data to compare distributions")
            st.markdown("</div>", unsafe_allow_html=True)

    with col_dist2:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;height:100%;">
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;margin-bottom:2px;">Categorical distributions</div>
<div style="font-size:0.8rem;color:#6D667F;margin-bottom:12px;">Category frequency across original and synthetic records</div>""",
            unsafe_allow_html=True
        )
        if has_real and has_syn and cat_comparisons:
            cat_keys = [c["column"] for c in cat_comparisons]
            sel_cat = st.selectbox("Category", cat_keys, key="comp_cat_key", label_visibility="collapsed")
            c_item = next(item for item in cat_comparisons if item["column"] == sel_cat)
            c_data = c_item["data"]
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=c_data["Category"],
                y=c_data["Real Proportion (%)"],
                name="Original",
                marker_color=config.COLOR_PRIMARY_PURPLE
            ))
            fig_bar.add_trace(go.Bar(
                x=c_data["Category"],
                y=c_data["Synthetic Proportion (%)"],
                name="Synthetic",
                marker_color="#388BFD"
            ))
            fig_bar.update_layout(barmode='group')
            apply_plotly_theme(fig_bar, height=260)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            render_empty_box("📊", "Category comparisons will appear here")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Statistical comparison Card
    st.markdown(
        """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;">
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;margin-bottom:2px;">Statistical comparison</div>
<div style="font-size:0.8rem;color:#6D667F;margin-bottom:14px;">Per-column summary statistics and two-sample Kolmogorov-Smirnov tests.</div>""",
        unsafe_allow_html=True
    )

    if has_real and has_syn and stat_df is not None and ks_df is not None:
        merged_rows = []
        for _, s_row in stat_df.iterrows():
            f_name = s_row["Feature"]
            ks_match = ks_df[ks_df["Feature"] == f_name]
            ks_stat = ks_match["KS Statistic"].values[0] if not ks_match.empty else "N/A"
            p_val = ks_match["P-Value"].values[0] if not ks_match.empty else "N/A"
            
            merged_rows.append({
                "Column": f_name,
                "Original mean / SD": f"{s_row['Real Mean']} / {s_row['Real Std']}",
                "Synthetic mean / SD": f"{s_row['Syn Mean']} / {s_row['Syn Std']}",
                "KS statistic": ks_stat,
                "p-value": p_val
            })
            
        st.dataframe(pd.DataFrame(merged_rows), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        render_empty_box("📋", "Upload and generate a dataset to calculate summary statistics and KS test results.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Bottom Row: Original correlation heatmap & Synthetic correlation heatmap
    c_hm1, c_hm2 = st.columns(2)

    with c_hm1:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;height:100%;">
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;margin-bottom:2px;">Original correlation heatmap</div>
<div style="font-size:0.8rem;color:#6D667F;margin-bottom:12px;">Relationships between numerical columns in your source data</div>""",
            unsafe_allow_html=True
        )
        if has_real and not r_corr.empty:
            fig_r = px.imshow(
                r_corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale=[[0, "#FFFFFF"], [0.5, config.COLOR_SOFT_PURPLE], [1.0, config.COLOR_PRIMARY_PURPLE]]
            )
            apply_plotly_theme(fig_r, height=270)
            st.plotly_chart(fig_r, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            render_empty_box("▦", "Upload a dataset to view correlations")
            st.markdown("</div>", unsafe_allow_html=True)

    with c_hm2:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;height:100%;">
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;margin-bottom:2px;">Synthetic correlation heatmap</div>
<div style="font-size:0.8rem;color:#6D667F;margin-bottom:12px;">Relationships between numerical columns in generated data</div>""",
            unsafe_allow_html=True
        )
        if has_syn and not s_corr.empty:
            fig_s = px.imshow(
                s_corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale=[[0, "#FFFFFF"], [0.5, "#D5E7FE"], [1.0, "#388BFD"]]
            )
            apply_plotly_theme(fig_s, height=270)
            st.plotly_chart(fig_s, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            render_empty_box("▦", "Generate data to view correlations")
            st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Results are calculated from your datasets. No sample metrics are displayed.")
