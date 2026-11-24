"""
SynData AI - Dataset Overview Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import config
from ui.components import render_page_header, render_stat_card, render_empty_state, apply_plotly_theme
from core.synthesizer import detect_column_types

def render_overview_view():
    render_page_header(
        title="Dataset Overview",
        subtitle="Explore detailed structural characteristics, column profiles, and statistical distributions.",
        icon="📊"
    )

    if st.session_state.get("real_df") is None:
        render_empty_state(
            title="No Dataset Available to Profile",
            description="Please upload a CSV file or load a demo dataset to explore structural metrics and visualizations.",
            icon="📊"
        )
        if st.button("➔ Go to Dataset Upload", type="primary"):
            st.session_state["nav"] = "upload"
            st.rerun()
        return

    df = st.session_state.real_df
    discrete_cols, num_cols = detect_column_types(df)

    # Tabs for structured viewing
    tab1, tab2, tab3 = st.tabs(["📋 Column Dictionary & Stats", "📈 Univariate Visualizations", "🔥 Correlation Analysis"])

    with tab1:
        st.markdown(f"<h4 style='color: {config.COLOR_DARK_PURPLE}; font-size: 1.05rem; font-weight: 700; margin-bottom: 0.8rem;'>Feature Profiling Table</h4>", unsafe_allow_html=True)
        
        prof_data = []
        for col in df.columns:
            n_missing = int(df[col].isnull().sum())
            pct_missing = round((n_missing / len(df)) * 100, 2)
            n_unique = int(df[col].nunique())
            dtype_str = str(df[col].dtype)
            col_type = "Categorical / Discrete" if col in discrete_cols else "Numerical Continuous"
            
            prof_data.append({
                "Column": col,
                "Type Category": col_type,
                "Data Type": dtype_str,
                "Unique Values": n_unique,
                "Missing Values": f"{n_missing} ({pct_missing}%)",
                "Sample Value": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A"
            })
            
        st.dataframe(pd.DataFrame(prof_data), use_container_width=True)

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        if num_cols:
            st.markdown(f"<h4 style='color: {config.COLOR_DARK_PURPLE}; font-size: 1.05rem; font-weight: 700; margin-bottom: 0.8rem;'>Numerical Descriptive Statistics</h4>", unsafe_allow_html=True)
            desc_df = df[num_cols].describe().round(2).T.reset_index().rename(columns={"index": "Feature"})
            st.dataframe(desc_df, use_container_width=True)

    with tab2:
        col_vis1, col_vis2 = st.columns(2)
        
        with col_vis1:
            st.markdown(f"<h4 style='color: {config.COLOR_DARK_PURPLE}; font-size: 1rem; font-weight: 700; margin-bottom: 0.6rem;'>Numerical Feature Histograms</h4>", unsafe_allow_html=True)
            if num_cols:
                selected_num = st.selectbox("Select Numerical Feature", num_cols, key="overview_num_select")
                fig_num = px.histogram(
                    df,
                    x=selected_num,
                    nbins=30,
                    marginal="box",
                    color_discrete_sequence=[config.COLOR_PRIMARY_PURPLE],
                    title=f"Distribution of '{selected_num}'"
                )
                apply_plotly_theme(fig_num, height=360)
                st.plotly_chart(fig_num, use_container_width=True)
            else:
                st.info("No continuous numerical features detected.")

        with col_vis2:
            st.markdown(f"<h4 style='color: {config.COLOR_DARK_PURPLE}; font-size: 1rem; font-weight: 700; margin-bottom: 0.6rem;'>Categorical Feature Frequencies</h4>", unsafe_allow_html=True)
            if discrete_cols:
                selected_cat = st.selectbox("Select Categorical Feature", discrete_cols, key="overview_cat_select")
                counts = df[selected_cat].value_counts().reset_index()
                counts.columns = [selected_cat, "Count"]
                fig_cat = px.bar(
                    counts,
                    x=selected_cat,
                    y="Count",
                    color_discrete_sequence=[config.COLOR_DARK_PURPLE],
                    title=f"Class Frequency: '{selected_cat}'"
                )
                apply_plotly_theme(fig_cat, height=360)
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("No categorical features detected.")

        # Missing values distribution chart
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: {config.COLOR_DARK_PURPLE}; font-size: 1rem; font-weight: 700; margin-bottom: 0.6rem;'>Missing Values by Feature</h4>", unsafe_allow_html=True)
        missing_counts = df.isnull().sum().reset_index()
        missing_counts.columns = ["Feature", "Missing_Count"]
        
        if missing_counts["Missing_Count"].sum() > 0:
            fig_missing = px.bar(
                missing_counts[missing_counts["Missing_Count"] > 0],
                x="Feature",
                y="Missing_Count",
                color_discrete_sequence=[config.COLOR_WARNING],
                title="Missing Values Count per Column"
            )
            apply_plotly_theme(fig_missing, height=280)
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.markdown(
                f"""
                <div style="background: #E8F8F0; border: 1px solid #C4EDD8; border-radius: 8px; padding: 10px 14px; color: #0E5E3D; font-size: 0.85rem;">
                    ✓ Zero missing values detected across all columns. The dataset is complete and ready for generative modeling.
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab3:
        st.markdown(f"<h4 style='color: {config.COLOR_DARK_PURPLE}; font-size: 1.05rem; font-weight: 700; margin-bottom: 0.8rem;'>Inter-Feature Correlation Matrix (Pearson)</h4>", unsafe_allow_html=True)
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale=[[0, "#FFFFFF"], [0.5, config.COLOR_SOFT_PURPLE], [1.0, config.COLOR_DARK_PURPLE]],
                title=f"Correlation Heatmap ({len(num_cols)} Features)"
            )
            apply_plotly_theme(fig_corr, height=450)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("At least two numerical features are required to compute a correlation matrix.")
            
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    c_btn1, _ = st.columns([1, 3])
    with c_btn1:
        if st.button("Proceed to Generate Synthetic Data ➔", type="primary", use_container_width=True):
            st.session_state["nav"] = "generate"
            st.rerun()
