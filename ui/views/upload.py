"""
SynData AI - Upload Dataset Page
Clean, robust, and general-purpose (works for ANY tabular CSV).
"""

import streamlit as st
import pandas as pd
from ui.components import render_top_bar, render_step_header, render_empty_box
from core.synthesizer import detect_column_types

def render_upload_view():
    render_top_bar("Upload Dataset")

    has_real = st.session_state.get("real_df") is not None
    badge_text = st.session_state.get("dataset_name", "No dataset uploaded") if has_real else "No dataset uploaded"

    render_step_header(
        step_tag="STEP 01 / YOUR DATA",
        title="Upload your dataset",
        subtitle="Every great synthetic dataset starts with real data. Let's bring yours in.",
        right_badge=badge_text
    )

    col_upload, col_tips = st.columns([1.7, 1])

    with col_upload:
        st.markdown(
            """<div style="background:#FAF8FE;border:1.5px dashed #D5C7F5;border-radius:12px;padding:28px 16px;text-align:center;margin-bottom:12px;">
<div style="font-size:2rem;color:#6C3BDB;margin-bottom:6px;">↑</div>
<div style="font-size:1rem;font-weight:700;color:#1E1534;margin-bottom:4px;">Drag & drop your dataset here</div>
<div style="font-size:0.82rem;color:#6D667F;">One CSV file is all you need to get started.</div>
</div>""",
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            label_visibility="collapsed",
            help="Select any tabular CSV file from your computer."
        )

        st.caption("SUPPORTED FORMAT: .CSV (Maximum file size: 50 MB)")

    with col_tips:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:18px 20px;">
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;margin-bottom:14px;">A few things before you upload</div>
<div style="margin-bottom:12px;">
<span style="color:#6C3BDB;font-weight:700;">✓</span> <b style="font-size:0.85rem;color:#1E1534;">CSV format</b>
<div style="font-size:0.78rem;color:#6D667F;margin-left:16px;">Export your spreadsheet as a standard .csv file.</div>
</div>
<div style="margin-bottom:12px;">
<span style="color:#6C3BDB;font-weight:700;">✓</span> <b style="font-size:0.85rem;color:#1E1534;">Column headers</b>
<div style="font-size:0.78rem;color:#6D667F;margin-left:16px;">Use clear, distinct names for each column.</div>
</div>
<div style="margin-bottom:14px;">
<span style="color:#6C3BDB;font-weight:700;">✓</span> <b style="font-size:0.85rem;color:#1E1534;">One row per record</b>
<div style="font-size:0.78rem;color:#6D667F;margin-left:16px;">Keep data organized in tabular rows and columns.</div>
</div>
<div style="background:#F8F6FE;border-radius:8px;padding:8px 12px;font-size:0.76rem;color:#6D667F;">
ℹ Review missing values before training.
</div>
</div>""",
            unsafe_allow_html=True
        )

    # Handle file upload
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if not df.empty:
                if st.session_state.get("dataset_name") != uploaded_file.name:
                    st.session_state.real_df = df
                    st.session_state.dataset_name = uploaded_file.name
                    st.session_state.syn_df = None
                    st.session_state.model_name = None
                    st.session_state.ml_results = None
                    st.toast(f"Loaded {uploaded_file.name} successfully!", icon="✅")
                    st.rerun()
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Section: Dataset overview using Streamlit native metrics (never breaks or leaks HTML!)
    st.markdown("<div style='font-size:1.05rem;font-weight:700;color:#1E1534;margin-bottom:10px;'>Dataset overview</div>", unsafe_allow_html=True)

    if has_real:
        df = st.session_state.real_df
        discrete_cols, num_cols = detect_column_types(df)
        total_rows_val = f"{len(df):,}"
        total_cols_val = f"{len(df.columns)}"
        missing_val = f"{int(df.isnull().sum().sum()):,}"
        dtypes_val = f"{len(num_cols)} Num, {len(discrete_cols)} Cat"
    else:
        total_rows_val = "—"
        total_cols_val = "—"
        missing_val = "—"
        dtypes_val = "—"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Total rows", value=total_rows_val)
    with m2:
        st.metric(label="Columns", value=total_cols_val)
    with m3:
        st.metric(label="Missing values", value=missing_val)
    with m4:
        st.metric(label="Data types", value=dtypes_val)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Section: Dataset preview
    st.markdown(
        """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;margin-bottom:16px;">
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;margin-bottom:2px;">Dataset preview</div>
<div style="font-size:0.8rem;color:#6D667F;margin-bottom:12px;">Rows, columns and inferred data types will appear here.</div>""",
        unsafe_allow_html=True
    )

    if has_real:
        st.dataframe(st.session_state.real_df.head(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        render_empty_box("📋", "No dataset to preview yet", "Upload a CSV to view the tabular data records.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Bottom action bar
    b_col_text, b_col_btn = st.columns([3.5, 1.3])
    with b_col_text:
        note = "Dataset verified. You can now proceed to model configuration." if has_real else "Upload a dataset to continue to model configuration."
        st.markdown(f"<div style='font-size:0.82rem;color:#6D667F;padding-top:10px;'>{note}</div>", unsafe_allow_html=True)
    with b_col_btn:
        if has_real:
            if st.button("Continue to Generation ➔", type="primary", use_container_width=True):
                st.session_state["nav"] = "generate"
                st.rerun()
        else:
            st.button("Continue to Generation ➔", disabled=True, use_container_width=True)
