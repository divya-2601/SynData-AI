"""
SynData AI - Dashboard (Home Page)
Universal tabular data platform matching Image 1 layout.
"""

import streamlit as st
import config
from ui.components import render_top_bar

def render_dashboard_view():
    render_top_bar("Dashboard")

    # Header Row with Top-Right Action Button
    col_head, col_btn = st.columns([3.5, 1])
    with col_head:
        st.markdown(
            """<div class="step-indicator">OVERVIEW</div>
<div style="font-size:1.65rem;font-weight:700;color:#1E1534;margin-bottom:4px;">Welcome to SynData AI</div>
<div style="font-size:0.88rem;color:#6D667F;margin-bottom:16px;">Generate realistic synthetic datasets while evaluating data quality and privacy risks.</div>""",
            unsafe_allow_html=True
        )
    with col_btn:
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        if st.button("↑ Upload dataset", type="primary", use_container_width=True, key="dash_top_upload"):
            st.session_state["nav"] = "upload"
            st.rerun()

    # Hero Banner: "Realistic data. Real possibilities."
    st.markdown(
        """<div style="background:linear-gradient(135deg, #F5F1FE 0%, #FAF8FE 100%);border:1px solid #EBE4FA;border-radius:14px;padding:22px 28px;margin-bottom:20px;display:flex;justify-content:space-between;align-items:center;">
<div style="max-width:560px;">
<div style="font-size:0.7rem;font-weight:700;color:#6C3BDB;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px;">YOUR DATA, REIMAGINED</div>
<div style="font-size:1.45rem;font-weight:800;color:#1E1534;margin-bottom:4px;">Realistic data. Real possibilities.</div>
<div style="font-size:0.86rem;color:#6D667F;">Start with a CSV. Generate, compare and evaluate in one place.</div>
</div>
<div style="display:flex;align-items:center;gap:12px;background:#FFFFFF;border:1px solid #ECE7F8;border-radius:12px;padding:12px 18px;">
<div style="text-align:center;"><div style="font-size:0.68rem;font-weight:700;color:#948CA5;">01</div><div style="font-size:1.2rem;">🗄️</div><div style="font-size:0.72rem;font-weight:600;color:#1E1534;">Your dataset</div></div>
<div style="color:#948CA5;font-size:0.9rem;">&rarr;</div>
<div style="text-align:center;"><div style="font-size:0.68rem;font-weight:700;color:#948CA5;">02</div><div style="font-size:1.2rem;">✦</div><div style="font-size:0.72rem;font-weight:600;color:#1E1534;">CTGAN / TVAE</div></div>
<div style="color:#948CA5;font-size:0.9rem;">&rarr;</div>
<div style="text-align:center;"><div style="font-size:0.68rem;font-weight:700;color:#948CA5;">03</div><div style="font-size:1.2rem;">📋</div><div style="font-size:0.72rem;font-weight:600;color:#1E1534;">Synthetic data</div></div>
</div>
</div>""",
        unsafe_allow_html=True
    )

    # 4 Stat Metric Cards (using clean Streamlit native metrics for 100% stability)
    has_real = st.session_state.get("real_df") is not None
    has_syn = st.session_state.get("syn_df") is not None

    orig_name = st.session_state.get("dataset_name", "No dataset") if has_real else "No dataset"
    records_val = f"{len(st.session_state.real_df):,}" if has_real else "—"
    syn_val = f"{len(st.session_state.syn_df):,}" if has_syn else "—"
    gen_status = "Generated" if has_syn else ("Ready to run" if has_real else "Not started")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric(label="Original Dataset", value=orig_name)
    with s2:
        st.metric(label="Total Records", value=records_val)
    with s3:
        st.metric(label="Synthetic Records", value=syn_val)
    with s4:
        st.metric(label="Generation Status", value=gen_status)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Section: How It Works
    st.markdown(
        """<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px;">
<div style="font-size:1rem;font-weight:700;color:#1E1534;">How It Works</div>
<div style="font-size:0.7rem;font-weight:700;color:#948CA5;letter-spacing:0.08em;text-transform:uppercase;">FROM SOURCE TO SYNTHETIC</div>
</div>""",
        unsafe_allow_html=True
    )

    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:10px;padding:14px;height:100%;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
<div style="width:30px;height:30px;border-radius:8px;background:#EDE8FD;color:#6C3BDB;display:flex;align-items:center;justify-content:center;font-size:14px;">↑</div>
<div style="font-size:0.75rem;color:#948CA5;font-weight:600;">01 &rarr;</div>
</div>
<div style="font-size:0.88rem;font-weight:700;color:#1E1534;margin-bottom:3px;">Upload Dataset</div>
<div style="font-size:0.76rem;color:#6D667F;">Bring your tabular data.</div>
</div>""",
            unsafe_allow_html=True
        )
    with h2:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:10px;padding:14px;height:100%;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
<div style="width:30px;height:30px;border-radius:8px;background:#EDE8FD;color:#6C3BDB;display:flex;align-items:center;justify-content:center;font-size:14px;">✦</div>
<div style="font-size:0.75rem;color:#948CA5;font-weight:600;">02 &rarr;</div>
</div>
<div style="font-size:0.88rem;font-weight:700;color:#1E1534;margin-bottom:3px;">Train AI Model</div>
<div style="font-size:0.76rem;color:#6D667F;">Choose CTGAN or TVAE.</div>
</div>""",
            unsafe_allow_html=True
        )
    with h3:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:10px;padding:14px;height:100%;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
<div style="width:30px;height:30px;border-radius:8px;background:#EDE8FD;color:#6C3BDB;display:flex;align-items:center;justify-content:center;font-size:14px;">🗄️</div>
<div style="font-size:0.75rem;color:#948CA5;font-weight:600;">03 &rarr;</div>
</div>
<div style="font-size:0.88rem;font-weight:700;color:#1E1534;margin-bottom:3px;">Generate Synthetic Data</div>
<div style="font-size:0.76rem;color:#6D667F;">Create new, realistic records.</div>
</div>""",
            unsafe_allow_html=True
        )
    with h4:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:10px;padding:14px;height:100%;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
<div style="width:30px;height:30px;border-radius:8px;background:#EDE8FD;color:#6C3BDB;display:flex;align-items:center;justify-content:center;font-size:14px;">📊</div>
<div style="font-size:0.75rem;color:#948CA5;font-weight:600;">04</div>
</div>
<div style="font-size:0.88rem;font-weight:700;color:#1E1534;margin-bottom:3px;">Evaluate Results</div>
<div style="font-size:0.76rem;color:#6D667F;">Understand quality and privacy.</div>
</div>""",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Section: Recent Activity
    act_col, act_btn_col = st.columns([4, 1.2])
    with act_col:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;display:flex;align-items:center;gap:14px;">
<div style="width:36px;height:36px;border-radius:9px;background:#EDE8FD;color:#6C3BDB;display:flex;align-items:center;justify-content:center;font-size:16px;">🕒</div>
<div>
<div style="font-size:0.9rem;font-weight:700;color:#1E1534;">A fresh start for your data</div>
<div style="font-size:0.78rem;color:#6D667F;">Your uploads, generation runs and evaluations will appear here.</div>
</div>
</div>""",
            unsafe_allow_html=True
        )
    with act_btn_col:
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        if st.button("↑ Upload dataset", key="recent_dash_btn", use_container_width=True):
            st.session_state["nav"] = "upload"
            st.rerun()
