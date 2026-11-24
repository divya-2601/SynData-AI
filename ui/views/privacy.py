"""
SynData AI - Privacy Analysis Page
Clean, empirical privacy risk auditing with nearest-neighbor distance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import config
from ui.components import render_top_bar, render_step_header, render_empty_box, apply_plotly_theme
from core.privacy import compute_privacy_metrics

def render_privacy_view():
    render_top_bar("Privacy Analysis")

    has_real = st.session_state.get("real_df") is not None
    has_syn = st.session_state.get("syn_df") is not None
    real_df = st.session_state.get("real_df")
    syn_df = st.session_state.get("syn_df")

    c_head, c_btn = st.columns([3.5, 1.4])
    with c_head:
        render_step_header(
            step_tag="STEP 03 / PRIVACY",
            title="Privacy, with perspective",
            subtitle="Understand record similarity and potential disclosure risks in your synthetic data."
        )
    with c_btn:
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        if has_real and has_syn:
            run_clicked = st.button("🛡️ Run privacy analysis", type="primary", use_container_width=True, key="do_priv_eval")
        else:
            run_clicked = False
            st.button("🛡️ Run privacy analysis", disabled=True, use_container_width=True)

    if run_clicked and has_real and has_syn:
        with st.spinner("Computing high-dimensional nearest-neighbor metrics..."):
            privacy_info = compute_privacy_metrics(real_df, syn_df)
            st.session_state.privacy_info = privacy_info

    privacy_info = st.session_state.get("privacy_info")
    has_priv = privacy_info is not None

    # Top 4 Metric Cards using native Streamlit metrics (100% stable!)
    avg_nnd_val = f"{privacy_info['avg_nnd']:.3f}" if has_priv else "—"
    min_nnd_val = f"{privacy_info['min_nnd']:.3f}" if has_priv else "—"
    susp_val = f"{privacy_info['suspicious_count']}" if has_priv else "—"
    risk_val = privacy_info['risk_level'] if has_priv else "Not assessed"

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric(label="Average NN distance", value=avg_nnd_val)
    with p2:
        st.metric(label="Minimum NN distance", value=min_nnd_val)
    with p3:
        st.metric(label="Suspiciously similar records", value=susp_val)
    with p4:
        st.metric(label="Privacy risk", value=risk_val)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Middle Row: Record similarity analysis & Reading risk indicators
    col_sim, col_ind = st.columns([1.6, 1])

    with col_sim:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;height:100%;">
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;margin-bottom:2px;">Record similarity analysis</div>
<div style="font-size:0.8rem;color:#6D667F;margin-bottom:12px;">Distances between synthetic records and their nearest original neighbours.</div>""",
            unsafe_allow_html=True
        )
        if has_priv:
            nnd_series = pd.Series(privacy_info['distance_distribution'])
            fig_nnd = px.histogram(
                nnd_series,
                nbins=25,
                color_discrete_sequence=[config.COLOR_PRIMARY_PURPLE],
                labels={"value": "Normalized Distance to Nearest Real Record"}
            )
            apply_plotly_theme(fig_nnd, height=260)
            st.plotly_chart(fig_nnd, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            render_empty_box("🛡️", "No privacy analysis yet", "Upload and generate data to calculate nearest-neighbour distances.")
            st.markdown("</div>", unsafe_allow_html=True)

    with col_ind:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;height:100%;">
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;margin-bottom:2px;">Reading risk indicators</div>
<div style="font-size:0.8rem;color:#6D667F;margin-bottom:16px;">Labels describe findings, not a privacy guarantee.</div>
<div style="display:flex;flex-direction:column;gap:12px;">
<div style="display:flex;align-items:center;gap:10px;"><span style="background:#E8F8F0;color:#16A36A;padding:3px 10px;border-radius:14px;font-size:0.75rem;font-weight:700;">Low Risk</span><span style="font-size:0.78rem;color:#6D667F;">Lower observed similarity</span></div>
<div style="display:flex;align-items:center;gap:10px;"><span style="background:#FEF7EA;color:#E5A12B;padding:3px 10px;border-radius:14px;font-size:0.75rem;font-weight:700;">Moderate Risk</span><span style="font-size:0.78rem;color:#6D667F;">Review potentially close matches</span></div>
<div style="display:flex;align-items:center;gap:10px;"><span style="background:#FDF1F1;color:#DC4545;padding:3px 10px;border-radius:14px;font-size:0.75rem;font-weight:700;">High Risk</span><span style="font-size:0.78rem;color:#6D667F;">Investigate near-duplicate records</span></div>
</div>
</div>""",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Card: Records to review
    st.markdown(
        """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;">
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;margin-bottom:2px;">Records to review</div>
<div style="font-size:0.8rem;color:#6D667F;margin-bottom:12px;">Potentially close matches will appear here after analysis.</div>""",
        unsafe_allow_html=True
    )

    if has_priv:
        dists = privacy_info['distance_distribution']
        sorted_indices = np.argsort(dists)[:6]
        
        review_data = []
        for rank, idx in enumerate(sorted_indices, 1):
            d_val = dists[idx]
            status = "Suspicious Match" if d_val <= privacy_info['threshold_used'] else "Safe Distance"
            review_data.append({
                "Synthetic record": f"Record #{idx + 1}",
                "Nearest original record": f"Vector #{rank}",
                "Distance": f"{d_val:.4f}",
                "Review status": status
            })
            
        st.dataframe(pd.DataFrame(review_data), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        render_empty_box("📋", "No calculated results. Run an analysis to identify records that need review.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Bottom Callout
    st.markdown(
        """<div style="background:#FAF8FE;border:1px solid #ECE7F5;border-radius:10px;padding:14px 18px;">
<div style="font-size:0.85rem;font-weight:700;color:#1E1534;margin-bottom:4px;">ℹ What these results can &mdash; and can't &mdash; tell you</div>
<div style="font-size:0.78rem;color:#6D667F;line-height:1.5;">Nearest-neighbour distance measures similarity, not anonymity. Results depend on feature scaling, distance metric and the chosen threshold. Low observed similarity does not prove privacy or rule out membership inference. Review the methodology before sharing generated data.</div>
</div>""",
        unsafe_allow_html=True
    )
