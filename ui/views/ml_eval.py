"""
SynData AI - Machine Learning Utility Evaluation Page
Train on Synthetic, Test on Real (TSTR) benchmark for any tabular dataset.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import config
from ui.components import render_top_bar, render_step_header, render_empty_box, apply_plotly_theme
from core.ml_eval import identify_candidate_targets, evaluate_ml_utility

def render_ml_eval_view():
    render_top_bar("ML Evaluation")

    has_real = st.session_state.get("real_df") is not None
    has_syn = st.session_state.get("syn_df") is not None
    real_df = st.session_state.get("real_df")
    syn_df = st.session_state.get("syn_df")

    c_head, c_btn = st.columns([3.5, 1.4])
    with c_head:
        render_step_header(
            step_tag="STEP 03 / MODEL UTILITY",
            title="Real-world utility, measured",
            subtitle="Train on synthetic data. Test on real data. Understand what your model can learn."
        )

    candidates = identify_candidate_targets(real_df) if has_real else ["Select a target after upload"]

    # Filter Bar
    c_t1, c_t2, c_badge = st.columns([1.5, 2, 1.2])
    with c_t1:
        target_col = st.selectbox("Target column", options=candidates, index=0, disabled=not has_real)
    with c_t2:
        st.text_input("Evaluation method", value="Train on Synthetic, Test on Real (TSTR)", disabled=True)
    with c_badge:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        eval_badge = "Ready to run" if (has_real and has_syn) else "No evaluation yet"
        st.markdown(f"<div style='text-align:right;'><span style='background:#F4F2F8;color:#6D667F;padding:4px 12px;border-radius:14px;font-size:0.75rem;font-weight:600;'>{eval_badge}</span></div>", unsafe_allow_html=True)

    with c_btn:
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        if has_real and has_syn and target_col in real_df.columns:
            eval_clicked = st.button("📊 Run evaluation", type="primary", use_container_width=True, key="do_ml_eval")
        else:
            eval_clicked = False
            st.button("📊 Run evaluation", disabled=True, use_container_width=True)

    if eval_clicked and has_real and has_syn and target_col in real_df.columns:
        with st.spinner(f"Evaluating TSTR model on target '{target_col}'..."):
            try:
                res = evaluate_ml_utility(
                    real_df=real_df,
                    syn_df=syn_df,
                    target_column=target_col,
                    test_size=0.30,
                    random_state=42
                )
                st.session_state.ml_results = res
            except Exception as e:
                st.error(f"Evaluation error: {e}")

    ml_res = st.session_state.get("ml_results")
    has_res = ml_res is not None

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 4 Metric Cards using native Streamlit metrics
    acc_val = f"{ml_res['tstr_accuracy']:.3f}" if has_res else "—"
    f1_val = f"{ml_res['tstr_f1']:.3f}" if has_res else "—"
    
    if has_res and "metrics_table" in ml_res:
        m_tbl = ml_res["metrics_table"]
        p_row = m_tbl[m_tbl["Metric"] == "Precision"]
        r_row = m_tbl[m_tbl["Metric"] == "Recall"]
        prec_val = f"{p_row['Synthetic Model (TSTR)'].values[0]:.3f}" if not p_row.empty else "—"
        rec_val = f"{r_row['Synthetic Model (TSTR)'].values[0]:.3f}" if not r_row.empty else "—"
    else:
        prec_val = "—"
        rec_val = "—"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Accuracy", value=acc_val)
    with m2:
        st.metric(label="Precision", value=prec_val)
    with m3:
        st.metric(label="Recall", value=rec_val)
    with m4:
        st.metric(label="F1-score", value=f1_val)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Middle Row: Model performance comparison & A fair comparison
    col_perf, col_fair = st.columns([1.6, 1])

    with col_perf:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;height:100%;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;">
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;">Model performance comparison</div>
<div style="font-size:0.75rem;color:#6D667F;">
<span style="color:#6C3BDB;">●</span> Synthetic-trained &nbsp; <span style="color:#388BFD;">●</span> Real baseline
</div>
</div>
<div style="font-size:0.8rem;color:#6D667F;margin-bottom:12px;">Synthetic-trained model vs. real-trained baseline on the same real test set.</div>""",
            unsafe_allow_html=True
        )

        if has_res and "metrics_table" in ml_res:
            m_df = ml_res["metrics_table"]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=m_df["Metric"],
                y=m_df["Synthetic Model (TSTR)"],
                name="Synthetic-trained",
                marker_color=config.COLOR_PRIMARY_PURPLE
            ))
            fig_bar.add_trace(go.Bar(
                x=m_df["Metric"],
                y=m_df["Real Model (TRTR)"],
                name="Real baseline",
                marker_color="#388BFD"
            ))
            fig_bar.update_layout(barmode="group", yaxis=dict(range=[0, 1.05]))
            apply_plotly_theme(fig_bar, height=260)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            render_empty_box("📊", "Your model comparison starts here", "Run an evaluation to calculate metrics and compare performance.")
            st.markdown("</div>", unsafe_allow_html=True)

    with col_fair:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;height:100%;display:flex;flex-direction:column;justify-content:space-between;">
<div>
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;margin-bottom:2px;">A fair comparison</div>
<div style="font-size:0.8rem;color:#6D667F;margin-bottom:14px;">Two training sets. One held-out test set.</div>
<div style="background:#FAF8FE;border:1px solid #ECE7F5;border-radius:8px;padding:12px;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;">
<div><div style="font-size:0.82rem;font-weight:700;color:#1E1534;">Train on synthetic</div><div style="font-size:0.72rem;color:#6D667F;">Synthetic model</div></div>
<div style="font-size:0.78rem;color:#6C3BDB;font-weight:600;">&rarr; Test on real</div>
</div>
<div style="background:#FAF8FE;border:1px solid #ECE7F5;border-radius:8px;padding:12px;display:flex;justify-content:space-between;align-items:center;">
<div><div style="font-size:0.82rem;font-weight:700;color:#1E1534;">Train on real</div><div style="font-size:0.72rem;color:#6D667F;">Baseline model</div></div>
<div style="font-size:0.78rem;color:#388BFD;font-weight:600;">&rarr; Test on real</div>
</div>
</div>
<div style="font-size:0.74rem;color:#948CA5;margin-top:14px;">Keep the test data separate from training.</div>
</div>""",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Detailed Results Table Card
    st.markdown(
        """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;">
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;margin-bottom:12px;">Detailed results</div>""",
        unsafe_allow_html=True
    )

    if has_res and "metrics_table" in ml_res:
        m_df = ml_res["metrics_table"]
        get_val = lambda m_name, col: m_df[m_df["Metric"] == m_name][col].values[0] if not m_df[m_df["Metric"] == m_name].empty else 0.0
        
        detail_rows = [
            {
                "Training data": "Real baseline (TRTR)",
                "Accuracy": f"{get_val('Accuracy', 'Real Model (TRTR)'):.4f}",
                "Precision": f"{get_val('Precision', 'Real Model (TRTR)'):.4f}",
                "Recall": f"{get_val('Recall', 'Real Model (TRTR)'):.4f}",
                "F1-score": f"{get_val('F1-Score', 'Real Model (TRTR)'):.4f}"
            },
            {
                "Training data": "Synthetic model (TSTR)",
                "Accuracy": f"{get_val('Accuracy', 'Synthetic Model (TSTR)'):.4f}",
                "Precision": f"{get_val('Precision', 'Synthetic Model (TSTR)'):.4f}",
                "Recall": f"{get_val('Recall', 'Synthetic Model (TSTR)'):.4f}",
                "F1-score": f"{get_val('F1-Score', 'Synthetic Model (TSTR)'):.4f}"
            }
        ]
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        render_empty_box("📋", "No calculated metrics yet. Upload data, generate synthetic records and run an evaluation.")
        st.markdown("</div>", unsafe_allow_html=True)
