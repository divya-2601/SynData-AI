"""
SynData AI - Reports & Downloads Page
Clean export cards for CSV datasets and PDF audit reports.
"""

import streamlit as st
import pandas as pd
from ui.components import render_top_bar, render_step_header
from core.pdf_reports import (
    generate_statistical_pdf,
    generate_privacy_pdf,
    generate_ml_eval_pdf
)

def render_reports_view():
    render_top_bar("Reports & Downloads")

    real_df = st.session_state.get("real_df")
    syn_df = st.session_state.get("syn_df")
    dataset_name = st.session_state.get("dataset_name", "dataset.csv")
    model_name = st.session_state.get("model_name", "CTGAN")
    quality_info = st.session_state.get("quality_info")
    stat_df = st.session_state.get("stat_df")
    ks_df = st.session_state.get("ks_df")
    privacy_info = st.session_state.get("privacy_info")
    ml_info = st.session_state.get("ml_results")

    ready_count = 0
    if syn_df is not None: ready_count += 1
    if quality_info is not None: ready_count += 1
    if privacy_info is not None: ready_count += 1
    if ml_info is not None: ready_count += 1

    render_step_header(
        step_tag="STEP 04 / EXPORT",
        title="Your insights, ready to share",
        subtitle="Export your synthetic dataset and the evidence behind it, all in one place."
    )

    # Top Banner Box
    st.markdown(
        f"""<div style="background:#FAF8FE;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;">
<div style="display:flex;align-items:center;gap:12px;">
<div style="width:36px;height:36px;border-radius:9px;background:#EDE8FD;color:#6C3BDB;display:flex;align-items:center;justify-content:center;font-size:16px;">↓</div>
<div>
<div style="font-size:0.92rem;font-weight:700;color:#1E1534;">Your report collection starts here</div>
<div style="font-size:0.78rem;color:#6D667F;">Complete a generation run and evaluations to unlock your downloads.</div>
</div>
</div>
<div style="background:#F4F2F8;color:#6D667F;padding:4px 12px;border-radius:14px;font-size:0.75rem;font-weight:600;">{ready_count} files ready</div>
</div>""",
        unsafe_allow_html=True
    )

    # 4 Cards in 2x2 Grid
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;min-height:160px;display:flex;flex-direction:column;justify-content:space-between;">
<div>
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
<div style="width:30px;height:30px;border-radius:8px;background:#EDE8FD;color:#6C3BDB;display:flex;align-items:center;justify-content:center;font-size:14px;">🗄️</div>
<span style="background:#F4F2F8;color:#6D667F;padding:2px 8px;border-radius:10px;font-size:0.72rem;font-weight:600;">CSV</span>
</div>
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;">Synthetic Dataset</div>
<div style="font-size:0.78rem;color:#6D667F;margin-top:4px;">New synthetic records, ready for analysis and downstream machine learning.</div>
</div>
</div>""",
            unsafe_allow_html=True
        )
        if syn_df is not None:
            csv_bytes = syn_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="↓ Download CSV",
                data=csv_bytes,
                file_name=f"syndata_records_{model_name.lower()}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
                key="rep_csv_dl_new"
            )
        else:
            st.button("↓ Download CSV", disabled=True, use_container_width=True, key="rep_csv_dis_new")

    with c2:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;min-height:160px;display:flex;flex-direction:column;justify-content:space-between;">
<div>
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
<div style="width:30px;height:30px;border-radius:8px;background:#EDE8FD;color:#6C3BDB;display:flex;align-items:center;justify-content:center;font-size:14px;">📊</div>
<span style="background:#F4F2F8;color:#6D667F;padding:2px 8px;border-radius:10px;font-size:0.72rem;font-weight:600;">PDF</span>
</div>
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;">Data Quality Report</div>
<div style="font-size:0.78rem;color:#6D667F;margin-top:4px;">Distribution comparisons, summary statistics and correlation analysis in one report.</div>
</div>
</div>""",
            unsafe_allow_html=True
        )
        if quality_info is not None and syn_df is not None:
            try:
                pdf_buf = generate_statistical_pdf(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    real_df=real_df,
                    syn_df=syn_df,
                    stat_df=stat_df if stat_df is not None else pd.DataFrame(),
                    ks_df=ks_df if ks_df is not None else pd.DataFrame(),
                    quality_info=quality_info
                )
                st.download_button(
                    label="↓ Download report",
                    data=pdf_buf,
                    file_name=f"syndata_quality_report_{dataset_name}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                    key="rep_qual_dl_new"
                )
            except Exception as e:
                st.error(f"Error compiling PDF: {e}")
        else:
            st.button("↓ Download report", disabled=True, use_container_width=True, key="rep_qual_dis_new")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;min-height:160px;display:flex;flex-direction:column;justify-content:space-between;">
<div>
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
<div style="width:30px;height:30px;border-radius:8px;background:#EDE8FD;color:#6C3BDB;display:flex;align-items:center;justify-content:center;font-size:14px;">🛡️</div>
<span style="background:#F4F2F8;color:#6D667F;padding:2px 8px;border-radius:10px;font-size:0.72rem;font-weight:600;">REPORT</span>
</div>
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;">Privacy Analysis Report</div>
<div style="font-size:0.78rem;color:#6D667F;margin-top:4px;">Nearest-neighbour findings, similar records and the limitations of your privacy analysis.</div>
</div>
</div>""",
            unsafe_allow_html=True
        )
        if privacy_info is not None:
            try:
                pdf_buf = generate_privacy_pdf(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    privacy_info=privacy_info
                )
                st.download_button(
                    label="↓ Download report",
                    data=pdf_buf,
                    file_name=f"syndata_privacy_report_{dataset_name}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                    key="rep_priv_dl_new"
                )
            except Exception as e:
                st.error(f"Error compiling PDF: {e}")
        else:
            st.button("↓ Download report", disabled=True, use_container_width=True, key="rep_priv_dis_new")

    with c4:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;min-height:160px;display:flex;flex-direction:column;justify-content:space-between;">
<div>
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
<div style="width:30px;height:30px;border-radius:8px;background:#EDE8FD;color:#6C3BDB;display:flex;align-items:center;justify-content:center;font-size:14px;">📄</div>
<span style="background:#F4F2F8;color:#6D667F;padding:2px 8px;border-radius:10px;font-size:0.72rem;font-weight:600;">REPORT</span>
</div>
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;">ML Evaluation Report</div>
<div style="font-size:0.78rem;color:#6D667F;margin-top:4px;">Train-on-synthetic, test-on-real metrics compared with a real-trained baseline.</div>
</div>
</div>""",
            unsafe_allow_html=True
        )
        if ml_info is not None:
            try:
                pdf_buf = generate_ml_eval_pdf(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    ml_info=ml_info
                )
                st.download_button(
                    label="↓ Download report",
                    data=pdf_buf,
                    file_name=f"syndata_ml_report_{dataset_name}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                    key="rep_ml_dl_new"
                )
            except Exception as e:
                st.error(f"Error compiling PDF: {e}")
        else:
            st.button("↓ Download report", disabled=True, use_container_width=True, key="rep_ml_dis_new")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Bottom Callout: Before you export
    st.markdown(
        """<div style="background:#FAF8FE;border:1px solid #ECE7F5;border-radius:10px;padding:14px 18px;">
<div style="font-size:0.85rem;font-weight:700;color:#1E1534;margin-bottom:4px;">ℹ Before you export</div>
<div style="font-size:0.78rem;color:#6D667F;line-height:1.5;">Review data quality, privacy findings and model performance together before using or sharing your synthetic dataset. Reports only become available after their corresponding analysis is complete. No sample results are included.</div>
</div>""",
        unsafe_allow_html=True
    )
