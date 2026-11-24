"""
SynData AI - Generate Synthetic Data Page
Clean, robust, and reliable generative model training.
"""

import streamlit as st
import pandas as pd
from ui.components import render_top_bar, render_step_header, render_empty_box
from core.synthesizer import train_and_generate

def render_generate_view():
    render_top_bar("Generate Synthetic Data")

    has_real = st.session_state.get("real_df") is not None
    has_syn = st.session_state.get("syn_df") is not None
    real_df = st.session_state.get("real_df")
    dataset_name = st.session_state.get("dataset_name", "No dataset selected")

    render_step_header(
        step_tag="STEP 02 / GENERATE",
        title="Create your synthetic dataset",
        subtitle="Choose a model, configure your run and turn source data into new possibilities."
    )

    if not has_real:
        st.markdown(
            """<div style="background:#EDE8FD;border-radius:10px;padding:12px 18px;display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;">
<div style="font-size:0.85rem;color:#1E1534;">ℹ Upload a dataset to unlock generation.</div>
</div>""",
            unsafe_allow_html=True
        )

    col_config, col_status = st.columns([1.7, 1])

    with col_config:
        st.markdown(
            """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:18px 20px;">
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;margin-bottom:2px;">Model configuration</div>
<div style="font-size:0.8rem;color:#6D667F;margin-bottom:14px;">Configure how your synthetic records are generated.</div>""",
            unsafe_allow_html=True
        )

        options_list = [dataset_name] if has_real else ["No dataset selected"]
        st.selectbox("Source dataset", options=options_list, index=0, disabled=not has_real)

        c_m1, c_m2 = st.columns(2)
        with c_m1:
            model_selected = st.selectbox("Model", options=["CTGAN", "TVAE"], index=0)
            model_sub = "Adversarial model for tabular data" if model_selected == "CTGAN" else "Variational autoencoder for tabular data"
            st.caption(model_sub)

        with c_m2:
            epochs = st.number_input("Training epochs", min_value=5, max_value=300, value=25, step=5)
            st.caption("More epochs may increase training time")

        c_s1, c_s2 = st.columns(2)
        with c_s1:
            default_samples = min(len(real_df), 1000) if has_real else 500
            n_samples = st.number_input("Synthetic records", min_value=10, max_value=10000, value=default_samples, step=50)
            st.caption("Number of new records to generate")

        with c_s2:
            seed = st.number_input("Random seed (optional)", min_value=1, max_value=99999, value=42, step=1)
            st.caption("Set a seed for reproducible runs")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_status:
        run_badge = "Completed" if has_syn else "Not started"
        st.markdown(
            f"""<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:18px 20px;height:100%;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;">
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;">Run status</div>
<div style="background:#F4F2F8;color:#6D667F;padding:3px 10px;border-radius:12px;font-size:0.72rem;font-weight:600;">{run_badge}</div>
</div>
<div style="font-size:0.8rem;color:#6D667F;margin-bottom:16px;">Your next run starts here.</div>
<div style="display:flex;flex-direction:column;gap:12px;">
<div style="display:flex;gap:10px;align-items:center;"><div style="width:24px;height:24px;border-radius:6px;background:#EDE8FD;color:#6C3BDB;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;">1</div><div><div style="font-size:0.82rem;font-weight:700;color:#1E1534;">Dataset validation</div><div style="font-size:0.72rem;color:#6D667F;">Check your source data</div></div></div>
<div style="display:flex;gap:10px;align-items:center;"><div style="width:24px;height:24px;border-radius:6px;background:#EDE8FD;color:#6C3BDB;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;">2</div><div><div style="font-size:0.82rem;font-weight:700;color:#1E1534;">Model training</div><div style="font-size:0.72rem;color:#6D667F;">Learn the data patterns</div></div></div>
<div style="display:flex;gap:10px;align-items:center;"><div style="width:24px;height:24px;border-radius:6px;background:#EDE8FD;color:#6C3BDB;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;">3</div><div><div style="font-size:0.82rem;font-weight:700;color:#1E1534;">Synthetic generation</div><div style="font-size:0.72rem;color:#6D667F;">Create new records</div></div></div>
<div style="display:flex;gap:10px;align-items:center;"><div style="width:24px;height:24px;border-radius:6px;background:#EDE8FD;color:#6C3BDB;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;">4</div><div><div style="font-size:0.82rem;font-weight:700;color:#1E1534;">Ready to evaluate</div><div style="font-size:0.72rem;color:#6D667F;">Compare quality and privacy</div></div></div>
</div>
</div>""",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Action bar below configuration
    b_left, b_right = st.columns([3.5, 1.4])
    with b_left:
        note = "Ready to train. Click to start synthesis run." if has_real else "Default settings are shown. Upload your dataset to edit and start a run."
        st.markdown(f"<div style='font-size:0.82rem;color:#6D667F;padding-top:10px;'>{note}</div>", unsafe_allow_html=True)
    with b_right:
        if has_real:
            generate_clicked = st.button("✦ Generate Synthetic Data", type="primary", use_container_width=True, key="do_gen_btn")
        else:
            generate_clicked = False
            st.button("✦ Generate Synthetic Data", disabled=True, use_container_width=True)

    if generate_clicked and has_real:
        p_bar = st.progress(0, text="Initializing model architecture...")
        try:
            def on_progress(pct, msg):
                p_bar.progress(int(pct * 100), text=msg)

            syn_df, exec_info = train_and_generate(
                df=real_df,
                model_type=model_selected,
                epochs=int(epochs),
                batch_size=200,
                num_samples=int(n_samples),
                random_seed=int(seed),
                progress_callback=on_progress
            )
            st.session_state.syn_df = syn_df
            st.session_state.model_name = model_selected
            st.session_state.exec_info = exec_info
            st.session_state.ml_results = None
            p_bar.empty()
            st.toast(f"Generated {len(syn_df):,} records with {model_selected}!", icon="✨")
            st.rerun()
        except Exception as e:
            p_bar.empty()
            st.error(f"Generative training error: {e}")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Generated records section
    st.markdown(
        """<div style="background:#FFFFFF;border:1px solid #ECE7F5;border-radius:12px;padding:16px 20px;margin-bottom:16px;">
<div style="font-size:0.95rem;font-weight:700;color:#1E1534;margin-bottom:2px;">Generated records</div>
<div style="font-size:0.8rem;color:#6D667F;margin-bottom:12px;">A preview will appear when generation completes.</div>""",
        unsafe_allow_html=True
    )

    if has_syn:
        syn_df = st.session_state.syn_df
        st.dataframe(syn_df.head(10), use_container_width=True)
        csv_bytes = syn_df.to_csv(index=False).encode('utf-8')
        d1, d2 = st.columns([1.2, 2])
        with d1:
            st.download_button(
                label="📥 Download CSV",
                data=csv_bytes,
                file_name=f"syn_data_{st.session_state.get('model_name', 'ctgan').lower()}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
        with d2:
            if st.button("Proceed to Data Comparison ➔", use_container_width=True):
                st.session_state["nav"] = "comparison"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        render_empty_box("🗄️", "No synthetic records yet", "Configure settings and click generate above to produce synthetic records.")
        st.markdown("</div>", unsafe_allow_html=True)
