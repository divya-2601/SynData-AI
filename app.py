"""
SynData AI - Synthetic Data Generation & Evaluation Platform
Clean, stable Streamlit frontend matching the reference layout.
"""

import streamlit as st

st.set_page_config(
    page_title="SynData AI - Synthetic Data Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

import config
from ui.styles import apply_custom_styles
from ui.views.dashboard import render_dashboard_view
from ui.views.upload import render_upload_view
from ui.views.generate import render_generate_view
from ui.views.comparison import render_comparison_view
from ui.views.privacy import render_privacy_view
from ui.views.ml_eval import render_ml_eval_view
from ui.views.reports import render_reports_view

def init_session_state():
    """Initializes global state variables across multi-page navigation."""
    defaults = {
        "nav": "dashboard",
        "real_df": None,
        "syn_df": None,
        "dataset_name": None,
        "model_name": None,
        "exec_info": None,
        "quality_info": None,
        "stat_df": None,
        "ks_df": None,
        "privacy_info": None,
        "ml_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def render_sidebar():
    """Renders the light enterprise sidebar matching the reference screenshots."""
    with st.sidebar:
        st.markdown(
            """<div style="display:flex;align-items:center;gap:10px;padding:4px 6px 16px 6px;margin-bottom:8px;">
<div style="width:34px;height:34px;background:#6C3BDB;border-radius:9px;display:flex;align-items:center;justify-content:center;color:#FFFFFF;font-size:18px;">🗄️</div>
<div style="font-size:1.2rem;font-weight:700;color:#1E1534;">
SynData <span style="font-size:0.72rem;font-weight:700;color:#6C3BDB;vertical-align:super;">AI</span>
</div>
</div>
<div style="font-size:0.68rem;font-weight:700;color:#948CA5;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;padding-left:8px;">WORKSPACE</div>""",
            unsafe_allow_html=True
        )

        current_nav = st.session_state.get("nav", "dashboard")

        for item in config.NAV_ITEMS:
            item_id = item["id"]
            label = f"{item['icon']}  {item['label']}"
            is_active = (current_nav == item_id)
            btn_kind = "primary" if is_active else "secondary"
            
            if st.button(label, key=f"side_nav_{item_id}", type=btn_kind, use_container_width=True):
                st.session_state["nav"] = item_id
                st.rerun()

        # Sidebar Card: Your first synthetic dataset
        st.markdown(
            """<div style="background:#FAF8FE;border:1px solid #ECE7F5;border-radius:12px;padding:14px;margin-top:24px;margin-bottom:12px;">
<div style="font-size:0.84rem;font-weight:700;color:#1E1534;margin-bottom:2px;">Your first synthetic dataset</div>
<div style="font-size:0.75rem;color:#6D667F;margin-bottom:12px;">A little data. A lot of possibility.</div>
</div>""",
            unsafe_allow_html=True
        )

        if st.button("📖 Quick start guide →", key="sidebar_quickstart_btn", use_container_width=True):
            st.session_state["nav"] = "upload"
            st.rerun()

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        st.markdown(
            """<div>
<div style="display:flex;align-items:center;gap:8px;font-size:0.8rem;color:#6D667F;padding:4px 8px;">
<span>❓</span><span>Help & documentation</span>
</div>
<div style="font-size:0.72rem;color:#948CA5;padding:10px 8px 4px 8px;">
SynData AI / Research workspace
</div>
</div>""",
            unsafe_allow_html=True
        )

def main():
    init_session_state()
    apply_custom_styles()
    render_sidebar()

    current_page = st.session_state.get("nav", "dashboard")

    if current_page == "dashboard":
        render_dashboard_view()
    elif current_page == "upload":
        render_upload_view()
    elif current_page == "generate":
        render_generate_view()
    elif current_page == "comparison":
        render_comparison_view()
    elif current_page == "privacy":
        render_privacy_view()
    elif current_page == "ml_eval":
        render_ml_eval_view()
    elif current_page == "reports":
        render_reports_view()
    else:
        render_dashboard_view()

if __name__ == "__main__":
    main()
