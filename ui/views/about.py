"""
SynData AI - About Project Page
"""

import streamlit as st
import config
from ui.components import render_page_header

def render_about_view():
    render_page_header(
        title="About SynData AI",
        subtitle="Next-generation generative platform for privacy-preserving synthetic tabular data.",
        icon="💡"
    )

    # Core Mission & Vision
    st.markdown(
        f"""
        <div class="syndata-card" style="border-left: 5px solid {config.COLOR_PRIMARY_PURPLE};">
            <h3 style="color: {config.COLOR_DARK_PURPLE}; margin: 0 0 10px 0; font-size: 1.3rem;">
                Empowering Safe Data Sharing Through Generative AI
            </h3>
            <p style="color: {config.COLOR_PRIMARY_TEXT}; font-size: 0.95rem; line-height: 1.6; margin: 0;">
                <b>SynData AI</b> solves one of modern computing's greatest dilemmas: the tension between extracting maximum value from enterprise data assets and strictly preserving confidential user privacy. By leveraging state-of-the-art deep generative models including Conditional GANs (CTGAN) and Variational Autoencoders (TVAE), the platform synthesizes mathematically coherent artificial datasets that capture empirical relationships without reproducing sensitive original patient, financial, or consumer records.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="syndata-card" style="height: 100%;">
                <div class="card-title">🚨 The Problem</div>
                <div class="card-description" style="font-size: 0.9rem; line-height: 1.6;">
                    Real-world datasets contain critical intellectual property, proprietary business metrics, and Personally Identifiable Information (PII).
                    Strict regulatory frameworks such as <b>GDPR, HIPAA, and CCPA</b> make cross-departmental sharing, vendor onboarding, and open machine learning research nearly impossible due to re-identification liabilities and data breach risks.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="syndata-card" style="height: 100%;">
                <div class="card-title">💡 The Solution</div>
                <div class="card-description" style="font-size: 0.9rem; line-height: 1.6;">
                    SynData AI trains unsupervised neural networks that learn the <b>multimodal continuous and discrete probability distributions</b> governing tabular records.
                    Once trained, the platform samples novel synthetic records from the learned manifold. These records exhibit high statistical fidelity, preserve cross-column correlations, and pass rigorous privacy and ML utility audits.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Technology Stack
    st.markdown(
        f"""
        <div style="margin-bottom: 1rem;">
            <h3 style="color: {config.COLOR_DARK_PURPLE}; font-size: 1.2rem; font-weight: 700; margin-bottom: 4px;">
                Technology Stack & System Architecture
            </h3>
            <p style="color: {config.COLOR_SECONDARY_TEXT}; font-size: 0.88rem; margin: 0;">
                Built on industry-standard open-source machine learning and visualization technologies.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown(
            f"""
            <div class="step-card">
                <div style="font-size: 1.5rem; margin-bottom: 6px;">🧠</div>
                <div class="step-title">Generative AI</div>
                <div class="step-desc">CTGAN, TVAE, PyTorch, RDT (Reversible Data Transforms)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with t2:
        st.markdown(
            f"""
            <div class="step-card">
                <div style="font-size: 1.5rem; margin-bottom: 6px;">📊</div>
                <div class="step-title">Data & Statistics</div>
                <div class="step-desc">Pandas, NumPy, SciPy (Two-Sample KS Test), Scikit-learn</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with t3:
        st.markdown(
            f"""
            <div class="step-card">
                <div style="font-size: 1.5rem; margin-bottom: 6px;">🎨</div>
                <div class="step-title">Frontend & UI</div>
                <div class="step-desc">Streamlit, Custom CSS, Plotly Interactive Visuals, HTML5</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with t4:
        st.markdown(
            f"""
            <div class="step-card">
                <div style="font-size: 1.5rem; margin-bottom: 6px;">📑</div>
                <div class="step-title">Audit & Reporting</div>
                <div class="step-desc">ReportLab 5.0 (PDF Engine), High-dimensional Nearest Neighbors</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Industry Applications
    st.markdown(
        f"""
        <div style="margin-bottom: 1rem;">
            <h3 style="color: {config.COLOR_DARK_PURPLE}; font-size: 1.2rem; font-weight: 700; margin-bottom: 4px;">
                Industry Applications
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    app1, app2, app3, app4, app5 = st.columns(5)
    with app1:
        st.markdown(
            f"""
            <div class="syndata-card" style="text-align: center; padding: 1.2rem 0.8rem;">
                <div style="font-size: 1.8rem; margin-bottom: 6px;">🏥</div>
                <div style="font-weight: 700; color: {config.COLOR_DARK_PURPLE}; font-size: 0.92rem;">Healthcare</div>
                <div style="font-size: 0.78rem; color: {config.COLOR_SECONDARY_TEXT}; margin-top: 4px;">HIPAA-safe clinical trial sharing and medical research.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with app2:
        st.markdown(
            f"""
            <div class="syndata-card" style="text-align: center; padding: 1.2rem 0.8rem;">
                <div style="font-size: 1.8rem; margin-bottom: 6px;">🏦</div>
                <div style="font-weight: 700; color: {config.COLOR_DARK_PURPLE}; font-size: 0.92rem;">Banking & Finance</div>
                <div style="font-size: 0.78rem; color: {config.COLOR_SECONDARY_TEXT}; margin-top: 4px;">Synthetic credit risk, fraud modeling, and stress testing.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with app3:
        st.markdown(
            f"""
            <div class="syndata-card" style="text-align: center; padding: 1.2rem 0.8rem;">
                <div style="font-size: 1.8rem; margin-bottom: 6px;">🎓</div>
                <div style="font-weight: 700; color: {config.COLOR_DARK_PURPLE}; font-size: 0.92rem;">Education & EdTech</div>
                <div style="font-size: 0.78rem; color: {config.COLOR_SECONDARY_TEXT}; margin-top: 4px;">Student outcome modeling without exposing FERPA records.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with app4:
        st.markdown(
            f"""
            <div class="syndata-card" style="text-align: center; padding: 1.2rem 0.8rem;">
                <div style="font-size: 1.8rem; margin-bottom: 6px;">🔬</div>
                <div style="font-weight: 700; color: {config.COLOR_DARK_PURPLE}; font-size: 0.92rem;">AI Research</div>
                <div style="font-size: 0.78rem; color: {config.COLOR_SECONDARY_TEXT}; margin-top: 4px;">Augmenting rare classes and synthetic data benchmark suites.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with app5:
        st.markdown(
            f"""
            <div class="syndata-card" style="text-align: center; padding: 1.2rem 0.8rem;">
                <div style="font-size: 1.8rem; margin-bottom: 6px;">💻</div>
                <div style="font-weight: 700; color: {config.COLOR_DARK_PURPLE}; font-size: 0.92rem;">Software QA</div>
                <div style="font-size: 0.78rem; color: {config.COLOR_SECONDARY_TEXT}; margin-top: 4px;">Realistic staging environments and load testing databases.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
