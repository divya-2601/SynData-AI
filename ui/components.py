"""
SynData AI - Core UI Component Helpers
Cleanly renders headers, breadcrumbs, and metrics without markdown indentation issues.
"""

import streamlit as st
import plotly.graph_objects as go
import config

def render_top_bar(page_name: str):
    """Renders top breadcrumb bar with status badge."""
    html = f"""<div style="display:flex;justify-content:space-between;align-items:center;padding-bottom:14px;border-bottom:1px solid #ECE7F5;margin-bottom:20px;">
<div style="font-size:0.85rem;color:#6D667F;display:flex;align-items:center;gap:6px;">
<span>⊞</span><span>Workspace</span><span>&gt;</span><span style="color:#1E1534;font-weight:600;">{page_name}</span>
</div>
<div style="background:#FAF9FD;border:1px solid #ECE7F5;border-radius:20px;padding:4px 12px;font-size:0.75rem;color:#6D667F;display:flex;align-items:center;gap:6px;">
<span style="width:7px;height:7px;background:#16A36A;border-radius:50%;display:inline-block;"></span>
<span>Local workspace</span><span>&bull;</span><span>Ready</span>
</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)

def render_step_header(step_tag: str, title: str, subtitle: str, right_badge: str = ""):
    """Renders step tag, heading, and subtitle cleanly."""
    badge_html = f'<div style="background:#F4F2F8;color:#6D667F;padding:4px 12px;border-radius:16px;font-size:0.75rem;font-weight:600;">{right_badge}</div>' if right_badge else ''
    html = f"""<div style="margin-bottom:20px;">
<div style="display:flex;justify-content:space-between;align-items:flex-start;">
<div>
<div style="font-size:0.72rem;font-weight:700;color:#6C3BDB;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">{step_tag}</div>
<div style="font-size:1.65rem;font-weight:700;color:#1E1534;letter-spacing:-0.02em;margin-bottom:4px;">{title}</div>
</div>
{badge_html}
</div>
<div style="font-size:0.88rem;color:#6D667F;line-height:1.45;">{subtitle}</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)

def render_empty_box(icon: str, title: str, subtitle: str = ""):
    """Renders a clean empty state placeholder."""
    sub_html = f'<div style="font-size:0.78rem;color:#948CA5;margin-top:4px;">{subtitle}</div>' if subtitle else ''
    html = f"""<div style="text-align:center;padding:36px 16px;background:#FAF9FD;border:1px dashed #ECE7F5;border-radius:12px;margin:8px 0;">
<div style="width:38px;height:38px;border-radius:9px;background:#EDE8FD;color:#6C3BDB;display:inline-flex;align-items:center;justify-content:center;font-size:18px;margin-bottom:8px;">{icon}</div>
<div style="font-size:0.88rem;font-weight:600;color:#6D667F;">{title}</div>
{sub_html}
</div>"""
    st.markdown(html, unsafe_allow_html=True)

def apply_plotly_theme(fig: go.Figure, height: int = 340) -> go.Figure:
    """Styles Plotly figures to match the clean Deep Purple & White theme."""
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAF9FD",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#1E1534", size=11),
        margin=dict(l=35, r=25, t=35, b=35),
        height=height,
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#ECE7F5",
            borderwidth=1,
            font=dict(size=10.5)
        ),
        xaxis=dict(
            gridcolor="#ECE7F5",
            zerolinecolor="#ECE7F5",
            showline=True,
            linecolor="#ECE7F5",
            tickfont=dict(color="#6D667F", size=10)
        ),
        yaxis=dict(
            gridcolor="#ECE7F5",
            zerolinecolor="#ECE7F5",
            showline=True,
            linecolor="#ECE7F5",
            tickfont=dict(color="#6D667F", size=10)
        )
    )
    return fig
