"""
SynData AI - Global Clean SaaS Stylesheet
Deep Purple & White clean modern design system without fragile nested HTML blocks.
"""

import streamlit as st

def apply_custom_styles():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

            /* Global App Typography & Theme */
            html, body, [class*="css"] {
                font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
                color: #1E1534;
            }

            .stApp {
                background-color: #FFFFFF !important;
            }

            .main .block-container {
                padding-top: 1.5rem !important;
                padding-bottom: 3.5rem !important;
                max-width: 1250px !important;
            }

            /* Sidebar Styling - Light White with Clean Violet Border */
            section[data-testid="stSidebar"] {
                background-color: #FFFFFF !important;
                border-right: 1px solid #ECE7F5 !important;
                width: 270px !important;
            }

            section[data-testid="stSidebar"] div.block-container {
                padding-top: 1.25rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }

            /* Sidebar Buttons */
            section[data-testid="stSidebar"] .stButton > button {
                width: 100% !important;
                border-radius: 8px !important;
                padding: 9px 12px !important;
                text-align: left !important;
                font-weight: 500 !important;
                font-size: 0.88rem !important;
                border: 1px solid transparent !important;
                background-color: transparent !important;
                color: #5F5870 !important;
                transition: all 0.15s ease !important;
                display: flex !important;
                justify-content: flex-start !important;
                align-items: center !important;
                gap: 8px !important;
                margin-bottom: 4px !important;
            }

            section[data-testid="stSidebar"] .stButton > button:hover {
                background-color: #F8F6FE !important;
                color: #6C3BDB !important;
            }

            section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
                background-color: #EDE8FD !important;
                color: #6C3BDB !important;
                font-weight: 700 !important;
                border-left: 3px solid #6C3BDB !important;
                border-radius: 8px !important;
            }

            /* Primary Action Buttons */
            .main .stButton > button[kind="primary"] {
                background-color: #6C3BDB !important;
                color: #FFFFFF !important;
                border-radius: 8px !important;
                font-weight: 600 !important;
                font-size: 0.88rem !important;
                padding: 8px 18px !important;
                border: none !important;
                box-shadow: 0 1px 3px rgba(108, 59, 219, 0.25) !important;
                transition: all 0.15s ease !important;
            }

            .main .stButton > button[kind="primary"]:hover {
                background-color: #5629BD !important;
                box-shadow: 0 3px 8px rgba(108, 59, 219, 0.35) !important;
            }

            .main .stButton > button:not([kind="primary"]) {
                background-color: #FFFFFF !important;
                color: #1E1534 !important;
                border: 1px solid #ECE7F5 !important;
                border-radius: 8px !important;
                font-weight: 500 !important;
                font-size: 0.86rem !important;
                padding: 8px 16px !important;
                transition: all 0.15s ease !important;
            }

            .main .stButton > button:not([kind="primary"]):hover {
                border-color: #6C3BDB !important;
                color: #6C3BDB !important;
                background-color: #FAF8FE !important;
            }

            /* Metric Cards Styling */
            div[data-testid="stMetric"] {
                background-color: #FFFFFF !important;
                border: 1px solid #ECE7F5 !important;
                border-radius: 12px !important;
                padding: 14px 18px !important;
                box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
            }

            div[data-testid="stMetricLabel"] {
                font-size: 0.78rem !important;
                font-weight: 600 !important;
                color: #6D667F !important;
                text-transform: capitalize !important;
            }

            div[data-testid="stMetricValue"] {
                font-size: 1.45rem !important;
                font-weight: 700 !important;
                color: #1E1534 !important;
            }

            /* Clean Form Inputs */
            div[data-baseweb="input"] > div, 
            div[data-baseweb="select"] > div {
                border-radius: 8px !important;
                border-color: #ECE7F5 !important;
            }

            div[data-baseweb="input"] > div:focus-within, 
            div[data-baseweb="select"] > div:focus-within {
                border-color: #6C3BDB !important;
                box-shadow: 0 0 0 1px #6C3BDB !important;
            }

            /* Table Borders */
            div[data-testid="stDataFrame"] {
                border: 1px solid #ECE7F5 !important;
                border-radius: 10px !important;
            }

            /* Hide Default Streamlit Menu / Footer */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header[data-testid="stHeader"] {background-color: transparent !important;}
        </style>
        """,
        unsafe_allow_html=True
    )
