import streamlit as st


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: #0f172a;
                color: #ffffff;
            }
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }
            h1, h2, h3, h4, p, li, span, label {
                color: #ffffff !important;
            }
            .nv-subtitle {
                color: #cbd5e1 !important;
                margin-top: -0.5rem;
                margin-bottom: 1.2rem;
                font-size: 1.05rem;
            }
            .nv-card {
                background: #111c34;
                border: 1px solid #1f2a44;
                border-radius: 14px;
                padding: 1rem 1.1rem;
                margin-bottom: 1rem;
            }
            .nv-card-title {
                font-size: 1.08rem;
                font-weight: 650;
                margin-bottom: 0.45rem;
            }
            .nv-muted {
                color: #cbd5e1 !important;
            }
            [data-testid="stVerticalBlockBorderWrapper"] {
                background: #111c34;
                border: 1px solid #1f2a44;
                border-radius: 14px;
            }
            .stButton button {
                border-radius: 10px;
                border: 1px solid #334155;
                background: #1d4ed8;
                color: white;
                font-weight: 600;
            }
            .stButton button:hover {
                background: #2563eb;
                border-color: #3b82f6;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
