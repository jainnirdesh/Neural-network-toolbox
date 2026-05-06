from typing import Optional

import streamlit as st


def glass_card(title: str, body: str, footer: Optional[str] = None) -> None:
    """Render a glassmorphism-style card using HTML/CSS and Streamlit markdown.

    Keep HTML minimal so it works across Streamlit deployments.
    """
    st.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.04);
            backdrop-filter: blur(8px);
            padding: 18px;
            border-radius: 12px;
            margin-bottom: 12px;
            transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
        " onmouseover="this.style.transform='translateY(-6px)'; this.style.boxShadow='0 8px 30px rgba(0,0,0,0.5)';" onmouseout="this.style.transform=''; this.style.boxShadow='';">
            <div style="font-weight:700; font-size:1.05rem; color:var(--text-primary);">{title}</div>
            <div style="color:var(--text-secondary); margin-top:6px;">{body}</div>
            {f'<div style="margin-top:8px; color:var(--text-tertiary); font-size:0.9rem">{footer}</div>' if footer else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def loading_spinner(message: str = "Loading..."):
    """Small helper to display a centered loading message and spinner."""
    with st.spinner(message):
        pass
