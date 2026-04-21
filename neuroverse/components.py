from typing import Dict, List

import streamlit as st


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f"<p class='nv-subtitle'>{subtitle}</p>", unsafe_allow_html=True)


def text_card(title: str, body_md: str) -> None:
    st.markdown(
        f"""
        <div class=\"nv-card\">
            <div class=\"nv-card-title\">{title}</div>
            <div class=\"nv-muted\">{body_md}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_how_it_works(steps: List[str]) -> None:
    st.subheader("How it Works")
    cols = st.columns(len(steps))
    for idx, step in enumerate(steps):
        with cols[idx]:
            text_card(f"Step {idx + 1}", step)


def render_use_cases(use_cases: List[str]) -> None:
    st.subheader("Use Cases")
    cols = st.columns(len(use_cases))
    for idx, name in enumerate(use_cases):
        with cols[idx]:
            text_card(name, "Practical neural network application scenario.")


def render_key_terms(terms: Dict[str, str]) -> None:
    st.subheader("Key Terms")
    parts = []
    for term, meaning in terms.items():
        parts.append(f"<p><strong>{term}:</strong> {meaning}</p>")
    text_card("Beginner Glossary", "".join(parts))
