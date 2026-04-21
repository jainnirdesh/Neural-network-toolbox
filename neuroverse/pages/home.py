import streamlit as st

from neuroverse.components import page_header, render_key_terms, text_card


def render_home_page() -> None:
    page_header("NeuroVerse AI Lab", "Learn and experiment with neural-network-powered modules")
    text_card(
        "What is this?",
        """
        NeuroVerse AI Lab is a beginner-friendly interactive environment to understand core machine learning
        modules: Classification, Regression, Image Recognition, and Neural Network visualization.
        """,
    )

    entries = [
        ("Classification", "Learn class probabilities and training curves."),
        ("Regression", "Predict continuous values using fitted trends."),
        ("Image Recognition", "Upload images and inspect top predictions."),
        ("Neural Network Visualizer", "See neurons, layers, and connection flow."),
        ("About", "Project overview and learning goals."),
    ]

    cols = [None, None, None]
    for idx, (name, desc) in enumerate(entries):
        if idx % 3 == 0:
            cols = list(st.columns(3))
        with cols[idx % 3]:
            text_card(name, desc)

    render_key_terms(
        {
            "Model": "A system that learns patterns from data.",
            "Training": "Process of improving model behavior using examples.",
            "Prediction": "Model output for new input data.",
            "Inference": "Using a trained model to make predictions.",
        }
    )
