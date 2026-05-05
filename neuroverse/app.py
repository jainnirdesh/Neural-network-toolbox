import os
import sys

import streamlit as st

# Allow running this file directly (e.g. `streamlit run neuroverse/app.py`).
if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from neuroverse.pages.about import render_about_page
from neuroverse.pages.classification import render_classification_page
from neuroverse.pages.home import render_home_page
from neuroverse.pages.image_recognition import render_image_recognition_page
from neuroverse.pages.neural_network_visualizer import render_neural_network_visualizer_page
from neuroverse.pages.propagation import render_propagation_page
from neuroverse.pages.regression import render_regression_page
from neuroverse.styles import apply_global_styles


PAGES = {
    "Home": render_home_page,
    "Classification": render_classification_page,
    "Regression": render_regression_page,
    "Image Recognition": render_image_recognition_page,
    "Neural Network Visualizer": render_neural_network_visualizer_page,
    "Forward & Backward Propagation": render_propagation_page,
    "About": render_about_page,
}


def run_app() -> None:
    st.set_page_config(page_title="NeuroVerse AI Lab", layout="wide")
    apply_global_styles()

    st.sidebar.title("NeuroVerse AI Lab")
    page = st.sidebar.radio("Navigate", list(PAGES.keys()))
    PAGES[page]()
