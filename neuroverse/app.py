import streamlit as st

from neuroverse.pages.about import render_about_page
from neuroverse.pages.classification import render_classification_page
from neuroverse.pages.home import render_home_page
from neuroverse.pages.image_recognition import render_image_recognition_page
from neuroverse.pages.neural_network_visualizer import render_neural_network_visualizer_page
from neuroverse.pages.regression import render_regression_page
from neuroverse.styles import apply_global_styles


PAGES = {
    "Home": render_home_page,
    "Classification": render_classification_page,
    "Regression": render_regression_page,
    "Image Recognition": render_image_recognition_page,
    "Neural Network Visualizer": render_neural_network_visualizer_page,
    "About": render_about_page,
}


def run_app() -> None:
    st.set_page_config(page_title="NeuroVerse AI Lab", layout="wide")
    apply_global_styles()

    st.sidebar.title("NeuroVerse AI Lab")
    page = st.sidebar.radio("Navigate", list(PAGES.keys()))
    PAGES[page]()
