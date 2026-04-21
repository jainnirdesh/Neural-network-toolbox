import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from neuroverse.components import (
    page_header,
    render_how_it_works,
    render_key_terms,
    render_use_cases,
    text_card,
)


def render_regression_page() -> None:
    page_header("Regression Model", "Predict continuous values using neural networks")

    text_card(
        "What is Regression?",
        """
        <ul>
            <li>Regression predicts continuous numerical values.</li>
            <li><strong>Example:</strong> House price prediction.</li>
            <li>Neural networks learn relationships between variables.</li>
        </ul>
        """,
    )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("### Interactive UI")
            degree = st.slider("Degree", 1, 5, 2)
            learning_rate = st.slider("Learning Rate", 0.001, 1.0, 0.03, 0.001, key="reg_lr")
            epochs = st.slider("Epochs", 10, 100, 60, key="reg_epochs")
            dataset = st.selectbox("Dataset", ["Housing", "Student Scores"])
            st.caption(
                f"Using dummy {dataset} dataset | Degree={degree}, lr={learning_rate:.3f}, epochs={epochs}."
            )

    rng = np.random.default_rng(12 if dataset == "Housing" else 18)
    x = np.linspace(0, 10, 80)
    target_curve = 2.2 * x + 5 if dataset == "Housing" else 0.7 * x**2 + 3
    y = target_curve + rng.normal(0, 4.0, size=len(x))

    coeff = np.polyfit(x, y, deg=degree)
    poly = np.poly1d(coeff)
    y_fit = poly(x)

    with right:
        with st.container(border=True):
            st.markdown("### Graph / Output")
            fig, ax = plt.subplots(figsize=(6, 4.2))
            ax.scatter(x, y, alpha=0.65, label="Data points", color="#60a5fa")
            ax.plot(x, y_fit, linewidth=2.5, label="Fitted curve", color="#f97316")
            ax.set_title("Scatter Plot with Fitted Curve")
            ax.set_xlabel("Input Feature")
            ax.set_ylabel("Target Value")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.patch.set_facecolor("#111c34")
            ax.set_facecolor("#111c34")
            ax.tick_params(colors="white")
            ax.title.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.legend(facecolor="#111c34", edgecolor="#1f2a44", labelcolor="white")
            st.pyplot(fig)

    render_how_it_works([
        "Input Variables",
        "Weights",
        "Activation",
        "Continuous Output",
    ])

    render_use_cases([
        "House Price Prediction",
        "Stock Forecasting",
        "Sales Prediction",
        "Temperature Prediction",
    ])

    render_key_terms(
        {
            "Regression": "Predicting a number, not a category.",
            "Fitted Curve": "A line or curve that best matches the data trend.",
            "Degree": "Controls curve complexity in polynomial fitting.",
            "Continuous Output": "Any numeric value in a range.",
        }
    )
