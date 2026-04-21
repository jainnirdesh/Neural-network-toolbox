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
from neuroverse.data import get_training_curves


def render_classification_page() -> None:
    page_header("Classification Model", "Train and understand classification using neural networks")

    text_card(
        "What is Classification?",
        """
        Classification is the task of assigning an input to one of multiple categories.<br><br>
        <strong>Example:</strong> Spam vs Not Spam emails.<br><br>
        <ul>
            <li>Neural networks learn patterns in labeled examples.</li>
            <li>Each layer transforms input features into richer representations.</li>
            <li>The final layer outputs probabilities for each class.</li>
        </ul>
        """,
    )

    col_left, col_right = st.columns(2)
    with col_left:
        with st.container(border=True):
            st.markdown("### Interactive UI")
            st.markdown("#### Model Configuration")
            learning_rate = st.slider("Learning Rate", 0.001, 1.0, 0.05, 0.001)
            epochs = st.slider("Epochs", 10, 100, 40, 1)
            hidden_layers = st.slider("Hidden Layers", 1, 5, 2, 1)
            dataset = st.selectbox("Dataset", ["Iris", "MNIST"])
            optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
            run_training = st.button("Run Training", key="cls_train")

            if run_training or "cls_probs" not in st.session_state:
                rng = np.random.default_rng()
                st.session_state.cls_probs = rng.dirichlet(np.ones(3), size=1)[0]

            st.caption(
                f"Configured with {dataset}, {optimizer}, {hidden_layers} hidden layers, lr={learning_rate:.3f}."
            )

    with col_right:
        with st.container(border=True):
            st.markdown("### Predictions")
            probs = st.session_state.get("cls_probs", np.array([0.33, 0.33, 0.34]))
            class_names = ["Class A", "Class B", "Class C"]
            for name, prob in zip(class_names, probs):
                st.write(f"{name}: {prob * 100:.1f}%")
                st.progress(float(prob))

        with st.container(border=True):
            st.markdown("### Why this prediction?")
            top_idx = int(np.argmax(probs))
            top_class = class_names[top_idx]
            st.write(
                f"The model predicted {top_class} because it has the highest probability score "
                "and matches learned patterns from training data."
            )

    st.subheader("Graph / Output")
    x_epoch, loss, acc = get_training_curves(epochs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(x_epoch, loss, color="#f97316", linewidth=2)
    axes[0].set_title("Training Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x_epoch, acc, color="#22c55e", linewidth=2)
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)

    fig.patch.set_facecolor("#111c34")
    for ax in axes:
        ax.set_facecolor("#111c34")
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    st.pyplot(fig)

    render_how_it_works([
        "Input Features",
        "Apply Weights",
        "Activation Functions",
        "Output Probability",
    ])

    render_use_cases([
        "Spam Detection",
        "Disease Prediction",
        "Fraud Detection",
        "Image Classification",
    ])

    render_key_terms(
        {
            "Epoch": "One full pass through the training data.",
            "Learning Rate": "How fast the model updates its parameters.",
            "Optimizer": "Method used to reduce prediction error during training.",
            "Probability": "Model confidence score for each class.",
        }
    )
