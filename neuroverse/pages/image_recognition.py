import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from neuroverse.components import (
    page_header,
    render_how_it_works,
    render_key_terms,
    render_use_cases,
    text_card,
)
from neuroverse.data import create_heatmap, simulate_image_predictions


def render_image_recognition_page() -> None:
    page_header("Image Recognition", "Classify images using neural networks")

    text_card(
        "What is Image Recognition?",
        """
        Image recognition means identifying objects or patterns inside an image.<br><br>
        It commonly uses Convolutional Neural Networks (CNNs) to detect visual features.<br><br>
        <strong>Example:</strong> Detecting handwritten digits or animals in photos.
        """,
    )

    with st.container(border=True):
        st.markdown("### Interactive UI")
        st.markdown("#### Upload Section")
        uploaded = st.file_uploader("Drag & drop image upload", type=["png", "jpg", "jpeg"])
        st.markdown("#### Controls")
        model_selector = st.selectbox("Model", ["CNN", "Pretrained"])
        threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
        run_infer = st.button("Run Recognition", key="img_rec_btn")

    output_col, heat_col = st.columns(2)
    top_labels, top_probs = simulate_image_predictions()

    with output_col:
        with st.container(border=True):
            st.markdown("### Graph / Output")
            image_array = None
            if uploaded is not None:
                image = Image.open(uploaded).convert("RGB")
                image_array = np.array(image)
                st.image(image, caption="Uploaded Image", use_container_width=True)
            else:
                st.info("Upload an image to view live output. Showing dummy predictions.")

            for label, prob in zip(top_labels, top_probs):
                st.write(f"{label}: {prob * 100:.1f}%")
                st.progress(float(prob))
                if prob < threshold:
                    st.caption(f"Below confidence threshold ({threshold:.2f})")

            if run_infer:
                st.caption(f"Inference simulated using {model_selector} model.")

            pred_label = top_labels[int(np.argmax(top_probs))]
            text_card(
                "Explain Prediction",
                f"The model focused on dominant visual patterns and predicted <strong>{pred_label}</strong> "
                "because it scored highest among candidate classes.",
            )

    with heat_col:
        with st.container(border=True):
            st.markdown("### Simulated Heatmap (Grad-CAM style)")
            if uploaded is not None and image_array is not None:
                heat = create_heatmap((image_array.shape[0], image_array.shape[1]))
            else:
                heat = np.random.default_rng().random((96, 96))

            fig, ax = plt.subplots(figsize=(6, 4.2))
            ax.imshow(heat, cmap="inferno")
            ax.axis("off")
            fig.patch.set_facecolor("#111c34")
            ax.set_facecolor("#111c34")
            st.pyplot(fig)

    render_how_it_works([
        "Image Input",
        "Filters",
        "Feature Maps",
        "Classification",
    ])

    render_use_cases([
        "Face Recognition",
        "Medical Imaging",
        "Self-driving Cars",
        "Security Systems",
    ])

    render_key_terms(
        {
            "CNN": "A neural network specialized for images.",
            "Feature Map": "Important visual patterns detected by filters.",
            "Confidence": "How sure the model is about a prediction.",
            "Grad-CAM": "A heatmap showing where the model looked most.",
        }
    )
