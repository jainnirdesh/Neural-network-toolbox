import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from neuroverse.components import page_header, render_how_it_works, render_key_terms, render_use_cases, text_card


def _layer_positions(layer_sizes: List[int]) -> List[List[Tuple[float, float]]]:
    positions: List[List[Tuple[float, float]]] = []
    x_coords = np.linspace(0.1, 0.9, len(layer_sizes))
    for layer_idx, size in enumerate(layer_sizes):
        if size == 1:
            y_coords = [0.5]
        else:
            y_coords = np.linspace(0.15, 0.85, size)
        positions.append([(x_coords[layer_idx], float(y)) for y in y_coords])
    return positions


def _collect_connections(positions: List[List[Tuple[float, float]]]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    connections = []
    for idx in range(len(positions) - 1):
        for src in positions[idx]:
            for dst in positions[idx + 1]:
                connections.append((src, dst))
    return connections


def _draw_network(layer_sizes: List[int], active_connections: int = 0) -> plt.Figure:
    positions = _layer_positions(layer_sizes)
    connections = _collect_connections(positions)

    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.patch.set_facecolor("#111c34")
    ax.set_facecolor("#111c34")
    ax.axis("off")

    for conn_idx, ((x1, y1), (x2, y2)) in enumerate(connections):
        is_active = conn_idx < active_connections
        color = "#22d3ee" if is_active else "#475569"
        width = 1.8 if is_active else 0.8
        alpha = 0.9 if is_active else 0.4
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=alpha)

    for layer in positions:
        for x, y in layer:
            circle = plt.Circle((x, y), 0.018, color="#60a5fa", ec="#dbeafe", lw=1.1)
            ax.add_patch(circle)

    names = ["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Output"]
    for i, x in enumerate(np.linspace(0.1, 0.9, len(layer_sizes))):
        ax.text(x, 0.96, names[i], color="white", ha="center", fontsize=10)

    return fig


def render_neural_network_visualizer_page() -> None:
    page_header("Neural Network Visualizer", "Understand how neural networks process data")

    text_card(
        "What is a Neural Network?",
        """
        Neural networks are inspired by how the human brain processes information.<br><br>
        They consist of connected layers: <strong>Input</strong>, <strong>Hidden</strong>, and <strong>Output</strong>.<br><br>
        During training, they learn patterns from data to make predictions.
        """,
    )

    with st.container(border=True):
        st.markdown("### Interactive UI")
        c1, c2, c3, c4, c5 = st.columns(5)
        input_n = c1.number_input("Input neurons", min_value=1, max_value=10, value=4)
        h1_n = c2.number_input("Hidden layer 1 neurons", min_value=1, max_value=10, value=6)
        h2_n = c3.number_input("Hidden layer 2 neurons", min_value=1, max_value=10, value=5)
        h3_n = c4.number_input("Hidden layer 3 neurons", min_value=1, max_value=10, value=4)
        output_n = c5.number_input("Output neurons", min_value=1, max_value=5, value=3)
        animate = st.button("Animate Network")

    layer_sizes = [int(input_n), int(h1_n), int(h2_n), int(h3_n), int(output_n)]

    with st.container(border=True):
        st.markdown("### Graph / Output")
        total_connections = len(_collect_connections(_layer_positions(layer_sizes)))

        placeholder = st.empty()
        if animate:
            frames = min(22, max(8, total_connections // 8))
            for step in range(1, frames + 1):
                active = int((step / frames) * total_connections)
                fig = _draw_network(layer_sizes, active_connections=active)
                placeholder.pyplot(fig)
                time.sleep(0.08)
            st.caption("Signal flow animation complete.")
        else:
            fig = _draw_network(layer_sizes, active_connections=0)
            placeholder.pyplot(fig)

    text_card(
        "How Neural Networks Work",
        """
        Input values are combined through weighted sums, passed through activation functions,
        and transformed into output predictions.
        """,
    )

    render_how_it_works([
        "Input",
        "Weighted Sum",
        "Activation",
        "Output",
    ])

    render_use_cases([
        "Speech Recognition",
        "Recommendation Systems",
        "AI Assistants",
    ])

    render_key_terms(
        {
            "Neuron": "A small computation unit that processes incoming signals.",
            "Weight": "Connection strength between two neurons.",
            "Bias": "An extra value added to help shift and improve neuron output.",
            "Activation Function": "Adds non-linearity so the model can learn complex patterns.",
        }
    )
