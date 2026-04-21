from typing import List, Tuple

import numpy as np


def get_training_curves(epochs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(1, epochs + 1)
    rng = np.random.default_rng(7)
    loss = np.exp(-x / max(epochs / 5, 1)) + rng.normal(0, 0.02, size=epochs)
    acc = 0.45 + (1 - np.exp(-x / max(epochs / 4, 1))) * 0.5 + rng.normal(0, 0.015, size=epochs)
    return x, np.clip(loss, 0, None), np.clip(acc, 0, 1)


def simulate_image_predictions() -> Tuple[List[str], np.ndarray]:
    labels = ["Cat", "Dog", "Bird", "Car", "Tree", "Digit"]
    rng = np.random.default_rng()
    probs = rng.dirichlet(np.ones(len(labels)))
    top_idx = np.argsort(probs)[::-1][:3]
    top_labels = [labels[i] for i in top_idx]
    top_probs = probs[top_idx]
    return top_labels, top_probs


def create_heatmap(image_shape: Tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    center_x = 0.6 * w
    center_y = 0.4 * h
    spread = max(h, w) * 0.22
    heat = np.exp(-((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2 * spread**2))
    noise = np.random.default_rng().normal(0, 0.05, size=(h, w))
    return np.clip(heat + noise, 0, 1)
