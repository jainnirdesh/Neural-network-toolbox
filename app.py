"""
╔══════════════════════════════════════════════════════════════╗
║              NeuroCraft AI Lab — app.py                      ║
║  A full-featured AI/ML learning & experimentation platform   ║
╚══════════════════════════════════════════════════════════════╝

Run with:
    streamlit run app.py

Dependencies: see requirements.txt
"""

# ─────────────────────────────────────────────
#  Standard library
# ─────────────────────────────────────────────
import json
import math
import os
import time
import random

# ─────────────────────────────────────────────
#  Data / ML
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, mean_squared_error, confusion_matrix,
    classification_report
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# ─────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
#  Streamlit
# ─────────────────────────────────────────────
import streamlit as st

# ════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be the very first Streamlit call)
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroCraft AI Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════
#  GLOBAL CSS  — dark-mode, custom typography, card styles
# ════════════════════════════════════════════════════════════════
GLOBAL_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg-primary:    #0d0f1a;
    --bg-secondary:  #12162a;
    --bg-card:       #181c34;
    --bg-card-hover: #1e2440;
    --accent:        #6c63ff;
    --accent-2:      #00d4aa;
    --accent-3:      #ff6584;
    --text-primary:  #e8eaf6;
    --text-muted:    #8892b0;
    --border:        rgba(108,99,255,0.25);
    --glow:          0 0 24px rgba(108,99,255,0.35);
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1224 0%, #12162a 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Main area ── */
.main .block-container {
    padding: 2rem 3rem 4rem 3rem;
    max-width: 1200px;
}

/* ── Hero section ── */
.hero-wrap {
    background: linear-gradient(135deg, #181c34 0%, #1a1040 50%, #0d1f3c 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3.5rem 4rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(108,99,255,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6c63ff, #00d4aa, #ff6584);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin: 0;
}
.hero-sub {
    color: var(--text-muted);
    font-size: 1.15rem;
    margin-top: 0.8rem;
    font-weight: 400;
    line-height: 1.6;
}
.hero-badge {
    display: inline-block;
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.4);
    color: #a89cff;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    margin-bottom: 1rem;
    letter-spacing: 0.05em;
}

/* ── Cards ── */
.nc-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    transition: all 0.25s ease;
}
.nc-card:hover {
    background: var(--bg-card-hover);
    border-color: rgba(108,99,255,0.55);
    box-shadow: var(--glow);
    transform: translateY(-2px);
}
.nc-card-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: #a89cff;
    margin-bottom: 0.5rem;
}
.nc-card-body { color: var(--text-muted); font-size: 0.92rem; line-height: 1.65; }

/* ── Section headings ── */
.section-head {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 2rem 0 0.3rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-sub {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-bottom: 1.8rem;
}
.divider { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }

/* ── Metric pill ── */
.metric-pill {
    background: rgba(0, 212, 170, 0.12);
    border: 1px solid rgba(0, 212, 170, 0.3);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    text-align: center;
}
.metric-pill .val {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #00d4aa;
}
.metric-pill .lbl { color: var(--text-muted); font-size: 0.8rem; margin-top: 0.2rem; }

/* ── Chat bubbles ── */
.chat-user {
    background: rgba(108,99,255,0.18);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 14px 14px 4px 14px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0 0.5rem 15%;
    color: #d0ccff;
    font-size: 0.93rem;
    line-height: 1.55;
}
.chat-ai {
    background: rgba(0, 212, 170, 0.1);
    border: 1px solid rgba(0, 212, 170, 0.25);
    border-radius: 4px 14px 14px 14px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 15% 0.5rem 0;
    color: var(--text-primary);
    font-size: 0.93rem;
    line-height: 1.55;
}
.chat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-bottom: 0.25rem;
}

/* ── Concept pill tags ── */
.tag {
    display: inline-block;
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.3);
    color: #a89cff;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.76rem;
    margin: 2px;
}

/* ── Progress bar ── */
.xp-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    margin: 6px 0;
}
.xp-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #6c63ff, #00d4aa);
    transition: width 0.6s ease;
}

/* ── Streamlit widget overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #5a52e0);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.4rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.92rem;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 4px 14px rgba(108,99,255,0.35);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #7c74ff, #6a62f0);
    box-shadow: 0 6px 20px rgba(108,99,255,0.5);
    transform: translateY(-1px);
}
.stSelectbox label, .stSlider label, .stTextInput label,
.stRadio label, .stTextArea label, .stNumberInput label {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}
.stSlider .stSlider { color: var(--accent) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: var(--text-muted);
    font-weight: 500;
    font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(108,99,255,0.25) !important;
    color: #a89cff !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* ── Info / warning boxes ── */
.stAlert { border-radius: 10px !important; }
div[data-baseweb="notification"] { border-radius: 10px !important; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(108,99,255,0.4); border-radius: 3px; }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  SESSION STATE  — XP gamification + chat history
# ════════════════════════════════════════════════════════════════
if "xp" not in st.session_state:
    st.session_state.xp = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "completed_topics" not in st.session_state:
    st.session_state.completed_topics = set()
if "lang" not in st.session_state:
    st.session_state.lang = "English"


def get_anthropic_api_key() -> str:
    """Resolve the Anthropic API key from the UI, secrets, or environment."""
    ui_key = st.session_state.get("anthropic_api_key", "").strip()
    if ui_key:
        return ui_key

    try:
        secret_key = st.secrets.get("ANTHROPIC_API_KEY", "").strip()
    except Exception:
        secret_key = ""
    if secret_key:
        return secret_key

    return os.getenv("ANTHROPIC_API_KEY", "").strip()


def add_xp(amount: int, reason: str = ""):
    """Award XP and surface a toast-style message."""
    st.session_state.xp += amount
    if reason:
        st.toast(f"⚡ +{amount} XP — {reason}", icon="🏆")


def xp_level():
    """Return (level, title, next_threshold) based on current XP."""
    thresholds = [(0, "🥚 Newbie"), (100, "🔬 Explorer"), (250, "🧪 Experimenter"),
                  (500, "🤖 Builder"), (1000, "🧠 AI Wizard"), (2000, "🚀 NeuroCraft Master")]
    level = 0
    title = thresholds[0][1]
    for i, (t, lbl) in enumerate(thresholds):
        if st.session_state.xp >= t:
            level = i
            title = lbl
    next_t = thresholds[min(level + 1, len(thresholds) - 1)][0]
    return level, title, next_t


# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <div style='font-family:Space Mono,monospace; font-size:1.35rem; font-weight:700;
                    background:linear-gradient(90deg,#6c63ff,#00d4aa);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text;'>
            🧠 NeuroCraft
        </div>
        <div style='color:#8892b0; font-size:0.75rem; letter-spacing:0.08em; margin-top:2px;'>
            AI LAB v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(108,99,255,0.2); margin: 0.5rem 0 1rem 0'>", unsafe_allow_html=True)

    # XP bar
    lvl, lvl_title, next_t = xp_level()
    xp_pct = min(100, int(st.session_state.xp / max(next_t, 1) * 100)) if next_t > 0 else 100
    st.markdown(f"""
    <div style='padding: 0 0.5rem;'>
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;'>
            <span style='font-family:Space Mono,monospace; font-size:0.78rem; color:#a89cff;'>{lvl_title}</span>
            <span style='font-family:Space Mono,monospace; font-size:0.72rem; color:#00d4aa;'>{st.session_state.xp} XP</span>
        </div>
        <div class='xp-bar-wrap'>
            <div class='xp-bar-fill' style='width:{xp_pct}%;'></div>
        </div>
        <div style='color:#8892b0; font-size:0.7rem; margin-top:3px;'>Next level at {next_t} XP</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(108,99,255,0.2); margin: 1rem 0;'>", unsafe_allow_html=True)

    # Language toggle
    lang = st.radio("🌐 Language", ["English", "हिंदी"], horizontal=True,
                    index=0 if st.session_state.lang == "English" else 1)
    st.session_state.lang = lang

    st.markdown("<hr style='border-color:rgba(108,99,255,0.2); margin: 0.8rem 0;'>", unsafe_allow_html=True)

    # Navigation
    nav_opts = {
        "🏠 Home":          "Home",
        "📘 Learn":         "Learn",
        "🧪 Playground":    "Playground",
        "📊 Visualize":     "Visualize",
        "⚙️ Experiment":    "Experiment",
        "💬 AI Assistant":  "Chatbot",
        "ℹ️ About":         "About",
    }
    page = st.radio("Navigation", list(nav_opts.keys()), label_visibility="collapsed")
    active = nav_opts[page]

    st.markdown("<hr style='border-color:rgba(108,99,255,0.2); margin: 1rem 0;'>", unsafe_allow_html=True)

    # Completed topics
    done = len(st.session_state.completed_topics)
    st.markdown(f"""
    <div style='padding:0 0.5rem;'>
        <div style='color:#8892b0; font-size:0.75rem; margin-bottom:4px;'>📖 Topics Completed</div>
        <div style='font-family:Space Mono,monospace; font-size:1.3rem; color:#00d4aa; font-weight:700;'>{done}/8</div>
    </div>
    """, unsafe_allow_html=True)

    # API Key for chatbot
    st.markdown("<hr style='border-color:rgba(108,99,255,0.2); margin: 1rem 0;'>", unsafe_allow_html=True)
    api_key = st.text_input("🔑 Anthropic API Key (optional)",
                            key="anthropic_api_key",
                            type="password",
                            placeholder="sk-ant-...",
                            help="Paste your Anthropic API key to unlock AI-powered chat. Leave blank to use smart simulation mode.")


# ════════════════════════════════════════════════════════════════
#  TRANSLATION HELPERS  (English / Hindi)
# ════════════════════════════════════════════════════════════════
TRANSLATIONS = {
    "What is Machine Learning?": "मशीन लर्निंग क्या है?",
    "Machine Learning (ML) is a subset of Artificial Intelligence that enables computers to learn from data without being explicitly programmed.":
        "मशीन लर्निंग (ML) कृत्रिम बुद्धिमत्ता का एक हिस्सा है जो कंप्यूटर को बिना स्पष्ट प्रोग्राम किए डेटा से सीखने में सक्षम बनाता है।",
}

def t(key: str) -> str:
    if st.session_state.lang == "हिंदी" and key in TRANSLATIONS:
        return TRANSLATIONS[key]
    return key


# ════════════════════════════════════════════════════════════════
#  MATPLOTLIB DARK THEME HELPER
# ════════════════════════════════════════════════════════════════
def dark_fig(w=8, h=4.5, subplots=None):
    """Return a matplotlib figure pre-styled for the dark theme."""
    edge_rgba = (108 / 255, 99 / 255, 1.0, 0.3)
    grid_rgba = (108 / 255, 99 / 255, 1.0, 0.08)
    kwargs = dict(figsize=(w, h))
    if subplots:
        fig, axes = plt.subplots(*subplots, **kwargs)
    else:
        fig, axes = plt.subplots(**kwargs)
    fig.patch.set_facecolor("#181c34")
    ax_list = axes if subplots else [axes]
    for ax in (ax_list if hasattr(ax_list, '__iter__') else [ax_list]):
        ax.set_facecolor("#12162a")
        ax.tick_params(colors="#8892b0", labelsize=8)
        ax.xaxis.label.set_color("#8892b0")
        ax.yaxis.label.set_color("#8892b0")
        ax.title.set_color("#e8eaf6")
        for spine in ax.spines.values():
            spine.set_edgecolor(edge_rgba)
        ax.grid(True, color=grid_rgba, linestyle="--", linewidth=0.6)
    return (fig, axes) if subplots else (fig, axes)


# ════════════════════════════════════════════════════════════════
#  ──────────────────  P A G E S  ──────────────────
# ════════════════════════════════════════════════════════════════

# ─────────────────────────────  HOME  ──────────────────────────
def page_home():
    st.markdown("""
    <div class='hero-wrap'>
        <div class='hero-badge'>● LIVE PLATFORM</div>
        <h1 class='hero-title'>NeuroCraft<br>AI Lab</h1>
        <p class='hero-sub'>
            Your interactive playground for Machine Learning & Neural Networks.<br>
            Learn concepts, run experiments, visualise models — all in one place.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    col1, col2, col3 = st.columns(3)
    cards = [
        ("📘", "Learn", "Beginner-friendly theory on ML, neural nets, activation functions, and more."),
        ("🧪", "Playground", "Run live Linear Regression, Classification, and Sentiment demos interactively."),
        ("📊", "Visualize", "See accuracy curves, loss graphs, confusion matrices, and decision boundaries."),
        ("⚙️", "Experiment", "Tweak hyperparameters — learning rate, epochs, model type — and watch changes."),
        ("💬", "AI Assistant", "Chat with an AI tutor that explains concepts, corrects you, and quizzes you."),
        ("🏆", "Gamification", "Earn XP as you explore topics and run experiments. Level up your skills!"),
    ]
    cols = [col1, col2, col3, col1, col2, col3]
    for (icon, title, desc), col in zip(cards, cols):
        col.markdown(f"""
        <div class='nc-card'>
            <div style='font-size:1.8rem; margin-bottom:0.5rem;'>{icon}</div>
            <div class='nc-card-title'>{title}</div>
            <div class='nc-card-body'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Quick stats
    st.markdown("<div class='section-head'>📈 Your Progress</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    lvl, lvl_title, _ = xp_level()
    stats = [
        (str(st.session_state.xp), "Total XP Earned"),
        (str(len(st.session_state.completed_topics)), "Topics Completed"),
        (str(lvl), "Current Level"),
        (lvl_title, "Rank"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], stats):
        col.markdown(f"""
        <div class='metric-pill'>
            <div class='val' style='font-size:1.3rem;'>{val}</div>
            <div class='lbl'>{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Quick tips
    st.markdown("<div class='section-head'>💡 Quick Tips</div>", unsafe_allow_html=True)
    tips = [
        "Start with **📘 Learn** to build your conceptual foundation.",
        "Try **🧪 Playground** demos — every experiment earns you XP!",
        "Use the **💬 AI Assistant** to ask any ML question in plain English.",
        "Add your Anthropic API key in the sidebar to unlock smarter chat.",
        "Switch to **हिंदी** in the sidebar for Hindi explanations.",
    ]
    for tip in tips:
        st.markdown(f"""
        <div class='nc-card' style='padding:0.9rem 1.2rem; margin-bottom:0.6rem;'>
            <div class='nc-card-body'>→ {tip}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────  LEARN  ─────────────────────────
LEARN_TOPICS = {
    "🤖 What is ML?": {
        "key": "ml_basics",
        "content": """
**Machine Learning (ML)** is a branch of Artificial Intelligence where computers learn patterns
from data rather than following hard-coded rules.

**Classic programming** → You give the computer rules + data → It produces answers.  
**Machine Learning** → You give the computer data + answers → It figures out the rules!

### Why does it matter?
- Powers recommendation systems (Netflix, YouTube, Spotify)
- Enables speech recognition (Siri, Alexa)
- Drives fraud detection in banking
- Makes self-driving cars possible

### Three steps in every ML project
1. **Collect & clean data** — garbage in, garbage out
2. **Train a model** — the algorithm learns from examples
3. **Evaluate & deploy** — check performance on unseen data
""",
        "quiz": ("What does a machine learning model learn from?", ["Rules written by humans", "Data and examples", "Random guesses", "The internet"], 1),
    },
    "📂 Types of ML": {
        "key": "ml_types",
        "content": """
### Supervised Learning
- Model trained on **labelled** input-output pairs
- Goal: predict the output for new inputs
- Examples: spam detection, house price prediction, image classification

### Unsupervised Learning
- No labels — model finds **hidden patterns** on its own
- Examples: customer segmentation, anomaly detection, topic modelling

### Reinforcement Learning
- An **agent** takes actions in an environment and learns from **rewards/penalties**
- Examples: game-playing AI (AlphaGo), robot navigation, trading bots

| Type | Labels? | Goal | Example |
|------|---------|------|---------|
| Supervised | ✅ Yes | Predict output | Email spam filter |
| Unsupervised | ❌ No | Find structure | Customer groups |
| Reinforcement | 🎮 Rewards | Maximise reward | Chess AI |
""",
        "quiz": ("Which ML type uses rewards and penalties?", ["Supervised", "Unsupervised", "Reinforcement", "Transfer"], 2),
    },
    "🕸️ Neural Networks": {
        "key": "neural_nets",
        "content": """
### Inspired by the brain!
A neural network is made of **neurons** arranged in **layers**:

```
Input Layer → Hidden Layer(s) → Output Layer
   [x₁]          [h₁ h₂ h₃]        [ŷ]
   [x₂]    →→→   [h₄ h₅ h₆]   →→→  
   [x₃]          [h₇ h₈ h₉]
```

### Key Concepts
- **Neuron**: Receives inputs, applies weights, passes through activation function
- **Weight (w)**: How important each connection is — learned during training
- **Bias (b)**: A constant added to shift the activation
- **Forward pass**: Data flows input → output to make a prediction
- **Backpropagation**: Error flows backwards to update weights

### Formula inside each neuron
```
output = activation(w₁x₁ + w₂x₂ + ... + b)
```

### Deep vs Shallow
- **Shallow**: 1 hidden layer — fine for simple problems
- **Deep**: Many hidden layers — can learn complex representations (images, speech)
""",
        "quiz": ("What does backpropagation update?", ["Input data", "Activation functions", "Weights and biases", "Layer count"], 2),
    },
    "⚡ Activation Functions": {
        "key": "activations",
        "content": """
Activation functions add **non-linearity** — without them, a neural network is just a fancy linear equation.

### ReLU — Rectified Linear Unit  *(most popular)*
```
f(x) = max(0, x)
```
- Output: 0 for negative x, x for positive x
- Fast to compute, avoids vanishing gradient
- Used in hidden layers of most modern networks

### Sigmoid  *(squashes to 0–1)*
```
f(x) = 1 / (1 + e^(-x))
```
- Output always between 0 and 1
- Used in binary classification output layers
- Suffers from vanishing gradient for deep networks

### Tanh  *(squashes to -1 to 1)*
```
f(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
```
- Zero-centred (better than Sigmoid in practice)
- Good for hidden layers in RNNs

### Softmax  *(multi-class probabilities)*
```
f(xᵢ) = eˣⁱ / Σeˣʲ
```
- Output sums to 1.0 — a probability distribution
- Used in the output layer for multi-class classification
""",
        "quiz": ("Which activation function is used in binary classification outputs?", ["ReLU", "Tanh", "Sigmoid", "Softmax"], 2),
    },
    "📉 Loss Functions": {
        "key": "loss_fns",
        "content": """
A **loss function** measures how wrong the model's predictions are. Training minimises this.

### Mean Squared Error (MSE) — Regression
```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```
- Penalises large errors heavily (squared)
- Use when target is a continuous number (price, temperature)

### Binary Cross-Entropy — Binary Classification
```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```
- Perfect for yes/no predictions
- Works with Sigmoid output

### Categorical Cross-Entropy — Multi-class
```
L = -Σ yᵢ·log(ŷᵢ)
```
- Combine with Softmax output
- Standard for image classification

> 💡 **Tip**: The goal of training is to follow the gradient of the loss downhill (Gradient Descent) until we reach a minimum.
""",
        "quiz": ("What does gradient descent minimise?", ["Accuracy", "Model size", "Loss function", "Number of layers"], 2),
    },
    "🎯 Overfitting vs Underfitting": {
        "key": "overfitting",
        "content": """
### Underfitting (High Bias)
- Model is **too simple** to capture the patterns
- High training error + High validation error
- Fix: Add more layers/features, train longer, reduce regularisation

### Overfitting (High Variance)
- Model **memorises training data** and fails on new data
- Low training error + High validation error
- Fix: More data, Dropout, Regularisation (L1/L2), Early stopping

### The Goldilocks Zone
```
         Training Error   Validation Error
Underfit:   High      →     High
Just right: Low       →     Low (close to training)
Overfit:    Very Low  →     High
```

### Regularisation Techniques
- **L2 (Ridge)**: Penalises large weights — keeps model smooth
- **L1 (Lasso)**: Drives some weights to zero — feature selection
- **Dropout**: Randomly disables neurons during training — forces redundancy
- **Early stopping**: Stop training when validation loss stops improving
""",
        "quiz": ("What characterises an overfitted model?", ["High training error", "Low training, high validation error", "Same train and val error", "No patterns learned"], 1),
    },
    "🔄 Gradient Descent": {
        "key": "grad_descent",
        "content": """
**Gradient Descent** is the optimisation algorithm that trains most neural networks.

### The Intuition
Imagine you're blindfolded on a hilly landscape and want to reach the valley (minimum loss).
You feel which direction slopes downward and take a step. Repeat until flat.

### Algorithm
```
θ = θ - α × ∇L(θ)

θ = parameters (weights)
α = learning rate
∇L = gradient of loss
```

### Learning Rate (α)
- **Too small**: Converges very slowly — takes forever
- **Too large**: Overshoots minimum — loss bounces or explodes
- **Just right**: Smooth convergence in reasonable time

### Variants
| Variant | Batch Size | Pro | Con |
|---------|-----------|-----|-----|
| Batch GD | Full dataset | Stable gradients | Slow per update |
| Stochastic (SGD) | 1 sample | Fast updates | Noisy |
| Mini-batch | 32–256 | Best of both | Tuning needed |

### Popular Optimisers
- **Adam**: Adaptive learning rates — most popular choice
- **RMSProp**: Good for RNNs
- **AdaGrad**: Better for sparse data
""",
        "quiz": ("What does a high learning rate cause?", ["Slow convergence", "Overfitting", "Overshooting / divergence", "Underfitting"], 2),
    },
    "📦 Model Evaluation": {
        "key": "evaluation",
        "content": """
### Train / Validation / Test Split
- **Training set** (70%): Model learns from this
- **Validation set** (15%): Tune hyperparameters
- **Test set** (15%): Final, unbiased performance estimate (touch only once!)

### Classification Metrics
```
Accuracy  = correct predictions / total predictions
Precision = TP / (TP + FP)   ← "of those predicted positive, how many are?"
Recall    = TP / (TP + FN)   ← "of all positives, how many did we catch?"
F1 Score  = 2 × (P × R) / (P + R)
```

### Confusion Matrix
```
              Predicted
              Pos   Neg
Actual Pos  [ TP  | FN ]
       Neg  [ FP  | TN ]
```

### Regression Metrics
- **MAE**: Mean Absolute Error — easy to interpret
- **MSE**: Mean Squared Error — penalises large errors
- **R²**: How much variance is explained (1.0 = perfect)

### Cross-Validation
- Split data into K folds; train on K-1, test on 1; rotate
- Gives a robust estimate of generalisation performance
""",
        "quiz": ("What is the test set used for?", ["Training the model", "Tuning hyperparameters", "Final unbiased evaluation", "Data augmentation"], 2),
    },
}

def page_learn():
    st.markdown("<div class='section-head'>📘 Learning Hub</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Build your ML foundation with interactive theory, examples, and mini-quizzes.</div>", unsafe_allow_html=True)

    topic_keys = list(LEARN_TOPICS.keys())
    chosen = st.selectbox("📚 Choose a topic", topic_keys)
    info = LEARN_TOPICS[chosen]
    topic_id = info["key"]

    # Mark read
    if topic_id not in st.session_state.completed_topics:
        if st.button("✅ Mark as Read (+30 XP)"):
            st.session_state.completed_topics.add(topic_id)
            add_xp(30, f"Completed '{chosen}'")
            st.rerun()
    else:
        st.success(f"✅ Topic completed! You earned XP for this one.")

    st.markdown(f"<div class='nc-card'>", unsafe_allow_html=True)
    st.markdown(info["content"])
    st.markdown("</div>", unsafe_allow_html=True)

    # Activation function visualiser
    if topic_id == "activations":
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("### 📈 Interactive Activation Function Visualiser")
        fn_choice = st.select_slider("Select function", ["ReLU", "Sigmoid", "Tanh", "Softmax/Identity"])
        x = np.linspace(-5, 5, 300)
        fig, ax = dark_fig(7, 3.5)
        colors = {"ReLU": "#6c63ff", "Sigmoid": "#00d4aa", "Tanh": "#ff6584", "Softmax/Identity": "#ffd700"}
        if fn_choice == "ReLU":
            y = np.maximum(0, x)
        elif fn_choice == "Sigmoid":
            y = 1 / (1 + np.exp(-x))
        elif fn_choice == "Tanh":
            y = np.tanh(x)
        else:
            y = x  # identity for illustration
        ax.plot(x, y, color=colors[fn_choice], linewidth=2.5, label=fn_choice)
        ax.axhline(0, color="#8892b0", linewidth=0.6, linestyle="--")
        ax.axvline(0, color="#8892b0", linewidth=0.6, linestyle="--")
        ax.set_title(f"{fn_choice} Activation", color="#e8eaf6", fontsize=11)
        ax.legend(facecolor="#181c34", edgecolor=(108 / 255, 99 / 255, 1.0, 0.3), labelcolor="#a89cff")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Gradient descent visualiser
    if topic_id == "grad_descent":
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("### 📉 Gradient Descent Visualiser")
        lr_demo = st.slider("Learning rate", 0.01, 0.99, 0.3, 0.01)
        x_vals = [3.5]
        for _ in range(20):
            x_v = x_vals[-1]
            grad = 2 * x_v          # derivative of x^2
            x_vals.append(x_v - lr_demo * grad)
        fig, ax = dark_fig(7, 3.5)
        xs = np.linspace(-4, 4, 300)
        ax.plot(xs, xs**2, color="#6c63ff", linewidth=2, label="Loss = x²")
        ax.scatter(x_vals, [v**2 for v in x_vals],
                   c=np.linspace(0, 1, len(x_vals)), cmap="plasma", s=50, zorder=5)
        ax.set_title("Gradient Descent on f(x)=x²", color="#e8eaf6", fontsize=11)
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("Loss")
        ax.legend(facecolor="#181c34", edgecolor=(108 / 255, 99 / 255, 1.0, 0.3), labelcolor="#a89cff")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Mini quiz
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Quick Quiz")
    q, options, correct_idx = info["quiz"]
    answer = st.radio(q, options, index=None, key=f"quiz_{topic_id}")
    if answer is not None:
        if options.index(answer) == correct_idx:
            st.success("🎉 Correct! Great job!")
            if f"quiz_{topic_id}_done" not in st.session_state:
                st.session_state[f"quiz_{topic_id}_done"] = True
                add_xp(20, "Quiz answered correctly")
        else:
            st.error(f"❌ Not quite. The answer is: **{options[correct_idx]}**")


# ──────────────────────────  PLAYGROUND  ───────────────────────
def page_playground():
    st.markdown("<div class='section-head'>🧪 Model Playground</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Run real ML models in your browser. Interact, predict, and explore.</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Linear Regression", "🔮 Classification", "💬 Sentiment Analysis"])

    # ── Tab 1: Linear Regression ──────────────────────────────
    with tab1:
        st.markdown("#### Linear Regression Demo")
        col_left, col_right = st.columns([1, 1.6])

        with col_left:
            data_src = st.radio("Data source", ["Generate synthetic", "Enter custom points"])
            if data_src == "Generate synthetic":
                n_pts = st.slider("Number of data points", 10, 200, 50)
                noise = st.slider("Noise level", 0.0, 5.0, 1.5, 0.1)
                seed = st.number_input("Random seed", 0, 9999, 42)
                np.random.seed(int(seed))
                X_raw = np.random.uniform(0, 10, n_pts)
                y_raw = 2.5 * X_raw + 5 + np.random.normal(0, noise, n_pts)
            else:
                st.markdown("<small style='color:#8892b0'>Enter comma-separated X and Y values</small>", unsafe_allow_html=True)
                x_txt = st.text_input("X values", "1,2,3,4,5,6,7,8,9,10")
                y_txt = st.text_input("Y values", "3,5,4,8,9,11,12,14,16,18")
                try:
                    X_raw = np.array([float(v) for v in x_txt.split(",")])
                    y_raw = np.array([float(v) for v in y_txt.split(",")])
                except:
                    st.error("Invalid input — using defaults")
                    X_raw = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)
                    y_raw = np.array([3,5,4,8,9,11,12,14,16,18], dtype=float)

            predict_x = st.number_input("Predict Y for X =", value=float(np.max(X_raw) + 1))

            if st.button("▶ Run Regression", key="run_lr"):
                add_xp(25, "Ran Linear Regression")

        with col_right:
            X_2d = X_raw.reshape(-1, 1)
            model_lr = LinearRegression()
            model_lr.fit(X_2d, y_raw)
            y_pred_line = model_lr.predict(X_2d)
            pred_val = model_lr.predict([[predict_x]])[0]
            r2 = model_lr.score(X_2d, y_raw)
            mse_v = mean_squared_error(y_raw, y_pred_line)

            fig, ax = dark_fig(6.5, 4)
            ax.scatter(X_raw, y_raw, color="#6c63ff", alpha=0.7, s=45, label="Data points")
            sort_idx = np.argsort(X_raw)
            ax.plot(X_raw[sort_idx], y_pred_line[sort_idx], color="#00d4aa", linewidth=2.5, label="Regression line")
            ax.scatter([predict_x], [pred_val], color="#ff6584", s=120, zorder=6, label=f"Prediction ({pred_val:.2f})")
            ax.set_title("Linear Regression", color="#e8eaf6", fontsize=11)
            ax.set_xlabel("X"); ax.set_ylabel("Y")
            ax.legend(facecolor="#181c34", edgecolor=(108 / 255, 99 / 255, 1.0, 0.3), labelcolor="#a89cff", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            c1, c2, c3 = st.columns(3)
            eq = f"y = {model_lr.coef_[0]:.3f}x + {model_lr.intercept_:.3f}"
            for col, (v, l) in zip([c1, c2, c3], [(eq, "Equation"), (f"{r2:.4f}", "R² Score"), (f"{mse_v:.3f}", "MSE")]):
                col.markdown(f"<div class='metric-pill'><div class='val' style='font-size:0.95rem;'>{v}</div><div class='lbl'>{l}</div></div>", unsafe_allow_html=True)

    # ── Tab 2: Classification ─────────────────────────────────
    with tab2:
        st.markdown("#### Classification Demo — Iris Dataset")
        iris = load_iris()
        df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
        df_iris["target"] = iris.target
        df_iris["species"] = [iris.target_names[t] for t in iris.target]

        col_l, col_r = st.columns([1, 1.6])
        with col_l:
            feat_x = st.selectbox("Feature X", iris.feature_names, index=0)
            feat_y = st.selectbox("Feature Y", iris.feature_names, index=2)
            algo = st.selectbox("Algorithm", ["Logistic Regression", "K-Nearest Neighbours", "Decision Tree", "SVM"])
            test_sz = st.slider("Test split %", 10, 40, 20)

            algo_map = {
                "Logistic Regression": LogisticRegression(max_iter=200),
                "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5),
                "Decision Tree": DecisionTreeClassifier(max_depth=4),
                "SVM": SVC(probability=True),
            }
            clf = algo_map[algo]

            # Custom prediction
            st.markdown("##### 🔮 Predict a new sample")
            sl = st.number_input("Sepal length (cm)", 4.0, 8.0, 5.5, 0.1)
            sw = st.number_input("Sepal width (cm)", 2.0, 5.0, 3.0, 0.1)
            pl = st.number_input("Petal length (cm)", 1.0, 7.0, 4.0, 0.1)
            pw = st.number_input("Petal width (cm)", 0.1, 3.0, 1.2, 0.1)

            if st.button("▶ Classify", key="run_clf"):
                add_xp(25, "Ran Classification")

        with col_r:
            X_feat = df_iris[[feat_x, feat_y]].values
            y_clf = iris.target
            X_train, X_test, y_train, y_test = train_test_split(
                iris.data, y_clf, test_size=test_sz/100, random_state=42)
            X_train_2, X_test_2, _, _ = train_test_split(
                X_feat, y_clf, test_size=test_sz/100, random_state=42)

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_train)
            X_te_s = sc.transform(X_test)
            clf.fit(X_tr_s, y_train)
            acc = accuracy_score(y_test, clf.predict(X_te_s))

            # 2D decision boundary
            clf_2d = algo_map[algo].__class__(**algo_map[algo].get_params())
            sc_2d = StandardScaler()
            clf_2d.fit(sc_2d.fit_transform(X_train_2), y_train)

            x_min, x_max = X_feat[:, 0].min() - 0.5, X_feat[:, 0].max() + 0.5
            y_min, y_max = X_feat[:, 1].min() - 0.5, X_feat[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 120), np.linspace(y_min, y_max, 120))
            Z = clf_2d.predict(sc_2d.transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)

            fig, ax = dark_fig(6.5, 4.2)
            ax.contourf(xx, yy, Z, alpha=0.25, cmap="cool")
            colors_cls = ["#6c63ff", "#00d4aa", "#ff6584"]
            for cls_idx, (cls_name, col) in enumerate(zip(iris.target_names, colors_cls)):
                mask = y_clf == cls_idx
                ax.scatter(X_feat[mask, 0], X_feat[mask, 1], color=col, label=cls_name, s=40, alpha=0.8)
            ax.set_xlabel(feat_x, fontsize=8); ax.set_ylabel(feat_y, fontsize=8)
            ax.set_title(f"{algo} — Decision Boundary", color="#e8eaf6", fontsize=10)
            ax.legend(facecolor="#181c34", edgecolor=(108 / 255, 99 / 255, 1.0, 0.3), labelcolor="#a89cff", fontsize=7)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Prediction
            sample = np.array([[sl, sw, pl, pw]])
            pred_class = iris.target_names[clf.predict(sc.transform(sample))[0]]
            proba = clf.predict_proba(sc.transform(sample))[0]
            c1, c2 = st.columns(2)
            c1.markdown(f"<div class='metric-pill'><div class='val' style='font-size:1rem;'>🌺 {pred_class}</div><div class='lbl'>Predicted Species</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-pill'><div class='val'>{acc*100:.1f}%</div><div class='lbl'>Test Accuracy</div></div>", unsafe_allow_html=True)
            st.markdown(f"<small style='color:#8892b0'>Probabilities: {', '.join([f'{n}: {p:.2%}' for n,p in zip(iris.target_names, proba)])}</small>", unsafe_allow_html=True)

    # ── Tab 3: Sentiment Analysis ─────────────────────────────
    with tab3:
        st.markdown("#### Text Sentiment Analysis")
        st.markdown("<div class='nc-card'><div class='nc-card-body'>This demo uses a rule-based lexicon approach (VADER-style) — no API key needed.</div></div>", unsafe_allow_html=True)

        # Simple sentiment lexicon
        POSITIVE = {"great","good","excellent","amazing","wonderful","fantastic","love","happy",
                    "best","awesome","perfect","brilliant","outstanding","superb","joy","enjoy",
                    "pleased","delighted","magnificent","positive","beautiful","nice","helpful"}
        NEGATIVE = {"bad","terrible","awful","horrible","hate","worst","poor","disappointing",
                    "ugly","disgusting","failure","useless","boring","dull","wrong","terrible",
                    "frustrating","annoying","sad","angry","broken","trash","garbage","negative"}
        INTENSIFIERS = {"very","extremely","really","absolutely","totally","super","highly"}

        def analyse_sentiment(text: str):
            words = text.lower().split()
            score = 0
            for i, w in enumerate(words):
                clean = w.strip(".,!?;:'\"")
                mult = 1.5 if i > 0 and words[i-1].strip(".,!?") in INTENSIFIERS else 1.0
                if clean in POSITIVE: score += mult
                if clean in NEGATIVE: score -= mult
                if clean == "not" and i + 1 < len(words):
                    nxt = words[i+1].strip(".,!?")
                    if nxt in POSITIVE: score -= 1.5
                    if nxt in NEGATIVE: score += 1.5
            if score > 0.5: return "Positive 😊", score, "#00d4aa"
            if score < -0.5: return "Negative 😞", score, "#ff6584"
            return "Neutral 😐", score, "#ffd700"

        sample_texts = [
            "I absolutely love this product! It's amazing and works perfectly.",
            "This is the worst experience I've ever had. Totally disappointing.",
            "The weather is okay today. Nothing special.",
            "The team did a great job despite the difficult challenges.",
            "I hate waiting in long queues. It's so frustrating and boring.",
        ]

        user_text = st.text_area("Enter text to analyse", height=100,
                                 placeholder="Type or paste any text here…")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Quick examples:**")
            for s_txt in sample_texts:
                if st.button(s_txt[:50] + "…", key=f"sent_{hash(s_txt)}"):
                    user_text = s_txt

        with col_b:
            if user_text:
                label, score, color = analyse_sentiment(user_text)
                norm_score = max(-1, min(1, score / 5))
                bar_pct = int((norm_score + 1) / 2 * 100)
                st.markdown(f"""
                <div class='nc-card'>
                    <div class='nc-card-title' style='color:{color}; font-size:1.4rem;'>{label}</div>
                    <div class='nc-card-body'>Raw score: <strong style='color:{color};'>{score:+.2f}</strong></div>
                    <div class='xp-bar-wrap' style='margin-top:0.5rem;'>
                        <div class='xp-bar-fill' style='width:{bar_pct}%; background:linear-gradient(90deg, #ff6584, #ffd700, #00d4aa);'></div>
                    </div>
                    <div style='display:flex; justify-content:space-between; color:#8892b0; font-size:0.7rem; margin-top:4px;'>
                        <span>Negative</span><span>Neutral</span><span>Positive</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Analyse ▶", key="run_sent"):
                    add_xp(15, "Ran Sentiment Analysis")


# ──────────────────────────  VISUALISE  ────────────────────────
def page_visualize():
    st.markdown("<div class='section-head'>📊 Visualization Studio</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Explore model performance through beautiful, interactive charts.</div>", unsafe_allow_html=True)

    v_tab1, v_tab2, v_tab3, v_tab4 = st.tabs(["📉 Loss & Accuracy", "🗃️ Confusion Matrix", "🌐 Feature Importance", "📦 Dataset Explorer"])

    # ── Training curves ───────────────────────────────────────
    with v_tab1:
        col_ctrl, col_plot = st.columns([1, 2])
        with col_ctrl:
            arch = st.selectbox("Architecture", ["Simple (1 hidden)", "Medium (3 hidden)", "Deep (6 hidden)"])
            opt = st.selectbox("Optimiser", ["Adam", "SGD", "RMSProp"])
            lr_v = st.select_slider("Learning rate", [0.0001, 0.001, 0.01, 0.1, 0.5])
            n_ep = st.slider("Epochs", 10, 200, 50)
            overfit = st.checkbox("Simulate overfitting", False)
            if st.button("▶ Train Model", key="train_curves"):
                add_xp(30, "Trained a model")

        with col_plot:
            np.random.seed(42)
            epochs = np.arange(1, n_ep + 1)
            decay = {"Adam": 0.85, "SGD": 0.90, "RMSProp": 0.87}[opt]
            depth_bonus = {"Simple (1 hidden)": 0, "Medium (3 hidden)": 0.02, "Deep (6 hidden)": 0.05}[arch]

            # Simulated learning curves
            train_loss = 2.5 * np.exp(-lr_v * 8 * epochs / n_ep * decay) + 0.08 + depth_bonus + np.random.normal(0, 0.015, n_ep)
            train_loss = np.maximum(0.02, train_loss)
            train_acc = 1 - train_loss / 3 + np.random.normal(0, 0.01, n_ep)
            train_acc = np.clip(train_acc, 0, 1)

            if overfit:
                val_loss = train_loss + 0.05 + 0.6 * np.linspace(0, 1, n_ep)**1.5 + np.random.normal(0, 0.02, n_ep)
                val_acc = train_acc - 0.05 - 0.25 * np.linspace(0, 1, n_ep)**1.5 + np.random.normal(0, 0.015, n_ep)
            else:
                val_loss = train_loss + 0.04 + np.random.normal(0, 0.02, n_ep)
                val_acc = train_acc - 0.03 + np.random.normal(0, 0.012, n_ep)
            val_loss = np.maximum(0.03, val_loss)
            val_acc = np.clip(val_acc, 0, 1)

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss Curve", "Accuracy Curve"))
            fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train Loss",
                                     line=dict(color="#6c63ff", width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Val Loss",
                                     line=dict(color="#ff6584", width=2, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=train_acc, name="Train Acc",
                                     line=dict(color="#00d4aa", width=2)), row=1, col=2)
            fig.add_trace(go.Scatter(x=epochs, y=val_acc, name="Val Acc",
                                     line=dict(color="#ffd700", width=2, dash="dot")), row=1, col=2)
            fig.update_layout(
                paper_bgcolor="#181c34", plot_bgcolor="#12162a",
                font=dict(color="#8892b0", size=11),
                height=380,
                legend=dict(bgcolor="#181c34", bordercolor="rgba(108,99,255,0.3)", borderwidth=1),
            )
            fig.update_xaxes(gridcolor="rgba(108,99,255,0.1)", zerolinecolor="rgba(108,99,255,0.2)")
            fig.update_yaxes(gridcolor="rgba(108,99,255,0.1)", zerolinecolor="rgba(108,99,255,0.2)")
            st.plotly_chart(fig, use_container_width=True)

            if overfit and np.max(val_loss) > np.max(train_loss) * 1.3:
                st.warning("⚠️ **Overfitting detected!** Validation loss diverges from training loss. Try Dropout or more data.")

    # ── Confusion matrix ─────────────────────────────────────
    with v_tab2:
        iris = load_iris()
        clf_cm = st.selectbox("Classifier for CM", ["Logistic Regression", "K-Nearest Neighbours", "Decision Tree", "SVM"], key="cm_clf")
        clf_map = {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(max_depth=4),
            "SVM": SVC(),
        }
        X_ir = iris.data; y_ir = iris.target
        Xtr, Xte, ytr, yte = train_test_split(X_ir, y_ir, test_size=0.25, random_state=42)
        sc_ir = StandardScaler()
        clf_ir = clf_map[clf_cm]
        clf_ir.fit(sc_ir.fit_transform(Xtr), ytr)
        y_pred_cm = clf_ir.predict(sc_ir.transform(Xte))
        cm = confusion_matrix(yte, y_pred_cm)
        acc_ir = accuracy_score(yte, y_pred_cm)

        fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=iris.target_names, y=iris.target_names,
                           color_continuous_scale=[[0,"#12162a"],[0.5,"#6c63ff"],[1,"#00d4aa"]],
                           text_auto=True)
        fig_cm.update_layout(paper_bgcolor="#181c34", plot_bgcolor="#12162a",
                             font=dict(color="#8892b0"), height=380,
                             title=dict(text=f"Confusion Matrix — Accuracy: {acc_ir:.2%}", font=dict(color="#e8eaf6")))
        st.plotly_chart(fig_cm, use_container_width=True)
        with st.expander("📋 Full Classification Report"):
            report = classification_report(yte, y_pred_cm, target_names=iris.target_names)
            st.code(report, language="text")

    # ── Feature importance ────────────────────────────────────
    with v_tab3:
        iris = load_iris()
        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt.fit(iris.data, iris.target)
        importances = dt.feature_importances_
        feat_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        fig_fi = go.Figure(go.Bar(
            x=importances, y=feat_names, orientation="h",
            marker=dict(color=["#6c63ff","#00d4aa","#ff6584","#ffd700"],
                        line=dict(color="rgba(255,255,255,0.1)", width=1)),
        ))
        fig_fi.update_layout(
            paper_bgcolor="#181c34", plot_bgcolor="#12162a",
            font=dict(color="#8892b0"), height=320,
            title=dict(text="Feature Importance — Decision Tree (Iris)", font=dict(color="#e8eaf6")),
            xaxis=dict(title="Importance", gridcolor="rgba(108,99,255,0.1)"),
            yaxis=dict(gridcolor="rgba(108,99,255,0.1)"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.markdown("<div class='nc-card'><div class='nc-card-body'>Petal features (length & width) are far more discriminative than sepal features for classifying Iris species. Decision Trees use this information to split nodes.</div></div>", unsafe_allow_html=True)

    # ── Dataset explorer ─────────────────────────────────────
    with v_tab4:
        iris = load_iris()
        df_ex = pd.DataFrame(iris.data, columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
        df_ex["Species"] = [iris.target_names[t] for t in iris.target]
        color_map = {"setosa": "#6c63ff", "versicolor": "#00d4aa", "virginica": "#ff6584"}
        x_ax = st.selectbox("X axis", df_ex.columns[:-1].tolist(), index=0, key="de_x")
        y_ax = st.selectbox("Y axis", df_ex.columns[:-1].tolist(), index=2, key="de_y")
        fig_sc = px.scatter(df_ex, x=x_ax, y=y_ax, color="Species", color_discrete_map=color_map,
                            hover_data=df_ex.columns.tolist(),
                            title="Iris Dataset Explorer")
        fig_sc.update_layout(paper_bgcolor="#181c34", plot_bgcolor="#12162a",
                             font=dict(color="#8892b0"),
                             legend=dict(bgcolor="#181c34", bordercolor="rgba(108,99,255,0.3)", borderwidth=1))
        fig_sc.update_xaxes(gridcolor="rgba(108,99,255,0.1)")
        fig_sc.update_yaxes(gridcolor="rgba(108,99,255,0.1)")
        st.plotly_chart(fig_sc, use_container_width=True)
        st.dataframe(df_ex.sample(10, random_state=42).reset_index(drop=True),
                     use_container_width=True, height=220)


# ──────────────────────────  EXPERIMENT  ───────────────────────
def page_experiment():
    st.markdown("<div class='section-head'>⚙️ Experiment Lab</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Tweak hyperparameters in real time and observe their effect on model performance.</div>", unsafe_allow_html=True)

    exp_tab1, exp_tab2 = st.tabs(["🧠 Neural Network Simulator", "📐 Regression Tuner"])

    # ── NN Simulator ──────────────────────────────────────────
    with exp_tab1:
        col_l, col_r = st.columns([1, 1.6])
        with col_l:
            st.markdown("#### Hyperparameters")
            lr_nn = st.select_slider("Learning Rate", [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0], value=0.01)
            n_epochs = st.slider("Epochs", 10, 300, 80)
            batch_sz = st.select_slider("Batch Size", [8, 16, 32, 64, 128, 256], value=32)
            hidden_layers = st.slider("Hidden Layers", 1, 6, 2)
            neurons_per = st.select_slider("Neurons per Layer", [8, 16, 32, 64, 128, 256], value=64)
            dropout = st.slider("Dropout Rate", 0.0, 0.7, 0.0, 0.05)
            l2_reg = st.slider("L2 Regularisation", 0.0, 0.1, 0.0, 0.001, format="%.3f")
            activation_nn = st.selectbox("Activation", ["ReLU", "Sigmoid", "Tanh"])
            overfit_nn = st.checkbox("Introduce Overfitting", False)

            if st.button("▶ Simulate Training", key="sim_nn"):
                add_xp(40, "Ran Neural Net Experiment")

        with col_r:
            np.random.seed(1234)
            epochs_arr = np.arange(1, n_epochs + 1)

            # Simulate convergence speed based on LR, depth, batch size
            lr_effect = math.log(lr_nn + 1e-6) / math.log(0.01 + 1e-6)
            depth_penalty = 1 + 0.03 * hidden_layers
            batch_effect = math.log2(batch_sz) / math.log2(32)

            base_loss = 2.2 / (1 + 0.08 * lr_effect * epochs_arr / batch_effect) + 0.07 * depth_penalty
            train_l = base_loss + np.random.normal(0, 0.03, n_epochs) + l2_reg * 0.5
            train_l = np.maximum(0.05, train_l)
            train_a = np.clip(0.97 - train_l / 3 - dropout * 0.05, 0, 1) + np.random.normal(0, 0.01, n_epochs)

            if overfit_nn:
                overfit_curve = 0.5 * np.linspace(0, 1, n_epochs) ** 1.4
                val_l = train_l + 0.07 + overfit_curve + np.random.normal(0, 0.025, n_epochs)
                val_a = train_a - 0.07 - overfit_curve * 0.6 + np.random.normal(0, 0.015, n_epochs)
            else:
                val_l = train_l + 0.05 + np.random.normal(0, 0.02, n_epochs)
                val_a = train_a - 0.04 + np.random.normal(0, 0.012, n_epochs)
            val_l = np.maximum(0.06, val_l)
            val_a = np.clip(val_a, 0, 1)

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))
            for (y_data, name, color, dash, row, col) in [
                (train_l, "Train Loss", "#6c63ff", "solid", 1, 1),
                (val_l, "Val Loss",   "#ff6584", "dot",   1, 1),
                (train_a, "Train Acc", "#00d4aa", "solid", 1, 2),
                (val_a, "Val Acc",    "#ffd700", "dot",   1, 2),
            ]:
                fig.add_trace(go.Scatter(x=epochs_arr, y=y_data, name=name,
                                         line=dict(color=color, width=2, dash=dash)), row=row, col=col)
            fig.update_layout(paper_bgcolor="#181c34", plot_bgcolor="#12162a",
                              font=dict(color="#8892b0"), height=360,
                              legend=dict(bgcolor="#181c34", bordercolor="rgba(108,99,255,0.3)", borderwidth=1))
            fig.update_xaxes(gridcolor="rgba(108,99,255,0.1)")
            fig.update_yaxes(gridcolor="rgba(108,99,255,0.1)")
            st.plotly_chart(fig, use_container_width=True)

            # Insight panel
            final_val_acc = float(np.clip(val_a[-1], 0, 1))
            final_val_loss = float(np.maximum(0.06, val_l[-1]))
            col_a, col_b, col_c = st.columns(3)
            col_a.markdown(f"<div class='metric-pill'><div class='val'>{final_val_acc:.1%}</div><div class='lbl'>Final Val Accuracy</div></div>", unsafe_allow_html=True)
            col_b.markdown(f"<div class='metric-pill'><div class='val'>{final_val_loss:.4f}</div><div class='lbl'>Final Val Loss</div></div>", unsafe_allow_html=True)
            col_c.markdown(f"<div class='metric-pill'><div class='val'>{hidden_layers}×{neurons_per}</div><div class='lbl'>Architecture</div></div>", unsafe_allow_html=True)

            # Automated tips
            tips_exp = []
            if lr_nn >= 0.5:     tips_exp.append("⚠️ Learning rate is high — may cause instability or divergence.")
            if lr_nn <= 0.0001:  tips_exp.append("⚠️ Learning rate is very low — training will be slow.")
            if dropout > 0.5:    tips_exp.append("ℹ️ High dropout may underfit; try 0.2–0.4 for most problems.")
            if overfit_nn:       tips_exp.append("🔴 Overfitting detected. Consider Dropout, more data, or L2 regularisation.")
            if hidden_layers > 4 and lr_nn >= 0.1:
                tips_exp.append("⚠️ Deep network + high LR: try learning rate warmup or batch normalisation.")
            if not tips_exp:     tips_exp.append("✅ Hyperparameters look reasonable. Good starting point!")
            for tip in tips_exp:
                st.info(tip)

    # ── Regression Tuner ─────────────────────────────────────
    with exp_tab2:
        col_l2, col_r2 = st.columns([1, 1.6])
        with col_l2:
            st.markdown("#### Regression Settings")
            n_samples_r = st.slider("Samples", 30, 500, 100)
            noise_r = st.slider("Noise σ", 0.0, 10.0, 2.0, 0.5)
            degree_r = st.slider("Polynomial Degree", 1, 8, 1)
            reg_type = st.selectbox("Regularisation", ["None (OLS)", "Ridge (L2)", "Lasso (L1)"])
            alpha_r = 0.0
            if reg_type != "None (OLS)":
                alpha_r = st.slider("Regularisation α", 0.001, 10.0, 1.0, 0.001)
            show_ci = st.checkbox("Show confidence band", True)
            if st.button("▶ Fit Model", key="fit_reg"):
                add_xp(20, "Ran Regression Tuner")

        with col_r2:
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import Ridge, Lasso
            from sklearn.pipeline import Pipeline

            np.random.seed(42)
            X_r = np.sort(np.random.uniform(-3, 3, n_samples_r))
            y_r = 0.5 * X_r ** 2 - X_r + 2 + np.random.normal(0, noise_r, n_samples_r)

            if reg_type == "Ridge (L2)":
                base_est = Ridge(alpha=alpha_r)
            elif reg_type == "Lasso (L1)":
                base_est = Lasso(alpha=alpha_r, max_iter=5000)
            else:
                base_est = LinearRegression()

            pipe = Pipeline([("poly", PolynomialFeatures(degree=degree_r)), ("reg", base_est)])
            pipe.fit(X_r.reshape(-1, 1), y_r)
            x_plot = np.linspace(-3.2, 3.2, 300)
            y_plot = pipe.predict(x_plot.reshape(-1, 1))
            r2_r = pipe.score(X_r.reshape(-1, 1), y_r)

            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=X_r, y=y_r, mode="markers",
                                       marker=dict(color="#6c63ff", size=5, opacity=0.7), name="Data"))
            fig_r.add_trace(go.Scatter(x=x_plot, y=y_plot,
                                       line=dict(color="#00d4aa", width=2.5), name=f"Degree-{degree_r} fit"))
            if show_ci:
                residuals = y_r - pipe.predict(X_r.reshape(-1, 1))
                sigma = np.std(residuals)
                fig_r.add_trace(go.Scatter(x=np.concatenate([x_plot, x_plot[::-1]]),
                                           y=np.concatenate([y_plot + 2*sigma, (y_plot - 2*sigma)[::-1]]),
                                           fill="toself", fillcolor="rgba(0,212,170,0.08)",
                                           line=dict(color="rgba(0,212,170,0)"), name="95% CI"))
            fig_r.update_layout(paper_bgcolor="#181c34", plot_bgcolor="#12162a",
                                font=dict(color="#8892b0"), height=360,
                                title=dict(text=f"Polynomial Regression (deg={degree_r}, R²={r2_r:.4f})", font=dict(color="#e8eaf6")),
                                legend=dict(bgcolor="#181c34", bordercolor="rgba(108,99,255,0.3)", borderwidth=1))
            fig_r.update_xaxes(gridcolor="rgba(108,99,255,0.1)")
            fig_r.update_yaxes(gridcolor="rgba(108,99,255,0.1)")
            st.plotly_chart(fig_r, use_container_width=True)

            if degree_r > 5 and noise_r < 2:
                st.warning("📊 Degree > 5 is likely overfitting. Notice the wild swings at the edges!")
            elif degree_r == 1:
                st.info("📊 Linear fit may underfit the curved data. Try degree 2 or 3.")
            else:
                st.success(f"✅ Reasonable fit! R² = {r2_r:.4f}")


# ──────────────────────────  CHATBOT  ──────────────────────────
# ── Simulated responses (no API needed) ────────────────────────
SIMULATED_QA = {
    "overfitting": "**Overfitting** occurs when your model memorises the training data rather than learning general patterns. Signs: very low training loss but high validation loss. Fix it with: more data, dropout, L2 regularisation, or early stopping.",
    "learning rate": "The **learning rate** controls how large a step gradient descent takes each iteration. Too high → divergence or bouncing. Too low → painfully slow convergence. A typical starting point is 0.001 with Adam optimiser.",
    "neural network": "A **neural network** is a stack of layers, each containing neurons that apply weighted sums + activation functions. Deep networks (many layers) can learn complex hierarchical features automatically from raw data.",
    "relu": "**ReLU** (Rectified Linear Unit) — `f(x) = max(0, x)` — is the most widely used activation function. It avoids the vanishing gradient problem, is cheap to compute, and works well in practice for hidden layers.",
    "sigmoid": "**Sigmoid** squashes input to (0, 1): `f(x) = 1/(1+e^(-x))`. It is used in binary classification output layers. Its downside: saturates (gradients vanish) for very large/small inputs, making deep training hard.",
    "gradient descent": "**Gradient Descent** is the algorithm that minimises the loss function by iteratively moving parameters in the direction of the negative gradient. Variants: Batch GD, SGD, Mini-batch GD, Adam.",
    "backpropagation": "**Backpropagation** is the algorithm to compute gradients efficiently in a neural network. It applies the chain rule layer-by-layer from the output back to the input, then gradient descent updates the weights.",
    "dropout": "**Dropout** randomly zeros out a fraction of neurons during training. This forces the network to learn redundant representations, which reduces overfitting dramatically. Typical rates: 0.2–0.5.",
    "batch normalisation": "**Batch Normalisation** normalises the activations within a mini-batch to have zero mean and unit variance. This speeds up training, allows higher learning rates, and has a mild regularising effect.",
    "transformer": "**Transformers** are the architecture behind GPT, BERT, and Claude. They use self-attention to weigh the importance of different tokens in a sequence — no recurrence needed. Key paper: 'Attention is All You Need' (2017).",
    "cnn": "A **Convolutional Neural Network (CNN)** uses convolutional filters to detect spatial features (edges, textures, shapes) in images. They are translation-invariant and parameter-efficient compared to fully-connected networks.",
    "rnn": "A **Recurrent Neural Network (RNN)** processes sequences step-by-step, maintaining a hidden state. Variants like LSTM and GRU handle long-range dependencies better by using gating mechanisms to control information flow.",
    "accuracy": "**Accuracy** = (correct predictions) / (total predictions). It's simple but misleading for imbalanced datasets. Always pair it with Precision, Recall, and F1-score for a complete picture.",
    "precision": "**Precision** = TP / (TP + FP). Of everything predicted positive, how many are actually positive? High precision = few false alarms. Important when false positives are costly (e.g., spam filters).",
    "recall": "**Recall** = TP / (TP + FN). Of all real positives, how many did we catch? High recall = few missed positives. Important when missing a positive is dangerous (e.g., disease detection).",
    "f1": "**F1 Score** = 2×(Precision×Recall)/(Precision+Recall). The harmonic mean of P and R. Use it when you need a single metric balancing both, especially on imbalanced datasets.",
    "iris": "The **Iris dataset** is a classic benchmark with 150 samples from 3 species (setosa, versicolor, virginica), each described by 4 measurements. It is linearly separable (mostly) and great for classification demos.",
    "supervised": "**Supervised Learning** trains a model on labelled pairs (X, y). The model learns to predict y from X. Examples: regression (continuous y) and classification (categorical y).",
    "unsupervised": "**Unsupervised Learning** finds hidden structure in unlabelled data. Examples include K-Means clustering, PCA for dimensionality reduction, and autoencoders for representation learning.",
}

def simulate_response(question: str) -> str:
    q_lower = question.lower().strip()

    greetings = ("hi", "hello", "hey", "good morning", "good afternoon", "good evening")
    if q_lower in greetings or any(q_lower.startswith(f"{g} ") for g in greetings):
        return (
            "Hi. I’m NeuroCraft AI. I can explain ML concepts, compare algorithms, or walk you through \
            a model step by step. Try asking about overfitting, gradient descent, dropout, transformers, \
            precision/recall, or any topic from the sidebar."
        )

    if any(phrase in q_lower for phrase in ("what can you do", "help me", "how can you help", "who are you")):
        return (
            "I can help with machine learning and AI topics like neural networks, classification, \
            regression, loss functions, activation functions, regularisation, and evaluation metrics. \
            I can also explain the playground demos and suggest next experiments."
        )

    for keyword, answer in SIMULATED_QA.items():
        if keyword in q_lower:
            return answer

    # Generic fallback
    return (
        "I can help with machine learning and AI topics. If you want a quick answer, try a specific \
        concept like **overfitting, gradient descent, dropout, transformers, precision, recall, or \
        batch normalisation**. You can also ask me to compare two algorithms or explain a demo on the page."
    )


def call_anthropic_api(messages: list, api_key: str) -> str:
    """Call the Anthropic API with conversation history."""
    import urllib.request
    system_prompt = (
        "You are NeuroCraft AI, an expert ML tutor embedded in an interactive AI learning platform. "
        "Explain machine learning, deep learning, and AI concepts in clear, beginner-friendly language. "
        "Use examples, analogies, equations, and bullet points where helpful. "
        "Keep answers focused and under 300 words unless the user asks for depth. "
        "Encourage exploration and learning."
    )
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 800,
        "system": system_prompt,
        "messages": messages,
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]
    except Exception as e:
        return f"⚠️ API error: {str(e)}. Falling back to simulation mode."


def page_chatbot():
    st.markdown("<div class='section-head'>💬 AI Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Ask anything about ML, deep learning, and AI — get instant expert answers.</div>", unsafe_allow_html=True)

    api_key = get_anthropic_api_key()
    using_api = bool(api_key and api_key.startswith("sk-ant"))
    mode_badge = ("🟢 Live AI Mode (Anthropic)" if using_api else "🟡 Simulation Mode")
    if using_api:
        source = "sidebar" if st.session_state.get("anthropic_api_key", "").strip() else ("Streamlit secrets" if getattr(st, "secrets", None) and st.secrets.get("ANTHROPIC_API_KEY", "").strip() else "environment")
        status_text = f"Powered by Claude Sonnet using the {source} API key."
    else:
        status_text = "Add your Anthropic API key in the sidebar, or set ANTHROPIC_API_KEY, to enable live AI responses."
    st.markdown(f"<div class='nc-card' style='padding:0.7rem 1.2rem; margin-bottom:1rem;'><span class='tag'>{mode_badge}</span> {status_text}</div>", unsafe_allow_html=True)

    # Suggested questions
    st.markdown("**💡 Try asking:**")
    suggestions = ["What is overfitting?", "Explain gradient descent", "What is dropout?",
                   "How does backpropagation work?", "What's the difference between precision and recall?",
                   "Explain transformers simply", "What is batch normalisation?"]
    cols_sug = st.columns(4)
    for i, sug in enumerate(suggestions[:4]):
        if cols_sug[i].button(sug, key=f"sug_{i}"):
            st.session_state.chat_history.append({"role": "user", "content": sug})

    # Chat display
    if st.session_state.chat_history:
        st.markdown("<div style='max-height:420px; overflow-y:auto; padding-right:4px;'>", unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"<div class='chat-label'>You</div><div class='chat-user'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-label'>🧠 NeuroCraft AI</div><div class='chat-ai'>{msg['content']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Check if last message was user (needs response)
    if (st.session_state.chat_history and
            st.session_state.chat_history[-1]["role"] == "user"):
        with st.spinner("🧠 Thinking…"):
            if using_api:
                reply = call_anthropic_api(st.session_state.chat_history, api_key)
            else:
                time.sleep(0.4)
                reply = simulate_response(st.session_state.chat_history[-1]["content"])
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            add_xp(10, "Used AI Assistant")
            st.rerun()

    # Input
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        user_input = st.text_input("Ask your question…", placeholder="e.g. What is dropout and why is it used?",
                                   label_visibility="collapsed", key="chat_input")
    with col_btn:
        send = st.button("Send ➤", key="chat_send")

    if send and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        st.rerun()

    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()


# ──────────────────────────  ABOUT  ────────────────────────────
def page_about():
    st.markdown("""
    <div class='hero-wrap' style='padding: 2.5rem 3rem;'>
        <div class='hero-badge'>ABOUT THIS PROJECT</div>
        <h1 class='hero-title' style='font-size:2rem;'>NeuroCraft AI Lab</h1>
        <p class='hero-sub'>An open-source interactive ML learning & experimentation platform.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='nc-card'>
            <div class='nc-card-title'>🎯 Mission</div>
            <div class='nc-card-body'>
                Make AI/ML education accessible, interactive, and fun for everyone — from curious beginners
                to aspiring data scientists. Learn by doing, not just reading.
            </div>
        </div>
        <div class='nc-card'>
            <div class='nc-card-title'>🛠️ Tech Stack</div>
            <div class='nc-card-body'>
                <span class='tag'>Python 3.10+</span>
                <span class='tag'>Streamlit</span>
                <span class='tag'>scikit-learn</span>
                <span class='tag'>Plotly</span>
                <span class='tag'>Matplotlib</span>
                <span class='tag'>NumPy</span>
                <span class='tag'>Pandas</span>
                <span class='tag'>Anthropic Claude</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='nc-card'>
            <div class='nc-card-title'>✨ Features</div>
            <div class='nc-card-body'>
                ✅ 8 interactive ML theory topics<br>
                ✅ Live Linear Regression playground<br>
                ✅ Multi-algorithm Classification demo<br>
                ✅ Rule-based Sentiment Analysis<br>
                ✅ Animated training curves (Plotly)<br>
                ✅ Confusion matrix & feature importance<br>
                ✅ Neural network hyperparameter tuner<br>
                ✅ Polynomial regression explorer<br>
                ✅ AI Assistant (simulated + live API)<br>
                ✅ XP gamification system<br>
                ✅ English / Hindi language toggle<br>
                ✅ Dark mode UI throughout
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 🚀 How to Run")
    st.code("""# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py

# 3. (Optional) Set your Anthropic API key in the sidebar
#    to unlock AI-powered chat responses""", language="bash")

    st.markdown("### 📦 requirements.txt")
    st.code("""streamlit>=1.35.0
scikit-learn>=1.4.0
plotly>=5.22.0
matplotlib>=3.8.0
numpy>=1.26.0
pandas>=2.2.0""", language="text")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='nc-card' style='text-align:center;'>
        <div class='nc-card-body'>
            NeuroCraft AI Lab v1.0<br>
            <span class='tag'>Open Source</span> <span class='tag'>MIT License</span> <span class='tag'>Educational</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  ROUTER
# ════════════════════════════════════════════════════════════════
page_map = {
    "Home":       page_home,
    "Learn":      page_learn,
    "Playground": page_playground,
    "Visualize":  page_visualize,
    "Experiment": page_experiment,
    "Chatbot":    page_chatbot,
    "About":      page_about,
}

page_map[active]()
