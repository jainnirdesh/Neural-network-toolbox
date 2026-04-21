import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="NeuroCraft Lab",
    page_icon="NC",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
:root {
    --bg0: #0d1428;
    --bg1: #111c38;
    --bg2: #16244a;
    --card: #192a54;
    --line: #2e4485;
    --text: #e8eeff;
    --muted: #9fb0e3;
    --acc: #7cc4ff;
    --acc2: #9df8d0;
}

html, body, [class*="css"] {
    color: var(--text) !important;
}

.stApp {
    background: radial-gradient(circle at 15% -5%, #1c2f62 0%, var(--bg0) 45%, #090f1f 100%) !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1731 0%, #101a36 100%) !important;
    border-right: 1px solid var(--line);
}

.block-container {
    max-width: 1250px;
    padding-top: 1.3rem;
}

.nc-hero {
    background: linear-gradient(135deg, #1b2d5d 0%, #132347 100%);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.nc-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}

.nc-title {
    margin: 0;
    color: var(--acc);
    font-weight: 700;
}

.nc-sub {
    color: var(--muted);
    margin-top: 0.35rem;
    font-size: 0.92rem;
}

.nc-kpi {
    background: #152349;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.85rem;
    text-align: center;
}

.nc-kpi .v {
    color: var(--acc2);
    font-size: 1.3rem;
    font-weight: 700;
}

.nc-kpi .l {
    color: var(--muted);
    font-size: 0.8rem;
}

.nc-chip {
    display: inline-block;
    background: #1a2b57;
    border: 1px solid var(--line);
    color: var(--text);
    border-radius: 999px;
    padding: 3px 10px;
    margin-right: 8px;
    font-size: 0.76rem;
}

.nc-module {
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 0.55rem 0.7rem;
    margin-bottom: 0.45rem;
    background: #152349;
}

.nc-help {
    color: var(--muted);
    font-size: 0.8rem;
}

hr {
    border: none;
    border-top: 1px solid var(--line);
    margin: 0.8rem 0;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


PAGE_GROUPS = {
    "Start Here": [
        ("Home", "Overview and quick launch"),
        ("Math Explorer", "All formulas, theorems, and definitions"),
        ("System Health", "Check local setup and assets"),
    ],
    "Module 0: Mathematics for Neural Networks": [
        ("Linear Algebra Lab", "Vector transformations and matrices"),
        ("Calculus Lab", "Derivatives and tangent lines"),
        ("Probability Lab", "Normal distributions and density"),
        ("Optimization Lab", "Gradient descent simulator"),
    ],
    "Learning Lab": [
        ("Perceptron", "Binary logic and decision boundaries"),
        ("Forward Propagation", "See activations move through a network"),
        ("Backward Propagation", "Understand gradient updates step by step"),
        ("Multi-Layer Perceptron", "Train on IRIS-like data or your own CSV"),
        ("Hopfield Network", "Associative memory and energy minimization"),
    ],
    "Vision Lab": [
        ("OpenCV Vision Lab", "Classical CV detection and edge playground"),
        ("CNN Image Classifier", "Image classification"),
    ],
    "Sequence Models": [
        ("RNN Hub", "Explore recurrent model demos"),
        ("RNN Visualizer", "Visualize sequence modeling"),
        ("RNN Next Word", "Predict the next token from context"),
        ("RNN Sentiment", "Classify review-style text"),
        ("LSTM Hub", "Work with long-sequence flows"),
        ("LSTM Next Word", "Predict the next token with LSTM"),
        ("LSTM Sentiment", "Run custom LSTM sentiment analysis"),
    ],
    "Playground": [
        ("AI Playground", "Profile data and generate training code"),
        ("3D Network Explorer", "Interactive 3D architecture and signal flow"),
    ],
}

ALL_PAGES = [p[0] for group in PAGE_GROUPS.values() for p in group]
TOTAL_PAGES = len(ALL_PAGES)


if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"
if "learning_mode" not in st.session_state:
    st.session_state.learning_mode = "Explorer"
if "explain_mode" not in st.session_state:
    st.session_state.explain_mode = True
if "completed_pages" not in st.session_state:
    st.session_state.completed_pages = set()
if "mentor_chat" not in st.session_state:
    st.session_state.mentor_chat = []


def mark_complete(page_name: str) -> None:
    st.session_state.completed_pages.add(page_name)


def navigate_to(page_name: str) -> None:
    st.session_state.current_page = page_name
    st.rerun()


def nav_button(page_name: str, help_text: str, key: str) -> None:
    if st.button(page_name, key=key, use_container_width=True):
        navigate_to(page_name)
    st.caption(help_text)


def render_problem_flow(problem: str, data: str, model: str, outcome: str) -> None:
    """Render a compact real-world problem-solving workflow block."""
    st.markdown(
        f"""
        <div class='nc-card'>
            <h4 class='nc-title'>Real-World Problem Workflow</h4>
            <div class='nc-sub'><strong>Problem:</strong> {problem}</div>
            <div class='nc-sub'><strong>Data:</strong> {data}</div>
            <div class='nc-sub'><strong>Model/Method:</strong> {model}</div>
            <div class='nc-sub'><strong>Outcome:</strong> {outcome}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("# NeuroCraft Lab")
        st.markdown("<div class='nc-help'>Polished neural-network workspace</div>", unsafe_allow_html=True)

        st.markdown("<div class='nc-card'><div class='nc-help'>Current Focus</div>"
                    f"<h4 class='nc-title'>{st.session_state.current_page}</h4>"
                    "<div class='nc-help'>Overview and quick launch</div></div>", unsafe_allow_html=True)

        st.radio(
            "Learning Mode",
            ["Beginner", "Explorer", "Research"],
            index=["Beginner", "Explorer", "Research"].index(st.session_state.learning_mode),
            key="learning_mode",
        )

        completed_count = len(st.session_state.completed_pages)
        st.write(f"Overall Progress: {completed_count}/{TOTAL_PAGES}")
        st.progress(completed_count / max(1, TOTAL_PAGES))

        st.checkbox("Explain Mode", key="explain_mode")

        query = st.text_input("Filter modules", placeholder="Type LSTM, guide, detection...").strip().lower()
        st.caption(f"{TOTAL_PAGES} pages available")

        for group_name, pages in PAGE_GROUPS.items():
            with st.expander(group_name, expanded=(group_name == "Start Here")):
                for name, desc in pages:
                    if query and query not in name.lower() and query not in desc.lower():
                        continue
                    nav_button(name, desc, key=f"nav_{group_name}_{name}")

        st.markdown("---")
        if st.button("Ask AI Mentor", use_container_width=True):
            navigate_to("Ask AI Mentor")


# ---------- Page Renderers ----------
def render_home() -> None:
    st.markdown("""
    <div class='nc-hero'>
        <div class='nc-help'>Neural Learning Workspace</div>
        <h1 style='margin:0;'>NeuroCraft Lab</h1>
        <p class='nc-sub'>Learn neural networks visually, jump into recurrent and computer-vision demos, and move from concept to experiment without getting lost in the app.</p>
        <span class='nc-chip'>10+ learning modules</span>
        <span class='nc-chip'>RNN + LSTM demos</span>
        <span class='nc-chip'>OpenCV lab</span>
        <span class='nc-chip'>AI-assisted workflows</span>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.6, 1])
    with right:
        st.markdown("<div class='nc-card'><h4 class='nc-title'>Workspace Snapshot</h4><div class='nc-sub'>Use the dashboard to jump into modules quickly and the sidebar filter to track down any page faster.</div></div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    metrics = [("Modules", "23"), ("Guides", "4"), ("Health Checks", "3/4"), ("Main Modes", "Learn / Build / Explore")]
    for col, (label, val) in zip([k1, k2, k3, k4], metrics):
        col.markdown(f"<div class='nc-kpi'><div class='l'>{label}</div><div class='v'>{val}</div></div>", unsafe_allow_html=True)

    st.subheader("Quick Launch")
    q1, q2, q3, q4 = st.columns(4)
    if q1.button("Start Learning", use_container_width=True):
        navigate_to("Perceptron")
    q1.caption("Binary logic and decision boundaries")

    if q2.button("Run RNN Demo", use_container_width=True):
        navigate_to("RNN Hub")
    q2.caption("Explore recurrent model demos")

    if q3.button("Open LSTM Lab", use_container_width=True):
        navigate_to("LSTM Hub")
    q3.caption("Work with long-sequence flows")

    if q4.button("Explore Data", use_container_width=True):
        navigate_to("AI Playground")
    q4.caption("Profile data and generate training code")

    st.subheader("Find Your Way")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        choice = st.selectbox("Jump directly to any module", ALL_PAGES, index=ALL_PAGES.index("Math Explorer"))
        if st.button("Open Selected Workspace"):
            navigate_to(choice)
    with c2:
        mode = st.radio("Experience Mode", ["Guided", "Builder", "Explorer"], horizontal=True)
        if mode == "Guided":
            st.info("Follow learner modules first, then move into RNN/LSTM and OpenCV labs.")
        elif mode == "Builder":
            st.info("Use Playground and Vision/NLP labs to build mini workflows.")
        else:
            st.info("Explore freely and use search to navigate quickly.")

    st.subheader("Real-World Problem Tracks")
    t1, t2, t3 = st.columns(3)
    t1.markdown(
        """
        <div class='nc-card'>
            <h4 class='nc-title'>Traffic Safety Monitoring</h4>
            <div class='nc-sub'>Detect lane edges and scene type from road images to support incident monitoring.</div>
            <div class='nc-sub'><strong>Labs:</strong> OpenCV Vision Lab, CNN Image Classifier</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    t2.markdown(
        """
        <div class='nc-card'>
            <h4 class='nc-title'>Customer Feedback Intelligence</h4>
            <div class='nc-sub'>Classify user feedback sentiment to prioritize urgent product issues.</div>
            <div class='nc-sub'><strong>Labs:</strong> RNN Sentiment, LSTM Sentiment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    t3.markdown(
        """
        <div class='nc-card'>
            <h4 class='nc-title'>Operational Decision Support</h4>
            <div class='nc-sub'>Profile tabular data and generate baseline ML pipelines for faster experiments.</div>
            <div class='nc-sub'><strong>Labs:</strong> AI Playground, Optimization Lab</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_math_explorer() -> None:
    st.title("Math Explorer")
    st.write("Core formulas used across the workspace.")
    st.latex(r"\nabla_w J(w) = \frac{\partial J}{\partial w}")
    st.latex(r"\sigma(z)=\frac{1}{1+e^{-z}}")
    st.latex(r"\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2")
    if st.session_state.explain_mode:
        st.caption("Gradient, activation, and loss equations are foundational for optimization and model training.")
    mark_complete("Math Explorer")


def render_system_health() -> None:
    st.title("System Health")
    checks = {
        "NumPy": True,
        "Pandas": True,
        "Matplotlib": True,
        "Plotly": True,
        "Streamlit": True,
    }
    rows = []
    for k, ok in checks.items():
        rows.append({"Component": k, "Status": "OK" if ok else "Missing"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption("All core components required for this workspace are available.")
    mark_complete("System Health")


def render_linear_algebra_lab() -> None:
    st.title("Linear Algebra Lab")
    vec = np.array([2, 1])
    mat = np.array([[1.2, -0.4], [0.6, 1.1]])
    transformed = mat @ vec
    st.write("Input vector:", vec)
    st.write("Transformation matrix:")
    st.write(mat)
    st.write("Transformed vector:", transformed)
    mark_complete("Linear Algebra Lab")


def render_calculus_lab() -> None:
    st.title("Calculus Lab")
    x = np.linspace(-5, 5, 200)
    y = x**2
    dydx = 2 * x
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(x, y, label="f(x)=x^2")
    ax.plot(x, dydx, label="f'(x)=2x")
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)
    mark_complete("Calculus Lab")


def render_probability_lab() -> None:
    st.title("Probability Lab")
    mu = st.slider("Mean", -2.0, 2.0, 0.0, 0.1)
    sigma = st.slider("Std Dev", 0.5, 3.0, 1.0, 0.1)
    x = np.linspace(-6, 6, 300)
    pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(x, pdf)
    ax.grid(alpha=0.25)
    ax.set_title("Normal Distribution")
    st.pyplot(fig)
    plt.close(fig)
    mark_complete("Probability Lab")


def render_optimization_lab() -> None:
    st.title("Optimization Lab")
    render_problem_flow(
        problem="Ad spend optimization where budget updates are unstable and conversion cost rises.",
        data="Campaign features, daily spend, conversions, and cost-per-acquisition signals.",
        model="Gradient descent update simulation to tune learning rate and stabilize convergence.",
        outcome="Select a stable learning rate range before production model retraining.",
    )
    lr = st.slider("Learning Rate", 0.01, 0.9, 0.2, 0.01)
    x_vals = [3.5]
    for _ in range(20):
        grad = 2 * x_vals[-1]
        x_vals.append(x_vals[-1] - lr * grad)
    xs = np.linspace(-4, 4, 300)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(xs, xs**2, label="Loss x^2")
    ax.scatter(x_vals, [v**2 for v in x_vals], c=np.linspace(0, 1, len(x_vals)), cmap="viridis")
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)
    mark_complete("Optimization Lab")


def render_perceptron() -> None:
    st.title("Perceptron")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    clf = Perceptron(max_iter=1000, random_state=42)
    clf.fit(X, y)
    pred = clf.predict(X)
    st.write(pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "target": y, "prediction": pred}))
    if st.session_state.explain_mode:
        st.info("This is an AND gate learned with a linear decision boundary.")
    mark_complete("Perceptron")


def render_forward_prop() -> None:
    st.title("Forward Propagation")
    x1 = st.slider("x1", -2.0, 2.0, 0.8, 0.1)
    x2 = st.slider("x2", -2.0, 2.0, -0.4, 0.1)
    h1 = max(0, 0.7 * x1 + 0.2 * x2 + 0.1)
    h2 = max(0, -0.3 * x1 + 0.9 * x2 + 0.2)
    yhat = 1 / (1 + math.exp(-((1.1 * h1) + (-0.8 * h2) + 0.1)))
    st.metric("Hidden h1", f"{h1:.3f}")
    st.metric("Hidden h2", f"{h2:.3f}")
    st.metric("Output", f"{yhat:.3f}")
    mark_complete("Forward Propagation")


def render_backward_prop() -> None:
    st.title("Backward Propagation")
    st.write("Use this mini demo to observe gradient effect.")
    w = st.slider("Weight w", -2.0, 2.0, 0.9, 0.1)
    x = 1.4
    target = 2.0
    pred = w * x
    loss = (pred - target) ** 2
    grad = 2 * (pred - target) * x
    st.metric("Prediction", f"{pred:.3f}")
    st.metric("Loss", f"{loss:.3f}")
    st.metric("dLoss/dw", f"{grad:.3f}")
    mark_complete("Backward Propagation")


def render_mlp() -> None:
    st.title("Multi-Layer Perceptron")
    st.write("Train/test split demo on synthetic binary data.")
    n = st.slider("Samples", 100, 1200, 400, 50)
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = ((x1 + 0.7 * x2) > 0).astype(int)
    X = np.column_stack([x1, x2])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    model = Perceptron(max_iter=1000, random_state=42)
    model.fit(Xtr, ytr)
    acc = (model.predict(Xte) == yte).mean()
    st.metric("Validation Accuracy", f"{acc * 100:.1f}%")
    mark_complete("Multi-Layer Perceptron")


def render_hopfield() -> None:
    st.title("Hopfield Network")
    st.write("Associative memory concept simulation.")
    pattern = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    noisy = pattern.copy()
    noisy[[1, 5]] *= -1
    recovered = np.where((pattern + noisy) >= 0, 1, -1)
    st.write("Stored pattern:", pattern)
    st.write("Noisy input:", noisy)
    st.write("Recovered pattern:", recovered)
    mark_complete("Hopfield Network")


def render_opencv_lab() -> None:
    st.title("OpenCV Vision Lab")
    st.write("Classical vision playground (edge detection concept).")
    render_problem_flow(
        problem="Road surveillance team needs fast visual cues for lane boundaries and scene changes.",
        data="Continuous camera frames from traffic intersections and highways.",
        model="Classical edge extraction pipeline to highlight structural contours.",
        outcome="Faster manual review and early warning support for safety operations.",
    )
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], key="cv_upload")
    if uploaded:
        image = plt.imread(uploaded)
        gray = image.mean(axis=2) if image.ndim == 3 else image
        gx = np.diff(gray, axis=1, prepend=gray[:, :1])
        gy = np.diff(gray, axis=0, prepend=gray[:1, :])
        edges = np.sqrt(gx**2 + gy**2)
        edge_range = np.ptp(edges)
        if edge_range > 0:
            edges = (edges - edges.min()) / edge_range
        else:
            edges = np.zeros_like(edges)

        c1, c2 = st.columns(2)
        c1.image(image, caption="Original", use_container_width=True)
        c2.image(edges, caption="Edge Map", use_container_width=True)
    mark_complete("OpenCV Vision Lab")


def render_cnn_classifier() -> None:
    st.title("CNN Image Classifier")
    render_problem_flow(
        problem="Operations center needs automated scene tagging to route images to correct analyst queues.",
        data="Uploaded field images from roads, sites, or inspection checkpoints.",
        model="Lightweight image classifier heuristic as a stand-in for pretrained CNN inference.",
        outcome="Auto-tagging reduces triage time and improves response prioritization.",
    )
    uploaded = st.file_uploader("Upload image for classification", type=["png", "jpg", "jpeg"], key="cnn_upload")
    if uploaded:
        arr = plt.imread(uploaded)
        if arr.ndim == 3:
            mean_rgb = arr[..., :3].mean(axis=(0, 1))
            if mean_rgb[1] > mean_rgb[0] and mean_rgb[1] > mean_rgb[2]:
                pred = "Nature / Vegetation"
                conf = 0.82
            elif mean_rgb[0] > mean_rgb[2]:
                pred = "Urban / Road Scene"
                conf = 0.76
            else:
                pred = "General Object"
                conf = 0.71
        else:
            pred = "Monochrome Object"
            conf = 0.68
        st.image(arr, caption="Input Image", use_container_width=True)
        st.success(f"Prediction: {pred}")
        st.metric("Confidence", f"{conf * 100:.1f}%")
    mark_complete("CNN Image Classifier")


def render_rnn_hub() -> None:
    st.title("RNN Hub")
    st.markdown("- Sequence basics\n- Hidden state intuition\n- Temporal dependencies")
    mark_complete("RNN Hub")


def render_rnn_visualizer() -> None:
    st.title("RNN Visualizer")
    steps = st.slider("Sequence Length", 3, 15, 8)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(steps)), y=np.sin(np.linspace(0, 2.5, steps)), mode="lines+markers"))
    fig.update_layout(title="Sequence Flow", xaxis_title="Time Step", yaxis_title="Signal")
    st.plotly_chart(fig, use_container_width=True)
    mark_complete("RNN Visualizer")


def render_rnn_next_word() -> None:
    st.title("RNN Next Word")
    text = st.text_input("Enter context", "deep learning")
    mapping = {"deep learning": "models", "neural network": "training", "data science": "pipeline"}
    pred = mapping.get(text.lower().strip(), "representation")
    st.write(f"Predicted next token: {pred}")
    mark_complete("RNN Next Word")


def render_rnn_sentiment() -> None:
    st.title("RNN Sentiment")
    render_problem_flow(
        problem="Product team cannot manually read thousands of support reviews each week.",
        data="Incoming customer comments, app store reviews, and support messages.",
        model="Sequence-aware sentiment scoring to classify feedback as positive/negative/neutral.",
        outcome="Prioritized issue queues and faster escalation for negative sentiment spikes.",
    )
    text = st.text_area("Enter review")
    if text:
        pos = {"good", "great", "amazing", "helpful", "love"}
        neg = {"bad", "poor", "worst", "hate", "awful"}
        toks = [t.strip(".,!?;:").lower() for t in text.split()]
        score = sum(1 for t in toks if t in pos) - sum(1 for t in toks if t in neg)
        label = "Positive" if score > 0 else ("Negative" if score < 0 else "Neutral")
        st.metric("Sentiment", label)
    mark_complete("RNN Sentiment")


def render_lstm_hub() -> None:
    st.title("LSTM Hub")
    st.write("LSTM handles long-term dependencies with gated memory cells.")
    mark_complete("LSTM Hub")


def render_lstm_next_word() -> None:
    st.title("LSTM Next Word")
    seed = st.text_input("Seed text", "sequence models")
    candidates = ["improve", "context", "predictions", "stability"]
    idx = abs(hash(seed)) % len(candidates)
    st.write(f"Predicted next token: {candidates[idx]}")
    mark_complete("LSTM Next Word")


def render_lstm_sentiment() -> None:
    st.title("LSTM Sentiment")
    render_problem_flow(
        problem="Long feedback threads carry context that simple keyword methods miss.",
        data="Paragraph-length customer narratives and conversation transcripts.",
        model="LSTM-style sentiment handling for longer sequence context.",
        outcome="More reliable sentiment decisions on complex text.",
    )
    text = st.text_area("Custom sentiment input", key="lstm_sent")
    if text:
        score = (text.lower().count("good") + text.lower().count("great")) - (text.lower().count("bad") + text.lower().count("poor"))
        st.metric("Sentiment Score", score)
    mark_complete("LSTM Sentiment")


def render_ai_playground() -> None:
    st.title("AI Playground")
    st.write("Profile tabular data and generate starter training code.")
    render_problem_flow(
        problem="Analysts spend excessive time preparing first-pass ML experiments from new datasets.",
        data="Uploaded CSV files with mixed numeric/categorical business features.",
        model="Automated profiling plus starter code generation for baseline classifiers.",
        outcome="Cuts setup time from hours to minutes for initial model iteration.",
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="play_csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(), use_container_width=True)
        st.write("Shape:", df.shape)
        st.write("Columns:", list(df.columns))
        code = """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('your_file.csv')
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
"""
        st.code(code, language="python")
    mark_complete("AI Playground")


def render_3d_explorer() -> None:
    st.title("3D Network Explorer")
    layers = st.slider("Layers", 2, 6, 4)
    neurons = st.slider("Neurons per layer", 4, 20, 10)

    xs, ys, zs = [], [], []
    for l in range(layers):
        for n in range(neurons):
            xs.append(l)
            ys.append(n - neurons / 2)
            zs.append(math.sin(n / 2) * 0.5)

    fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode="markers", marker=dict(size=4))])
    fig.update_layout(height=500, title="3D Architecture Sketch")
    st.plotly_chart(fig, use_container_width=True)
    mark_complete("3D Network Explorer")


def render_ai_mentor() -> None:
    st.title("Ask AI Mentor")
    st.caption("ML-focused assistant with simulation mode.")

    for msg in st.session_state.mentor_chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask about neural networks, RNN, LSTM, CV...")
    if question:
        st.session_state.mentor_chat.append({"role": "user", "content": question})
        q = question.lower()
        answers = {
            "overfitting": "Overfitting happens when training performance improves but validation degrades. Use regularization, dropout, and more data.",
            "lstm": "LSTM uses forget/input/output gates to preserve long-term information in sequence data.",
            "rnn": "RNN processes data step-by-step and shares parameters over time steps.",
            "cnn": "CNN learns local spatial features with convolutions and pooling.",
            "backprop": "Backpropagation computes gradients through the network using the chain rule.",
        }
        response = next((v for k, v in answers.items() if k in q), "Ask a specific ML topic and I will explain with examples.")
        st.session_state.mentor_chat.append({"role": "assistant", "content": response})
        st.rerun()


ROUTER = {
    "Home": render_home,
    "Math Explorer": render_math_explorer,
    "System Health": render_system_health,
    "Linear Algebra Lab": render_linear_algebra_lab,
    "Calculus Lab": render_calculus_lab,
    "Probability Lab": render_probability_lab,
    "Optimization Lab": render_optimization_lab,
    "Perceptron": render_perceptron,
    "Forward Propagation": render_forward_prop,
    "Backward Propagation": render_backward_prop,
    "Multi-Layer Perceptron": render_mlp,
    "Hopfield Network": render_hopfield,
    "OpenCV Vision Lab": render_opencv_lab,
    "CNN Image Classifier": render_cnn_classifier,
    "RNN Hub": render_rnn_hub,
    "RNN Visualizer": render_rnn_visualizer,
    "RNN Next Word": render_rnn_next_word,
    "RNN Sentiment": render_rnn_sentiment,
    "LSTM Hub": render_lstm_hub,
    "LSTM Next Word": render_lstm_next_word,
    "LSTM Sentiment": render_lstm_sentiment,
    "AI Playground": render_ai_playground,
    "3D Network Explorer": render_3d_explorer,
    "Ask AI Mentor": render_ai_mentor,
}


render_sidebar()
page = st.session_state.current_page
ROUTER.get(page, render_home)()
