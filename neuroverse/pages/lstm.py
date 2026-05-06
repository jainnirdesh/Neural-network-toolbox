import streamlit as st
import numpy as np
import plotly.graph_objects as go

from neuroverse.components import page_header, text_card


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def compute_lstm_sequence(x_seq, Wx, Wh, b, h0=None, c0=None):
    # x_seq: (T, D), Wx: (D, 4H), Wh: (H, 4H), b: (4H,)
    T, D = x_seq.shape
    H = Wh.shape[0]
    hs = np.zeros((T, H))
    cs = np.zeros((T, H))
    if h0 is None:
        h_prev = np.zeros(H)
    else:
        h_prev = h0
    if c0 is None:
        c_prev = np.zeros(H)
    else:
        c_prev = c0

    for t in range(T):
        x = x_seq[t]
        z = x.dot(Wx) + h_prev.dot(Wh) + b
        i = sigmoid(z[:H])
        f = sigmoid(z[H:2*H])
        o = sigmoid(z[2*H:3*H])
        g = tanh(z[3*H:4*H])

        c = f * c_prev + i * g
        h = o * tanh(c)

        hs[t] = h
        cs[t] = c

        h_prev = h
        c_prev = c

    return hs, cs, {'i': i, 'f': f, 'o': o, 'g': g}


def render_lstm_page() -> None:
    page_header("LSTM Explorer", "Interactive LSTM cell demo and lightweight visualizations")

    text_card(
        "What is an LSTM?",
        """
        Long Short-Term Memory (LSTM) is a recurrent neural network cell designed to retain
        information across long temporal gaps using gated updates (input, forget, output).
        Use this demo to step through an LSTM cell on a short sequence and inspect gate activations.
        """,
    )

    st.subheader("Demo Settings")
    cols = st.columns((1, 2))
    with cols[0]:
        seq_len = st.slider("Sequence length (T)", 3, 40, 12)
        input_dim = st.slider("Input dim (D)", 1, 8, 1)
        hidden_size = st.slider("Hidden units (H)", 1, 32, 8)
        random_seed = st.number_input("Random seed", value=42, step=1)
        init_mode = st.selectbox("Weight init", ["random", "ones", "small"])

    with cols[1]:
        mode = st.radio("Mode", ["Step-through (educational)", "Run sequence (visualize outputs)"])
        show_weights = st.checkbox("Show raw weight ranges", value=False)

    np.random.seed(int(random_seed))

    # Generate input sequence (user can override later)
    x_seq = np.sin(np.linspace(0, 3 * np.pi, seq_len)).reshape(seq_len, 1)
    if input_dim > 1:
        x_seq = np.hstack([x_seq] * input_dim)

    # Initialize LSTM params
    D = input_dim
    H = hidden_size
    if init_mode == "random":
        Wx = np.random.randn(D, 4 * H) * 0.5
        Wh = np.random.randn(H, 4 * H) * 0.5
        b = np.zeros(4 * H)
    elif init_mode == "ones":
        Wx = np.ones((D, 4 * H)) * 0.1
        Wh = np.ones((H, 4 * H)) * 0.1
        b = np.zeros(4 * H)
    else:
        Wx = np.random.randn(D, 4 * H) * 0.1
        Wh = np.random.randn(H, 4 * H) * 0.1
        b = np.zeros(4 * H)

    if show_weights:
        st.write("Wx range:", Wx.min(), Wx.max())
        st.write("Wh range:", Wh.min(), Wh.max())

    if mode.startswith("Step"):
        st.markdown("#### Step-through LSTM cell")
        t_index = st.slider("Timestep to inspect", 1, seq_len, 1)

        hs, cs, last = compute_lstm_sequence(x_seq[:t_index], Wx, Wh, b)

        st.markdown("**Input sequence (first dims shown)**")
        st.write(np.round(x_seq[:t_index], 3))

        st.markdown("**Gate values at selected timestep**")
        fig = go.Figure()
        gates = {"input": last['i'], "forget": last['f'], "output": last['o'], "candidate": last['g']}
        for name, vals in gates.items():
            fig.add_trace(go.Bar(y=vals, name=name))
        fig.update_layout(barmode='group', height=320, title_text=f"Gate activations (t={t_index})")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Hidden state (h) and cell state (c) at this timestep**")
        st.write("h:", np.round(hs[-1], 3))
        st.write("c:", np.round(cs[-1], 3))

    else:
        st.markdown("#### Sequence visualization")
        hs, cs, _ = compute_lstm_sequence(x_seq, Wx, Wh, b)

        fig = go.Figure()
        # plot first hidden unit and cell state
        fig.add_trace(go.Scatter(y=hs[:, 0], mode='lines+markers', name='h[0]'))
        fig.add_trace(go.Scatter(y=cs[:, 0], mode='lines+markers', name='c[0]'))
        fig.add_trace(go.Scatter(y=x_seq[:, 0], mode='lines', name='input[0]'))
        fig.update_layout(height=360, title_text='Sequence: input vs hidden & cell (first unit)')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Hidden states (heatmap)**")
        st.image(_heatmap_image(hs), use_column_width=True)


def _heatmap_image(mat: np.ndarray):
    # create a simple matplotlib figure and return as PNG bytes
    import matplotlib.pyplot as plt
    from io import BytesIO

    fig, ax = plt.subplots(figsize=(6, 2 + mat.shape[1] * 0.15))
    im = ax.imshow(mat.T, aspect='auto', cmap='RdBu', interpolation='nearest')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Hidden unit')
    fig.colorbar(im, ax=ax)
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf
