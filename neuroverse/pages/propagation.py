"""
Forward and Backward Propagation Visualization Page
Interactive learning tool for neural network training concepts
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class SimpleNeuralNetwork:
    """3-layer neural network for educational purposes"""
    
    def __init__(self, input_size=2, hidden_size=3, output_size=1, learning_rate=0.01):
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        self.cache = {}
        
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward_pass(self, X):
        """Forward propagation through network"""
        self.cache['Z1'] = np.dot(X, self.W1) + self.b1
        self.cache['A1'] = self.relu(self.cache['Z1'])
        
        self.cache['Z2'] = np.dot(self.cache['A1'], self.W2) + self.b2
        self.cache['A2'] = self.sigmoid(self.cache['Z2'])
        
        self.cache['X'] = X
        return self.cache['A2']
    
    def backward_pass(self, y_true):
        """Backward propagation to calculate gradients"""
        m = self.cache['X'].shape[0]
        
        dZ2 = self.cache['A2'] - y_true
        dW2 = np.dot(self.cache['A1'].T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.cache['Z1'])
        dW1 = np.dot(self.cache['X'].T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    
    def calculate_loss(self, y_pred, y_true):
        """Binary cross-entropy loss"""
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def render_propagation_page():
    """Main page for forward and backward propagation visualization"""
    
    # Page styling
    st.set_page_config(layout="wide")
    
    # Hero Section with professional styling
    st.markdown("""
    <div style='margin-bottom: 2.5rem;'>
        <h1 style='margin-bottom: 0.5rem;'>🧠 Forward & Backward Propagation</h1>
        <p class='nv-subtitle'>Understand how neural networks learn through forward prediction and backward error correction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📚 Learn", "🎮 Interactive Demo", "📊 Visualize"])
    
    # ========== TAB 1: LEARN ==========
    with tab1:
        st.markdown("<h2>How Neural Networks Learn</h2>", unsafe_allow_html=True)
        
        # Concepts Section with Professional Cards
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div class='nv-card'>
                <div class='nv-card-title'>🔵 Forward Propagation</div>
                <div class='nv-card-subtitle'>The Prediction Phase</div>
                <p style='color: #cbd5e1; margin-bottom: 1rem;'>
                    The network receives input data and computes predictions by propagating values through layers 
                    using learned weights and activation functions.
                </p>
                <div style='background: rgba(99, 102, 241, 0.1); border-left: 3px solid #6366f1; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
                    <strong style='color: #93c5fd;'>Key Steps:</strong>
                    <ul style='margin-top: 0.5rem; color: #cbd5e1;'>
                        <li>Multiply inputs by weights: <code>Z = X·W</code></li>
                        <li>Add bias terms for adjustment</li>
                        <li>Apply non-linear activation: <code>A = activation(Z)</code></li>
                        <li>Pass to next layer recursively</li>
                    </ul>
                </div>
                <p style='color: #10b981; margin-top: 1rem; font-weight: 600;'>
                    ✓ Result: Network produces a prediction
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='nv-card'>
                <div class='nv-card-title'>🔴 Backward Propagation</div>
                <div class='nv-card-subtitle'>The Learning Phase</div>
                <p style='color: #cbd5e1; margin-bottom: 1rem;'>
                    The network calculates how far the prediction was from the target, then computes gradients 
                    to determine how each weight should be adjusted.
                </p>
                <div style='background: rgba(236, 72, 153, 0.1); border-left: 3px solid #ec4899; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
                    <strong style='color: #f472b6;'>Key Steps:</strong>
                    <ul style='margin-top: 0.5rem; color: #cbd5e1;'>
                        <li>Calculate error: <code>E = target - prediction</code></li>
                        <li>Propagate error backward through layers</li>
                        <li>Compute partial derivatives (gradients)</li>
                        <li>Update weights: <code>W ← W - α·∇L</code></li>
                    </ul>
                </div>
                <p style='color: #10b981; margin-top: 1rem; font-weight: 600;'>
                    ✓ Result: Network weights improve, error decreases
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Learning Rate Deep Dive
        st.markdown("<h3>Understanding Learning Rate (α)</h3>", unsafe_allow_html=True)
        st.markdown(
            "The learning rate controls how much we adjust weights in each iteration. It's a critical hyperparameter that balances convergence speed with stability.",
            help="Too high: weights oscillate and diverge. Too low: training is very slow."
        )
        
        lr_col1, lr_col2, lr_col3 = st.columns(3, gap="medium")
        
        with lr_col1:
            st.markdown("""
            <div class='nv-card' style='border-left: 4px solid #ef4444;'>
                <div class='nv-card-title'>⚡ High (0.1)</div>
                <div style='padding: 1rem; background: rgba(239, 68, 68, 0.05); border-radius: 8px; margin-top: 0.5rem;'>
                    <p><strong style='color: #10b981;'>✓ Pros:</strong></p>
                    <ul style='color: #cbd5e1; margin: 0.5rem 0;'>
                        <li>Trains faster</li>
                        <li>Escapes local minima</li>
                    </ul>
                    <p style='margin-top: 1rem;'><strong style='color: #ef4444;'>✗ Cons:</strong></p>
                    <ul style='color: #cbd5e1;'>
                        <li>May overshoot optimal</li>
                        <li>Unstable convergence</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with lr_col2:
            st.markdown("""
            <div class='nv-card' style='border-left: 4px solid #10b981; border: 2px solid #10b981;'>
                <div class='nv-card-title'>✨ Medium (0.01) — Recommended</div>
                <div style='padding: 1rem; background: rgba(16, 185, 129, 0.05); border-radius: 8px; margin-top: 0.5rem;'>
                    <p><strong style='color: #10b981;'>✓ Ideal Balance:</strong></p>
                    <ul style='color: #cbd5e1; margin: 0.5rem 0;'>
                        <li>Good convergence speed</li>
                        <li>Stable training</li>
                        <li>Reliable results</li>
                    </ul>
                    <p style='margin-top: 1rem; color: #10b981; font-weight: 600;'>👍 Start here for most tasks</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with lr_col3:
            st.markdown("""
            <div class='nv-card' style='border-left: 4px solid #f59e0b;'>
                <div class='nv-card-title'>🐢 Low (0.001)</div>
                <div style='padding: 1rem; background: rgba(245, 158, 11, 0.05); border-radius: 8px; margin-top: 0.5rem;'>
                    <p><strong style='color: #10b981;'>✓ Pros:</strong></p>
                    <ul style='color: #cbd5e1; margin: 0.5rem 0;'>
                        <li>Very stable</li>
                        <li>Fine-tuning</li>
                    </ul>
                    <p style='margin-top: 1rem;'><strong style='color: #f59e0b;'>✗ Cons:</strong></p>
                    <ul style='color: #cbd5e1;'>
                        <li>Very slow learning</li>
                        <li>More iterations needed</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # ========== TAB 2: INTERACTIVE DEMO ==========
    with tab2:
        st.markdown("<h2>Interactive Demonstration</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p class='nv-subtitle'>
            Adjust the inputs and learning rate below to see how the network processes data and improves its predictions in real-time.
        </p>
        """, unsafe_allow_html=True)
        
        # Input Controls Section
        st.markdown("<h3>Step 1: Configure Inputs</h3>", unsafe_allow_html=True)
        
        input_cols = st.columns(4, gap="medium")
        with input_cols[0]:
            st.markdown("<p style='color: #cbd5e1; font-size: 0.9rem; margin-bottom: 0.5rem;'><strong>Input 1</strong></p>", unsafe_allow_html=True)
            inp1 = st.slider("Input 1", -1.0, 1.0, 0.5, 0.1, key="inp1", label_visibility="collapsed")
        with input_cols[1]:
            st.markdown("<p style='color: #cbd5e1; font-size: 0.9rem; margin-bottom: 0.5rem;'><strong>Input 2</strong></p>", unsafe_allow_html=True)
            inp2 = st.slider("Input 2", -1.0, 1.0, 0.3, 0.1, key="inp2", label_visibility="collapsed")
        with input_cols[2]:
            st.markdown("<p style='color: #cbd5e1; font-size: 0.9rem; margin-bottom: 0.5rem;'><strong>Target Output</strong></p>", unsafe_allow_html=True)
            target = st.slider("Target Output", 0.0, 1.0, 0.8, 0.1, key="tgt", label_visibility="collapsed")
        with input_cols[3]:
            st.markdown("<p style='color: #cbd5e1; font-size: 0.9rem; margin-bottom: 0.5rem;'><strong>Learning Rate (α)</strong></p>", unsafe_allow_html=True)
            learning_rate = st.slider("Learning Rate (α)", 0.001, 0.1, 0.01, 0.001, key="lr_slider", label_visibility="collapsed")
        
        st.divider()
        
        # Computation Section
        st.markdown("<h3>Step 2: Network Computation</h3>", unsafe_allow_html=True)
        
        nn = SimpleNeuralNetwork(learning_rate=learning_rate)
        X = np.array([[inp1, inp2]])
        y = np.array([[target]])
        
        prediction = nn.forward_pass(X)
        loss = nn.calculate_loss(prediction, y)
        gradients = nn.backward_pass(y)
        
        # Results Section
        st.markdown("<h3>Step 3: Results</h3>", unsafe_allow_html=True)
        
        result_cols = st.columns(5, gap="medium")
        
        with result_cols[0]:
            st.markdown("""
            <div class='nv-card'>
                <div class='nv-card-subtitle'>Network Prediction</div>
                <div style='font-size: 1.75rem; font-weight: 700; color: #6366f1; margin: 0.5rem 0;'>
                    {:.4f}
                </div>
                <div style='font-size: 0.85rem; color: #94a3b8;'>Delta: {:.4f}</div>
            </div>
            """.format(prediction[0][0], prediction[0][0]-target), unsafe_allow_html=True)
        
        with result_cols[1]:
            st.markdown("""
            <div class='nv-card'>
                <div class='nv-card-subtitle'>Target Value</div>
                <div style='font-size: 1.75rem; font-weight: 700; color: #10b981; margin: 0.5rem 0;'>
                    {:.4f}
                </div>
                <div style='font-size: 0.85rem; color: #94a3b8;'>Expected output</div>
            </div>
            """.format(target), unsafe_allow_html=True)
        
        with result_cols[2]:
            error_val = abs(prediction[0][0] - target)
            st.markdown("""
            <div class='nv-card'>
                <div class='nv-card-subtitle'>Absolute Error</div>
                <div style='font-size: 1.75rem; font-weight: 700; color: #f59e0b; margin: 0.5rem 0;'>
                    {:.4f}
                </div>
                <div style='font-size: 0.85rem; color: #94a3b8;'>|Pred - Target|</div>
            </div>
            """.format(error_val), unsafe_allow_html=True)
        
        with result_cols[3]:
            st.markdown("""
            <div class='nv-card'>
                <div class='nv-card-subtitle'>Loss (BCE)</div>
                <div style='font-size: 1.75rem; font-weight: 700; color: #ec4899; margin: 0.5rem 0;'>
                    {:.6f}
                </div>
                <div style='font-size: 0.85rem; color: #94a3b8;'>Binary cross-entropy</div>
            </div>
            """.format(loss), unsafe_allow_html=True)
        
        with result_cols[4]:
            error_val = abs(prediction[0][0] - target)
            if error_val < 0.1:
                badge_color, badge_text, badge_emoji = "#10b981", "Excellent", "🎯"
            elif error_val < 0.3:
                badge_color, badge_text, badge_emoji = "#6366f1", "Good", "✨"
            else:
                badge_color, badge_text, badge_emoji = "#f59e0b", "Training", "🔄"
            
            st.markdown(f"""
            <div class='nv-card' style='text-align: center; border-left: 4px solid {badge_color};'>
                <div style='font-size: 2rem;'>{badge_emoji}</div>
                <div style='font-size: 1.2rem; font-weight: 700; color: {badge_color}; margin-top: 0.5rem;'>
                    {badge_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Network Architecture
        st.markdown("<h3>Step 4: Network Architecture</h3>", unsafe_allow_html=True)
        
        arch_cols = st.columns([1, 0.3, 1, 0.3, 1])
        
        with arch_cols[0]:
            st.markdown("""
            <div class='nv-card' style='text-align: center; background: rgba(99, 102, 241, 0.1); border-left: 4px solid #6366f1;'>
                <div style='font-size: 0.9rem; color: #cbd5e1; margin-bottom: 0.5rem;'>INPUT LAYER</div>
                <div style='font-size: 1.5rem; font-weight: 700; color: #6366f1;'>2</div>
                <div style='font-size: 0.75rem; color: #94a3b8;'>neurons</div>
            </div>
            """, unsafe_allow_html=True)
        
        with arch_cols[1]:
            st.markdown("<div style='text-align: center; font-size: 1.5rem; color: #6366f1; margin-top: 1.5rem;'>→</div>", unsafe_allow_html=True)
        
        with arch_cols[2]:
            st.markdown(f"""
            <div class='nv-card' style='text-align: center; background: rgba(236, 72, 153, 0.1); border-left: 4px solid #ec4899;'>
                <div style='font-size: 0.9rem; color: #cbd5e1; margin-bottom: 0.5rem;'>HIDDEN LAYER</div>
                <div style='font-size: 1.5rem; font-weight: 700; color: #ec4899;'>{nn.cache['A1'].shape[1]}</div>
                <div style='font-size: 0.75rem; color: #94a3b8;'>neurons (ReLU)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with arch_cols[3]:
            st.markdown("<div style='text-align: center; font-size: 1.5rem; color: #6366f1; margin-top: 1.5rem;'>→</div>", unsafe_allow_html=True)
        
        with arch_cols[4]:
            st.markdown("""
            <div class='nv-card' style='text-align: center; background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981;'>
                <div style='font-size: 0.9rem; color: #cbd5e1; margin-bottom: 0.5rem;'>OUTPUT LAYER</div>
                <div style='font-size: 1.5rem; font-weight: 700; color: #10b981;'>1</div>
                <div style='font-size: 0.75rem; color: #94a3b8;'>neuron (Sigmoid)</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Process Explanation
        st.markdown("<h3>Step 5: What Happened</h3>", unsafe_allow_html=True)
        
        process_cols = st.columns(3, gap="medium")
        
        with process_cols[0]:
            st.markdown("""
            <div class='nv-card' style='border-left: 4px solid #6366f1;'>
                <div class='nv-card-title'>1️⃣ Forward Pass</div>
                <div style='color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;'>
                    <strong>Input → Weights → Activations → Output</strong><br>
                    Data propagated through network layers to generate prediction
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with process_cols[1]:
            st.markdown("""
            <div class='nv-card' style='border-left: 4px solid #f59e0b;'>
                <div class='nv-card-title'>2️⃣ Error Calculation</div>
                <div style='color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;'>
                    <strong>Compare → Measure → Quantify</strong><br>
                    Prediction compared to target, loss computed
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with process_cols[2]:
            st.markdown("""
            <div class='nv-card' style='border-left: 4px solid #10b981;'>
                <div class='nv-card-title'>3️⃣ Backward Pass</div>
                <div style='color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;'>
                    <strong>Gradients → Update → Improve</strong><br>
                    Weights adjusted to reduce error in next iteration
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # ========== TAB 3: VISUALIZE ==========
    with tab3:
        st.markdown("<h2>Training Visualization</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p class='nv-subtitle'>
            Watch the network improve over multiple iterations. The loss should decrease while accuracy increases.
        </p>
        """, unsafe_allow_html=True)
        
        col_params, col_demo = st.columns([1, 2], gap="large")
        
        with col_params:
            st.markdown("<h3>Training Configuration</h3>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class='nv-card'>
                <div class='nv-card-subtitle'>Adjust these parameters to control training</div>
            </div>
            """, unsafe_allow_html=True)
            
            epochs_demo = st.slider("Number of Iterations", 10, 200, 50, 10, key="epochs_vis")
            lr_demo = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001, key="lr_vis")
            
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            
            if st.button("🚀 Start Training", key="train_btn", use_container_width=True):
                # Training loop
                nn_train = SimpleNeuralNetwork(learning_rate=lr_demo)
                
                # Generate training data
                np.random.seed(42)
                X_train = np.random.randn(30, 2)
                y_train = ((X_train[:, 0] > 0) ^ (X_train[:, 1] > 0)).astype(float).reshape(-1, 1)
                
                losses = []
                accuracies = []
                
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                chart_placeholder = st.empty()
                
                for epoch in range(epochs_demo):
                    output = nn_train.forward_pass(X_train)
                    loss = nn_train.calculate_loss(output, y_train)
                    nn_train.backward_pass(y_train)
                    
                    losses.append(loss)
                    acc = np.mean((output > 0.5).astype(float) == y_train)
                    accuracies.append(acc)
                    
                    # Update progress
                    progress = (epoch + 1) / epochs_demo
                    progress_placeholder.progress(progress)
                    
                    status_placeholder.markdown(f"""
                    <div class='nv-card'>
                        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.5rem;'>
                            <div>
                                <div class='nv-card-subtitle'>Iteration</div>
                                <div style='font-size: 1.3rem; font-weight: 700; color: #6366f1;'>{epoch+1}/{epochs_demo}</div>
                            </div>
                            <div>
                                <div class='nv-card-subtitle'>Progress</div>
                                <div style='font-size: 1.3rem; font-weight: 700; color: #10b981;'>{progress*100:.0f}%</div>
                            </div>
                            <div>
                                <div class='nv-card-subtitle'>Loss</div>
                                <div style='font-size: 1.3rem; font-weight: 700; color: #f59e0b;'>{loss:.4f}</div>
                            </div>
                            <div>
                                <div class='nv-card-subtitle'>Accuracy</div>
                                <div style='font-size: 1.3rem; font-weight: 700; color: #ec4899;'>{acc*100:.1f}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                status_placeholder.markdown(f"""
                <div class='nv-card' style='background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981;'>
                    <div style='font-weight: 700; color: #10b981;'>✅ Training Complete!</div>
                    <div style='margin-top: 0.5rem; color: #cbd5e1; font-size: 0.9rem;'>
                        Final Loss: <strong>{losses[-1]:.4f}</strong> | 
                        Final Accuracy: <strong>{accuracies[-1]*100:.1f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Create combined visualization
                with chart_placeholder.container():
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(
                            y=losses, mode='lines+markers',
                            name='Loss', 
                            line=dict(color='#f59e0b', width=3),
                            fill='tozeroy', 
                            fillcolor='rgba(245, 158, 11, 0.1)',
                            marker=dict(size=4)
                        ))
                        fig_loss.update_layout(
                            title="<b>Loss Over Time</b>",
                            xaxis_title="Iteration",
                            yaxis_title="Binary Cross-Entropy Loss",
                            hovermode='x unified',
                            template="plotly_dark",
                            height=400,
                            paper_bgcolor='rgba(15, 23, 42, 0.5)',
                            plot_bgcolor='rgba(30, 41, 59, 0.3)',
                            font=dict(color='#cbd5e1', family='Arial'),
                            title_font=dict(size=14)
                        )
                        st.plotly_chart(fig_loss, use_container_width=True, config={'responsive': True})
                    
                    with chart_col2:
                        fig_acc = go.Figure()
                        fig_acc.add_trace(go.Scatter(
                            y=accuracies, mode='lines+markers',
                            name='Accuracy', 
                            line=dict(color='#10b981', width=3),
                            fill='tozeroy', 
                            fillcolor='rgba(16, 185, 129, 0.1)',
                            marker=dict(size=4)
                        ))
                        fig_acc.update_layout(
                            title="<b>Accuracy Over Time</b>",
                            xaxis_title="Iteration",
                            yaxis_title="Accuracy",
                            hovermode='x unified',
                            template="plotly_dark",
                            height=400,
                            paper_bgcolor='rgba(15, 23, 42, 0.5)',
                            plot_bgcolor='rgba(30, 41, 59, 0.3)',
                            font=dict(color='#cbd5e1', family='Arial'),
                            title_font=dict(size=14)
                        )
                        st.plotly_chart(fig_acc, use_container_width=True, config={'responsive': True})
        
        with col_demo:
            st.markdown("<h3>How Training Works</h3>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class='nv-card' style='border-left: 4px solid #6366f1;'>
                <div class='nv-card-title'>🎯 Training Process</div>
                <ol style='color: #cbd5e1; margin-top: 1rem;'>
                    <li><strong>Initialize:</strong> Network starts with random weights</li>
                    <li><strong>Forward:</strong> Feed training data through network</li>
                    <li><strong>Compute Loss:</strong> Compare outputs to expected values</li>
                    <li><strong>Backward:</strong> Calculate gradients using backpropagation</li>
                    <li><strong>Update:</strong> Adjust weights by α × gradient</li>
                    <li><strong>Repeat:</strong> Steps 2-5 until convergence</li>
                </ol>
            </div>
            
            <div class='nv-card' style='border-left: 4px solid #10b981; margin-top: 1.5rem;'>
                <div class='nv-card-title'>✨ Key Metrics</div>
                <div style='margin-top: 1rem;'>
                    <div style='margin-bottom: 1rem;'>
                        <strong style='color: #f59e0b;'>Loss:</strong>
                        <p style='color: #cbd5e1; font-size: 0.9rem; margin-top: 0.25rem;'>
                            Measures how far predictions are from targets. Lower is better.
                        </p>
                    </div>
                    <div>
                        <strong style='color: #10b981;'>Accuracy:</strong>
                        <p style='color: #cbd5e1; font-size: 0.9rem; margin-top: 0.25rem;'>
                            Percentage of correct predictions. Higher is better.
                        </p>
                    </div>
                </div>
            </div>
            
            <div class='nv-card' style='border-left: 4px solid #ec4899; margin-top: 1.5rem;'>
                <div class='nv-card-title'>💡 Tips</div>
                <ul style='color: #cbd5e1; margin-top: 1rem;'>
                    <li>Start with <strong>0.01</strong> learning rate</li>
                    <li>Run <strong>50-100</strong> iterations for this task</li>
                    <li>Watch for <strong>Loss decreasing</strong></li>
                    <li>Expect <strong>Accuracy ≥ 80%</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; color: #94a3b8; border-top: 1px solid #334155;'>
        <div style='font-weight: 600; margin-bottom: 0.5rem;'>🎓 NeuroVerse AI Lab</div>
        <div style='font-size: 0.85rem;'>Educational Platform for Deep Learning & Neural Networks</div>
        <div style='font-size: 0.8rem; margin-top: 0.5rem; color: #64748b;'>
            Learn, experiment, and master forward and backward propagation concepts
        </div>
    </div>
    """, unsafe_allow_html=True)
