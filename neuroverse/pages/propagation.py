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
    
    # Title section with nice styling
    st.markdown("""
    <div style='text-align: center; padding: 30px 0;'>
        <h1 style='color: #FF6B6B; font-size: 3em;'>🧠 Forward & Backward Propagation</h1>
        <p style='font-size: 1.2em; color: #666;'>Learn how neural networks learn and improve through training</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📚 Learn", "🎮 Interactive Demo", "📊 Visualize"])
    
    # ========== TAB 1: LEARN ==========
    with tab1:
        st.header("How Neural Networks Learn")
        
        # Two column layout for concepts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔵 Forward Propagation
            
            **What it does:** Predicts the output
            
            The network processes input data through layers:
            
            1. **Multiply by weights** - Each input is scaled
            2. **Add bias** - Shift values to adjust predictions  
            3. **Apply activation** - Non-linear transformation
            4. **Pass to next layer** - Continue the process
            
            **Formula:**
            ```
            Z = X·W + b
            A = activation(Z)
            ```
            
            **Result:** Network makes a prediction for the given input
            """)
        
        with col2:
            st.markdown("""
            ### 🔴 Backward Propagation
            
            **What it does:** Learns from mistakes
            
            The network learns by calculating how wrong it was:
            
            1. **Calculate error** - Compare prediction vs target
            2. **Propagate backward** - Send error back through layers
            3. **Find gradients** - Measure how much each weight caused error
            4. **Update weights** - Adjust to reduce error next time
            
            **Formula:**
            ```
            ∂L/∂W = ∂L/∂A × ∂A/∂Z × ∂Z/∂W
            W_new = W_old - α·∂L/∂W
            ```
            
            **Result:** Weights are adjusted, network improves
            """)
        
        st.divider()
        
        # Learning rate explanation
        st.subheader("Understanding Learning Rate (α)")
        
        lr_col1, lr_col2, lr_col3 = st.columns(3)
        
        with lr_col1:
            st.info("""
            ### High Learning Rate (0.1)
            ✓ Learns faster
            ✗ May overshoot target
            ✗ Less stable
            """)
        
        with lr_col2:
            st.success("""
            ### Medium Learning Rate (0.01)
            ✓ Balanced speed
            ✓ More stable
            ✓ Recommended
            """)
        
        with lr_col3:
            st.warning("""
            ### Low Learning Rate (0.001)
            ✓ Very stable
            ✗ Learns slower
            ✗ Takes more iterations
            """)
    
    # ========== TAB 2: INTERACTIVE DEMO ==========
    with tab2:
        st.header("Try It Yourself!")
        
        # Input section
        st.subheader("Step 1: Set Network Inputs")
        
        input_cols = st.columns(4)
        with input_cols[0]:
            inp1 = st.slider("Input 1", -1.0, 1.0, 0.5, 0.1, key="inp1")
        with input_cols[1]:
            inp2 = st.slider("Input 2", -1.0, 1.0, 0.3, 0.1, key="inp2")
        with input_cols[2]:
            target = st.slider("Target Output", 0.0, 1.0, 0.8, 0.1, key="tgt")
        with input_cols[3]:
            learning_rate = st.slider("Learning Rate (α)", 0.001, 0.1, 0.01, 0.001, key="lr_slider")
        
        # Run the network
        st.subheader("Step 2: Network Computation")
        
        nn = SimpleNeuralNetwork(learning_rate=learning_rate)
        X = np.array([[inp1, inp2]])
        y = np.array([[target]])
        
        # Forward pass
        prediction = nn.forward_pass(X)
        loss = nn.calculate_loss(prediction, y)
        
        # Backward pass
        gradients = nn.backward_pass(y)
        
        # Display results in nice layout
        st.subheader("Step 3: Results")
        
        result_cols = st.columns(5)
        
        with result_cols[0]:
            st.metric(
                "Network Prediction",
                f"{prediction[0][0]:.4f}",
                delta=f"{prediction[0][0]-target:+.4f}",
                delta_color="inverse"
            )
        
        with result_cols[1]:
            st.metric("Target Value", f"{target:.4f}")
        
        with result_cols[2]:
            error_val = abs(prediction[0][0] - target)
            st.metric("Error", f"{error_val:.4f}")
        
        with result_cols[3]:
            st.metric("Loss (BCE)", f"{loss:.6f}")
        
        with result_cols[4]:
            # Color based on error
            if error_val < 0.1:
                color = "green"
                msg = "Excellent!"
            elif error_val < 0.3:
                color = "blue"
                msg = "Good"
            else:
                color = "orange"
                msg = "Training..."
            
            st.markdown(f"<div style='text-align: center; padding: 20px; background: {color}20; border-radius: 10px;'><b>{msg}</b></div>", unsafe_allow_html=True)
        
        # Network structure visualization
        st.subheader("Step 4: Network Architecture")
        
        arch_cols = st.columns(5)
        
        with arch_cols[0]:
            st.markdown("""
            <div style='text-align: center; padding: 20px; background: #E3F2FD; border-radius: 10px;'>
            <b>Input</b><br>
            Layer
            <hr>
            2 neurons
            </div>
            """, unsafe_allow_html=True)
        
        with arch_cols[1]:
            st.markdown("<div style='text-align: center; font-size: 2em; padding-top: 40px;'>→</div>", unsafe_allow_html=True)
        
        with arch_cols[2]:
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background: #F3E5F5; border-radius: 10px;'>
            <b>Hidden</b><br>
            Layer
            <hr>
            {nn.cache['A1'].shape[1]} neurons<br>
            (ReLU)
            </div>
            """, unsafe_allow_html=True)
        
        with arch_cols[3]:
            st.markdown("<div style='text-align: center; font-size: 2em; padding-top: 40px;'>→</div>", unsafe_allow_html=True)
        
        with arch_cols[4]:
            st.markdown("""
            <div style='text-align: center; padding: 20px; background: #FCE4EC; border-radius: 10px;'>
            <b>Output</b><br>
            Layer
            <hr>
            1 neuron<br>
            (Sigmoid)
            </div>
            """, unsafe_allow_html=True)
        
        # Process explanation
        st.subheader("Step 5: What Just Happened")
        
        process_cols = st.columns(3)
        
        with process_cols[0]:
            st.success("""
            **1️⃣ Forward Pass**
            - Input flows through network
            - Weights multiply inputs
            - Activations transform values
            - Network produced prediction
            """)
        
        with process_cols[1]:
            st.warning("""
            **2️⃣ Error Calculation**
            - Compared prediction to target
            - Calculated loss value
            - Measured how far off we are
            - Found room for improvement
            """)
        
        with process_cols[2]:
            st.info("""
            **3️⃣ Backward Pass**
            - Calculated gradients
            - Found which weights caused error
            - Updated weights by α × gradient
            - Network is now slightly better!
            """)
    
    # ========== TAB 3: VISUALIZE ==========
    with tab3:
        st.header("Training Visualization")
        
        st.markdown("""
        Watch how the network improves over multiple iterations!
        """)
        
        col_params, col_demo = st.columns([1, 2])
        
        with col_params:
            st.subheader("Training Parameters")
            epochs_demo = st.slider("Number of Iterations", 10, 200, 50, 10, key="epochs_vis")
            lr_demo = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001, key="lr_vis")
            
            if st.button("🚀 Start Training", key="train_btn"):
                # Training loop
                nn_train = SimpleNeuralNetwork(learning_rate=lr_demo)
                
                # Generate simple training data
                np.random.seed(42)
                X_train = np.random.randn(30, 2)
                y_train = ((X_train[:, 0] > 0) ^ (X_train[:, 1] > 0)).astype(float).reshape(-1, 1)
                
                losses = []
                accuracies = []
                
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                
                for epoch in range(epochs_demo):
                    output = nn_train.forward_pass(X_train)
                    loss = nn_train.calculate_loss(output, y_train)
                    nn_train.backward_pass(y_train)
                    
                    losses.append(loss)
                    acc = np.mean((output > 0.5).astype(float) == y_train)
                    accuracies.append(acc)
                    
                    if (epoch + 1) % max(1, epochs_demo // 10) == 0:
                        progress_bar.progress((epoch + 1) / epochs_demo)
                        status_placeholder.text(f"Iteration {epoch+1}/{epochs_demo} | Loss: {loss:.4f} | Accuracy: {acc:.1%}")
                
                status_placeholder.success(f"✅ Training Complete! Final Loss: {losses[-1]:.4f} | Final Accuracy: {accuracies[-1]:.1%}")
                
                # Plot results
                col_loss, col_acc = st.columns(2)
                
                with col_loss:
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(
                        y=losses, mode='lines+markers',
                        name='Loss', line=dict(color='#FF6B6B', width=3),
                        fill='tozeroy', fillcolor='rgba(255, 107, 107, 0.2)'
                    ))
                    fig_loss.update_layout(
                        title="Loss Over Time",
                        xaxis_title="Iteration",
                        yaxis_title="Loss",
                        hovermode='x unified',
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)
                
                with col_acc:
                    fig_acc = go.Figure()
                    fig_acc.add_trace(go.Scatter(
                        y=accuracies, mode='lines+markers',
                        name='Accuracy', line=dict(color='#4ECDC4', width=3),
                        fill='tozeroy', fillcolor='rgba(78, 205, 196, 0.2)'
                    ))
                    fig_acc.update_layout(
                        title="Accuracy Over Time",
                        xaxis_title="Iteration",
                        yaxis_title="Accuracy",
                        hovermode='x unified',
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)
        
        with col_demo:
            st.info("""
            ### How Training Works
            
            1. **Initial State:** Network has random weights
            2. **Forward Pass:** Input flows through network
            3. **Calculate Loss:** Compare output to expected value
            4. **Backward Pass:** Calculate how to improve
            5. **Update Weights:** Adjust weights using learning rate
            6. **Repeat:** Steps 2-5 continue until trained
            
            ### Key Insights
            
            - **Loss decreases:** Network is learning
            - **Accuracy increases:** Better predictions
            - **Gradient matters:** Shows direction to improve
            - **Learning rate controls:** Speed of learning
            """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; padding: 20px; color: #999;'>
    <small>🎓 NeuroVerse AI Lab - Forward & Backward Propagation Tutorial</small>
    </div>
    """, unsafe_allow_html=True)
