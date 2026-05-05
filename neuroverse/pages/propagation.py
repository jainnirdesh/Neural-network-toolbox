"""
Forward and Backward Propagation Visualization Page
Demonstrates how neural networks learn through forward pass and backpropagation
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class SimpleNeuralNetwork:
    """Simple 3-layer neural network for demonstration"""
    
    def __init__(self, input_size=2, hidden_size=3, output_size=1, learning_rate=0.01):
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        # Store values for visualization
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
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        return x * (1 - x)
    
    def forward_pass(self, X):
        """Forward propagation"""
        # Layer 1
        self.cache['Z1'] = np.dot(X, self.W1) + self.b1
        self.cache['A1'] = self.relu(self.cache['Z1'])
        
        # Layer 2 (output)
        self.cache['Z2'] = np.dot(self.cache['A1'], self.W2) + self.b2
        self.cache['A2'] = self.sigmoid(self.cache['Z2'])
        
        # Store input
        self.cache['X'] = X
        
        return self.cache['A2']
    
    def backward_pass(self, y_true):
        """Backward propagation"""
        m = self.cache['X'].shape[0]
        
        # Output layer gradient
        dZ2 = self.cache['A2'] - y_true
        dW2 = np.dot(self.cache['A1'].T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Hidden layer gradient
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.cache['Z1'])
        dW1 = np.dot(self.cache['X'].T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        
        return {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2,
            'dZ1': dZ1, 'dZ2': dZ2
        }
    
    def calculate_loss(self, y_pred, y_true):
        """Binary cross-entropy loss"""
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def render_propagation_page():
    """Render the propagation visualization page"""
    st.title("🧠 Forward & Backward Propagation")
    
    st.markdown("""
    Learn how neural networks learn through **forward propagation** (prediction) 
    and **backward propagation** (learning)!
    """)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🎮 Interactive Demo", "📊 Training Visualization"])
    
    # === TAB 1: Theory ===
    with tab1:
        st.header("What is Forward & Backward Propagation?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔵 Forward Propagation")
            st.markdown("""
            **Forward Pass** = Network की prediction
            
            1. Input को multiply करो weights से
            2. Bias add करो
            3. Activation function apply करो
            4. अगले layer को pass करो
            
            **Formula:**
            - Z = X·W + b
            - A = activation(Z)
            """)
            
            # Show forward pass visualization
            st.image("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 300 150'%3E%3Crect x='20' y='50' width='40' height='40' fill='%234285F4' /%3E%3Crect x='130' y='30' width='40' height='40' fill='%23EA4335' /%3E%3Crect x='130' y='80' width='40' height='40' fill='%23EA4335' /%3E%3Crect x='240' y='60' width='40' height='30' fill='%2334A853' /%3E%3Cline x1='60' y1='70' x2='130' y2='50' stroke='%23666' /%3E%3Cline x1='60' y1='70' x2='130' y2='100' stroke='%23666' /%3E%3Cline x1='170' y1='50' x2='240' y2='75' stroke='%23666' /%3E%3Cline x1='170' y1='100' x2='240' y2='75' stroke='%23666' /%3E%3Ctext x='25' y='77' font-size='12' fill='white'%3EInput%3C/text%3E%3Ctext x='125' y='63' font-size='12' fill='white'%3EHidden%3C/text%3E%3Ctext x='240' y='82' font-size='12' fill='white'%3EOutput%3C/text%3E%3C/svg%3E", use_column_width=True)
        
        with col2:
            st.subheader("🔴 Backward Propagation")
            st.markdown("""
            **Backward Pass** = Network learning
            
            1. Output error calculate करो
            2. उसे पीछे की ओर propagate करो
            3. Gradients calculate करो
            4. Weights update करो
            
            **Formula:**
            - ∂L/∂W = ∂L/∂A × ∂A/∂Z × ∂Z/∂W
            - W_new = W_old - α·∂L/∂W
            """)
            
            st.markdown("""
            **Gradient Descent:**
            - α = learning rate
            - बड़ा α = तेज़ learning (unstable हो सकता है)
            - छोटा α = धीमी learning (ज़्यादा accurate)
            """)
    
    # === TAB 2: Interactive Demo ===
    with tab2:
        st.header("🎮 Live Demo: Single Forward & Backward Pass")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Input Values")
            input1 = st.slider("Input 1", -1.0, 1.0, 0.5, 0.1)
            input2 = st.slider("Input 2", -1.0, 1.0, 0.3, 0.1)
            target = st.slider("Target Output", 0.0, 1.0, 0.8, 0.1)
            learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
            
            show_gradients = st.checkbox("Show Gradient Values", value=False)
        
        with col2:
            # Initialize network
            nn = SimpleNeuralNetwork(learning_rate=learning_rate)
            X = np.array([[input1, input2]])
            y_true = np.array([[target]])
            
            # Forward pass
            output = nn.forward_pass(X)
            loss = nn.calculate_loss(output, y_true)
            
            # Backward pass
            gradients = nn.backward_pass(y_true)
            
            st.subheader("📊 Forward Pass Results")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Input", f"[{input1:.3f}, {input2:.3f}]")
            with col_b:
                st.metric("Prediction", f"{output[0][0]:.3f}")
            with col_c:
                st.metric("Target", f"{target:.3f}")
            
            st.metric("Loss (Binary CrossEntropy)", f"{loss:.4f}")
            
            if show_gradients:
                st.subheader("📉 Gradient Magnitudes")
                grad_data = {
                    "dW1": np.mean(np.abs(gradients['dW1'])),
                    "dW2": np.mean(np.abs(gradients['dW2'])),
                    "db1": np.mean(np.abs(gradients['db1'])),
                    "db2": np.mean(np.abs(gradients['db2'])),
                }
                
                fig = go.Figure(data=[
                    go.Bar(x=list(grad_data.keys()), y=list(grad_data.values()), 
                           marker=dict(color=['#4285F4', '#EA4335', '#FBBC04', '#34A853']))
                ])
                fig.update_layout(height=300, showlegend=False, 
                                template="plotly_dark", yaxis_title="Gradient Magnitude")
                st.plotly_chart(fig, use_container_width=True)
            
            # Visualize network activations
            st.subheader("🔗 Activation Values")
            
            activations = {
                "Input (X)": X[0],
                "Hidden (A1)": nn.cache['A1'][0],
                "Output (A2)": np.array([output[0][0]])
            }
            
            fig = make_subplots(
                rows=1, cols=3,
                specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "indicator"}]]
            )
            
            fig.add_trace(
                go.Bar(x=[f"x{i+1}" for i in range(len(activations["Input (X)"]))],
                       y=activations["Input (X)"],
                       name="Input", marker=dict(color='#4285F4')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=[f"h{i+1}" for i in range(len(activations["Hidden (A1)"]))],
                       y=activations["Hidden (A1)"],
                       name="Hidden", marker=dict(color='#EA4335')),
                row=1, col=2
            )
            
            # Output indicator
            st.metric("Network Output", f"{output[0][0]:.4f}", 
                     delta=f"Error: {abs(output[0][0] - target):.4f}", 
                     delta_color="inverse")
    
    # === TAB 3: Training Visualization ===
    with tab3:
        st.header("📈 Training Progress Over Multiple Iterations")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Training Parameters")
            epochs = st.slider("Number of Epochs", 10, 500, 100, 10)
            lr = st.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
            
            # Generate random training data (XOR problem)
            np.random.seed(42)
            X_train = np.random.randn(50, 2)
            y_train = ((X_train[:, 0] > 0) ^ (X_train[:, 1] > 0)).astype(float).reshape(-1, 1)
            
            train_button = st.button("🚀 Train Network", type="primary")
        
        with col2:
            if train_button:
                # Train network
                nn_train = SimpleNeuralNetwork(learning_rate=lr)
                losses = []
                accuracies = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for epoch in range(epochs):
                    output = nn_train.forward_pass(X_train)
                    loss = nn_train.calculate_loss(output, y_train)
                    nn_train.backward_pass(y_train)
                    
                    losses.append(loss)
                    accuracy = np.mean((output > 0.5).astype(float) == y_train)
                    accuracies.append(accuracy)
                    
                    if (epoch + 1) % max(1, epochs // 10) == 0:
                        progress_bar.progress((epoch + 1) / epochs)
                        status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Accuracy: {accuracy:.2%}")
                
                progress_bar.progress(1.0)
                status_text.text(f"✅ Training Complete! Final Loss: {losses[-1]:.4f} | Accuracy: {accuracies[-1]:.2%}")
                
                # Plot training curves
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Loss Over Epochs", "Accuracy Over Epochs")
                )
                
                fig.add_trace(
                    go.Scatter(y=losses, mode='lines', name='Loss',
                              line=dict(color='#EA4335', width=2),
                              fill='tozeroy'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(y=accuracies, mode='lines', name='Accuracy',
                              line=dict(color='#34A853', width=2),
                              fill='tozeroy'),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=False, template="plotly_dark")
                fig.update_yaxes(title_text="Loss", row=1, col=1)
                fig.update_yaxes(title_text="Accuracy", row=1, col=2)
                fig.update_xaxes(title_text="Epoch", row=1, col=1)
                fig.update_xaxes(title_text="Epoch", row=1, col=2)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Decision boundary visualization
                st.subheader("🎯 Learned Decision Boundary")
                
                x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
                y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
                
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                     np.linspace(y_min, y_max, 100))
                
                Z = nn_train.forward_pass(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                fig = go.Figure()
                
                fig.add_trace(go.Contourf(
                    x=xx[0], y=yy[:, 0], z=Z,
                    colorscale='RdBu', showscale=False,
                    name='Decision Boundary'
                ))
                
                fig.add_trace(go.Scatter(
                    x=X_train[:, 0], y=X_train[:, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=y_train[:, 0],
                        colorscale='Viridis',
                        showscale=False,
                        line=dict(width=1, color='white')
                    ),
                    name='Training Data'
                ))
                
                fig.update_layout(height=500, template="plotly_dark",
                                title="Neural Network Decision Boundary")
                st.plotly_chart(fig, use_container_width=True)
