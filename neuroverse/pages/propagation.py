"""
Forward and Backward Propagation Visualization Page
Demonstrates how neural networks learn through forward pass and backpropagation
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_propagation_page():
    """Render the propagation visualization page"""
    st.title("🧠 Forward & Backward Propagation")
    
    st.markdown("""
    Learn how neural networks learn through **forward propagation** (prediction) 
    and **backward propagation** (learning)!
    """)
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["📚 Theory", "🎮 Interactive Demo"])
    
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
            input1 = st.slider("Input 1", min_value=-1.0, max_value=1.0, value=0.5, step=0.1, key="demo_inp1")
            input2 = st.slider("Input 2", min_value=-1.0, max_value=1.0, value=0.3, step=0.1, key="demo_inp2")
            target = st.slider("Target Output", min_value=0.0, max_value=1.0, value=0.8, step=0.1, key="demo_tgt")
            learning_rate_demo = st.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001, key="demo_lr_val")
        
        with col2:
            # Initialize network
            nn = SimpleNeuralNetwork(learning_rate=learning_rate_demo)
            X = np.array([[input1, input2]])
            y_true = np.array([[target]])
            
            # Forward pass
            output = nn.forward_pass(X)
            loss = nn.calculate_loss(output, y_true)
            
            # Backward pass
            gradients = nn.backward_pass(y_true)
            
            st.subheader("📊 Results")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Input", f"[{input1:.3f}, {input2:.3f}]")
            with col_b:
                st.metric("Prediction", f"{output[0][0]:.3f}")
            with col_c:
                st.metric("Target", f"{target:.3f}")
            
            st.metric("Loss", f"{loss:.4f}")
            
            st.info(f"""
            **Network Learned:**
            - Forward: Input → {nn.cache['A1'].shape[1]} hidden neurons → 1 output
            - Backward: Calculated gradients to improve predictions
            - Next iteration: Weights adjusted by learning rate × gradients
            """)

