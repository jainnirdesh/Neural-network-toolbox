import streamlit as st
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Neural Network Toolbox",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better integration
st.markdown("""
    <style>
        .main {
            padding: 0;
        }
        iframe {
            border: none;
            width: 100%;
            height: 100vh;
        }
    </style>
    """, unsafe_allow_html=True)

# Get the dist folder path
dist_path = Path(__file__).parent / "dist"

if dist_path.exists():
    # Read the built React app
    html_file = dist_path / "index.html"
    
    if html_file.exists():
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Serve the React app
        st.components.v1.html(html_content, height=1200, scrolling=True)
    else:
        st.error("❌ Build files not found. Please run: npm run build")
else:
    st.error("❌ dist folder not found. Please run: npm run build")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 12px;">
        Neural Network Toolbox v1.0 | Streamlit Deployment
    </div>
    """, unsafe_allow_html=True)
