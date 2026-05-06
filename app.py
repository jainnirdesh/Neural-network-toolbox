import streamlit as st
import os
from pathlib import Path
import base64

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
        html, body, [class*="css"]  {
            margin: 0;
            padding: 0;
        }
        .main {
            padding: 0;
            margin: 0;
        }
        iframe[title="streamlit_app"] {
            border: none;
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
        
        # Serve the React app in an iframe
        st.components.v1.html(html_content, height=1200, scrolling=False)
    else:
        st.error("❌ Build files not found. Please run: npm run build")
        st.info("Run this command locally: npm run build")
else:
    st.error("❌ dist folder not found.")
    st.info("Please run 'npm run build' before deploying")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 11px; margin-top: 20px;">
        <p>Neural Network Toolbox v1.0 | Nirdesh Jain (2301420025) | Streamlit Deployment</p>
    </div>
    """, unsafe_allow_html=True)
