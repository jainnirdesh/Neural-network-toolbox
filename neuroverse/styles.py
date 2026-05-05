import streamlit as st


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            /* Core Design System */
            :root {
                --primary: #6366f1;
                --primary-dark: #4f46e5;
                --primary-light: #818cf8;
                --accent: #ec4899;
                --success: #10b981;
                --warning: #f59e0b;
                --danger: #ef4444;
                --bg-dark: #0f172a;
                --bg-darker: #0a0f1f;
                --surface: #1e293b;
                --surface-hover: #334155;
                --border: #334155;
                --text-primary: #f1f5f9;
                --text-secondary: #cbd5e1;
                --text-tertiary: #94a3b8;
            }
            
            /* Main App Container */
            .stApp {
                background: linear-gradient(135deg, #0f172a 0%, #1a1f3a 50%, #0f172a 100%);
                color: var(--text-primary);
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;
            }
            
            .block-container {
                padding-top: 2.5rem;
                padding-bottom: 2.5rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            
            /* Typography */
            h1 {
                color: var(--text-primary) !important;
                font-size: 2.5rem !important;
                font-weight: 800 !important;
                letter-spacing: -0.02em !important;
                margin-bottom: 1rem !important;
                background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            h2 {
                color: var(--text-primary) !important;
                font-size: 1.875rem !important;
                font-weight: 700 !important;
                margin-top: 1.5rem !important;
                margin-bottom: 1rem !important;
            }
            
            h3 {
                color: var(--text-primary) !important;
                font-size: 1.25rem !important;
                font-weight: 600 !important;
                margin-top: 1rem !important;
                margin-bottom: 0.75rem !important;
            }
            
            h4, h5, h6 {
                color: var(--text-primary) !important;
                font-weight: 600 !important;
            }
            
            p, li, span {
                color: var(--text-secondary) !important;
                line-height: 1.6 !important;
                font-size: 0.95rem !important;
            }
            
            strong {
                color: var(--text-primary) !important;
                font-weight: 600 !important;
            }
            
            /* Subtitle Style */
            .nv-subtitle {
                color: var(--text-secondary) !important;
                margin-top: -0.5rem;
                margin-bottom: 1.5rem;
                font-size: 1.05rem;
                font-weight: 400;
            }
            
            /* Card Components */
            .nv-card {
                background: rgba(30, 41, 59, 0.6);
                backdrop-filter: blur(10px);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                transition: all 0.3s ease;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            }
            
            .nv-card:hover {
                background: rgba(30, 41, 59, 0.8);
                border-color: var(--primary);
                box-shadow: 0 8px 24px rgba(99, 102, 241, 0.1);
                transform: translateY(-2px);
            }
            
            .nv-card-title {
                font-size: 1.1rem;
                font-weight: 700;
                margin-bottom: 0.75rem;
                color: var(--text-primary) !important;
            }
            
            .nv-card-subtitle {
                font-size: 0.9rem;
                color: var(--text-tertiary) !important;
                margin-bottom: 1rem;
            }
            
            /* Muted Text */
            .nv-muted {
                color: var(--text-tertiary) !important;
                font-size: 0.875rem;
            }
            
            /* Sidebar */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0a0f1f 0%, #1a2332 100%);
                border-right: 1px solid var(--border);
            }
            
            [data-testid="stSidebarNav"] {
                padding: 1.5rem 0;
            }
            
            /* Containers and Blocks */
            [data-testid="stVerticalBlockBorderWrapper"] {
                background: rgba(30, 41, 59, 0.5);
                backdrop-filter: blur(8px);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 1.5rem;
                transition: all 0.3s ease;
            }
            
            [data-testid="stVerticalBlockBorderWrapper"]:hover {
                background: rgba(30, 41, 59, 0.7);
                border-color: var(--primary);
            }
            
            /* Buttons */
            .stButton button {
                background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
                color: white;
                font-weight: 600;
                border: none;
                border-radius: 12px;
                padding: 0.75rem 1.5rem !important;
                font-size: 0.95rem;
                transition: all 0.3s ease;
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
                cursor: pointer;
            }
            
            .stButton button:hover {
                background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
            }
            
            .stButton button:active {
                transform: translateY(0);
                box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
            }
            
            /* Input Fields */
            .stTextInput input, .stTextArea textarea, .stNumberInput input {
                background: rgba(51, 65, 85, 0.5);
                border: 1px solid var(--border) !important;
                border-radius: 10px;
                color: var(--text-primary) !important;
                padding: 0.75rem !important;
                transition: all 0.3s ease;
            }
            
            .stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
                background: rgba(51, 65, 85, 0.7);
                border-color: var(--primary) !important;
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
                outline: none;
            }
            
            /* Select */
            .stSelectbox [data-baseweb=select] {
                background: rgba(51, 65, 85, 0.5);
                border-radius: 10px;
            }
            
            /* Sliders */
            .stSlider [data-baseweb=slider] {
                padding: 0.5rem;
            }
            
            /* Tabs */
            [data-baseweb="tab-list"] {
                border-bottom: 2px solid var(--border) !important;
                gap: 0.5rem;
            }
            
            [data-baseweb="tab"] {
                color: var(--text-secondary) !important;
                border-radius: 8px 8px 0 0;
                font-weight: 600;
            }
            
            [data-baseweb="tab"][aria-selected="true"] {
                color: var(--primary) !important;
                border-bottom: 3px solid var(--primary) !important;
            }
            
            /* Alerts and Callouts */
            .stAlert {
                border-radius: 12px;
                border-left: 4px solid;
                padding: 1rem;
            }
            
            [data-testid="stAlert"] {
                background: rgba(30, 41, 59, 0.6);
                border-radius: 12px;
            }
            
            /* Info Boxes */
            .info-box {
                background: rgba(99, 102, 241, 0.1);
                border: 1px solid var(--primary);
                border-radius: 12px;
                padding: 1.25rem;
                margin: 1rem 0;
            }
            
            .success-box {
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid var(--success);
                border-radius: 12px;
                padding: 1.25rem;
                margin: 1rem 0;
            }
            
            .warning-box {
                background: rgba(245, 158, 11, 0.1);
                border: 1px solid var(--warning);
                border-radius: 12px;
                padding: 1.25rem;
                margin: 1rem 0;
            }
            
            /* Divider */
            .stDivider {
                border-color: var(--border) !important;
                margin: 2rem 0 !important;
            }
            
            /* Metrics */
            [data-testid="metric-container"] {
                background: rgba(30, 41, 59, 0.6);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 1.25rem;
                transition: all 0.3s ease;
            }
            
            [data-testid="metric-container"]:hover {
                background: rgba(30, 41, 59, 0.8);
                border-color: var(--primary);
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
            }
            
            /* Code Blocks */
            code {
                background: rgba(51, 65, 85, 0.5);
                color: #93c5fd;
                padding: 0.2rem 0.5rem;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 0.85rem;
            }
            
            pre {
                background: rgba(15, 23, 42, 0.8);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 1rem;
                overflow-x: auto;
            }
            
            /* Radio and Checkbox */
            [data-baseweb="radio"] {
                margin: 0.5rem 0;
            }
            
            /* Scrollbar */
            ::-webkit-scrollbar {
                width: 10px;
                height: 10px;
            }
            
            ::-webkit-scrollbar-track {
                background: transparent;
            }
            
            ::-webkit-scrollbar-thumb {
                background: rgba(99, 102, 241, 0.4);
                border-radius: 5px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: rgba(99, 102, 241, 0.6);
            }
            
            /* Animations */
            @keyframes fadeIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            @keyframes slideInFromLeft {
                from {
                    opacity: 0;
                    transform: translateX(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            
            .fade-in {
                animation: fadeIn 0.5s ease-out;
            }
            
            .slide-in {
                animation: slideInFromLeft 0.5s ease-out;
            }
            
        </style>
        """,
        unsafe_allow_html=True,
    )
