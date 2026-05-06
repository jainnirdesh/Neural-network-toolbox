import re
import shutil
import subprocess
from pathlib import Path

import streamlit as st

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

BASE_DIR = Path(__file__).parent
DIST_DIR = BASE_DIR / "dist"


def ensure_frontend_build() -> bool:
    index_file = DIST_DIR / "index.html"
    if index_file.exists():
        return True

    npm_path = shutil.which("npm")
    if npm_path is None:
        return False

    try:
        subprocess.run([npm_path, "install"], cwd=BASE_DIR, check=True, capture_output=True, text=True)
        subprocess.run([npm_path, "run", "build"], cwd=BASE_DIR, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return False

    return index_file.exists()


def inline_frontend_assets(html_content: str) -> str:
    def read_asset(asset_href: str) -> str:
        asset_path = (DIST_DIR / asset_href.lstrip("/")).resolve()
        return asset_path.read_text(encoding="utf-8")

    def replace_stylesheet(match: re.Match[str]) -> str:
        asset_href = match.group(1)
        if not asset_href.startswith("/assets/"):
            return match.group(0)

        asset_file = DIST_DIR / asset_href.lstrip("/")
        if not asset_file.exists():
            return match.group(0)

        return f"<style>{read_asset(asset_href)}</style>"

    def replace_script(match: re.Match[str]) -> str:
        asset_href = match.group(1)
        if not asset_href.startswith("/assets/"):
            return match.group(0)

        asset_file = DIST_DIR / asset_href.lstrip("/")
        if not asset_file.exists():
            return match.group(0)

        script_content = read_asset(asset_href).replace("</", "<\\/")
        return f'<script type="module">{script_content}</script>'

    html_content = re.sub(
        r'<link[^>]*rel="stylesheet"[^>]*href="([^"]+)"[^>]*>',
        replace_stylesheet,
        html_content,
    )
    html_content = re.sub(
        r'<script[^>]*type="module"[^>]*src="([^"]+)"[^>]*></script>',
        replace_script,
        html_content,
    )
    return html_content


if ensure_frontend_build():
    html_file = DIST_DIR / "index.html"
    html_content = html_file.read_text(encoding="utf-8")
    rendered_html = inline_frontend_assets(html_content)
    st.components.v1.html(rendered_html, height=1200, scrolling=False)
else:
    st.error("Frontend build missing.")
    st.info("Run `npm install` and `npm run build`, or let Streamlit Cloud execute setup.sh during deployment.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 11px; margin-top: 20px;">
        <p>Neural Network Toolbox v1.0 | Nirdesh Jain (2301420025) | Streamlit Deployment</p>
    </div>
    """, unsafe_allow_html=True)
