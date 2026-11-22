"""
CropCare AI – Intelligent Crop Disease Detection System
Main Entry Point & Streamlit Navigation Router
"""

import sys
from pathlib import Path

# Add project root directory to sys.path to guarantee clean imports
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

# Streamlit Page Configuration (Must be first Streamlit command)
st.set_page_config(
    page_title="CropCare AI – Intelligent Crop Disease Detection System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Database Schema
from modules.database import init_db
init_db()

# Load Custom CSS Styling
CSS_PATH = ROOT_DIR / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Import Page Views
from pages.dashboard import render_dashboard_page
from pages.disease_detection import render_disease_detection_page
from pages.history import render_history_page
from pages.crop_information import render_crop_information_page
from pages.analytics import render_analytics_page
from pages.about import render_about_page


def main():
    """Main application lifecycle and sidebar routing."""

    # Sidebar Brand Header
    st.sidebar.markdown("""
        <div class="sidebar-brand">
            <h2>🌿 CropCare AI</h2>
            <p>Intelligent Crop Disease Detection System</p>
        </div>
    """, unsafe_allow_html=True)

    # Navigation Pages
    pages = {
        "Dashboard": render_dashboard_page,
        "Disease Detection": render_disease_detection_page,
        "Detection History": render_history_page,
        "Crop Information": render_crop_information_page,
        "Analytics": render_analytics_page,
        "About Project": render_about_page,
    }

    # Handle internal page switches from buttons
    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = "Dashboard"

    nav_options = list(pages.keys())
    current_index = nav_options.index(st.session_state["nav_selection"]) if st.session_state["nav_selection"] in nav_options else 0

    nav_icons = {
        "Dashboard": "📊",
        "Disease Detection": "🔬",
        "Detection History": "📜",
        "Crop Information": "📚",
        "Analytics": "📈",
        "About Project": "ℹ️"
    }

    selected_page_name = st.sidebar.radio(
        "Navigation",
        options=nav_options,
        index=current_index,
        format_func=lambda x: f"{nav_icons.get(x, '•')} {x}",
        label_visibility="collapsed"
    )

    # Sync session state
    st.session_state["nav_selection"] = selected_page_name

    # Sidebar IoT Hardware Status Indicator
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style="padding: 0.5rem 0; font-size: 0.8rem; color: #495057;">
            <div style="font-weight: 700; color: #1b4332; margin-bottom: 0.2rem;">IoT Edge Node Status</div>
            <div style="display: flex; align-items: center; gap: 0.4rem; color: #2d6a4f; font-weight: 600;">
                <span style="height: 9px; width: 9px; background-color: #52b788; border-radius: 50%; display: inline-block;"></span>
                ESP32-CAM Client Ready
            </div>
            <div style="font-size: 0.72rem; color: #8a99a8; margin-top: 0.2rem;">Endpoint: /capture (HTTP)</div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar System Metadata
    st.sidebar.markdown("""
        <div style="margin-top: 1.5rem; font-size: 0.72rem; color: #8a99a8; border-top: 1px solid #e2ece5; padding-top: 0.8rem;">
            <b>CropCare AI System</b><br>
            IoT & AI-Based Crop Disease Detection<br>
            MobileNetV2 + PSO Optimization
        </div>
    """, unsafe_allow_html=True)

    # Render selected page
    render_func = pages.get(selected_page_name, render_dashboard_page)
    render_func()


if __name__ == "__main__":
    main()
