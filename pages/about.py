"""
CropCare AI - About Project Page
Provides comprehensive system documentation, architecture workflow diagrams,
objective breakdowns, and IoT edge hardware integration specifications.
"""

import streamlit as st
from config import BENCHMARK_STATS
from modules.camera import get_hardware_specs


def render_about_page():
    """Renders the About Project interface."""

    # Project Header
    st.markdown("""
        <div class="hero-banner">
            <span class="badge-tag">Project Documentation & Architecture</span>
            <h1 style="margin-top: 0.5rem;">IoT-Based Crop Disease Detection System Using Artificial Intelligence</h1>
            <p class="subtitle">CropCare AI – Intelligent Crop Health Monitoring and Disease Classification Platform</p>
        </div>
    """, unsafe_allow_html=True)

    # Project Overview & Motivation
    st.markdown("### 🌾 1. Project Overview & Motivation")
    st.markdown("""
        Plant diseases pose a severe threat to global food security, causing estimated annual crop yield losses
        of 20% to 40% worldwide. Traditional manual scouting by agricultural specialists is labor-intensive,
        subjective, slow, and inaccessible to smallholder farmers in remote farming regions.

        This project introduces **CropCare AI**, an end-to-end cyber-physical IoT system combining
        **Deep Learning (MobileNetV2)** with an **ESP32-CAM wireless microcontroller** and a modern **Streamlit web dashboard**.
        The system enables automated, rapid, and objective identification of foliar crop diseases across 5 vital agricultural crops
        from digital photographs.
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Project Objectives
    st.markdown("### 🎯 2. Core Project Objectives")
    obj_c1, obj_c2 = st.columns(2)

    with obj_c1:
        st.markdown("""
            * **Automated Foliage Disease Diagnosis:** Detect and classify crop diseases directly from individual leaf images without manual laboratory testing.
            * **MobileNetV2 Transfer Learning:** Deploy a lightweight, depthwise-separable convolutional neural network optimized for fast inference with minimal compute footprint.
            * **Metaheuristic Optimization (PSO):** Explore Particle Swarm Optimization to fine-tune network hyper-parameters and feature weighting for improved convergence.
        """)

    with obj_c2:
        st.markdown("""
            * **IoT Edge Image Acquisition:** Integrate an ESP32-CAM microcontroller module equipped with an OV2640 sensor for wireless image capture from agricultural fields.
            * **Modern Web Interface:** Develop a responsive, full-featured Streamlit dashboard with SQLite history logging, real-time analytics, and CSV exports.
            * **Agronomic Decision Support:** Provide actionable non-chemical cultural management advisories to mitigate disease spread sustainably.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # End-to-End System Workflow
    st.markdown("### 🔄 3. End-to-End System Workflow Architecture")
    st.markdown("""
        The complete diagnostic pipeline flows from edge acquisition to host inference and persistent storage:
    """)

    # Visual Workflow Diagram Cards
    st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 1.5rem; margin-bottom: 1.5rem;">
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 0.8rem; text-align: center;">
                <div style="flex: 1; min-width: 130px; background: #f4fbf6; padding: 0.9rem; border-radius: 10px; border: 1px solid #74c69d;">
                    <div style="font-size: 1.5rem;">📷</div>
                    <div style="font-weight: 700; color: #1b4332; font-size: 0.9rem;">1. Image Acquisition</div>
                    <div style="font-size: 0.75rem; color: #6c757d;">ESP32-CAM / Upload</div>
                </div>
                <div style="font-size: 1.2rem; color: #52b788; font-weight: bold;">➔</div>
                <div style="flex: 1; min-width: 130px; background: #f4fbf6; padding: 0.9rem; border-radius: 10px; border: 1px solid #74c69d;">
                    <div style="font-size: 1.5rem;">⚙️</div>
                    <div style="font-weight: 700; color: #1b4332; font-size: 0.9rem;">2. Preprocessing</div>
                    <div style="font-size: 0.75rem; color: #6c757d;">224×224 px, [-1, 1]</div>
                </div>
                <div style="font-size: 1.2rem; color: #52b788; font-weight: bold;">➔</div>
                <div style="flex: 1; min-width: 130px; background: #f4fbf6; padding: 0.9rem; border-radius: 10px; border: 1px solid #74c69d;">
                    <div style="font-size: 1.5rem;">🧠</div>
                    <div style="font-weight: 700; color: #1b4332; font-size: 0.9rem;">3. MobileNetV2</div>
                    <div style="font-size: 0.75rem; color: #6c757d;">Depthwise CNN + PSO</div>
                </div>
                <div style="font-size: 1.2rem; color: #52b788; font-weight: bold;">➔</div>
                <div style="flex: 1; min-width: 130px; background: #f4fbf6; padding: 0.9rem; border-radius: 10px; border: 1px solid #74c69d;">
                    <div style="font-size: 1.5rem;">🏷️</div>
                    <div style="font-weight: 700; color: #1b4332; font-size: 0.9rem;">4. Classification</div>
                    <div style="font-size: 0.75rem; color: #6c757d;">17 Pathology Classes</div>
                </div>
                <div style="font-size: 1.2rem; color: #52b788; font-weight: bold;">➔</div>
                <div style="flex: 1; min-width: 130px; background: #f4fbf6; padding: 0.9rem; border-radius: 10px; border: 1px solid #74c69d;">
                    <div style="font-size: 1.5rem;">💾</div>
                    <div style="font-weight: 700; color: #1b4332; font-size: 0.9rem;">5. History & Logs</div>
                    <div style="font-size: 0.75rem; color: #6c757d;">SQLite & Plotly Data</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Deep Learning Model & Benchmark Notice
    st.markdown("### 🔬 4. Deep Learning Architecture & Evaluation Metrics")
    st.markdown(f"""
        The core vision classifier is constructed upon Google's **MobileNetV2** architecture,
        which leverages inverted residual blocks and depthwise separable convolutions to dramatically
        reduce computational complexity (FLOPs) and model parameter size while preserving high representational capacity.

        **Quantitative Benchmark Results:**
        * **Baseline MobileNetV2 Test Accuracy:** `{BENCHMARK_STATS['baseline_test_accuracy']}%`
        * **Baseline Macro F1-Score:** `{BENCHMARK_STATS['baseline_macro_f1']}%`
        * **Baseline Test Loss:** `{BENCHMARK_STATS['baseline_test_loss']}`
        * **PSO-Optimized MobileNetV2 Accuracy:** `{BENCHMARK_STATS['pso_test_accuracy']}%`
        * **Dataset Volume:** `27,494 images` partitioned into 17 target classes across 5 crops.
    """)

    st.warning("""
        **⚠️ Important Benchmark Notice for Demonstrations:**
        The 96.97% test accuracy represents the model's empirical baseline performance on the curated, holdout test dataset.
        It is an offline research evaluation benchmark, **not a guaranteed confidence metric for every arbitrary leaf photograph**.
        Field variables such as motion blur, harsh shadows, occlusions, and severe overexposure can impact real-time prediction confidence.
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Hardware & ESP32-CAM Architecture
    st.markdown("### 📷 5. IoT Edge Hardware Architecture (ESP32-CAM)")
    st.markdown("""
        The hardware acquisition module employs an **ESP32-CAM (Ai-Thinker)** board, serving as a low-cost,
        Wi-Fi enabled edge camera node:
    """)

    specs = get_hardware_specs()
    spec_table = [{"Component / Parameter": k.replace("_", " ").title(), "Technical Specification": v} for k, v in specs.items()]
    st.table(spec_table)

    st.markdown("""
        **Inference Topology Note:**
        Deep learning inference is executed centrally on the host computer or cloud server (CPU/GPU) rather than directly
        on the micro-controller. The ESP32-CAM handles image sensor acquisition and HTTP streaming over the local Wi-Fi subnet,
        transmitting raw JPEG frames to the host workstation.
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Technology Stack Badges
    st.markdown("### 🛠️ 6. Full Technology Stack")
    st.markdown("""
        - **Frontend Framework:** Streamlit (Custom responsive multi-page architecture)
        - **Backend:** Python 3.10+
        - **Deep Learning:** TensorFlow 2.x, Keras 3.x
        - **Vision Architecture:** MobileNetV2 (Depthwise Separable CNN)
        - **Optimization:** Particle Swarm Optimization (PSO)
        - **Computer Vision & Image Processing:** Pillow (PIL), OpenCV (cv2)
        - **Interactive Visualization:** Plotly Graph Objects & Express
        - **Persistent Database:** SQLite 3
        - **IoT Edge Microcontroller:** ESP32-CAM (OV2640 2MP Camera Sensor)
    """)

    st.markdown("---")
    st.caption("CropCare AI • IoT-Based Crop Disease Detection System Using Artificial Intelligence")
