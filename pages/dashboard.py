"""
CropCare AI - Home Dashboard Page
Presents model architecture specifications, benchmark dataset statistics,
live detection activity KPIs, recent diagnostic records, and distribution charts.
"""

import streamlit as st
import pandas as pd
from config import BENCHMARK_STATS, SUPPORTED_CROPS
from modules.database import get_dashboard_statistics
from modules.analytics import (
    create_health_donut_chart,
    create_crop_distribution_bar,
    create_disease_frequency_bar
)


def render_dashboard_page():
    """Renders the CropCare AI Home Dashboard."""

    # Hero Banner
    st.markdown("""
        <div class="hero-banner">
            <span class="badge-tag">AI & IoT in Agriculture</span>
            <h1 style="margin-top: 0.5rem;">CropCare AI – Intelligent Crop Disease Detection System</h1>
            <p class="subtitle">AI-Powered Plant Health Monitoring and Disease Classification</p>
        </div>
    """, unsafe_allow_html=True)

    # Introductory description
    st.markdown("""
        **CropCare AI** is an intelligent agricultural decision-support system designed to empower farmers,
        agronomists, and researchers with automated crop disease classification. Leveraging a transfer-learned
        **MobileNetV2** deep convolutional neural network enhanced with **Particle Swarm Optimization (PSO)**,
        the platform diagnoses 17 distinct health conditions across 5 commercially critical crops from RGB leaf imagery.
    """)

    st.markdown("---")

    # SECTION 1: DATASET & MODEL BENCHMARKS (Explicitly Labeled)
    st.markdown("""
        <div class="section-header">
            <span>🔬</span>
            <h3>Model Architecture & Training Dataset Benchmarks</h3>
        </div>
        <p style="color: #6c757d; font-size: 0.88rem; margin-top: -0.5rem;">
            Fixed evaluation metrics achieved during model training on the 17-class PlantVillage benchmark dataset.
        </p>
    """, unsafe_allow_html=True)

    b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)

    with b_col1:
        st.markdown(f"""
            <div class="card-metric card-benchmark">
                <span class="metric-icon">🌽</span>
                <div class="metric-title">Supported Crops</div>
                <div class="metric-value">{BENCHMARK_STATS['supported_crops_count']}</div>
                <div class="metric-caption">Corn, Grape, Pepper, Potato, Tomato</div>
            </div>
        """, unsafe_allow_html=True)

    with b_col2:
        st.markdown(f"""
            <div class="card-metric card-benchmark">
                <span class="metric-icon">🏷️</span>
                <div class="metric-title">Disease Classes</div>
                <div class="metric-value">{BENCHMARK_STATS['total_classes_count']}</div>
                <div class="metric-caption">12 Pathologies + 5 Healthy States</div>
            </div>
        """, unsafe_allow_html=True)

    with b_col3:
        st.markdown(f"""
            <div class="card-metric card-benchmark">
                <span class="metric-icon">🧠</span>
                <div class="metric-title">Architecture</div>
                <div class="metric-value" style="font-size: 1.45rem;">{BENCHMARK_STATS['model_architecture']}</div>
                <div class="metric-caption">Depthwise Separable CNN + PSO</div>
            </div>
        """, unsafe_allow_html=True)

    with b_col4:
        st.markdown(f"""
            <div class="card-metric card-benchmark">
                <span class="metric-icon">🎯</span>
                <div class="metric-title">Baseline Accuracy</div>
                <div class="metric-value">{BENCHMARK_STATS['baseline_test_accuracy']}%</div>
                <div class="metric-caption">Test Loss: {BENCHMARK_STATS['baseline_test_loss']} (Holdout)</div>
            </div>
        """, unsafe_allow_html=True)

    with b_col5:
        st.markdown(f"""
            <div class="card-metric card-benchmark">
                <span class="metric-icon">📁</span>
                <div class="metric-title">Training Images</div>
                <div class="metric-value">{BENCHMARK_STATS['total_training_images']:,}</div>
                <div class="metric-caption">Augmented Leaf Dataset</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # SECTION 2: LIVE SYSTEM ACTIVITY (SQLite Live Data)
    stats = get_dashboard_statistics()

    st.markdown("""
        <div class="section-header">
            <span>📡</span>
            <h3>Live Detection Activity & Field Statistics</h3>
        </div>
        <p style="color: #6c757d; font-size: 0.88rem; margin-top: -0.5rem;">
            Real-time diagnostics recorded in the local SQLite database from user scans and IoT camera acquisitions.
        </p>
    """, unsafe_allow_html=True)

    l_col1, l_col2, l_col3, l_col4 = st.columns(4)

    with l_col1:
        st.markdown(f"""
            <div class="card-metric card-live">
                <span class="metric-icon">🔍</span>
                <div class="metric-title">Total Live Scans</div>
                <div class="metric-value">{stats['total_scans']}</div>
                <div class="metric-caption">Stored in database</div>
            </div>
        """, unsafe_allow_html=True)

    with l_col2:
        st.markdown(f"""
            <div class="card-metric card-live">
                <span class="metric-icon">🌿</span>
                <div class="metric-title">Healthy Scans</div>
                <div class="metric-value" style="color: #2d6a4f;">{stats['healthy_count']}</div>
                <div class="metric-caption">{stats['healthy_pct']}% of total recorded</div>
            </div>
        """, unsafe_allow_html=True)

    with l_col3:
        st.markdown(f"""
            <div class="card-metric card-live">
                <span class="metric-icon">⚠️</span>
                <div class="metric-title">Diseased Cases</div>
                <div class="metric-value" style="color: #d90429;">{stats['diseased_count']}</div>
                <div class="metric-caption">{stats['diseased_pct']}% pathogen incidence</div>
            </div>
        """, unsafe_allow_html=True)

    with l_col4:
        st.markdown(f"""
            <div class="card-metric card-live">
                <span class="metric-icon">⚡</span>
                <div class="metric-title">Mean Confidence</div>
                <div class="metric-value">{stats['avg_confidence']}%</div>
                <div class="metric-caption">Across historical predictions</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # SECTION 3: VISUAL CHARTS ROW
    c_col1, c_col2 = st.columns([1, 1])

    with c_col1:
        st.markdown("##### 🩺 Foliage Health Distribution")
        donut_fig = create_health_donut_chart(stats["healthy_count"], stats["diseased_count"])
        st.plotly_chart(donut_fig, use_container_width=True)

    with c_col2:
        st.markdown("##### 🌾 Crop-Wise Diagnostic Volume")
        crop_fig = create_crop_distribution_bar(stats["crop_distribution"])
        st.plotly_chart(crop_fig, use_container_width=True)

    # SECTION 4: RECENT DETECTIONS TABLE
    st.markdown("""
        <div class="section-header">
            <span>🕒</span>
            <h3>Recent Foliage Scans</h3>
        </div>
    """, unsafe_allow_html=True)

    recent = stats.get("recent_records", [])
    if recent:
        table_rows = []
        for r in recent:
            status_style = "color: #2d6a4f; font-weight: 700;" if r["health_status"] == "Healthy" else "color: #d90429; font-weight: 700;"
            table_rows.append({
                "ID": f"#{r['id']}",
                "Date & Time": r["timestamp"],
                "Crop": r["crop_name"],
                "Diagnosed Condition": r["disease_name"],
                "Confidence": f"{r['confidence']:.2f}%",
                "Health Status": r["health_status"],
                "Model": r.get("model_used", "MobileNetV2")
            })

        df_recent = pd.DataFrame(table_rows)
        st.dataframe(
            df_recent,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.markdown("""
            <div class="empty-state">
                <div class="icon">🍃</div>
                <h4>No Scans Logged Yet</h4>
                <p>Navigate to <b>Disease Detection</b> in the sidebar to upload a leaf photograph and run your first diagnosis.</p>
            </div>
        """, unsafe_allow_html=True)

    # Quick Action Callouts
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 🚀 Quick Actions")
    qa_col1, qa_col2, qa_col3 = st.columns(3)

    with qa_col1:
        if st.button("📸 Run New Leaf Diagnosis", use_container_width=True, type="primary"):
            st.session_state["nav_selection"] = "Disease Detection"
            st.rerun()

    with qa_col2:
        if st.button("📊 Explore System Analytics", use_container_width=True):
            st.session_state["nav_selection"] = "Analytics"
            st.rerun()

    with qa_col3:
        if st.button("📖 Review Crop Pathologies", use_container_width=True):
            st.session_state["nav_selection"] = "Crop Information"
            st.rerun()
