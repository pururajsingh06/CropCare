"""
CropCare AI - Visual Analytics Page
Renders interactive Plotly visual analytics from the SQLite detection database,
including health status donuts, disease distributions, and model benchmark comparisons.
"""

import streamlit as st
import pandas as pd
from modules.database import get_dashboard_statistics, get_detections_dataframe
from modules.analytics import (
    create_health_donut_chart,
    create_crop_distribution_bar,
    create_disease_frequency_bar,
    create_confidence_distribution_chart,
    create_daily_trend_chart,
    create_benchmark_comparison_chart
)
from config import BENCHMARK_STATS


def render_analytics_page():
    """Renders the Analytics dashboard interface."""

    st.markdown("""
        <div class="section-header">
            <span>📊</span>
            <h3>Diagnostic Analytics & Model Evaluation</h3>
        </div>
        <p style="color: #6c757d; font-size: 0.9rem; margin-top: -0.5rem;">
            Interactive visual insights compiled from live field diagnosis records and benchmark deep learning performance metrics.
        </p>
    """, unsafe_allow_html=True)

    # Top-level tabs: Live Field Analytics vs Model Benchmarks
    tab_field, tab_benchmark = st.tabs([
        "📈 Live Field Detection Analytics",
        "🎯 AI Model Benchmark & PSO Comparison"
    ])

    # TAB 1: LIVE FIELD ANALYTICS (From SQLite)
    with tab_field:
        stats = get_dashboard_statistics()
        df = get_detections_dataframe()

        if stats["total_scans"] == 0:
            st.markdown("""
                <div class="empty-state">
                    <div class="icon">📊</div>
                    <h4>No Live Detection History Recorded</h4>
                    <p>Visual analytics require at least one recorded leaf diagnosis. Navigate to <b>Disease Detection</b> to run your first scan and generate real-time metrics.</p>
                </div>
            """, unsafe_allow_html=True)
            return

        # KPI Summary Metric Cards
        st.markdown("##### 📌 Operational KPI Summary")
        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

        with kpi_c1:
            st.markdown(f"""
                <div class="card-metric card-live">
                    <div class="metric-title">Total Logged Scans</div>
                    <div class="metric-value">{stats['total_scans']}</div>
                    <div class="metric-caption">Unique field records</div>
                </div>
            """, unsafe_allow_html=True)

        with kpi_c2:
            st.markdown(f"""
                <div class="card-metric card-live">
                    <div class="metric-title">Healthy Ratio</div>
                    <div class="metric-value" style="color: #2d6a4f;">{stats['healthy_pct']}%</div>
                    <div class="metric-caption">{stats['healthy_count']} healthy leaves</div>
                </div>
            """, unsafe_allow_html=True)

        with kpi_c3:
            st.markdown(f"""
                <div class="card-metric card-live">
                    <div class="metric-title">Pathology Incidence</div>
                    <div class="metric-value" style="color: #d90429;">{stats['diseased_pct']}%</div>
                    <div class="metric-caption">{stats['diseased_count']} diseased leaves</div>
                </div>
            """, unsafe_allow_html=True)

        with kpi_c4:
            st.markdown(f"""
                <div class="card-metric card-live">
                    <div class="metric-title">Average Confidence</div>
                    <div class="metric-value">{stats['avg_confidence']}%</div>
                    <div class="metric-caption">Across all diagnoses</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual Analytics Charts Row 1
        st.markdown("##### 🩺 Foliage Health and Crop Volume")
        r1_c1, r1_c2 = st.columns([1, 1])

        with r1_c1:
            donut_fig = create_health_donut_chart(stats["healthy_count"], stats["diseased_count"])
            st.plotly_chart(donut_fig, use_container_width=True)

        with r1_c2:
            crop_fig = create_crop_distribution_bar(stats["crop_distribution"])
            st.plotly_chart(crop_fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual Analytics Charts Row 2
        st.markdown("##### 🦠 Pathology Breakdown and Prediction Confidence")
        r2_c1, r2_c2 = st.columns([1, 1])

        with r2_c1:
            disease_fig = create_disease_frequency_bar(stats["disease_distribution"])
            st.plotly_chart(disease_fig, use_container_width=True)

        with r2_c2:
            conf_fig = create_confidence_distribution_chart(df)
            st.plotly_chart(conf_fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual Analytics Charts Row 3: Daily Timeline
        st.markdown("##### 📅 Temporal Diagnostic Activity")
        trend_fig = create_daily_trend_chart(df)
        st.plotly_chart(trend_fig, use_container_width=True)

    # TAB 2: AI MODEL BENCHMARK & PSO COMPARISON
    with tab_benchmark:
        st.markdown("##### 🔬 MobileNetV2 Deep Learning & PSO Optimization Benchmark")
        st.caption("Quantitative evaluation metrics measured on the independent 17-class holdout test set.")

        bm_col1, bm_col2, bm_col3, bm_col4 = st.columns(4)
        with bm_col1:
            st.metric("Baseline Test Accuracy", f"{BENCHMARK_STATS['baseline_test_accuracy']}%", "Baseline")
        with bm_col2:
            st.metric("PSO Optimized Accuracy", f"{BENCHMARK_STATS['pso_test_accuracy']}%", "-0.24% diff")
        with bm_col3:
            st.metric("Baseline Macro F1", f"{BENCHMARK_STATS['baseline_macro_f1']}%")
        with bm_col4:
            st.metric("Training Dataset Size", f"{BENCHMARK_STATS['total_training_images']:,} images")

        st.markdown("<br>", unsafe_allow_html=True)

        # Plotly comparison chart
        st.plotly_chart(create_benchmark_comparison_chart(), use_container_width=True)

        # Comparison Data Table
        st.markdown("##### 📋 Quantitative Metric Comparison Matrix")
        comparison_table = pd.DataFrame([
            {
                "Model Variant": "MobileNetV2 Baseline",
                "Test Accuracy": f"{BENCHMARK_STATS['baseline_test_accuracy']}%",
                "Test Loss": f"{BENCHMARK_STATS['baseline_test_loss']}",
                "Macro Precision": f"{BENCHMARK_STATS['baseline_macro_precision']}%",
                "Macro Recall": f"{BENCHMARK_STATS['baseline_macro_recall']}%",
                "Macro F1-Score": f"{BENCHMARK_STATS['baseline_macro_f1']}%"
            },
            {
                "Model Variant": "MobileNetV2 + PSO Optimization",
                "Test Accuracy": f"{BENCHMARK_STATS['pso_test_accuracy']}%",
                "Test Loss": f"{BENCHMARK_STATS['pso_test_loss']}",
                "Macro Precision": f"{BENCHMARK_STATS['pso_macro_precision']}%",
                "Macro Recall": f"{BENCHMARK_STATS['pso_macro_recall']}%",
                "Macro F1-Score": f"{BENCHMARK_STATS['pso_macro_f1']}%"
            }
        ])
        st.dataframe(comparison_table, use_container_width=True, hide_index=True)

        st.markdown("""
            > **Important Project Note:** The 96.97% test accuracy and 96.73% PSO accuracy represent offline
            > evaluation benchmarks on the 27,494 image dataset. In real-world field applications, factors such
            > as ambient lighting, leaf angle, camera focus, and background clutter may influence live confidence scores.
        """)
