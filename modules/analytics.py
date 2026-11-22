"""
CropCare AI - Analytics Visualization Module
Generates professional Plotly charts with an agricultural color palette
from SQLite historical detection data and model benchmark statistics.
"""

from typing import Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from config import BENCHMARK_STATS


# Agricultural Design Palette
COLOR_HEALTHY = "#2d6a4f"       # Forest green
COLOR_DISEASED = "#d90429"      # Crimson red / amber accent
COLOR_ACCENT = "#52b788"        # Mint leaf green
COLOR_SECONDARY = "#40916c"     # Deep foliage
COLOR_LIGHT_BG = "#f8fbf9"      # Clean off-white
CROP_COLORS = ["#2d6a4f", "#40916c", "#52b788", "#74c69d", "#95d5b2"]


def create_health_donut_chart(healthy_count: int, diseased_count: int) -> go.Figure:
    """
    Creates a modern donut chart showing healthy vs diseased proportion.
    """
    total = healthy_count + diseased_count
    if total == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No scan records logged yet.<br>Perform a leaf scan to view distribution.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#6c757d")
        )
        fig.update_layout(height=320, margin=dict(t=30, b=30, l=30, r=30))
        return fig

    labels = ["Healthy Foliage", "Diseased Foliage"]
    values = [healthy_count, diseased_count]
    colors = [COLOR_HEALTHY, COLOR_DISEASED]

    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
            textinfo="percent+value",
            hoverinfo="label+value+percent",
            textfont=dict(size=14, family="sans-serif"),
            insidetextorientation="horizontal"
        )
    ])

    fig.update_layout(
        title=dict(text="<b>Foliage Health Status Ratio</b>", font=dict(size=16, color="#1b4332")),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        height=340,
        margin=dict(t=50, b=40, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_crop_distribution_bar(crop_distribution: Dict[str, int]) -> go.Figure:
    """
    Creates a bar chart displaying total scans per crop.
    """
    if not crop_distribution:
        fig = go.Figure()
        fig.add_annotation(
            text="No crop data logged yet.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#6c757d")
        )
        fig.update_layout(height=320, margin=dict(t=30, b=30, l=30, r=30))
        return fig

    crops = list(crop_distribution.keys())
    counts = list(crop_distribution.values())

    fig = go.Figure(data=[
        go.Bar(
            x=crops,
            y=counts,
            text=counts,
            textposition="auto",
            marker=dict(
                color=CROP_COLORS[:len(crops)],
                line=dict(color="#1b4332", width=1)
            )
        )
    ])

    fig.update_layout(
        title=dict(text="<b>Crop-Wise Scan Frequency</b>", font=dict(size=16, color="#1b4332")),
        xaxis=dict(title="Crop Type", tickangle=-15),
        yaxis=dict(title="Number of Scans", dtick=1),
        height=340,
        margin=dict(t=50, b=50, l=40, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_disease_frequency_bar(disease_distribution: Dict[str, int]) -> go.Figure:
    """
    Creates a horizontal bar chart displaying diagnosed diseases.
    """
    if not disease_distribution:
        fig = go.Figure()
        fig.add_annotation(
            text="No disease records found in history.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#6c757d")
        )
        fig.update_layout(height=320, margin=dict(t=30, b=30, l=30, r=30))
        return fig

    diseases = list(disease_distribution.keys())
    counts = list(disease_distribution.values())

    fig = go.Figure(data=[
        go.Bar(
            y=diseases,
            x=counts,
            orientation="h",
            text=counts,
            textposition="auto",
            marker=dict(
                color="#e76f51",
                line=dict(color="#264653", width=1)
            )
        )
    ])

    fig.update_layout(
        title=dict(text="<b>Diagnosed Disease Breakdown</b>", font=dict(size=16, color="#1b4332")),
        xaxis=dict(title="Detections Count"),
        yaxis=dict(title="Disease Type", autorange="reversed"),
        height=360,
        margin=dict(t=50, b=40, l=150, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_confidence_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Creates a histogram showing model prediction confidence distribution.
    """
    if df.empty or "confidence" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No confidence records logged yet.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#6c757d")
        )
        fig.update_layout(height=320, margin=dict(t=30, b=30, l=30, r=30))
        return fig

    fig = go.Figure(data=[
        go.Histogram(
            x=df["confidence"],
            nbinsx=15,
            marker=dict(
                color="#40916c",
                line=dict(color="#1b4332", width=1)
            ),
            opacity=0.85
        )
    ])

    fig.update_layout(
        title=dict(text="<b>Model Prediction Confidence Spread</b>", font=dict(size=16, color="#1b4332")),
        xaxis=dict(title="Confidence Score (%)", range=[50, 102]),
        yaxis=dict(title="Frequency"),
        height=320,
        margin=dict(t=50, b=40, l=40, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_daily_trend_chart(df: pd.DataFrame) -> go.Figure:
    """
    Creates a trend area chart showing scan volume over time.
    """
    if df.empty or "timestamp" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No temporal detection trends yet.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#6c757d")
        )
        fig.update_layout(height=320, margin=dict(t=30, b=30, l=30, r=30))
        return fig

    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["timestamp"]).dt.date
    daily_counts = df_copy.groupby("date").size().reset_index(name="scans")

    fig = go.Figure(data=[
        go.Scatter(
            x=daily_counts["date"],
            y=daily_counts["scans"],
            mode="lines+markers",
            fill="tozeroy",
            line=dict(color="#2d6a4f", width=3),
            marker=dict(size=7, color="#1b4332"),
            fillcolor="rgba(82, 183, 136, 0.25)"
        )
    ])

    fig.update_layout(
        title=dict(text="<b>Daily Scan Activity Timeline</b>", font=dict(size=16, color="#1b4332")),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Scans Recorded"),
        height=320,
        margin=dict(t=50, b=40, l=40, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_benchmark_comparison_chart() -> go.Figure:
    """
    Creates a grouped bar comparison between MobileNetV2 Baseline and PSO Optimization.
    """
    metrics = ["Test Accuracy (%)", "Macro Precision (%)", "Macro Recall (%)", "Macro F1 (%)"]
    baseline_vals = [
        BENCHMARK_STATS["baseline_test_accuracy"],
        BENCHMARK_STATS["baseline_macro_precision"],
        BENCHMARK_STATS["baseline_macro_recall"],
        BENCHMARK_STATS["baseline_macro_f1"]
    ]
    pso_vals = [
        BENCHMARK_STATS["pso_test_accuracy"],
        BENCHMARK_STATS["pso_macro_precision"],
        BENCHMARK_STATS["pso_macro_recall"],
        BENCHMARK_STATS["pso_macro_f1"]
    ]

    fig = go.Figure(data=[
        go.Bar(
            name="Baseline MobileNetV2",
            x=metrics,
            y=baseline_vals,
            text=[f"{v:.2f}%" for v in baseline_vals],
            textposition="auto",
            marker=dict(color="#2d6a4f")
        ),
        go.Bar(
            name="PSO-Optimized MobileNetV2",
            x=metrics,
            y=pso_vals,
            text=[f"{v:.2f}%" for v in pso_vals],
            textposition="auto",
            marker=dict(color="#52b788")
        )
    ])

    fig.update_layout(
        barmode="group",
        title=dict(text="<b>Model Benchmark Comparison (Test Dataset Evaluation)</b>", font=dict(size=16, color="#1b4332")),
        yaxis=dict(title="Score (%)", range=[90, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=360,
        margin=dict(t=70, b=40, l=40, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig
