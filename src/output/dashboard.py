"""Streamlit dashboard for real-time visualization."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def draw_tracked_frame(
    frame: np.ndarray,
    tracks: list[dict],
    cluster_colors: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Draw bounding boxes and labels on frame.

    Args:
        frame: BGR image
        tracks: List of track dictionaries with bbox, cluster_id, cluster_label, track_id
        cluster_colors: Optional color mapping per cluster

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    default_colors = [
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    for track in tracks:
        bbox = track.get("bbox", (0, 0, 0, 0))
        cluster_id = track.get("cluster_id", -1)
        cluster_label = track.get("cluster_label", "Unknown")
        track_id = track.get("track_id", 0)

        # Get color
        if cluster_colors and cluster_id in cluster_colors:
            color = cluster_colors[cluster_id]
        elif cluster_id >= 0:
            color = default_colors[cluster_id % len(default_colors)]
        else:
            color = (128, 128, 128)  # Gray for unknown

        x1, y1, x2, y2 = bbox

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{cluster_label} #{track_id}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            annotated,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    return annotated


def create_flow_chart(flow_stats: dict) -> go.Figure:
    """Create a flow rate chart from stats."""
    clusters = flow_stats.get("clusters", {})

    if not clusters:
        fig = go.Figure()
        fig.add_annotation(text="No cluster data yet", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    labels = list(clusters.keys())
    active_counts = [c.get("active_count", 0) for c in clusters.values()]
    flow_rates = [c.get("flow_rate", 0) for c in clusters.values()]

    fig = go.Figure(data=[
        go.Bar(name="Active Objects", x=labels, y=active_counts, marker_color="blue"),
        go.Bar(name="Flow Rate (/min)", x=labels, y=flow_rates, marker_color="green"),
    ])

    fig.update_layout(
        title="Cluster Statistics",
        barmode="group",
        xaxis_title="Cluster",
        yaxis_title="Count",
        height=300,
    )

    return fig


def create_cluster_scatter(embeddings_2d: np.ndarray, labels: np.ndarray) -> go.Figure:
    """Create 2D scatter plot of clusters."""
    df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "cluster": [f"Cluster {l}" if l >= 0 else "Noise" for l in labels],
    })

    fig = px.scatter(
        df, x="x", y="y", color="cluster",
        title="Object Clusters (UMAP projection)",
        height=400,
    )

    return fig


def main():
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="SASD - Object Flow Tracker",
        page_icon="🚗",
        layout="wide",
    )

    st.title("🚗 Semantic Object Stream Discovery")
    st.markdown("Real-time unsupervised object clustering and flow tracking")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    video_source = st.sidebar.text_input(
        "Video Source",
        value="",
        placeholder="RTSP URL, YouTube URL, or file path",
    )

    warmup_duration = st.sidebar.slider(
        "Warmup Duration (seconds)",
        min_value=30,
        max_value=300,
        value=60,
    )

    n_clusters = st.sidebar.number_input(
        "Number of Clusters (0 = auto)",
        min_value=0,
        max_value=20,
        value=0,
    )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Feed")
        video_placeholder = st.empty()

        # Status indicator
        status_placeholder = st.empty()

    with col2:
        st.subheader("Flow Statistics")
        stats_placeholder = st.empty()

        st.subheader("Cluster Distribution")
        chart_placeholder = st.empty()

    # Bottom section
    st.subheader("Cluster Visualization")
    scatter_placeholder = st.empty()

    # Cluster labels editing
    st.subheader("Cluster Labels")
    st.markdown("Assign meaningful names to discovered clusters:")

    labels_col1, labels_col2, labels_col3 = st.columns(3)
    with labels_col1:
        cluster_0_label = st.text_input("Cluster 0", value="Sedan", key="c0")
    with labels_col2:
        cluster_1_label = st.text_input("Cluster 1", value="Pickup Truck", key="c1")
    with labels_col3:
        cluster_2_label = st.text_input("Cluster 2", value="Commercial Truck", key="c2")

    # Run button
    if st.sidebar.button("Start Processing", type="primary"):
        if not video_source:
            st.error("Please enter a video source")
            return

        st.sidebar.success(f"Processing: {video_source}")

        # Demo mode - show placeholder
        status_placeholder.info("🔄 Pipeline starting... Warmup phase in progress")

        # In a real implementation, this would connect to the pipeline
        # For now, show demo data
        for i in range(10):
            time.sleep(0.5)
            status_placeholder.info(f"🔄 Warmup: {(i + 1) * 10}% complete ({(i + 1) * 6}s)")

        status_placeholder.success("✅ Warmup complete! Clusters discovered.")

        # Demo stats
        demo_stats = {
            "total_objects_tracked": 156,
            "active_tracks": 12,
            "clusters": {
                cluster_0_label: {"active_count": 5, "flow_rate": 12.3},
                cluster_1_label: {"active_count": 4, "flow_rate": 8.7},
                cluster_2_label: {"active_count": 3, "flow_rate": 4.2},
            },
        }

        stats_placeholder.json(demo_stats)
        chart_placeholder.plotly_chart(create_flow_chart(demo_stats), use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "Built with DINOv2 + HDBSCAN | "
        "[Documentation](https://github.com/yourrepo/sasd)"
    )


if __name__ == "__main__":
    main()
