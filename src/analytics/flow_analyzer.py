"""Flow analytics module for tracking statistics per cluster."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from src.tracking.tracker import Track


@dataclass
class ClusterFlowStats:
    """Flow statistics for a single cluster."""

    cluster_id: int
    cluster_label: str = "Unknown"

    # Counts
    total_count: int = 0  # Total unique tracks seen in this cluster
    active_count: int = 0  # Currently active tracks

    # Flow rates (per minute)
    flow_rate: float = 0.0  # Objects per minute

    # Velocity stats
    mean_speed: float = 0.0

    # Track IDs seen in this cluster
    seen_track_ids: set[int] = field(default_factory=set)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "cluster_id": self.cluster_id,
            "cluster_label": self.cluster_label,
            "total_count": self.total_count,
            "active_count": self.active_count,
            "flow_rate": round(self.flow_rate, 2),
            "mean_speed": round(self.mean_speed, 2),
        }


@dataclass
class OverallFlowStats:
    """Overall flow statistics across all clusters."""

    total_objects_detected: int = 0
    total_objects_tracked: int = 0
    active_tracks: int = 0

    # Per-cluster stats
    cluster_stats: dict[int, ClusterFlowStats] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_objects_detected": self.total_objects_detected,
            "total_objects_tracked": self.total_objects_tracked,
            "active_tracks": self.active_tracks,
            "clusters": {k: v.to_dict() for k, v in self.cluster_stats.items()},
        }


class FlowAnalyzer:
    """Analyzer for computing flow statistics from tracked objects."""

    def __init__(self, flow_window: float = 60.0):
        """Initialize flow analyzer.

        Args:
            flow_window: Time window for flow rate calculation (seconds)
        """
        self.flow_window = flow_window

        self._stats = OverallFlowStats()
        self._start_time = time.time()

        # Track IDs we've seen globally (for total count)
        self._seen_track_ids: set[int] = set()

        # Track arrivals over time for flow rate: list of (timestamp, cluster_id, track_id)
        self._arrivals: list[tuple[float, int, int]] = []

    def update(self, tracks: list["Track"], cluster_names: dict[int, str] | None = None) -> OverallFlowStats:
        """Update statistics with current tracks.

        Args:
            tracks: List of currently active tracks
            cluster_names: Optional mapping of cluster IDs to names

        Returns:
            Updated flow statistics
        """
        current_time = time.time()

        # Update active count
        self._stats.active_tracks = len(tracks)

        # Group tracks by cluster_label (class name) instead of cluster_id
        # This ensures all classes show up even if not clustered
        label_tracks: dict[str, list["Track"]] = defaultdict(list)
        for track in tracks:
            # Use cluster_label which includes class name and anomaly status
            label = track.cluster_label if track.cluster_label and track.cluster_label != "Unknown" else track.yolo_class_name
            if label:  # Skip empty labels
                label_tracks[label].append(track)

            # Track new arrivals
            if track.track_id not in self._seen_track_ids:
                self._seen_track_ids.add(track.track_id)
                self._stats.total_objects_tracked += 1
                # Use label hash as cluster_id for arrivals tracking
                self._arrivals.append((current_time, hash(label) if label else -1, track.track_id))

        # Trim old arrivals outside flow window
        cutoff = current_time - self.flow_window
        self._arrivals = [(t, c, tid) for t, c, tid in self._arrivals if t > cutoff]

        # Reset all active counts to 0 first (so classes with no tracks show 0)
        for stats in self._stats.cluster_stats.values():
            stats.active_count = 0

        # Update per-label stats (using label as key instead of cluster_id)
        for label, track_list in label_tracks.items():
            if not label:
                continue

            # Use hash of label as pseudo cluster_id for storage
            label_key = hash(label) % 10000  # Keep it small for readability
            
            if label_key not in self._stats.cluster_stats:
                self._stats.cluster_stats[label_key] = ClusterFlowStats(
                    cluster_id=label_key,
                    cluster_label=label,
                )

            stats = self._stats.cluster_stats[label_key]
            stats.cluster_label = label  # Always update label

            # Active count
            stats.active_count = len(track_list)

            # Update total count for this label
            for track in track_list:
                if track.track_id not in stats.seen_track_ids:
                    stats.seen_track_ids.add(track.track_id)
                    stats.total_count += 1

            # Calculate flow rate from arrivals with this label
            label_hash = hash(label) if label else -1
            label_arrivals = [(t, tid) for t, c, tid in self._arrivals if c == label_hash]
            if label_arrivals:
                time_span = current_time - label_arrivals[0][0]
                if time_span > 0:
                    stats.flow_rate = (len(label_arrivals) / time_span) * 60
                else:
                    stats.flow_rate = 0.0
            else:
                stats.flow_rate = 0.0

            # Speed stats
            if track_list:
                speeds = [t.speed for t in track_list]
                stats.mean_speed = float(np.mean(speeds)) if speeds else 0.0

        return self._stats

    def get_cluster_stats(self, cluster_id: int) -> ClusterFlowStats | None:
        """Get stats for a specific cluster."""
        return self._stats.cluster_stats.get(cluster_id)

    def get_all_stats(self) -> OverallFlowStats:
        """Get all statistics."""
        return self._stats

    def get_flow_summary(self) -> dict:
        """Get a summary suitable for logging/display."""
        elapsed = time.time() - self._start_time
        summary = {
            "elapsed_time": round(elapsed, 1),
            "active_tracks": self._stats.active_tracks,
            "total_tracked": self._stats.total_objects_tracked,
            "clusters": {},
        }

        for cluster_id, stats in self._stats.cluster_stats.items():
            summary["clusters"][stats.cluster_label] = {
                "active": stats.active_count,
                "total": stats.total_count,
                "flow_rate": f"{stats.flow_rate:.1f}/min",
                "avg_speed": f"{stats.mean_speed:.1f}",
            }

        return summary

    def reset(self):
        """Reset all statistics."""
        self._stats = OverallFlowStats()
        self._start_time = time.time()
        self._seen_track_ids.clear()
        self._arrivals.clear()
        logger.info("Flow analyzer reset")
