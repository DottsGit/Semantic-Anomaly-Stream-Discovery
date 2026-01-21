"""Per-class clustering manager for anomaly detection.

This module provides class-aware clustering where objects are grouped by their
YOLO class (e.g., car, truck, bus) and clustered independently within each class.
Anomalies are detected as either HDBSCAN noise points or outliers far from centroids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from src.clustering.clusterer import ObjectClusterer, ClusterResult, ClusterInfo


# Define distinct base colors for common YOLO/COCO classes (BGR format)
# Made more visually distinct from each other
CLASS_BASE_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (0, 140, 255),         # Bright Orange
    "truck": (0, 180, 0),          # Green
    "bus": (255, 100, 0),          # Blue
    "motorcycle": (255, 0, 200),   # Pink-Magenta
    "bicycle": (255, 200, 0),      # Cyan
    "person": (0, 255, 0),         # Bright Green (distinct from car)
    "boat": (180, 0, 180),         # Purple
    "train": (100, 100, 255),      # Light Red
    "airplane": (255, 150, 150),   # Light Blue
}

# Default color for unknown classes
DEFAULT_BASE_COLOR = (150, 150, 150)  # Light Grey


def get_class_color(class_name: str, is_anomaly: bool = False) -> tuple[int, int, int]:
    """Get display color for a class, with darker/shifted variant for anomalies.
    
    Args:
        class_name: The YOLO class name
        is_anomaly: If True, return anomaly variant color
        
    Returns:
        BGR color tuple
    """
    base = CLASS_BASE_COLORS.get(class_name.lower(), DEFAULT_BASE_COLOR)
    
    if is_anomaly:
        # Shift anomaly color: darken and shift toward red
        b, g, r = base
        return (
            max(0, b - 60),      # Less blue
            max(0, g - 80),      # Less green
            min(255, r + 100),   # More red
        )
    return base


@dataclass
class PerClassClusterResult:
    """Combined clustering result for all classes."""
    
    # Results per class: class_name -> ClusterResult
    class_results: dict[str, ClusterResult] = field(default_factory=dict)
    
    # Classes that were skipped due to insufficient samples
    skipped_classes: dict[str, int] = field(default_factory=dict)
    
    # Total stats
    total_clusters: int = 0
    total_samples: int = 0
    
    @property
    def clustered_classes(self) -> list[str]:
        """Get list of classes that were successfully clustered."""
        return list(self.class_results.keys())


@dataclass 
class ClassPrediction:
    """Prediction result for a single embedding."""
    
    class_name: str
    cluster_id: int
    is_anomaly: bool
    cluster_label: str  # e.g., "car" or "car (anomaly)"
    color: tuple[int, int, int]


class ClassAwareClusterer:
    """Manages per-class clustering for anomaly detection.
    
    Instead of clustering all objects together, this clusters objects
    within each YOLO class separately, enabling intra-class anomaly detection.
    """
    
    def __init__(
        self,
        min_samples_percentage: float = 0.05,  # 5% of total samples
        min_samples_absolute: int = 15,  # Minimum floor
        algorithm: str = "hdbscan",
        min_cluster_size: int = 10,
        use_pca: bool = True,
        pca_n_components: int = 32,
        cluster_scale: float = 0.025,
        anomaly_threshold_percentile: float = 95.0,
    ):
        """Initialize the class-aware clusterer.
        
        Args:
            min_samples_percentage: Minimum samples as % of total (default 5%)
            min_samples_absolute: Absolute minimum floor (default 15)
            algorithm: Clustering algorithm (hdbscan, kmeans, dbscan)
            min_cluster_size: Minimum cluster size for HDBSCAN
            use_pca: Whether to use PCA dimensionality reduction
            pca_n_components: Target PCA dimensions
            cluster_scale: Dynamic cluster size scale factor  
            anomaly_threshold_percentile: Percentile for distance-based anomaly detection
        """
        self.min_samples_percentage = min_samples_percentage
        self.min_samples_absolute = min_samples_absolute
        self.algorithm = algorithm
        self.min_cluster_size = min_cluster_size
        self.use_pca = use_pca
        self.pca_n_components = pca_n_components
        self.cluster_scale = cluster_scale
        self.anomaly_threshold_percentile = anomaly_threshold_percentile
        
        # Per-class clusterers
        self._clusterers: dict[str, ObjectClusterer] = {}
        
        # Per-class anomaly thresholds (distance from centroid)
        self._anomaly_thresholds: dict[str, float] = {}
        
        # Last combined result
        self._last_result: PerClassClusterResult | None = None
        self._fitted = False
    
    def _calculate_min_samples(self, total_samples: int) -> int:
        """Calculate minimum samples threshold as percentage of total."""
        pct_min = int(total_samples * self.min_samples_percentage)
        return max(pct_min, self.min_samples_absolute)
    
    def _create_clusterer(self, class_name: str, min_samples: int) -> ObjectClusterer:
        """Create a new clusterer for a class."""
        return ObjectClusterer(
            algorithm=self.algorithm,
            min_cluster_size=self.min_cluster_size,
            use_pca=self.use_pca,
            pca_n_components=self.pca_n_components,
            min_samples=min_samples,
            cluster_scale=self.cluster_scale,
        )
    
    def _compute_anomaly_threshold(
        self, 
        embeddings: np.ndarray, 
        cluster_result: ClusterResult
    ) -> float:
        """Compute distance threshold for anomaly detection.
        
        Points further than this distance from their nearest centroid
        are considered anomalies.
        """
        if not cluster_result.clusters:
            return float('inf')
        
        # Get all centroids
        centroids = np.vstack([c.centroid for c in cluster_result.clusters.values()])
        
        # Compute distance of each point to its assigned centroid
        from scipy.spatial.distance import cdist
        distances = cdist(embeddings, centroids, metric="euclidean")
        min_distances = np.min(distances, axis=1)
        
        # Use percentile as threshold
        threshold = np.percentile(min_distances, self.anomaly_threshold_percentile)
        return float(threshold)
    
    def fit(self, embeddings_by_class: dict[str, np.ndarray]) -> PerClassClusterResult:
        """Fit clusterers on embeddings grouped by class.
        
        Args:
            embeddings_by_class: Dict mapping class_name to embeddings array
            
        Returns:
            PerClassClusterResult with per-class clustering results
        """
        result = PerClassClusterResult()
        
        # Calculate total samples to determine dynamic threshold
        total_samples = sum(len(e) for e in embeddings_by_class.values())
        min_threshold = self._calculate_min_samples(total_samples)
        
        logger.info(f"Minimum samples threshold: {min_threshold} ({self.min_samples_percentage*100:.0f}% of {total_samples})")
        
        for class_name, embeddings in embeddings_by_class.items():
            n_samples = len(embeddings)
            
            if n_samples < min_threshold:
                logger.info(
                    f"Skipping class '{class_name}': {n_samples} samples "
                    f"< {min_threshold} minimum"
                )
                result.skipped_classes[class_name] = n_samples
                continue
            
            logger.info(f"Clustering class '{class_name}' ({n_samples} samples)...")
            
            # Create and fit clusterer for this class
            clusterer = self._create_clusterer(class_name, min_threshold)
            class_result = clusterer.fit(embeddings)
            
            # Update cluster labels to include class name
            for cluster_id, info in class_result.clusters.items():
                info.label = class_name
                info.color = get_class_color(class_name, is_anomaly=False)
            
            # Store clusterer and result
            self._clusterers[class_name] = clusterer
            result.class_results[class_name] = class_result
            
            # Compute anomaly threshold for this class
            self._anomaly_thresholds[class_name] = self._compute_anomaly_threshold(
                embeddings, class_result
            )
            
            result.total_clusters += class_result.n_clusters
            result.total_samples += n_samples
            
            logger.info(
                f"  '{class_name}': {class_result.n_clusters} clusters, "
                f"{class_result.noise_count} noise points"
            )
        
        self._last_result = result
        self._fitted = True
        
        logger.info(
            f"Per-class clustering complete: {result.total_clusters} total clusters "
            f"across {len(result.class_results)} classes"
        )
        
        return result
    
    def predict(
        self, 
        class_name: str, 
        embedding: np.ndarray
    ) -> ClassPrediction:
        """Predict cluster and anomaly status for a single embedding.
        
        Args:
            class_name: The YOLO class of the object
            embedding: The DINO embedding vector
            
        Returns:
            ClassPrediction with cluster_id, is_anomaly, label, and color
        """
        # Ensure embedding is 2D
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Check if we have a clusterer for this class
        if class_name not in self._clusterers:
            # Class wasn't clustered (too few samples or unseen)
            return ClassPrediction(
                class_name=class_name,
                cluster_id=-1,
                is_anomaly=False,  # Can't determine anomaly without clustering
                cluster_label=class_name,
                color=get_class_color(class_name, is_anomaly=False),
            )
        
        clusterer = self._clusterers[class_name]
        cluster_ids = clusterer.predict(embedding)
        cluster_id = int(cluster_ids[0])
        
        # Check if anomaly
        is_anomaly = False
        
        # Method 1: HDBSCAN noise points (cluster_id == -1)
        if cluster_id == -1:
            is_anomaly = True
        else:
            # Method 2: Distance-based anomaly detection
            if class_name in self._anomaly_thresholds:
                threshold = self._anomaly_thresholds[class_name]
                
                # Get distance to assigned centroid
                result = clusterer.result
                if result and cluster_id in result.clusters:
                    centroid = result.clusters[cluster_id].centroid
                    distance = np.linalg.norm(embedding[0] - centroid)
                    is_anomaly = distance > threshold
        
        # Build label
        cluster_label = f"{class_name} (anomaly)" if is_anomaly else class_name
        
        return ClassPrediction(
            class_name=class_name,
            cluster_id=cluster_id,
            is_anomaly=is_anomaly,
            cluster_label=cluster_label,
            color=get_class_color(class_name, is_anomaly=is_anomaly),
        )
    
    def predict_batch(
        self,
        class_names: list[str],
        embeddings: np.ndarray,
    ) -> list[ClassPrediction]:
        """Predict clusters and anomaly status for multiple embeddings.
        
        Args:
            class_names: List of YOLO class names, one per embedding
            embeddings: Array of embeddings (n_samples, embedding_dim)
            
        Returns:
            List of ClassPrediction objects
        """
        predictions = []
        for i, (class_name, embedding) in enumerate(zip(class_names, embeddings)):
            pred = self.predict(class_name, embedding)
            predictions.append(pred)
        return predictions
    
    def get_clusterer(self, class_name: str) -> ObjectClusterer | None:
        """Get the clusterer for a specific class."""
        return self._clusterers.get(class_name)
    
    def get_class_result(self, class_name: str) -> ClusterResult | None:
        """Get clustering result for a specific class."""
        if self._last_result and class_name in self._last_result.class_results:
            return self._last_result.class_results[class_name]
        return None
    
    @property
    def is_fitted(self) -> bool:
        """Check if clusterer has been fitted."""
        return self._fitted
    
    @property
    def result(self) -> PerClassClusterResult | None:
        """Get the last clustering result."""
        return self._last_result
    
    @property
    def clustered_classes(self) -> list[str]:
        """Get list of classes that have been clustered."""
        return list(self._clusterers.keys())
