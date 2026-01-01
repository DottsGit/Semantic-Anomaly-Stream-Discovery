"""Configuration settings for the SOSD pipeline."""

from pydantic import Field
from pydantic_settings import BaseSettings


class VideoSettings(BaseSettings):
    """Video ingestion settings."""

    source: str = Field(default="", description="Video source URL or file path")
    source_type: str = Field(
        default="auto", description="Source type: auto, rtsp, hls, youtube, file"
    )
    fps_limit: int = Field(default=10, description="Max FPS to process (reduces load)")
    resolution: tuple[int, int] = Field(
        default=(640, 480), description="Processing resolution (width, height)"
    )
    buffer_size: int = Field(default=30, description="Frame buffer size")


class DetectionSettings(BaseSettings):
    """Object detection settings."""

    model: str = Field(default="yolov8n.pt", description="YOLO model for base detection")
    confidence_threshold: float = Field(default=0.5, description="Detection confidence threshold")
    nms_threshold: float = Field(default=0.4, description="Non-max suppression threshold")
    classes: list[int] | None = Field(
        default=None, description="Filter to specific COCO classes (None = all)"
    )
    min_box_area: int = Field(default=1000, description="Minimum bounding box area in pixels")


class FeatureSettings(BaseSettings):
    """DINOv2 feature extraction settings."""

    model: str = Field(default="facebook/dinov2-base", description="DINOv2 model variant")
    device: str = Field(default="cuda", description="Device: cuda, cpu, mps")
    batch_size: int = Field(default=16, description="Batch size for feature extraction")
    embedding_dim: int = Field(default=768, description="DINOv2-base embedding dimension")


class ClusteringSettings(BaseSettings):
    """Unsupervised clustering settings."""

    warmup_duration: int = Field(default=60, description="Warmup duration in seconds")
    min_samples: int = Field(default=50, description="Min samples before clustering")
    algorithm: str = Field(default="hdbscan", description="Clustering: kmeans, hdbscan, dbscan")
    n_clusters: int | None = Field(
        default=None, description="Number of clusters (None = auto for HDBSCAN)"
    )
    min_cluster_size: int = Field(default=10, description="HDBSCAN min cluster size")
    recluster_interval: int = Field(default=300, description="Recluster every N seconds")
    use_umap: bool = Field(default=True, description="Use UMAP for dimensionality reduction")
    umap_n_components: int = Field(default=32, description="UMAP target dimensions")


class TrackingSettings(BaseSettings):
    """Object tracking settings."""

    max_age: int = Field(default=30, description="Max frames to keep lost track")
    min_hits: int = Field(default=3, description="Min hits before track is confirmed")
    iou_threshold: float = Field(default=0.3, description="IOU threshold for matching")


class OutputSettings(BaseSettings):
    """Output and analytics settings."""

    enable_dashboard: bool = Field(default=True, description="Enable Streamlit dashboard")
    dashboard_port: int = Field(default=8501, description="Dashboard port")

    enable_pubsub: bool = Field(default=False, description="Enable Pub/Sub output")
    pubsub_project: str = Field(default="", description="GCP project ID")
    pubsub_topic: str = Field(default="sosd-events", description="Pub/Sub topic name")

    enable_bigquery: bool = Field(default=False, description="Enable BigQuery output")
    bigquery_dataset: str = Field(default="sosd", description="BigQuery dataset")
    bigquery_table: str = Field(default="tracking_events", description="BigQuery table")

    log_interval: int = Field(default=5, description="Log stats every N seconds")


class Settings(BaseSettings):
    """Main settings container."""

    video: VideoSettings = Field(default_factory=VideoSettings)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    clustering: ClusteringSettings = Field(default_factory=ClusteringSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)

    class Config:
        env_prefix = "SOSD_"
        env_nested_delimiter = "__"


settings = Settings()
