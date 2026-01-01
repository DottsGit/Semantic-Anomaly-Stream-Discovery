"""Video ingestion module."""

from src.ingestion.video_source import (
    BufferedVideoSource,
    Frame,
    OpenCVSource,
    SourceType,
    VideoSource,
    YouTubeSource,
    create_video_source,
    detect_source_type,
)

__all__ = [
    "BufferedVideoSource",
    "Frame",
    "OpenCVSource",
    "SourceType",
    "VideoSource",
    "YouTubeSource",
    "create_video_source",
    "detect_source_type",
]
