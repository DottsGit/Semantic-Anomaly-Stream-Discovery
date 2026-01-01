"""Object detection module."""

from src.detection.detector import (
    COCO_ANIMAL_CLASSES,
    COCO_PERSON_CLASS,
    COCO_VEHICLE_CLASSES,
    Detection,
    ObjectDetector,
)

__all__ = [
    "COCO_ANIMAL_CLASSES",
    "COCO_PERSON_CLASS",
    "COCO_VEHICLE_CLASSES",
    "Detection",
    "ObjectDetector",
]
