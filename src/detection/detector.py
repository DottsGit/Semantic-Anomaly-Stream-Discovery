"""Object detection module using YOLOv8 for base detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger

from src.ingestion.video_source import Frame


@dataclass
class Detection:
    """A detected object with bounding box and metadata."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    frame_number: int
    timestamp: float

    @property
    def center(self) -> tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        """Get area of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def crop_from(self, image: np.ndarray, padding: int = 0) -> np.ndarray:
        """Crop this detection from an image with optional padding."""
        x1, y1, x2, y2 = self.bbox
        h, w = image.shape[:2]

        # Add padding with bounds checking
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        return image[y1:y2, x1:x2]


class ObjectDetector:
    """YOLOv8-based object detector."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        classes: list[int] | None = None,
        min_box_area: int = 1000,
        device: str = "cuda",
    ):
        """Initialize the detector.

        Args:
            model_name: YOLO model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            confidence_threshold: Minimum detection confidence
            nms_threshold: Non-max suppression IoU threshold
            classes: List of COCO class IDs to detect (None = all)
            min_box_area: Minimum bounding box area in pixels
            device: Device to run inference on
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.classes = classes
        self.min_box_area = min_box_area
        self.device = device if torch.cuda.is_available() else "cpu"

        self._model = None
        self._class_names: dict[int, str] = {}

    def load(self) -> None:
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model: {self.model_name}")
            self._model = YOLO(self.model_name)
            self._model.to(self.device)

            # Get class names
            self._class_names = self._model.names
            logger.info(f"Loaded model with {len(self._class_names)} classes on {self.device}")

        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise

    def detect(self, frame: Frame) -> list[Detection]:
        """Run detection on a frame.

        Args:
            frame: Input video frame

        Returns:
            List of Detection objects
        """
        if self._model is None:
            self.load()

        results = self._model(
            frame.image,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            classes=self.classes,
            verbose=False,
        )[0]

        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            # Skip small detections
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_box_area:
                continue

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=self._class_names.get(cls_id, "unknown"),
                    frame_number=frame.frame_number,
                    timestamp=frame.timestamp,
                )
            )

        return detections

    def detect_batch(self, frames: list[Frame]) -> list[list[Detection]]:
        """Run detection on a batch of frames.

        Args:
            frames: List of input frames

        Returns:
            List of detection lists, one per frame
        """
        if self._model is None:
            self.load()

        images = [f.image for f in frames]
        results = self._model(
            images,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            classes=self.classes,
            verbose=False,
        )

        all_detections = []
        for frame, result in zip(frames, results):
            frame_detections = []
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                area = (x2 - x1) * (y2 - y1)
                if area < self.min_box_area:
                    continue

                frame_detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=self._class_names.get(cls_id, "unknown"),
                        frame_number=frame.frame_number,
                        timestamp=frame.timestamp,
                    )
                )
            all_detections.append(frame_detections)

        return all_detections


# Common COCO class IDs for reference
COCO_VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
COCO_PERSON_CLASS = [0]
COCO_ANIMAL_CLASSES = list(range(14, 24))  # bird through teddy bear
