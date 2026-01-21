"""DINOv3 feature extraction module for unsupervised object representation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from loguru import logger


from src.detection.detector import Detection


# Define stats for normalization (DINOv3 / ImageNet stats)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)



@dataclass
class ObjectFeature:
    """Feature vector for a detected object."""

    detection: Detection
    embedding: np.ndarray  # DINOv3 embedding vector
    crop_image: np.ndarray | None = None  # Optional: store the cropped image

    @property
    def vector(self) -> np.ndarray:
        """Alias for embedding."""
        return self.embedding


class DINOv3Extractor:
    """Feature extractor using DINOv3 vision transformer.

    Uses timm (Ross Wightman's library) for the DINOv3 model.
    """

    def __init__(
        self,
        model_name: str = "timm/vit_base_patch16_dinov3.lvd1689m",
        device: str = "cuda",
        batch_size: int = 16,
        image_size: int = 224,
    ):
        """Initialize DINOv3 feature extractor.

        Args:
            model_name: HuggingFace/timm model name for DINOv3
                - timm/vit_small_patch16_dinov3.lvd1689m
                - timm/vit_base_patch16_dinov3.lvd1689m
                - timm/vit_large_patch16_dinov3.lvd1689m
                - timm/vit_giant_patch14_dinov3.lvd1689m
            device: Device to run on (cuda, cpu, mps)
            batch_size: Batch size for feature extraction
            image_size: Input image size (DINOv3 usually 224 or 518)
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.image_size = image_size

        self._model = None
        self._embedding_dim: int | None = None

    def load(self) -> None:
        """Load the DINOv3 model."""
        try:
            import timm
            
            logger.info(f"Loading DINOv3 model (timm): {self.model_name}")

            # Create model with num_classes=0 to get pooling/embedding support
            self._model = timm.create_model(
                self.model_name, 
                pretrained=True, 
                num_classes=0, 
            )
            self._model.to(self.device)
            self._model.eval()

            # Get embedding dimension
            self._embedding_dim = self._model.num_features
            logger.info(
                f"Loaded DINOv3 with {self._embedding_dim}-dim embeddings on {self.device}"
            )

        except ImportError:
            logger.error("timm not installed. Run: pip install timm")
            raise

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            self.load()
        return self._embedding_dim or 768

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess a crop for DINOv3.

        Args:
            crop: BGR image crop from OpenCV

        Returns:
            RGB image resized for model input
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        # DINOv3/timm usually expects bicubic interpolation for best results
        resized = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        
        # Normalize (H, W, C) -> (C, H, W)
        img = resized.astype(np.float32) / 255.0
        
        # Standardize using ImageNet stats
        # (x - mean) / std
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        
        # Transpose to (C, H, W)
        img = img.transpose(2, 0, 1)

        return img.astype(np.float32)

    def extract(
        self,
        image: np.ndarray,
        detections: list[Detection],
        store_crops: bool = False,
    ) -> list[ObjectFeature]:
        """Extract DINOv3 features for detected objects.

        Args:
            image: Full frame image (BGR)
            detections: List of detections to extract features for
            store_crops: Whether to store cropped images in results

        Returns:
            List of ObjectFeature with embeddings
        """
        if self._model is None:
            self.load()

        if not detections:
            return []

        # Crop and preprocess all detections
        crops = []
        valid_detections = []

        for det in detections:
            try:
                crop = det.crop_from(image, padding=5)
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue
                crops.append(self._preprocess_crop(crop))
                valid_detections.append((det, crop if store_crops else None))
            except Exception as e:
                logger.warning(f"Failed to crop detection: {e}")
                continue

        if not crops:
            return []

        # Process in batches
        all_embeddings = []

        for i in range(0, len(crops), self.batch_size):
            batch_crops = crops[i : i + self.batch_size]
            
            # Stack into tensor
            batch_tensor = np.stack(batch_crops)
            batch_tensor = torch.from_numpy(batch_tensor).to(self.device)
            
            if self.device == "cuda":
                # Use float16/bfloat16 for inference speed if available
                # But careful with DINOv3 RoPE issues - though timm handles this safely now
                # We'll stick to model's dtype or default float32 for safety
                pass

            with torch.no_grad():
                # For num_classes=0, timm returns the pooled output directly
                outputs = self._model(batch_tensor)

            embeddings = outputs.cpu().numpy()
            all_embeddings.append(embeddings)

        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)

        # Create ObjectFeature instances
        features = []
        for (det, crop), embedding in zip(valid_detections, all_embeddings):
            features.append(
                ObjectFeature(
                    detection=det,
                    embedding=embedding,
                    crop_image=crop,
                )
            )

        return features

    def extract_single(
        self, image: np.ndarray, detection: Detection, store_crop: bool = False
    ) -> ObjectFeature | None:
        """Extract feature for a single detection.

        Args:
            image: Full frame image
            detection: Single detection
            store_crop: Whether to store the crop

        Returns:
            ObjectFeature or None if extraction fails
        """
        features = self.extract(image, [detection], store_crop)
        return features[0] if features else None


class FeatureBuffer:
    """Buffer for collecting features during warmup phase."""

    def __init__(self, max_size: int = 10000):
        """Initialize feature buffer.

        Args:
            max_size: Maximum number of features to store
        """
        self.max_size = max_size
        self._features: list[ObjectFeature] = []
        self._features_by_class: dict[str, list[ObjectFeature]] = {}
        self._embeddings: np.ndarray | None = None
        self._dirty = True  # Track if embeddings array needs rebuild

    def add(self, features: list[ObjectFeature]) -> None:
        """Add features to the buffer (legacy method, also groups by class)."""
        for feature in features:
            self._add_single_internal(feature)
        self._dirty = True

    def _add_single_internal(self, feature: ObjectFeature) -> None:
        """Internal method to add a single feature."""
        self._features.append(feature)
        
        # Also group by class
        class_name = feature.detection.class_name
        if class_name not in self._features_by_class:
            self._features_by_class[class_name] = []
        self._features_by_class[class_name].append(feature)

        # Trim if over max size (keep most recent)
        if len(self._features) > self.max_size:
            # Remove oldest from main list
            removed = self._features.pop(0)
            # Also remove from class dict
            removed_class = removed.detection.class_name
            if removed_class in self._features_by_class:
                class_features = self._features_by_class[removed_class]
                if class_features and class_features[0] is removed:
                    class_features.pop(0)
                if not class_features:
                    del self._features_by_class[removed_class]

    def add_single(self, feature: ObjectFeature) -> None:
        """Add a single feature."""
        self._add_single_internal(feature)
        self._dirty = True

    @property
    def embeddings(self) -> np.ndarray:
        """Get all embeddings as a numpy array."""
        if self._dirty or self._embeddings is None:
            if self._features:
                self._embeddings = np.vstack([f.embedding for f in self._features])
            else:
                self._embeddings = np.array([])
            self._dirty = False
        return self._embeddings

    @property
    def features(self) -> list[ObjectFeature]:
        """Get all features."""
        return self._features

    def get_class_names(self) -> list[str]:
        """Get list of all class names in the buffer."""
        return list(self._features_by_class.keys())

    def get_class_counts(self) -> dict[str, int]:
        """Get count of features per class."""
        return {name: len(feats) for name, feats in self._features_by_class.items()}

    def get_class_embeddings(self, class_name: str) -> np.ndarray:
        """Get embeddings for a specific class.
        
        Args:
            class_name: The YOLO class name (e.g., "car", "truck")
            
        Returns:
            Array of embeddings for that class, or empty array if class not found
        """
        if class_name not in self._features_by_class:
            return np.array([])
        features = self._features_by_class[class_name]
        if not features:
            return np.array([])
        return np.vstack([f.embedding for f in features])

    def get_class_features(self, class_name: str) -> list[ObjectFeature]:
        """Get features for a specific class."""
        return self._features_by_class.get(class_name, [])

    def get_embeddings_by_class(self) -> dict[str, np.ndarray]:
        """Get all embeddings grouped by class.
        
        Returns:
            Dict mapping class_name to numpy array of embeddings
        """
        result = {}
        for class_name in self._features_by_class:
            embeddings = self.get_class_embeddings(class_name)
            if len(embeddings) > 0:
                result[class_name] = embeddings
        return result

    def __len__(self) -> int:
        return len(self._features)

    def clear(self) -> None:
        """Clear the buffer."""
        self._features = []
        self._features_by_class = {}
        self._embeddings = None
        self._dirty = True


# Backward-compatible alias
DINOv2Extractor = DINOv3Extractor
