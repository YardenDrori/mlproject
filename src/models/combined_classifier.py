"""Combined classifier for fruit type + freshness inference."""

from typing import Dict, List, Tuple

import torch
from PIL import Image

from .fruit_classifier import FruitClassifier
from .freshness_classifier import FreshnessClassifier
from ..data.transforms import get_val_transforms
from ..utils.device import get_device


class CombinedFruitClassifier:
    """
    Combined inference for fruit type and freshness.

    Loads both trained models and produces outputs like:
    "fresh apple", "rotten banana", etc.
    """

    def __init__(
        self,
        fruit_checkpoint: str,
        freshness_checkpoint: str,
        fruit_classes: List[str],
        freshness_classes: List[str] = None,
        img_size: int = 224,
        device: torch.device = None
    ):
        """
        Initialize the combined classifier.

        Args:
            fruit_checkpoint: Path to fruit classifier checkpoint
            freshness_checkpoint: Path to freshness classifier checkpoint
            fruit_classes: List of fruit class names
            freshness_classes: List of freshness class names (default: fresh, rotten)
            img_size: Image size for inference
            device: Torch device (auto-detected if None)
        """
        self.device = device or get_device()
        self.img_size = img_size
        self.fruit_classes = fruit_classes
        self.freshness_classes = freshness_classes or ["fresh", "rotten"]
        self.transform = get_val_transforms(img_size)

        # Load fruit classifier
        self.fruit_model = FruitClassifier(
            num_classes=len(self.fruit_classes),
            pretrained=False
        )
        self._load_checkpoint(self.fruit_model, fruit_checkpoint)
        self.fruit_model.to(self.device)
        self.fruit_model.eval()

        # Load freshness classifier
        self.freshness_model = FreshnessClassifier(
            num_classes=len(self.freshness_classes),
            pretrained=False
        )
        self._load_checkpoint(self.freshness_model, freshness_checkpoint)
        self.freshness_model.to(self.device)
        self.freshness_model.eval()

    def _load_checkpoint(self, model: torch.nn.Module, checkpoint_path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

    def predict_single(self, image_path: str) -> Tuple[str, Dict]:
        """
        Predict fruit type and freshness for a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (combined_label, confidence_dict)
            - combined_label: e.g., "fresh apple" or "rotten banana"
            - confidence_dict: Detailed confidence scores
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get predictions from both models
            fruit_logits = self.fruit_model(tensor)
            freshness_logits = self.freshness_model(tensor)

            # Get probabilities
            fruit_probs = torch.softmax(fruit_logits, dim=1)
            freshness_probs = torch.softmax(freshness_logits, dim=1)

            # Get predicted classes
            fruit_idx = fruit_probs.argmax(dim=1).item()
            freshness_idx = freshness_probs.argmax(dim=1).item()

            fruit_name = self.fruit_classes[fruit_idx]
            freshness_name = self.freshness_classes[freshness_idx]

            # Combine: "fresh apple" or "rotten banana"
            combined_label = f"{freshness_name} {fruit_name}"

            confidence = {
                "fruit": {
                    "label": fruit_name,
                    "confidence": fruit_probs[0, fruit_idx].item()
                },
                "freshness": {
                    "label": freshness_name,
                    "confidence": freshness_probs[0, freshness_idx].item()
                }
            }

        return combined_label, confidence

    def predict_batch(self, image_paths: List[str]) -> List[Tuple[str, Dict]]:
        """
        Batch prediction for multiple images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of (combined_label, confidence_dict) tuples
        """
        results = []
        for path in image_paths:
            result = self.predict_single(path)
            results.append(result)
        return results
