"""Fruit type classifier model using EfficientNet-B0."""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class FruitClassifier(nn.Module):
    """
    Fruit type classifier using transfer learning with EfficientNet-B0.

    Identifies different types of fruits (apple, banana, orange, etc.)
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.2
    ):
        """
        Initialize the fruit classifier.

        Args:
            num_classes: Number of fruit classes to predict
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone weights (for faster training)
            dropout: Dropout rate for classifier head
        """
        super().__init__()

        # Load pretrained EfficientNet-B0
        if pretrained:
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = efficientnet_b0(weights=None)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Unfreeze backbone weights for fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get the number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
