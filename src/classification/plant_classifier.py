"""
PlantVision AI — Plant Identification Model

Phase 2 (Week 5). A MobileNetV2-based transfer-learning classifier.
Trained on the PlantVillage dataset (38 classes: 14 crop species crossed
with healthy/diseased states). This same model doubles as the "disease
detection" signal for PlantVillage-derived classes, since in this dataset
plant identity and disease state are encoded together in one label
(e.g. "Tomato___Early_blight"). A dedicated disease-only head can be split
out later (Month 3) if the project needs plant ID and disease prediction
to be independently confident.

Status: IMPLEMENTED (architecture). Not yet trained — see
training/train_plant_classifier.py and reports/ for actual run results.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class PlantClassifier(nn.Module):
    """MobileNetV2 backbone + a fresh classification head.

    Args:
        num_classes: number of output classes (38 for PlantVillage color set).
        pretrained: if True, load ImageNet weights for the backbone.
        freeze_backbone: if True, freeze all backbone weights and train only
            the new head (fast, good for a first baseline). Set False for a
            later fine-tuning pass once the head has converged.
        dropout: dropout probability in the classification head.
    """

    def __init__(
        self,
        num_classes: int = 38,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # MobileNetV2's classifier is Sequential(Dropout, Linear(1280, 1000)).
        # We keep the feature extractor and replace the classifier head.
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        in_features = backbone.last_channel  # 1280 for MobileNetV2
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        if freeze_backbone:
            self.set_backbone_trainable(False)

        self.num_classes = num_classes

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze the backbone. Call with True for a fine-tuning
        pass after the head has already converged (see MODEL_PLAN.md)."""
        for param in self.features.parameters():
            param.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(
    num_classes: int = 38,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> PlantClassifier:
    """Convenience factory used by the training script."""
    return PlantClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )


if __name__ == "__main__":
    # Quick sanity check: forward pass with random input, no training.
    model = build_model(num_classes=38)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)  # expected: torch.Size([2, 38])
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")
