"""Model architecture helpers."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    mobilenet_v2,
    resnet18,
    resnet34,
)

from model.constants import DEFAULT_ARCHITECTURE


def build_classifier(
    num_classes: int,
    architecture: str = DEFAULT_ARCHITECTURE,
    pretrained: bool = True,
) -> nn.Module:
    """Build a transfer-learning classifier for validation or disease detection."""
    if architecture == "mobilenet_v2":
        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        model = mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if architecture == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if architecture == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported architecture: {architecture}")
