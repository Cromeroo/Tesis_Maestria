"""Arquitecturas CNN. Sin bug, sin duplicados (v1 tenía `nn.ReLU(),-`)."""
import torch.nn as nn


class SimpleCNN(nn.Module):
    """CNN de la tesis original (128x128 -> 3 clases)."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))


def build_model(backbone: str = "simple_cnn", num_classes: int = 3) -> nn.Module:
    """backbone: "simple_cnn" (tesis) | "efficientnet_b0" (mejora propuesta)."""
    if backbone == "efficientnet_b0":
        from torchvision import models
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if backbone != "simple_cnn":
        raise ValueError(f"backbone desconocido: {backbone}")
    return SimpleCNN(num_classes=num_classes)
