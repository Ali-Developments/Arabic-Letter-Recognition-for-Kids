"""
Model Architectures for Arabic Handwritten Character Recognition.
Includes:
1. ArabicCNN (Lightweight Custom 4-Block CNN)
2. ArabicMobileNetV3 (Adapted MobileNetV3-Small for 1-channel 32x32 Arabic input)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class ArabicCNN(nn.Module):
    """
    Lightweight, robust Convolutional Neural Network for 28-class Arabic character recognition.
    Optimized for fast CPU inference (<2ms) and high accuracy on children's handwriting.
    """

    def __init__(self, num_classes: int = 28):
        super().__init__()
        # Block 1: 32x32 -> 16x16
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)

        # Block 2: 16x16 -> 8x8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.25)

        # Classifier
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Flatten & FC
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.drop3(x)
        x = self.fc2(x)
        return x


class ArabicMobileNetV3(nn.Module):
    """
    MobileNetV3-Small adapted for 1-channel grayscale 32x32 isolated Arabic letter recognition.
    Initializes from pretrained ImageNet weights with first-layer RGB filter channel averaging.
    Uses depthwise-separable convolutions and squeeze-and-excitation blocks for dot discrimination.
    """

    def __init__(self, num_classes: int = 28, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)

        # Adapt first conv from 3 channels to 1 channel
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained and old_conv.weight is not None:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        base.features[0][0] = new_conv
        self.features = base.features
        self.avgpool = base.avgpool

        # 28-class classification head
        in_features = 576
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
