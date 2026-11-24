import torch
import torch.nn as nn
from typing import Dict, Optional


class ForensicFeatureCNN(nn.Module):
    """
    Custom small convolutional neural network for facial trait extraction.
    Designed for training from scratch (no ImageNet/VGG/ResNet pretrained weights).
    """
    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 256,
        image_size: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.image_size = image_size

        self.conv_blocks = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 4 * 4, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass extracting forensic feature embeddings.
        Args:
            x: Input image tensor of shape (batch_size, in_channels, height, width)
        Returns:
            embedding: Latent feature vector of shape (batch_size, feature_dim)
        """
        feats = self.conv_blocks(x)
        embeddings = self.embedding_head(feats)
        return embeddings

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

