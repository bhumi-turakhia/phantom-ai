"""
Independent Multi-Label Forensic Attribute Evaluator (ForensicFeatureCNN).
Trained strictly from scratch on CelebA Split B1 to evaluate 20 GCIT attributes.
Independent of Generator and Critic; does not use pretrained weights.
Deep feature extraction used for Custom Feature Fréchet Distance (C-FD).
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.gcit_translator import GCIT_LAYOUT


def weights_init_evaluator(m: nn.Module) -> None:
    """
    Kaiming (He) normal weight initialization for ForensicFeatureCNN.
    Prevents vanishing/exploding gradients with LeakyReLU non-linearities.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class ForensicFeatureCNN(nn.Module):
    """
    Independent Deep Convolutional Network for multi-label facial attribute classification.

    Architecture:
      Input: (B, 3, 64, 64) in [-1.0, 1.0].
      4 Convolutional stages with residual depth, Batch Normalization, and LeakyReLU(0.1).
      Global Adaptive Average Pooling producing a 512-dimensional feature embedding psi(x).
      Multi-label classification head outputting 20 attribute logits.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_attributes: int = 20,
        feature_dim: int = 512,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_attributes = num_attributes
        self.feature_dim = feature_dim

        # Block 1: 64x64 -> 32x32 (Channels: 3 -> 64)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate * 0.5),
        )

        # Block 2: 32x32 -> 16x16 (Channels: 64 -> 128)
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate * 0.75),
        )

        # Block 3: 16x16 -> 8x8 (Channels: 128 -> 256)
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate),
        )

        # Block 4: 8x8 -> 4x4 (Channels: 256 -> 512)
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_attributes, bias=True),
        )

        self.apply(weights_init_evaluator)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts deep feature representation psi(x) in R^{512}.
        Used for computing Custom Feature Fréchet Distance (C-FD).
        """
        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        psi_x = h.view(h.size(0), -1)
        return psi_x

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, 3, 64, 64) image tensor in [-1.0, 1.0].
            return_features: If True, returns (logits, features).

        Returns:
            logits: (B, 20) raw unnormalized logits for the 20 GCIT attributes.
            features: Optional (B, 512) feature embedding psi(x).
        """
        features = self.extract_features(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits, None

    def predict_attributes(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference helper: returns predicted binary attribute vector {0, 1} and probabilities [0, 1].
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = torch.sigmoid(logits)
            binary_preds = (probs >= threshold).float()
        return binary_preds, probs


def calculate_attribute_pos_weights(
    train_attributes: torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """
    Computes class imbalance pos_weight vector for BCEWithLogitsLoss:
      pos_weight_k = (N_total - N_pos_k) / (N_pos_k + epsilon)
    Prevents rare attributes (e.g. bald, gray hair) from being dominated by majority negative classes.

    Args:
        train_attributes: (N, 20) tensor with labels in {-1.0, 1.0} or {0.0, 1.0}.
    Returns:
        pos_weights: (20,) tensor.
    """
    # Normalize to {0.0, 1.0} if in {-1.0, 1.0}
    if train_attributes.min() < 0.0:
        targets = (train_attributes + 1.0) / 2.0
    else:
        targets = train_attributes

    n_samples = targets.size(0)
    pos_counts = torch.sum(targets, dim=0)
    neg_counts = n_samples - pos_counts

    pos_weights = neg_counts / (pos_counts + epsilon)
    # Clip extreme weights to avoid numerical instability
    pos_weights = torch.clamp(pos_weights, min=0.1, max=50.0)
    return pos_weights

