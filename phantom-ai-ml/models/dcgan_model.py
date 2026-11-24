import torch
import torch.nn as nn
from typing import Tuple


def weights_init_dcgan(m: nn.Module) -> None:
    """
    Standard DCGAN weight initialization from N(0, 0.02).
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ConditionalDCGANGenerator(nn.Module):
    """
    Custom Deep Convolutional GAN Generator for conditional facial composite generation experiments.
    Configurable for 64x64 or 128x128 resolution.
    Strictly initialized with random weights (no pretrained weights).
    """
    def __init__(
        self,
        latent_dim: int = 100,
        condition_dim: int = 32,
        out_channels: int = 3,
        feature_maps: int = 64,
        image_size: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.image_size = image_size
        self.out_channels = out_channels
        self.feature_maps = feature_maps

        in_features = latent_dim + condition_dim

        if image_size == 64:
            self.network = nn.Sequential(
                # Input is latent vector Z + condition vector, projected to (feature_maps * 8) x 4 x 4
                nn.ConvTranspose2d(in_features, feature_maps * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(feature_maps * 8),
                nn.ReLU(True),
                # 4x4 -> 8x8
                nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 4),
                nn.ReLU(True),
                # 8x8 -> 16x16
                nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 2),
                nn.ReLU(True),
                # 16x16 -> 32x32
                nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps),
                nn.ReLU(True),
                # 32x32 -> 64x64
                nn.ConvTranspose2d(feature_maps, out_channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
        elif image_size == 128:
            self.network = nn.Sequential(
                # 1x1 -> 4x4
                nn.ConvTranspose2d(in_features, feature_maps * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(feature_maps * 16),
                nn.ReLU(True),
                # 4x4 -> 8x8
                nn.ConvTranspose2d(feature_maps * 16, feature_maps * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 8),
                nn.ReLU(True),
                # 8x8 -> 16x16
                nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 4),
                nn.ReLU(True),
                # 16x16 -> 32x32
                nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 2),
                nn.ReLU(True),
                # 32x32 -> 64x64
                nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps),
                nn.ReLU(True),
                # 64x64 -> 128x128
                nn.ConvTranspose2d(feature_maps, out_channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
        else:
            raise ValueError(f"Unsupported image_size: {image_size}. Supported sizes: 64, 128.")

        self.apply(weights_init_dcgan)

    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise: Tensor of shape (batch_size, latent_dim)
            condition: Tensor of shape (batch_size, condition_dim)
        Returns:
            generated_image: Tensor of shape (batch_size, out_channels, image_size, image_size) in [-1, 1]
        """
        combined = torch.cat([noise, condition], dim=1)
        combined = combined.view(combined.size(0), -1, 1, 1)
        return self.network(combined)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConditionalDCGANDiscriminator(nn.Module):
    """
    Custom Deep Convolutional GAN Discriminator for conditional validity classification.
    """
    def __init__(
        self,
        in_channels: int = 3,
        condition_dim: int = 32,
        feature_maps: int = 64,
        image_size: int = 64,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.condition_dim = condition_dim

        # Condition projection layer to spatial feature map
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, image_size * image_size),
            nn.LeakyReLU(0.2, inplace=True),
        )

        total_channels = in_channels + 1  # 1 channel for spatial condition map

        if image_size == 64:
            self.network = nn.Sequential(
                # 64x64 -> 32x32
                nn.Conv2d(total_channels, feature_maps, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # 32x32 -> 16x16
                nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # 16x16 -> 8x8
                nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # 8x8 -> 4x4
                nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # 4x4 -> 1x1
                nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid(),
            )
        elif image_size == 128:
            self.network = nn.Sequential(
                # 128x128 -> 64x64
                nn.Conv2d(total_channels, feature_maps, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # 64x64 -> 32x32
                nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # 32x32 -> 16x16
                nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # 16x16 -> 8x8
                nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # 8x8 -> 4x4
                nn.Conv2d(feature_maps * 8, feature_maps * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 16),
                nn.LeakyReLU(0.2, inplace=True),
                # 4x4 -> 1x1
                nn.Conv2d(feature_maps * 16, 1, 4, 1, 0, bias=False),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unsupported image_size: {image_size}.")

        self.apply(weights_init_dcgan)

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Tensor of shape (batch_size, in_channels, image_size, image_size)
            condition: Tensor of shape (batch_size, condition_dim)
        Returns:
            probability: Real vs fake probability score in [0, 1] of shape (batch_size, 1, 1, 1)
        """
        batch_size = image.size(0)
        cond_map = self.condition_proj(condition).view(batch_size, 1, self.image_size, self.image_size)
        combined = torch.cat([image, cond_map], dim=1)
        return self.network(combined).view(batch_size, -1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

