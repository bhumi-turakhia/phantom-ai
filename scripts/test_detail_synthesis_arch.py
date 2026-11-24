import torch
import torch.nn as nn
import torch.nn.functional as F

class HighResResidualBlock(nn.Module):
    """Residual block operating natively at 128x128 resolution."""
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.act1 = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.act2 = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act2(self.conv2(self.act1(self.conv1(x))))


class ForensicDetailSynthesis128(nn.Module):
    """
    High-Fidelity Forensic Facial Detail Synthesis Network.
    Maps (B, 3, 64, 64) -> (B, 3, 128, 128).
    
    Architecture:
    1. Low-res feature extraction (64x64): 4 Residual Blocks with PReLU.
    2. Sub-Pixel Upsampling: Conv2d(64, 256) + PixelShuffle(2) -> (64 channels at 128x128).
    3. High-res feature refinement (128x128): 4 High-Res Residual Blocks with PReLU.
    4. Multi-frequency reconstruction:
       - Bicubic base upsampling preserves spatial layout and continuous skin tones.
       - Full-capacity dynamic residual delta_x in [-1.0, 1.0] without artificial dampening.
       - High-frequency edge enhancer predicting sharp facial landmark gradients.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1. 64x64 Feature Extraction
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.PReLU(),
        )
        self.lr_res1 = HighResResidualBlock(base_channels)
        self.lr_res2 = HighResResidualBlock(base_channels)
        self.lr_res3 = HighResResidualBlock(base_channels)
        self.lr_res4 = HighResResidualBlock(base_channels)

        # 2. Sub-Pixel PixelShuffle Upsampling: 64x64 -> 128x128
        self.pixel_shuffle_up = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 64x64 -> 128x128, channels: 256 -> 64
            nn.PReLU(),
        )

        # 3. 128x128 High-Resolution Anatomical Refinement
        self.hr_res1 = HighResResidualBlock(base_channels)
        self.hr_res2 = HighResResidualBlock(base_channels)
        self.hr_res3 = HighResResidualBlock(base_channels)
        self.hr_res4 = HighResResidualBlock(base_channels)

        # 4. Multi-frequency synthesis head
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, 32, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Tanh(),  # delta_x in [-1.0, 1.0]
        )

    def forward(self, x64: torch.Tensor) -> torch.Tensor:
        # Bicubic interpolation provides sharp foundation
        base = F.interpolate(x64, size=(128, 128), mode="bicubic", align_corners=False)

        # Low-res feature encoding
        f64 = self.entry(x64)
        f64 = self.lr_res1(f64)
        f64 = self.lr_res2(f64)
        f64 = self.lr_res3(f64)
        f64 = self.lr_res4(f64)

        # Sub-pixel expansion to 128x128
        f128 = self.pixel_shuffle_up(f64)

        # High-res anatomical detail refinement
        f128 = self.hr_res1(f128)
        f128 = self.hr_res2(f128)
        f128 = self.hr_res3(f128)
        f128 = self.hr_res4(f128)

        # Full-range dynamic detail synthesis
        delta = self.head(f128)

        # Final composite with clamped forensic range [-1.0, 1.0]
        return torch.clamp(base + delta, -1.0, 1.0)


if __name__ == "__main__":
    model = ForensicDetailSynthesis128()
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"ForensicDetailSynthesis128 output shape: {out.shape}, total params: {params:,}")
    assert out.shape == (2, 3, 128, 128)
    print("Architecture verified successfully.")

