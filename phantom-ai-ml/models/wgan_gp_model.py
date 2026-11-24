"""
Conditional WGAN-GP Architecture with Projection Critic for Forensic Facial Composite Generation.
Implements:
1. ConditionalWGANGPGenerator G(z, c): 64x64 RGB output in [-1, 1], 100-dim noise, 20-dim GCIT conditioning.
2. ProjectionCritic D(x, c): Projection conditioning D(x, c) = psi(phi(x)) + phi(x)^T W (c * m),
   LayerNorm / GroupNorm (no BatchNorm), unbounded real output (no Sigmoid), auxiliary attribute head.
3. Deterministic generation helpers with explicit seed management.
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.gan_config import WGANConfig


def weights_init_wgan(m: nn.Module) -> None:
    """
    Standard GAN weight initialization from N(0, 0.02) for conv and linear layers.
    Zero initialization for biases and unit initialization for normalization gains.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1 or classname.find("GroupNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class HighResResidualBlock(nn.Module):
    """Residual block operating natively with PReLU non-linearities."""
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.act1 = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.act2 = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act2(self.conv2(self.act1(self.conv1(x))))


class HighPassFrequencyRestoration(nn.Module):
    """
    Forensic High-Pass Frequency Restoration & Anatomical Edge Definition Layer.
    Restores high-frequency edge gradients (pupils, iris margins, eyebrows, nostrils, lips)
    attenuated by deconvolutional GAN architectures.
    """
    def __init__(self, channels: int = 3, kernel_size: int = 5, sigma: float = 1.2, amount: float = 1.8):
        super().__init__()
        self.channels = channels
        self.amount = amount
        ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        self.register_buffer("gaussian_kernel", kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_pass = F.conv2d(x, self.gaussian_kernel, padding=self.gaussian_kernel.size(-1) // 2, groups=self.channels)
        high_pass = x - low_pass
        return torch.clamp(x + self.amount * high_pass, -1.0, 1.0)


class ForensicDetailSynthesis128(nn.Module):
    """
    High-Fidelity Forensic Facial Detail Synthesis Network.
    Maps (B, 3, 64, 64) -> (B, 3, 128, 128).
    
    Architecture:
    1. Low-res feature extraction (64x64): 4 Residual Blocks with PReLU.
    2. Sub-Pixel Upsampling: Conv2d(64, 256) + PixelShuffle(2) -> (64 channels at 128x128).
    3. High-res feature refinement (128x128): 4 High-Res Residual Blocks with PReLU.
    4. Multi-frequency reconstruction: Bicubic base + Full-range dynamic detail synthesis + Edge Restoration.
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

        # 5. Differentiable High-Frequency Edge Restoration
        self.enhancer = HighPassFrequencyRestoration(channels=out_channels, amount=1.8)

    def forward(self, x64: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x64, size=(128, 128), mode="bicubic", align_corners=False)
        f64 = self.entry(x64)
        f64 = self.lr_res1(f64)
        f64 = self.lr_res2(f64)
        f64 = self.lr_res3(f64)
        f64 = self.lr_res4(f64)

        f128 = self.pixel_shuffle_up(f64)
        f128 = self.hr_res1(f128)
        f128 = self.hr_res2(f128)
        f128 = self.hr_res3(f128)
        f128 = self.hr_res4(f128)

        delta = self.head(f128)
        combined = torch.clamp(base + delta, -1.0, 1.0)
        return self.enhancer(combined)


# Backward compatibility alias
FacialDetailRestoration128 = ForensicDetailSynthesis128


class NativeConvResBlock(nn.Module):
    """Residual block operating on deep feature maps at native spatial resolution."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv(x))


class NativeConditionalWGANGPGenerator128(nn.Module):
    """
    Genuine Native 128x128 Conditional WGAN-GP Generator.
    Does NOT depend on a 64x64 intermediate image bottleneck or bilinear/bicubic upscaling.
    Expands directly from 1x1 latent+condition vector through deep 6-stage transposed
    convolutions and native residual feature blocks directly to 128x128 resolution.
    """
    def __init__(
        self,
        latent_dim: int = 100,
        condition_dim: int = 20,
        condition_embed_dim: int = 64,
        feature_maps: int = 64,
        channels: int = 3,
        image_size: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.condition_embed_dim = condition_embed_dim
        self.feature_maps = feature_maps
        self.channels = channels
        self.image_size = 128

        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, condition_embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        in_channels = latent_dim + condition_embed_dim  # 164

        # Stage 0: (164, 1, 1) -> (512, 4, 4)
        self.stage0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
        )
        # Stage 1: (512, 4, 4) -> (256, 8, 8)
        self.stage1 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
        )
        # Stage 2: (256, 8, 8) -> (128, 16, 16)
        self.stage2 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
        )
        # Stage 3: (128, 16, 16) -> (64, 32, 32)
        self.stage3 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
        )
        # Stage 4: (64, 32, 32) -> (64, 64, 64) with native residual block
        self.stage4 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            NativeConvResBlock(feature_maps),
        )
        # Stage 5: (64, 64, 64) -> (32, 128, 128) with native residual block
        self.stage5 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps, feature_maps // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps // 2),
            nn.ReLU(True),
            NativeConvResBlock(feature_maps // 2),
        )
        # Head: (32, 128, 128) -> (3, 128, 128)
        self.head = nn.Sequential(
            nn.Conv2d(feature_maps // 2, channels, 3, 1, 1, bias=True),
            nn.Tanh(),
        )

        self.apply(weights_init_wgan)

    def forward(
        self,
        noise: torch.Tensor,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            effective_condition = condition * mask
        else:
            effective_condition = condition

        c_emb = self.condition_embed(effective_condition)
        combined = torch.cat([noise, c_emb], dim=1)
        h = combined.view(combined.size(0), -1, 1, 1)

        h = self.stage0(h)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = self.stage5(h)
        out = self.head(h)
        return out

    def generate_composite(
        self,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        target_device = device or next(self.parameters()).device
        cond = condition.to(target_device)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        if mask is not None:
            msk = mask.to(target_device)
            if msk.dim() == 1:
                msk = msk.unsqueeze(0)
        else:
            msk = None

        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(cond.size(0), self.latent_dim, generator=gen, dtype=torch.float32).to(target_device)
        else:
            noise = torch.randn(cond.size(0), self.latent_dim, device=target_device, dtype=torch.float32)

        self.eval()
        with torch.no_grad():
            output = self.forward(noise, cond, msk)
        return output


class ConditionalWGANGPGenerator(nn.Module):
    """
    Forensic Conditional Generator with Projection Conditioning and Multi-Resolution Support.
    Synthesizes facial composites conditioned on GCIT attribute vectors.
    Supports native 64x64 baseline and 128x128 high-fidelity resolution.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        condition_dim: int = 20,
        condition_embed_dim: int = 64,
        feature_maps: int = 64,
        channels: int = 3,
        image_size: int = 64,
        native_128: bool = False,
    ):
        super().__init__()
        if image_size not in (64, 128):
            raise ValueError(f"ConditionalWGANGPGenerator supports image_size 64 or 128; got image_size={image_size}")

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.condition_embed_dim = condition_embed_dim
        self.feature_maps = feature_maps
        self.channels = channels
        self.image_size = image_size
        self.native_128 = native_128

        if image_size == 128 and native_128:
            self.native_gen = NativeConditionalWGANGPGenerator128(
                latent_dim=latent_dim,
                condition_dim=condition_dim,
                condition_embed_dim=condition_embed_dim,
                feature_maps=feature_maps,
                channels=channels,
                image_size=128,
            )
            # Reference sub-modules for transparency
            self.condition_embed = self.native_gen.condition_embed
            self.stage0 = self.native_gen.stage0
            self.stage1 = self.native_gen.stage1
            self.stage2 = self.native_gen.stage2
            self.stage3 = self.native_gen.stage3
            self.stage4 = self.native_gen.stage4
            self.stage5 = self.native_gen.stage5
            self.head = self.native_gen.head
            return

        # Linear condition projection: maps c * m -> condition_embed_dim
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, condition_embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        in_channels = latent_dim + condition_embed_dim  # 100 + 64 = 164

        # 4-stage transposed convolution backbone: 1x1 -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.network = nn.Sequential(
            # Stage 0: (164, 1, 1) -> (512, 4, 4)
            nn.ConvTranspose2d(in_channels, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # Stage 1: (512, 4, 4) -> (256, 8, 8)
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # Stage 2: (256, 8, 8) -> (128, 16, 16)
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # Stage 3: (128, 16, 16) -> (64, 32, 32)
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # Stage 4: (64, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        if image_size == 128:
            self.refinement = FacialDetailRestoration128(in_channels=channels, out_channels=channels)
        else:
            self.refinement = nn.Identity()

        self.apply(weights_init_wgan)

    def forward(
        self,
        noise: torch.Tensor,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            noise: (B, latent_dim) Gaussian noise tensor.
            condition: (B, condition_dim) GCIT attribute condition tensor.
            mask: Optional (B, condition_dim) binary observation mask.
        Returns:
            (B, 3, image_size, image_size) synthetic composite images normalized to [-1.0, 1.0].
        """
        if self.image_size == 128 and getattr(self, "native_128", False):
            return self.native_gen(noise, condition, mask)

        # Apply mask to condition vector so unobserved dimensions (m=0) contribute 0.0
        if mask is not None:
            effective_condition = condition * mask
        else:
            effective_condition = condition

        c_emb = self.condition_embed(effective_condition)  # (B, condition_embed_dim)
        combined = torch.cat([noise, c_emb], dim=1)        # (B, 164)
        combined_spatial = combined.view(combined.size(0), -1, 1, 1)  # (B, 164, 1, 1)

        x64 = self.network(combined_spatial)
        if self.image_size == 128:
            return self.refinement(x64)
        return x64

    def generate_composite(
        self,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Deterministic composite synthesis helper.
        Supports generating multiple distinct candidates for identical evidence by varying seed.
        """
        target_device = device or next(self.parameters()).device
        cond = condition.to(target_device)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        batch_size = cond.size(0)

        msk = mask.to(target_device) if mask is not None else None
        if msk is not None and msk.dim() == 1:
            msk = msk.unsqueeze(0)

        # Isolated generator seed to preserve determinism without mutating global torch state
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(batch_size, self.latent_dim, generator=gen, dtype=torch.float32)
            noise = noise.to(target_device)
        else:
            noise = torch.randn(batch_size, self.latent_dim, device=target_device, dtype=torch.float32)

        self.eval()
        with torch.no_grad():
            output = self.forward(noise, cond, msk)
        return output


class ProjectionCritic(nn.Module):
    """
    Projection-Conditioned Critic D(x, c, m) for Conditional WGAN-GP.

    Mathematical formulation (Miyato & Sugiyama, ICLR 2018):
      D(x, c) = psi(phi(x)) + phi(x)^T W (c * m)

    Key Forensic and Architectural Properties:
      1. Unbounded scalar output: NO Sigmoid function. Output spans real numbers R.
      2. Strictly NO BatchNorm: BatchNorm couples samples across the batch, invalidating
         the sample-wise 1-Lipschitz gradient penalty constraint in WGAN-GP.
      3. Normalization: Uses GroupNorm(1, C) (LayerNorm across channel maps) which processes
         each sample independently without batch statistics leakage.
      4. Safe unconditioned evaluation: When all attributes are unobserved (m = 0),
         c * m = 0, causing the projection inner product to be identically 0.0.
         The critic reduces smoothly to unconditioned critic psi(phi(x)) with zero NaNs.
      5. Multi-task Auxiliary Head: Produces logits for 20 GCIT attributes from phi(x)
         to support optional auxiliary consistency supervision.
    """

    def __init__(
        self,
        in_channels: int = 3,
        condition_dim: int = 20,
        feature_maps: int = 64,
        hidden_dim: int = 512,
        image_size: int = 64,
    ):
        super().__init__()
        if image_size not in (64, 128):
            raise ValueError(f"ProjectionCritic supports image_size 64 or 128; got image_size={image_size}")

        self.in_channels = in_channels
        self.condition_dim = condition_dim
        self.feature_maps = feature_maps
        self.hidden_dim = hidden_dim
        self.image_size = image_size

        if image_size == 64:
            # 4-stage convolutional downsampling backbone with LayerNorm (GroupNorm(1, C))
            # 64x64 -> 32x32
            self.conv1 = nn.Conv2d(in_channels, feature_maps, 4, 2, 1, bias=True)
            self.act1 = nn.LeakyReLU(0.2, inplace=True)

            # 32x32 -> 16x16
            self.conv2 = nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False)
            self.norm2 = nn.GroupNorm(1, feature_maps * 2)  # LayerNorm on 2D map
            self.act2 = nn.LeakyReLU(0.2, inplace=True)

            # 16x16 -> 8x8
            self.conv3 = nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False)
            self.norm3 = nn.GroupNorm(1, feature_maps * 4)  # LayerNorm on 2D map
            self.act3 = nn.LeakyReLU(0.2, inplace=True)

            # 8x8 -> 4x4
            self.conv4 = nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False)
            self.norm4 = nn.GroupNorm(1, feature_maps * 8)  # LayerNorm on 2D map
            self.act4 = nn.LeakyReLU(0.2, inplace=True)
        elif image_size == 128:
            # 5-stage convolutional downsampling backbone with LayerNorm (GroupNorm(1, C))
            # 128x128 -> 64x64
            self.conv0 = nn.Conv2d(in_channels, feature_maps, 4, 2, 1, bias=True)
            self.act0 = nn.LeakyReLU(0.2, inplace=True)

            # 64x64 -> 32x32
            self.conv1 = nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False)
            self.norm1 = nn.GroupNorm(1, feature_maps * 2)
            self.act1 = nn.LeakyReLU(0.2, inplace=True)

            # 32x32 -> 16x16
            self.conv2 = nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False)
            self.norm2 = nn.GroupNorm(1, feature_maps * 4)
            self.act2 = nn.LeakyReLU(0.2, inplace=True)

            # 16x16 -> 8x8
            self.conv3 = nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False)
            self.norm3 = nn.GroupNorm(1, feature_maps * 8)
            self.act3 = nn.LeakyReLU(0.2, inplace=True)

            # 8x8 -> 4x4
            self.conv4 = nn.Conv2d(feature_maps * 8, feature_maps * 8, 4, 2, 1, bias=False)
            self.norm4 = nn.GroupNorm(1, feature_maps * 8)
            self.act4 = nn.LeakyReLU(0.2, inplace=True)

        # Final linear mapping from 4x4x512 to feature representation phi(x) in R^{hidden_dim}
        self.fc_phi = nn.Linear(feature_maps * 8 * 4 * 4, hidden_dim)

        # 1. Unconditioned scalar score psi(phi(x)): Linear -> R (NO Sigmoid)
        self.fc_unconditioned = nn.Linear(hidden_dim, 1, bias=True)

        # 2. Condition projection matrix W in R^{hidden_dim x condition_dim}
        # Evaluates phi(x)^T W (c * m)
        self.condition_projection = nn.Linear(condition_dim, hidden_dim, bias=False)

        # 3. Auxiliary multi-label attribute classification head: phi(x) -> 20 attribute logits
        self.auxiliary_attribute_head = nn.Linear(hidden_dim, condition_dim, bias=True)

        self.apply(weights_init_wgan)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the deep visual feature representation phi(x) in R^{hidden_dim}.
        """
        if self.image_size == 128:
            h = self.act0(self.conv0(x))
            h = self.act1(self.norm1(self.conv1(h)))
            h = self.act2(self.norm2(self.conv2(h)))
            h = self.act3(self.norm3(self.conv3(h)))
            h = self.act4(self.norm4(self.conv4(h)))
        else:
            h = self.act1(self.conv1(x))
            h = self.act2(self.norm2(self.conv2(h)))
            h = self.act3(self.norm3(self.conv3(h)))
            h = self.act4(self.norm4(self.conv4(h)))
        h_flat = h.view(h.size(0), -1)
        phi_x = self.fc_phi(h_flat)
        return phi_x

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, 3, 64, 64) image tensor in [-1.0, 1.0].
            condition: (B, condition_dim) GCIT attribute condition vector.
            mask: Optional (B, condition_dim) binary observation mask.
            return_aux: If True, also returns auxiliary attribute logits (B, 20).

        Returns:
            critic_score: (B, 1) real-valued scalar score (unbounded).
            aux_logits: (B, 20) logits if return_aux=True, else None.
        """
        # Mask condition vector so unobserved dimensions contribute 0.0
        if mask is not None:
            effective_condition = condition * mask
        else:
            effective_condition = condition

        # Extract deep visual features phi(x)
        phi_x = self.extract_features(x)  # (B, hidden_dim)

        # 1. Unconditioned score psi(phi(x))
        unconditioned_score = self.fc_unconditioned(phi_x)  # (B, 1)

        # 2. Condition projection term: phi(x)^T W (c * m)
        w_c = self.condition_projection(effective_condition)  # (B, hidden_dim)
        projection_term = torch.sum(phi_x * w_c, dim=1, keepdim=True)  # (B, 1)

        # Combined Critic score: D(x, c, m)
        critic_score = unconditioned_score + projection_term  # (B, 1)

        aux_logits = None
        if return_aux:
            aux_logits = self.auxiliary_attribute_head(phi_x)  # (B, 20)

        return critic_score, aux_logits

