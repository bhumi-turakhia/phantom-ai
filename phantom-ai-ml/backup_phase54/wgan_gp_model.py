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


class ConditionalWGANGPGenerator(nn.Module):
    """
    Conditional Generator G(z, c) synthesizing 64x64 RGB facial composites.

    Input:
      - z: Latent noise vector ~ N(0, I) of dimension latent_dim (100).
      - c: GCIT condition vector of dimension condition_dim (20), values in [-1.0, 1.0].
      - m: Optional binary mask of dimension (20,), values in {0.0, 1.0}.

    Conditioning Mechanism:
      c * m is projected via a dedicated condition embedding MLP into R^{condition_embed_dim}.
      The condition embedding is concatenated with z: [z; e_c] in R^{100 + 64} = R^{164}.
      The combined latent representation is expanded through 4 transposed convolution blocks
      with Batch Normalization and ReLU activations, concluding with a Tanh activation
      producing pixel values in [-1.0, 1.0].
    """

    def __init__(
        self,
        latent_dim: int = 100,
        condition_dim: int = 20,
        condition_embed_dim: int = 64,
        feature_maps: int = 64,
        channels: int = 3,
        image_size: int = 64,
    ):
        super().__init__()
        if image_size != 64:
            raise ValueError(f"Baseline generator currently designed for 64x64; got image_size={image_size}")

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.condition_embed_dim = condition_embed_dim
        self.feature_maps = feature_maps
        self.channels = channels
        self.image_size = image_size

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
            (B, 3, 64, 64) synthetic composite images normalized to [-1.0, 1.0].
        """
        # Apply mask to condition vector so unobserved dimensions (m=0) contribute 0.0
        if mask is not None:
            effective_condition = condition * mask
        else:
            effective_condition = condition

        c_emb = self.condition_embed(effective_condition)  # (B, condition_embed_dim)
        combined = torch.cat([noise, c_emb], dim=1)        # (B, 164)
        combined_spatial = combined.view(combined.size(0), -1, 1, 1)  # (B, 164, 1, 1)

        return self.network(combined_spatial)

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
        if image_size != 64:
            raise ValueError(f"Critic currently designed for 64x64; got image_size={image_size}")

        self.in_channels = in_channels
        self.condition_dim = condition_dim
        self.feature_maps = feature_maps
        self.hidden_dim = hidden_dim
        self.image_size = image_size

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

