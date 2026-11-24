"""
Training Objective Functions for Conditional WGAN-GP Architecture.
Implements:
1. Wasserstein adversarial critic and generator objectives.
2. 1-Lipschitz Gradient Penalty with sample-wise interpolation.
3. Masked auxiliary attribute consistency objective (ignoring unobserved attributes, zero-division safe).
4. Bounded mode-seeking diversity objective.
5. Centralized Loss Manager and gradient clipping utilities.
"""

from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.gan_config import WGANConfig


def compute_wasserstein_critic_loss(
    critic_real: torch.Tensor,
    critic_fake: torch.Tensor,
) -> torch.Tensor:
    """
    Standard Wasserstein critic loss:
      L_D_wass = E[D(G(z, c))] - E[D(x_real)]
    Critic aims to maximize D(x_real) and minimize D(x_fake), so minimizing this loss
    drives D(x_real) higher and D(x_fake) lower.
    """
    return torch.mean(critic_fake) - torch.mean(critic_real)


def compute_wasserstein_generator_loss(
    critic_fake: torch.Tensor,
) -> torch.Tensor:
    """
    Standard Wasserstein generator loss:
      L_G_wass = -E[D(G(z, c))]
    Generator aims to fool the critic into scoring fake images as real.
    """
    return -torch.mean(critic_fake)


def compute_gradient_penalty(
    critic: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    condition: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    lambda_gp: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the 1-Lipschitz Gradient Penalty (Gulrajani et al., 2017) with projection conditioning.

    Args:
        critic: ProjectionCritic model.
        real_images: (B, 3, 64, 64) real image batch.
        fake_images: (B, 3, 64, 64) synthesized image batch (detached).
        condition: (B, 20) condition vector.
        mask: Optional (B, 20) observation mask.
        lambda_gp: Gradient penalty weight factor (default: 10.0).

    Returns:
        gradient_penalty: Scaled penalty term lambda_gp * E[(||grad_x_hat D||_2 - 1)^2].
        grad_norm: Mean gradient norm for logging and diagnostic monitoring.
    """
    batch_size = real_images.size(0)
    device = real_images.device

    # Random interpolation factor epsilon ~ U(0, 1) per sample
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device, dtype=real_images.dtype)
    x_hat = epsilon * real_images + (1.0 - epsilon) * fake_images.detach()
    x_hat.requires_grad_(True)

    # Evaluate critic on interpolated images with condition
    d_hat, _ = critic(x_hat, condition, mask, return_aux=False)

    # Compute analytical gradients wrt interpolated input x_hat
    grad_outputs = torch.ones_like(d_hat)
    gradients = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Flatten spatial gradient tensors per sample: (B, 3 * 64 * 64)
    gradients = gradients.view(batch_size, -1)

    # Compute Euclidean L2 norm per sample with epsilon stabilizer to prevent sqrt(0) NaN
    grad_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # 1-Lipschitz penalty: penalize deviations from norm 1.0
    gp = torch.mean((grad_norm - 1.0) ** 2) * lambda_gp

    return gp, torch.mean(grad_norm).detach()


def compute_masked_auxiliary_loss(
    aux_logits: torch.Tensor,
    target_condition: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes masked multi-label attribute classification loss.
    Only evaluates attributes actively observed by witness (mask = 1.0).
    Safely ignores unobserved attributes without dividing by zero.

    Args:
        aux_logits: (B, 20) raw logits predicted by critic auxiliary head.
        target_condition: (B, 20) true condition targets in [-1.0, 1.0].
        mask: (B, 20) binary observation mask in {0.0, 1.0}.

    Returns:
        loss: Masked binary cross-entropy loss scalar.
    """
    # Convert targets from [-1.0, 1.0] to [0.0, 1.0] for BCE
    target_binary = (target_condition + 1.0) / 2.0

    # Unreduced element-wise binary cross entropy
    bce = F.binary_cross_entropy_with_logits(aux_logits, target_binary, reduction="none")

    if mask is not None:
        masked_bce = bce * mask
        total_observed = torch.sum(mask)

        # Zero-division safety: if zero attributes are observed in batch, loss is 0.0
        if total_observed.item() == 0.0:
            return torch.zeros((), device=aux_logits.device, dtype=aux_logits.dtype)

        return torch.sum(masked_bce) / torch.clamp(total_observed, min=1.0)
    else:
        return torch.mean(bce)


def compute_bounded_mode_seeking_loss(
    generator: nn.Module,
    condition: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    latent_dim: int = 100,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Computes bounded mode-seeking diversity loss (Mao et al., CVPR 2019).
    Encourages the generator to synthesize perceptually distinct composites when given
    different random noise seeds for identical evidence conditions.
    Strictly bounded by threshold tau to avoid gradient explosions.

    Args:
        generator: ConditionalWGANGPGenerator model.
        condition: (B, 20) condition vector.
        mask: Optional (B, 20) observation mask.
        latent_dim: Latent noise vector dimension.
        threshold: Desired diversity margin tau.

    Returns:
        loss: Bounded diversity loss scalar.
    """
    batch_size = condition.size(0)
    device = condition.device

    # Sample two independent noise vectors
    z1 = torch.randn(batch_size, latent_dim, device=device)
    z2 = torch.randn(batch_size, latent_dim, device=device)

    # Generate two composites under the same condition
    img1 = generator(z1, condition, mask)
    img2 = generator(z2, condition, mask)

    # Normalized L1 difference in latent space
    diff_z = torch.mean(torch.abs(z1 - z2), dim=1, keepdim=True) + 1e-6
    # Normalized L1 difference in pixel space
    diff_img = torch.mean(torch.abs(img1 - img2), dim=[1, 2, 3], keepdim=True)

    # Diversity ratio
    diversity_ratio = diff_img / diff_z

    # Bounded penalty: penalizes ratio falling below target threshold
    ms_loss = torch.mean(torch.clamp(threshold - diversity_ratio, min=0.0))
    return ms_loss


def clip_model_gradients(model: nn.Module, max_norm: float = 10.0) -> float:
    """
    Safely clips parameter gradients to prevent explosion in deep critics/generators.
    Returns the total gradient norm before clipping.
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return float(total_norm)


class WGANGPLossManager:
    """
    Orchestrates loss computation, gradient penalties, auxiliary objectives,
    and parameter updates for Conditional WGAN-GP.
    """

    def __init__(self, config: WGANConfig):
        self.config = config

    def compute_critic_loss(
        self,
        critic: nn.Module,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes total critic loss:
          L_D = L_D_wass + lambda_gp * L_gp + lambda_aux * L_aux_real
        """
        # 1. Critic score on real images (with auxiliary attribute prediction)
        d_real, aux_real = critic(real_images, condition, mask, return_aux=True)

        # 2. Critic score on fake images (detached from generator graph)
        d_fake, _ = critic(fake_images.detach(), condition, mask, return_aux=False)

        # 3. Wasserstein loss
        loss_wass = compute_wasserstein_critic_loss(d_real, d_fake)

        # 4. Gradient penalty
        gp, grad_norm = compute_gradient_penalty(
            critic=critic,
            real_images=real_images,
            fake_images=fake_images,
            condition=condition,
            mask=mask,
            lambda_gp=self.config.lambda_gp,
        )

        # 5. Masked auxiliary attribute consistency loss on real images
        loss_aux = torch.zeros((), device=real_images.device)
        if self.config.lambda_aux > 0.0 and aux_real is not None:
            loss_aux = compute_masked_auxiliary_loss(
                aux_logits=aux_real,
                target_condition=condition,
                mask=mask,
            ) * self.config.lambda_aux

        total_d_loss = loss_wass + gp + loss_aux

        return {
            "total_d_loss": total_d_loss,
            "loss_wass": loss_wass,
            "gradient_penalty": gp,
            "gradient_norm": grad_norm,
            "loss_aux": loss_aux,
            "d_real_mean": torch.mean(d_real).detach(),
            "d_fake_mean": torch.mean(d_fake).detach(),
        }

    def compute_generator_loss(
        self,
        generator: nn.Module,
        critic: nn.Module,
        fake_images: torch.Tensor,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        compute_ms: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes total generator loss:
          L_G = L_G_wass + lambda_aux * L_aux_fake + [lambda_ms * L_ms]
        """
        # 1. Critic score on synthesized images
        d_fake, aux_fake = critic(fake_images, condition, mask, return_aux=True)

        # 2. Wasserstein generator loss
        loss_wass = compute_wasserstein_generator_loss(d_fake)

        # 3. Masked auxiliary attribute consistency loss on fake images
        loss_aux = torch.zeros((), device=fake_images.device)
        if self.config.lambda_aux > 0.0 and aux_fake is not None:
            loss_aux = compute_masked_auxiliary_loss(
                aux_logits=aux_fake,
                target_condition=condition,
                mask=mask,
            ) * self.config.lambda_aux

        # 4. Optional bounded mode-seeking diversity loss
        loss_ms = torch.zeros((), device=fake_images.device)
        if compute_ms and self.config.lambda_ms > 0.0:
            loss_ms = compute_bounded_mode_seeking_loss(
                generator=generator,
                condition=condition,
                mask=mask,
                latent_dim=self.config.latent_dim,
                threshold=self.config.ms_target_threshold,
            ) * self.config.lambda_ms

        total_g_loss = loss_wass + loss_aux + loss_ms

        return {
            "total_g_loss": total_g_loss,
            "loss_wass": loss_wass,
            "loss_aux": loss_aux,
            "loss_ms": loss_ms,
        }

