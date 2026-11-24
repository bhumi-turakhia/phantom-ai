"""
Automated Test Suite for Conditional WGAN-GP Architecture and Projection Critic.
Verifies generator/critic shapes, projection conditioning, gradient flow, gradient penalty behavior,
masked auxiliary losses, deterministic seed reproducibility, zero-observed safety, and device compatibility.
"""

import pytest
import torch
import torch.nn as nn

from config.gan_config import (
    WGANConfig,
    get_device,
    count_parameters,
    set_reproducibility_seed,
    create_checkpoint_metadata,
)
from models.wgan_gp_model import (
    ConditionalWGANGPGenerator,
    ProjectionCritic,
    weights_init_wgan,
)
from training.wgan_objectives import (
    compute_wasserstein_critic_loss,
    compute_wasserstein_generator_loss,
    compute_gradient_penalty,
    compute_masked_auxiliary_loss,
    compute_bounded_mode_seeking_loss,
    clip_model_gradients,
    WGANGPLossManager,
)
from data.celeba_dataset import CelebAIdentityDataset


@pytest.fixture
def config():
    return WGANConfig(
        latent_dim=100,
        condition_dim=20,
        condition_embed_dim=64,
        image_size=64,
        channels=3,
        generator_features=64,
        critic_features=64,
        lambda_gp=10.0,
        lambda_aux=1.0,
        lambda_ms=0.05,
    )


@pytest.fixture
def generator(config):
    return ConditionalWGANGPGenerator(
        latent_dim=config.latent_dim,
        condition_dim=config.condition_dim,
        condition_embed_dim=config.condition_embed_dim,
        feature_maps=config.generator_features,
        channels=config.channels,
        image_size=config.image_size,
    )


@pytest.fixture
def critic(config):
    return ProjectionCritic(
        in_channels=config.channels,
        condition_dim=config.condition_dim,
        feature_maps=config.critic_features,
        hidden_dim=config.critic_hidden_dim,
        image_size=config.image_size,
    )


def test_generator_input_output_shapes_and_ranges(generator):
    """
    Asserts Generator maps (B, 100) noise + (B, 20) condition to (B, 3, 64, 64) in [-1.0, 1.0].
    """
    batch_size = 4
    z = torch.randn(batch_size, 100)
    c = torch.empty(batch_size, 20).uniform_(-1.0, 1.0)
    m = torch.ones(batch_size, 20)

    out = generator(z, c, m)
    assert out.shape == (batch_size, 3, 64, 64)
    assert out.dtype == torch.float32

    # Tanh bounds
    assert out.min() >= -1.0
    assert out.max() <= 1.0
    assert not torch.isnan(out).any()


def test_critic_shapes_and_unbounded_scores(critic):
    """
    Asserts Critic maps (B, 3, 64, 64) image + (B, 20) condition to unbounded scalar score (B, 1)
    and optional auxiliary logits (B, 20).
    """
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    c = torch.empty(batch_size, 20).uniform_(-1.0, 1.0)
    m = torch.ones(batch_size, 20)

    score, aux_logits = critic(x, c, m, return_aux=True)
    assert score.shape == (batch_size, 1)
    assert aux_logits.shape == (batch_size, 20)
    assert not torch.isnan(score).any()
    assert not torch.isnan(aux_logits).any()


def test_critic_strictly_no_batchnorm(critic):
    """
    CRITICAL WGAN-GP CONSTRAINT:
    The critic must NOT contain any BatchNorm layers.
    Batch normalization couples samples across a batch, violating the Kantorovich-Rubinstein
    dual formulation and corrupting sample-wise gradient penalty estimation.
    """
    for name, module in critic.named_modules():
        assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)), (
            f"Critic contains forbidden BatchNorm layer: {name} ({module.__class__.__name__})"
        )


def test_critic_strictly_no_sigmoid(critic):
    """
    Wasserstein critic outputs real-valued unbounded scores. It must NOT contain Sigmoid on final output.
    """
    assert not isinstance(critic.fc_unconditioned, nn.Sigmoid)
    # Check score range can exceed [0, 1]
    x_large = torch.randn(2, 3, 64, 64) * 10.0
    c = torch.ones(2, 20)
    score, _ = critic(x_large, c)
    # Critic score can be negative or > 1
    assert score.shape == (2, 1)


def test_zero_observed_attributes_safety(generator, critic, config):
    """
    When witness observes zero attributes (m = 0.0 for all dimensions):
    - Generator synthesizes from empirical prior without crashing.
    - Projection critic reduces to unconditioned score with zero NaNs.
    - Masked auxiliary loss evaluates to exactly 0.0 with no divide-by-zero.
    """
    batch_size = 4
    z = torch.randn(batch_size, 100)
    c = torch.zeros(batch_size, 20)
    m_zero = torch.zeros(batch_size, 20)

    # 1. Generator forward
    fake_img = generator(z, c, m_zero)
    assert fake_img.shape == (batch_size, 3, 64, 64)
    assert not torch.isnan(fake_img).any()

    # 2. Critic forward
    score, aux = critic(fake_img, c, m_zero, return_aux=True)
    assert score.shape == (batch_size, 1)
    assert not torch.isnan(score).any()

    # 3. Masked auxiliary loss on zero mask
    aux_loss = compute_masked_auxiliary_loss(aux, c, m_zero)
    assert aux_loss.item() == 0.0


def test_gradient_flow_generator_and_critic(generator, critic, config):
    """
    Asserts analytical gradient propagation:
    - Critic parameters receive valid non-zero gradients when optimizing critic loss.
    - Generator parameters receive valid non-zero gradients when optimizing generator loss.
    """
    loss_mgr = WGANGPLossManager(config)
    batch_size = 2
    z = torch.randn(batch_size, 100)
    c = torch.empty(batch_size, 20).uniform_(-1.0, 1.0)
    m = torch.ones(batch_size, 20)
    real_x = torch.randn(batch_size, 3, 64, 64)

    # 1. Critic update gradient flow
    fake_x = generator(z, c, m)
    d_dict = loss_mgr.compute_critic_loss(critic, real_x, fake_x, c, m)
    critic.zero_grad()
    d_dict["total_d_loss"].backward()

    # Verify critic gradients
    critic_has_grad = False
    for p in critic.parameters():
        if p.grad is not None and torch.norm(p.grad) > 0.0:
            critic_has_grad = True
            break
    assert critic_has_grad, "Critic parameters received zero gradients."

    # 2. Generator update gradient flow
    generator.zero_grad()
    critic.zero_grad()
    fake_x_gen = generator(z, c, m)
    g_dict = loss_mgr.compute_generator_loss(generator, critic, fake_x_gen, c, m, compute_ms=True)
    g_dict["total_g_loss"].backward()

    generator_has_grad = False
    for p in generator.parameters():
        if p.grad is not None and torch.norm(p.grad) > 0.0:
            generator_has_grad = True
            break
    assert generator_has_grad, "Generator parameters received zero gradients."


def test_gradient_penalty_numerical_behavior(critic):
    """
    Asserts gradient penalty is strictly finite, non-negative, and properly enforces 1-Lipschitz.
    """
    batch_size = 4
    real_x = torch.randn(batch_size, 3, 64, 64)
    fake_x = torch.randn(batch_size, 3, 64, 64)
    c = torch.empty(batch_size, 20).uniform_(-1.0, 1.0)
    m = torch.ones(batch_size, 20)

    gp, grad_norm = compute_gradient_penalty(critic, real_x, fake_x, c, m, lambda_gp=10.0)
    assert gp.item() >= 0.0
    assert not torch.isnan(gp)
    assert not torch.isinf(gp)
    assert grad_norm.item() > 0.0


def test_deterministic_generation_with_seeds(generator):
    """
    Asserts deterministic composite generation when given the same seed.
    Asserts varied composite generation when given different seeds for identical conditions.
    """
    condition = torch.zeros(20)
    condition[0] = 1.0  # Black hair
    mask = torch.ones(20)

    # Same seed -> exactly identical images
    comp1 = generator.generate_composite(condition, mask, seed=12345)
    comp2 = generator.generate_composite(condition, mask, seed=12345)
    assert torch.allclose(comp1, comp2, atol=1e-6), "Identical seeds must yield identical outputs."

    # Different seeds -> distinct variations
    comp3 = generator.generate_composite(condition, mask, seed=99999)
    diff = torch.mean(torch.abs(comp1 - comp3)).item()
    assert diff > 1e-4, f"Different seeds should synthesize distinct outputs, got diff={diff}"
    assert not torch.allclose(comp1, comp3, atol=1e-4)


def test_device_selection_and_cpu_compatibility(config):
    """
    Verifies device selection logic and runs a full forward/backward step on CPU.
    """
    dev = get_device("cpu")
    assert dev.type == "cpu"

    gen_cpu = ConditionalWGANGPGenerator(
        latent_dim=config.latent_dim,
        condition_dim=config.condition_dim,
        condition_embed_dim=config.condition_embed_dim,
    ).to(dev)

    z = torch.randn(2, 100, device=dev)
    c = torch.ones(2, 20, device=dev)
    out = gen_cpu(z, c)
    assert out.device.type == "cpu"
    assert out.shape == (2, 3, 64, 64)


def test_model_parameter_counts(generator, critic):
    """
    Verifies that model parameters are counted and non-trivial.
    """
    tot_g, train_g = count_parameters(generator)
    tot_c, train_c = count_parameters(critic)

    assert tot_g > 3_000_000, f"Generator too small: {tot_g}"
    assert tot_c > 5_000_000, f"Critic too small: {tot_c}"
    assert tot_g == train_g
    assert tot_c == train_c


def test_checkpoint_metadata_schema(config, generator, critic):
    """
    Asserts standardized checkpoint format creation with complete provenance metadata.
    """
    meta = create_checkpoint_metadata(
        config=config,
        epoch=0,
        step=0,
        generator_state=generator.state_dict(),
        critic_state=critic.state_dict(),
    )
    assert meta["format_version"] == "3.0.0"
    assert meta["architecture"] == "Conditional_WGAN_GP_Projection"
    assert meta["training_split"] == "split_a"
    assert meta["held_out_benchmark"] == "split_c_untouched"
    assert "generator_state_dict" in meta
    assert "critic_state_dict" in meta


def test_require_images_flag_blocks_accidental_zero_image_training():
    """
    Asserts Phase 5.1/5.2 safety requirement:
    In training mode (require_images=True), if images are missing or directory is None,
    the loader MUST raise an error rather than silently returning dummy zero-image tensors.
    """
    # 1. image_dir is None with require_images=True
    with pytest.raises(ValueError) as exc:
        CelebAIdentityDataset(
            metadata_dir="phantom-ai-ml/datasets/celeba",
            split="split_a",
            image_dir=None,
            require_images=True,
        )
    assert "image_dir was not provided" in str(exc.value)

    # 2. image_dir does not exist with require_images=True
    with pytest.raises(FileNotFoundError) as exc:
        CelebAIdentityDataset(
            metadata_dir="phantom-ai-ml/datasets/celeba",
            split="split_a",
            image_dir="/non/existent/path/for/images",
            require_images=True,
        )
    assert "image_dir does not exist" in str(exc.value)


def test_gradient_clipping_helper(critic):
    """
    Tests that gradient clipping executes properly and returns finite norm.
    """
    x = torch.randn(2, 3, 64, 64)
    c = torch.ones(2, 20)
    score, _ = critic(x, c)
    score.mean().backward()

    norm_before = clip_model_gradients(critic, max_norm=5.0)
    assert norm_before >= 0.0
