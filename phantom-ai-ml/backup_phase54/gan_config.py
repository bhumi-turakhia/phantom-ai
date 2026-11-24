"""
Centralized Configuration for Phantom AI Conditional WGAN-GP Architecture.
Specifies model dimensions, conditioning settings, optimization parameters, device selection,
and checkpoint format definitions.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


@dataclass
class WGANConfig:
    # Latent and Conditioning Dimensions
    latent_dim: int = 100               # Dimension of Gaussian noise vector z ~ N(0, I)
    condition_dim: int = 20            # 20-dimensional GCIT layout
    condition_embed_dim: int = 64      # Linear projection dimension for c * m

    # Image Resolution & Channels
    image_size: int = 64               # 64x64 baseline resolution
    channels: int = 3                  # RGB channels
    output_range: Tuple[float, float] = (-1.0, 1.0)

    # Architectural Feature Map Scales
    generator_features: int = 64       # Base feature map multiplier for G
    critic_features: int = 64          # Base feature map multiplier for D
    critic_hidden_dim: int = 512       # Dimension of penultimate feature vector phi(x)

    # Loss Objective Weights
    lambda_gp: float = 10.0            # WGAN-GP 1-Lipschitz penalty weight
    lambda_aux: float = 1.0            # Masked auxiliary attribute consistency loss weight
    lambda_ms: float = 0.05            # Bounded mode-seeking diversity loss weight
    ms_target_threshold: float = 0.5   # Diversity margin tau for mode-seeking

    # Optimizer Hyperparameters (Heusel et al. / TTUR / Gulrajani et al.)
    lr_g: float = 1e-4                 # Generator learning rate
    lr_d: float = 1e-4                 # Critic learning rate (TTUR: 1e-4 or 2e-4)
    beta1: float = 0.0                 # Standard Adam beta1 for WGAN-GP (Gulrajani et al.)
    beta2: float = 0.9                 # Standard Adam beta2 for WGAN-GP
    critic_steps_per_g_step: int = 5   # Critic iterations per generator update (n_critic)
    grad_clip_norm: float = 10.0       # Optional gradient norm clipping threshold

    # Runtime and Hardware
    device: str = "auto"               # 'auto', 'mps', 'cpu', or 'cuda'
    seed: int = 42                     # Default global reproducibility seed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WGANConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def get_device(requested: str = "auto") -> torch.device:
    """
    Selects the optimal hardware device:
    - If Apple Silicon MPS is available and requested != 'cpu', select 'mps'.
    - If CUDA is available, select 'cuda'.
    - Otherwise, falls back to 'cpu'.
    """
    if requested == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    elif requested == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")
    elif requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    else:
        return torch.device("cpu")


def set_reproducibility_seed(seed: int = 42) -> None:
    """
    Sets deterministic seeds across torch, numpy, and python random.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Returns (total_parameters, trainable_parameters).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def create_checkpoint_metadata(
    config: WGANConfig,
    epoch: int,
    step: int,
    generator_state: Dict[str, Any],
    critic_state: Dict[str, Any],
    g_optimizer_state: Optional[Dict[str, Any]] = None,
    d_optimizer_state: Optional[Dict[str, Any]] = None,
    evaluator_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Standardized, self-describing checkpoint dictionary schema for Phantom AI.
    """
    import time
    return {
        "format_version": "3.0.0",
        "architecture": "Conditional_WGAN_GP_Projection",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epoch": epoch,
        "step": step,
        "config": config.to_dict(),
        "generator_state_dict": generator_state,
        "critic_state_dict": critic_state,
        "g_optimizer_state_dict": g_optimizer_state,
        "d_optimizer_state_dict": d_optimizer_state,
        "evaluator_metrics": evaluator_metrics or {},
        "training_split": "split_a",
        "evaluator_split": "split_b",
        "held_out_benchmark": "split_c_untouched",
    }

