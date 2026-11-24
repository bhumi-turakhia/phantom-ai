"""
Phantom AI 3.0 — Forensic Facial Composite Generator Inference Engine.
Loads frozen Conditional WGAN-GP Generator checkpoint and generates condition-consistent 64x64 facial composites.
Supports reproducible random seeds, custom GCIT condition vectors, partial observation masks, and PIL/tensor outputs.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from config.gan_config import WGANConfig, get_device
from models.wgan_gp_model import ConditionalWGANGPGenerator
from utils.gcit_translator import GCIT_LAYOUT, translate_evidence_to_gcit
from schemas.attributes import EvidenceState, AttributeEvidence, AttributeCategory


DEFAULT_CHECKPOINT = (
    "phantom-ai-ml/checkpoints/cgan_generator_128.pt"
    if os.path.exists("phantom-ai-ml/checkpoints/cgan_generator_128.pt")
    else "phantom-ai-ml/checkpoints/cgan_generator_phase54.pt"
)


CATEGORY_MAP: Dict[str, AttributeCategory] = {
    "face_shape": AttributeCategory.FACE_SHAPE,
    "hair": AttributeCategory.HAIR,
    "hair_color": AttributeCategory.HAIR,
    "hair_style": AttributeCategory.HAIR,
    "hair_length": AttributeCategory.HAIR,
    "hairline": AttributeCategory.HAIRLINE,
    "eyes": AttributeCategory.EYES,
    "eye_color": AttributeCategory.EYES,
    "eye_shape": AttributeCategory.EYES,
    "eyebrows": AttributeCategory.EYEBROWS,
    "eyebrow_shape": AttributeCategory.EYEBROWS,
    "nose": AttributeCategory.NOSE,
    "nose_shape": AttributeCategory.NOSE,
    "lips": AttributeCategory.LIPS,
    "lip_shape": AttributeCategory.LIPS,
    "facial_hair": AttributeCategory.FACIAL_HAIR,
    "distinguishing_marks": AttributeCategory.DISTINGUISHING_MARKS,
    "other_observable": AttributeCategory.OTHER_OBSERVABLE,
    "accessories": AttributeCategory.OTHER_OBSERVABLE,
}


def normalize_category(cat_str: str) -> AttributeCategory:
    clean = str(cat_str).lower().strip().replace(" ", "_")
    return CATEGORY_MAP.get(clean, AttributeCategory.OTHER_OBSERVABLE)


def load_generator(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    device: Optional[torch.device] = None,
) -> ConditionalWGANGPGenerator:
    """
    Loads and freezes ConditionalWGANGPGenerator from a saved checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Generator checkpoint not found at: {checkpoint_path}")

    target_device = device or get_device("auto")
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)

    config_dict = checkpoint.get("config", {})
    if isinstance(config_dict, dict) and config_dict:
        config = WGANConfig.from_dict(config_dict)
    else:
        config = WGANConfig()

    generator = ConditionalWGANGPGenerator(
        latent_dim=config.latent_dim,
        condition_dim=config.condition_dim,
        condition_embed_dim=config.condition_embed_dim,
        feature_maps=config.generator_features,
        channels=config.channels,
        image_size=config.image_size,
    ).to(target_device)

    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    for param in generator.parameters():
        param.requires_grad = False

    return generator


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    Converts a normalized (3, H, W) tensor in [-1.0, 1.0] to a PIL Image in RGB [0, 255].
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    img_np = image_tensor.detach().cpu().numpy()
    # Normalize from [-1.0, 1.0] to [0, 255]
    img_np = np.clip((img_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
    img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, 3)
    return Image.fromarray(img_np, mode="RGB")


def generate_forensic_composite(
    condition_vector: Optional[Union[torch.Tensor, List[float], np.ndarray]] = None,
    condition_mask: Optional[Union[torch.Tensor, List[float], np.ndarray]] = None,
    forensic_profile: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    device: Optional[torch.device] = None,
    generator_model: Optional[ConditionalWGANGPGenerator] = None,
) -> Tuple[Image.Image, torch.Tensor]:
    """
    Generates a single facial composite from GCIT conditions or a forensic evidence profile.

    Args:
        condition_vector: (20,) vector in [-1.0, 1.0].
        condition_mask: (20,) binary mask in {0.0, 1.0}.
        forensic_profile: Optional dict of attribute evidence states to be translated.
        seed: Optional integer seed for deterministic latent noise generation.
        checkpoint_path: Path to generator weights.
        device: PyTorch device.
        generator_model: Optional pre-loaded generator instance.

    Returns:
        pil_image: 64x64 RGB PIL Image.
        tensor_image: (1, 3, 64, 64) normalized PyTorch tensor in [-1.0, 1.0].
    """
    target_device = device or get_device("auto")

    # Load generator if not provided
    if generator_model is None:
        generator = load_generator(checkpoint_path, target_device)
    else:
        generator = generator_model.to(target_device)
        generator.eval()

    # Determine condition and mask
    if forensic_profile is not None:
        typed_profile = {}
        for k, v in forensic_profile.items():
            norm_cat = normalize_category(k)
            if isinstance(v, AttributeEvidence):
                typed_profile[norm_cat.value] = v
            elif isinstance(v, dict):
                typed_profile[norm_cat.value] = AttributeEvidence(
                    category=norm_cat,
                    value=v.get("value"),
                    state=EvidenceState(v.get("state", "explicit")),
                    source_text=v.get("source_text", ""),
                )
            elif isinstance(v, str):
                typed_profile[norm_cat.value] = AttributeEvidence(
                    category=norm_cat,
                    value=v,
                    state=EvidenceState.EXPLICIT,
                    source_text=v,
                )
        c_tensor, m_tensor, _ = translate_evidence_to_gcit(typed_profile)
        c_tensor = c_tensor.unsqueeze(0).to(target_device)
        m_tensor = m_tensor.unsqueeze(0).to(target_device)
    elif condition_vector is not None:
        if isinstance(condition_vector, (list, np.ndarray)):
            c_tensor = torch.tensor(condition_vector, dtype=torch.float32)
        else:
            c_tensor = condition_vector.clone().detach().float()
        if c_tensor.dim() == 1:
            c_tensor = c_tensor.unsqueeze(0)
        c_tensor = c_tensor.to(target_device)

        if condition_mask is not None:
            if isinstance(condition_mask, (list, np.ndarray)):
                m_tensor = torch.tensor(condition_mask, dtype=torch.float32)
            else:
                m_tensor = condition_mask.clone().detach().float()
            if m_tensor.dim() == 1:
                m_tensor = m_tensor.unsqueeze(0)
            m_tensor = m_tensor.to(target_device)
        else:
            m_tensor = torch.ones_like(c_tensor)
    else:
        # Unconditioned default: all 0.0 with 0.0 mask
        c_tensor = torch.zeros((1, 20), dtype=torch.float32, device=target_device)
        m_tensor = torch.zeros((1, 20), dtype=torch.float32, device=target_device)

    # Sample latent noise
    latent_dim = generator.latent_dim
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        z = torch.randn(1, latent_dim, generator=gen, dtype=torch.float32).to(target_device)
    else:
        z = torch.randn(1, latent_dim, device=target_device, dtype=torch.float32)

    with torch.no_grad():
        fake_tensor = generator(z, c_tensor, m_tensor)  # (1, 3, 64, 64)

    pil_img = tensor_to_pil(fake_tensor)
    return pil_img, fake_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate facial composite from GCIT conditions.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Generator checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for latent noise")
    parser.add_argument("--output", type=str, default="composite_output.png", help="Output image file path")
    parser.add_argument("--profile-json", type=str, default=None, help="Path to JSON file with forensic profile")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cpu', 'mps')")
    parser.add_argument("--scale", type=int, default=4, help="Display upscaling factor (e.g. 4 for 256x256)")
    args = parser.parse_args()

    dev = get_device(args.device)
    profile = None
    if args.profile_json and os.path.exists(args.profile_json):
        with open(args.profile_json, "r", encoding="utf-8") as f:
            profile = json.load(f)

    pil_img, _ = generate_forensic_composite(
        forensic_profile=profile,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        device=dev,
    )

    if args.scale > 1:
        new_size = (pil_img.width * args.scale, pil_img.height * args.scale)
        pil_img = pil_img.resize(new_size, Image.Resampling.NEAREST)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    pil_img.save(args.output)
    print(f"Generated facial composite saved to: {args.output}")


if __name__ == "__main__":
    main()
