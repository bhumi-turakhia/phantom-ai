"""
Phantom AI 3.0 — Phase 5.4: Conditional WGAN-GP Training & Generative Evaluation Engine.
Trains ConditionalWGANGPGenerator and ProjectionCritic on CelebA Split A1 (Generator Training).
Validates on Split A2 (Generator Validation) and selects best checkpoint.
Evaluates final frozen generator on untouched Split C (Held-out Benchmark) using frozen ForensicFeatureCNN.
"""

import os
import sys
import json
import time
import math
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
import numpy as np
from PIL import Image

from config.gan_config import WGANConfig, get_device, set_reproducibility_seed, count_parameters
from data.celeba_dataset import CelebAIdentityDataset
from models.wgan_gp_model import (
    ConditionalWGANGPGenerator,
    ProjectionCritic,
    weights_init_wgan,
)
from models.forensic_evaluator import ForensicFeatureCNN
from training.wgan_objectives import WGANGPLossManager, clip_model_gradients
from utils.generative_eval_metrics import (
    extract_evaluator_features,
    compute_cfd_from_features,
    compute_fav_metrics,
    compute_seed_diversity,
)
from utils.gcit_translator import GCIT_LAYOUT


EXPECTED_EVALUATOR_SHA256 = "f52ae5a8df4ae574432f353fd77fc605f6dcf758d67d492b447c139a79e78834"
DEFAULT_EVALUATOR_PATH = "phantom-ai-ml/checkpoints/forensic_evaluator_split_b.pt"


@dataclass
class GeneratorTrainingConfig:
    # Training Hyperparameters
    batch_size: int = 64
    epochs: int = 5
    n_critic: int = 5
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    beta1: float = 0.0
    beta2: float = 0.9
    weight_decay: float = 1e-4
    attribute_dropout: float = 0.20
    lambda_gp: float = 10.0
    lambda_aux: float = 1.0
    lambda_ms: float = 0.05
    ms_threshold: float = 0.5
    grad_clip_norm: float = 10.0

    # Dataset & Hardware
    device: str = "auto"
    seed: int = 42
    metadata_dir: str = "phantom-ai-ml/datasets/celeba"
    image_dir: str = "phantom-ai-ml/datasets/celeba/img_align_celeba"
    evaluator_checkpoint: str = DEFAULT_EVALUATOR_PATH
    output_checkpoint_dir: str = "phantom-ai-ml/checkpoints"
    output_viz_dir: str = "phantom-ai-ml/visualizations"
    results_json: str = "phantom-ai-ml/datasets/celeba/cgan_training_and_evaluation_results.json"
    generator_checkpoint_path: str = "phantom-ai-ml/checkpoints/cgan_generator_phase54.pt"
    critic_checkpoint_path: str = "phantom-ai-ml/checkpoints/cgan_critic_phase54.pt"

    # Evaluation Subsampling
    max_train_steps_per_epoch: Optional[int] = 150  # Set to control training duration per epoch
    val_samples: int = 512
    final_test_samples: int = 1024


def get_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def create_fixed_evaluation_profiles(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Creates 16 deterministic, forensically diverse condition profiles for tracking visual evolution.
    """
    c = torch.zeros((16, 20), dtype=torch.float32)
    m = torch.zeros((16, 20), dtype=torch.float32)
    labels = []

    def set_attr(row: int, name: str, val: float):
        idx = GCIT_LAYOUT[name]
        c[row, idx] = val
        m[row, idx] = 1.0

    # Profile 0: Black wavy hair, high cheekbones, no beard
    set_attr(0, "hair_black", 1.0)
    set_attr(0, "hair_wavy", 1.0)
    set_attr(0, "face_cheekbones", 1.0)
    set_attr(0, "facial_no_beard", 1.0)
    labels.append("Profile 0: Black Wavy Hair, High Cheekbones, No Beard")

    # Profile 1: Gray hair, receding hairline, mustache, glasses
    set_attr(1, "hair_gray", 1.0)
    set_attr(1, "hairline_receding", 1.0)
    set_attr(1, "facial_mustache", 1.0)
    set_attr(1, "glasses", 1.0)
    labels.append("Profile 1: Gray Hair, Receding Hairline, Mustache, Glasses")

    # Profile 2: Blond hair, goatee, big nose
    set_attr(2, "hair_blond", 1.0)
    set_attr(2, "facial_goatee", 1.0)
    set_attr(2, "nose_big", 1.0)
    labels.append("Profile 2: Blond Hair, Goatee, Big Nose")

    # Profile 3: Brown straight hair, bangs, big lips
    set_attr(3, "hair_brown", 1.0)
    set_attr(3, "hair_straight", 1.0)
    set_attr(3, "hair_bangs", 1.0)
    set_attr(3, "lips_big", 1.0)
    labels.append("Profile 3: Brown Straight Hair, Bangs, Big Lips")

    # Profile 4: Bald, 5 o'clock shadow stubble, high cheekbones
    set_attr(4, "hair_bald", 1.0)
    set_attr(4, "facial_stubble", 1.0)
    set_attr(4, "face_cheekbones", 1.0)
    labels.append("Profile 4: Bald, Stubble, High Cheekbones")

    # Profile 5: Black hair, oval face, pointy nose
    set_attr(5, "hair_black", 1.0)
    set_attr(5, "face_oval", 1.0)
    set_attr(5, "nose_pointy", 1.0)
    labels.append("Profile 5: Black Hair, Oval Face, Pointy Nose")

    # Profile 6: Brown wavy hair, sideburns, no mustache
    set_attr(6, "hair_brown", 1.0)
    set_attr(6, "hair_wavy", 1.0)
    set_attr(6, "facial_sideburns", 1.0)
    set_attr(6, "facial_mustache", -1.0)
    labels.append("Profile 6: Brown Wavy Hair, Sideburns, No Mustache")

    # Profile 7: Gray hair, glasses, oval face
    set_attr(7, "hair_gray", 1.0)
    set_attr(7, "glasses", 1.0)
    set_attr(7, "face_oval", 1.0)
    labels.append("Profile 7: Gray Hair, Glasses, Oval Face")

    # Profile 8: Blond hair, mustache, glasses
    set_attr(8, "hair_blond", 1.0)
    set_attr(8, "facial_mustache", 1.0)
    set_attr(8, "glasses", 1.0)
    labels.append("Profile 8: Blond Hair, Mustache, Glasses")

    # Profile 9: Blond wavy hair, high cheekbones, big lips
    set_attr(9, "hair_blond", 1.0)
    set_attr(9, "hair_wavy", 1.0)
    set_attr(9, "face_cheekbones", 1.0)
    set_attr(9, "lips_big", 1.0)
    labels.append("Profile 9: Blond Wavy Hair, High Cheekbones, Big Lips")

    # Profile 10: Black straight hair, goatee, big nose
    set_attr(10, "hair_black", 1.0)
    set_attr(10, "hair_straight", 1.0)
    set_attr(10, "facial_goatee", 1.0)
    set_attr(10, "nose_big", 1.0)
    labels.append("Profile 10: Black Straight Hair, Goatee, Big Nose")

    # Profile 11: Bald head, glasses, mustache
    set_attr(11, "hair_bald", 1.0)
    set_attr(11, "glasses", 1.0)
    set_attr(11, "facial_mustache", 1.0)
    labels.append("Profile 11: Bald Head, Glasses, Mustache")

    # Profile 12: Brown hair, bangs, glasses
    set_attr(12, "hair_brown", 1.0)
    set_attr(12, "hair_bangs", 1.0)
    set_attr(12, "glasses", 1.0)
    labels.append("Profile 12: Brown Hair, Bangs, Glasses")

    # Profile 13: Black hair, stubble, oval face
    set_attr(13, "hair_black", 1.0)
    set_attr(13, "facial_stubble", 1.0)
    set_attr(13, "face_oval", 1.0)
    labels.append("Profile 13: Black Hair, Stubble, Oval Face")

    # Profile 14: Blond straight hair, pointy nose
    set_attr(14, "hair_blond", 1.0)
    set_attr(14, "hair_straight", 1.0)
    set_attr(14, "nose_pointy", 1.0)
    labels.append("Profile 14: Blond Straight Hair, Pointy Nose")

    # Profile 15: Unconditioned baseline (all attributes unobserved: mask = 0.0)
    # c is all 0.0, m is all 0.0
    labels.append("Profile 15: Unconditioned Baseline (Mask = 0.0)")

    return c.to(device), m.to(device), labels


def save_grid_image(
    images_tensor: torch.Tensor,
    output_path: str,
    rows: int = 4,
    cols: int = 4,
    scale: int = 2,
) -> None:
    """
    Saves a batch of N images as a tiled PNG image grid.
    """
    n = images_tensor.shape[0]
    # Rescale from [-1.0, 1.0] to [0, 255]
    imgs_np = np.clip((images_tensor.detach().cpu().numpy() + 1.0) * 127.5, 0, 255).astype(np.uint8)
    h, w = imgs_np.shape[2], imgs_np.shape[3]

    grid_h = rows * h
    grid_w = cols * w
    grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for idx in range(min(n, rows * cols)):
        r = idx // cols
        c = idx % cols
        img = np.transpose(imgs_np[idx], (1, 2, 0))  # (H, W, 3)
        grid_img[r * h : (r + 1) * h, c * w : (c + 1) * w] = img

    pil_grid = Image.fromarray(grid_img, mode="RGB")
    if scale > 1:
        pil_grid = pil_grid.resize((grid_w * scale, grid_h * scale), Image.Resampling.NEAREST)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pil_grid.save(output_path)


def train_conditional_wgan_gp(
    config: Optional[GeneratorTrainingConfig] = None,
) -> Dict[str, Any]:
    cfg = config or GeneratorTrainingConfig()
    set_reproducibility_seed(cfg.seed)
    device = get_device(cfg.device)

    print("=" * 80)
    print("PHANTOM AI 3.0 — PHASE 5.4: CONDITIONAL WGAN-GP TRAINING & GENERATIVE EVALUATION")
    print(f"Device: {device} | Epochs: {cfg.epochs} | Batch Size: {cfg.batch_size} | n_critic: {cfg.n_critic}")
    print(f"Generator LR: {cfg.lr_g} | Critic LR: {cfg.lr_d} | Attribute Dropout: {cfg.attribute_dropout}")
    print(f"lambda_gp: {cfg.lambda_gp} | lambda_aux: {cfg.lambda_aux} | lambda_ms: {cfg.lambda_ms}")
    print("=" * 80)

    # 1. Pre-training verification: Evaluator Checkpoint Hash
    if not os.path.exists(cfg.evaluator_checkpoint):
        raise FileNotFoundError(f"Frozen evaluator checkpoint missing: {cfg.evaluator_checkpoint}")

    actual_eval_hash = get_file_sha256(cfg.evaluator_checkpoint)
    print(f"Evaluator Checkpoint SHA-256: {actual_eval_hash}")
    if actual_eval_hash != EXPECTED_EVALUATOR_SHA256:
        raise ValueError(
            f"Evaluator checkpoint hash mismatch!\nExpected: {EXPECTED_EVALUATOR_SHA256}\nActual:   {actual_eval_hash}"
        )
    print("[PASS] Evaluator checkpoint cryptographic integrity verified.")

    # 2. Pre-training verification: Dataset & Manifests
    gen_manifest_path = os.path.join(cfg.metadata_dir, "generator_split_a_manifest.json")
    if not os.path.exists(gen_manifest_path):
        raise FileNotFoundError(f"Generator split manifest missing at {gen_manifest_path}")

    with open(gen_manifest_path, "r", encoding="utf-8") as f:
        gen_manifest = json.load(f)
    print(
        f"[PASS] Split A Manifest: A1={gen_manifest['partitions']['split_a1']['image_count']} imgs, "
        f"A2={gen_manifest['partitions']['split_a2']['image_count']} imgs."
    )

    # 3. Load Frozen Independent Evaluator
    print("\nLoading Frozen Independent Evaluator (ForensicFeatureCNN)...")
    evaluator = ForensicFeatureCNN().to(device)
    eval_state = torch.load(cfg.evaluator_checkpoint, map_location=device, weights_only=False)
    evaluator.load_state_dict(eval_state["model_state_dict"])
    evaluator.eval()
    for param in evaluator.parameters():
        param.requires_grad = False
    print(f"[PASS] Evaluator loaded in eval() mode with zero gradients ({count_parameters(evaluator)[0]:,} params).")

    # 4. Initialize Generator and Critic Models
    print("\nInitializing Conditional WGAN-GP Architecture...")
    wgan_cfg = WGANConfig(
        latent_dim=100,
        condition_dim=20,
        condition_embed_dim=64,
        image_size=64,
        channels=3,
        generator_features=64,
        critic_features=64,
        critic_hidden_dim=512,
        lambda_gp=cfg.lambda_gp,
        lambda_aux=cfg.lambda_aux,
        lambda_ms=cfg.lambda_ms,
        ms_target_threshold=cfg.ms_threshold,
    )
    generator = ConditionalWGANGPGenerator(
        latent_dim=wgan_cfg.latent_dim,
        condition_dim=wgan_cfg.condition_dim,
        condition_embed_dim=wgan_cfg.condition_embed_dim,
        feature_maps=wgan_cfg.generator_features,
        channels=wgan_cfg.channels,
        image_size=wgan_cfg.image_size,
    ).to(device)

    critic = ProjectionCritic(
        in_channels=wgan_cfg.channels,
        condition_dim=wgan_cfg.condition_dim,
        feature_maps=wgan_cfg.critic_features,
        hidden_dim=wgan_cfg.critic_hidden_dim,
        image_size=wgan_cfg.image_size,
    ).to(device)

    generator.apply(weights_init_wgan)
    critic.apply(weights_init_wgan)
    loss_manager = WGANGPLossManager(wgan_cfg)

    g_total, g_trainable = count_parameters(generator)
    d_total, d_trainable = count_parameters(critic)
    print(f"[PASS] Generator initialized: {g_total:,} parameters.")
    print(f"[PASS] Critic initialized: {d_total:,} parameters.")

    # 5. Optimizers
    opt_G = AdamW(generator.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    opt_D = AdamW(critic.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

    # 6. Datasets and Loaders
    print("\nPreparing Identity-Disjoint Datasets...")
    train_dataset = CelebAIdentityDataset(
        metadata_dir=cfg.metadata_dir,
        split="split_a1",
        image_dir=cfg.image_dir,
        attribute_dropout=cfg.attribute_dropout,
        require_images=True,
    )
    val_dataset = CelebAIdentityDataset(
        metadata_dir=cfg.metadata_dir,
        split="split_a2",
        image_dir=cfg.image_dir,
        attribute_dropout=0.0,
        require_images=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    val_indices = list(range(min(cfg.val_samples, len(val_dataset))))
    val_subset = Subset(val_dataset, val_indices)
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    print(f"[PASS] Split A1 Train Dataset: {len(train_dataset):,} samples.")
    print(f"[PASS] Split A2 Validation Subset: {len(val_subset):,} samples.")

    # Pre-extract real validation features from Split A2 for fast C-FD calculation
    print("Pre-extracting frozen evaluator features for Split A2 validation...")
    val_real_images = []
    val_conditions = []
    val_masks = []
    for b in val_loader:
        val_real_images.append(b["image"])
        val_conditions.append(b["condition"])
        val_masks.append(b["mask"])
    val_real_images_tensor = torch.cat(val_real_images, dim=0)
    val_conditions_tensor = torch.cat(val_conditions, dim=0)
    val_masks_tensor = torch.cat(val_masks, dim=0)

    val_real_features = extract_evaluator_features(
        evaluator=evaluator,
        images=val_real_images_tensor,
        batch_size=cfg.batch_size,
        device=str(device),
    )
    print(f"[PASS] Split A2 real features extracted: {val_real_features.shape}.")

    # Fixed evaluation noise and conditions for longitudinal grid tracking
    fixed_cond, fixed_mask, fixed_labels = create_fixed_evaluation_profiles(device)
    fixed_gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    fixed_z = torch.randn(16, 100, generator=fixed_gen, dtype=torch.float32).to(device)

    # Initial baseline grid (epoch 0, untrained random initialization)
    generator.eval()
    with torch.no_grad():
        init_grid_imgs = generator(fixed_z, fixed_cond, fixed_mask)
    save_grid_image(
        init_grid_imgs,
        os.path.join(cfg.output_viz_dir, "cgan_phase54_epoch_0_untrained.png"),
    )
    print(f"[INFO] Untrained baseline image grid saved to {cfg.output_viz_dir}/cgan_phase54_epoch_0_untrained.png")

    # 7. Training Loop
    history: List[Dict[str, Any]] = []
    best_composite_score = float("inf")  # C-FD - 100 * FAV_F1 (lower is better)
    best_epoch = -1
    best_val_metrics: Dict[str, Any] = {}
    best_gen_weights: Optional[Dict[str, Any]] = None
    best_crit_weights: Optional[Dict[str, Any]] = None

    total_training_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        generator.train()
        critic.train()

        running_d_loss = 0.0
        running_g_loss = 0.0
        running_w_dist = 0.0
        running_gp = 0.0
        running_d_aux = 0.0
        running_g_aux = 0.0
        running_g_ms = 0.0
        running_d_grad_norm = 0.0
        running_g_grad_norm = 0.0

        d_steps_count = 0
        g_steps_count = 0

        train_iter = iter(train_loader)
        max_steps = cfg.max_train_steps_per_epoch or len(train_loader) // cfg.n_critic

        for step in range(max_steps):
            # -----------------------------------------------------------------
            # 1. Train Critic for n_critic steps
            # -----------------------------------------------------------------
            for _ in range(cfg.n_critic):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                x_real = batch["image"].to(device)
                c_real = batch["condition"].to(device)
                m_real = batch["mask"].to(device)
                b_size = x_real.size(0)

                z = torch.randn(b_size, 100, device=device)
                with torch.no_grad():
                    x_fake = generator(z, c_real, m_real)

                opt_D.zero_grad()
                d_dict = loss_manager.compute_critic_loss(
                    critic=critic,
                    real_images=x_real,
                    fake_images=x_fake,
                    condition=c_real,
                    mask=m_real,
                )
                d_loss = d_dict["total_d_loss"]
                d_loss.backward()

                d_grad_norm = clip_model_gradients(critic, cfg.grad_clip_norm)
                opt_D.step()

                running_d_loss += d_loss.item()
                running_w_dist += (d_dict["d_real_mean"] - d_dict["d_fake_mean"]).item()
                running_gp += d_dict["gradient_penalty"].item()
                running_d_aux += d_dict["loss_aux"].item()
                running_d_grad_norm += d_grad_norm
                d_steps_count += 1

            # -----------------------------------------------------------------
            # 2. Train Generator for 1 step
            # -----------------------------------------------------------------
            opt_G.zero_grad()
            z = torch.randn(b_size, 100, device=device)
            # Re-synthesize with gradient graph attached
            x_fake = generator(z, c_real, m_real)

            g_dict = loss_manager.compute_generator_loss(
                generator=generator,
                critic=critic,
                fake_images=x_fake,
                condition=c_real,
                mask=m_real,
                compute_ms=True,
            )
            g_loss = g_dict["total_g_loss"]
            g_loss.backward()

            g_grad_norm = clip_model_gradients(generator, cfg.grad_clip_norm)
            opt_G.step()

            running_g_loss += g_loss.item()
            running_g_aux += g_dict["loss_aux"].item()
            running_g_ms += g_dict["loss_ms"].item()
            running_g_grad_norm += g_grad_norm
            g_steps_count += 1

            if (step + 1) % 50 == 0 or (step + 1) == max_steps:
                cur_d = running_d_loss / max(1, d_steps_count)
                cur_g = running_g_loss / max(1, g_steps_count)
                cur_w = running_w_dist / max(1, d_steps_count)
                print(
                    f"Epoch [{epoch}/{cfg.epochs}] Step [{step + 1}/{max_steps}] "
                    f"D_Loss: {cur_d:.4f} | G_Loss: {cur_g:.4f} | W_Dist: {cur_w:.4f}"
                )

        epoch_duration = time.time() - epoch_start
        avg_d_loss = running_d_loss / max(1, d_steps_count)
        avg_g_loss = running_g_loss / max(1, g_steps_count)
        avg_w_dist = running_w_dist / max(1, d_steps_count)
        avg_gp = running_gp / max(1, d_steps_count)
        avg_d_aux = running_d_aux / max(1, d_steps_count)
        avg_g_aux = running_g_aux / max(1, g_steps_count)
        avg_g_ms = running_g_ms / max(1, g_steps_count)
        avg_d_norm = running_d_grad_norm / max(1, d_steps_count)
        avg_g_norm = running_g_grad_norm / max(1, g_steps_count)

        # -----------------------------------------------------------------
        # 3. Epoch Visualization: Fixed Condition Grid
        # -----------------------------------------------------------------
        generator.eval()
        with torch.no_grad():
            grid_imgs = generator(fixed_z, fixed_cond, fixed_mask)
        viz_path = os.path.join(cfg.output_viz_dir, f"cgan_phase54_epoch_{epoch}.png")
        save_grid_image(grid_imgs, viz_path)

        # -----------------------------------------------------------------
        # 4. Epoch Validation on Split A2
        # -----------------------------------------------------------------
        print(f"\n--- Running Split A2 Validation for Epoch {epoch} ---")
        val_start = time.time()
        with torch.no_grad():
            # Generate fake images for the validation subset using true val conditions
            val_fakes = []
            for i in range(0, len(val_subset), cfg.batch_size):
                b_cond = val_conditions_tensor[i : i + cfg.batch_size].to(device)
                b_mask = val_masks_tensor[i : i + cfg.batch_size].to(device)
                b_z = torch.randn(b_cond.size(0), 100, device=device)
                b_fake = generator(b_z, b_cond, b_mask)
                val_fakes.append(b_fake.cpu())
            val_fakes_tensor = torch.cat(val_fakes, dim=0)

        # Extract features of generated images
        val_fake_features = extract_evaluator_features(
            evaluator=evaluator,
            images=val_fakes_tensor,
            batch_size=cfg.batch_size,
            device=str(device),
        )

        # Compute C-FD on 512-dim features
        val_cfd = compute_cfd_from_features(val_real_features, val_fake_features)

        # Compute FAV metrics using frozen evaluator
        fav_results = compute_fav_metrics(
            evaluator=evaluator,
            fake_images=val_fakes_tensor,
            target_conditions=val_conditions_tensor,
            target_masks=val_masks_tensor,
            device=str(device),
        )

        # Compute seed diversity (MS-SSIM across 5 random seeds per profile)
        diversity_results = compute_seed_diversity(
            generator=generator,
            conditions=val_conditions_tensor,
            num_seeds=5,
            device=str(device),
        )

        val_duration = time.time() - val_start

        # Composite selection score: C-FD penalized by attribute compliance
        # Lower score is superior
        fav_f1 = fav_results["fav_macro_f1"]
        composite_score = val_cfd - (100.0 * fav_f1)

        print(
            f"Epoch {epoch} Summary ({epoch_duration:.1f}s train, {val_duration:.1f}s val):\n"
            f"  D_Loss: {avg_d_loss:.4f} | G_Loss: {avg_g_loss:.4f} | W_Dist: {avg_w_dist:.4f} | GP: {avg_gp:.4f}\n"
            f"  Val C-FD: {val_cfd:.4f} | Val FAV Macro F1: {fav_f1:.4f} | Val FAV Accuracy: {fav_results['fav_macro_accuracy']:.4f}\n"
            f"  Seed Diversity (1 - MS-SSIM): {diversity_results['seed_diversity_score']:.4f} | Composite Score: {composite_score:.4f}"
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_duration_s": round(epoch_duration, 2),
            "val_duration_s": round(val_duration, 2),
            "d_loss": round(avg_d_loss, 4),
            "g_loss": round(avg_g_loss, 4),
            "wasserstein_distance": round(avg_w_dist, 4),
            "gradient_penalty": round(avg_gp, 4),
            "d_aux_loss": round(avg_d_aux, 4),
            "g_aux_loss": round(avg_g_aux, 4),
            "g_mode_seeking_loss": round(avg_g_ms, 4),
            "d_grad_norm": round(avg_d_norm, 4),
            "g_grad_norm": round(avg_g_norm, 4),
            "val_cfd": round(val_cfd, 4),
            "val_fav_macro_f1": fav_results["fav_macro_f1"],
            "val_fav_micro_f1": fav_results["fav_micro_f1"],
            "val_fav_macro_accuracy": fav_results["fav_macro_accuracy"],
            "val_seed_diversity_score": diversity_results["seed_diversity_score"],
            "val_mean_ms_ssim": diversity_results["mean_pairwise_ms_ssim"],
            "composite_selection_score": round(composite_score, 4),
            "viz_grid_path": viz_path,
        }
        history.append(epoch_metrics)

        # Track best model checkpoint based on Split A2 validation
        if composite_score < best_composite_score:
            best_composite_score = composite_score
            best_epoch = epoch
            best_val_metrics = epoch_metrics
            best_gen_weights = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
            best_crit_weights = {k: v.cpu().clone() for k, v in critic.state_dict().items()}
            print(f"  >>> New Best Checkpoint at Epoch {epoch} (Score: {composite_score:.4f})")

    total_train_time = time.time() - total_training_start
    print(f"\n[DONE] Generator training complete in {total_train_time:.1f}s. Best Epoch: {best_epoch}")

    # -----------------------------------------------------------------
    # 8. Save Best Frozen Generator Checkpoint
    # -----------------------------------------------------------------
    assert best_gen_weights is not None
    assert best_crit_weights is not None

    os.makedirs(cfg.output_checkpoint_dir, exist_ok=True)
    generator_save_dict = {
        "generator_state_dict": best_gen_weights,
        "config": wgan_cfg.to_dict(),
        "epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "hash_salt": gen_manifest.get("hash_salt", ""),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    torch.save(generator_save_dict, cfg.generator_checkpoint_path)

    critic_save_dict = {
        "critic_state_dict": best_crit_weights,
        "config": wgan_cfg.to_dict(),
        "epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    torch.save(critic_save_dict, cfg.critic_checkpoint_path)

    gen_ckpt_hash = get_file_sha256(cfg.generator_checkpoint_path)
    crit_ckpt_hash = get_file_sha256(cfg.critic_checkpoint_path)
    print(f"Frozen Generator Checkpoint: {cfg.generator_checkpoint_path} (SHA-256: {gen_ckpt_hash})")
    print(f"Frozen Critic Checkpoint:    {cfg.critic_checkpoint_path} (SHA-256: {crit_ckpt_hash})")

    # -----------------------------------------------------------------
    # 9. Final Benchmark Evaluation on Strictly Untouched Split C
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK EVALUATION ON UNTOUCHED SPLIT C (SEALED HELD-OUT DATASET)")
    print("=" * 80)

    # Load frozen best generator into clean model
    best_gen_model = ConditionalWGANGPGenerator(
        latent_dim=wgan_cfg.latent_dim,
        condition_dim=wgan_cfg.condition_dim,
        condition_embed_dim=wgan_cfg.condition_embed_dim,
        feature_maps=wgan_cfg.generator_features,
        channels=wgan_cfg.channels,
        image_size=wgan_cfg.image_size,
    ).to(device)
    best_gen_model.load_state_dict(best_gen_weights)
    best_gen_model.eval()

    test_c_dataset = CelebAIdentityDataset(
        metadata_dir=cfg.metadata_dir,
        split="split_c",
        image_dir=cfg.image_dir,
        attribute_dropout=0.0,
        require_images=True,
    )
    test_indices = list(range(min(cfg.final_test_samples, len(test_c_dataset))))
    test_subset = Subset(test_c_dataset, test_indices)
    test_loader = DataLoader(
        test_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    print(f"Loaded Split C Test Subset: {len(test_subset):,} images from {len(test_c_dataset):,} total Split C.")

    test_real_images = []
    test_conditions = []
    test_masks = []
    for b in test_loader:
        test_real_images.append(b["image"])
        test_conditions.append(b["condition"])
        test_masks.append(b["mask"])
    test_real_images_tensor = torch.cat(test_real_images, dim=0)
    test_conditions_tensor = torch.cat(test_conditions, dim=0)
    test_masks_tensor = torch.cat(test_masks, dim=0)

    # Extract real Split C features
    print("Extracting real Split C feature representations with frozen evaluator...")
    real_c_features = extract_evaluator_features(
        evaluator=evaluator,
        images=test_real_images_tensor,
        batch_size=cfg.batch_size,
        device=str(device),
    )

    # Synthesize corresponding fake images for Split C
    print("Synthesizing condition-matched composites with frozen generator...")
    with torch.no_grad():
        test_fakes = []
        for i in range(0, len(test_subset), cfg.batch_size):
            b_cond = test_conditions_tensor[i : i + cfg.batch_size].to(device)
            b_mask = test_masks_tensor[i : i + cfg.batch_size].to(device)
            b_z = torch.randn(b_cond.size(0), 100, device=device)
            b_fake = best_gen_model(b_z, b_cond, b_mask)
            test_fakes.append(b_fake.cpu())
        test_fakes_tensor = torch.cat(test_fakes, dim=0)

    # Extract synthetic feature representations
    fake_c_features = extract_evaluator_features(
        evaluator=evaluator,
        images=test_fakes_tensor,
        batch_size=cfg.batch_size,
        device=str(device),
    )

    # Final C-FD calculation
    final_cfd = compute_cfd_from_features(real_c_features, fake_c_features)

    # Final FAV metrics across active conditions
    final_fav = compute_fav_metrics(
        evaluator=evaluator,
        fake_images=test_fakes_tensor,
        target_conditions=test_conditions_tensor,
        target_masks=test_masks_tensor,
        device=str(device),
    )

    # Final seed diversity (MS-SSIM across 5 random seeds)
    final_diversity = compute_seed_diversity(
        generator=best_gen_model,
        conditions=test_conditions_tensor,
        num_seeds=5,
        device=str(device),
    )

    # Final disaggregated demographic subgroup analysis
    # Uses Male/Female and Young/Not-Young attribute combinations in CelebA Split C
    print("Computing disaggregated demographic subgroup analysis on Split C...")
    # Map attribute column indices from list_attr_celeba
    attr_file = os.path.join(cfg.metadata_dir, "list_attr_celeba.txt")
    with open(attr_file, "r", encoding="utf-8") as f:
        f.readline()
        header = f.readline().split()
        male_col = header.index("Male")
        young_col = header.index("Young")

    # Read demographic flags for test subset
    split_c_image_names = [test_c_dataset.image_names[idx] for idx in test_indices]
    name_to_subgroup: Dict[str, str] = {}
    with open(attr_file, "r", encoding="utf-8") as f:
        f.readline()
        f.readline()
        for line in f:
            parts = line.split()
            if not parts:
                continue
            name = parts[0]
            if name in split_c_image_names:
                is_male = parts[1 + male_col] == "1"
                is_young = parts[1 + young_col] == "1"
                if is_young and not is_male:
                    subgroup = "young_female"
                elif is_young and is_male:
                    subgroup = "young_male"
                elif not is_young and not is_male:
                    subgroup = "older_female"
                else:
                    subgroup = "older_male"
                name_to_subgroup[name] = subgroup

    subgroup_metrics: Dict[str, Dict[str, Any]] = {}
    for sg_name in ["young_female", "young_male", "older_female", "older_male"]:
        sg_indices = [
            i for i, name in enumerate(split_c_image_names)
            if name_to_subgroup.get(name) == sg_name
        ]
        if len(sg_indices) < 10:
            subgroup_metrics[sg_name] = {"sample_count": len(sg_indices), "status": "insufficient_samples"}
            continue

        sg_fakes = test_fakes_tensor[sg_indices]
        sg_conds = test_conditions_tensor[sg_indices]
        sg_masks = test_masks_tensor[sg_indices]
        sg_real_feat = real_c_features[sg_indices]
        sg_fake_feat = fake_c_features[sg_indices]

        sg_cfd = compute_cfd_from_features(sg_real_feat, sg_fake_feat)
        sg_fav = compute_fav_metrics(
            evaluator=evaluator,
            fake_images=sg_fakes,
            target_conditions=sg_conds,
            target_masks=sg_masks,
            device=str(device),
        )

        subgroup_metrics[sg_name] = {
            "sample_count": len(sg_indices),
            "c_fd": round(sg_cfd, 4),
            "fav_macro_f1": sg_fav["fav_macro_f1"],
            "fav_macro_accuracy": sg_fav["fav_macro_accuracy"],
            "fav_micro_f1": sg_fav["fav_micro_f1"],
        }
        print(
            f"  Subgroup '{sg_name}' (N={len(sg_indices)}): "
            f"C-FD: {sg_cfd:.4f} | FAV Macro F1: {sg_fav['fav_macro_f1']:.4f} | Acc: {sg_fav['fav_macro_accuracy']:.4f}"
        )

    # 10. Assemble and Save Results JSON
    final_results: Dict[str, Any] = {
        "dataset": "CelebA",
        "phase": "Phase 5.4: Conditional WGAN-GP Training & Generative Evaluation",
        "generator_checkpoint": {
            "path": cfg.generator_checkpoint_path,
            "sha256": gen_ckpt_hash,
            "best_epoch": best_epoch,
            "total_parameters": g_total,
        },
        "critic_checkpoint": {
            "path": cfg.critic_checkpoint_path,
            "sha256": crit_ckpt_hash,
            "best_epoch": best_epoch,
            "total_parameters": d_total,
        },
        "evaluator_checkpoint": {
            "path": cfg.evaluator_checkpoint,
            "sha256": actual_eval_hash,
            "frozen": True,
        },
        "hyperparameters": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "n_critic": cfg.n_critic,
            "lr_g": cfg.lr_g,
            "lr_d": cfg.lr_d,
            "beta1": cfg.beta1,
            "beta2": cfg.beta2,
            "attribute_dropout": cfg.attribute_dropout,
            "lambda_gp": cfg.lambda_gp,
            "lambda_aux": cfg.lambda_aux,
            "lambda_ms": cfg.lambda_ms,
            "ms_threshold": cfg.ms_threshold,
            "grad_clip_norm": cfg.grad_clip_norm,
        },
        "training_history": history,
        "best_epoch_validation_metrics_split_a2": best_val_metrics,
        "split_c_held_out_benchmark": {
            "test_sample_count": len(test_subset),
            "final_custom_feature_frechet_distance_c_fd": round(final_cfd, 4),
            "final_fav_macro_f1": final_fav["fav_macro_f1"],
            "final_fav_macro_precision": final_fav["fav_macro_precision"],
            "final_fav_macro_recall": final_fav["fav_macro_recall"],
            "final_fav_macro_accuracy": final_fav["fav_macro_accuracy"],
            "final_fav_micro_f1": final_fav["fav_micro_f1"],
            "final_fav_micro_accuracy": final_fav["fav_micro_accuracy"],
            "seed_diversity": {
                "mean_pairwise_ms_ssim": final_diversity["mean_pairwise_ms_ssim"],
                "seed_diversity_score": final_diversity["seed_diversity_score"],
                "mean_pairwise_pixel_l1": final_diversity["mean_pairwise_pixel_l1"],
            },
            "per_attribute_fav": final_fav["per_attribute"],
            "demographic_subgroups": subgroup_metrics,
        },
        "total_training_duration_s": round(total_train_time, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    os.makedirs(os.path.dirname(os.path.abspath(cfg.results_json)), exist_ok=True)
    with open(cfg.results_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY OF PHASE 5.4 RESULTS")
    print(f"Best Epoch: {best_epoch} | Total Training Time: {total_train_time:.1f}s")
    print(f"Split C Final C-FD: {final_cfd:.4f}")
    print(f"Split C Final FAV Macro F1: {final_fav['fav_macro_f1']:.4f} | FAV Macro Acc: {final_fav['fav_macro_accuracy']:.4f}")
    print(f"Split C Seed Diversity Score (1 - MS-SSIM): {final_diversity['seed_diversity_score']:.4f}")
    print(f"Results JSON written to: {cfg.results_json}")
    print("=" * 80)

    return final_results


if __name__ == "__main__":
    train_conditional_wgan_gp()
