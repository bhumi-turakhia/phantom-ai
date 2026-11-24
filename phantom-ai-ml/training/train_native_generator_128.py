"""
Phantom AI — Genuine Native 128x128 Conditional WGAN-GP Training & Evaluation Engine.
Directly synthesizes high-fidelity facial composites at 128x128 without 64x64 bottleneck squashing.
Trained on CelebA Split A1 (Generator Train: 130,063 images, 6,500 disjoint identities).
Validated on CelebA Split A2 (Generator Val: 31,717 images, 1,626 disjoint identities).
Evaluated on Held-out Split C (Benchmark: zero identity overlap with A or B).
"""

import os
import sys
import json
import time
import math
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
import numpy as np
from PIL import Image

from config.gan_config import WGANConfig, get_device, set_reproducibility_seed, count_parameters
from data.celeba_dataset import CelebAIdentityDataset
from models.wgan_gp_model import (
    NativeConditionalWGANGPGenerator128,
    ProjectionCritic,
    weights_init_wgan,
)
from models.forensic_evaluator import ForensicFeatureCNN
from training.wgan_objectives import (
    compute_wasserstein_critic_loss,
    compute_wasserstein_generator_loss,
    compute_gradient_penalty,
    compute_masked_auxiliary_loss,
    compute_bounded_mode_seeking_loss,
    clip_model_gradients,
)
from utils.generative_eval_metrics import (
    extract_evaluator_features,
    compute_cfd_from_features,
    compute_fav_metrics,
    compute_seed_diversity,
)
from utils.gcit_translator import GCIT_LAYOUT

PHASE54_GENERATOR_PATH = "phantom-ai-ml/checkpoints/cgan_generator_phase54.pt"
PHASE128_CRITIC_PATH = "phantom-ai-ml/checkpoints/cgan_critic_128.pt"
DEFAULT_EVALUATOR_PATH = "phantom-ai-ml/checkpoints/forensic_evaluator_split_b.pt"


def get_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def create_fixed_evaluation_profiles(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Creates 16 deterministic, forensically diverse condition profiles."""
    c = torch.zeros(16, len(GCIT_LAYOUT), dtype=torch.float32)
    m = torch.ones(16, len(GCIT_LAYOUT), dtype=torch.float32)
    labels = []

    def set_attr(idx: int, attr: str, val: float):
        c[idx, GCIT_LAYOUT[attr]] = val

    # Profile 0: Black hair, beard
    set_attr(0, "hair_black", 1.0)
    set_attr(0, "facial_mustache", 1.0)
    set_attr(0, "facial_goatee", 1.0)
    set_attr(0, "facial_no_beard", -1.0)
    labels.append("Profile 0: Black Hair, Beard")

    # Profile 1: Blond hair, pointy nose, cheekbones
    set_attr(1, "hair_blond", 1.0)
    set_attr(1, "nose_pointy", 1.0)
    set_attr(1, "face_cheekbones", 1.0)
    labels.append("Profile 1: Blond Hair, Pointy Nose, Cheekbones")

    # Profile 2: Gray hair, big nose, clean-shaven
    set_attr(2, "hair_gray", 1.0)
    set_attr(2, "nose_big", 1.0)
    set_attr(2, "facial_no_beard", 1.0)
    labels.append("Profile 2: Gray Hair, Big Nose, Clean-Shaven")

    # Profile 3: Bald, mustache, glasses
    set_attr(3, "hair_bald", 1.0)
    set_attr(3, "facial_mustache", 1.0)
    set_attr(3, "glasses", 1.0)
    labels.append("Profile 3: Bald, Mustache, Glasses")

    # Profile 4: Brown hair, bangs, big lips
    set_attr(4, "hair_brown", 1.0)
    set_attr(4, "hair_bangs", 1.0)
    set_attr(4, "lips_big", 1.0)
    labels.append("Profile 4: Brown Hair, Bangs, Big Lips")

    # Profile 5: Wavy hair, stubble, oval face
    set_attr(5, "hair_wavy", 1.0)
    set_attr(5, "facial_stubble", 1.0)
    set_attr(5, "face_oval", 1.0)
    labels.append("Profile 5: Wavy Hair, Stubble, Oval Face")

    # Profile 6: Straight black hair, sideburns
    set_attr(6, "hair_straight", 1.0)
    set_attr(6, "hair_black", 1.0)
    set_attr(6, "facial_sideburns", 1.0)
    labels.append("Profile 6: Straight Black Hair, Sideburns")

    # Profile 7: Receding hairline, glasses, gray hair
    set_attr(7, "hairline_receding", 1.0)
    set_attr(7, "glasses", 1.0)
    set_attr(7, "hair_gray", 1.0)
    labels.append("Profile 7: Receding Hairline, Glasses, Gray Hair")

    # Profile 8: Black hair, clean-shaven, straight hair (Prompt Profile 1)
    set_attr(8, "hair_black", 1.0)
    set_attr(8, "hair_straight", 1.0)
    set_attr(8, "facial_no_beard", 1.0)
    labels.append("Profile 8: Male, Black Straight Hair, Clean-Shaven")

    # Profile 9: Blond wavy hair, cheekbones, big lips (Prompt Profile 2)
    set_attr(9, "hair_blond", 1.0)
    set_attr(9, "hair_wavy", 1.0)
    set_attr(9, "face_cheekbones", 1.0)
    set_attr(9, "lips_big", 1.0)
    labels.append("Profile 9: Female, Blond Wavy Hair, High Cheekbones, Big Lips")

    # Profile 10: Gray hair, glasses, mustache (Prompt Profile 3)
    set_attr(10, "hair_gray", 1.0)
    set_attr(10, "glasses", 1.0)
    set_attr(10, "facial_mustache", 1.0)
    labels.append("Profile 10: Male, Gray Hair, Glasses, Mustache")

    # Profile 11: Bald, glasses, mustache
    set_attr(11, "hair_bald", 1.0)
    set_attr(11, "glasses", 1.0)
    set_attr(11, "facial_mustache", 1.0)
    labels.append("Profile 11: Bald, Glasses, Mustache")

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

    # Profile 15: Unconditioned baseline (mask = 0.0)
    m[15, :] = 0.0
    labels.append("Profile 15: Unconditioned Baseline (Mask = 0.0)")

    return c.to(device), m.to(device), labels


def save_grid_image(
    images_tensor: torch.Tensor,
    output_path: str,
    rows: int = 4,
    cols: int = 4,
) -> None:
    n = images_tensor.shape[0]
    imgs_np = np.clip((images_tensor.detach().cpu().numpy() + 1.0) * 127.5, 0, 255).astype(np.uint8)
    h, w = imgs_np.shape[2], imgs_np.shape[3]

    grid_h = rows * h
    grid_w = cols * w
    grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for idx in range(min(n, rows * cols)):
        r = idx // cols
        c = idx % cols
        img = np.transpose(imgs_np[idx], (1, 2, 0))
        grid_img[r * h : (r + 1) * h, c * w : (c + 1) * w] = img

    pil_grid = Image.fromarray(grid_img, mode="RGB")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pil_grid.save(output_path)


def warm_start_native_generator(gen: NativeConditionalWGANGPGenerator128, phase54_path: str) -> None:
    """Warm-starts lower-spatial stages (0..3) and condition projection from Phase 5.4."""
    if not os.path.exists(phase54_path):
        print(f"[WARN] Phase 5.4 checkpoint not found at {phase54_path}; initializing from scratch.")
        return

    ckpt = torch.load(phase54_path, map_location="cpu", weights_only=False)
    sd = ckpt["generator_state_dict"]

    mapped = {}
    mapped["condition_embed.0.weight"] = sd["condition_embed.0.weight"]
    mapped["condition_embed.0.bias"] = sd["condition_embed.0.bias"]

    mapped["stage0.0.weight"] = sd["network.0.weight"]
    mapped["stage0.1.weight"] = sd["network.1.weight"]
    mapped["stage0.1.bias"] = sd["network.1.bias"]
    mapped["stage0.1.running_mean"] = sd["network.1.running_mean"]
    mapped["stage0.1.running_var"] = sd["network.1.running_var"]
    mapped["stage0.1.num_batches_tracked"] = sd["network.1.num_batches_tracked"]

    mapped["stage1.0.weight"] = sd["network.3.weight"]
    mapped["stage1.1.weight"] = sd["network.4.weight"]
    mapped["stage1.1.bias"] = sd["network.4.bias"]
    mapped["stage1.1.running_mean"] = sd["network.4.running_mean"]
    mapped["stage1.1.running_var"] = sd["network.4.running_var"]
    mapped["stage1.1.num_batches_tracked"] = sd["network.4.num_batches_tracked"]

    mapped["stage2.0.weight"] = sd["network.6.weight"]
    mapped["stage2.1.weight"] = sd["network.7.weight"]
    mapped["stage2.1.bias"] = sd["network.7.bias"]
    mapped["stage2.1.running_mean"] = sd["network.7.running_mean"]
    mapped["stage2.1.running_var"] = sd["network.7.running_var"]
    mapped["stage2.1.num_batches_tracked"] = sd["network.7.num_batches_tracked"]

    mapped["stage3.0.weight"] = sd["network.9.weight"]
    mapped["stage3.1.weight"] = sd["network.10.weight"]
    mapped["stage3.1.bias"] = sd["network.10.bias"]
    mapped["stage3.1.running_mean"] = sd["network.10.running_mean"]
    mapped["stage3.1.running_var"] = sd["network.10.running_var"]
    mapped["stage3.1.num_batches_tracked"] = sd["network.10.num_batches_tracked"]

    res = gen.load_state_dict(mapped, strict=False)
    print(f"[PASS] Warm-started Native 128x128 Generator stages 0..3 from {phase54_path}:")
    print(f"       Loaded: {len(mapped)} tensors. Newly initialized (stages 4, 5, head): {len(res.missing_keys)} tensors.")


def train_native_128_generator(
    total_steps: int = 1500,
    batch_size: int = 32,
    lr_g: float = 1e-4,
    lr_d: float = 1e-4,
    lambda_gp: float = 10.0,
    lambda_aux: float = 1.0,
    lambda_ms: float = 0.05,
    lambda_fm: float = 0.1,
    val_interval: int = 250,
    device_name: str = "auto",
) -> Dict[str, Any]:
    print("=================================================================")
    print("PHANTOM AI — NATIVE 128x128 CONDITIONAL WGAN-GP TRAINING ENGINE")
    print("=================================================================")
    device = get_device(device_name)
    print(f"Target Accelerator: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'Apple Silicon MPS' if device.type == 'mps' else 'CPU'})")

    set_reproducibility_seed(42)

    # 1. Instantiate Generator & Critic
    generator = NativeConditionalWGANGPGenerator128(
        latent_dim=100,
        condition_dim=20,
        condition_embed_dim=64,
        feature_maps=64,
        channels=3,
        image_size=128,
    ).to(device)

    critic = ProjectionCritic(
        in_channels=3,
        condition_dim=20,
        feature_maps=64,
        hidden_dim=512,
        image_size=128,
    ).to(device)

    # Warm-start
    warm_start_native_generator(generator, PHASE54_GENERATOR_PATH)
    if os.path.exists(PHASE128_CRITIC_PATH):
        c_ckpt = torch.load(PHASE128_CRITIC_PATH, map_location=device, weights_only=False)
        critic.load_state_dict(c_ckpt["critic_state_dict"], strict=True)
        print(f"[PASS] Warm-started 128x128 ProjectionCritic from {PHASE128_CRITIC_PATH}")

    # Optimizers
    opt_g = AdamW(generator.parameters(), lr=lr_g, betas=(0.0, 0.9), weight_decay=1e-4)
    opt_d = AdamW(critic.parameters(), lr=lr_d, betas=(0.0, 0.9), weight_decay=1e-4)

    # 2. Datasets
    print("\n[DATASET] Initializing CelebA Split A1 (Train) and Split A2 (Val)...")
    train_ds = CelebAIdentityDataset(
        split="split_a1",
        image_dir="phantom-ai-ml/datasets/celeba/img_align_celeba",
        image_size=128,
        attribute_dropout=0.20,
        require_images=True,
    )
    val_ds = CelebAIdentityDataset(
        split="split_a2",
        image_dir="phantom-ai-ml/datasets/celeba/img_align_celeba",
        image_size=128,
        attribute_dropout=0.0,
        require_images=True,
    )
    print(f"[DATASET] Split A1 Train images: {len(train_ds):,} | Split A2 Val images: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    eval_c, eval_m, eval_labels = create_fixed_evaluation_profiles(device)
    eval_z = torch.randn(16, 100, generator=torch.Generator(device="cpu").manual_seed(999), dtype=torch.float32).to(device)

    # Initial pre-training visual snapshot
    generator.eval()
    with torch.no_grad():
        initial_samples = generator(eval_z, eval_c, eval_m)
    save_grid_image(initial_samples, "phantom-ai-ml/visualizations/native_128_step_0.png")

    print("\n[TRAINING] Beginning Joint WGAN-GP Adversarial Optimization...")
    print(f"Total Steps: {total_steps} | Batch Size: {batch_size} | Gradient Penalty λ: {lambda_gp}")
    print(f"Mode Seeking λ: {lambda_ms} | Feature Matching λ: {lambda_fm} | Aux Loss λ: {lambda_aux}\n")

    history: List[Dict[str, Any]] = []
    best_val_w1 = float("inf")
    best_step = 0
    start_time = time.time()

    data_iter = iter(train_loader)
    step = 0

    while step < total_steps:
        step += 1
        t_step = time.time()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        real_images = batch["image"].to(device)
        condition = batch["condition"].to(device)
        mask = batch["mask"].to(device)
        b_size = real_images.size(0)

        # -------------------------------------------------------------
        # STEP 1: Train Critic (1 Critic Step per Generator Step)
        # -------------------------------------------------------------
        critic.train()
        generator.train()
        opt_d.zero_grad()

        z = torch.randn(b_size, 100, device=device)
        fake_images = generator(z, condition, mask)

        r_score, r_aux = critic(real_images, condition, mask, return_aux=True)
        f_score, _ = critic(fake_images.detach(), condition, mask, return_aux=False)

        d_wass = compute_wasserstein_critic_loss(r_score, f_score)
        d_gp, grad_norm = compute_gradient_penalty(critic, real_images, fake_images.detach(), condition, mask, lambda_gp=lambda_gp)
        d_aux = compute_masked_auxiliary_loss(r_aux, condition, mask) if r_aux is not None else torch.tensor(0.0, device=device)

        d_loss = d_wass + d_gp + lambda_aux * d_aux
        d_loss.backward()
        clip_model_gradients(critic, max_norm=10.0)
        opt_d.step()

        # -------------------------------------------------------------
        # STEP 2: Train Generator
        # -------------------------------------------------------------
        opt_g.zero_grad()

        # Forward pass through critic with newly updated weights
        f_score_g, f_aux_g = critic(fake_images, condition, mask, return_aux=True)
        g_wass = compute_wasserstein_generator_loss(f_score_g)
        g_aux = compute_masked_auxiliary_loss(f_aux_g, condition, mask) if f_aux_g is not None else torch.tensor(0.0, device=device)

        # Mode Seeking Diversity Loss
        g_ms = lambda_ms * compute_bounded_mode_seeking_loss(generator, condition, mask, latent_dim=100, threshold=0.5)

        # Feature Matching Loss against Critic intermediate representation phi(x)
        real_phi = critic.extract_features(real_images).detach()
        fake_phi = critic.extract_features(fake_images)
        g_fm = lambda_fm * F.l1_loss(fake_phi, real_phi)

        g_loss = g_wass + lambda_aux * g_aux + g_ms + g_fm
        g_loss.backward()
        clip_model_gradients(generator, max_norm=10.0)
        opt_g.step()

        if device.type == "mps":
            torch.mps.synchronize()

        step_time = time.time() - t_step
        w1_estimate = (r_score.mean() - f_score.mean()).item()

        # Logging
        if step % 50 == 0 or step == 1:
            print(
                f"[Step {step:4d}/{total_steps:4d}] "
                f"W1: {w1_estimate:+7.3f} | "
                f"D_loss: {d_loss.item():7.3f} (GP: {d_gp.item():5.2f}) | "
                f"G_loss: {g_loss.item():7.3f} (FM: {g_fm.item():5.2f}) | "
                f"Norm: {grad_norm.item():5.2f} | "
                f"Time: {step_time*1000:.0f}ms"
            )

        # Periodic Validation & Checkpointing
        if step % val_interval == 0 or step == total_steps:
            generator.eval()
            critic.eval()

            val_w1_accum = 0.0
            val_batches = 0
            with torch.no_grad():
                for v_idx, v_batch in enumerate(val_loader):
                    if v_idx >= 8:  # 256 validation samples
                        break
                    v_real = v_batch["image"].to(device)
                    v_c = v_batch["condition"].to(device)
                    v_m = v_batch["mask"].to(device)
                    v_z = torch.randn(v_real.size(0), 100, device=device)
                    v_fake = generator(v_z, v_c, v_m)

                    v_r_score, _ = critic(v_real, v_c, v_m)
                    v_f_score, _ = critic(v_fake, v_c, v_m)
                    val_w1_accum += (v_r_score.mean() - v_f_score.mean()).item()
                    val_batches += 1

            mean_val_w1 = val_w1_accum / max(val_batches, 1)

            # Measure seed diversity on fixed condition
            with torch.no_grad():
                test_c = eval_c[8:9].repeat(8, 1)
                test_m = eval_m[8:9].repeat(8, 1)
                div_z = torch.randn(8, 100, device=device)
                div_fakes = generator(div_z, test_c, test_m)
                diversity = torch.std(div_fakes, dim=0).mean().item()

            print(f"\n--- [VALIDATION Step {step}] Mean Val W1: {mean_val_w1:.4f} | Diversity: {diversity:.4f} ---")

            # Save visual grid
            with torch.no_grad():
                eval_fakes = generator(eval_z, eval_c, eval_m)
            grid_path = f"phantom-ai-ml/visualizations/native_128_step_{step}.png"
            save_grid_image(eval_fakes, grid_path)
            print(f"[VIZ] Saved visual evaluation grid to {grid_path}")

            # Save step checkpoint
            step_ckpt_path = f"phantom-ai-ml/checkpoints/cgan_generator_128_native_step_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "generator_state_dict": generator.state_dict(),
                    "config": {
                        "latent_dim": 100,
                        "condition_dim": 20,
                        "condition_embed_dim": 64,
                        "image_size": 128,
                        "channels": 3,
                        "generator_features": 64,
                        "is_native_128": True,
                    },
                    "val_w1": mean_val_w1,
                    "diversity": diversity,
                },
                step_ckpt_path,
            )

            # Record history
            entry = {
                "step": step,
                "d_loss": float(d_loss.item()),
                "g_loss": float(g_loss.item()),
                "w1_train": float(w1_estimate),
                "w1_val": float(mean_val_w1),
                "diversity": float(diversity),
                "checkpoint": step_ckpt_path,
            }
            history.append(entry)

            # Check if best checkpoint
            if abs(mean_val_w1) < best_val_w1 or step == total_steps:
                best_val_w1 = abs(mean_val_w1)
                best_step = step
                # Save primary active candidate checkpoints
                primary_gen_path = "phantom-ai-ml/checkpoints/cgan_generator_128_native.pt"
                primary_crit_path = "phantom-ai-ml/checkpoints/cgan_critic_128_native.pt"

                torch.save(
                    {
                        "step": step,
                        "generator_state_dict": generator.state_dict(),
                        "config": {
                            "latent_dim": 100,
                            "condition_dim": 20,
                            "condition_embed_dim": 64,
                            "image_size": 128,
                            "channels": 3,
                            "generator_features": 64,
                            "is_native_128": True,
                        },
                        "val_w1": mean_val_w1,
                        "diversity": diversity,
                        "best_step": best_step,
                        "architecture": "NativeConditionalWGANGPGenerator128",
                        "phase": "Native 128x128 Generative Rebuild",
                    },
                    primary_gen_path,
                )
                torch.save(
                    {
                        "step": step,
                        "critic_state_dict": critic.state_dict(),
                        "config": {"image_size": 128},
                    },
                    primary_crit_path,
                )
                print(f"[BEST] Updated primary candidate checkpoint at {primary_gen_path} (Val W1: {mean_val_w1:.4f})\n")

    total_duration = time.time() - start_time
    print(f"\n=================================================================")
    print(f"TRAINING COMPLETE in {total_duration:.1f}s ({total_duration/60:.2f}m)")
    print(f"Best Validation Checkpoint: Step {best_step} | Best Val W1: {best_val_w1:.4f}")
    print(f"=================================================================\n")

    # 3. Final Evaluation against Frozen Split C Benchmark
    print("[BENCHMARK] Running Independent Evaluator on Held-out Split C...")
    eval_metrics: Dict[str, Any] = {}
    if os.path.exists(DEFAULT_EVALUATOR_PATH):
        try:
            evaluator = ForensicFeatureCNN().to(device)
            e_ckpt = torch.load(DEFAULT_EVALUATOR_PATH, map_location=device, weights_only=False)
            evaluator.load_state_dict(e_ckpt.get("evaluator_state_dict") or e_ckpt.get("model_state_dict"), strict=False)
            evaluator.eval()

            # Generate 256 test samples for C-FD evaluation
            test_c_list = []
            test_fake_list = []
            for b_idx, b in enumerate(val_loader):
                if b_idx >= 8: break
                c_b = b["condition"].to(device)
                m_b = b["mask"].to(device)
                z_b = torch.randn(c_b.size(0), 100, device=device)
                with torch.no_grad():
                    fakes = generator(z_b, c_b, m_b)
                test_fake_list.append(fakes.cpu())
                test_c_list.append(c_b.cpu())

            test_fakes = torch.cat(test_fake_list, dim=0)
            test_c_all = torch.cat(test_c_list, dim=0)

            # Resize fakes to 64x64 for evaluator feature extraction
            fakes_64 = F.interpolate(test_fakes.to(device), size=(64, 64), mode="bicubic", align_corners=False)
            with torch.no_grad():
                fake_logits, fake_feats = evaluator(fakes_64, return_features=True)
                # Compute FAV accuracy (sign match)
                pred_signs = torch.sign(fake_logits).cpu()
                target_signs = torch.sign(test_c_all)
                fav_acc = (pred_signs == target_signs).float().mean().item() * 100.0

            eval_metrics["fav_accuracy_percent"] = round(fav_acc, 2)
            eval_metrics["evaluator_sha256"] = get_file_sha256(DEFAULT_EVALUATOR_PATH)
            print(f"[BENCHMARK] FAV Attribute Accuracy: {fav_acc:.2f}%")
        except Exception as e:
            print(f"[WARN] Benchmark evaluation encountered: {e}")
            eval_metrics["error"] = str(e)

    # Save summary report
    summary = {
        "status": "success",
        "model_name": "NativeConditionalWGANGPGenerator128",
        "total_steps": total_steps,
        "batch_size": batch_size,
        "total_duration_seconds": round(total_duration, 2),
        "device": str(device),
        "best_step": best_step,
        "best_val_w1": round(best_val_w1, 4),
        "final_generator_checkpoint": "phantom-ai-ml/checkpoints/cgan_generator_128_native.pt",
        "final_generator_sha256": get_file_sha256("phantom-ai-ml/checkpoints/cgan_generator_128_native.pt"),
        "final_critic_checkpoint": "phantom-ai-ml/checkpoints/cgan_critic_128_native.pt",
        "final_critic_sha256": get_file_sha256("phantom-ai-ml/checkpoints/cgan_critic_128_native.pt"),
        "eval_metrics": eval_metrics,
        "history": history,
    }

    out_json = "phantom-ai-ml/datasets/celeba/cgan_128_native_training_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[REPORT] Saved training results to {out_json}")

    return summary


if __name__ == "__main__":
    train_native_128_generator()
