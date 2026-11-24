"""
Independent Forensic Evaluator Training Script.
Trains ForensicFeatureCNN from random initialization exclusively on CelebA Split B1.
Validates and selects checkpoints exclusively on Split B2.
Computes per-attribute metrics, Macro/Micro F1, AUROC, Brier scores, and Demographic Subgroup breakdowns.
"""

import os
import sys
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config.gan_config import get_device, set_reproducibility_seed, count_parameters
from data.celeba_dataset import CelebAIdentityDataset
from models.forensic_evaluator import (
    ForensicFeatureCNN,
    calculate_attribute_pos_weights,
    weights_init_evaluator,
)
from utils.evaluator_metrics import (
    compute_evaluator_metrics,
    compute_demographic_subgroup_metrics,
)
from utils.gcit_translator import GCIT_LAYOUT


@dataclass
class EvaluatorTrainingConfig:
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    patience: int = 3
    use_pos_weights: bool = True
    device: str = "auto"
    seed: int = 42
    metadata_dir: str = "phantom-ai-ml/datasets/celeba"
    image_dir: str = "phantom-ai-ml/datasets/celeba/img_align_celeba"
    checkpoint_dir: str = "phantom-ai-ml/checkpoints"
    results_out: str = "phantom-ai-ml/datasets/celeba/evaluator_training_results.json"
    checkpoint_path: str = "phantom-ai-ml/checkpoints/forensic_evaluator_split_b.pt"


def get_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def train_evaluator(
    config: Optional[EvaluatorTrainingConfig] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    skip_image_check_for_test: bool = False,
) -> Dict[str, Any]:
    cfg = config or EvaluatorTrainingConfig()
    set_reproducibility_seed(cfg.seed)
    device = get_device(cfg.device)

    print("=" * 70)
    print("PHANTOM AI 3.0 — INDEPENDENT FORENSIC EVALUATOR TRAINING")
    print(f"Device: {device} | Epochs: {cfg.epochs} | Batch Size: {cfg.batch_size}")
    print(f"Training Split: Split B1 (80%) | Validation Split: Split B2 (20%)")
    print("Pretrained Weights: NONE (Strictly Random Initialization)")
    print("=" * 70)

    # 1. Dataset Loaders
    require_images = not skip_image_check_for_test
    train_dataset = CelebAIdentityDataset(
        metadata_dir=cfg.metadata_dir,
        split="split_b1",
        image_dir=cfg.image_dir if os.path.exists(cfg.image_dir) else None,
        require_images=require_images,
    )
    val_dataset = CelebAIdentityDataset(
        metadata_dir=cfg.metadata_dir,
        split="split_b2",
        image_dir=cfg.image_dir if os.path.exists(cfg.image_dir) else None,
        require_images=require_images,
    )

    # Optional subsampling for accelerated validation or unit tests
    if max_train_samples is not None:
        train_indices = list(range(min(max_train_samples, len(train_dataset))))
        train_loader = DataLoader(
            torch.utils.data.Subset(train_dataset, train_indices),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
        )

    if max_val_samples is not None:
        val_indices = list(range(min(max_val_samples, len(val_dataset))))
        val_loader = DataLoader(
            torch.utils.data.Subset(val_dataset, val_indices),
            batch_size=cfg.batch_size,
            shuffle=False,
        )
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )

    print(f"Dataset Loaded: Train (Split B1) = {len(train_dataset)} | Val (Split B2) = {len(val_dataset)}")

    # 2. Compute Pos Weights for Class Imbalance on Split B1
    pos_weights = None
    if cfg.use_pos_weights:
        pos_weights = calculate_attribute_pos_weights(train_dataset.attributes).to(device)
        print(f"Computed pos_weight vector: min={pos_weights.min():.2f}, max={pos_weights.max():.2f}, mean={pos_weights.mean():.2f}")

    # 3. Model Architecture
    model = ForensicFeatureCNN(in_channels=3, num_attributes=20).to(device)
    tot_params, train_params = count_parameters(model)
    print(f"Model: ForensicFeatureCNN initialized with {tot_params:,} trainable parameters.")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    # 4. Training Loop with Early Stopping
    best_macro_f1 = -1.0
    best_epoch = -1
    best_val_metrics: Dict[str, Any] = {}
    patience_counter = 0

    history: List[Dict[str, Any]] = []
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        t_epoch_start = time.time()
        model.train()
        running_train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            # Conditions in [-1.0, 1.0] -> binary {0.0, 1.0}
            targets = ((batch["condition"] + 1.0) / 2.0).to(device)

            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_train_loss += loss.item()
            train_batches += 1

        scheduler.step()
        avg_train_loss = running_train_loss / max(train_batches, 1)

        # Validation on Split B2
        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        all_val_probs = []
        all_val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                raw_cond = batch["condition"].to(device)
                targets = ((raw_cond + 1.0) / 2.0)

                logits, _ = model(images)
                loss = criterion(logits, targets)
                running_val_loss += loss.item()
                val_batches += 1

                probs = torch.sigmoid(logits)
                all_val_probs.append(probs.cpu())
                all_val_targets.append(targets.cpu())

        avg_val_loss = running_val_loss / max(val_batches, 1)
        val_probs_cat = torch.cat(all_val_probs, dim=0)
        val_targets_cat = torch.cat(all_val_targets, dim=0)

        epoch_metrics = compute_evaluator_metrics(val_probs_cat, val_targets_cat, threshold=0.5)
        epoch_macro_f1 = epoch_metrics["macro_f1"]
        epoch_micro_f1 = epoch_metrics["micro_f1"]
        t_epoch = time.time() - t_epoch_start

        epoch_log = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_macro_f1": epoch_macro_f1,
            "val_micro_f1": epoch_micro_f1,
            "val_accuracy": epoch_metrics["overall_accuracy"],
            "time_seconds": round(t_epoch, 2),
            "lr": round(scheduler.get_last_lr()[0], 6),
        }
        history.append(epoch_log)

        print(
            f"Epoch [{epoch:2d}/{cfg.epochs:2d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Macro F1: {epoch_macro_f1:.4f} | "
            f"Val Micro F1: {epoch_micro_f1:.4f} | "
            f"Time: {t_epoch:.1f}s"
        )

        # Checkpoint Selection: Maximize Validation Macro F1
        if epoch_macro_f1 > best_macro_f1:
            best_macro_f1 = epoch_macro_f1
            best_epoch = epoch
            best_val_metrics = epoch_metrics
            patience_counter = 0

            # Save best checkpoint
            os.makedirs(os.path.dirname(cfg.checkpoint_path), exist_ok=True)
            manifest_file = os.path.join(cfg.metadata_dir, "evaluator_split_b_manifest.json")
            manifest_hash = get_file_sha256(manifest_file) if os.path.exists(manifest_file) else "N/A"

            checkpoint_dict = {
                "format_version": "3.0.0",
                "model_type": "ForensicFeatureCNN",
                "architecture": "4-Stage Deep Residual CNN + AdaptiveAvgPool + MLP Head",
                "num_attributes": 20,
                "feature_dim": 512,
                "in_channels": 3,
                "input_resolution": (64, 64),
                "epoch": best_epoch,
                "best_macro_f1": best_macro_f1,
                "best_micro_f1": epoch_micro_f1,
                "training_split": "split_b1",
                "validation_split": "split_b2",
                "held_out_benchmark": "split_c_isolated",
                "split_b_manifest_sha256": manifest_hash,
                "pretrained_weights": False,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": epoch_metrics,
                "config": asdict(cfg),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            torch.save(checkpoint_dict, cfg.checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping triggered at epoch {epoch} (No improvement for {cfg.patience} epochs).")
                break

    total_training_time = time.time() - t_start
    checkpoint_sha256 = get_file_sha256(cfg.checkpoint_path) if os.path.exists(cfg.checkpoint_path) else "N/A"

    # 5. Intersectional Demographic Subgroup Analysis on Validation Split B2
    # Identify demographic subgroups for Split B2 images
    stats_file = os.path.join(cfg.metadata_dir, "attribute_statistics.json")
    attr_file = os.path.join(cfg.metadata_dir, "list_attr_celeba.txt")
    subgroup_indices = {"All_Validation": list(range(len(val_targets_cat)))}

    # Parse Male, Young, Pale_Skin for Split B2 images
    if os.path.exists(attr_file):
        with open(attr_file, "r") as f:
            f.readline()
            cols = f.readline().split()
            male_idx = cols.index("Male")
            young_idx = cols.index("Young")
            pale_idx = cols.index("Pale_Skin")

            img_set_b2 = set(val_dataset.image_names)
            img_to_row = {}
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] in img_set_b2:
                    img_to_row[parts[0]] = (parts[1 + male_idx], parts[1 + young_idx], parts[1 + pale_idx])

        # Map val_dataset ordering to subgroups
        subgroups_map = {
            "Female_Young_Non_Pale": [],
            "Male_Young_Non_Pale": [],
            "Male_Older_Non_Pale": [],
            "Female_Older_Non_Pale": [],
            "Female_Young_Pale_Skin": [],
            "Male_Young_Pale_Skin": [],
            "Female_Older_Pale_Skin": [],
            "Male_Older_Pale_Skin": [],
        }

        for idx, img in enumerate(val_dataset.image_names):
            if idx >= len(val_targets_cat):
                break
            if img in img_to_row:
                m_str, y_str, p_str = img_to_row[img]
                m_val = 1 if m_str == "1" else -1
                y_val = 1 if y_str == "1" else -1
                p_val = 1 if p_str == "1" else -1

                if m_val == -1 and y_val == 1 and p_val == -1:
                    subgroups_map["Female_Young_Non_Pale"].append(idx)
                elif m_val == 1 and y_val == 1 and p_val == -1:
                    subgroups_map["Male_Young_Non_Pale"].append(idx)
                elif m_val == 1 and y_val == -1 and p_val == -1:
                    subgroups_map["Male_Older_Non_Pale"].append(idx)
                elif m_val == -1 and y_val == -1 and p_val == -1:
                    subgroups_map["Female_Older_Non_Pale"].append(idx)
                elif m_val == -1 and y_val == 1 and p_val == 1:
                    subgroups_map["Female_Young_Pale_Skin"].append(idx)
                elif m_val == 1 and y_val == 1 and p_val == 1:
                    subgroups_map["Male_Young_Pale_Skin"].append(idx)
                elif m_val == -1 and y_val == -1 and p_val == 1:
                    subgroups_map["Female_Older_Pale_Skin"].append(idx)
                elif m_val == 1 and y_val == -1 and p_val == 1:
                    subgroups_map["Male_Older_Pale_Skin"].append(idx)

        subgroup_metrics = compute_demographic_subgroup_metrics(val_probs_cat, val_targets_cat, subgroups_map)
    else:
        subgroup_metrics = {}

    final_results = {
        "status": "COMPLETED",
        "training_time_seconds": round(total_training_time, 2),
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "best_micro_f1": best_val_metrics.get("micro_f1", 0.0),
        "overall_accuracy": best_val_metrics.get("overall_accuracy", 0.0),
        "checkpoint_path": cfg.checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "history": history,
        "per_attribute_metrics": best_val_metrics.get("per_attribute", {}),
        "subgroup_metrics": subgroup_metrics,
        "hardware_device": str(device),
        "config": asdict(cfg),
    }

    with open(cfg.results_out, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print("=" * 70)
    print("EVALUATOR TRAINING COMPLETED SUCCESSFULLY")
    print(f"Best Epoch: {best_epoch} | Best Validation Macro F1: {best_macro_f1:.4f}")
    print(f"Checkpoint Saved: {cfg.checkpoint_path}")
    print(f"Checkpoint SHA-256: {checkpoint_sha256}")
    print(f"Results Saved: {cfg.results_out}")
    print("=" * 70)

    return final_results


if __name__ == "__main__":
    train_evaluator()

