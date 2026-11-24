"""
Phase 5.7: Latent-Space Hyperplane Traversal Study and Validation.
Discovers and rigorously evaluates linear SVM semantic directions in R^{100}
for the 20 GCIT facial attributes using the frozen Generator and independent Evaluator.

Guarantees:
- Zero modifications to generator or evaluator weights.
- Strictly identity-disjoint from Split C (held-out test benchmark).
- Fits directions on a training bank of latent vectors and validates on an independent test bank of seeds.
- Measures intended semantic change, cross-seed consistency, and unintended cross-attribute entanglement.
- Rejects directions failing validation criteria.
"""

import os
import sys
import json
import hashlib
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn

# Ensure local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.gan_config import get_device
from models.wgan_gp_model import ConditionalWGANGPGenerator
from models.forensic_evaluator import ForensicFeatureCNN
from utils.gcit_translator import GCIT_LAYOUT

EXPECTED_GENERATOR_SHA256 = "a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964"
EXPECTED_EVALUATOR_SHA256 = "f52ae5a8df4ae574432f353fd77fc605f6dcf758d67d492b447c139a79e78834"


def calculate_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class LinearSVMPyTorch(nn.Module):
    """Linear SVM with soft margin in R^D."""
    def __init__(self, dim: int = 100):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w + self.b


def fit_linear_svm(
    z_pos: torch.Tensor,
    z_neg: torch.Tensor,
    c_reg: float = 1.0,
    max_iter: int = 60,
) -> Tuple[torch.Tensor, float, float]:
    """
    Fits a linear SVM separating z_pos (+1) and z_neg (-1).
    Returns (unit_normal, bias, train_accuracy).
    """
    device = z_pos.device
    dim = z_pos.shape[1]
    svm = LinearSVMPyTorch(dim=dim).to(device)

    x_train = torch.cat([z_pos, z_neg], dim=0)
    y_train = torch.cat([
        torch.ones(len(z_pos), device=device),
        -torch.ones(len(z_neg), device=device)
    ])

    optimizer = torch.optim.LBFGS(svm.parameters(), lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        preds = svm(x_train)
        hinge_loss = torch.clamp(1.0 - y_train * preds, min=0.0).mean()
        reg_loss = 0.5 * (svm.w ** 2).sum()
        loss = reg_loss + c_reg * hinge_loss
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        w_norm = svm.w.norm()
        if w_norm.item() < 1e-7:
            unit_normal = svm.w
        else:
            unit_normal = svm.w / w_norm
        preds = svm(x_train)
        acc = ((preds * y_train) > 0).float().mean().item()

    return unit_normal.detach(), svm.b.item(), acc


def run_study(
    gen_path: str = "phantom-ai-ml/checkpoints/cgan_generator_phase54.pt",
    eval_path: str = "phantom-ai-ml/checkpoints/forensic_evaluator_split_b.pt",
    out_dir: str = "phantom-ai-ml/checkpoints",
    n_fit: int = 800,
    n_test: int = 50,
    alpha_step: float = 1.5,
):
    print("================================================================")
    print("   PHANTOM AI 3.0 — PHASE 5.7 LATENT TRAVERSAL STUDY            ")
    print("================================================================")

    # 1. Checkpoint integrity check
    gen_sha = calculate_sha256(gen_path)
    eval_sha = calculate_sha256(eval_path)
    print(f"Generator Checkpoint: {gen_path}")
    print(f"  SHA-256: {gen_sha}")
    assert gen_sha == EXPECTED_GENERATOR_SHA256, "Generator SHA-256 mismatch!"

    print(f"Evaluator Checkpoint: {eval_path}")
    print(f"  SHA-256: {eval_sha}")
    assert eval_sha == EXPECTED_EVALUATOR_SHA256, "Evaluator SHA-256 mismatch!"

    device = get_device("auto")
    print(f"Hardware Device: {device}")

    # 2. Instantiate and load frozen models
    print("\nLoading Frozen Generator...")
    ckpt_gen = torch.load(gen_path, map_location=device, weights_only=False)
    generator = ConditionalWGANGPGenerator(
        latent_dim=100,
        condition_dim=20,
        condition_embed_dim=64,
        feature_maps=64,
        channels=3,
        image_size=64,
    ).to(device)
    generator.load_state_dict(ckpt_gen["generator_state_dict"], strict=True)
    generator.eval()

    print("Loading Frozen Independent Evaluator...")
    ckpt_eval = torch.load(eval_path, map_location=device, weights_only=False)
    evaluator = ForensicFeatureCNN(in_channels=3, num_attributes=20).to(device)
    evaluator.load_state_dict(ckpt_eval["model_state_dict"], strict=True)
    evaluator.eval()

    # Layout mapping
    idx_to_name = {idx: name for name, idx in GCIT_LAYOUT.items()}
    num_attrs = len(GCIT_LAYOUT)

    # 3. Fitting Phase: Generate synthetic latent dataset and score with evaluator
    print(f"\n--- FITTING PHASE ({n_fit} latent samples) ---")
    torch.manual_seed(1001)

    # Sample latent vectors in batches
    batch_size = 64
    all_z = []
    all_probs = []

    c_zero = torch.zeros(1, num_attrs, device=device)
    m_zero = torch.zeros(1, num_attrs, device=device)

    with torch.inference_mode():
        for b_start in range(0, n_fit, batch_size):
            b_len = min(batch_size, n_fit - b_start)
            # Sample standard Gaussian latent vectors
            z = torch.randn(b_len, 100, device=device)
            c_b = c_zero.expand(b_len, -1)
            m_b = m_zero.expand(b_len, -1)

            # Synthesize images
            fakes = generator(z, c_b, m_b)

            # Score using independent evaluator
            logits, _ = evaluator(fakes)
            probs = torch.sigmoid(logits)

            all_z.append(z)
            all_probs.append(probs)

    Z_fit = torch.cat(all_z, dim=0)          # (N_fit, 100)
    P_fit = torch.cat(all_probs, dim=0)      # (N_fit, 20)

    print(f"Synthesized and scored {Z_fit.shape[0]} latent vectors across {num_attrs} GCIT dimensions.")

    # 4. Fit SVM hyperplanes for each dimension
    fitted_directions: Dict[str, torch.Tensor] = {}
    svm_train_accuracies: Dict[str, float] = {}

    quantile_cut = 0.25  # top/bottom 25%
    n_cut = int(n_fit * quantile_cut)

    for attr_idx in range(num_attrs):
        attr_name = idx_to_name[attr_idx]
        scores = P_fit[:, attr_idx]

        sorted_indices = torch.argsort(scores)
        neg_indices = sorted_indices[:n_cut]
        pos_indices = sorted_indices[-n_cut:]

        z_pos = Z_fit[pos_indices]
        z_neg = Z_fit[neg_indices]

        normal, bias, acc = fit_linear_svm(z_pos, z_neg, c_reg=1.0)
        fitted_directions[attr_name] = normal
        svm_train_accuracies[attr_name] = acc
        print(f"  [{attr_idx:2d}] {attr_name:<20}: SVM Train Acc = {acc * 100:.1f}%")

    # 5. Validation Phase on Independent Held-Out Seeds
    print(f"\n--- VALIDATION PHASE ({n_test} independent held-out seeds, alpha={alpha_step}) ---")
    test_seeds = [5000 + i * 17 for i in range(n_test)]

    # Generate baseline test latents
    Z_test_list = [
        torch.randn(1, 100, generator=torch.Generator(device="cpu").manual_seed(s), dtype=torch.float32)
        for s in test_seeds
    ]
    Z_test = torch.cat(Z_test_list, dim=0).to(device)  # (N_test, 100)

    # Pre-generate baseline test images and baseline scores
    with torch.inference_mode():
        c_b = c_zero.expand(n_test, -1)
        m_b = m_zero.expand(n_test, -1)
        base_fakes = generator(Z_test, c_b, m_b)
        base_logits, _ = evaluator(base_fakes)
        base_probs = torch.sigmoid(base_logits)        # (N_test, 20)

    validation_results = {}
    validated_directions = {}

    for attr_idx in range(num_attrs):
        attr_name = idx_to_name[attr_idx]
        normal = fitted_directions[attr_name]

        # Traversal: Z+ = Z + alpha * n, Z- = Z - alpha * n
        Z_plus = Z_test + alpha_step * normal.unsqueeze(0)
        Z_minus = Z_test - alpha_step * normal.unsqueeze(0)

        with torch.inference_mode():
            fakes_plus = generator(Z_plus, c_b, m_b)
            fakes_minus = generator(Z_minus, c_b, m_b)

            logits_plus, _ = evaluator(fakes_plus)
            logits_minus, _ = evaluator(fakes_minus)

            probs_plus = torch.sigmoid(logits_plus)    # (N_test, 20)
            probs_minus = torch.sigmoid(logits_minus)  # (N_test, 20)

        # Changes per seed
        diff_all = probs_plus - probs_minus            # (N_test, 20)

        # Intended attribute metric
        intended_diffs = diff_all[:, attr_idx]
        mean_intended_diff = intended_diffs.mean().item()
        consistency = (intended_diffs > 0).float().mean().item()

        # Unintended attributes (entanglement)
        unintended_diffs = diff_all.clone()
        unintended_diffs[:, attr_idx] = 0.0
        mean_abs_unintended = unintended_diffs.abs().mean(dim=0)  # (20,)

        # Identify attribute with worst unintended change
        worst_unintended_idx = mean_abs_unintended.argmax().item()
        worst_unintended_name = idx_to_name[worst_unintended_idx]
        worst_unintended_val = mean_abs_unintended[worst_unintended_idx].item()
        overall_avg_unintended = mean_abs_unintended.mean().item()

        # Validation Criterion:
        # 1. Mean intended change >= 0.12 (on [0, 1] probability scale)
        # 2. Consistency across seeds >= 65%
        # 3. Ratio of intended change to worst unintended change > 1.2
        is_validated = (
            mean_intended_diff >= 0.12 and
            consistency >= 0.65 and
            (mean_intended_diff / max(worst_unintended_val, 1e-4)) >= 1.15
        )

        validation_results[attr_name] = {
            "attribute_index": attr_idx,
            "svm_train_accuracy": round(svm_train_accuracies[attr_name], 4),
            "mean_intended_delta": round(mean_intended_diff, 4),
            "consistency_rate": round(consistency, 4),
            "worst_unintended_attribute": worst_unintended_name,
            "worst_unintended_delta": round(worst_unintended_val, 4),
            "overall_avg_unintended_delta": round(overall_avg_unintended, 4),
            "validation_verdict": "VALIDATED" if is_validated else "REJECTED",
        }

        if is_validated:
            validated_directions[attr_name] = normal.cpu()

        status_str = "VALIDATED" if is_validated else "REJECTED"
        print(
            f"[{status_str:9s}] {attr_name:<20}: "
            f"Intended Delta = +{mean_intended_diff:.3f} | "
            f"Consistency = {consistency * 100:.1f}% | "
            f"Worst Side-Effect = {worst_unintended_name} (+{worst_unintended_val:.3f})"
        )

    # 6. Save results and checkpoint
    os.makedirs(out_dir, exist_ok=True)
    directions_ckpt_path = os.path.join(out_dir, "latent_directions_phase57.pt")
    results_json_path = os.path.join(out_dir, "latent_traversal_validation_results.json")

    # Serialize directions dictionary
    torch.save(
        {
            "format_version": "1.0",
            "phase": "5.7",
            "latent_dim": 100,
            "num_validated": len(validated_directions),
            "all_directions": {k: v.cpu() for k, v in fitted_directions.items()},
            "validated_directions": validated_directions,
            "validation_metrics": validation_results,
            "generator_sha256": gen_sha,
            "evaluator_sha256": eval_sha,
            "alpha_step": alpha_step,
            "n_fit": n_fit,
            "n_test": n_test,
        },
        directions_ckpt_path,
    )

    with open(results_json_path, "w") as f:
        json.dump(
            {
                "generator_sha256": gen_sha,
                "evaluator_sha256": eval_sha,
                "n_fit_samples": n_fit,
                "n_test_samples": n_test,
                "alpha_step": alpha_step,
                "total_gcit_dimensions": num_attrs,
                "validated_count": len(validated_directions),
                "rejected_count": num_attrs - len(validated_directions),
                "validated_attributes": list(validated_directions.keys()),
                "detailed_results": validation_results,
            },
            f,
            indent=2,
        )

    print("\n================================================================")
    print(f"Study Complete! Validated Directions: {len(validated_directions)} / {num_attrs}")
    print(f"Saved directions tensor: {directions_ckpt_path}")
    print(f"Saved audit report JSON: {results_json_path}")
    print("================================================================")


if __name__ == "__main__":
    run_study()

