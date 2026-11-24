"""
Generative Evaluation Metrics for Conditional WGAN-GP.
Implements:
1. Custom Feature Fréchet Distance (C-FD) using the frozen ForensicFeatureCNN (no ImageNet/Inception).
2. Forensic Attribute Verification (FAV) precision, recall, and F1 across active GCIT conditions.
3. Multi-Scale Structural Similarity (MS-SSIM) and SSIM diversity metrics across random noise seeds.
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.gcit_translator import GCIT_LAYOUT


# ============================================================================
# 1. Custom Feature Fréchet Distance (C-FD)
# ============================================================================

def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Computes Fréchet Distance d^2 = ||mu1 - mu2||_2^2 + Tr(sigma1 + sigma2 - 2*(sigma1*sigma2)^(1/2))
    using symmetric eigenvalue decomposition for strict numerical stability and zero external dependencies.
    """
    diff = mu1 - mu2
    diff_sq = np.dot(diff, diff)

    # Regularize covariance matrices
    cov1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    cov2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    # Symmetric square root of cov1: cov1 = V1 @ D1 @ V1.T => cov1^(1/2) = V1 @ sqrt(D1) @ V1.T
    w1, v1 = np.linalg.eigh(cov1)
    w1 = np.maximum(w1, 0.0)
    sqrt_cov1 = v1 @ np.diag(np.sqrt(w1)) @ v1.T

    # Form symmetric matrix M = cov1^(1/2) @ cov2 @ cov1^(1/2)
    m_mat = sqrt_cov1 @ cov2 @ sqrt_cov1
    w_m = np.linalg.eigvalsh(m_mat)
    w_m = np.maximum(w_m, 0.0)
    trace_term = np.sum(np.sqrt(w_m))

    trace_cov1 = np.trace(cov1)
    trace_cov2 = np.trace(cov2)

    cfd = diff_sq + trace_cov1 + trace_cov2 - 2.0 * trace_term
    return float(max(cfd, 0.0))


@torch.no_grad()
def extract_evaluator_features(
    evaluator: nn.Module,
    images: torch.Tensor,
    batch_size: int = 128,
    device: str = "cpu",
) -> np.ndarray:
    """
    Extracts 512-dim bottleneck features psi(x) from ForensicFeatureCNN.
    """
    evaluator.eval()
    all_features = []
    n = images.shape[0]

    for i in range(0, n, batch_size):
        batch = images[i : i + batch_size].to(device)
        if batch.shape[-1] != 64 or batch.shape[-2] != 64:
            batch = F.interpolate(batch, size=(64, 64), mode="bilinear", align_corners=False)
        feat = evaluator.extract_features(batch)
        all_features.append(feat.detach().cpu().numpy())

    return np.concatenate(all_features, axis=0)


def compute_features_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes mean vector mu and covariance matrix Sigma for feature representations.
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_cfd_from_features(
    real_features: np.ndarray,
    fake_features: np.ndarray,
) -> float:
    """
    Computes C-FD directly from pre-extracted real and fake feature matrices.
    """
    mu_r, sigma_r = compute_features_statistics(real_features)
    mu_g, sigma_g = compute_features_statistics(fake_features)
    return calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)


# ============================================================================
# 2. Forensic Attribute Verification (FAV)
# ============================================================================

@torch.no_grad()
def compute_fav_metrics(
    evaluator: nn.Module,
    fake_images: torch.Tensor,
    target_conditions: torch.Tensor,
    target_masks: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluates whether generated images exhibit the requested GCIT attributes.
    Uses frozen ForensicFeatureCNN to predict attributes and compares against active conditions.

    Args:
        evaluator: Frozen ForensicFeatureCNN in eval mode.
        fake_images: (N, 3, 64, 64) synthetic images in [-1.0, 1.0].
        target_conditions: (N, 20) requested condition vector in [-1.0, 1.0].
        target_masks: Optional (N, 20) active condition mask. If None, mask = (conditions != 0.0).
        threshold: Classification probability threshold (default: 0.5).
        device: Execution device.

    Returns:
        Dict with overall macro/micro F1, precision, recall, accuracy, and per-attribute breakdown.
    """
    evaluator.eval()
    n = fake_images.shape[0]
    batch_size = 128
    logits_list = []

    for i in range(0, n, batch_size):
        batch = fake_images[i : i + batch_size].to(device)
        if batch.shape[-1] != 64 or batch.shape[-2] != 64:
            batch = F.interpolate(batch, size=(64, 64), mode="bilinear", align_corners=False)
        logits, _ = evaluator(batch)
        logits_list.append(logits.detach().cpu())

    pred_logits = torch.cat(logits_list, dim=0)
    pred_probs = torch.sigmoid(pred_logits)
    # Binary predictions: 1.0 if prob >= threshold else -1.0
    pred_binary = torch.where(pred_probs >= threshold, 1.0, -1.0)

    conditions = target_conditions.cpu()
    if target_masks is None:
        masks = (conditions != 0.0).float()
    else:
        masks = target_masks.cpu()

    # Per-attribute metrics
    per_attribute: Dict[str, Dict[str, float]] = {}
    f1_list = []
    prec_list = []
    rec_list = []
    acc_list = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    idx_to_attr = {idx: name for name, idx in GCIT_LAYOUT.items()}

    for dim_idx in range(len(GCIT_LAYOUT)):
        attr_name = idx_to_attr.get(dim_idx, f"attr_{dim_idx}")
        active = masks[:, dim_idx] > 0.5

        if active.sum().item() == 0:
            per_attribute[attr_name] = {
                "active_count": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }
            continue

        tgt = conditions[active, dim_idx]
        prd = pred_binary[active, dim_idx]

        tp = ((tgt == 1.0) & (prd == 1.0)).sum().item()
        fp = ((tgt == -1.0) & (prd == 1.0)).sum().item()
        fn = ((tgt == 1.0) & (prd == -1.0)).sum().item()
        tn = ((tgt == -1.0) & (prd == -1.0)).sum().item()

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        per_attribute[attr_name] = {
            "active_count": int(active.sum().item()),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "accuracy": round(float(acc), 4),
        }

        f1_list.append(f1)
        prec_list.append(prec)
        rec_list.append(rec)
        acc_list.append(acc)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    macro_prec = float(np.mean(prec_list)) if prec_list else 0.0
    macro_rec = float(np.mean(rec_list)) if rec_list else 0.0
    macro_acc = float(np.mean(acc_list)) if acc_list else 0.0

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    total_evaluated = total_tp + total_tn + total_fp + total_fn
    micro_acc = (total_tp + total_tn) / total_evaluated if total_evaluated > 0 else 0.0

    return {
        "fav_macro_f1": round(macro_f1, 4),
        "fav_macro_precision": round(macro_prec, 4),
        "fav_macro_recall": round(macro_rec, 4),
        "fav_macro_accuracy": round(macro_acc, 4),
        "fav_micro_f1": round(micro_f1, 4),
        "fav_micro_precision": round(micro_prec, 4),
        "fav_micro_recall": round(micro_rec, 4),
        "fav_micro_accuracy": round(micro_acc, 4),
        "total_active_evaluations": total_evaluated,
        "per_attribute": per_attribute,
    }


# ============================================================================
# 3. Structural Similarity (SSIM / MS-SSIM) Diversity Metrics
# ============================================================================

def _gaussian_window_1d(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _create_gaussian_window_2d(window_size: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    w1d = _gaussian_window_1d(window_size, sigma).unsqueeze(1)
    w2d = w1d.mm(w1d.t()).unsqueeze(0).unsqueeze(0)
    window = w2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim_tensor(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    c1: float = 0.01 ** 2,
    c2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    Computes SSIM between two batches of RGB images normalized to [0.0, 1.0].
    """
    channels = img1.size(1)
    window = _create_gaussian_window_2d(window_size, sigma, channels).to(img1.device, dtype=img1.dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    return ssim_map.mean(dim=(1, 2, 3))


def ms_ssim_tensor(
    img1: torch.Tensor,
    img2: torch.Tensor,
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Computes Multi-Scale SSIM (MS-SSIM) over 3 scales for 64x64 images.
    Input images must be in [0.0, 1.0].
    """
    if weights is None:
        weights = [0.0448, 0.2856, 0.6696]  # 3-scale normalized weights

    levels = len(weights)
    mcs_list = []

    for i in range(levels):
        if i == levels - 1:
            ssim_val = ssim_tensor(img1, img2, window_size=7, sigma=1.5)
            mcs_list.append(ssim_val)
        else:
            # Downsample by 2
            ssim_val = ssim_tensor(img1, img2, window_size=7, sigma=1.5)
            mcs_list.append(ssim_val)
            img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)

    # Weighted product
    overall_ms_ssim = torch.ones_like(mcs_list[0])
    for w, mcs in zip(weights, mcs_list):
        overall_ms_ssim = overall_ms_ssim * (torch.clamp(mcs, min=1e-6) ** w)

    return overall_ms_ssim


@torch.no_grad()
def compute_seed_diversity(
    generator: nn.Module,
    conditions: torch.Tensor,
    num_seeds: int = 5,
    latent_dim: int = 100,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Measures visual variation across different random latent seeds z for identical condition vectors c.
    Generates num_seeds images per condition vector, computes pairwise MS-SSIM and pixel L1 differences.

    Note: MS-SSIM measures spatial pixel-level structural similarity, not high-level semantic divergence.
    Diversity = 1.0 - mean_ms_ssim.

    Returns:
        Dict with mean_ms_ssim, seed_diversity_score (1 - MS-SSIM), and mean_pairwise_l1.
    """
    generator.eval()
    num_profiles = min(conditions.shape[0], 64)
    cond_subset = conditions[:num_profiles].to(device)

    all_ms_ssim_pairs = []
    all_l1_pairs = []

    for idx in range(num_profiles):
        c = cond_subset[idx : idx + 1]  # (1, 20)
        c_expanded = c.expand(num_seeds, -1)
        z = torch.randn(num_seeds, latent_dim, device=device)

        gen_imgs = generator(z, c_expanded)  # (K, 3, 64, 64) in [-1.0, 1.0]
        # Rescale to [0.0, 1.0] for SSIM calculation
        imgs_01 = (gen_imgs + 1.0) * 0.5
        imgs_01 = torch.clamp(imgs_01, 0.0, 1.0)

        # Pairwise comparisons
        for i in range(num_seeds):
            for j in range(i + 1, num_seeds):
                im_i = imgs_01[i : i + 1]
                im_j = imgs_01[j : j + 1]
                ms_val = ms_ssim_tensor(im_i, im_j).item()
                l1_val = torch.abs(im_i - im_j).mean().item()
                all_ms_ssim_pairs.append(ms_val)
                all_l1_pairs.append(l1_val)

    mean_ms_ssim = float(np.mean(all_ms_ssim_pairs)) if all_ms_ssim_pairs else 1.0
    diversity_score = float(max(0.0, 1.0 - mean_ms_ssim))
    mean_l1 = float(np.mean(all_l1_pairs)) if all_l1_pairs else 0.0

    return {
        "mean_pairwise_ms_ssim": round(mean_ms_ssim, 4),
        "seed_diversity_score": round(diversity_score, 4),
        "mean_pairwise_pixel_l1": round(mean_l1, 4),
        "num_evaluated_profiles": num_profiles,
        "seeds_per_profile": num_seeds,
    }

