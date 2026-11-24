"""
Forensic Evaluator Metrics Module.
Calculates per-attribute Precision, Recall, F1, AUROC, Average Precision (PR-AUC),
Confusion Matrices, Brier Calibration Scores, and Demographic Subgroup Disaggregation.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import numpy as np
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from utils.gcit_translator import GCIT_LAYOUT


GCIT_INDEX_TO_NAME: Dict[int, str] = {idx: name for name, idx in GCIT_LAYOUT.items()}


def _fast_auroc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Pure NumPy exact AUROC computation using empirical ROC curve trapezoid integration."""
    pos_count = int(np.sum(y_true == 1))
    neg_count = int(np.sum(y_true == 0))
    if pos_count == 0 or neg_count == 0:
        return None
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    tpr = np.r_[0, tps / tps[-1]]
    fpr = np.r_[0, fps / fps[-1]]
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz_fn is not None:
        return float(trapz_fn(tpr, fpr))
    return None


def _fast_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Pure NumPy Average Precision (PR-AUC) computation."""
    pos_count = int(np.sum(y_true == 1))
    if pos_count == 0:
        return None
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_true = y_true[desc_score_indices]
    tps = np.cumsum(y_true)
    precision = tps / np.arange(1, len(y_true) + 1)
    recall = tps / tps[-1]
    auc = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]) + recall[0] * precision[0])
    return round(auc, 4)


def compute_evaluator_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes comprehensive multi-label evaluation metrics for the 20 GCIT attributes.

    Args:
        probs: (N, 20) predicted probabilities in [0.0, 1.0].
        targets: (N, 20) ground truth binary labels in {0.0, 1.0} or {-1.0, 1.0}.
        threshold: Classification decision threshold (default: 0.5).

    Returns:
        metrics: Dictionary containing per-attribute and aggregated evaluation metrics.
    """
    # Normalize targets to {0.0, 1.0} if in {-1.0, 1.0}
    if targets.min() < 0.0:
        targets_bin = ((targets + 1.0) / 2.0).float()
    else:
        targets_bin = targets.float()

    probs_np = probs.detach().cpu().numpy()
    targets_np = targets_bin.detach().cpu().numpy()
    preds_np = (probs_np >= threshold).astype(np.float32)

    n_samples, n_attrs = targets_np.shape
    per_attribute: Dict[str, Dict[str, Any]] = {}

    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    f1_list = []

    for k in range(n_attrs):
        attr_name = GCIT_INDEX_TO_NAME.get(k, f"attr_{k}")
        y_true = targets_np[:, k]
        y_pred = preds_np[:, k]
        y_prob = probs_np[:, k]

        tp = int(np.sum((y_pred == 1.0) & (y_true == 1.0)))
        fp = int(np.sum((y_pred == 1.0) & (y_true == 0.0)))
        tn = int(np.sum((y_pred == 0.0) & (y_true == 0.0)))
        fn = int(np.sum((y_pred == 0.0) & (y_true == 1.0)))

        pos_count = int(np.sum(y_true == 1.0))
        neg_count = int(np.sum(y_true == 0.0))
        prevalence = pos_count / max(n_samples, 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / n_samples if n_samples > 0 else 0.0
        brier = float(np.mean((y_prob - y_true) ** 2))

        # AUROC and Average Precision (PR-AUC)
        auroc = None
        pr_auc = None
        if pos_count > 0 and neg_count > 0:
            if SKLEARN_AVAILABLE:
                try:
                    auroc = float(roc_auc_score(y_true, y_prob))
                    pr_auc = float(average_precision_score(y_true, y_prob))
                except Exception:
                    auroc = _fast_auroc(y_true, y_prob)
                    pr_auc = _fast_pr_auc(y_true, y_prob)
            else:
                auroc = _fast_auroc(y_true, y_prob)
                pr_auc = _fast_pr_auc(y_true, y_prob)

        per_attribute[attr_name] = {
            "index": k,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "pos_count": pos_count,
            "neg_count": neg_count,
            "prevalence": round(prevalence, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "brier_score": round(brier, 4),
            "auroc": round(auroc, 4) if auroc is not None else None,
            "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        }

        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list))
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2.0 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    overall_accuracy = (total_tp + total_tn) / (n_samples * n_attrs) if (n_samples * n_attrs) > 0 else 0.0

    return {
        "n_samples": n_samples,
        "threshold": threshold,
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "micro_precision": round(micro_prec, 4),
        "micro_recall": round(micro_rec, 4),
        "overall_accuracy": round(overall_accuracy, 4),
        "per_attribute": per_attribute,
    }


def compute_demographic_subgroup_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    subgroup_indices: Dict[str, List[int]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluates Macro F1 and accuracy disaggregated across demographic intersectional buckets.
    Prevents majority group performance from masking severe failure on underrepresented demographics.
    """
    subgroup_results: Dict[str, Any] = {}

    for group_name, idx_list in subgroup_indices.items():
        if len(idx_list) == 0:
            continue
        sub_probs = probs[idx_list]
        sub_targets = targets[idx_list]

        sub_metrics = compute_evaluator_metrics(sub_probs, sub_targets, threshold=threshold)
        subgroup_results[group_name] = {
            "sample_count": len(idx_list),
            "macro_f1": sub_metrics["macro_f1"],
            "micro_f1": sub_metrics["micro_f1"],
            "overall_accuracy": sub_metrics["overall_accuracy"],
        }

    return subgroup_results

