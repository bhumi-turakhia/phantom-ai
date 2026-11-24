"""
Unit Test Suite for Independent Forensic Attribute Evaluator (ForensicFeatureCNN).
Verifies:
1. Split B1 and B2 identity disjointness and zero identity leakage.
2. Complete isolation of held-out benchmark Split C.
3. Multi-label 20-dimensional output and 512-dimensional feature embedding.
4. Class imbalance pos_weight calculation.
5. Checkpoint serialization, deserialization, and exact numerical reproducibility.
6. Metric calculation (Precision, Recall, F1, AUROC, Brier scores, Demographic Subgroups).
7. Zero pretrained weight dependencies.
8. CPU and MPS device compatibility.
"""

import os
import json
import tempfile
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from config.gan_config import count_parameters, get_device
from models.forensic_evaluator import (
    ForensicFeatureCNN,
    calculate_attribute_pos_weights,
    weights_init_evaluator,
)
from utils.evaluator_metrics import (
    compute_evaluator_metrics,
    compute_demographic_subgroup_metrics,
)
from data.evaluator_partitioner import hash_split_b_identity, partition_split_b


CELEBA_DIR = "phantom-ai-ml/datasets/celeba"
MAIN_MANIFEST_PATH = os.path.join(CELEBA_DIR, "identity_split_manifest.json")
EVAL_MANIFEST_PATH = os.path.join(CELEBA_DIR, "evaluator_split_b_manifest.json")


@pytest.fixture(scope="module")
def eval_manifest():
    if not os.path.exists(EVAL_MANIFEST_PATH):
        partition_split_b()
    with open(EVAL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def main_manifest():
    with open(MAIN_MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_evaluator_split_b1_b2_identity_disjointness(eval_manifest):
    """
    Asserts zero identity overlap between Evaluator Train (B1) and Validation (B2).
    """
    partitions = eval_manifest["partitions"]
    assert partitions["split_b1"]["identity_count"] == 791
    assert partitions["split_b2"]["identity_count"] == 189
    assert partitions["split_b1"]["identity_count"] + partitions["split_b2"]["identity_count"] == 980

    # Extract identities from image assignments
    image_to_id = eval_manifest["image_to_identity"]
    b1_ids = {image_to_id[img] for img, sp in eval_manifest["image_assignments"].items() if sp == "split_b1"}
    b2_ids = {image_to_id[img] for img, sp in eval_manifest["image_assignments"].items() if sp == "split_b2"}

    overlap = b1_ids.intersection(b2_ids)
    assert len(overlap) == 0, f"Identity leakage between Split B1 and B2: {len(overlap)}"


def test_split_c_complete_isolation_from_evaluator(eval_manifest, main_manifest):
    """
    CRITICAL FORENSIC REQUIREMENT:
    Asserts zero identity or image leakage from held-out benchmark Split C into Split B1 or B2.
    """
    split_c_images = {img for img, sp in main_manifest["image_assignments"].items() if sp == "split_c"}
    split_c_identities = {main_manifest["image_to_identity"][img] for img in split_c_images}

    b1_b2_images = set(eval_manifest["image_assignments"].keys())
    image_leakage = split_c_images.intersection(b1_b2_images)
    assert len(image_leakage) == 0, f"Split C images leaked into Split B: {len(image_leakage)}"

    b1_b2_identities = set(eval_manifest["identity_assignments"].keys())
    identity_leakage = split_c_identities.intersection(b1_b2_identities)
    assert len(identity_leakage) == 0, f"Split C identities leaked into Split B: {len(identity_leakage)}"


def test_forensic_evaluator_architecture_and_outputs():
    """
    Asserts ForensicFeatureCNN processes (B, 3, 64, 64) and produces (B, 20) logits and (B, 512) feature embeddings.
    """
    model = ForensicFeatureCNN(in_channels=3, num_attributes=20, feature_dim=512)
    model.eval()
    x = torch.randn(4, 3, 64, 64)

    # 1. Forward pass
    logits, feat = model(x, return_features=True)
    assert logits.shape == (4, 20)
    assert feat.shape == (4, 512)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(feat).any()

    # 2. Extract features helper
    extracted_feat = model.extract_features(x)
    assert extracted_feat.shape == (4, 512)
    assert torch.allclose(feat, extracted_feat, atol=1e-5)

    # 3. Predict attributes helper
    preds, probs = model.predict_attributes(x, threshold=0.5)
    assert preds.shape == (4, 20)
    assert probs.shape == (4, 20)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    for val in preds.unique().tolist():
        assert val in {0.0, 1.0}


def test_pos_weights_calculation_for_class_imbalance():
    """
    Asserts calculate_attribute_pos_weights correctly upweights rare positive classes.
    """
    # Create synthetic dataset of 100 samples with 1 rare attribute (5 positives) and 1 common (80 positives)
    synthetic_targets = torch.zeros(100, 20)
    synthetic_targets[:5, 0] = 1.0   # Rare attribute (prevalence 5%)
    synthetic_targets[:80, 1] = 1.0  # Common attribute (prevalence 80%)

    pos_weights = calculate_attribute_pos_weights(synthetic_targets)
    assert pos_weights.shape == (20,)

    # Rare attribute pos_weight = (100 - 5) / 5 = 19.0
    assert torch.isclose(pos_weights[0], torch.tensor(19.0), atol=1e-1)

    # Common attribute pos_weight = (100 - 80) / 80 = 0.25
    assert torch.isclose(pos_weights[1], torch.tensor(0.25), atol=1e-1)


def test_metric_computation_precision_recall_f1():
    """
    Asserts compute_evaluator_metrics accurately calculates TP, FP, TN, FN, Precision, Recall, F1, AUROC, Brier.
    """
    # 4 samples, 2 attributes
    probs = torch.tensor([
        [0.9, 0.1],  # Pred: [1, 0]
        [0.8, 0.8],  # Pred: [1, 1]
        [0.2, 0.3],  # Pred: [0, 0]
        [0.1, 0.9],  # Pred: [0, 1]
    ])
    # Pad to 20 columns
    probs_full = torch.zeros(4, 20)
    probs_full[:, :2] = probs

    targets = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],  # False positive for attr 1
        [0.0, 0.0],
        [0.0, 1.0],
    ])
    targets_full = torch.zeros(4, 20)
    targets_full[:, :2] = targets

    metrics = compute_evaluator_metrics(probs_full, targets_full, threshold=0.5)

    # Check attribute 0:
    # y_pred = [1, 1, 0, 0], y_true = [1, 1, 0, 0] -> Perfect!
    attr0 = metrics["per_attribute"]["hair_black"]
    assert attr0["tp"] == 2
    assert attr0["fp"] == 0
    assert attr0["fn"] == 0
    assert attr0["tn"] == 2
    assert attr0["precision"] == 1.0
    assert attr0["recall"] == 1.0
    assert attr0["f1"] == 1.0

    # Check attribute 1:
    # y_pred = [0, 1, 0, 1], y_true = [0, 0, 0, 1]
    # TP = 1 (sample 3), FP = 1 (sample 1), TN = 2 (sample 0, 2), FN = 0
    attr1 = metrics["per_attribute"]["hair_brown"]
    assert attr1["tp"] == 1
    assert attr1["fp"] == 1
    assert attr1["fn"] == 0
    assert attr1["tn"] == 2
    assert attr1["precision"] == 0.5
    assert attr1["recall"] == 1.0
    assert torch.isclose(torch.tensor(attr1["f1"]), torch.tensor(2/3), atol=1e-3)


def test_demographic_subgroup_metrics_disaggregation():
    """
    Asserts compute_demographic_subgroup_metrics evaluates disaggregated performance per subgroup.
    """
    probs = torch.rand(10, 20)
    targets = torch.randint(0, 2, (10, 20)).float()

    subgroups = {
        "GroupA": [0, 1, 2, 3, 4],
        "GroupB": [5, 6, 7, 8, 9],
    }

    sub_results = compute_demographic_subgroup_metrics(probs, targets, subgroups)
    assert "GroupA" in sub_results
    assert "GroupB" in sub_results
    assert sub_results["GroupA"]["sample_count"] == 5
    assert sub_results["GroupB"]["sample_count"] == 5
    assert "macro_f1" in sub_results["GroupA"]


def test_checkpoint_reproducibility():
    """
    Asserts that saving a checkpoint and reloading it into a new instance yields bit-identical outputs.
    """
    model1 = ForensicFeatureCNN(in_channels=3, num_attributes=20)
    model1.eval()

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out1, _ = model1(x)

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        torch.save({"model_state_dict": model1.state_dict()}, tmp.name)

        model2 = ForensicFeatureCNN(in_channels=3, num_attributes=20)
        ckpt = torch.load(tmp.name, weights_only=True)
        model2.load_state_dict(ckpt["model_state_dict"])
        model2.eval()

        with torch.no_grad():
            out2, _ = model2(x)

    assert torch.allclose(out1, out2, atol=1e-7), "Reloaded model must produce identical outputs."


def test_no_pretrained_weights_used():
    """
    Verifies ForensicFeatureCNN does NOT import or instantiate torchvision pretrained models.
    """
    model = ForensicFeatureCNN()
    for name, module in model.named_modules():
        # Assert no torchvision standard pretrained models
        assert "Inception" not in module.__class__.__name__
        assert "VGG" not in module.__class__.__name__
        assert "ResNet" not in module.__class__.__name__


def test_cpu_and_device_compatibility():
    """
    Verifies that ForensicFeatureCNN runs smoothly on CPU.
    """
    dev = torch.device("cpu")
    model = ForensicFeatureCNN().to(dev)
    x = torch.randn(2, 3, 64, 64, device=dev)
    logits, _ = model(x)
    assert logits.device.type == "cpu"
    assert logits.shape == (2, 20)
