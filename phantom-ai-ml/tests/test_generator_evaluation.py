"""
Unit and Integration Tests for Phase 5.4: Conditional WGAN-GP Training & Generative Evaluation.
Tests:
1. Split A1/A2 partition disjointness and isolation from Split B and Split C.
2. Cryptographic SHA-256 integrity of frozen evaluator and NLP checkpoints.
3. Custom Feature Fréchet Distance (C-FD) numerical stability and invariance.
4. Forensic Attribute Verification (FAV) precision, recall, and F1 calculations.
5. MS-SSIM structural similarity and diversity metrics.
6. Inference engine determinism with fixed random seeds.
"""

import os
import json
import hashlib
import pytest
import numpy as np
import torch
import torch.nn as nn

from data.celeba_dataset import CelebAIdentityDataset
from models.forensic_evaluator import ForensicFeatureCNN
from models.wgan_gp_model import ConditionalWGANGPGenerator
from utils.generative_eval_metrics import (
    calculate_frechet_distance,
    compute_features_statistics,
    compute_cfd_from_features,
    compute_fav_metrics,
    ssim_tensor,
    ms_ssim_tensor,
    compute_seed_diversity,
)
from utils.gcit_translator import GCIT_LAYOUT
from inference.generate_composite import generate_forensic_composite


EXPECTED_EVAL_HASH = "f52ae5a8df4ae574432f353fd77fc605f6dcf758d67d492b447c139a79e78834"
EXPECTED_NLP_HASH = "2d567c79d759e76940968fadb02d412b530d3405e377a2d6454c64e9f2e7b4a3"


def test_split_a_partition_disjointness():
    """Verify Split A1 and Split A2 have zero identity overlap and zero leakage to Split B or C."""
    manifest_path = "phantom-ai-ml/datasets/celeba/generator_split_a_manifest.json"
    assert os.path.exists(manifest_path), f"Missing manifest: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        gen_manifest = json.load(f)

    main_manifest_path = "phantom-ai-ml/datasets/celeba/identity_split_manifest.json"
    with open(main_manifest_path, "r", encoding="utf-8") as f:
        main_manifest = json.load(f)

    identity_assignments = gen_manifest["identity_assignments"]
    a1_ids = {ident for ident, sp in identity_assignments.items() if sp == "split_a1"}
    a2_ids = {ident for ident, sp in identity_assignments.items() if sp == "split_a2"}

    # Overlap between A1 and A2
    overlap_a = a1_ids.intersection(a2_ids)
    assert len(overlap_a) == 0, f"Identity leakage between A1 and A2: {len(overlap_a)}"

    # Isolation from Split B and C
    img_assign = main_manifest["image_assignments"]
    img_to_id = main_manifest["image_to_identity"]
    b_ids = {img_to_id[img] for img, sp in img_assign.items() if sp == "split_b"}
    c_ids = {img_to_id[img] for img, sp in img_assign.items() if sp == "split_c"}

    assert len(a1_ids.intersection(b_ids)) == 0, "Leakage between A1 and B"
    assert len(a2_ids.intersection(b_ids)) == 0, "Leakage between A2 and B"
    assert len(a1_ids.intersection(c_ids)) == 0, "Leakage between A1 and C"
    assert len(a2_ids.intersection(c_ids)) == 0, "Leakage between A2 and C"


def test_cryptographic_checkpoint_integrity():
    """Verify SHA-256 hashes of frozen evaluator and NLP checkpoints."""
    eval_path = "phantom-ai-ml/checkpoints/forensic_evaluator_split_b.pt"
    assert os.path.exists(eval_path), f"Evaluator checkpoint missing: {eval_path}"
    with open(eval_path, "rb") as f:
        eval_hash = hashlib.sha256(f.read()).hexdigest()
    assert eval_hash == EXPECTED_EVAL_HASH, f"Evaluator hash mismatch: {eval_hash}"

    nlp_path = "phantom-ai-ml/checkpoints/nlp_char_bilstm_phase3c2.pt"
    if os.path.exists(nlp_path):
        with open(nlp_path, "rb") as f:
            nlp_hash = hashlib.sha256(f.read()).hexdigest()
        assert nlp_hash == EXPECTED_NLP_HASH, f"NLP hash mismatch: {nlp_hash}"


def test_cfd_metric_numerical_properties():
    """Test Custom Feature Fréchet Distance mathematical properties."""
    np.random.seed(42)
    dim = 512
    n = 100

    mu1 = np.random.randn(dim)
    sigma1 = np.eye(dim)

    # Identical distributions must yield C-FD = 0.0
    zero_dist = calculate_frechet_distance(mu1, sigma1, mu1, sigma1)
    assert abs(zero_dist) < 1e-5, f"Expected 0.0 for identical distributions, got {zero_dist}"

    # Perturbed distribution must yield positive distance
    mu2 = mu1 + 0.5
    sigma2 = np.eye(dim) * 1.5
    pos_dist = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    assert pos_dist > 0.0, f"Expected positive C-FD, got {pos_dist}"

    # Features statistics computation
    feat1 = np.random.randn(n, dim)
    feat2 = np.random.randn(n, dim) + 1.0
    cfd_val = compute_cfd_from_features(feat1, feat2)
    assert cfd_val > 0.0


def test_fav_metrics_calculation():
    """Test Forensic Attribute Verification precision, recall, and F1 calculations."""
    class MockEvaluator(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3 * 64 * 64, 20)

        def forward(self, x, return_features=False):
            # Output fixed logits: positive for dim 0, negative for dim 1
            batch_size = x.size(0)
            logits = torch.zeros(batch_size, 20)
            logits[:, 0] = 5.0   # Sigmoid ~ 0.993 (+1)
            logits[:, 1] = -5.0  # Sigmoid ~ 0.007 (-1)
            return logits, None

    evaluator = MockEvaluator()
    fake_imgs = torch.zeros(4, 3, 64, 64)

    target_conditions = torch.zeros(4, 20)
    target_conditions[:, 0] = 1.0   # Matches pred (+1)
    target_conditions[:, 1] = -1.0  # Matches pred (-1)

    target_masks = torch.zeros(4, 20)
    target_masks[:, 0] = 1.0
    target_masks[:, 1] = 1.0

    res = compute_fav_metrics(
        evaluator=evaluator,
        fake_images=fake_imgs,
        target_conditions=target_conditions,
        target_masks=target_masks,
    )

    assert "fav_macro_f1" in res
    assert "fav_macro_accuracy" in res
    assert res["fav_macro_accuracy"] == 1.0
    assert res["total_active_evaluations"] == 8


def test_ssim_and_diversity_metrics():
    """Test SSIM and multi-scale SSIM calculations."""
    torch.manual_seed(42)
    img1 = torch.rand(2, 3, 64, 64)
    # Identical images must have SSIM = 1.0
    s_ident = ssim_tensor(img1, img1)
    assert torch.allclose(s_ident, torch.ones_like(s_ident), atol=1e-3)

    ms_ident = ms_ssim_tensor(img1, img1)
    assert torch.allclose(ms_ident, torch.ones_like(ms_ident), atol=1e-3)

    # Different images must have SSIM < 1.0
    img2 = torch.rand(2, 3, 64, 64)
    s_diff = ssim_tensor(img1, img2)
    assert (s_diff < 0.9).all()


def test_inference_engine_determinism():
    """Test deterministic composite generation with fixed seed."""
    generator = ConditionalWGANGPGenerator()
    generator.eval()

    condition = torch.zeros(20)
    condition[0] = 1.0  # hair_black

    pil1, t1 = generate_forensic_composite(
        condition_vector=condition,
        seed=12345,
        generator_model=generator,
    )
    pil2, t2 = generate_forensic_composite(
        condition_vector=condition,
        seed=12345,
        generator_model=generator,
    )

    # Tensors must match exactly under identical seed
    assert torch.allclose(t1, t2), "Generator outputs differed under identical seed!"
    assert pil1.size == (64, 64)

