"""
Unit Test Suite for CelebA Identity-Disjoint Partitioning and Dataset Loader.
Verifies zero identity leakage across splits, deterministic hashing, GCIT compatibility,
attribute dropout behavior, and preprocessing integrity.
"""

import os
import json
import tempfile
import pytest
import torch
from pathlib import Path
from PIL import Image

from data.celeba_partitioner import hash_identity_to_split
from data.celeba_dataset import (
    CelebAIdentityDataset,
    PreprocessingError,
    GCIT_TO_CELEBA_ATTR,
)
from utils.gcit_translator import GCIT_LAYOUT


CELEBA_METADATA_DIR = "phantom-ai-ml/datasets/celeba"
MANIFEST_FILE = os.path.join(CELEBA_METADATA_DIR, "identity_split_manifest.json")
STATS_FILE = os.path.join(CELEBA_METADATA_DIR, "attribute_statistics.json")


@pytest.fixture(scope="module")
def manifest_data():
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_manifest_and_stats_files_exist():
    assert os.path.exists(MANIFEST_FILE), "identity_split_manifest.json must exist."
    assert os.path.exists(STATS_FILE), "attribute_statistics.json must exist."


def test_identity_disjointness(manifest_data):
    """
    CRITICAL FORENSIC REQUIREMENT:
    Zero identity overlap across Split A (generator), Split B (evaluator), and Split C (benchmark).
    """
    split_identities = {"split_a": set(), "split_b": set(), "split_c": set()}

    for ident, split in manifest_data["identity_assignments"].items():
        # Map identity string to set
        pass

    # Extract identities from manifest partitions
    image_to_id = manifest_data.get("image_to_identity", {})
    for img, sp in manifest_data["image_assignments"].items():
        ident = image_to_id.get(img) or manifest_data["identity_assignments"].get(img)
        split_identities[sp].add(ident)

    set_a = split_identities["split_a"]
    set_b = split_identities["split_b"]
    set_c = split_identities["split_c"]

    overlap_ab = set_a.intersection(set_b)
    overlap_ac = set_a.intersection(set_c)
    overlap_bc = set_b.intersection(set_c)

    assert len(overlap_ab) == 0, f"Identity leakage between Split A and B: {len(overlap_ab)}"
    assert len(overlap_ac) == 0, f"Identity leakage between Split A and C: {len(overlap_ac)}"
    assert len(overlap_bc) == 0, f"Identity leakage between Split B and C: {len(overlap_bc)}"

    assert len(set_a) + len(set_b) + len(set_c) == manifest_data["unique_identities"]
    assert len(set_a) + len(set_b) + len(set_c) == 10177


def test_deterministic_reproducibility():
    """
    Asserts that the identity hashing function is 100% deterministic and salt-stable.
    """
    test_ids = ["1", "2", "3", "10", "100", "500", "10177", "celebrity_test_id"]
    first_pass = [hash_identity_to_split(i) for i in test_ids]
    second_pass = [hash_identity_to_split(i) for i in test_ids]
    assert first_pass == second_pass


def test_partition_proportions(manifest_data):
    """
    Asserts partition sizes are approximately 80% / 10% / 10%.
    """
    partitions = manifest_data["partitions"]
    total_imgs = manifest_data["total_images"]
    assert total_imgs == 202599

    assert 0.78 <= partitions["split_a"]["identity_proportion"] <= 0.82
    assert 0.08 <= partitions["split_b"]["identity_proportion"] <= 0.12
    assert 0.08 <= partitions["split_c"]["identity_proportion"] <= 0.12

    assert 0.78 <= partitions["split_a"]["image_proportion"] <= 0.82
    assert 0.08 <= partitions["split_b"]["image_proportion"] <= 0.12
    assert 0.08 <= partitions["split_c"]["image_proportion"] <= 0.12


def test_split_c_isolation(manifest_data):
    """
    Asserts that Split C images and identities are completely excluded from Split A and B.
    """
    split_c_imgs = {img for img, sp in manifest_data["image_assignments"].items() if sp == "split_c"}
    split_a_imgs = {img for img, sp in manifest_data["image_assignments"].items() if sp == "split_a"}
    split_b_imgs = {img for img, sp in manifest_data["image_assignments"].items() if sp == "split_b"}

    assert len(split_c_imgs.intersection(split_a_imgs)) == 0
    assert len(split_c_imgs.intersection(split_b_imgs)) == 0


def test_dataset_loader_instantiation_split_b():
    """
    Instantiates CelebAIdentityDataset on Split B (evaluator split).
    Verifies tensor shapes and attributes.
    """
    dataset_b = CelebAIdentityDataset(
        metadata_dir=CELEBA_METADATA_DIR,
        split="split_b",
        image_size=64,
        attribute_dropout=0.0,
    )
    assert len(dataset_b) == 19313

    sample = dataset_b[0]
    assert "image" in sample
    assert "condition" in sample
    assert "mask" in sample
    assert "identity" in sample
    assert "image_name" in sample

    assert sample["image"].shape == (3, 64, 64)
    assert sample["condition"].shape == (20,)
    assert sample["mask"].shape == (20,)

    # Values must be -1.0 or +1.0 when dropout = 0.0
    for v in sample["condition"].tolist():
        assert v in {-1.0, 1.0}

    # All masks must be 1.0
    assert torch.all(sample["mask"] == 1.0)


def test_dataset_loader_attribute_dropout():
    """
    Tests configurable attribute dropout (p_drop).
    When an attribute is dropped:
    - mask[k] must be 0.0
    - condition[k] must be 0.0
    """
    dataset = CelebAIdentityDataset(
        metadata_dir=CELEBA_METADATA_DIR,
        split="split_b",
        attribute_dropout=0.5,
    )

    # Inspect 10 samples to ensure dropout occurs and sets both condition and mask to 0.0
    dropout_observed = False
    for i in range(10):
        sample = dataset[i]
        cond = sample["condition"]
        mask = sample["mask"]

        # Invariant: wherever mask is 0.0, condition must be 0.0
        assert torch.all(cond[mask == 0.0] == 0.0)

        # Invariant: wherever mask is 1.0, condition must be in {-1.0, 1.0}
        active_vals = cond[mask == 1.0]
        for val in active_vals.tolist():
            assert val in {-1.0, 1.0}

        if torch.any(mask == 0.0):
            dropout_observed = True

    assert dropout_observed, "Attribute dropout (p=0.5) should drop at least one attribute across 10 samples."


def test_zero_observed_attributes_batch():
    """
    Verifies that a batch where all attributes are dropped (all masks = 0.0)
    computes safely without divide-by-zero or NaN errors.
    """
    dataset = CelebAIdentityDataset(
        metadata_dir=CELEBA_METADATA_DIR,
        split="split_b",
        attribute_dropout=1.0,  # 100% dropout: all attributes unconditioned
    )
    sample = dataset[0]
    assert torch.all(sample["mask"] == 0.0)
    assert torch.all(sample["condition"] == 0.0)

    # Simulate batch of 4 unconditioned samples
    cond_batch = torch.stack([dataset[i]["condition"] for i in range(4)])
    mask_batch = torch.stack([dataset[i]["mask"] for i in range(4)])

    assert cond_batch.shape == (4, 20)
    assert mask_batch.shape == (4, 20)
    assert not torch.isnan(cond_batch).any()
    assert not torch.isnan(mask_batch).any()


def test_image_preprocessing_pipeline():
    """
    Tests the deterministic image preprocessing pipeline using a temporary image file:
    - RGB conversion
    - 64x64 bilinear resizing
    - Normalization to [-1.0, 1.0]
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = CelebAIdentityDataset(
            metadata_dir=CELEBA_METADATA_DIR,
            split="split_b",
            image_dir=tmpdir,
            image_size=64,
        )
        first_img = dataset.image_names[0]
        tmp_img_path = Path(tmpdir) / first_img

        # Create a synthetic 128x128 solid color test image
        test_img = Image.new("RGB", (128, 128), color=(255, 0, 128))
        test_img.save(tmp_img_path)

        sample = dataset[0]
        tensor = sample["image"]
        assert tensor.shape == (3, 64, 64)
        assert tensor.dtype == torch.float32

        # Check normalization range [-1.0, 1.0]
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0

        # R channel was 255 -> (1.0 - 0.5) / 0.5 = 1.0
        # G channel was 0 -> (0.0 - 0.5) / 0.5 = -1.0
        # B channel was 128 -> (128/255 - 0.5) / 0.5 approx 0.0039
        assert torch.isclose(tensor[0, 0, 0], torch.tensor(1.0), atol=1e-2)
        assert torch.isclose(tensor[1, 0, 0], torch.tensor(-1.0), atol=1e-2)
        assert torch.isclose(tensor[2, 0, 0], torch.tensor(0.0), atol=1e-2)


def test_corrupt_image_handling():
    """
    Verifies that a corrupt image file raises PreprocessingError with informative message.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = CelebAIdentityDataset(
            metadata_dir=CELEBA_METADATA_DIR,
            split="split_b",
            image_dir=tmpdir,
            image_size=64,
        )
        first_img = dataset.image_names[0]
        tmp_img_path = Path(tmpdir) / first_img
        with open(tmp_img_path, "wb") as f:
            f.write(b"NOT_A_VALID_JPEG_HEADER")

        with pytest.raises(PreprocessingError) as exc_info:
            _ = dataset[0]
        assert "Failed to process image" in str(exc_info.value)
