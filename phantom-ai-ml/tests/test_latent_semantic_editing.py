"""
Unit and Integration Test Suite for Phase 5.7 Semantic Editing and Latent Traversal.
Tests:
1. Continuous GCIT conditioning traversal.
2. Latent-space hyperplane traversal in Z (validated directions only).
3. Rejection of unvalidated directions and incompatible attributes.
4. Fixed-seed reproducibility and before-and-after baseline comparison.
5. FastAPI endpoints (/latent-directions and /edit-composite).
"""

import os
import pytest
import torch
from fastapi.testclient import TestClient

from app.main import app
from services.composite_generation_service import (
    CompositeGenerationService,
    IncompatibleAttributeError,
)
from schemas.requests import (
    EditCompositeRequest,
    LatentTraversalItem,
)
from utils.gcit_translator import GCIT_LAYOUT

client = TestClient(app)


def test_latent_directions_file_and_metadata():
    """Asserts Phase 5.7 directions file exists, is valid, and contains the 5 validated directions."""
    service = CompositeGenerationService()
    dirs_info = service.get_latent_directions_info()

    assert dirs_info.status == "success"
    assert dirs_info.total_dimensions == 20
    assert dirs_info.validated_count == 5
    assert dirs_info.rejected_count == 15

    expected_validated = {"hair_blond", "hair_gray", "hair_bangs", "facial_goatee", "glasses"}
    assert set(dirs_info.validated_attributes) == expected_validated

    # Verify unit normal vector property
    for attr in expected_validated:
        normal = service.latent_directions[attr]
        assert normal.shape == (100,)
        norm_val = normal.norm().item()
        assert abs(norm_val - 1.0) < 1e-4, f"Direction for {attr} must be unit normal, got {norm_val}"


def test_continuous_gcit_traversal_effect():
    """Asserts that continuous GCIT adjustment modifies generator output with a fixed seed."""
    service = CompositeGenerationService(device=torch.device("cpu"))
    base_profile = {
        "hair": {"category": "hair", "value": "black wavy hair", "state": "explicit"},
    }

    req_base = EditCompositeRequest(
        forensic_profile=base_profile,
        seed=1234,
        include_original=True,
    )
    res_base = service.edit_composite(req_base)

    # Now apply continuous adjustment on nose_big = +0.8
    req_adj = EditCompositeRequest(
        forensic_profile=base_profile,
        seed=1234,
        continuous_gcit_adjustments={"nose_big": 0.8},
        include_original=True,
    )
    res_adj = service.edit_composite(req_adj)

    assert res_adj.status == "success"
    assert res_adj.applied_continuous_adjustments["nose_big"] == 0.8
    assert res_adj.edited_composite.seed == 1234
    # Image must differ from original because of the +0.8 nose adjustment
    assert res_adj.edited_composite.image_base64 != res_adj.original_composite.image_base64


def test_latent_traversal_effect():
    """Asserts that stepping along validated latent direction modifies generator output with fixed seed."""
    service = CompositeGenerationService(device=torch.device("cpu"))
    base_profile = {
        "hair": {"category": "hair", "value": "black hair", "state": "explicit"},
    }

    req_trav = EditCompositeRequest(
        forensic_profile=base_profile,
        seed=5678,
        latent_traversals=[LatentTraversalItem(attribute="glasses", alpha=1.5)],
        include_original=True,
    )
    res_trav = service.edit_composite(req_trav)

    assert res_trav.status == "success"
    assert len(res_trav.applied_latent_traversals) == 1
    assert res_trav.applied_latent_traversals[0]["attribute"] == "glasses"
    assert res_trav.applied_latent_traversals[0]["alpha"] == 1.5
    # Image must differ from original because of the latent step
    assert res_trav.edited_composite.image_base64 != res_trav.original_composite.image_base64


def test_rejection_of_unvalidated_latent_direction():
    """Asserts that attempting latent traversal on unvalidated directions is strictly rejected."""
    service = CompositeGenerationService(device=torch.device("cpu"))
    base_profile = {
        "hair": {"category": "hair", "value": "black hair", "state": "explicit"},
    }

    unvalidated_examples = ["smile_intensity", "facial_expression", "hair_bald", "hair_straight", "nose_big"]
    for attr in unvalidated_examples:
        req = EditCompositeRequest(
            forensic_profile=base_profile,
            seed=42,
            latent_traversals=[LatentTraversalItem(attribute=attr, alpha=1.0)],
        )
        with pytest.raises(ValueError, match="not an empirically validated latent direction"):
            service.edit_composite(req)


def test_rejection_of_incompatible_continuous_adjustments():
    """Asserts that continuous GCIT adjustments violating physical compatibility rules are rejected."""
    service = CompositeGenerationService(device=torch.device("cpu"))
    base_profile = {
        "hair": {"category": "hair", "value": "black hair", "state": "explicit"},
    }

    # Attempt setting bald = 1.0 and bangs = 1.0 simultaneously
    req = EditCompositeRequest(
        forensic_profile=base_profile,
        seed=42,
        continuous_gcit_adjustments={"hair_bald": 1.0, "hair_bangs": 1.0},
    )
    with pytest.raises(IncompatibleAttributeError, match="cannot co-occur"):
        service.edit_composite(req)


def test_fixed_seed_determinism_across_runs():
    """Asserts that identical seeds and adjustments yield 100% bit-exact identical images."""
    service = CompositeGenerationService(device=torch.device("cpu"))
    base_profile = {
        "hair": {"category": "hair", "value": "black hair", "state": "explicit"},
        "facial_hair": {"category": "facial_hair", "value": "clean-shaven", "state": "negated"},
    }

    req = EditCompositeRequest(
        forensic_profile=base_profile,
        seed=9999,
        continuous_gcit_adjustments={"lips_big": 0.6},
        latent_traversals=[LatentTraversalItem(attribute="hair_blond", alpha=0.8)],
        include_original=True,
    )

    res1 = service.edit_composite(req)
    res2 = service.edit_composite(req)

    assert res1.edited_composite.image_base64 == res2.edited_composite.image_base64
    assert res1.original_composite.image_base64 == res2.original_composite.image_base64


def test_fastapi_endpoints():
    """Tests GET /latent-directions and POST /edit-composite via FastAPI TestClient."""
    # 1. GET /latent-directions
    resp_dirs = client.get("/latent-directions")
    assert resp_dirs.status_code == 200
    data_dirs = resp_dirs.json()
    assert data_dirs["status"] == "success"
    assert data_dirs["validated_count"] == 5
    assert "glasses" in data_dirs["validated_attributes"]
    assert len(data_dirs["unsupported_features"]) >= 4

    # 2. POST /edit-composite (Valid)
    payload_valid = {
        "forensic_profile": {
            "hair": {"category": "hair", "value": "black hair", "state": "explicit"},
        },
        "seed": 42,
        "continuous_gcit_adjustments": {"nose_pointy": 0.7},
        "latent_traversals": [{"attribute": "hair_gray", "alpha": 1.0}],
        "include_original": True,
    }
    resp_edit = client.post("/edit-composite", json=payload_valid)
    assert resp_edit.status_code == 200
    data_edit = resp_edit.json()
    assert data_edit["status"] == "success"
    assert data_edit["edited_composite"]["width"] in (64, 128)
    assert data_edit["original_composite"] is not None

    # 3. POST /edit-composite (Unvalidated Direction Rejection)
    payload_invalid = {
        "forensic_profile": {"hair": {"value": "black", "state": "explicit"}},
        "seed": 42,
        "latent_traversals": [{"attribute": "smile_intensity", "alpha": 1.0}],
    }
    resp_invalid = client.post("/edit-composite", json=payload_invalid)
    assert resp_invalid.status_code == 400
    assert "not an empirically validated latent direction" in resp_invalid.json()["detail"]

