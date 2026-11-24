import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "device_type" in data["device"]
    assert "torch_version" in data["device"]


def test_model_status_endpoint():
    response = client.get("/model-status")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data

    # Verify all 4 custom model architectures are listed
    models = data["models"]
    assert "nlp_bilstm" in models
    assert "uncertainty_classifier" in models
    assert "cnn_feature_extractor" in models
    assert "dcgan_generator" in models
    assert "dcgan_discriminator" in models

    # Verify honest reporting: nlp_bilstm is trained; other models remain untrained
    assert models["nlp_bilstm"]["is_trained"] is True
    assert models["nlp_bilstm"]["checkpoint_exists"] is True
    assert models["nlp_bilstm"]["parameter_count"] > 0
    assert models["uncertainty_classifier"]["is_trained"] is False
    assert models["uncertainty_classifier"]["checkpoint_exists"] is False
    assert data["all_models_trained"] is False


def test_analyze_description_endpoint():
    response = client.post(
        "/analyze-description",
        json={"witness_text": "The subject had an oval face, short black hair, and brown eyes. Maybe had a small scar on the chin."}
    )
    assert response.status_code == 200
    data = response.json()

    # Model is trained: reports success and neural sequence extraction
    assert data["status"] == "success"
    assert data["is_model_trained"] is True
    assert "attributes" in data
    assert "hair" in data["attributes"]
    assert "eyes" in data["attributes"]
    assert "overall_confidence" in data
    assert "audit_notes" in data


def test_analyze_description_empty_text_validation():
    response = client.post("/analyze-description", json={"witness_text": ""})
    assert response.status_code == 422  # Unprocessable Entity from Pydantic


def test_validate_attributes_endpoint():
    # Valid request
    res_valid = client.post(
        "/validate-attributes",
        json={"attributes": {"face_shape": "oval", "hair": "short black"}}
    )
    assert res_valid.status_code == 200
    assert res_valid.json()["valid"] is True

    # Contradictory request
    res_contra = client.post(
        "/validate-attributes",
        json={"attributes": {"hair": "bald and long curly"}}
    )
    assert res_contra.status_code == 200
    assert res_contra.json()["valid"] is False
    assert len(res_contra.json()["contradictions_detected"]) > 0


def test_evaluate_endpoint_missing_checkpoint():
    # Model without checkpoint must return model_not_trained
    response = client.post("/evaluate", json={"model_name": "uncertainty"})
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "model_not_trained"
    assert data["is_trained"] is False
    assert data["metrics"] is None
    assert "No checkpoint found" in data["evaluation_details"]


def test_evaluate_endpoint_trained_nlp_checkpoint():
    # Trained model with evaluation results on held-out test split
    response = client.post("/evaluate", json={"model_name": "nlp"})
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "evaluated_on_test_split"
    assert data["is_trained"] is True
    assert data["metrics"] is not None
    assert "precision" in data["metrics"]
    assert "f1" in data["metrics"]
    assert "held-out test set" in data["evaluation_details"]
