import pytest
from pydantic import ValidationError
from schemas.attributes import (
    EvidenceState,
    AttributeCategory,
    AttributeEvidence,
    ForensicAttributeSet,
)
from schemas.requests import (
    AnalyzeDescriptionRequest,
    ValidateAttributesRequest,
    EvaluateRequest,
)


def test_attribute_evidence_schema():
    ev = AttributeEvidence(
        category=AttributeCategory.HAIR,
        state=EvidenceState.EXPLICIT,
        value="short brown hair",
        confidence=0.9,
    )
    assert ev.category == "hair"
    assert ev.state == "explicit"
    assert ev.confidence == 0.9


def test_confidence_boundary_validation():
    with pytest.raises(ValidationError):
        AttributeEvidence(
            category=AttributeCategory.EYES,
            state=EvidenceState.UNKNOWN,
            confidence=1.5,  # Out of range [0.0, 1.0]
        )


def test_forensic_attribute_set_defaults_to_unknown():
    fas = ForensicAttributeSet()
    # All attributes must start in UNKNOWN state, never silently fabricated as definite
    assert fas.face_shape.state == EvidenceState.UNKNOWN
    assert fas.hair.state == EvidenceState.UNKNOWN
    assert fas.facial_hair.state == EvidenceState.UNKNOWN
    assert fas.eyes.state == EvidenceState.UNKNOWN
    assert fas.nose.state == EvidenceState.UNKNOWN
    assert fas.lips.state == EvidenceState.UNKNOWN
    assert fas.distinguishing_marks.state == EvidenceState.UNKNOWN


def test_analyze_request_validation():
    req = AnalyzeDescriptionRequest(witness_text="Suspect had black curly hair.")
    assert req.witness_text == "Suspect had black curly hair."
    assert req.confidence_threshold == 0.5

    with pytest.raises(ValidationError):
        AnalyzeDescriptionRequest(witness_text="")  # Empty text rejected

