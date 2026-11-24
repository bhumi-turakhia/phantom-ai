from schemas.attributes import AttributeCategory, EvidenceState
from services.uncertainty_service import UncertaintyService
from services.evidence_service import EvidenceService


def test_uncertainty_cue_detection():
    service = UncertaintyService()
    text = "The suspect was maybe thirty years old, I think he had short hair."
    cues = service.detect_uncertainty_cues(text)
    assert len(cues) >= 2
    assert any("maybe" in c for c in cues)
    assert any("i think" in c for c in cues)


def test_contradiction_detection():
    service = UncertaintyService()
    text = "The witness first stated the suspect was bald, but then claimed long hair was visible."
    contradictions = service.detect_contradictions(text, {})
    assert len(contradictions) >= 1
    assert "bald" in contradictions[0] and "long hair" in contradictions[0]


def test_unknown_attribute_preservation():
    service = UncertaintyService()
    # Unmentioned attribute must NEVER be converted into a definite attribute
    evidence = service.resolve_attribute_state(
        category=AttributeCategory.FACIAL_HAIR,
        extracted_value=None,
        source_sentence="",
        confidence=0.0
    )
    assert evidence.state == EvidenceState.UNKNOWN
    assert evidence.value is None
    assert evidence.confidence is None or evidence.confidence == 0.0


def test_evidence_service_validation():
    uncertainty_service = UncertaintyService()
    evidence_service = EvidenceService(uncertainty_service=uncertainty_service)

    # Valid payload
    is_valid, errors, contradictions, states = evidence_service.validate_attribute_payload({
        "hair": "short black hair",
        "eyes": "dark brown eyes",
    })
    assert is_valid is True
    assert len(errors) == 0

    # Contradictory payload
    is_valid_contra, errors_c, contradictions_c, _ = evidence_service.validate_attribute_payload({
        "hair": "bald long wavy",
    })
    assert is_valid_contra is False
    assert len(contradictions_c) >= 1

