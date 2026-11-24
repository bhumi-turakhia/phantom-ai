import pytest
from schemas.attributes import AttributeCategory, EvidenceState
from services.nlp_service import EXPECTED_PHASE3C2_SHA256
from app.main import nlp_service as default_nlp_service, uncertainty_service as default_uncertainty_service

@pytest.fixture
def nlp_service():
    return default_nlp_service

@pytest.fixture
def uncertainty_service():
    return default_uncertainty_service

def test_checkpoint_sha256_integrity(nlp_service):
    """Verify that the loaded checkpoint is strictly the Phase 3C.2 model."""
    meta = nlp_service.get_model_metadata_info()
    assert meta["checkpoint_sha256"] == EXPECTED_PHASE3C2_SHA256
    assert meta["phase"] == "3C.2"
    assert meta["is_trained"] is True

def test_explicit_attributes_and_exact_offsets(nlp_service):
    """Test explicit facial features and verify zero-drift character offsets."""
    text = "The suspect had piercing blue eyes and a sharp jawline."
    status, is_trained, attrs, contradictions, uncertainties, conf, notes = nlp_service.analyze_description(text)

    assert status == "success"
    assert is_trained is True

    # Check eyes
    eyes = attrs[AttributeCategory.EYES.value]
    assert eyes.state == EvidenceState.EXPLICIT
    assert eyes.value is not None
    assert eyes.start_char is not None and eyes.end_char is not None
    assert text[eyes.start_char:eyes.end_char] == eyes.source_spans[0]

    # Check jawline / face shape
    face = attrs[AttributeCategory.FACE_SHAPE.value]
    assert face.state == EvidenceState.EXPLICIT
    assert face.value is not None
    assert text[face.start_char:face.end_char] == face.source_spans[0]

def test_uncertain_descriptions(nlp_service):
    """Test that hedging cues convert attributes to UNCERTAIN state and degrade extraction score."""
    text = "I think he possibly had a receding hairline and maybe hazel eyes."
    status, is_trained, attrs, contradictions, uncertainties, conf, notes = nlp_service.analyze_description(text)

    assert len(uncertainties) > 0
    assert "maybe" in uncertainties or "possibly" in uncertainties or "i think" in uncertainties

    hairline = attrs[AttributeCategory.HAIRLINE.value]
    assert hairline.state == EvidenceState.UNCERTAIN
    assert hairline.confidence is None  # Artificial heuristic confidence removed in Phase 4.2

    eyes = attrs[AttributeCategory.EYES.value]
    assert eyes.state == EvidenceState.UNCERTAIN
    assert eyes.confidence is None

def test_negated_descriptions_suppress_positive_features(nlp_service):
    """Verify negation safety layer prevents negated features from becoming positive observations."""
    text = "He was clean-shaven with no beard or mustache, but had a prominent nose."
    status, is_trained, attrs, contradictions, uncertainties, conf, notes = nlp_service.analyze_description(text)

    facial_hair = attrs[AttributeCategory.FACIAL_HAIR.value]
    assert facial_hair.state == EvidenceState.NEGATED
    assert facial_hair.value is None  # Feature MUST be suppressed from active reconstruction
    assert facial_hair.extracted_by == "negation_safety_rule"
    assert text[facial_hair.start_char:facial_hair.end_char] == facial_hair.source_spans[0]

def test_distinguishing_marks_extraction(nlp_service):
    """Test extraction of distinguishing scars, marks, or tattoos."""
    text = "The individual displayed an aquiline nose and a notched left eyebrow."
    status, is_trained, attrs, contradictions, uncertainties, conf, notes = nlp_service.analyze_description(text)

    marks = attrs[AttributeCategory.DISTINGUISHING_MARKS.value]
    assert marks.state == EvidenceState.EXPLICIT
    assert marks.value is not None
    assert text[marks.start_char:marks.end_char] == marks.source_spans[0]

def test_non_facial_observation_separation(nlp_service):
    """Test that non-facial scene observations (clothing, items) are categorized as other_observable."""
    text = "The suspect was wearing a dark blue hoodie and carrying a heavy backpack."
    status, is_trained, attrs, contradictions, uncertainties, conf, notes = nlp_service.analyze_description(text)

    other = attrs[AttributeCategory.OTHER_OBSERVABLE.value]
    assert other.state == EvidenceState.EXPLICIT
    assert "hoodie" in other.value.lower()
    assert text[other.start_char:other.end_char] == other.source_spans[0]

    # Facial categories must remain unobserved / unknown
    assert attrs[AttributeCategory.NOSE.value].state == EvidenceState.UNKNOWN
    assert attrs[AttributeCategory.FACIAL_HAIR.value].state == EvidenceState.UNKNOWN

def test_contradiction_detection(uncertainty_service):
    """Test contradictory statements detection across conflicting witness statements."""
    statement_a = "Witness A reported the suspect had a full beard and dark brown hair."
    statement_b = "Witness B stated the suspect was clean shaven."
    combined = f"{statement_a} {statement_b}"

    contradictions = uncertainty_service.detect_contradictions(combined, {})
    assert len(contradictions) > 0
    # Both statements must be preserved without majority voting
    assert "Witness A" in statement_a
    assert "Witness B" in statement_b

def test_completely_bald_not_falsely_negated(nlp_service):
    """Verify Phase 4.2 fix: 'completely bald' is affirmative hair state, not a false negation."""
    text = "The suspect was completely bald."
    status, is_trained, attrs, contradictions, uncertainties, conf, notes = nlp_service.analyze_description(text)

    hair = attrs[AttributeCategory.HAIR.value]
    # Completely bald must NOT be falsely classified as NEGATED
    assert hair.state != EvidenceState.NEGATED

def test_uncalibrated_confidence_score_properties(nlp_service):
    """Verify that artificial heuristic confidence values are removed in Phase 4.2 in favor of uninflated evidence states."""
    explicit_text = "Suspect had blue eyes and short brown hair."
    _, _, explicit_attrs, _, _, conf_explicit, _ = nlp_service.analyze_description(explicit_text)

    uncertain_text = "I think maybe he might have had blue eyes."
    _, _, uncertain_attrs, _, uncertainties, conf_uncertain, _ = nlp_service.analyze_description(uncertain_text)

    # Confidence values are suppressed to avoid misleading investigators
    assert conf_explicit is None
    assert conf_uncertain is None

    # Evidence states and linguistic uncertainty markers remain distinct
    assert explicit_attrs[AttributeCategory.EYES.value].state == EvidenceState.EXPLICIT
    assert uncertain_attrs[AttributeCategory.EYES.value].state == EvidenceState.UNCERTAIN
    assert len(uncertainties) > 0
