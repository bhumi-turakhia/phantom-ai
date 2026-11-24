"""
Tests for GCIT translation layer:
Verifies evidence-to-dataset translation, polarity-aware mapping, unconditioned feature isolation,
and conflict detection.
"""

import pytest
import torch

from schemas.attributes import EvidenceState, AttributeCategory, AttributeEvidence
from utils.gcit_translator import (
    translate_evidence_to_gcit,
    GCIT_LAYOUT,
    TranslationConflictError,
    IncompatibleAttributeError,
)


def test_no_beard_inverted_polarity_translation():
    """
    Mandatory test:
    - Witness says 'no beard' (EvidenceState.NEGATED).
    - CelebA No_Beard source annotation is positive (+1) for absence of beard.
    - GCIT facial_no_beard must be +1.0 with mask = 1.0.
    - Unobserved mustache, goatee, and stubble dimensions must remain strictly 0.0 with mask = 0.0.
    """
    profile = {
        AttributeCategory.FACIAL_HAIR.value: AttributeEvidence(
            category=AttributeCategory.FACIAL_HAIR,
            state=EvidenceState.NEGATED,
            value=None,  # Negated facial hair ("no beard")
            notes="Witness stated suspect had no beard.",
        )
    }

    c_gen, mask, warnings = translate_evidence_to_gcit(profile)

    # Inverted polarity check: No_Beard in CelebA is positive for absence of beard!
    idx_no_beard = GCIT_LAYOUT["facial_no_beard"]
    assert c_gen[idx_no_beard] == 1.0
    assert mask[idx_no_beard] == 1.0

    # Crucial check: mustache, goatee, sideburns, stubble must NOT be altered or assumed clean-shaven
    assert c_gen[GCIT_LAYOUT["facial_mustache"]] == 0.0
    assert mask[GCIT_LAYOUT["facial_mustache"]] == 0.0

    assert c_gen[GCIT_LAYOUT["facial_goatee"]] == 0.0
    assert mask[GCIT_LAYOUT["facial_goatee"]] == 0.0

    assert c_gen[GCIT_LAYOUT["facial_stubble"]] == 0.0
    assert mask[GCIT_LAYOUT["facial_stubble"]] == 0.0

    assert c_gen[GCIT_LAYOUT["facial_sideburns"]] == 0.0
    assert mask[GCIT_LAYOUT["facial_sideburns"]] == 0.0


def test_normal_polarity_negation_not_bald():
    """
    Test normal polarity negation:
    - Witness says 'not bald' (EvidenceState.NEGATED, value='bald').
    - CelebA Bald is normal polarity (presence = +1, absence = -1).
    - GCIT hair_bald must be -1.0 with mask = 1.0.
    - Hair texture/bangs must remain 0.0 with mask = 0.0 (does not invent long or styled hair).
    """
    profile = {
        AttributeCategory.HAIR.value: AttributeEvidence(
            category=AttributeCategory.HAIR,
            state=EvidenceState.NEGATED,
            value="bald",
        )
    }

    c_gen, mask, warnings = translate_evidence_to_gcit(profile)

    idx_bald = GCIT_LAYOUT["hair_bald"]
    assert c_gen[idx_bald] == -1.0
    assert mask[idx_bald] == 1.0

    # Must NOT invent bangs or straight texture
    assert c_gen[GCIT_LAYOUT["hair_bangs"]] == 0.0
    assert mask[GCIT_LAYOUT["hair_bangs"]] == 0.0
    assert c_gen[GCIT_LAYOUT["hair_straight"]] == 0.0
    assert mask[GCIT_LAYOUT["hair_straight"]] == 0.0


def test_explicit_mustache_does_not_affect_beard_or_stubble():
    """
    Test that explicit mustache does not set clean-shaven or beard presence.
    """
    profile = {
        AttributeCategory.FACIAL_HAIR.value: AttributeEvidence(
            category=AttributeCategory.FACIAL_HAIR,
            state=EvidenceState.EXPLICIT,
            value="thick dark mustache",
        )
    }

    c_gen, mask, warnings = translate_evidence_to_gcit(profile)

    assert c_gen[GCIT_LAYOUT["facial_mustache"]] == 1.0
    assert mask[GCIT_LAYOUT["facial_mustache"]] == 1.0

    # Beard and stubble remain unconditioned
    assert c_gen[GCIT_LAYOUT["facial_no_beard"]] == 0.0
    assert mask[GCIT_LAYOUT["facial_no_beard"]] == 0.0
    assert c_gen[GCIT_LAYOUT["facial_stubble"]] == 0.0
    assert mask[GCIT_LAYOUT["facial_stubble"]] == 0.0


def test_unknown_and_uncertain_states_are_strictly_unconditioned():
    """
    Test that UNKNOWN and UNCERTAIN states produce exact 0.0 and mask = 0.0.
    """
    profile = {
        AttributeCategory.NOSE.value: AttributeEvidence(
            category=AttributeCategory.NOSE,
            state=EvidenceState.UNCERTAIN,
            value="maybe a pointy nose",
        ),
        AttributeCategory.HAIRLINE.value: AttributeEvidence(
            category=AttributeCategory.HAIRLINE,
            state=EvidenceState.UNKNOWN,
            value=None,
        ),
    }

    c_gen, mask, warnings = translate_evidence_to_gcit(profile)

    assert c_gen[GCIT_LAYOUT["nose_pointy"]] == 0.0
    assert mask[GCIT_LAYOUT["nose_pointy"]] == 0.0
    assert c_gen[GCIT_LAYOUT["nose_big"]] == 0.0
    assert mask[GCIT_LAYOUT["nose_big"]] == 0.0

    assert c_gen[GCIT_LAYOUT["hairline_receding"]] == 0.0
    assert mask[GCIT_LAYOUT["hairline_receding"]] == 0.0


def test_contradictory_state_hard_blocks_generation():
    """
    Test that active contradictory evidence raises TranslationConflictError,
    preventing execution until resolved by investigator.
    """
    profile = {
        AttributeCategory.HAIR.value: AttributeEvidence(
            category=AttributeCategory.HAIR,
            state=EvidenceState.CONTRADICTORY,
            value="blonde hair vs dark hair",
        )
    }

    with pytest.raises(TranslationConflictError) as exc_info:
        translate_evidence_to_gcit(profile)

    assert "unresolved contradictory testimony" in str(exc_info.value).lower()


def test_impossible_attribute_combination_rejected():
    """
    Test that physically impossible combinations (e.g. bald + bangs) raise IncompatibleAttributeError.
    """
    profile = {
        AttributeCategory.HAIR.value: AttributeEvidence(
            category=AttributeCategory.HAIR,
            state=EvidenceState.EXPLICIT,
            value="completely bald with long bangs",
        )
    }

    with pytest.raises(IncompatibleAttributeError) as exc_info:
        translate_evidence_to_gcit(profile)

    assert "physically impossible" in str(exc_info.value).lower() or "contradictory" in str(exc_info.value).lower()


def test_bald_with_straight_hair_emits_advisory_warning():
    """
    Test Phase 5.2 empirical update:
    In CelebA, bald co-occurs with straight hair in 71 images (receding hair / fringe).
    It should NOT crash as physically impossible; it should succeed and emit an advisory warning.
    """
    profile = {
        AttributeCategory.HAIR.value: AttributeEvidence(
            category=AttributeCategory.HAIR,
            state=EvidenceState.EXPLICIT,
            value="bald with straight hair on sides",
        )
    }

    c_gen, mask, warnings = translate_evidence_to_gcit(profile)
    assert c_gen[GCIT_LAYOUT["hair_bald"]] == 1.0
    assert c_gen[GCIT_LAYOUT["hair_straight"]] == 1.0
    assert any("76 cases" in w or "receding hairline" in w for w in warnings)


def test_wavy_hair_emits_advisory_limitation_warning():
    """
    Test that wavy/curly hair triggers limitation warning regarding CelebA's lack of afro/tight coil annotations.
    """
    profile = {
        AttributeCategory.HAIR.value: AttributeEvidence(
            category=AttributeCategory.HAIR,
            state=EvidenceState.EXPLICIT,
            value="curly afro hair",
        )
    }

    c_gen, mask, warnings = translate_evidence_to_gcit(profile)

    assert c_gen[GCIT_LAYOUT["hair_wavy"]] == 1.0
    assert mask[GCIT_LAYOUT["hair_wavy"]] == 1.0
    assert any("tightly coiled" in w.lower() or "waves only" in w.lower() for w in warnings)


def test_unannotated_modalities_emit_advisory_warnings():
    """
    Test that unannotated attributes (eye color, scars) remain unconditioned and emit advisory warnings.
    """
    profile = {
        AttributeCategory.EYES.value: AttributeEvidence(
            category=AttributeCategory.EYES,
            state=EvidenceState.EXPLICIT,
            value="hazel blue eyes",
        ),
        AttributeCategory.DISTINGUISHING_MARKS.value: AttributeEvidence(
            category=AttributeCategory.DISTINGUISHING_MARKS,
            state=EvidenceState.EXPLICIT,
            value="scar across left eyebrow",
        ),
    }

    c_gen, mask, warnings = translate_evidence_to_gcit(profile)

    # All GCIT values for unannotated modalities remain strictly 0.0 with mask = 0.0
    assert (c_gen == 0.0).all()
    assert (mask == 0.0).all()
    assert len(warnings) == 2
    assert any("eye color is unannotated" in w.lower() for w in warnings)
    assert any("distinguishing marks" in w.lower() for w in warnings)

