"""
GCIT Translation Layer: Translates forensic evidence states into Generator-Conditioning Input Tensors.
Enforces polarity-aware mapping (normal vs inverted annotations) and strictly isolates unconditioned attributes.
"""

from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import torch

from schemas.attributes import EvidenceState, AttributeCategory, AttributeEvidence


class AnnotationPolarity(str, Enum):
    NORMAL = "normal"      # Positive presence = +1, Absence / Negation = -1 (e.g. Mustache, Bald)
    INVERTED = "inverted"  # Positive absence = +1, Presence = -1 (e.g. No_Beard)


# Exact 20-dimensional GCIT layout
GCIT_LAYOUT: Dict[str, int] = {
    # Hair Color (Indices 0..4)
    "hair_black": 0,
    "hair_brown": 1,
    "hair_blond": 2,
    "hair_gray": 3,
    "hair_bald": 4,
    
    # Hair Structure (Indices 5..8)
    "hair_bangs": 5,
    "hair_straight": 6,
    "hair_wavy": 7,
    "hairline_receding": 8,

    # Facial Hair (Indices 9..13)
    "facial_mustache": 9,
    "facial_goatee": 10,
    "facial_sideburns": 11,
    "facial_stubble": 12,
    "facial_no_beard": 13,

    # Facial Geometry (Indices 14..18)
    "nose_big": 14,
    "nose_pointy": 15,
    "lips_big": 16,
    "face_oval": 17,
    "face_cheekbones": 18,

    # Accessories (Index 19)
    "glasses": 19,
}

# Polarity specification: indicates how the underlying CelebA annotation behaves
GCIT_POLARITY: Dict[str, AnnotationPolarity] = {
    "hair_black": AnnotationPolarity.NORMAL,
    "hair_brown": AnnotationPolarity.NORMAL,
    "hair_blond": AnnotationPolarity.NORMAL,
    "hair_gray": AnnotationPolarity.NORMAL,
    "hair_bald": AnnotationPolarity.NORMAL,
    "hair_bangs": AnnotationPolarity.NORMAL,
    "hair_straight": AnnotationPolarity.NORMAL,
    "hair_wavy": AnnotationPolarity.NORMAL,
    "hairline_receding": AnnotationPolarity.NORMAL,
    "facial_mustache": AnnotationPolarity.NORMAL,
    "facial_goatee": AnnotationPolarity.NORMAL,
    "facial_sideburns": AnnotationPolarity.NORMAL,
    "facial_stubble": AnnotationPolarity.NORMAL,
    "facial_no_beard": AnnotationPolarity.INVERTED,  # CelebA: No_Beard (+1 = absent, -1 = present)
    "nose_big": AnnotationPolarity.NORMAL,
    "nose_pointy": AnnotationPolarity.NORMAL,
    "lips_big": AnnotationPolarity.NORMAL,
    "face_oval": AnnotationPolarity.NORMAL,
    "face_cheekbones": AnnotationPolarity.NORMAL,
    "glasses": AnnotationPolarity.NORMAL,
}


class TranslationConflictError(ValueError):
    """Raised when active contradictory evidence has not been resolved by investigator."""
    pass


class IncompatibleAttributeError(ValueError):
    """Raised when mutually exclusive attributes are specified simultaneously."""
    pass


def translate_evidence_to_gcit(
    evidence_profile: Dict[str, AttributeEvidence]
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Translates an investigator-approved forensic evidence profile into a 20-dim GCIT tensor.

    Args:
        evidence_profile: Dictionary mapping category string to AttributeEvidence.

    Returns:
        c_gen: Tensor of shape (20,) in [-1.0, 1.0], with 0.0 for unconditioned attributes.
        mask: Tensor of shape (20,) in {0.0, 1.0}, indicating actively conditioned features.
        warnings: List of advisory strings (unsupported modalities, rare combinations).
    """
    c_gen = torch.zeros(len(GCIT_LAYOUT), dtype=torch.float32)
    mask = torch.zeros(len(GCIT_LAYOUT), dtype=torch.float32)
    warnings: List[str] = []

    # 1. First pass: Scan for unresolved contradictory states
    for cat_name, ev in evidence_profile.items():
        if ev.state == EvidenceState.CONTRADICTORY:
            raise TranslationConflictError(
                f"Cannot generate composite: Category '{cat_name}' contains unresolved contradictory testimony. "
                "Investigator review and resolution required before generation can execute."
            )

    # 2. Translate explicit and negated evidence into polarity-aware GCIT values
    for cat_name, ev in evidence_profile.items():
        state = ev.state
        val_str = (ev.value or "").lower().strip()

        # UNKNOWN or UNCERTAIN states are strictly unconditioned
        if state in [EvidenceState.UNKNOWN, EvidenceState.UNCERTAIN]:
            continue

        # Handle Facial Hair Category
        if cat_name in [AttributeCategory.FACIAL_HAIR.value, "facial_hair"]:
            if state == EvidenceState.NEGATED:
                # Witness says "no beard", "clean-shaven", etc.
                # In CelebA, No_Beard is INVERTED: +1 means absence of beard!
                idx = GCIT_LAYOUT["facial_no_beard"]
                c_gen[idx] = +1.0  # CelebA No_Beard = +1 enforces absence of beard
                mask[idx] = 1.0

                if "mustache" in val_str or "no mustache" in val_str:
                    m_idx = GCIT_LAYOUT["facial_mustache"]
                    c_gen[m_idx] = -1.0  # Normal polarity: -1.0 suppresses mustache
                    mask[m_idx] = 1.0
            elif state == EvidenceState.EXPLICIT:
                if "mustache" in val_str:
                    idx = GCIT_LAYOUT["facial_mustache"]
                    c_gen[idx] = +1.0
                    mask[idx] = 1.0
                if "goatee" in val_str:
                    idx = GCIT_LAYOUT["facial_goatee"]
                    c_gen[idx] = +1.0
                    mask[idx] = 1.0
                if "sideburn" in val_str:
                    idx = GCIT_LAYOUT["facial_sideburns"]
                    c_gen[idx] = +1.0
                    mask[idx] = 1.0
                if "stubble" in val_str or "shadow" in val_str:
                    idx = GCIT_LAYOUT["facial_stubble"]
                    c_gen[idx] = +1.0
                    mask[idx] = 1.0
                if "beard" in val_str and "no beard" not in val_str and "goatee" not in val_str:
                    # Inverted polarity: CelebA No_Beard = -1 means beard is present
                    idx = GCIT_LAYOUT["facial_no_beard"]
                    c_gen[idx] = -1.0
                    mask[idx] = 1.0
                    warnings.append(
                        "CelebA does not contain a dedicated 'full beard' label; "
                        "conditioning beard presence via inverted No_Beard may yield variable beard styles."
                    )

        # Handle Hair Category
        elif cat_name in [AttributeCategory.HAIR.value, "hair"]:
            if state == EvidenceState.NEGATED:
                if "bald" in val_str:
                    # Normal polarity: -1.0 suppresses baldness
                    idx = GCIT_LAYOUT["hair_bald"]
                    c_gen[idx] = -1.0
                    mask[idx] = 1.0
            elif state == EvidenceState.EXPLICIT:
                if "bald" in val_str:
                    idx = GCIT_LAYOUT["hair_bald"]
                    c_gen[idx] = +1.0
                    mask[idx] = 1.0
                if "black" in val_str:
                    idx = GCIT_LAYOUT["hair_black"]
                    c_gen[idx] = +1.0
                    mask[idx] = 1.0
                if "brown" in val_str:
                    idx = GCIT_LAYOUT["hair_brown"]
                    c_gen[idx] = +1.0
                    mask[idx] = 1.0
                if "blond" in val_str or "blonde" in val_str:
                    idx = GCIT_LAYOUT["hair_blond"]
                    c_gen[idx] = +1.0
                    mask[idx] = 1.0
                if "gray" in val_str or "grey" in val_str or "white" in val_str:
                    idx = GCIT_LAYOUT["hair_gray"]
                    c_gen[idx] = +1.0
                    mask[idx] = 1.0
                if "straight" in val_str:
                    idx = GCIT_LAYOUT["hair_straight"]
                    c_gen[idx] = +1.0
                    mask[idx] = 1.0
                if "wavy" in val_str or "curly" in val_str:
                    idx = GCIT_LAYOUT["hair_wavy"]
                    c_gen[idx] = +1.0
                    mask[idx] = 1.0
                    if "curly" in val_str or "coiled" in val_str or "afro" in val_str:
                        warnings.append(
                            "CelebA Wavy_Hair only annotates loose waves; tightly coiled/afro hair is unsupported."
                        )
                if "bang" in val_str or "fringe" in val_str:
                    idx = GCIT_LAYOUT["hair_bangs"]
                    c_gen[idx] = +1.0
                    mask[idx] = 1.0

        # Handle Hairline Category
        elif cat_name in [AttributeCategory.HAIRLINE.value, "hairline"]:
            if state == EvidenceState.EXPLICIT and "receding" in val_str:
                idx = GCIT_LAYOUT["hairline_receding"]
                c_gen[idx] = +1.0
                mask[idx] = 1.0

        # Handle Nose Category
        elif cat_name in [AttributeCategory.NOSE.value, "nose"]:
            if state == EvidenceState.EXPLICIT:
                if "big" in val_str or "large" in val_str or "broad" in val_str:
                    idx = GCIT_LAYOUT["nose_big"]
                    c_gen[idx] = +1.0
                    mask[idx] = 1.0
                if "pointy" in val_str or "sharp" in val_str or "pointed" in val_str:
                    idx = GCIT_LAYOUT["nose_pointy"]
                    c_gen[idx] = +1.0
                    mask[idx] = 1.0
            elif state == EvidenceState.NEGATED:
                if "big" in val_str or "large" in val_str:
                    idx = GCIT_LAYOUT["nose_big"]
                    c_gen[idx] = -1.0  # Suppresses big nose
                    mask[idx] = 1.0

        # Handle Lips Category
        elif cat_name in [AttributeCategory.LIPS.value, "lips"]:
            if state == EvidenceState.EXPLICIT and ("big" in val_str or "thick" in val_str or "full" in val_str):
                idx = GCIT_LAYOUT["lips_big"]
                c_gen[idx] = +1.0
                mask[idx] = 1.0
            elif state == EvidenceState.NEGATED and ("big" in val_str or "thick" in val_str):
                idx = GCIT_LAYOUT["lips_big"]
                c_gen[idx] = -1.0  # Suppresses thick lips
                mask[idx] = 1.0

        # Handle Face Shape Category
        elif cat_name in [AttributeCategory.FACE_SHAPE.value, "face_shape"]:
            if state == EvidenceState.EXPLICIT:
                if "oval" in val_str:
                    idx = GCIT_LAYOUT["face_oval"]
                    c_gen[idx] = +1.0
                    mask[idx] = 1.0
                if "cheekbone" in val_str or "high cheekbone" in val_str:
                    idx = GCIT_LAYOUT["face_cheekbones"]
                    c_gen[idx] = +1.0
                    mask[idx] = 1.0

        # Handle Other Observable / Eyeglasses
        elif cat_name in [AttributeCategory.OTHER_OBSERVABLE.value, "other_observable"]:
            if "glass" in val_str:
                idx = GCIT_LAYOUT["glasses"]
                if state == EvidenceState.EXPLICIT:
                    c_gen[idx] = +1.0
                    mask[idx] = 1.0
                elif state == EvidenceState.NEGATED:
                    c_gen[idx] = -1.0
                    mask[idx] = 1.0

        # Unannotated Modalities (Eyes color, Scars)
        elif cat_name in [AttributeCategory.EYES.value, "eyes"]:
            if state == EvidenceState.EXPLICIT and any(c in val_str for c in ["blue", "green", "hazel", "brown"]):
                warnings.append(
                    "Eye color is unannotated in CelebA and remains unconditioned; "
                    "iris pigmentation will be drawn from empirical prior."
                )
        elif cat_name in [AttributeCategory.DISTINGUISHING_MARKS.value, "distinguishing_marks"]:
            if state == EvidenceState.EXPLICIT:
                warnings.append(
                    "Distinguishing marks (scars/tattoos) are unannotated in CelebA and remain unconditioned."
                )

    # 3. Post-translation compatibility validation
    # Updated in Phase 5.2 based on empirical co-occurrence audit:
    # 'bald' and 'bangs' is 0 in CelebA and anatomically contradictory (bangs require frontal hair).
    # 'bald' + 'straight' (71 cases) and 'bald' + 'wavy' (5 cases) exist in CelebA (receding/side fringe),
    # so they emit advisory warnings rather than hard crashes.
    if c_gen[GCIT_LAYOUT["hair_bald"]] == 1.0:
        if c_gen[GCIT_LAYOUT["hair_bangs"]] == 1.0:
            raise IncompatibleAttributeError(
                "Anatomically contradictory / physically impossible attribute combination: 'bald' cannot co-occur with 'bangs' "
                "(bangs require frontal scalp hair, which is absent in bald conditioning)."
            )
        if c_gen[GCIT_LAYOUT["hair_wavy"]] == 1.0 or c_gen[GCIT_LAYOUT["hair_straight"]] == 1.0:
            warnings.append(
                "Observed combination: 'bald' co-occurring with hair texture ('straight'/'wavy'). "
                "In CelebA this occurs in 76 cases (0.037%), typically representing fringe hair or receding hairline."
            )

    return c_gen, mask, warnings

