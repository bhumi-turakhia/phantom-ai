from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class EvidenceState(str, Enum):
    EXPLICIT = "explicit"           # Definite statement from witness
    NEGATED = "negated"             # Explicitly negated statement ("no beard", "not bald") [Phase 3C.2]
    UNCERTAIN = "uncertain"         # Hedged statement ("maybe", "possibly", "I think")
    UNKNOWN = "unknown"             # Not mentioned or observed
    CONTRADICTORY = "contradictory" # Mutually conflicting statements in testimony


class AttributeCategory(str, Enum):
    FACE_SHAPE = "face_shape"
    HAIR = "hair"
    HAIRLINE = "hairline"
    EYES = "eyes"
    EYEBROWS = "eyebrows"
    NOSE = "nose"
    LIPS = "lips"
    FACIAL_HAIR = "facial_hair"
    DISTINGUISHING_MARKS = "distinguishing_marks"
    OTHER_OBSERVABLE = "other_observable"


# Explicitly forbidden demographic attributes to prevent biased or sensitive attribute inference
FORBIDDEN_DEMOGRAPHIC_CATEGORIES = {
    "race",
    "ethnicity",
    "skin_color",
    "complexion_race",
    "religion",
    "gender",
    "sex",
    "sexual_orientation",
    "caste",
    "nationality",
    "political_affiliation",
}


class AttributeEvidence(BaseModel):
    category: AttributeCategory
    state: EvidenceState
    value: Optional[str] = None
    confidence: Optional[float] = Field(ge=0.0, le=1.0, default=None)
    source_spans: List[str] = Field(default_factory=list)
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    extracted_by: Optional[str] = "nlp_model"  # 'nlp_model', 'negation_safety_rule', 'investigator'
    uncertainty_cues: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    model_config = ConfigDict(use_enum_values=True)


class ForensicAttributeSet(BaseModel):
    face_shape: AttributeEvidence = Field(
        default_factory=lambda: AttributeEvidence(category=AttributeCategory.FACE_SHAPE, state=EvidenceState.UNKNOWN)
    )
    hair: AttributeEvidence = Field(
        default_factory=lambda: AttributeEvidence(category=AttributeCategory.HAIR, state=EvidenceState.UNKNOWN)
    )
    hairline: AttributeEvidence = Field(
        default_factory=lambda: AttributeEvidence(category=AttributeCategory.HAIRLINE, state=EvidenceState.UNKNOWN)
    )
    facial_hair: AttributeEvidence = Field(
        default_factory=lambda: AttributeEvidence(category=AttributeCategory.FACIAL_HAIR, state=EvidenceState.UNKNOWN)
    )
    eyes: AttributeEvidence = Field(
        default_factory=lambda: AttributeEvidence(category=AttributeCategory.EYES, state=EvidenceState.UNKNOWN)
    )
    eyebrows: AttributeEvidence = Field(
        default_factory=lambda: AttributeEvidence(category=AttributeCategory.EYEBROWS, state=EvidenceState.UNKNOWN)
    )
    nose: AttributeEvidence = Field(
        default_factory=lambda: AttributeEvidence(category=AttributeCategory.NOSE, state=EvidenceState.UNKNOWN)
    )
    lips: AttributeEvidence = Field(
        default_factory=lambda: AttributeEvidence(category=AttributeCategory.LIPS, state=EvidenceState.UNKNOWN)
    )
    distinguishing_marks: AttributeEvidence = Field(
        default_factory=lambda: AttributeEvidence(category=AttributeCategory.DISTINGUISHING_MARKS, state=EvidenceState.UNKNOWN)
    )
    other_observable: AttributeEvidence = Field(
        default_factory=lambda: AttributeEvidence(category=AttributeCategory.OTHER_OBSERVABLE, state=EvidenceState.UNKNOWN)
    )
