from typing import Dict, Any, List, Tuple
from schemas.attributes import (
    AttributeCategory,
    AttributeEvidence,
    EvidenceState,
    ForensicAttributeSet,
)
from services.uncertainty_service import UncertaintyService


class EvidenceService:
    """
    Manages forensic evidence synthesis and attribute validation.
    """
    def __init__(self, uncertainty_service: UncertaintyService):
        self.uncertainty_service = uncertainty_service

    def validate_attribute_payload(self, raw_attributes: Dict[str, Any]) -> Tuple[bool, List[str], List[str], Dict[str, str]]:
        """
        Validates an incoming attribute dictionary.
        Returns: (is_valid, validation_errors, contradictions, verified_states)
        """
        errors = []
        contradictions = []
        verified_states = {}

        valid_categories = {c.value for c in AttributeCategory}

        # Check for unknown keys
        for key in raw_attributes.keys():
            if key not in valid_categories:
                errors.append(f"Unrecognized attribute category: '{key}'. Permitted categories: {sorted(list(valid_categories))}")

        # Validate values and states
        for cat in valid_categories:
            val = raw_attributes.get(cat)
            if val is None or val == "" or val == "unknown":
                verified_states[cat] = EvidenceState.UNKNOWN.value
            elif isinstance(val, dict):
                state = val.get("state", EvidenceState.UNKNOWN.value)
                verified_states[cat] = state
            else:
                verified_states[cat] = EvidenceState.EXPLICIT.value

        # Check internal contradictions
        hair_val = str(raw_attributes.get("hair", "")).lower()
        if "bald" in hair_val and ("long" in hair_val or "curly" in hair_val):
            contradictions.append("Contradiction: Subject hair cannot be simultaneously described as bald and long/curly.")

        facial_hair = str(raw_attributes.get("facial_hair", "")).lower()
        if "clean shaven" in facial_hair and ("beard" in facial_hair or "mustache" in facial_hair):
            contradictions.append("Contradiction: Subject facial hair cannot be simultaneously clean shaven and bearded.")

        is_valid = len(errors) == 0 and len(contradictions) == 0
        return is_valid, errors, contradictions, verified_states

