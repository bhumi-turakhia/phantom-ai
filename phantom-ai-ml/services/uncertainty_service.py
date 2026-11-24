import re
from typing import List, Dict, Tuple, Any, Optional
from schemas.attributes import EvidenceState, AttributeCategory, AttributeEvidence


# Explicit linguistic markers denoting uncertainty and hedging
UNCERTAINTY_CUES = [
    r"\bmaybe\b",
    r"\bpossibly\b",
    r"\bperhaps\b",
    r"\bi think\b",
    r"\bi believe\b",
    r"\bnot sure\b",
    r"\bnot certain\b",
    r"\bunsure\b",
    r"\buncertain\b",
    r"\bcould be\b",
    r"\bmight be\b",
    r"\bsort of\b",
    r"\bkind of\b",
    r"\bvaguely\b",
    r"\bhard to tell\b",
    r"\bappeared to be\b",
    r"\bappeared to have\b",
    r"\bappeared to\b",
    r"\bappears to be\b",
    r"\bappears to have\b",
    r"\bappears to\b",
    r"\bseemed to be\b",
    r"\bseemed to have\b",
    r"\bseemed to\b",
    r"\bseems to be\b",
    r"\bseems to have\b",
    r"\bseems to\b",
    r"\blooked somewhat like\b",
    r"\blooked like\b",
    r"\blooks like\b",
    r"\bseemed like\b",
    r"\bcannot recall\b",
    r"\bcould not recall\b",
    r"\bcan't recall\b",
    r"\bunable to recall\b",
    r"\bcannot remember\b",
    r"\bcould not remember\b",
    r"\bcan't remember\b",
    r"\bhard to remember\b",
    r"\bunable to remember\b",
    r"\bcannot say for sure\b",
    r"\bcan't say for sure\b",
    r"\bnot positive\b",
    r"\bnot 100% sure\b",
]

# Attribute conflict / contradiction patterns (term1, term2, category)
CONTRADICTION_PAIRS = [
    ("bald", "long hair", AttributeCategory.HAIR),
    ("bald", "ponytail", AttributeCategory.HAIR),
    ("bald", "wavy hair", AttributeCategory.HAIR),
    ("bald", "thick hair", AttributeCategory.HAIR),
    ("blond hair", "black hair", AttributeCategory.HAIR),
    ("blonde hair", "black hair", AttributeCategory.HAIR),
    ("blond hair", "brown hair", AttributeCategory.HAIR),
    ("blonde hair", "brown hair", AttributeCategory.HAIR),
    ("black hair", "brown hair", AttributeCategory.HAIR),
    ("short hair", "long hair", AttributeCategory.HAIR),
    ("clean shaven", "full beard", AttributeCategory.FACIAL_HAIR),
    ("clean-shaven", "full beard", AttributeCategory.FACIAL_HAIR),
    ("clean shaven", "goatee", AttributeCategory.FACIAL_HAIR),
    ("clean-shaven", "goatee", AttributeCategory.FACIAL_HAIR),
    ("clean shaven", "mustache", AttributeCategory.FACIAL_HAIR),
    ("clean-shaven", "mustache", AttributeCategory.FACIAL_HAIR),
    ("no beard", "thick beard", AttributeCategory.FACIAL_HAIR),
    ("no beard", "full beard", AttributeCategory.FACIAL_HAIR),
    ("no beard", "goatee", AttributeCategory.FACIAL_HAIR),
    ("blonde", "jet black", AttributeCategory.HAIR),
    ("blonde", "dark black", AttributeCategory.HAIR),
    ("blue eyes", "dark brown eyes", AttributeCategory.EYES),
    ("blue eyes", "brown eyes", AttributeCategory.EYES),
    ("green eyes", "brown eyes", AttributeCategory.EYES),
    ("round face", "angular sharp face", AttributeCategory.FACE_SHAPE),
    ("without glasses", "wire-rim glasses", AttributeCategory.OTHER_OBSERVABLE),
    ("no glasses", "wearing glasses", AttributeCategory.OTHER_OBSERVABLE),
]


def _term_is_affirmatively_present(term: str, lower_text: str) -> bool:
    """
    Checks if a term is present affirmatively (i.e. not preceded by negation like 'not bald', 'never bald').
    Terms that are inherently negative (e.g. 'no beard', 'without glasses') are exempt from this check.
    """
    if term.startswith("no ") or term.startswith("without ") or "clean" in term:
        return term in lower_text

    # Search for term with word boundaries
    pattern = rf"\b{re.escape(term)}\b"
    for m in re.finditer(pattern, lower_text):
        preceding = lower_text[max(0, m.start() - 35):m.start()]
        # Check if preceded by negation words or compound negation like 'no beard or '
        if (
            re.search(r"\b(not|never|definitely\s+not|without|no|neither)\s+$", preceding)
            or re.search(r"\b(no|neither)\s+[a-z\s-]+\s+(or|nor)\s+$", preceding)
        ):
            continue
        return True
    return False


class UncertaintyService:
    """
    Forensic uncertainty and contradiction detection engine.
    Ensures that unknown attributes are never silently converted into definite traits.
    """

    def detect_uncertainty_cues(self, text: str) -> List[str]:
        """
        Scans text for hedging and uncertainty markers.
        """
        cues_found = []
        lower_text = text.lower()
        for cue_pattern in UNCERTAINTY_CUES:
            if re.search(cue_pattern, lower_text):
                cues_found.append(cue_pattern.replace(r"\b", ""))
        return cues_found

    def detect_contradictions(self, text: str, attributes: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Detects mutually conflicting attribute claims within witness text or attributes.
        """
        contradictions = []
        lower_text = text.lower()

        for item in CONTRADICTION_PAIRS:
            term1, term2 = item[0], item[1]
            if _term_is_affirmatively_present(term1, lower_text) and _term_is_affirmatively_present(term2, lower_text):
                contradictions.append(
                    f"Contradictory statements detected in testimony regarding features: '{term1}' vs '{term2}'"
                )

        return contradictions

    def detect_contradictions_with_categories(self, text: str) -> List[Tuple[str, str, AttributeCategory]]:
        """
        Detects mutually conflicting attribute claims and maps each conflict to its target AttributeCategory.
        """
        conflicts = []
        lower_text = text.lower()
        for term1, term2, category in CONTRADICTION_PAIRS:
            if _term_is_affirmatively_present(term1, lower_text) and _term_is_affirmatively_present(term2, lower_text):
                conflicts.append((term1, term2, category))
        return conflicts

    def detect_scope_cues(self, witness_text: str, start_char: Optional[int], end_char: Optional[int]) -> List[str]:
        """
        Linguistically scopes uncertainty cues to the specific clause or governing phrase of the attribute span.
        Prevents uncertainty cues in coordinate or contrasting clauses from bleeding across independent observations.
        """
        if start_char is None or end_char is None:
            return []

        span = witness_text[start_char:end_char].lower()
        # Direct cue inside span
        for cp in UNCERTAINTY_CUES:
            if re.search(cp, span):
                return [cp.replace(r"\b", "")]

        # Determine clause boundaries (delimiters: .!? ; and coordinate contrastives like ', but', ', however')
        clause_delims = list(re.finditer(r"[.!?]+|;\s*|,\s*(?:but|however|whereas|although|while)\s*", witness_text))
        c_start = 0
        c_end = len(witness_text)
        for m in clause_delims:
            if m.end() <= start_char:
                c_start = m.end()
            elif m.start() >= end_char:
                c_end = m.start()
                break

        clause = witness_text[c_start:c_end]
        rel_start = start_char - c_start
        pre_text = clause[:rel_start]

        # If span is in a 'with ...' prepositional phrase (e.g. 'seems to have a receding hairline with brown hair'),
        # the main verb hedge governs the direct object, not the accompaniment phrase unless hedged independently.
        if re.search(r"\bwith\s+[a-z0-9\s-]*$", pre_text, re.IGNORECASE):
            m_with = re.search(r"\bwith\s+", pre_text, re.IGNORECASE)
            if m_with:
                pre_text = pre_text[m_with.end():]

        # Check for explicit certainty assertions (e.g. 'definitely', 'certainly', 'without doubt')
        if re.search(r"\b(definitely|certainly|positively|without\s+doubt)\b", pre_text, re.IGNORECASE):
            return []

        cues_found = []
        for cp in UNCERTAINTY_CUES:
            if re.search(cp, pre_text, re.IGNORECASE):
                cues_found.append(cp.replace(r"\b", ""))
        # Check if preceding clause was a pure epistemic hedge (e.g. 'It was hard to tell, but...', 'Not sure, but...')
        if c_start > 0:
            prev_chunk = witness_text[:c_start].strip().rstrip(",;").strip().lower()
            # If prev_chunk matches an introductory hedge, include its cues
            for cp in UNCERTAINTY_CUES:
                if re.search(cp, prev_chunk):
                    # Check if prev_chunk is predominantly the hedge itself
                    cleaned = re.sub(r"^(it\s+was\s+|i\s*('m|am)\s+|i\s+)", "", prev_chunk)
                    if any(re.search(cp, cleaned) for cp in [r"\bhard to tell\b", r"\bnot sure\b", r"\bnot certain\b", r"\bunsure\b"]):
                        cues_found.append(cp.replace(r"\b", ""))
                        break

        return cues_found

    def resolve_attribute_state(
        self,
        category: AttributeCategory,
        extracted_value: Optional[str],
        source_sentence: str,
        confidence: float,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
        extracted_by: str = "nlp_model",
    ) -> AttributeEvidence:
        """
        Determines the evidence state: explicit, uncertain, unknown, or contradictory.
        Enforces rule: Unknown attributes MUST remain UNKNOWN with 0 confidence.
        Preserves exact character offsets and verbatim spans.
        """
        # Rule 1: Unknown features cannot be hallucinated into definite features
        if not extracted_value or not extracted_value.strip():
            return AttributeEvidence(
                category=category,
                state=EvidenceState.UNKNOWN,
                value=None,
                confidence=None,
                source_spans=[],
                start_char=None,
                end_char=None,
                extracted_by="system_default",
                uncertainty_cues=[],
                notes="Feature was not observed or mentioned in witness testimony."
            )

        cues = self.detect_uncertainty_cues(source_sentence)

        # Rule 2: If uncertainty markers surround the attribute, classify as UNCERTAIN
        if cues:
            return AttributeEvidence(
                category=category,
                state=EvidenceState.UNCERTAIN,
                value=extracted_value.strip(),
                confidence=None,
                source_spans=[extracted_value.strip()],
                start_char=start_char,
                end_char=end_char,
                extracted_by=extracted_by,
                uncertainty_cues=cues,
                notes=f"Witness expressed uncertainty: {', '.join(cues)}."
            )

        # Rule 3: Confirmed definite observation
        return AttributeEvidence(
            category=category,
            state=EvidenceState.EXPLICIT,
            value=extracted_value.strip(),
            confidence=None,
            source_spans=[extracted_value.strip()],
            start_char=start_char,
            end_char=end_char,
            extracted_by=extracted_by,
            uncertainty_cues=[],
            notes="Feature observed with definite witness assertion."
        )

