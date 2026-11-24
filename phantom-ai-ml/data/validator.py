import re
from typing import List, Dict, Any, Tuple, Set, Optional
from collections import Counter
from schemas.attributes import (
    AttributeCategory,
    EvidenceState,
    FORBIDDEN_DEMOGRAPHIC_CATEGORIES,
)
from schemas.dataset import (
    DatasetSample,
    AttributeAnnotation,
    DatasetValidationSummary,
)
from services.uncertainty_service import CONTRADICTION_PAIRS


class DatasetValidator:
    """
    Forensic Dataset Quality Control and Schema Validation Suite.
    Ensures strict adherence to forensic evidence standards, prevents demographic bias,
    validates verbatim character offsets, and detects data contamination.
    """

    def __init__(self):
        self.forbidden_categories = {c.lower() for c in FORBIDDEN_DEMOGRAPHIC_CATEGORIES}

    def validate_sample(self, sample: DatasetSample) -> Tuple[bool, List[str], List[str]]:
        """
        Validates an individual dataset sample.
        Returns: (is_valid, errors, warnings)
        """
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Check witness text
        if not sample.witness_text or not sample.witness_text.strip():
            errors.append(f"Sample '{sample.sample_id}': witness_text cannot be empty.")
            return False, errors, warnings

        # 2. Check provenance and synthetic flag
        if not sample.metadata:
            errors.append(f"Sample '{sample.sample_id}': missing sample metadata.")
        elif not sample.metadata.source:
            warnings.append(f"Sample '{sample.sample_id}': metadata source is empty.")

        # 3. Check for forbidden demographic markers
        lower_text = sample.witness_text.lower()
        for forbidden in self.forbidden_categories:
            if re.search(rf"\b{re.escape(forbidden)}\b", lower_text):
                warnings.append(
                    f"Sample '{sample.sample_id}': Witness text contains potential sensitive term '{forbidden}'. "
                    "Ensure sensitive demographic attributes are not inferred or modeled."
                )

        # 4. Validate annotations
        occupied_spans: List[Tuple[int, int, str, str]] = []  # (start, end, category, annotation_id)

        for ann in sample.annotations:
            # Check forbidden categories in annotation
            cat_str = (
                ann.category.value
                if isinstance(ann.category, AttributeCategory)
                else str(ann.category).lower()
            )
            if cat_str in self.forbidden_categories:
                errors.append(
                    f"Sample '{sample.sample_id}', Annotation '{ann.annotation_id}': "
                    f"Forbidden demographic category '{cat_str}'. Demographic attributes are strictly prohibited."
                )

            # Check UNKNOWN state
            if ann.certainty == EvidenceState.UNKNOWN.value:
                if ann.source_span is not None and ann.source_span.strip() != "":
                    errors.append(
                        f"Sample '{sample.sample_id}', Annotation '{ann.annotation_id}': "
                        f"UNKNOWN state must have empty source_span, got '{ann.source_span}'."
                    )
                if ann.start_char is not None or ann.end_char is not None:
                    errors.append(
                        f"Sample '{sample.sample_id}', Annotation '{ann.annotation_id}': "
                        f"UNKNOWN state must not specify start_char or end_char."
                    )
                continue

            # Non-unknown states must have valid offsets
            if ann.start_char is None or ann.end_char is None:
                errors.append(
                    f"Sample '{sample.sample_id}', Annotation '{ann.annotation_id}': "
                    f"State '{ann.certainty}' missing start_char or end_char."
                )
                continue

            if ann.start_char < 0:
                errors.append(
                    f"Sample '{sample.sample_id}', Annotation '{ann.annotation_id}': "
                    f"Negative start_char ({ann.start_char})."
                )
            if ann.end_char > len(sample.witness_text):
                errors.append(
                    f"Sample '{sample.sample_id}', Annotation '{ann.annotation_id}': "
                    f"end_char ({ann.end_char}) exceeds witness_text length ({len(sample.witness_text)})."
                )
            if ann.start_char >= ann.end_char:
                errors.append(
                    f"Sample '{sample.sample_id}', Annotation '{ann.annotation_id}': "
                    f"start_char ({ann.start_char}) must be strictly less than end_char ({ann.end_char})."
                )

            # Verbatim character offset verification
            if 0 <= ann.start_char < ann.end_char <= len(sample.witness_text):
                actual_text = sample.witness_text[ann.start_char:ann.end_char]
                if actual_text != ann.source_span:
                    errors.append(
                        f"Sample '{sample.sample_id}', Annotation '{ann.annotation_id}': "
                        f"Character offset mismatch! Text at [{ann.start_char}:{ann.end_char}] is "
                        f"'{actual_text}', but source_span is '{ann.source_span}'."
                    )

            # Check for overlapping spans with conflicting categories
            for prev_start, prev_end, prev_cat, prev_id in occupied_spans:
                if max(ann.start_char, prev_start) < min(ann.end_char, prev_end):
                    if prev_cat != cat_str:
                        errors.append(
                            f"Sample '{sample.sample_id}': Conflicting overlapping spans detected between "
                            f"Annotation '{ann.annotation_id}' ({cat_str}: [{ann.start_char}:{ann.end_char}]) and "
                            f"Annotation '{prev_id}' ({prev_cat}: [{prev_start}:{prev_end}])."
                        )
            occupied_spans.append((ann.start_char, ann.end_char, cat_str, ann.annotation_id))

        # 5. Check for semantic contradictions in witness text
        for item in CONTRADICTION_PAIRS:
            term1, term2 = item[0], item[1]
            if term1 in lower_text and term2 in lower_text:
                has_contradictory_flag = any(
                    ann.certainty == EvidenceState.CONTRADICTORY.value
                    for ann in sample.annotations
                )
                if not has_contradictory_flag:
                    warnings.append(
                        f"Sample '{sample.sample_id}': Text contains conflicting terms ('{term1}' vs '{term2}') "
                        "but no annotation is marked with certainty 'contradictory'."
                    )

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def validate_dataset(self, samples: List[DatasetSample]) -> DatasetValidationSummary:
        """
        Runs quality control over an entire collection of dataset samples,
        checking for duplicates, missing annotations, offset fidelity, and summary statistics.
        """
        all_errors: List[str] = []
        all_warnings: List[str] = []
        valid_count = 0
        invalid_count = 0

        # Duplicate detection structures
        exact_text_seen: Dict[str, str] = {}  # text -> sample_id
        normalized_text_seen: Dict[str, str] = {}  # norm_text -> sample_id
        duplicate_pairs: List[Tuple[str, str]] = []

        # Category and Certainty counts
        category_counter: Counter = Counter()
        certainty_counter: Counter = Counter()
        total_annotations = 0
        token_lengths: List[int] = []

        for sample in samples:
            # 1. Sample ID uniqueness check
            pass

            # 2. Duplicate detection
            norm_text = re.sub(r"\s+", " ", sample.witness_text.strip().lower())
            if sample.witness_text in exact_text_seen:
                prev_id = exact_text_seen[sample.witness_text]
                all_errors.append(
                    f"Duplicate sample detected! '{sample.sample_id}' has identical witness_text to '{prev_id}'."
                )
                duplicate_pairs.append((sample.sample_id, prev_id))
            elif norm_text in normalized_text_seen:
                prev_id = normalized_text_seen[norm_text]
                all_warnings.append(
                    f"Near-duplicate sample detected! '{sample.sample_id}' normalized text matches '{prev_id}'."
                )
            else:
                exact_text_seen[sample.witness_text] = sample.sample_id
                normalized_text_seen[norm_text] = sample.sample_id

            # 3. Individual sample validation
            is_valid, sample_errs, sample_warns = self.validate_sample(sample)
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
            all_errors.extend(sample_errs)
            all_warnings.extend(sample_warns)

            # 4. Statistics aggregation
            words = sample.witness_text.split()
            token_lengths.append(len(words))

            total_annotations += len(sample.annotations)
            for ann in sample.annotations:
                cat_str = (
                    ann.category.value
                    if isinstance(ann.category, AttributeCategory)
                    else str(ann.category).lower()
                )
                certainty_str = (
                    ann.certainty.value
                    if isinstance(ann.certainty, EvidenceState)
                    else str(ann.certainty).lower()
                )
                category_counter[cat_str] += 1
                certainty_counter[certainty_str] += 1

        avg_token_length = (
            round(sum(token_lengths) / len(token_lengths), 2)
            if token_lengths else 0.0
        )

        statistics = {
            "total_samples": len(samples),
            "valid_samples": valid_count,
            "invalid_samples": invalid_count,
            "duplicate_samples_detected": len(duplicate_pairs),
            "total_annotations": total_annotations,
            "category_distribution": dict(category_counter),
            "certainty_distribution": dict(certainty_counter),
            "avg_words_per_sample": avg_token_length,
            "all_synthetic": all(s.metadata.is_synthetic for s in samples) if samples else False,
        }

        return DatasetValidationSummary(
            is_valid=len(all_errors) == 0,
            total_samples=len(samples),
            valid_samples=valid_count,
            invalid_samples=invalid_count,
            total_annotations=total_annotations,
            errors=all_errors,
            warnings=all_warnings,
            statistics=statistics,
        )

