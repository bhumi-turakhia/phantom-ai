from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from schemas.attributes import (
    AttributeCategory,
    EvidenceState,
    FORBIDDEN_DEMOGRAPHIC_CATEGORIES,
)


class AttributeAnnotation(BaseModel):
    """
    Structured annotation of a single facial attribute observation in a witness description.
    Enforces exact span boundaries and character indexing.
    """
    annotation_id: str = Field(description="Unique identifier for this annotation record")
    category: AttributeCategory = Field(description="Observable facial attribute category")
    attribute_value: Optional[str] = Field(
        default=None,
        description="Standardized or extracted attribute value (None if unknown)"
    )
    source_span: Optional[str] = Field(
        default=None,
        description="Exact verbatim substring from the witness description (None if unknown)"
    )
    start_char: Optional[int] = Field(
        default=None,
        description="0-based start character offset in original witness text (None if unknown)"
    )
    end_char: Optional[int] = Field(
        default=None,
        description="0-based end character offset (exclusive) in original witness text (None if unknown)"
    )
    certainty: EvidenceState = Field(
        default=EvidenceState.UNKNOWN,
        description="Certainty level: explicit, uncertain, unknown, or contradictory"
    )
    annotator_id: str = Field(
        description="Identifier of the annotator or automated pipeline"
    )
    annotation_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp of annotation creation"
    )
    uncertainty_cues: List[str] = Field(
        default_factory=list,
        description="Linguistic hedging cues associated with this attribute"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Auditing notes or reasoning for this annotation"
    )

    model_config = ConfigDict(use_enum_values=True)

    @field_validator("category")
    @classmethod
    def validate_non_demographic(cls, v: Any) -> Any:
        cat_str = v.value if isinstance(v, AttributeCategory) else str(v).lower()
        if cat_str in FORBIDDEN_DEMOGRAPHIC_CATEGORIES:
            raise ValueError(
                f"Forbidden attribute category '{cat_str}'. Demographic and sensitive "
                "attributes must never be annotated or inferred."
            )
        return v

    @model_validator(mode="after")
    def validate_span_consistency(self) -> "AttributeAnnotation":
        # Rule: UNKNOWN state must not have character spans or non-null attribute values
        if self.certainty == EvidenceState.UNKNOWN.value:
            if self.source_span is not None and self.source_span.strip() != "":
                raise ValueError(
                    f"Annotation '{self.annotation_id}': UNKNOWN evidence state cannot have "
                    f"a non-empty source_span ('{self.source_span}')."
                )
            if self.start_char is not None or self.end_char is not None:
                raise ValueError(
                    f"Annotation '{self.annotation_id}': UNKNOWN evidence state must have "
                    "None for start_char and end_char."
                )
        else:
            # For EXPLICIT, UNCERTAIN, and CONTRADICTORY, offsets and spans must be specified
            if self.source_span is None or not self.source_span.strip():
                raise ValueError(
                    f"Annotation '{self.annotation_id}': State '{self.certainty}' requires "
                    "a non-empty source_span."
                )
            if self.start_char is None or self.end_char is None:
                raise ValueError(
                    f"Annotation '{self.annotation_id}': State '{self.certainty}' requires "
                    "valid start_char and end_char offsets."
                )
            if self.start_char < 0:
                raise ValueError(
                    f"Annotation '{self.annotation_id}': start_char ({self.start_char}) cannot be negative."
                )
            if self.end_char <= self.start_char:
                raise ValueError(
                    f"Annotation '{self.annotation_id}': end_char ({self.end_char}) must be "
                    f"strictly greater than start_char ({self.start_char})."
                )
        return self


class SampleMetadata(BaseModel):
    """
    Dataset sample provenance, licensing, and synthetic identification metadata.
    """
    dataset_version: str = Field(default="1.0.0", description="Dataset semantic version")
    source: str = Field(description="Origin source of sample (e.g. Synthetic Benchmark)")
    license: str = Field(default="CC-BY-4.0-Synthetic", description="Data licensing terms")
    is_synthetic: bool = Field(
        default=True,
        description="Whether this sample is synthetic data (MUST be True for generated test data)"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 creation timestamp"
    )
    notes: Optional[str] = Field(default=None, description="Additional provenance notes")

    model_config = ConfigDict(use_enum_values=True)


class DatasetSample(BaseModel):
    """
    A single witness description sample with its associated attribute annotations.
    Preserves the original witness description text verbatim without modification.
    """
    sample_id: str = Field(description="Unique sample identifier (e.g. SYNTH-001)")
    witness_text: str = Field(
        min_length=1,
        description="Verbatim witness description text. Must never be altered."
    )
    annotations: List[AttributeAnnotation] = Field(
        default_factory=list,
        description="List of attribute annotations for this sample"
    )
    metadata: SampleMetadata = Field(
        description="Sample metadata including version, license, and synthetic flag"
    )

    model_config = ConfigDict(use_enum_values=True)

    def verify_offsets(self) -> List[str]:
        """
        Verifies that all annotation character offsets correspond exactly
        to the verbatim witness description text.
        Returns a list of error messages (empty if all valid).
        """
        errors = []
        for ann in self.annotations:
            if ann.certainty == EvidenceState.UNKNOWN.value:
                continue
            if ann.start_char is None or ann.end_char is None:
                errors.append(
                    f"Sample {self.sample_id}, Annotation {ann.annotation_id}: missing offsets."
                )
                continue
            if ann.end_char > len(self.witness_text):
                errors.append(
                    f"Sample {self.sample_id}, Annotation {ann.annotation_id}: "
                    f"end_char ({ann.end_char}) exceeds witness text length ({len(self.witness_text)})."
                )
                continue
            actual_slice = self.witness_text[ann.start_char:ann.end_char]
            if actual_slice != ann.source_span:
                errors.append(
                    f"Sample {self.sample_id}, Annotation {ann.annotation_id}: "
                    f"Character span mismatch! Offset [{ann.start_char}:{ann.end_char}] yields "
                    f"'{actual_slice}', expected '{ann.source_span}'."
                )
        return errors


class ForensicDataset(BaseModel):
    """
    Complete dataset container with provenance, license, version, and samples list.
    """
    name: str = Field(description="Dataset name (e.g. Phantom AI Synthetic Forensic Benchmark)")
    version: str = Field(default="1.0.0", description="Dataset semantic version")
    license: str = Field(default="CC-BY-4.0-Synthetic", description="Dataset licensing terms")
    source: str = Field(description="Data provenance source")
    is_synthetic: bool = Field(
        default=True,
        description="Flag denoting synthetic sample collection"
    )
    samples: List[DatasetSample] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    model_config = ConfigDict(use_enum_values=True)


class DatasetValidationSummary(BaseModel):
    """
    Comprehensive quality control summary report.
    """
    is_valid: bool
    total_samples: int
    valid_samples: int
    invalid_samples: int
    total_annotations: int
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]

