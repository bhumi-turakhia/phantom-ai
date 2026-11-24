from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from schemas.attributes import AttributeEvidence, ForensicAttributeSet


class ModelMetadataInfo(BaseModel):
    name: str = "CharAwareBiLSTMAttributeLabeler"
    architecture: str = "CharAwareBiLSTM-CRF"
    phase: str = "3C.2"
    checkpoint: str = "nlp_char_bilstm_phase3c2.pt"
    checkpoint_sha256: str = "2d567c79d759e76940968fadb02d412b530d3405e377a2d6454c64e9f2e7b4a3"
    device: str
    is_trained: bool = True


class AnalyzeDescriptionRequest(BaseModel):
    witness_text: str = Field(..., min_length=1, max_length=10000, description="Raw witness statement text")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    case_id: Optional[str] = Field(default=None, description="Optional associated case ID for audit logging")


class AnalyzeDescriptionResponse(BaseModel):
    status: str = Field(..., description="Execution status: e.g. success or model_not_trained")
    is_model_trained: bool
    attributes: Dict[str, AttributeEvidence]
    contradictions: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    overall_confidence: Optional[float] = Field(ge=0.0, le=1.0, default=None)
    audit_notes: str
    model: Optional[ModelMetadataInfo] = None


class ValidateAttributesRequest(BaseModel):
    attributes: Dict[str, Any] = Field(..., description="Dictionary of proposed facial attributes to validate")


class ValidateAttributesResponse(BaseModel):
    valid: bool
    validation_errors: List[str] = Field(default_factory=list)
    contradictions_detected: List[str] = Field(default_factory=list)
    verified_states: Dict[str, str] = Field(default_factory=dict)
    summary: str


class EvaluateRequest(BaseModel):
    model_name: str = Field(..., description="Target model: nlp, uncertainty, cnn, or dcgan")
    dataset_name: Optional[str] = Field(default=None, description="Dataset identifier")


class EvaluateResponse(BaseModel):
    model_name: str
    status: str
    is_trained: bool
    metrics: Optional[Dict[str, float]] = None
    evaluation_details: str


class ModelStatusItem(BaseModel):
    model_name: str
    architecture: str
    is_trained: bool
    checkpoint_exists: bool
    checkpoint_path: str
    parameter_count: int
    input_shape: str
    output_shape: str
    notes: str


class ModelStatusResponse(BaseModel):
    service: str = "Phantom AI Custom ML Service"
    device: Dict[str, Any]
    models: Dict[str, ModelStatusItem]
    all_models_trained: bool


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    device: Dict[str, Any]
    timestamp: str


class CompositeVariationItem(BaseModel):
    variation_index: int
    seed: int
    image_base64: str
    data_url: str
    width: int = 64
    height: int = 64


class GenerateCompositeRequest(BaseModel):
    forensic_profile: Dict[str, Any] = Field(
        ...,
        description="Investigator-approved forensic evidence profile mapping categories to AttributeEvidence",
    )
    seed: Optional[int] = Field(default=None, description="Optional random seed for reproducible noise")
    num_variations: int = Field(default=1, ge=1, le=8, description="Number of visual composite variations")
    case_id: Optional[str] = Field(default=None, description="Associated case ID")
    witness_id: Optional[str] = Field(default=None, description="Associated approved witness description ID")


class GenerateCompositeResponse(BaseModel):
    status: str = "success"
    generator: Dict[str, Any]
    variations: List[CompositeVariationItem]
    conditioned_attributes: List[str] = Field(default_factory=list)
    unconditioned_attributes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    latency_ms: float
    timestamp: str


class LatentTraversalItem(BaseModel):
    attribute: str = Field(..., description="Validated GCIT attribute name for latent space traversal")
    alpha: float = Field(default=0.0, ge=-3.0, le=3.0, description="Step size along unit normal vector")


class EditCompositeRequest(BaseModel):
    forensic_profile: Dict[str, Any] = Field(
        ...,
        description="Investigator-approved baseline forensic evidence profile",
    )
    seed: int = Field(..., description="Fixed random seed for deterministic latent reconstruction")
    continuous_gcit_adjustments: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="Continuous adjustments in [-1.0, +1.0] for supported GCIT dimensions",
    )
    latent_traversals: Optional[List[LatentTraversalItem]] = Field(
        default_factory=list,
        description="Latent space hyperplane traversal steps along empirically validated directions",
    )
    include_original: bool = Field(default=True, description="Whether to include original baseline composite for before-and-after comparison")
    case_id: Optional[str] = None
    witness_id: Optional[str] = None


class EditCompositeResponse(BaseModel):
    status: str = "success"
    generator: Dict[str, Any]
    edited_composite: CompositeVariationItem
    original_composite: Optional[CompositeVariationItem] = None
    applied_continuous_adjustments: Dict[str, float] = Field(default_factory=dict)
    applied_latent_traversals: List[Dict[str, Any]] = Field(default_factory=list)
    conditioned_attributes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    latency_ms: float
    timestamp: str


class LatentDirectionsResponse(BaseModel):
    status: str = "success"
    total_dimensions: int
    validated_count: int
    rejected_count: int
    validated_attributes: List[str]
    detailed_metrics: Dict[str, Any]
    unsupported_features: List[str]
    timestamp: str


