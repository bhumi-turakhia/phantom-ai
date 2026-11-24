from datetime import datetime, timezone
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    CHECKPOINTS_DIR,
    DATASETS_DIR,
    HOST,
    PORT,
    SERVICE_NAME,
    VERSION,
)
from utils.device import resolve_device, get_device_info, print_device_banner
from utils.seed import set_global_seed
from utils.preprocessing import TAGS

from schemas.requests import (
    HealthResponse,
    ModelStatusResponse,
    ModelStatusItem,
    AnalyzeDescriptionRequest,
    AnalyzeDescriptionResponse,
    ValidateAttributesRequest,
    ValidateAttributesResponse,
    EvaluateRequest,
    EvaluateResponse,
    ModelMetadataInfo,
    GenerateCompositeRequest,
    GenerateCompositeResponse,
    EditCompositeRequest,
    EditCompositeResponse,
    LatentDirectionsResponse,
)

from models.nlp_model import BiLSTMAttributeLabeler
from models.uncertainty_model import EvidentialUncertaintyClassifier
from models.cnn_model import ForensicFeatureCNN
from models.dcgan_model import ConditionalDCGANGenerator, ConditionalDCGANDiscriminator

from services.uncertainty_service import UncertaintyService
from services.evidence_service import EvidenceService
from services.nlp_service import NLPService
from services.evaluation_service import EvaluationService
from services.composite_generation_service import (
    CompositeGenerationService,
    CheckpointUnavailableError,
)
from utils.gcit_translator import (
    TranslationConflictError,
    IncompatibleAttributeError,
)


# Set deterministic random seed
set_global_seed(42)

# Resolve execution hardware
device = resolve_device()
device_info = get_device_info()

# Initialize core services
uncertainty_service = UncertaintyService()
evidence_service = EvidenceService(uncertainty_service=uncertainty_service)

nlp_checkpoint_path = os.path.join(CHECKPOINTS_DIR, "nlp_char_bilstm_phase3c2.pt")
nlp_service = NLPService(
    checkpoint_path=nlp_checkpoint_path,
    device=device,
    uncertainty_service=uncertainty_service,
)
evaluation_service = EvaluationService(checkpoints_dir=CHECKPOINTS_DIR)

cgan_native_candidate = os.path.join(CHECKPOINTS_DIR, "cgan_generator_128_native.pt")
cgan_hd_candidate = os.path.join(CHECKPOINTS_DIR, "cgan_generator_128_hd.pt")
cgan_128_candidate = os.path.join(CHECKPOINTS_DIR, "cgan_generator_128.pt")
if os.path.exists(cgan_native_candidate):
    cgan_checkpoint_path = cgan_native_candidate
elif os.path.exists(cgan_hd_candidate):
    cgan_checkpoint_path = cgan_hd_candidate
elif os.path.exists(cgan_128_candidate):
    cgan_checkpoint_path = cgan_128_candidate
else:
    cgan_checkpoint_path = os.path.join(CHECKPOINTS_DIR, "cgan_generator_phase54.pt")
composite_service = CompositeGenerationService(
    checkpoint_path=cgan_checkpoint_path,
    device=device,
)

# Instantiate models in memory to verify parameter sizing and architecture definitions
_nlp_model_arch = BiLSTMAttributeLabeler(vocab_size=1000, num_tags=len(TAGS))
_uncertainty_arch = EvidentialUncertaintyClassifier(input_dim=512)
_cnn_arch = ForensicFeatureCNN(image_size=64)
_dcgan_gen_arch = ConditionalDCGANGenerator(image_size=64)
_dcgan_disc_arch = ConditionalDCGANDiscriminator(image_size=64)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print_device_banner()
    yield

app = FastAPI(
    title=SERVICE_NAME,
    version=VERSION,
    description="Phantom AI custom Python ML inference and training foundation (Zero external APIs, PyTorch, Apple MPS/CUDA support)",
    lifespan=lifespan,
)

# Bind strictly to internal communications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5001", "http://127.0.0.1:5001"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Exposes service health, selected hardware acceleration device, and system status.
    """
    return HealthResponse(
        status="healthy",
        service=SERVICE_NAME,
        version=VERSION,
        device=get_device_info(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/model-status", response_model=ModelStatusResponse)
async def model_status():
    """
    Returns exact training and checkpoint status for all custom model architectures.
    Accurately indicates whether models are trained or untrained.
    """
    nlp_path = nlp_checkpoint_path
    unc_path = os.path.join(CHECKPOINTS_DIR, "uncertainty_model.pt")
    cnn_path = os.path.join(CHECKPOINTS_DIR, "cnn_feature_extractor.pt")
    dcgan_path = os.path.join(CHECKPOINTS_DIR, "dcgan_generator.pt")

    models_info = {
        "nlp_bilstm": ModelStatusItem(
            model_name="CharAwareBiLSTMAttributeLabeler",
            architecture="Character-Aware Bidirectional LSTM with Linear-Chain CRF",
            is_trained=nlp_service.is_trained,
            checkpoint_exists=os.path.exists(nlp_path),
            checkpoint_path=nlp_path,
            parameter_count=nlp_service.model.count_parameters() if nlp_service.model else _nlp_model_arch.count_parameters(),
            input_shape="(batch_size, seq_len)",
            output_shape=f"(batch_size, seq_len, num_tags={len(TAGS)})",
            notes=f"Phase 3C.2 Character-Aware BiLSTM-CRF (SHA-256: {nlp_service.model_metadata.get('sha256', 'unknown')}).",
        ),
        "uncertainty_classifier": ModelStatusItem(
            model_name="EvidentialUncertaintyClassifier",
            architecture="Evidential Deep Learning (Dirichlet Prior Network)",
            is_trained=os.path.exists(unc_path),
            checkpoint_exists=os.path.exists(unc_path),
            checkpoint_path=unc_path,
            parameter_count=_uncertainty_arch.count_parameters(),
            input_shape="(batch_size, 512)",
            output_shape="(batch_size, 7, 4)",
            notes="Quantifies epistemic uncertainty across 4 evidence states (explicit, uncertain, unknown, contradictory).",
        ),
        "cnn_feature_extractor": ModelStatusItem(
            model_name="ForensicFeatureCNN",
            architecture="4-Block Convolutional Feature Extractor with LayerNorm & LeakyReLU",
            is_trained=os.path.exists(cnn_path),
            checkpoint_exists=os.path.exists(cnn_path),
            checkpoint_path=cnn_path,
            parameter_count=_cnn_arch.count_parameters(),
            input_shape="(batch_size, 3, 64, 64)",
            output_shape="(batch_size, 256)",
            notes="Custom small CNN feature extractor for future training from scratch.",
        ),
        "dcgan_generator": ModelStatusItem(
            model_name="ConditionalDCGANGenerator",
            architecture="Deep Convolutional GAN Generator (Transposed Convolutions)",
            is_trained=os.path.exists(dcgan_path),
            checkpoint_exists=os.path.exists(dcgan_path),
            checkpoint_path=dcgan_path,
            parameter_count=_dcgan_gen_arch.count_parameters(),
            input_shape="(batch_size, z_dim=100 + cond_dim=32)",
            output_shape="(batch_size, 3, 64, 64)",
            notes="Configurable resolution generator for future conditional composite synthesis.",
        ),
        "dcgan_discriminator": ModelStatusItem(
            model_name="ConditionalDCGANDiscriminator",
            architecture="Deep Convolutional GAN Discriminator (Spectral-Norm Compatible)",
            is_trained=os.path.exists(dcgan_path),
            checkpoint_exists=os.path.exists(dcgan_path),
            checkpoint_path=dcgan_path,
            parameter_count=_dcgan_disc_arch.count_parameters(),
            input_shape="(batch_size, 4, 64, 64)",
            output_shape="(batch_size, 1)",
            notes="Discriminates real vs fake conditional composite pairs.",
        ),
        "cgan_generator": ModelStatusItem(
            model_name="ConditionalWGANGPGenerator",
            architecture="Conditional WGAN-GP with Projection Conditioning",
            is_trained=composite_service.is_loaded,
            checkpoint_exists=os.path.exists(cgan_checkpoint_path),
            checkpoint_path=cgan_checkpoint_path,
            parameter_count=composite_service.model_metadata.get("parameter_count", 4102336),
            input_shape="(batch_size, z_dim=100 + cond_embed=64)",
            output_shape="(batch_size, 3, 64, 64)",
            notes=f"Phase 5.4 Conditional WGAN-GP generator (SHA-256: {composite_service.model_metadata.get('checkpoint_sha256', 'unknown')}).",
        ),
    }

    all_trained = all(m.is_trained for m in models_info.values())

    return ModelStatusResponse(
        service=SERVICE_NAME,
        device=get_device_info(),
        models=models_info,
        all_models_trained=all_trained,
    )


@app.post("/analyze-description", response_model=AnalyzeDescriptionResponse)
async def analyze_description(request: AnalyzeDescriptionRequest):
    """
    Extracts forensic facial attributes from witness statements.
    If the neural model is untrained, explicitly returns model_not_trained status.
    Never fabricates model predictions.
    """
    status, is_trained, attributes, contradictions, uncertainties, overall_conf, notes = (
        nlp_service.analyze_description(
            witness_text=request.witness_text,
            confidence_threshold=request.confidence_threshold,
        )
    )

    model_meta = nlp_service.get_model_metadata_info()

    return AnalyzeDescriptionResponse(
        status=status,
        is_model_trained=is_trained,
        attributes=attributes,
        contradictions=contradictions,
        uncertainties=uncertainties,
        overall_confidence=overall_conf,
        audit_notes=notes,
        model=ModelMetadataInfo(**model_meta) if model_meta else None,
    )


@app.post("/validate-attributes", response_model=ValidateAttributesResponse)
async def validate_attributes(request: ValidateAttributesRequest):
    """
    Validates proposed facial attributes against forensic constraints and contradiction rules.
    """
    is_valid, errors, contradictions, verified_states = (
        evidence_service.validate_attribute_payload(request.attributes)
    )

    summary = (
        "All attributes validated successfully."
        if is_valid
        else f"Attribute validation detected {len(errors)} formatting error(s) and {len(contradictions)} contradiction(s)."
    )

    return ValidateAttributesResponse(
        valid=is_valid,
        validation_errors=errors,
        contradictions_detected=contradictions,
        verified_states=verified_states,
        summary=summary,
    )


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_model(request: EvaluateRequest):
    """
    Evaluates model performance metrics.
    If no checkpoint exists, returns explicit model_not_trained status without fabricating metrics.
    """
    return evaluation_service.evaluate_model(
        model_name=request.model_name,
        dataset_name=request.dataset_name,
    )


@app.post("/generate-composite", response_model=GenerateCompositeResponse)
async def generate_composite(request: GenerateCompositeRequest):
    """
    Synthesizes condition-consistent forensic facial composites using the frozen Phase 5.4 Conditional WGAN-GP.
    Translates investigator-approved evidence profiles into GCIT conditioning vectors, validates against
    contradictions and incompatibilities, and outputs actual base64 PNG composites.
    """
    try:
        return composite_service.generate_composites(request)
    except TranslationConflictError as err:
        raise HTTPException(status_code=409, detail=str(err))
    except IncompatibleAttributeError as err:
        raise HTTPException(status_code=422, detail=str(err))
    except CheckpointUnavailableError as err:
        raise HTTPException(status_code=503, detail=str(err))
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Generation pipeline error: {str(err)}")


@app.get("/latent-directions", response_model=LatentDirectionsResponse)
async def get_latent_directions():
    """
    Returns discovery and empirical evaluation metadata for latent directions in R^{100}.
    Clearly identifies validated vs rejected dimensions and discloses unsupported features.
    """
    try:
        return composite_service.get_latent_directions_info()
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve latent directions: {str(err)}")


@app.post("/edit-composite", response_model=EditCompositeResponse)
async def edit_composite(request: EditCompositeRequest):
    """
    Executes controlled semantic editing of an existing composite via:
    1. Continuous GCIT conditioning traversal.
    2. Latent space hyperplane traversal in Z (empirically validated directions only).
    Outputs both edited composite and baseline composite for fixed-seed side-by-side comparison.
    """
    try:
        return composite_service.edit_composite(request)
    except TranslationConflictError as err:
        raise HTTPException(status_code=409, detail=str(err))
    except IncompatibleAttributeError as err:
        raise HTTPException(status_code=422, detail=str(err))
    except CheckpointUnavailableError as err:
        raise HTTPException(status_code=503, detail=str(err))
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Semantic editing error: {str(err)}")

