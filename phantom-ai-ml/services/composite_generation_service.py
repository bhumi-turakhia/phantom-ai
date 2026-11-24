"""
Composite Generation Service for Phantom AI 3.0.
Integrates the frozen Phase 5.4 Conditional WGAN-GP generator with FastAPI.
Translates investigator-approved forensic evidence into GCIT conditioning vectors,
validates against contradictions and physical incompatibilities,
and synthesizes reproducible 64x64 facial composite variations.
"""

import os
import io
import time
import base64
import hashlib
import threading
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone

import torch
import numpy as np
from PIL import Image

from config.gan_config import WGANConfig, get_device
from models.wgan_gp_model import ConditionalWGANGPGenerator, NativeConditionalWGANGPGenerator128
from utils.gcit_translator import (
    GCIT_LAYOUT,
    translate_evidence_to_gcit,
    TranslationConflictError,
    IncompatibleAttributeError,
)
from schemas.attributes import EvidenceState, AttributeEvidence, AttributeCategory
from schemas.requests import (
    GenerateCompositeRequest,
    GenerateCompositeResponse,
    CompositeVariationItem,
    EditCompositeRequest,
    EditCompositeResponse,
    LatentTraversalItem,
    LatentDirectionsResponse,
)
from inference.generate_composite import normalize_category


EXPECTED_GENERATOR_SHA256 = "a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964"


class CheckpointUnavailableError(RuntimeError):
    """Raised when generator checkpoint is missing or corrupt."""
    pass


class CompositeGenerationService:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        # Prefer upgraded Native 128x128 checkpoint if available; fall back gracefully to HD 128, Phase 128 or Phase 5.4 64x64
        if checkpoint_path is None or checkpoint_path == "phantom-ai-ml/checkpoints/cgan_generator_phase54.pt":
            candidates = [
                "phantom-ai-ml/checkpoints/cgan_generator_128_native.pt",
                "checkpoints/cgan_generator_128_native.pt",
                "phantom-ai-ml/checkpoints/cgan_generator_128_hd.pt",
                "checkpoints/cgan_generator_128_hd.pt",
                "phantom-ai-ml/checkpoints/cgan_generator_128.pt",
                "checkpoints/cgan_generator_128.pt",
                "phantom-ai-ml/checkpoints/cgan_generator_phase54.pt",
                "checkpoints/cgan_generator_phase54.pt",
            ]
            chosen = "phantom-ai-ml/checkpoints/cgan_generator_phase54.pt"
            for c in candidates:
                if os.path.exists(c):
                    chosen = c
                    break
            self.checkpoint_path = chosen
        else:
            self.checkpoint_path = checkpoint_path

        self.device = device or get_device("auto")
        self.model: Optional[ConditionalWGANGPGenerator] = None
        self.model_metadata: Dict[str, Any] = {}
        self.is_loaded: bool = False
        self._inference_lock = threading.Lock()
        self.latent_directions: Dict[str, torch.Tensor] = {}
        self.latent_directions_metadata: Dict[str, Any] = {}

        self._load_model()
        self._load_latent_directions()

    def _load_latent_directions(self) -> None:
        """Loads Phase 5.7 empirically validated latent traversal directions."""
        candidates = [
            os.path.join(os.path.dirname(self.checkpoint_path), "latent_directions_phase57.pt"),
            os.path.join("phantom-ai-ml", "checkpoints", "latent_directions_phase57.pt"),
            os.path.join("checkpoints", "latent_directions_phase57.pt"),
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    ckpt = torch.load(p, map_location="cpu", weights_only=False)
                    self.latent_directions = ckpt.get("validated_directions", {})
                    self.latent_directions_metadata = ckpt.get("validation_metrics", {})
                    break
                except Exception:
                    pass

    def _load_model(self) -> None:
        if not os.path.exists(self.checkpoint_path):
            self.is_loaded = False
            return

        # Check cryptographic hash
        h = hashlib.sha256()
        with open(self.checkpoint_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        actual_sha256 = h.hexdigest()

        # Cryptographic integrity assertion for frozen Phase 5.4 checkpoint
        EXPECTED_CGAN_SHA256 = "a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964"
        if "cgan_generator_phase54" in os.path.basename(self.checkpoint_path):
            if actual_sha256 != EXPECTED_CGAN_SHA256:
                raise CheckpointUnavailableError(
                    f"Integrity check failed for Phase 5.4 CGAN checkpoint: expected {EXPECTED_CGAN_SHA256}, got {actual_sha256}"
                )

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except Exception:
            # Fall back to CPU if target accelerator fails during load
            self.device = torch.device("cpu")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        cfg_dict = checkpoint.get("config", {})
        config = WGANConfig.from_dict(cfg_dict) if cfg_dict else WGANConfig()
        sd = checkpoint["generator_state_dict"]
        is_native = (
            checkpoint.get("architecture") == "NativeConditionalWGANGPGenerator128"
            or checkpoint.get("config", {}).get("is_native_128", False)
            or "stage5.0.weight" in sd
            or "head.0.weight" in sd
        )

        if is_native:
            generator = NativeConditionalWGANGPGenerator128(
                latent_dim=config.latent_dim,
                condition_dim=config.condition_dim,
                condition_embed_dim=config.condition_embed_dim,
                feature_maps=config.generator_features,
                channels=config.channels,
                image_size=128,
            ).to(self.device)
        else:
            generator = ConditionalWGANGPGenerator(
                latent_dim=config.latent_dim,
                condition_dim=config.condition_dim,
                condition_embed_dim=config.condition_embed_dim,
                feature_maps=config.generator_features,
                channels=config.channels,
                image_size=config.image_size,
            ).to(self.device)

        generator.load_state_dict(sd, strict=True)
        generator.eval()
        for p in generator.parameters():
            p.requires_grad = False

        self.model = generator
        self.is_loaded = True
        self.model_metadata = {
            "model_name": "NativeConditionalWGANGPGenerator128" if is_native else "ConditionalWGANGPGenerator",
            "architecture": (
                "Native 128x128 Conditional WGAN-GP (Direct Transposed Convolutions + Residual Blocks)"
                if is_native
                else f"Conditional WGAN-GP with Projection Conditioning ({config.image_size}x{config.image_size})"
            ),
            "phase": "Native 128x128 Generative Rebuild" if is_native else "Phase 128 High-Resolution Quality Upgrade" if config.image_size >= 128 else "Phase 5.4 / 5.5 / 5.6",
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": actual_sha256,
            "best_epoch": checkpoint.get("step", checkpoint.get("epoch", 4)),
            "parameter_count": sum(p.numel() for p in generator.parameters()),
            "latent_dim": config.latent_dim,
            "condition_dim": config.condition_dim,
            "output_resolution": f"{config.image_size}x{config.image_size}",
            "device": str(self.device),
        }

    def generate_composites(
        self,
        request: GenerateCompositeRequest,
    ) -> GenerateCompositeResponse:
        t0 = time.time()

        if not self.is_loaded or self.model is None:
            # Attempt reload once in case checkpoint was added
            self._load_model()
            if not self.is_loaded or self.model is None:
                raise CheckpointUnavailableError(
                    f"Trained Conditional WGAN-GP generator checkpoint unavailable at '{self.checkpoint_path}'. "
                    "Ensure Phase 5.4 checkpoint is in place."
                )

        # 1. Parse and validate the approved forensic attribute profile
        raw_profile = request.forensic_profile
        if not raw_profile or not isinstance(raw_profile, dict):
            raise ValueError("forensic_profile must be a valid dictionary of attribute evidence.")

        typed_profile: Dict[str, AttributeEvidence] = {}
        for cat_key, raw_val in raw_profile.items():
            norm_cat = normalize_category(cat_key)
            if isinstance(raw_val, AttributeEvidence):
                typed_profile[norm_cat.value] = raw_val
            elif isinstance(raw_val, dict):
                state_str = str(raw_val.get("state", "explicit")).lower().strip()
                try:
                    state_enum = EvidenceState(state_str)
                except ValueError:
                    state_enum = EvidenceState.EXPLICIT

                val_str = raw_val.get("value")
                if val_str is not None:
                    val_str = str(val_str).strip()

                typed_profile[norm_cat.value] = AttributeEvidence(
                    category=norm_cat,
                    value=val_str,
                    state=state_enum,
                    source_spans=raw_val.get("source_spans", []),
                    source_text=raw_val.get("source_text", ""),
                )
            elif isinstance(raw_val, str):
                typed_profile[norm_cat.value] = AttributeEvidence(
                    category=norm_cat,
                    value=raw_val.strip(),
                    state=EvidenceState.EXPLICIT,
                    source_text=raw_val.strip(),
                )

        # 2. Translate into GCIT condition vector and mask
        c_tensor, m_tensor, warnings = translate_evidence_to_gcit(typed_profile)

        # Identify conditioned vs unconditioned attributes
        idx_to_dim = {idx: name for name, idx in GCIT_LAYOUT.items()}
        conditioned_attrs = []
        unconditioned_attrs = []
        for idx in range(len(GCIT_LAYOUT)):
            dim_name = idx_to_dim[idx]
            if m_tensor[idx].item() > 0.5:
                val = c_tensor[idx].item()
                sign = "+" if val > 0 else "-"
                conditioned_attrs.append(f"{dim_name} ({sign}1.0)")
            else:
                unconditioned_attrs.append(dim_name)

        # 3. Seed and variations bounds validation
        if request.seed is not None:
            if not isinstance(request.seed, int) or request.seed < 0 or request.seed >= 2**31 - 1:
                raise ValueError("Seed must be a non-negative integer less than 2^31 - 1.")

        num_vars = max(1, min(int(request.num_variations), 8))
        base_seed = request.seed if request.seed is not None else int(time.time() * 1000) % (2**31 - 1)
        seeds = [(base_seed + var_idx * 7919) % (2**31 - 1) for var_idx in range(num_vars)]

        # 4. Construct batched noise and conditioning tensors
        z_list = [
            torch.randn(1, self.model.latent_dim, generator=torch.Generator(device="cpu").manual_seed(s), dtype=torch.float32)
            for s in seeds
        ]
        z_batch = torch.cat(z_list, dim=0).to(self.device)
        c_batch = c_tensor.unsqueeze(0).repeat(num_vars, 1).to(self.device)
        m_batch = m_tensor.unsqueeze(0).repeat(num_vars, 1).to(self.device)

        # 5. Thread-safe batched forward pass with fallback
        with self._inference_lock:
            try:
                with torch.inference_mode():
                    fake_batch = self.model(z_batch, c_batch, m_batch)
                    # Vectorized uint8 scaling on device prior to host copy
                    imgs_uint8 = ((fake_batch.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
                    imgs_np = imgs_uint8.permute(0, 2, 3, 1).cpu().numpy()
            except RuntimeError as err:
                if self.device.type != "cpu":
                    warnings.append(f"Accelerator '{self.device}' failed ({err}); completed synthesis on CPU fallback.")
                    self.model.to("cpu")
                    try:
                        with torch.inference_mode():
                            fake_batch = self.model(z_batch.to("cpu"), c_batch.to("cpu"), m_batch.to("cpu"))
                            imgs_uint8 = ((fake_batch.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
                            imgs_np = imgs_uint8.permute(0, 2, 3, 1).numpy()
                    finally:
                        try:
                            self.model.to(self.device)
                        except Exception:
                            # If accelerator cannot re-acquire model (e.g. permanent OOM), demote primary device to CPU
                            self.device = torch.device("cpu")
                else:
                    raise

        # 6. Fast lossless PNG encoding
        variations: List[CompositeVariationItem] = []
        for var_idx in range(num_vars):
            pil_img = Image.fromarray(imgs_np[var_idx])
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG", compress_level=1)
            img_bytes = buffer.getvalue()
            b64_str = base64.b64encode(img_bytes).decode("utf-8")
            data_url = f"data:image/png;base64,{b64_str}"

            variations.append(
                CompositeVariationItem(
                    variation_index=var_idx + 1,
                    seed=seeds[var_idx],
                    image_base64=b64_str,
                    data_url=data_url,
                    width=pil_img.width,
                    height=pil_img.height,
                )
            )

        # Clear MPS memory cache if available to prevent fragmentation
        if self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        latency = (time.time() - t0) * 1000.0

        return GenerateCompositeResponse(
            status="success",
            generator=self.model_metadata,
            variations=variations,
            conditioned_attributes=conditioned_attrs,
            unconditioned_attributes=unconditioned_attrs,
            warnings=warnings,
            latency_ms=round(latency, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_latent_directions_info(self) -> LatentDirectionsResponse:
        """
        Returns the discovery and evaluation status of latent directions in R^{100}.
        Clearly distinguishes validated directions from rejected ones and unsupported features.
        """
        if not self.latent_directions:
            self._load_latent_directions()

        total = len(GCIT_LAYOUT)
        validated = list(self.latent_directions.keys())
        rejected = [name for name in GCIT_LAYOUT.keys() if name not in validated]

        return LatentDirectionsResponse(
            status="success",
            total_dimensions=total,
            validated_count=len(validated),
            rejected_count=len(rejected),
            validated_attributes=validated,
            detailed_metrics=self.latent_directions_metadata,
            unsupported_features=[
                "smile_intensity (excluded from GCIT; not an invariant forensic identifier)",
                "facial_expression (unsupported; emotional states not biometric traits)",
                "continuous_hair_length (only bald and bangs annotated in GCIT)",
                "arbitrary_3d_geometry (no 3D mesh representation or parametric landmark controls)",
                "eye_spacing_or_jawline_geometry (unsupported fine-grained geometric attributes)",
            ],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def edit_composite(
        self,
        request: EditCompositeRequest,
    ) -> EditCompositeResponse:
        """
        Executes controlled semantic editing via:
        1. Continuous GCIT conditioning traversal (c_i in [-1.0, +1.0]).
        2. Latent space hyperplane traversal in Z (z' = z + alpha * n).
        Preserves deterministic seed, validates incompatibilities, and outputs
        both the edited composite and baseline composite for fixed-seed comparison.
        """
        t0 = time.time()

        if not self.is_loaded or self.model is None:
            self._load_model()
            if not self.is_loaded or self.model is None:
                raise CheckpointUnavailableError(
                    f"Trained Conditional WGAN-GP generator checkpoint unavailable at '{self.checkpoint_path}'."
                )

        if not self.latent_directions:
            self._load_latent_directions()

        # 1. Parse and validate baseline profile
        typed_profile: Dict[str, AttributeEvidence] = {}
        for category_key, attr_data in request.forensic_profile.items():
            cat = normalize_category(category_key)
            if isinstance(attr_data, dict):
                val = attr_data.get("value")
                state_raw = attr_data.get("evidence_state", attr_data.get("state", "explicit"))
                try:
                    state = EvidenceState(state_raw)
                except ValueError:
                    state = EvidenceState.EXPLICIT
                typed_profile[cat.value] = AttributeEvidence(
                    category=cat,
                    value=val,
                    state=state,
                )
            elif isinstance(attr_data, AttributeEvidence):
                typed_profile[cat.value] = attr_data

        c_tensor, m_tensor, warnings = translate_evidence_to_gcit(typed_profile)
        c_tensor_orig = c_tensor.clone()
        m_tensor_orig = m_tensor.clone()

        # 2. Apply Continuous GCIT Adjustments (Part A)
        applied_adjustments: Dict[str, float] = {}
        if request.continuous_gcit_adjustments:
            for attr_name, val in request.continuous_gcit_adjustments.items():
                if attr_name not in GCIT_LAYOUT:
                    raise ValueError(f"Attribute '{attr_name}' is not in GCIT_LAYOUT. Valid attributes: {list(GCIT_LAYOUT.keys())}")
                clamped_val = max(-1.0, min(1.0, float(val)))
                idx = GCIT_LAYOUT[attr_name]
                c_tensor[idx] = clamped_val
                m_tensor[idx] = 1.0
                applied_adjustments[attr_name] = round(clamped_val, 3)

        # Incompatibility check on final adjusted conditioning
        bald_idx = GCIT_LAYOUT["hair_bald"]
        if c_tensor[bald_idx] > 0.0 and m_tensor[bald_idx] > 0.5:
            for conf_attr in ["hair_bangs", "hair_wavy", "hair_straight"]:
                conf_idx = GCIT_LAYOUT[conf_attr]
                if c_tensor[conf_idx] > 0.0 and m_tensor[conf_idx] > 0.5:
                    raise IncompatibleAttributeError(
                        f"Incompatible combination: 'hair_bald' cannot co-occur with '{conf_attr}'."
                    )

        # 3. Seed validation & Base Latent Reconstruction
        if not isinstance(request.seed, int) or request.seed < 0 or request.seed >= 2**31 - 1:
            raise ValueError("Seed must be a non-negative integer less than 2^31 - 1.")

        gen = torch.Generator(device="cpu").manual_seed(request.seed)
        z_base = torch.randn(1, self.model.latent_dim, generator=gen, dtype=torch.float32)
        z_edited = z_base.clone()

        # 4. Latent Space Hyperplane Traversal in Z (Part B)
        applied_traversals: List[Dict[str, Any]] = []
        if request.latent_traversals:
            for item in request.latent_traversals:
                attr = item.attribute
                alpha = float(item.alpha)
                if abs(alpha) < 1e-5:
                    continue

                if attr not in self.latent_directions:
                    raise ValueError(
                        f"Attribute '{attr}' is not an empirically validated latent direction. "
                        f"Validated directions: {list(self.latent_directions.keys())}. "
                        "Unsupported features such as smile intensity or continuous hair length cannot be traversed."
                    )

                normal = self.latent_directions[attr].to(dtype=torch.float32, device="cpu")
                z_edited = z_edited + alpha * normal
                applied_traversals.append({
                    "attribute": attr,
                    "alpha": round(alpha, 3),
                    "unit_norm": round(normal.norm().item(), 4),
                })

        # 5. Batched Execution (Edited + Optional Baseline)
        include_orig = bool(request.include_original)
        if include_orig:
            z_batch = torch.cat([z_edited, z_base], dim=0).to(self.device)
            c_batch = torch.stack([c_tensor, c_tensor_orig], dim=0).to(self.device)
            m_batch = torch.stack([m_tensor, m_tensor_orig], dim=0).to(self.device)
        else:
            z_batch = z_edited.to(self.device)
            c_batch = c_tensor.unsqueeze(0).to(self.device)
            m_batch = m_tensor.unsqueeze(0).to(self.device)

        with self._inference_lock:
            try:
                with torch.inference_mode():
                    fake_batch = self.model(z_batch, c_batch, m_batch)
                    imgs_uint8 = ((fake_batch.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
                    imgs_np = imgs_uint8.permute(0, 2, 3, 1).cpu().numpy()
            except RuntimeError as err:
                if self.device.type != "cpu":
                    warnings.append(f"Accelerator '{self.device}' failed ({err}); completed editing on CPU fallback.")
                    self.model.to("cpu")
                    try:
                        with torch.inference_mode():
                            fake_batch = self.model(z_batch.to("cpu"), c_batch.to("cpu"), m_batch.to("cpu"))
                            imgs_uint8 = ((fake_batch.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
                            imgs_np = imgs_uint8.permute(0, 2, 3, 1).numpy()
                    finally:
                        try:
                            self.model.to(self.device)
                        except Exception:
                            self.device = torch.device("cpu")
                else:
                    raise

        # 6. Encode Lossless PNG
        def encode_png(img_arr: np.ndarray) -> Tuple[str, str]:
            pil_img = Image.fromarray(img_arr)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG", compress_level=1)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return b64, f"data:image/png;base64,{b64}"

        b64_edit, url_edit = encode_png(imgs_np[0])
        img_h, img_w = imgs_np[0].shape[0], imgs_np[0].shape[1]
        edited_item = CompositeVariationItem(
            variation_index=1,
            seed=request.seed,
            image_base64=b64_edit,
            data_url=url_edit,
            width=img_w,
            height=img_h,
        )

        orig_item = None
        if include_orig and len(imgs_np) > 1:
            b64_orig, url_orig = encode_png(imgs_np[1])
            orig_item = CompositeVariationItem(
                variation_index=0,
                seed=request.seed,
                image_base64=b64_orig,
                data_url=url_orig,
                width=img_w,
                height=img_h,
            )

        # Summarize conditioned attributes
        idx_to_dim = {idx: name for name, idx in GCIT_LAYOUT.items()}
        conditioned_attrs = []
        for idx in range(len(GCIT_LAYOUT)):
            if m_tensor[idx].item() > 0.5:
                val = c_tensor[idx].item()
                sign = "+" if val >= 0 else ""
                conditioned_attrs.append(f"{idx_to_dim[idx]} ({sign}{val:.2f})")

        latency = (time.time() - t0) * 1000.0

        return EditCompositeResponse(
            status="success",
            generator=self.model_metadata,
            edited_composite=edited_item,
            original_composite=orig_item,
            applied_continuous_adjustments=applied_adjustments,
            applied_latent_traversals=applied_traversals,
            conditioned_attributes=conditioned_attrs,
            warnings=warnings,
            latency_ms=round(latency, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

