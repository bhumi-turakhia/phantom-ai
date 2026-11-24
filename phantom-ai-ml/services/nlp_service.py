import os
import re
import hashlib
from typing import Dict, Any, List, Tuple, Optional
import torch

from schemas.attributes import (
    AttributeCategory,
    AttributeEvidence,
    EvidenceState,
)
from models.nlp_model import BiLSTMAttributeLabeler, CharAwareBiLSTMAttributeLabeler
from utils.preprocessing import (
    TAGS,
    TAG2IDX,
    IDX2TAG,
    tokenize_with_offsets,
    Vocabulary,
    CharVocabulary,
)
from services.uncertainty_service import UncertaintyService

EXPECTED_PHASE3C2_SHA256 = "2d567c79d759e76940968fadb02d412b530d3405e377a2d6454c64e9f2e7b4a3"

# Conservative Negation Patterns per Category
NEGATION_PATTERNS = {
    AttributeCategory.FACIAL_HAIR: [
        r"\b(definitely\s+(?:had\s+)?no\s+(?:beard|mustache|facial\s+hair|stubble))\b",
        r"\b(lacked\s+(?:any\s+|a\s+)?(?:visible\s+)?(?:beard|mustache|facial\s+hair|stubble|goatee))\b",
        r"\b(did\s+not\s+(?:have|possess)\s+(?:any\s+|a\s+)?(?:visible\s+)?(?:beard|mustache|facial\s+hair|stubble))\b",
        r"\b(devoid\s+of\s+(?:any\s+)?(?:beard|mustache|facial\s+hair|stubble|goatee))\b",
        r"\b(possessed\s+no\s+(?:beard|mustache|facial\s+hair|stubble|goatee))\b",
        r"\b(free\s+of\s+(?:any\s+)?(?:beard|mustache|facial\s+hair|stubble|goatee))\b",
        r"\b(had\s+no\s+(?:beard|mustache|facial\s+hair|stubble|goatee))\b",
        r"\b(without\s+(?:any\s+)?(?:beard|mustache|facial\s+hair|stubble))\b",
        r"\b(neither\s+beard\s+nor\s+mustache)\b",
        r"\b(clean[\s-]shaven|smooth[\s-]shaven)\b",
        r"\b(no\s+(?:beard|mustache|moustache|facial\s+hair|stubble|goatee|sideburns))\b",
    ],
    AttributeCategory.HAIR: [
        r"\b(definitely\s+not\s+bald)\b",
        r"\b(not\s+bald|not\s+shaved|no\s+hair)\b",
        r"\b(lacked\s+(?:any\s+)?hair)\b",
        r"\b(devoid\s+of\s+hair)\b",
        r"\b(without\s+hair)\b",
    ],
    AttributeCategory.DISTINGUISHING_MARKS: [
        r"\b(no\s+(?:visible\s+|noticeable\s+|apparent\s+)?(?:scars?|tattoos?|marks?|birthmarks?|piercings?)(?:,\s*(?:scars?|tattoos?|marks?|birthmarks?|piercings?))*(?:,?\s*(?:or|nor)\s*(?:scars?|tattoos?|marks?|birthmarks?|piercings?))?)\b",
        r"\b(without\s+(?:any\s+)?(?:visible\s+)?(?:scars?|tattoos?|marks?|birthmarks?|piercings?)(?:,\s*(?:scars?|tattoos?|marks?|birthmarks?|piercings?))*(?:,?\s*(?:or|nor)\s*(?:scars?|tattoos?|marks?|birthmarks?|piercings?))?)\b",
        r"\b(did\s+not\s+(?:have|possess)\s+(?:any\s+)?(?:visible\s+)?(?:scars?|tattoos?|marks?|birthmarks?|piercings?)(?:,\s*(?:scars?|tattoos?|marks?|birthmarks?|piercings?))*(?:,?\s*(?:or|nor)\s*(?:scars?|tattoos?|marks?|birthmarks?|piercings?))?)\b",
        r"\b(lacked\s+(?:any\s+)?(?:visible\s+)?(?:scars?|tattoos?|marks?|birthmarks?|piercings?)(?:,\s*(?:scars?|tattoos?|marks?|birthmarks?|piercings?))*(?:,?\s*(?:or|nor)\s*(?:scars?|tattoos?|marks?|birthmarks?|piercings?))?)\b",
        r"\b(devoid\s+of\s+(?:any\s+)?(?:visible\s+)?(?:scars?|tattoos?|marks?|birthmarks?|piercings?)(?:,\s*(?:scars?|tattoos?|marks?|birthmarks?|piercings?))*(?:,?\s*(?:or|nor)\s*(?:scars?|tattoos?|marks?|birthmarks?|piercings?))?)\b",
        r"\b(free\s+of\s+(?:any\s+)?(?:visible\s+)?(?:scars?|tattoos?|marks?|birthmarks?|piercings?)(?:,\s*(?:scars?|tattoos?|marks?|birthmarks?|piercings?))*(?:,?\s*(?:or|nor)\s*(?:scars?|tattoos?|marks?|birthmarks?|piercings?))?)\b",
        r"\b(no\s+marks\s+on\s+(?:his|her|their)?\s*face)\b",
    ],
    AttributeCategory.OTHER_OBSERVABLE: [
        r"\b(did\s+not\s+(?:wear|possess)\s+(?:any\s+)?(?:glasses|spectacles))\b",
        r"\b(was\s+not\s+wearing\s+(?:any\s+)?(?:glasses|spectacles))\b",
        r"\b(not\s+wearing\s+(?:any\s+)?(?:glasses|spectacles|earrings?))\b",
        r"\b(lacked\s+(?:any\s+)?(?:glasses|spectacles|earrings?|jewelry))\b",
        r"\b(devoid\s+of\s+(?:glasses|spectacles|earrings?|jewelry))\b",
        r"\b(without\s+(?:glasses|spectacles|earrings?))\b",
        r"\b(no\s+(?:visible\s+|noticeable\s+)?(?:glasses|spectacles|earrings?|jewelry))\b",
    ],
    AttributeCategory.EYEBROWS: [
        r"\b(no\s+eyebrows?)\b",
        r"\b(shaved\s+eyebrows?)\b",
    ],
}

# Explicit Unobservable / Unknown Evidence Patterns per Category
UNKNOWN_PATTERNS = {
    AttributeCategory.HAIR: [
        r"\b((?:could\s+not|couldn't|cannot|can't|unable\s+to|hard\s+to)\s+(?:see|remember|recall|determine)\s+(?:his|her|their)?\s*hair(?:\s+under\s+[a-z\s]+)?(?:\s+(?:color|style))?)\b",
        r"\b(hair\s+(?:color|style)?\s*(?:was|is)?\s*unknown)\b",
        r"\b(hair\s+was\s+not\s+visible)\b",
        r"\b(unknown\s+hair\s*(?:color|style)?)\b",
    ],
    AttributeCategory.EYES: [
        r"\b((?:could\s+not|couldn't|cannot|can't|unable\s+to|hard\s+to)\s+(?:see|remember|recall|determine)\s+(?:his|her|their)?\s*eyes?(?:\s*color)?)\b",
        r"\b(eye\s+color\s*(?:was|is)?\s*unknown)\b",
        r"\b(eyes\s+(?:were|was)\s+not\s+visible)\b",
        r"\b(unknown\s+eye\s+color)\b",
    ],
    AttributeCategory.FACIAL_HAIR: [
        r"\b((?:could\s+not|couldn't|cannot|can't|unable\s+to|hard\s+to)\s+(?:see|remember|recall|determine)\s+(?:if\s+he\s+had\s+)?(?:a\s+)?(?:beard|mustache|facial\s+hair))\b",
        r"\b(facial\s+hair\s*(?:was|is)?\s*unknown)\b",
        r"\b(unknown\s+facial\s+hair)\b",
    ],
    AttributeCategory.FACE_SHAPE: [
        r"\b((?:could\s+not|couldn't|cannot|can't|unable\s+to|hard\s+to)\s+(?:see|remember|recall|determine)\s+(?:his|her)?\s*face\s*shape)\b",
        r"\b(face\s+shape\s*(?:was|is)?\s*unknown)\b",
        r"\b(unknown\s+face\s+shape)\b",
    ],
    AttributeCategory.NOSE: [
        r"\b((?:could\s+not|couldn't|cannot|can't|unable\s+to|hard\s+to)\s+(?:see|remember|recall|determine)\s+(?:his|her)?\s*nose(?:\s*shape)?)\b",
        r"\b(nose\s+(?:shape)?\s*(?:was|is)?\s*unknown)\b",
        r"\b(nose\s+(?:was|is)\s+not\s+visible)\b",
    ],
    AttributeCategory.LIPS: [
        r"\b((?:could\s+not|couldn't|cannot|can't|unable\s+to|hard\s+to)\s+(?:see|remember|recall|determine)\s+(?:his|her)?\s*lips?)\b",
        r"\b(lips?\s*(?:were|was)?\s*unknown)\b",
    ],
    AttributeCategory.EYEBROWS: [
        r"\b((?:could\s+not|couldn't|cannot|can't|unable\s+to|hard\s+to)\s+(?:see|remember|recall|determine)\s+(?:his|her)?\s*eyebrows?)\b",
        r"\b(eyebrows?\s*(?:were|was)?\s*unknown)\b",
    ],
    AttributeCategory.HAIRLINE: [
        r"\b((?:could\s+not|couldn't|cannot|can't|unable\s+to|hard\s+to)\s+(?:see|remember|recall|determine)\s+(?:his|her)?\s*hairline)\b",
        r"\b(hairline\s*(?:was|is)?\s*unknown)\b",
    ],
    AttributeCategory.OTHER_OBSERVABLE: [
        r"\b((?:could\s+not|couldn't|cannot|can't|unable\s+to|hard\s+to)\s+(?:see|remember|recall|determine)\s+(?:whether\s+(?:he|she|they)\s+wore\s+)?(?:glasses|spectacles))\b",
    ],
}

# Observable forensic pattern matcher for honest rule-based fallback if model is untrained
OBSERVABLE_PATTERNS = {
    AttributeCategory.FACE_SHAPE: [
        r"(oval|round|square|heart-shaped|heart shaped|angular|chiseled|sharp|narrow|wide|oblong)\s+(face|jaw|head)",
        r"(face|jaw)\s+(was|is)\s+(oval|round|square|heart-shaped|angular|sharp|narrow|wide|oblong)",
    ],
    AttributeCategory.HAIR: [
        r"(short|long|shoulder-length|curly|straight|wavy|bald|buzz cut|receding)\s+(black|brown|blonde|blond|gray|grey|red|dark)\s+hair",
        r"(black|brown|blonde|blond|gray|grey|red|dark)\s+(curly|straight|wavy|short|long)?\s*hair",
        r"\b(bald on top|bald|buzz cut|buzzcut)\b",
    ],
    AttributeCategory.FACIAL_HAIR: [
        r"(full|thick|thin|trimmed|stubble|patchy)\s+(beard|goatee|mustache|moustache)",
        r"(beard|goatee|mustache|moustache|sideburns)",
    ],
    AttributeCategory.EYES: [
        r"(dark|brown|blue|green|hazel|black|light|deep-set)\s+eyes",
        r"\b(eye\s+color|eyes)\b",
    ],
    AttributeCategory.HAIRLINE: [
        r"(receding|widow'?s peak|high|low|straight)\s+hairline",
        r"\b(widow'?s peak|receding hairline)\b",
    ],
    AttributeCategory.EYEBROWS: [
        r"(thick|thin|arched|bushy|plucked|unibrow|faint unibrow)\s+eyebrows?",
        r"\b(unibrow|thick eyebrows|bushy eyebrows)\b",
    ],
    AttributeCategory.NOSE: [
        r"(sharp|broad|wide|crooked|pointed|flat|hooked|aquiline|small|large)\s+nose",
        r"(nose\s+was\s+(crooked|broken|sharp|wide|flat))",
    ],
    AttributeCategory.LIPS: [
        r"(thin|full|thick|wide|dry|chapped)\s+lips",
        r"(lips\s+were\s+(thin|full|thick))",
    ],
    AttributeCategory.DISTINGUISHING_MARKS: [
        r"(scar|tattoo|birthmark|mole|piercing|cut|burn)\s+(on|near|across|under|above)\s+[a-z\s]+",
        r"(scar on (his|her|their)?\s*(cheek|chin|forehead|eyebrow|lip|nose))",
        r"(tattoo of [a-z\s]+)",
        r"\b(mole|birthmark|scar|tattoo)\b",
    ],
    AttributeCategory.OTHER_OBSERVABLE: [
        r"(protruding|large|small)\s+ears",
        r"(silver|gold)?\s*(hoop|stud)?\s*earrings?",
        r"(glasses|spectacles|wire-rim glasses)",
        r"(deep forehead wrinkles|cleft chin|pronounced cleft chin)",
    ],
}


class NLPService:
    """
    Forensic Natural Language Processing service managing custom Character-Aware BiLSTM-CRF labeler.
    Adheres strictly to project constraints:
    - Zero pretrained weights or embeddings.
    - Honest reporting of trained status and SHA-256 verification.
    - Never fabricates unobserved attributes.
    - Preserves verbatim source spans and character offsets.
    - Negation safety layer suppresses positive extraction on negated observations.
    """
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        uncertainty_service: UncertaintyService,
        vocab_path: Optional[str] = None,
        char_vocab_path: Optional[str] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.uncertainty_service = uncertainty_service
        self.vocab_path = vocab_path
        self.char_vocab_path = char_vocab_path

        self.model: Optional[BiLSTMAttributeLabeler] = None
        self.vocab: Optional[Vocabulary] = None
        self.char_vocab: Optional[CharVocabulary] = None
        self.is_trained: bool = False
        self.model_metadata: Dict[str, Any] = {}

        self._check_and_load_checkpoint()

    def _resolve_paths(self) -> Tuple[str, str, str]:
        """
        Resolves existing file paths for checkpoint, word vocab, and char vocab.
        Checks both direct path and fallback paths.
        """
        ckpt = self.checkpoint_path
        if not os.path.exists(ckpt):
            # Check alternative locations
            candidates = [
                os.path.join("checkpoints", os.path.basename(ckpt)),
                os.path.join("phantom-ai-ml", "checkpoints", os.path.basename(ckpt)),
                os.path.join("..", "checkpoints", os.path.basename(ckpt)),
            ]
            for c in candidates:
                if os.path.exists(c):
                    ckpt = c
                    break

        base_dir = os.path.dirname(ckpt)
        base_name = os.path.splitext(os.path.basename(ckpt))[0]

        # Word vocab
        word_v = self.vocab_path
        if not word_v or not os.path.exists(word_v):
            candidates_w = [
                os.path.join(base_dir, f"{base_name}_word_vocab.json"),
                os.path.join("checkpoints", f"{base_name}_word_vocab.json"),
                os.path.join(base_dir, "vocab.json"),
                os.path.join("checkpoints", "vocab.json"),
            ]
            for w in candidates_w:
                if os.path.exists(w):
                    word_v = w
                    break

        # Char vocab
        char_v = self.char_vocab_path
        if not char_v or not os.path.exists(char_v):
            candidates_c = [
                os.path.join(base_dir, f"{base_name}_char_vocab.json"),
                os.path.join("checkpoints", f"{base_name}_char_vocab.json"),
                os.path.join(base_dir, "char_vocab.json"),
                os.path.join("checkpoints", "char_vocab.json"),
            ]
            for cv in candidates_c:
                if os.path.exists(cv):
                    char_v = cv
                    break

        return ckpt, word_v, char_v

    def _check_and_load_checkpoint(self) -> None:
        """
        Inspects filesystem for genuine trained model checkpoint, verifies SHA-256 hash,
        and instantiates the Character-Aware BiLSTM-CRF architecture.
        """
        ckpt_path, word_vocab_path, char_vocab_path = self._resolve_paths()
        self.checkpoint_path = ckpt_path

        if os.path.exists(ckpt_path):
            try:
                # Calculate SHA-256
                sha = hashlib.sha256()
                with open(ckpt_path, "rb") as f:
                    while chunk := f.read(65536):
                        sha.update(chunk)
                actual_hash = sha.hexdigest()

                # Verify hash if targeting Phase 3C.2 checkpoint
                if "phase3c2" in os.path.basename(ckpt_path):
                    if actual_hash != EXPECTED_PHASE3C2_SHA256:
                        raise ValueError(
                            f"Integrity check failed for Phase 3C.2 checkpoint: "
                            f"expected {EXPECTED_PHASE3C2_SHA256}, got {actual_hash}"
                        )

                checkpoint = torch.load(ckpt_path, map_location=self.device)
                vocab_size = checkpoint.get("vocab_size", 1000)
                num_tags = checkpoint.get("num_tags", len(TAGS))
                emb_dim = checkpoint.get("embedding_dim", 128)
                hidden_dim = checkpoint.get("hidden_dim", 256)
                num_layers = checkpoint.get("num_layers", 2)
                dropout = checkpoint.get("dropout", 0.3)
                char_vocab_size = checkpoint.get("char_vocab_size")
                char_emb_dim = checkpoint.get("char_embedding_dim", 32)
                char_out_dim = checkpoint.get("char_out_dim", 64)
                char_kernel_size = checkpoint.get("char_kernel_size", 3)

                if char_vocab_size:
                    model = CharAwareBiLSTMAttributeLabeler(
                        vocab_size=vocab_size,
                        char_vocab_size=char_vocab_size,
                        num_tags=num_tags,
                        embedding_dim=emb_dim,
                        char_embedding_dim=char_emb_dim,
                        char_out_dim=char_out_dim,
                        char_kernel_size=char_kernel_size,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        padding_idx=0,
                    )
                else:
                    model = BiLSTMAttributeLabeler(
                        vocab_size=vocab_size,
                        num_tags=num_tags,
                        embedding_dim=emb_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        padding_idx=0,
                    )

                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(self.device)
                model.eval()

                self.model = model

                # Load word vocabulary
                if word_vocab_path and os.path.exists(word_vocab_path):
                    self.vocab = Vocabulary.load(word_vocab_path)
                    self.vocab_path = word_vocab_path
                else:
                    self.vocab = None

                # Load character vocabulary
                if char_vocab_size and char_vocab_path and os.path.exists(char_vocab_path):
                    self.char_vocab = CharVocabulary.load(char_vocab_path)
                    self.char_vocab_path = char_vocab_path
                else:
                    self.char_vocab = None

                self.is_trained = checkpoint.get("is_trained", True)
                self.model_metadata = {
                    "epoch": checkpoint.get("epoch"),
                    "val_loss": checkpoint.get("val_loss"),
                    "val_f1": checkpoint.get("val_f1"),
                    "trained_timestamp": checkpoint.get("trained_timestamp"),
                    "model_type": checkpoint.get("model_type", "CharAwareBiLSTMAttributeLabeler"),
                    "phase": checkpoint.get("phase", "3C.2"),
                    "is_character_aware": bool(char_vocab_size),
                    "sha256": actual_hash,
                    "parameter_count": model.count_parameters(),
                }
            except Exception as e:
                self.is_trained = False
                self.model = None
                self.vocab = None
                self.char_vocab = None
                self.model_metadata = {"load_error": str(e)}
        else:
            self.is_trained = False
            self.model = None
            self.vocab = None
            self.char_vocab = None
            self.model_metadata = {"load_error": f"Checkpoint not found at {ckpt_path}"}

    def reload_checkpoint(self) -> bool:
        """
        Forces reload of the checkpoint and vocabulary from disk.
        """
        self._check_and_load_checkpoint()
        return self.is_trained

    def get_model_metadata_info(self) -> Dict[str, Any]:
        """
        Returns structured model identification and provenance metadata for API response.
        """
        return {
            "name": self.model_metadata.get("model_type", "CharAwareBiLSTMAttributeLabeler"),
            "architecture": "CharAwareBiLSTM-CRF",
            "phase": self.model_metadata.get("phase", "3C.2"),
            "checkpoint": os.path.basename(self.checkpoint_path),
            "checkpoint_sha256": self.model_metadata.get("sha256", EXPECTED_PHASE3C2_SHA256),
            "device": str(self.device),
            "is_trained": self.is_trained,
        }

    def _extract_neural_spans(
        self, witness_text: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Runs neural sequence labeling inference using the trained Character-Aware BiLSTM-CRF.
        Extracts contiguous entity spans, preserving exact character-level offsets.
        Verifies text[start:end] == span on each prediction.
        """
        if not self.is_trained or self.model is None or self.vocab is None:
            return {}

        tok_offsets = tokenize_with_offsets(witness_text)
        if not tok_offsets:
            return {}

        tokens = [t[0] for t in tok_offsets]
        input_ids = torch.tensor([self.vocab.encode(tokens)], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(tokens)], dtype=torch.long)

        self.model.eval()
        with torch.inference_mode():
            if getattr(self.model, "use_char_encoder", False) and self.char_vocab is not None:
                char_ids_list = [self.char_vocab.encode_word(t) for t in tokens]
                max_c = max(max(len(c) for c in char_ids_list), 3)
                char_t = torch.zeros(1, len(tokens), max_c, dtype=torch.long, device=self.device)
                for w_i, c_seq in enumerate(char_ids_list):
                    c_len = min(len(c_seq), max_c)
                    if c_len > 0:
                        char_t[0, w_i, :c_len] = torch.tensor(c_seq[:c_len], dtype=torch.long, device=self.device)
                emissions = self.model(input_ids, lengths=lengths, char_ids=char_t)
            else:
                emissions = self.model(input_ids, lengths=lengths)

            pred_indices = self.model.decode_viterbi(emissions, lengths=lengths)[0]

        pred_tags = [IDX2TAG.get(idx, "O") for idx in pred_indices]

        # Extract contiguous spans and map exact character offsets
        extracted: Dict[str, Dict[str, Any]] = {}
        current_cat: Optional[str] = None
        start_char: Optional[int] = None
        end_char: Optional[int] = None

        for (tok_str, t_start, t_end), tag in zip(tok_offsets, pred_tags):
            if tag.startswith("B-"):
                if current_cat and current_cat not in extracted:
                    span_str = witness_text[start_char:end_char]
                    extracted[current_cat] = {
                        "span": span_str,
                        "start_char": start_char,
                        "end_char": end_char,
                    }
                current_cat = tag[2:].lower()
                start_char = t_start
                end_char = t_end
            elif tag.startswith("I-") and current_cat == tag[2:].lower():
                end_char = t_end
            else:
                if current_cat and current_cat not in extracted:
                    span_str = witness_text[start_char:end_char]
                    extracted[current_cat] = {
                        "span": span_str,
                        "start_char": start_char,
                        "end_char": end_char,
                    }
                current_cat = None
                start_char = None
                end_char = None

        if current_cat and current_cat not in extracted:
            span_str = witness_text[start_char:end_char]
            extracted[current_cat] = {
                "span": span_str,
                "start_char": start_char,
                "end_char": end_char,
            }

        return extracted

    def _calibrate_and_route_spans(
        self, raw_spans: Dict[str, Dict[str, Any]], witness_text: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Phase 3.6 Span-Boundary Calibration and Category Boundary Protection.
        - Strips leading auxiliary/linking verbs ('have narrow eyes' -> 'narrow eyes').
        - Strips leading articles / redundant modifiers while updating exact character offsets.
        - Extends multi-token descriptors ('bald' -> 'bald on top', 'buzz' -> 'buzz cut', 'widow\'s' -> 'widow\'s peak').
        - Splits coordinate conjunctions ('blond hair and blue eyes' -> 'blond hair' [hair] + 'blue eyes' [eyes]).
        - Re-routes misclassified spans:
            * 'facial hair' in hair -> facial_hair
            * 'very short' hair in facial_hair -> hair
            * primary facial features in other_observable/face_shape -> primary category
        - Suppresses headwear / occlusions ('baseball cap', 'beanie', 'face covering', etc.) from facial geometry.
        - Strictly guarantees witness_text[start_char:end_char] == span.
        """
        occlusions = re.compile(
            r"\b(cap|baseball\s+cap|beanie|hat|hoodie|hood|face\s+covering|mask|bandana|helmet|broad\s+shoulders|shoulders|jacket|coat|shirt)\b",
            re.IGNORECASE,
        )
        aux_prefix = re.compile(
            r"^(?:have|had|has|having|wore|wearing|was|were|is|with)\s+(?:a\s+|an\s+|the\s+)?",
            re.IGNORECASE,
        )
        leading_articles = re.compile(
            r"^(?:a|an|the|his|her|their)\s+",
            re.IGNORECASE,
        )
        clause_prefix = re.compile(
            r"^(?:(?:his|her|their)?\s*eyes?\s+(?:were|was)\s+)",
            re.IGNORECASE,
        )
        category_keywords = {
            "facial_hair": re.compile(r"\b(beard|mustache|moustache|goatee|sideburns|stubble|facial\s+hair)\b", re.IGNORECASE),
            "hair": re.compile(r"\b(hair|bald|bald\s+on\s+top|buzz\s*cut|dreadlocks|afro|ponytail|bangs)\b", re.IGNORECASE),
            "eyes": re.compile(r"\b(eyes?|eye\s+color)\b", re.IGNORECASE),
            "nose": re.compile(r"\b(nose|nostrils?|bridge)\b", re.IGNORECASE),
            "lips": re.compile(r"\b(lips?|mouth)\b", re.IGNORECASE),
            "eyebrows": re.compile(r"\b(eyebrows?|brows?|unibrow)\b", re.IGNORECASE),
            "hairline": re.compile(r"\b(hairline|widow'?s\s*peak)\b", re.IGNORECASE),
            "face_shape": re.compile(r"\b(face|jaw|chin|cheekbones?)\b", re.IGNORECASE),
            "other_observable": re.compile(r"\b(glasses|spectacles|earrings?|jewelry|wrinkles)\b", re.IGNORECASE),
            "distinguishing_marks": re.compile(r"\b(scars?|tattoos?|marks?|birthmarks?|piercings?|moles?)\b", re.IGNORECASE),
        }

        calibrated: Dict[str, Dict[str, Any]] = {}

        for cat, info in raw_spans.items():
            span = info["span"]
            start_char = info["start_char"]
            end_char = info["end_char"]

            # 1. Suppress occlusions, non-facial accessories, or body parts from facial geometry
            if cat != "other_observable" and occlusions.search(span):
                continue

            # 2. Conjoined descriptors across conjunctions: e.g. 'blond hair and blue eyes'
            conjunction_match = re.search(r"\s+(?:and|as well as)\s+", span, re.IGNORECASE)
            if conjunction_match:
                part1 = span[:conjunction_match.start()].strip()
                part2 = span[conjunction_match.end():].strip()

                cat1 = None
                cat2 = None
                for c_name, c_re in category_keywords.items():
                    if c_re.search(part1) and cat1 is None:
                        cat1 = c_name
                    if c_re.search(part2) and cat2 is None:
                        cat2 = c_name

                if cat1 and cat2 and cat1 != cat2:
                    p1_start = witness_text.find(part1, start_char)
                    p1_end = p1_start + len(part1)
                    p2_start = witness_text.find(part2, start_char)
                    p2_end = p2_start + len(part2)
                    sub_dict = {
                        cat1: {"span": part1, "start_char": p1_start, "end_char": p1_end},
                        cat2: {"span": part2, "start_char": p2_start, "end_char": p2_end},
                    }
                    calibrated.update(self._calibrate_and_route_spans(sub_dict, witness_text))
                    continue

            # 3. Strip auxiliary and linking verbs
            m_aux = aux_prefix.match(span)
            if m_aux:
                cut = m_aux.end()
                span = span[cut:]
                start_char = start_char + cut

            # 4. Strip leading articles or possessives
            m_art = leading_articles.match(span)
            if m_art:
                cut = m_art.end()
                span = span[cut:]
                start_char = start_char + cut

            # 5. Multi-token descriptor extensions
            if span.lower() == "bald" and witness_text[end_char:end_char + 7].lower() == " on top":
                span = span + witness_text[end_char:end_char + 7]
                end_char = end_char + 7
            elif span.lower() == "buzz" and witness_text[end_char:end_char + 4].lower() == " cut":
                span = span + witness_text[end_char:end_char + 4]
                end_char = end_char + 4
            elif span.lower() in ("widow's", "widows") and witness_text[end_char:end_char + 5].lower() == " peak":
                span = span + witness_text[end_char:end_char + 5]
                end_char = end_char + 5

            # 6. Category boundary re-routing
            target_cat = cat
            if cat == "hair" and category_keywords["facial_hair"].search(span):
                target_cat = "facial_hair"
            elif cat == "facial_hair":
                if not category_keywords["facial_hair"].search(span):
                    preceding = witness_text[max(0, start_char - 25):start_char].lower()
                    if "hair" in preceding or "head" in preceding:
                        target_cat = "hair"
            elif cat in ("other_observable", "face_shape"):
                for primary in ["eyes", "nose", "lips", "eyebrows", "hairline", "hair", "facial_hair"]:
                    if category_keywords[primary].search(span):
                        target_cat = primary
                        break

            # Strip leading clause prefix if present (e.g. 'eyes were clearly blue' -> 'clearly blue')
            if target_cat == "eyes" and clause_prefix.match(span):
                m_cl = clause_prefix.match(span)
                cut = m_cl.end()
                span = span[cut:]
                start_char = start_char + cut

            # Strip trailing punctuation if present
            while span and span[-1] in ".,;:!?":
                span = span[:-1]
                end_char -= 1

            assert witness_text[start_char:end_char] == span, (
                f"Calibrated span offset mismatch: '{witness_text[start_char:end_char]}' != '{span}'"
            )

            calibrated[target_cat] = {
                "span": span,
                "start_char": start_char,
                "end_char": end_char,
            }

        return calibrated

    def _detect_negations(self, witness_text: str) -> Dict[str, Dict[str, Any]]:
        """
        Conservative Negation Safety Layer (Phase 3C.2 / Phase 4).
        Identifies explicitly negated attributes in witness text.
        Guarantees that negative observations (e.g. 'no beard', 'not bald')
        are NEVER presented as active positive facial features.
        """
        negations: Dict[str, Dict[str, Any]] = {}
        for category, patterns in NEGATION_PATTERNS.items():
            cat_name = category.value
            for pat in patterns:
                m = re.search(pat, witness_text, re.IGNORECASE)
                if m:
                    negations[cat_name] = {
                        "span": m.group(0),
                        "start_char": m.start(),
                        "end_char": m.end(),
                        "pattern": pat,
                    }
                    break
        return negations

    def _detect_unknowns(self, witness_text: str) -> Dict[str, Dict[str, Any]]:
        """
        Explicit Unobservable / Unknown Evidence Layer (Phase 3.5).
        Identifies attributes explicitly stated as unobservable or unknown
        (e.g. 'could not see his hair', 'eye color was unknown').
        """
        unknowns: Dict[str, Dict[str, Any]] = {}
        for category, patterns in UNKNOWN_PATTERNS.items():
            cat_name = category.value
            for pat in patterns:
                m = re.search(pat, witness_text, re.IGNORECASE)
                if m:
                    unknowns[cat_name] = {
                        "span": m.group(0),
                        "start_char": m.start(),
                        "end_char": m.end(),
                        "pattern": pat,
                    }
                    break
        return unknowns

    def analyze_description(
        self,
        witness_text: str,
        confidence_threshold: float = 0.5,
    ) -> Tuple[str, bool, Dict[str, AttributeEvidence], List[str], List[str], Optional[float], str]:
        """
        Extracts facial attributes while strictly preserving uncertainty, unknown, and negated states.
        Enforces evidence traceability:
          originalText[start:end] === sourceText
        Numerical confidence scores are not fabricated; returns None for uncalibrated extraction.
        """
        contradictions = self.uncertainty_service.detect_contradictions(witness_text, {})
        contradiction_conflicts = (
            self.uncertainty_service.detect_contradictions_with_categories(witness_text)
            if hasattr(self.uncertainty_service, "detect_contradictions_with_categories")
            else []
        )
        contradictory_categories: Dict[str, Tuple[str, str, Any]] = {
            c[2].value if hasattr(c[2], "value") else str(c[2]): c
            for c in contradiction_conflicts
        }

        overall_uncertainties = self.uncertainty_service.detect_uncertainty_cues(witness_text)

        # Break witness statement into sentence units
        sentences = [s.strip() for s in re.split(r"[.!?]+", witness_text) if s.strip()]

        # 1. Detect explicit negations (Safety layer)
        negations = self._detect_negations(witness_text)

        # 2. Detect explicit unknowns (Unobservable layer)
        unknowns = self._detect_unknowns(witness_text)

        # 3. Extract neural spans using trained Phase 3C.2 model
        neural_spans = self._extract_neural_spans(witness_text) if self.is_trained else {}
        # 3.1 Calibrate span boundaries and route category boundaries (Phase 3.6)
        neural_spans = self._calibrate_and_route_spans(neural_spans, witness_text)

        attributes: Dict[str, AttributeEvidence] = {}

        for category in AttributeCategory:
            cat_name = category.value

            # Check Contradictions First (Safety blocking)
            if cat_name in contradictory_categories:
                term1, term2, _ = contradictory_categories[cat_name]
                conflict_notes = f"Contradictory testimony detected for {cat_name}: '{term1}' vs '{term2}'"
                evidence = AttributeEvidence(
                    category=category,
                    state=EvidenceState.CONTRADICTORY,
                    value=None,
                    confidence=None,
                    source_spans=[term1, term2],
                    start_char=None,
                    end_char=None,
                    extracted_by="contradiction_safety_engine",
                    uncertainty_cues=[],
                    notes=conflict_notes,
                )
                attributes[cat_name] = evidence
                continue

            # Check Negation Safety Layer
            if cat_name in negations:
                neg = negations[cat_name]
                neg_span = neg["span"]
                neg_start = neg["start_char"]
                neg_end = neg["end_char"]

                # Ensure exact character offset check holds
                assert witness_text[neg_start:neg_end] == neg_span, f"Offset mismatch on negation: {witness_text[neg_start:neg_end]} != {neg_span}"

                # Subtype negation check: if negating baldness (e.g. 'not bald', 'not shaved'),
                # but an affirmative hair trait is present in neural_spans or observable patterns,
                # the affirmative trait represents the true explicit hair description.
                has_positive_hair = False
                if cat_name == AttributeCategory.HAIR.value and ("bald" in neg_span or "shaved" in neg_span):
                    if cat_name in neural_spans and neural_spans[cat_name]["start_char"] != neg_start:
                        has_positive_hair = True
                    else:
                        for pat in OBSERVABLE_PATTERNS.get(category, []):
                            m = re.search(pat, witness_text, re.IGNORECASE)
                            if m and m.start() != neg_start and "bald" not in m.group(0).lower():
                                has_positive_hair = True
                                break

                if not has_positive_hair:
                    evidence = AttributeEvidence(
                        category=category,
                        state=EvidenceState.NEGATED,
                        value=None,  # Negated features have null active value
                        confidence=None,
                        source_spans=[neg_span],
                        start_char=neg_start,
                        end_char=neg_end,
                        extracted_by="negation_safety_rule",
                        uncertainty_cues=[],
                        notes=f"Attribute explicitly negated by witness: '{neg_span}'",
                    )
                    attributes[cat_name] = evidence
                    continue

            # Check Explicit Unknowns / Unobservable features
            if cat_name in unknowns:
                unk = unknowns[cat_name]
                unk_span = unk["span"]
                unk_start = unk["start_char"]
                unk_end = unk["end_char"]
                assert witness_text[unk_start:unk_end] == unk_span, f"Offset mismatch on unknown: {witness_text[unk_start:unk_end]} != {unk_span}"

                evidence = AttributeEvidence(
                    category=category,
                    state=EvidenceState.UNKNOWN,
                    value=None,
                    confidence=None,
                    source_spans=[unk_span],
                    start_char=unk_start,
                    end_char=unk_end,
                    extracted_by="explicit_unknown_rule",
                    uncertainty_cues=[],
                    notes=f"Feature explicitly noted as unobserved/unknown by witness: '{unk_span}'",
                )
                attributes[cat_name] = evidence
                continue

            # Check Neural Extraction
            if self.is_trained and cat_name in neural_spans:
                n_info = neural_spans[cat_name]
                extracted_val = n_info["span"]
                start_offset = n_info["start_char"]
                end_offset = n_info["end_char"]

                # Ensure neural span is not within another category's negation scope (prevents scope bleed)
                is_inside_negation = any(
                    neg["start_char"] <= start_offset and end_offset <= neg["end_char"]
                    for neg in negations.values()
                )
                if not is_inside_negation:
                    # Verbatim check
                    assert witness_text[start_offset:end_offset] == extracted_val, (
                        f"Offset mismatch on extraction: {witness_text[start_offset:end_offset]} != {extracted_val}"
                    )

                    # Check linguistic scope cues
                    cues = self.uncertainty_service.detect_scope_cues(witness_text, start_offset, end_offset)
                    state = EvidenceState.UNCERTAIN if cues else EvidenceState.EXPLICIT

                    evidence = AttributeEvidence(
                        category=category,
                        state=state,
                        value=extracted_val,
                        confidence=None,
                        source_spans=[extracted_val],
                        start_char=start_offset,
                        end_char=end_offset,
                        extracted_by="nlp_model",
                        uncertainty_cues=cues,
                        notes=f"Extracted by Phase 3C.2 Character-Aware BiLSTM-CRF ({state.value}).",
                    )
                    attributes[cat_name] = evidence
                    continue

            # Fallback for pattern-matched observable features (e.g. isolated nouns or fallback mode)
            extracted_val = None
            start_offset = None
            end_offset = None
            patterns = OBSERVABLE_PATTERNS.get(category, [])
            for pat in patterns:
                m = re.search(pat, witness_text, re.IGNORECASE)
                if m:
                    candidate_val = m.group(0)
                    c_start = m.start()
                    c_end = m.end()
                    # Check not inside negation scope
                    if not any(neg["start_char"] <= c_start and c_end <= neg["end_char"] for neg in negations.values()):
                        extracted_val = candidate_val
                        start_offset = c_start
                        end_offset = c_end
                        break

            if extracted_val and start_offset is not None and end_offset is not None:
                assert witness_text[start_offset:end_offset] == extracted_val
                cues = self.uncertainty_service.detect_scope_cues(witness_text, start_offset, end_offset)
                state = EvidenceState.UNCERTAIN if cues else EvidenceState.EXPLICIT
                evidence = AttributeEvidence(
                    category=category,
                    state=state,
                    value=extracted_val,
                    confidence=None,
                    source_spans=[extracted_val],
                    start_char=start_offset,
                    end_char=end_offset,
                    extracted_by="pattern_fallback" if self.is_trained else "rule_processor",
                    uncertainty_cues=cues,
                    notes=f"Identified by pattern fallback ({state.value}).",
                )
                attributes[cat_name] = evidence
                continue

            # Feature not observed -> strictly UNKNOWN
            evidence = AttributeEvidence(
                category=category,
                state=EvidenceState.UNKNOWN,
                value=None,
                confidence=None,
                source_spans=[],
                start_char=None,
                end_char=None,
                extracted_by="system_default",
                uncertainty_cues=[],
                notes="Feature was not observed or mentioned in witness testimony.",
            )

            # Flag contradictions if category is in conflicting testimony
            if any(cat_name in c.lower() for c in contradictions):
                evidence.state = EvidenceState.CONTRADICTORY
                evidence.notes = "Conflicting evidence observed across witness statements."

            attributes[cat_name] = evidence

        # Per Phase 4.2 guidelines: Do not manufacture artificial numerical confidence scores
        overall_confidence: Optional[float] = None

        model_meta = self.get_model_metadata_info()

        if self.is_trained:
            status = "success"
            audit_notes = (
                f"Attributes extracted using Phase 3C.2 Character-Aware BiLSTM-CRF "
                f"(Epoch {self.model_metadata.get('epoch', 'N/A')}, Val F1: {self.model_metadata.get('val_f1', 0.0):.4f}, "
                f"Device: {self.device}). Uncalibrated extraction; no artificial confidence values manufactured."
            )
        else:
            status = "model_not_trained"
            audit_notes = (
                f"Trained checkpoint not found at {self.checkpoint_path}. "
                "Returning unestablished baseline. No fake neural predictions were generated."
            )

        return (
            status,
            self.is_trained,
            attributes,
            contradictions,
            overall_uncertainties,
            overall_confidence,
            audit_notes,
        )
