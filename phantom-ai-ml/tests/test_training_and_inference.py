import os
import json
import pytest
import tempfile
import torch

from utils.device import resolve_device
from utils.seed import set_global_seed
from utils.preprocessing import TAGS, TAG2IDX, IDX2TAG, Vocabulary, tokenize_with_offsets
from models.nlp_model import BiLSTMAttributeLabeler
from data.dataset_loader import DatasetLoader
from training.train_nlp import WitnessSequenceDataset, pad_collate_fn
from services.nlp_service import NLPService
from services.uncertainty_service import UncertaintyService
from schemas.attributes import AttributeCategory, EvidenceState


@pytest.fixture
def device():
    return resolve_device()


@pytest.fixture
def sample_vocab():
    v = Vocabulary()
    v.build_vocab([["the", "suspect", "had", "short", "black", "hair", "and", "brown", "eyes"]])
    return v


def test_training_data_loading(sample_vocab):
    loader = DatasetLoader()
    train_path = "datasets/nlp/train.jsonl"
    if not os.path.exists(train_path):
        train_path = os.path.join("phantom-ai-ml", train_path)
    assert os.path.exists(train_path), "Training dataset file must exist"

    samples = loader.load_jsonl(train_path, validate=True)
    assert len(samples) > 0

    dataset = WitnessSequenceDataset(samples, sample_vocab)
    assert len(dataset) == len(samples)

    seq, tags, length = dataset[0]
    assert isinstance(seq, torch.Tensor)
    assert isinstance(tags, torch.Tensor)
    assert length == len(seq)
    assert length > 0


def test_training_reproducibility():
    """
    Two models initialized with identical global seeds must have identical initial weights.
    """
    set_global_seed(999)
    model_1 = BiLSTMAttributeLabeler(vocab_size=100, num_tags=len(TAGS))

    set_global_seed(999)
    model_2 = BiLSTMAttributeLabeler(vocab_size=100, num_tags=len(TAGS))

    for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
        assert torch.allclose(p1, p2), "Model parameters must match exactly for identical random seeds."


def test_checkpoint_save_and_load(tmp_path, device):
    """
    Verifies that a model checkpoint can be saved and accurately reloaded without degradation.
    """
    model = BiLSTMAttributeLabeler(vocab_size=50, num_tags=10, embedding_dim=32, hidden_dim=64).to(device)
    ckpt_path = tmp_path / "test_checkpoint.pt"

    torch.save({
        "epoch": 5,
        "model_state_dict": model.state_dict(),
        "vocab_size": 50,
        "num_tags": 10,
        "embedding_dim": 32,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "val_loss": 0.25,
        "val_f1": 0.85,
        "is_trained": True,
    }, ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location=device)
    reloaded_model = BiLSTMAttributeLabeler(
        vocab_size=checkpoint["vocab_size"],
        num_tags=checkpoint["num_tags"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)
    reloaded_model.load_state_dict(checkpoint["model_state_dict"])

    for p1, p2 in zip(model.parameters(), reloaded_model.parameters()):
        assert torch.allclose(p1, p2), "Reloaded model weights must match saved weights."


def test_bio_viterbi_and_argmax_decoding(device):
    """
    Verifies Viterbi decoding and argmax sequence decoding algorithms.
    """
    num_tags = 5
    seq_len = 4
    batch_size = 2
    model = BiLSTMAttributeLabeler(vocab_size=20, num_tags=num_tags, hidden_dim=32).to(device)

    emissions = torch.randn(batch_size, seq_len, num_tags, device=device)
    lengths = torch.tensor([4, 2], dtype=torch.long)

    viterbi_preds = model.decode_viterbi(emissions, lengths=lengths)
    argmax_preds = model.decode_argmax(emissions, lengths=lengths)

    assert len(viterbi_preds) == batch_size
    assert len(viterbi_preds[0]) == 4
    assert len(viterbi_preds[1]) == 2

    assert len(argmax_preds) == batch_size
    assert len(argmax_preds[0]) == 4
    assert len(argmax_preds[1]) == 2


def test_model_configuration_compatibility():
    """
    Verifies that model_config.json contains all required architectural fields.
    """
    config_path = "checkpoints/model_config.json"
    if not os.path.exists(config_path):
        config_path = os.path.join("phantom-ai-ml", config_path)
    assert os.path.exists(config_path), "model_config.json must exist in checkpoints directory"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert config["model_type"] == "BiLSTMAttributeLabeler"
    assert config["random_initialization"] is True
    assert config["pretrained_weights"] is False
    assert config["num_tags"] == len(TAGS)
    assert "vocab_size" in config
    assert "embedding_dim" in config
    assert "hidden_dim" in config


def test_nlp_service_with_trained_checkpoint(device):
    """
    Verifies that NLPService loads the genuine checkpoint and performs neural inference.
    """
    ckpt_path = "checkpoints/nlp_bilstm.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join("phantom-ai-ml", ckpt_path)
    assert os.path.exists(ckpt_path), "Trained checkpoint nlp_bilstm.pt must exist"

    uncertainty_service = UncertaintyService()
    nlp_service = NLPService(
        checkpoint_path=ckpt_path,
        device=device,
        uncertainty_service=uncertainty_service,
    )

    assert nlp_service.is_trained is True
    assert nlp_service.model is not None
    assert nlp_service.vocab is not None

    status, is_trained, attributes, contradictions, uncertainties, conf, audit = nlp_service.analyze_description(
        "The suspect had an oval face, short dark curly hair, and brown eyes."
    )

    assert status == "success"
    assert is_trained is True
    assert "hair" in attributes
    assert "face_shape" in attributes
    assert "eyes" in attributes


def test_missing_checkpoint_behavior(device):
    """
    Verifies truthful fallback when checkpoint is absent.
    """
    uncertainty_service = UncertaintyService()
    nlp_service = NLPService(
        checkpoint_path="/non/existent/path/model.pt",
        device=device,
        uncertainty_service=uncertainty_service,
    )

    assert nlp_service.is_trained is False
    assert nlp_service.model is None

    status, is_trained, attributes, _, _, _, audit = nlp_service.analyze_description(
        "The suspect was tall with a bushy mustache."
    )
    assert status == "model_not_trained"
    assert is_trained is False
    assert "No fake neural predictions were generated" in audit


def test_span_alignment_and_no_hallucination(device):
    """
    Verifies that unobserved features remain strictly UNKNOWN with 0 confidence and null spans.
    """
    ckpt_path = "checkpoints/nlp_bilstm.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join("phantom-ai-ml", ckpt_path)

    uncertainty_service = UncertaintyService()
    nlp_service = NLPService(
        checkpoint_path=ckpt_path,
        device=device,
        uncertainty_service=uncertainty_service,
    )

    # Text that mentions ONLY nose
    status, _, attributes, _, _, _, _ = nlp_service.analyze_description(
        "The only thing I saw was a crooked nose."
    )

    # Nose should be observed
    assert attributes["nose"].state in [EvidenceState.EXPLICIT, EvidenceState.EXPLICIT.value]
    assert attributes["nose"].value is not None

    # Distinguishing marks should remain UNKNOWN - NEVER hallucinated
    dm = attributes["distinguishing_marks"]
    assert dm.state in [EvidenceState.UNKNOWN, EvidenceState.UNKNOWN.value]
    assert dm.confidence is None or dm.confidence == 0.0
    assert dm.value is None
    assert len(dm.source_spans) == 0


def test_phase3c2_checkpoint_integrity_and_loading(device):
    """
    Verifies that NLPService loads the Phase 3C.2 checkpoint with valid SHA-256 hash.
    """
    ckpt_path = "checkpoints/nlp_char_bilstm_phase3c2.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join("phantom-ai-ml", ckpt_path)
    assert os.path.exists(ckpt_path), "Phase 3C.2 checkpoint must exist"

    uncertainty_service = UncertaintyService()
    nlp_service = NLPService(
        checkpoint_path=ckpt_path,
        device=device,
        uncertainty_service=uncertainty_service,
    )

    assert nlp_service.is_trained is True
    assert nlp_service.model is not None
    assert nlp_service.vocab is not None
    assert nlp_service.char_vocab is not None
    meta = nlp_service.get_model_metadata_info()
    assert meta["checkpoint_sha256"] == "2d567c79d759e76940968fadb02d412b530d3405e377a2d6454c64e9f2e7b4a3"


def test_negation_safety_layer(device):
    """
    Verifies that explicitly negated features (e.g. 'no beard', 'clean-shaven')
    are NEVER classified as active positive attributes, and are marked as NEGATED.
    """
    ckpt_path = "checkpoints/nlp_char_bilstm_phase3c2.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join("phantom-ai-ml", ckpt_path)

    uncertainty_service = UncertaintyService()
    nlp_service = NLPService(
        checkpoint_path=ckpt_path,
        device=device,
        uncertainty_service=uncertainty_service,
    )

    text = "The suspect was clean-shaven with no beard, but had piercing blue eyes."
    status, is_trained, attributes, _, _, _, _ = nlp_service.analyze_description(text)

    # Facial hair MUST be marked NEGATED, not EXPLICIT positive
    fh = attributes["facial_hair"]
    assert fh.state in [EvidenceState.NEGATED, EvidenceState.NEGATED.value]
    assert fh.value is None  # Must not have positive value
    assert fh.extracted_by == "negation_safety_rule"
    assert fh.start_char is not None and fh.end_char is not None
    assert text[fh.start_char:fh.end_char] == fh.source_spans[0]

    # Eyes should be observed
    eyes = attributes["eyes"]
    assert eyes.state in [EvidenceState.EXPLICIT, EvidenceState.EXPLICIT.value]
    assert eyes.value == "piercing blue eyes"
    assert text[eyes.start_char:eyes.end_char] == eyes.value


def test_exact_character_offset_traceability(device):
    """
    Verifies that all extracted attributes satisfy: text[start:end] == span.
    """
    ckpt_path = "checkpoints/nlp_char_bilstm_phase3c2.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join("phantom-ai-ml", ckpt_path)

    uncertainty_service = UncertaintyService()
    nlp_service = NLPService(
        checkpoint_path=ckpt_path,
        device=device,
        uncertainty_service=uncertainty_service,
    )

    text = "The individual had a sharp jawline, short dark curly hair, and a small gold stud earring."
    status, is_trained, attributes, _, _, _, _ = nlp_service.analyze_description(text)

    for cat, ev in attributes.items():
        if ev.state in [EvidenceState.EXPLICIT, EvidenceState.UNCERTAIN, EvidenceState.NEGATED]:
            assert ev.start_char is not None
            assert ev.end_char is not None
            assert ev.end_char > ev.start_char
            slice_text = text[ev.start_char:ev.end_char]
            assert slice_text == ev.source_spans[0]


