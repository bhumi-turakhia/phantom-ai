import os
import pytest
import tempfile
from schemas.attributes import AttributeCategory, EvidenceState
from schemas.dataset import (
    DatasetSample,
    AttributeAnnotation,
    SampleMetadata,
    ForensicDataset,
)
from data.validator import DatasetValidator
from data.splitter import DatasetSplitter
from data.dataset_loader import DatasetLoader
from data.sample_dataset import get_synthetic_dataset
from utils.preprocessing import (
    tokenize_with_offsets,
    generate_bio_tags_from_spans,
    TAGS,
)
from evaluation.evaluate_metrics import (
    compute_sequence_labeling_metrics,
    compute_exact_span_metrics,
    compute_per_attribute_metrics,
    compute_annotation_coverage,
)


@pytest.fixture
def synthetic_dataset() -> ForensicDataset:
    return get_synthetic_dataset()


@pytest.fixture
def validator() -> DatasetValidator:
    return DatasetValidator()


@pytest.fixture
def loader() -> DatasetLoader:
    return DatasetLoader()


def test_valid_dataset_loading_json(synthetic_dataset, validator, loader):
    summary = validator.validate_dataset(synthetic_dataset.samples)
    assert summary.is_valid is True
    assert summary.total_samples == 24
    assert summary.valid_samples == 24
    assert summary.invalid_samples == 0
    assert len(summary.errors) == 0
    assert summary.statistics["all_synthetic"] is True


def test_character_offset_verification(synthetic_dataset):
    """
    Every character offset must strictly match the verbatim witness description.
    """
    for sample in synthetic_dataset.samples:
        errors = sample.verify_offsets()
        assert len(errors) == 0, f"Offset error in sample {sample.sample_id}: {errors}"
        for ann in sample.annotations:
            if ann.certainty != EvidenceState.UNKNOWN.value:
                assert ann.start_char is not None and ann.end_char is not None
                assert ann.source_span is not None
                assert sample.witness_text[ann.start_char:ann.end_char] == ann.source_span


def test_invalid_dataset_rejection_offset_mismatch(validator):
    bad_sample = DatasetSample(
        sample_id="BAD-001",
        witness_text="The suspect had dark curly hair.",
        annotations=[
            AttributeAnnotation(
                annotation_id="BAD-ANN-1",
                category=AttributeCategory.HAIR,
                attribute_value="dark curly hair",
                source_span="dark curly hair",
                start_char=4,  # Intentionally wrong offset (should be 16)
                end_char=19,
                certainty=EvidenceState.EXPLICIT,
                annotator_id="TEST-ANNOTATOR",
            )
        ],
        metadata=SampleMetadata(
            dataset_version="1.0.0",
            source="Test Suite",
            license="CC-BY-4.0-Synthetic",
            is_synthetic=True,
        ),
    )
    is_valid, errors, _ = validator.validate_sample(bad_sample)
    assert is_valid is False
    assert any("Character offset mismatch" in err for err in errors)


def test_invalid_dataset_rejection_out_of_bounds(validator):
    bad_sample = DatasetSample(
        sample_id="BAD-002",
        witness_text="Short description.",
        annotations=[
            AttributeAnnotation(
                annotation_id="BAD-ANN-2",
                category=AttributeCategory.NOSE,
                attribute_value="crooked nose",
                source_span="crooked nose",
                start_char=5,
                end_char=500,  # Far beyond witness_text length
                certainty=EvidenceState.EXPLICIT,
                annotator_id="TEST-ANNOTATOR",
            )
        ],
        metadata=SampleMetadata(
            dataset_version="1.0.0",
            source="Test Suite",
            license="CC-BY-4.0-Synthetic",
            is_synthetic=True,
        ),
    )
    is_valid, errors, _ = validator.validate_sample(bad_sample)
    assert is_valid is False
    assert any("exceeds witness_text length" in err for err in errors)


def test_forbidden_demographic_attributes_rejection():
    """
    Enforces that demographic categories (race, ethnicity, etc.) are strictly prohibited.
    """
    with pytest.raises(ValueError) as excinfo:
        AttributeAnnotation(
            annotation_id="ILLEGAL-DEMO-1",
            category="race",  # Forbidden demographic attribute
            attribute_value="caucasian",
            source_span="caucasian",
            start_char=0,
            end_char=9,
            certainty=EvidenceState.EXPLICIT,
            annotator_id="TEST-ANNOTATOR",
        )
    assert "Forbidden attribute category" in str(excinfo.value) or "Input should be" in str(excinfo.value)


def test_unknown_attributes_have_no_spans():
    """
    Unknown features must never have character spans or fabricated values.
    """
    with pytest.raises(ValueError):
        AttributeAnnotation(
            annotation_id="BAD-UNK-1",
            category=AttributeCategory.EYES,
            attribute_value="blue eyes",
            source_span="blue eyes",
            start_char=0,
            end_char=9,
            certainty=EvidenceState.UNKNOWN,  # Contradiction: UNKNOWN with span
            annotator_id="TEST-ANNOTATOR",
        )


def test_bio_tag_generation_alignment(synthetic_dataset):
    sample_1 = synthetic_dataset.samples[0]  # SYNTH-001
    tokens_with_offsets = tokenize_with_offsets(sample_1.witness_text)
    tokens, tags = generate_bio_tags_from_spans(
        sample_1.witness_text,
        sample_1.annotations,
        tokens_with_offsets=tokens_with_offsets,
    )

    assert len(tokens) == len(tags)
    assert "B-FACE_SHAPE" in tags
    assert "B-HAIR" in tags
    assert "B-EYES" in tags

    # Verify every tag exists in TAGS taxonomy
    for tag in tags:
        assert tag in TAGS


def test_contradiction_handling(synthetic_dataset, validator):
    """
    SYNTH-003 and SYNTH-011 test conflicting witness claims (e.g. bald vs long hair).
    """
    sample_3 = next(s for s in synthetic_dataset.samples if s.sample_id == "SYNTH-003")
    contradictory_anns = [
        ann for ann in sample_3.annotations
        if ann.certainty == EvidenceState.CONTRADICTORY.value
    ]
    assert len(contradictory_anns) == 2
    assert any("bald" in ann.source_span for ann in contradictory_anns)
    assert any("wavy hair" in ann.source_span for ann in contradictory_anns)

    is_valid, errors, _ = validator.validate_sample(sample_3)
    assert is_valid is True


def test_dataset_splitting_reproducibility(synthetic_dataset):
    splitter_1 = DatasetSplitter(seed=42)
    split_1 = splitter_1.split(synthetic_dataset.samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    splitter_2 = DatasetSplitter(seed=42)
    split_2 = splitter_2.split(synthetic_dataset.samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    # Identical seed must produce identical splits
    assert [s.sample_id for s in split_1["train"]] == [s.sample_id for s in split_2["train"]]
    assert [s.sample_id for s in split_1["val"]] == [s.sample_id for s in split_2["val"]]
    assert [s.sample_id for s in split_1["test"]] == [s.sample_id for s in split_2["test"]]

    # Total sample count preserved
    total = len(split_1["train"]) + len(split_1["val"]) + len(split_1["test"])
    assert total == len(synthetic_dataset.samples)


def test_dataset_split_leakage_prevention():
    """
    Near-duplicate witness statements must be grouped to prevent train/test leakage.
    """
    meta = SampleMetadata(dataset_version="1.0.0", source="Test", license="CC-BY-4.0-Synthetic", is_synthetic=True)
    samples = [
        DatasetSample(sample_id="D1", witness_text="Suspect had black hair and brown eyes.", annotations=[], metadata=meta),
        DatasetSample(sample_id="D2", witness_text="Suspect had black hair and brown eyes.", annotations=[], metadata=meta),  # Exact duplicate
        DatasetSample(sample_id="S3", witness_text="Person had oval face.", annotations=[], metadata=meta),
        DatasetSample(sample_id="S4", witness_text="Subject wore a buzzcut.", annotations=[], metadata=meta),
        DatasetSample(sample_id="S5", witness_text="Witness reported a thin mustache.", annotations=[], metadata=meta),
    ]
    splitter = DatasetSplitter(seed=123)
    split = splitter.split(samples, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    # D1 and D2 must both end up in the exact same split partition
    train_ids = {s.sample_id for s in split["train"]}
    val_ids = {s.sample_id for s in split["val"]}
    test_ids = {s.sample_id for s in split["test"]}

    in_train = ("D1" in train_ids) and ("D2" in train_ids)
    in_val = ("D1" in val_ids) and ("D2" in val_ids)
    in_test = ("D1" in test_ids) and ("D2" in test_ids)

    assert in_train or in_val or in_test, "Duplicate samples leaked across different partitions!"


def test_synthetic_dataset_identification(synthetic_dataset):
    assert synthetic_dataset.is_synthetic is True
    assert synthetic_dataset.license == "CC-BY-4.0-Synthetic"
    for s in synthetic_dataset.samples:
        assert s.metadata.is_synthetic is True
        assert s.metadata.license == "CC-BY-4.0-Synthetic"


def test_export_and_reload_consistency(synthetic_dataset, loader):
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test_dataset.json")
        jsonl_path = os.path.join(tmpdir, "test_dataset.jsonl")

        # Save JSON & reload
        loader.save_json(synthetic_dataset, json_path)
        reloaded_json = loader.load_json(json_path)
        assert len(reloaded_json.samples) == len(synthetic_dataset.samples)
        assert reloaded_json.samples[0].sample_id == synthetic_dataset.samples[0].sample_id
        assert reloaded_json.samples[0].witness_text == synthetic_dataset.samples[0].witness_text

        # Save JSONL & reload
        loader.save_jsonl(synthetic_dataset.samples, jsonl_path)
        reloaded_jsonl = loader.load_jsonl(jsonl_path)
        assert len(reloaded_jsonl) == len(synthetic_dataset.samples)

        # Export to BIO format
        bio_data = loader.export_to_bio_format(synthetic_dataset.samples)
        assert len(bio_data) == len(synthetic_dataset.samples)
        assert "tokens" in bio_data[0]
        assert "bio_tags" in bio_data[0]
        assert len(bio_data[0]["tokens"]) == len(bio_data[0]["bio_tags"])


def test_evaluation_metrics_calculations():
    # 1. Exact span metrics
    true_spans = [
        ("S1", 10, 20, "hair"),
        ("S1", 25, 35, "eyes"),
        ("S2", 5, 15, "nose"),
    ]
    # Predict 2 correct spans, 1 false positive, 1 false negative
    pred_spans = [
        ("S1", 10, 20, "hair"),  # TP
        ("S1", 25, 35, "eyes"),  # TP
        ("S2", 5, 18, "nose"),  # FP (wrong end)
    ]
    span_metrics = compute_exact_span_metrics(true_spans, pred_spans)
    assert span_metrics["tp"] == 2
    assert span_metrics["fp"] == 1
    assert span_metrics["fn"] == 1
    assert span_metrics["precision"] == round(2 / 3, 4)
    assert span_metrics["recall"] == round(2 / 3, 4)

    # 2. Per-attribute metrics
    per_cat = compute_per_attribute_metrics(true_spans, pred_spans, categories=["hair", "eyes", "nose"])
    assert per_cat["hair"]["f1"] == 1.0
    assert per_cat["eyes"]["f1"] == 1.0
    assert per_cat["nose"]["f1"] == 0.0

    # 3. Token-level metrics
    true_tags = [["O", "B-HAIR", "I-HAIR", "O"]]
    pred_tags = [["O", "B-HAIR", "I-HAIR", "O"]]
    token_metrics = compute_sequence_labeling_metrics(true_tags, pred_tags)
    assert token_metrics["f1"] == 1.0

