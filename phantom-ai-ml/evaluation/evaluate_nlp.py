import os
import json
import argparse
from typing import List, Dict, Any, Tuple, Set, Optional
from collections import defaultdict
import torch

from utils.device import resolve_device
from utils.preprocessing import (
    TAGS,
    TAG2IDX,
    IDX2TAG,
    tokenize_with_offsets,
    generate_bio_tags_from_spans,
    Vocabulary,
)
from models.nlp_model import BiLSTMAttributeLabeler
from data.dataset_loader import DatasetLoader
from schemas.attributes import AttributeCategory, EvidenceState
from schemas.dataset import DatasetSample
from services.nlp_service import OBSERVABLE_PATTERNS
from evaluation.evaluate_metrics import (
    compute_sequence_labeling_metrics,
    compute_exact_span_metrics,
    compute_per_attribute_metrics,
    compute_annotation_coverage,
)


def extract_spans_from_bio_tags(
    tokens_with_offsets: List[Tuple[str, int, int]],
    tags: List[str],
    sample_id: str = "",
) -> List[Tuple[str, int, int, str]]:
    """
    Extracts entity spans from aligned token offsets and BIO sequence tags.
    Returns: List of (sample_id, start_char, end_char, category_name)
    """
    spans = []
    current_cat: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    for (token_str, t_start, t_end), tag in zip(tokens_with_offsets, tags):
        if tag.startswith("B-"):
            if current_cat is not None:
                spans.append((sample_id, start_char, end_char, current_cat))
            current_cat = tag[2:].lower()
            start_char = t_start
            end_char = t_end
        elif tag.startswith("I-") and current_cat is not None and tag[2:].lower() == current_cat:
            end_char = t_end
        else:
            if current_cat is not None:
                spans.append((sample_id, start_char, end_char, current_cat))
                current_cat = None
                start_char = None
                end_char = None

    if current_cat is not None:
        spans.append((sample_id, start_char, end_char, current_cat))

    return spans


def run_rule_based_baseline(sample: DatasetSample) -> List[Tuple[str, int, int, str]]:
    """
    Rule-based regex baseline for forensic attribute extraction.
    Used for objective comparison against custom neural model on the exact same test split.
    """
    import re
    spans = []
    text = sample.witness_text

    for cat in AttributeCategory:
        patterns = OBSERVABLE_PATTERNS.get(cat, [])
        for pat in patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                spans.append((sample.sample_id, match.start(), match.end(), cat.value))

    return spans


def build_confusion_matrix(
    all_true_tags: List[List[str]],
    all_pred_tags: List[List[str]],
    tag_list: List[str] = TAGS,
) -> Dict[str, Dict[str, int]]:
    """
    Constructs a tag-to-tag confusion matrix across all sequence tokens.
    """
    matrix: Dict[str, Dict[str, int]] = {t: {p: 0 for p in tag_list} for t in tag_list}
    for trues, preds in zip(all_true_tags, all_pred_tags):
        for t, p in zip(trues, preds):
            if t in matrix and p in matrix[t]:
                matrix[t][p] += 1
    return matrix


def evaluate_trained_nlp_model(
    test_path: str = "datasets/nlp/test.jsonl",
    checkpoints_dir: str = "checkpoints",
    output_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs comprehensive, rigorous evaluation on the held-out test split.
    Adheres strictly to honesty constraints:
    - Zero fabricated metrics.
    - Test set used exclusively for evaluation.
    - Explicit disclosure that synthetic benchmark results do not claim real-world forensic accuracy.
    """
    checkpoint_file = os.path.join(checkpoints_dir, "nlp_bilstm.pt")
    vocab_file = os.path.join(checkpoints_dir, "vocab.json")

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_file}. Train model first.")
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_file}.")

    # Load test data
    if not os.path.exists(test_path):
        alt_test = os.path.join("phantom-ai-ml", test_path)
        if os.path.exists(alt_test):
            test_path = alt_test
        else:
            raise FileNotFoundError(f"Test split not found at {test_path}")

    loader_tool = DatasetLoader()
    test_samples = loader_tool.load_jsonl(test_path, validate=True)

    # Load vocabulary and model checkpoint
    device = resolve_device()
    vocab = Vocabulary.load(vocab_file)
    checkpoint = torch.load(checkpoint_file, map_location=device)

    model = BiLSTMAttributeLabeler(
        vocab_size=checkpoint["vocab_size"],
        num_tags=checkpoint["num_tags"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        padding_idx=0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Ground truth vs Model vs Rule-based Baseline collection
    all_true_tokens: List[List[str]] = []
    all_true_tags: List[List[str]] = []

    model_pred_tags: List[List[str]] = []

    ground_truth_spans: List[Tuple[str, int, int, str]] = []
    model_pred_spans: List[Tuple[str, int, int, str]] = []
    baseline_pred_spans: List[Tuple[str, int, int, str]] = []

    error_analysis_cases: List[Dict[str, Any]] = []

    with torch.no_grad():
        for sample in test_samples:
            tok_offsets = tokenize_with_offsets(sample.witness_text)
            tokens, true_tags = generate_bio_tags_from_spans(
                sample.witness_text,
                sample.annotations,
                tokens_with_offsets=tok_offsets,
            )
            all_true_tokens.append(tokens)
            all_true_tags.append(true_tags)

            # Ground truth spans
            for ann in sample.annotations:
                if (
                    ann.certainty != EvidenceState.UNKNOWN.value
                    and ann.start_char is not None
                    and ann.end_char is not None
                ):
                    cat_name = (
                        ann.category.value
                        if isinstance(ann.category, AttributeCategory)
                        else str(ann.category).lower()
                    )
                    ground_truth_spans.append(
                        (sample.sample_id, ann.start_char, ann.end_char, cat_name)
                    )

            # Baseline spans
            b_spans = run_rule_based_baseline(sample)
            baseline_pred_spans.extend(b_spans)

            # Model inference
            input_ids = torch.tensor([vocab.encode(tokens)], dtype=torch.long, device=device)
            lengths = torch.tensor([len(tokens)], dtype=torch.long)
            emissions = model(input_ids, lengths=lengths)

            decoded = model.decode_viterbi(emissions, lengths=lengths)[0]
            pred_tags = [IDX2TAG[t] for t in decoded]
            model_pred_tags.append(pred_tags)

            # Model predicted spans
            m_spans = extract_spans_from_bio_tags(tok_offsets, pred_tags, sample_id=sample.sample_id)
            model_pred_spans.extend(m_spans)

            # Log errors for analysis if prediction differs from ground truth
            sample_gt_spans = [
                s for s in ground_truth_spans if s[0] == sample.sample_id
            ]
            if set(m_spans) != set(sample_gt_spans):
                error_analysis_cases.append({
                    "sample_id": sample.sample_id,
                    "text": sample.witness_text,
                    "ground_truth_spans": [
                        {"category": s[3], "start": s[1], "end": s[2], "span": sample.witness_text[s[1]:s[2]]}
                        for s in sample_gt_spans
                    ],
                    "model_spans": [
                        {"category": s[3], "start": s[1], "end": s[2], "span": sample.witness_text[s[1]:s[2]]}
                        for s in m_spans
                    ],
                    "tokens": tokens,
                    "true_tags": true_tags,
                    "pred_tags": pred_tags,
                })

    # Metric Computations
    token_metrics = compute_sequence_labeling_metrics(all_true_tags, model_pred_tags, ignore_tag="O")
    span_metrics = compute_exact_span_metrics(ground_truth_spans, model_pred_spans)
    per_attribute_metrics = compute_per_attribute_metrics(ground_truth_spans, model_pred_spans)

    # Baseline comparison metrics
    baseline_span_metrics = compute_exact_span_metrics(ground_truth_spans, baseline_pred_spans)
    baseline_per_attribute = compute_per_attribute_metrics(ground_truth_spans, baseline_pred_spans)

    # Confusion Matrix
    confusion_mat = build_confusion_matrix(all_true_tags, model_pred_tags)

    # Coverage
    coverage_stats = compute_annotation_coverage(test_samples)

    report = {
        "evaluation_scope": "Held-out synthetic test set (datasets/nlp/test.jsonl)",
        "test_sample_count": len(test_samples),
        "total_test_tokens": sum(len(t) for t in all_true_tokens),
        "total_ground_truth_spans": len(ground_truth_spans),
        "trained_model_metrics": {
            "token_level": token_metrics,
            "exact_span_level": span_metrics,
            "per_attribute": per_attribute_metrics,
        },
        "baseline_comparison": {
            "baseline_type": "Deterministic Regex Pattern Matcher",
            "exact_span_level": baseline_span_metrics,
            "per_attribute": baseline_per_attribute,
        },
        "confusion_matrix": confusion_mat,
        "coverage": coverage_stats,
        "error_analysis": {
            "total_errors": len(error_analysis_cases),
            "representative_failures": error_analysis_cases[:5],
        },
        "disclaimer": (
            "NOTICE: Evaluation was conducted strictly on the synthetic forensic benchmark. "
            "These metrics demonstrate successful pipeline integration and model convergence from scratch, "
            "but do NOT establish production-grade forensic accuracy on real witness statements."
        ),
    }

    if output_report_path is None:
        output_report_path = os.path.join(checkpoints_dir, "evaluation_results.json")

    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 70)
    print("PHANTOM AI 3.0 — HELD-OUT TEST EVALUATION REPORT")
    print(f"Test Samples: {len(test_samples)} | Ground Truth Spans: {len(ground_truth_spans)}")
    print(f"Model Token-level F1: {token_metrics['f1']:.4f} (P: {token_metrics['precision']:.4f}, R: {token_metrics['recall']:.4f})")
    print(f"Model Exact-Span F1:  {span_metrics['f1']:.4f} (P: {span_metrics['precision']:.4f}, R: {span_metrics['recall']:.4f})")
    print(f"Baseline Exact-Span F1: {baseline_span_metrics['f1']:.4f} (P: {baseline_span_metrics['precision']:.4f}, R: {baseline_span_metrics['recall']:.4f})")
    print("=" * 70)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained BiLSTM-CRF on held-out test split.")
    parser.add_argument("--test-path", type=str, default="datasets/nlp/test.jsonl")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    evaluate_trained_nlp_model(
        test_path=args.test_path,
        checkpoints_dir=args.checkpoints_dir,
    )
