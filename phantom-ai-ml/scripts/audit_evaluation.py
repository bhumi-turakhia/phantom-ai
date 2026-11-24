import os
import json
import torch
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any

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
from evaluation.evaluate_nlp import extract_spans_from_bio_tags, run_rule_based_baseline


def audit_metrics_and_predictions():
    test_path = "datasets/nlp/test.jsonl"
    if not os.path.exists(test_path):
        test_path = os.path.join("phantom-ai-ml", test_path)
    
    checkpoints_dir = "checkpoints"
    if not os.path.exists(os.path.join(checkpoints_dir, "nlp_bilstm.pt")):
        checkpoints_dir = os.path.join("phantom-ai-ml", "checkpoints")

    loader = DatasetLoader()
    test_samples = loader.load_jsonl(test_path, validate=True)
    device = resolve_device()

    vocab = Vocabulary.load(os.path.join(checkpoints_dir, "vocab.json"))
    ckpt = torch.load(os.path.join(checkpoints_dir, "nlp_bilstm.pt"), map_location=device)

    model = BiLSTMAttributeLabeler(
        vocab_size=ckpt["vocab_size"],
        num_tags=ckpt["num_tags"],
        embedding_dim=ckpt["embedding_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
        padding_idx=0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # TASK 1: Recompute token-level and span-level raw counts
    # Token-level
    token_tp = 0
    token_fp = 0
    token_fn = 0
    token_tn = 0

    all_true_tokens = []
    all_true_tags = []
    all_pred_tags = []

    # Span-level
    ground_truth_spans = []
    model_pred_spans = []
    baseline_pred_spans = []

    error_samples = []

    with torch.no_grad():
        for s_idx, sample in enumerate(test_samples):
            tok_offsets = tokenize_with_offsets(sample.witness_text)
            tokens, true_tags = generate_bio_tags_from_spans(
                sample.witness_text,
                sample.annotations,
                tokens_with_offsets=tok_offsets,
            )

            # Ground truth spans (excluding UNKNOWN which have no spans)
            gt_spans_for_sample = []
            for ann in sample.annotations:
                if ann.certainty != EvidenceState.UNKNOWN.value and ann.start_char is not None and ann.end_char is not None:
                    cat_name = ann.category.value if isinstance(ann.category, AttributeCategory) else str(ann.category).lower()
                    span_item = (sample.sample_id, ann.start_char, ann.end_char, cat_name)
                    ground_truth_spans.append(span_item)
                    gt_spans_for_sample.append(span_item)

            # Baseline spans
            b_spans = run_rule_based_baseline(sample)
            baseline_pred_spans.extend(b_spans)

            # Neural model forward
            input_ids = torch.tensor([vocab.encode(tokens)], dtype=torch.long, device=device)
            lengths = torch.tensor([len(tokens)], dtype=torch.long)
            emissions = model(input_ids, lengths=lengths)

            decoded = model.decode_viterbi(emissions, lengths=lengths)[0]
            pred_tags = [IDX2TAG[t] for t in decoded]

            all_true_tokens.append(tokens)
            all_true_tags.append(true_tags)
            all_pred_tags.append(pred_tags)

            # Token level counts
            for t, p in zip(true_tags, pred_tags):
                if t == "O" and p == "O":
                    token_tn += 1
                elif t != "O" and p == t:
                    token_tp += 1
                elif t == "O" and p != "O":
                    token_fp += 1
                elif t != "O" and p == "O":
                    token_fn += 1
                else:  # t != 'O' and p != 'O' but t != p
                    token_fp += 1
                    token_fn += 1

            # Model spans
            m_spans = extract_spans_from_bio_tags(tok_offsets, pred_tags, sample_id=sample.sample_id)
            model_pred_spans.extend(m_spans)

            # Check if this sample has errors
            if set(m_spans) != set(gt_spans_for_sample):
                error_samples.append({
                    "sample_id": sample.sample_id,
                    "text": sample.witness_text,
                    "gt_spans": gt_spans_for_sample,
                    "pred_spans": m_spans,
                    "tokens": tokens,
                    "true_tags": true_tags,
                    "pred_tags": pred_tags,
                })

    # Token metrics calculation
    t_prec = token_tp / (token_tp + token_fp) if (token_tp + token_fp) > 0 else 0.0
    t_rec = token_tp / (token_tp + token_fn) if (token_tp + token_fn) > 0 else 0.0
    t_f1 = 2 * t_prec * t_rec / (t_prec + t_rec) if (t_prec + t_rec) > 0 else 0.0

    # Span metrics calculation (Model)
    gt_set = set(ground_truth_spans)
    pred_set = set(model_pred_spans)
    span_tp = len(gt_set.intersection(pred_set))
    span_fp = len(pred_set.difference(gt_set))
    span_fn = len(gt_set.difference(pred_set))

    s_prec = span_tp / (span_tp + span_fp) if (span_tp + span_fp) > 0 else 0.0
    s_rec = span_tp / (span_tp + span_fn) if (span_tp + span_fn) > 0 else 0.0
    s_f1 = 2 * s_prec * s_rec / (s_prec + s_rec) if (s_prec + s_rec) > 0 else 0.0

    # Span metrics calculation (Baseline)
    base_set = set(baseline_pred_spans)
    base_tp = len(gt_set.intersection(base_set))
    base_fp = len(base_set.difference(gt_set))
    base_fn = len(gt_set.difference(base_set))
    b_prec = base_tp / (base_tp + base_fp) if (base_tp + base_fp) > 0 else 0.0
    b_rec = base_tp / (base_tp + base_fn) if (base_tp + base_fn) > 0 else 0.0
    b_f1 = 2 * b_prec * b_rec / (b_prec + b_rec) if (b_prec + b_rec) > 0 else 0.0

    # Per-category raw counts
    per_cat_gt = Counter([s[3] for s in ground_truth_spans])
    per_cat_pred = Counter([s[3] for s in model_pred_spans])
    per_cat_tp = Counter([s[3] for s in gt_set.intersection(pred_set)])

    print("=== TASK 1 AUDIT RESULTS ===")
    print(f"Total Test Samples: {len(test_samples)}")
    print(f"Total Test Tokens: {sum(len(t) for t in all_true_tokens)}")
    print(f"Token-Level Raw Counts: TP={token_tp}, FP={token_fp}, FN={token_fn}, TN={token_tn}")
    print(f"Token Metrics: Prec={t_prec:.4f}, Rec={t_rec:.4f}, F1={t_f1:.4f}")
    print(f"Span-Level Raw Counts: TP={span_tp}, FP={span_fp}, FN={span_fn}")
    print(f"Total Ground Truth Spans: {len(ground_truth_spans)}, Total Model Predicted Spans: {len(model_pred_spans)}")
    print(f"Model Exact-Span: Prec={s_prec:.4f}, Rec={s_rec:.4f}, F1={s_f1:.4f}")
    print(f"Baseline Span Raw Counts: TP={base_tp}, FP={base_fp}, FN={base_fn}")
    print(f"Baseline Exact-Span: Prec={b_prec:.4f}, Rec={b_rec:.4f}, F1={b_f1:.4f}")
    print("\nPer-Category Raw Counts (GT, Pred, TP):")
    for cat in sorted(list(set(list(per_cat_gt.keys()) + list(per_cat_pred.keys())))):
        gt_c = per_cat_gt[cat]
        pred_c = per_cat_pred[cat]
        tp_c = per_cat_tp[cat]
        prec = tp_c / pred_c if pred_c > 0 else 0.0
        rec = tp_c / gt_c if gt_c > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"  {cat:22s} | GT={gt_c:2d} | Pred={pred_c:3d} | TP={tp_c:2d} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}")

    print(f"\nSamples with Errors: {len(error_samples)} / {len(test_samples)}")

    # Save detailed error log for Task 2
    with open("checkpoints/audit_error_samples.json", "w", encoding="utf-8") as f:
        json.dump(error_samples, f, indent=2)

    return error_samples


if __name__ == "__main__":
    audit_metrics_and_predictions()

