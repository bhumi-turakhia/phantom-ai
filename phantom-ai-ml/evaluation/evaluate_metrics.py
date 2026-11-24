from typing import List, Dict, Any, Tuple, Set, Optional
from collections import defaultdict
from schemas.attributes import AttributeCategory, EvidenceState
from schemas.dataset import DatasetSample, AttributeAnnotation


def compute_sequence_labeling_metrics(
    true_tags: List[List[str]],
    pred_tags: List[List[str]],
    ignore_tag: str = "O"
) -> Dict[str, float]:
    """
    Computes strict token-level Precision, Recall, and F1 score for sequence labeling.
    """
    tp = 0
    fp = 0
    fn = 0

    for trues, preds in zip(true_tags, pred_tags):
        for t, p in zip(trues, preds):
            if t == ignore_tag and p == ignore_tag:
                continue
            elif t == p:
                tp += 1
            elif t == ignore_tag and p != ignore_tag:
                fp += 1
            elif t != ignore_tag and p == ignore_tag:
                fn += 1
            else:
                fp += 1
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_exact_span_metrics(
    true_spans: List[Tuple[str, int, int, str]],
    pred_spans: List[Tuple[str, int, int, str]],
) -> Dict[str, Any]:
    """
    Computes exact span-level Precision, Recall, and F1 metrics.
    A true positive requires an exact match on (sample_id, start_char, end_char, category).

    Args:
        true_spans: List of (sample_id, start_char, end_char, category)
        pred_spans: List of (sample_id, start_char, end_char, category)
    """
    true_set: Set[Tuple[str, int, int, str]] = set(true_spans)
    pred_set: Set[Tuple[str, int, int, str]] = set(pred_spans)

    tp_set = true_set.intersection(pred_set)
    fp_set = pred_set.difference(true_set)
    fn_set = true_set.difference(pred_set)

    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_ground_truth_spans": len(true_set),
        "total_predicted_spans": len(pred_set),
    }


def compute_per_attribute_metrics(
    true_spans: List[Tuple[str, int, int, str]],
    pred_spans: List[Tuple[str, int, int, str]],
    categories: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Computes exact span Precision, Recall, and F1 broken down per attribute category.
    """
    if categories is None:
        categories = [c.value for c in AttributeCategory]

    per_cat_true: Dict[str, Set[Tuple[str, int, int]]] = defaultdict(set)
    per_cat_pred: Dict[str, Set[Tuple[str, int, int]]] = defaultdict(set)

    for sid, start, end, cat in true_spans:
        per_cat_true[cat.lower()].add((sid, start, end))

    for sid, start, end, cat in pred_spans:
        per_cat_pred[cat.lower()].add((sid, start, end))

    results: Dict[str, Dict[str, float]] = {}

    for cat in categories:
        cat_lower = cat.lower()
        t_set = per_cat_true[cat_lower]
        p_set = per_cat_pred[cat_lower]

        tp = len(t_set.intersection(p_set))
        fp = len(p_set.difference(t_set))
        fn = len(t_set.difference(p_set))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        results[cat_lower] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "ground_truth_count": len(t_set),
            "predicted_count": len(p_set),
        }

    return results


def compute_annotation_coverage(samples: List[DatasetSample]) -> Dict[str, Any]:
    """
    Computes annotation coverage and density statistics across a dataset.
    """
    total_samples = len(samples)
    if total_samples == 0:
        return {"total_samples": 0}

    all_categories = [c.value for c in AttributeCategory]
    category_sample_counts: Dict[str, int] = defaultdict(int)
    certainty_counts: Dict[str, int] = defaultdict(int)
    total_annotations = 0
    total_active_spans = 0

    for sample in samples:
        observed_categories_in_sample: Set[str] = set()
        for ann in sample.annotations:
            total_annotations += 1
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
            certainty_counts[certainty_str] += 1

            if certainty_str != EvidenceState.UNKNOWN.value:
                total_active_spans += 1
                observed_categories_in_sample.add(cat_str)

        for cat in observed_categories_in_sample:
            category_sample_counts[cat] += 1

    per_category_coverage = {
        cat: round(category_sample_counts[cat] / total_samples * 100, 1)
        for cat in all_categories
    }

    return {
        "total_samples": total_samples,
        "total_annotations": total_annotations,
        "total_active_spans": total_active_spans,
        "avg_active_spans_per_sample": round(total_active_spans / total_samples, 2),
        "per_category_coverage_percent": per_category_coverage,
        "certainty_distribution": dict(certainty_counts),
    }
