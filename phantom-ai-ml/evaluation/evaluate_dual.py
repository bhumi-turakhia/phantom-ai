"""
evaluate_dual.py
PHANTOM AI 3.0 — Phase 3B.2

Dual evaluation runner:
  A. In-domain held-out synthetic test set (datasets/nlp/test.jsonl — 83 samples)
  B. Out-of-frame independent evaluation corpus (datasets/nlp/out_of_frame_eval.jsonl — 106 samples)

Evaluates:
  - Trained BiLSTM-CRF (CRF NLL, Phase 3B.2 checkpoint)
  - Rule-based regex baseline

Reports:
  - Token-level precision / recall / F1 with TP, FP, FN
  - Exact-span precision / recall / F1 with TP, FP, FN
  - Per-category exact-span metrics
  - Failure cases
  - Side-by-side comparison: Neural CRF vs Rule-based Baseline

CONSTRAINTS:
  - The out-of-frame corpus is used ONLY for evaluation — zero gradient updates.
  - The test split is evaluated ONLY once (after final model selection).
  - No metrics are fabricated or modified.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import torch

# Resolve project path so script can run from any cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
import sys
sys.path.insert(0, _ML_ROOT)

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
)


# ---------------------------------------------------------------------------
# Utility: extract char-level spans from BIO tag sequence
# ---------------------------------------------------------------------------

def extract_spans(
    tokens_with_offsets: List[Tuple[str, int, int]],
    tags: List[str],
    sample_id: str = "",
) -> List[Tuple[str, int, int, str]]:
    spans = []
    current_cat: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    for (tok, t_start, t_end), tag in zip(tokens_with_offsets, tags):
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
    if current_cat is not None:
        spans.append((sample_id, start_char, end_char, current_cat))
    return spans


# ---------------------------------------------------------------------------
# Utility: rule-based baseline
# ---------------------------------------------------------------------------

def run_baseline(sample: DatasetSample) -> List[Tuple[str, int, int, str]]:
    import re
    spans = []
    text = sample.witness_text
    for cat in AttributeCategory:
        patterns = OBSERVABLE_PATTERNS.get(cat, [])
        for pat in patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                spans.append((sample.sample_id, match.start(), match.end(), cat.value))
    return spans


# ---------------------------------------------------------------------------
# Utility: compute raw token TP/FP/FN
# ---------------------------------------------------------------------------

def token_raw_counts(
    all_true: List[List[str]],
    all_pred: List[List[str]],
    ignore_tag: str = "O",
) -> Tuple[int, int, int]:
    tp = fp = fn = 0
    for trues, preds in zip(all_true, all_pred):
        for t, p in zip(trues, preds):
            if t == ignore_tag and p == ignore_tag:
                continue
            if t != ignore_tag and p == t:
                tp += 1
            elif t == ignore_tag and p != ignore_tag:
                fp += 1
            elif t != ignore_tag and p == ignore_tag:
                fn += 1
            elif t != ignore_tag and p != ignore_tag and t != p:
                # Wrong label — counts as FP + FN
                fp += 1
                fn += 1
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Utility: compute raw span TP/FP/FN
# ---------------------------------------------------------------------------

def span_raw_counts(
    gold: List[Tuple],
    pred: List[Tuple],
) -> Tuple[int, int, int]:
    gold_set = set(gold)
    pred_set = set(pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Core evaluation of one dataset split
# ---------------------------------------------------------------------------

def evaluate_split(
    samples: List[DatasetSample],
    model: BiLSTMAttributeLabeler,
    vocab: Vocabulary,
    device: torch.device,
    split_name: str,
) -> Dict[str, Any]:
    """
    Evaluates both the neural CRF model and the rule-based baseline on a split.
    Returns a structured result dict with all metrics and failure cases.
    No gradient updates are performed.
    """
    all_true_tags: List[List[str]] = []
    all_pred_tags: List[List[str]] = []

    gold_spans: List[Tuple] = []
    model_spans: List[Tuple] = []
    baseline_spans: List[Tuple] = []

    failures: List[Dict] = []

    model.eval()
    with torch.no_grad():
        for sample in samples:
            tok_offsets = tokenize_with_offsets(sample.witness_text)
            tokens, true_tags = generate_bio_tags_from_spans(
                sample.witness_text,
                sample.annotations,
                tokens_with_offsets=tok_offsets,
            )
            all_true_tags.append(true_tags)

            # Ground truth spans (exclude UNKNOWN annotations)
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
                    gold_spans.append(
                        (sample.sample_id, ann.start_char, ann.end_char, cat_name)
                    )

            # Model inference
            input_ids = torch.tensor(
                [vocab.encode(tokens)], dtype=torch.long, device=device
            )
            lengths = torch.tensor([len(tokens)], dtype=torch.long)
            emissions = model(input_ids, lengths=lengths)
            decoded = model.decode_viterbi(emissions, lengths=lengths)[0]
            pred_tags = [IDX2TAG[t] for t in decoded]
            all_pred_tags.append(pred_tags)

            m_spans = extract_spans(tok_offsets, pred_tags, sample_id=sample.sample_id)
            model_spans.extend(m_spans)

            # Baseline inference
            b_spans = run_baseline(sample)
            baseline_spans.extend(b_spans)

            # Failure cases
            sample_gold = [s for s in gold_spans if s[0] == sample.sample_id]
            if set(m_spans) != set(sample_gold):
                failures.append({
                    "sample_id": sample.sample_id,
                    "text": sample.witness_text,
                    "gold_spans": [
                        {"category": s[3], "start": s[1], "end": s[2],
                         "text": sample.witness_text[s[1]:s[2]]}
                        for s in sample_gold
                    ],
                    "model_spans": [
                        {"category": s[3], "start": s[1], "end": s[2],
                         "text": sample.witness_text[s[1]:s[2]]}
                        for s in m_spans
                    ],
                    "true_tags": true_tags,
                    "pred_tags": pred_tags,
                })

    # ---- Token-level metrics ----
    tok_metrics = compute_sequence_labeling_metrics(all_true_tags, all_pred_tags, ignore_tag="O")
    tok_tp, tok_fp, tok_fn = token_raw_counts(all_true_tags, all_pred_tags)

    # ---- Span-level metrics ----
    span_metrics = compute_exact_span_metrics(gold_spans, model_spans)
    span_tp, span_fp, span_fn = span_raw_counts(gold_spans, model_spans)
    per_cat = compute_per_attribute_metrics(gold_spans, model_spans)

    # ---- Baseline metrics ----
    base_span_metrics = compute_exact_span_metrics(gold_spans, baseline_spans)
    base_span_tp, base_span_fp, base_span_fn = span_raw_counts(gold_spans, baseline_spans)
    base_per_cat = compute_per_attribute_metrics(gold_spans, baseline_spans)

    return {
        "split": split_name,
        "sample_count": len(samples),
        "gold_span_count": len(gold_spans),
        "neural_crf": {
            "token_level": {
                **tok_metrics,
                "tp": tok_tp,
                "fp": tok_fp,
                "fn": tok_fn,
            },
            "exact_span": {
                **span_metrics,
                "tp": span_tp,
                "fp": span_fp,
                "fn": span_fn,
            },
            "per_category": per_cat,
        },
        "rule_based_baseline": {
            "exact_span": {
                **base_span_metrics,
                "tp": base_span_tp,
                "fp": base_span_fp,
                "fn": base_span_fn,
            },
            "per_category": base_per_cat,
        },
        "failure_cases": {
            "total": len(failures),
            "representative": failures[:8],
        },
        "disclaimer": (
            "NOTICE: Results are measured on synthetic forensic benchmark data only. "
            "They demonstrate pipeline integration and model learning from scratch "
            "but do NOT establish production forensic accuracy on real witness statements."
        ),
    }


# ---------------------------------------------------------------------------
# Main dual evaluation runner
# ---------------------------------------------------------------------------

def run_dual_evaluation(
    test_path: str = "datasets/nlp/test.jsonl",
    oof_path: str = "datasets/nlp/out_of_frame_eval.jsonl",
    checkpoints_dir: str = "checkpoints",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:

    # --- Resolve paths ---
    def resolve(p: str) -> str:
        if os.path.exists(p):
            return p
        alt = os.path.join(_ML_ROOT, p)
        if os.path.exists(alt):
            return alt
        # try from project root
        proj_root = os.path.abspath(os.path.join(_ML_ROOT, ".."))
        alt2 = os.path.join(proj_root, p)
        if os.path.exists(alt2):
            return alt2
        raise FileNotFoundError(f"Cannot find file: {p}")

    test_path = resolve(test_path)
    oof_path = resolve(oof_path)
    checkpoints_dir = resolve(checkpoints_dir) if not os.path.isabs(checkpoints_dir) else checkpoints_dir

    chk_file = os.path.join(checkpoints_dir, "nlp_bilstm.pt")
    vocab_file = os.path.join(checkpoints_dir, "vocab.json")

    for f in [chk_file, vocab_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file not found: {f}")

    # --- Load model ---
    device = resolve_device()
    vocab = Vocabulary.load(vocab_file)
    chk = torch.load(chk_file, map_location=device)

    model = BiLSTMAttributeLabeler(
        vocab_size=chk["vocab_size"],
        num_tags=chk["num_tags"],
        embedding_dim=chk["embedding_dim"],
        hidden_dim=chk["hidden_dim"],
        num_layers=chk["num_layers"],
        dropout=chk["dropout"],
        padding_idx=0,
    )
    model.load_state_dict(chk["model_state_dict"])
    model.to(device)
    model.eval()

    training_objective = chk.get("training_objective", "unknown — likely CrossEntropyLoss (Phase 3B.1)")
    phase = chk.get("phase", "unknown")

    print("=" * 70)
    print("PHANTOM AI 3.0 — PHASE 3B.2 DUAL EVALUATION")
    print(f"Device: {device}")
    print(f"Checkpoint phase: {phase}")
    print(f"Training objective: {training_objective}")
    print(f"Best epoch: {chk.get('epoch', 'N/A')} | Val F1: {chk.get('val_f1', 'N/A')}")
    print("=" * 70)

    # --- Load datasets ---
    loader = DatasetLoader()
    test_samples = loader.load_jsonl(test_path, validate=True)
    oof_samples = loader.load_jsonl(oof_path, validate=False)  # OOF may not have full schema

    print(f"Loaded {len(test_samples)} in-domain test samples")
    print(f"Loaded {len(oof_samples)} out-of-frame eval samples")

    # --- Evaluate A: In-domain test set ---
    print("\n--- Evaluating A: In-domain test set ---")
    results_A = evaluate_split(test_samples, model, vocab, device, "A_in_domain_test")

    # --- Evaluate B: Out-of-frame corpus ---
    print("--- Evaluating B: Out-of-frame evaluation corpus ---")
    results_B = evaluate_split(oof_samples, model, vocab, device, "B_out_of_frame_eval")

    # --- Report ---
    report = {
        "audit": "Phase 3B.2 — CRF Training Verification and Out-of-Frame Generalisation",
        "checkpoint": {
            "file": chk_file,
            "training_objective": training_objective,
            "phase": phase,
            "best_epoch": chk.get("epoch"),
            "val_f1_at_best": chk.get("val_f1"),
        },
        "set_A_in_domain_test": results_A,
        "set_B_out_of_frame_eval": results_B,
    }

    # Print summary table
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    for label, r in [("A  IN-DOMAIN TEST (83 samples)", results_A),
                     ("B  OUT-OF-FRAME  (106 samples)", results_B)]:
        nn = r["neural_crf"]
        rb = r["rule_based_baseline"]
        print(f"\n  {label}")
        print(f"    Gold spans       : {r['gold_span_count']}")
        print(f"    Neural  Token F1 : {nn['token_level']['f1']:.4f}  (P={nn['token_level']['precision']:.4f}  R={nn['token_level']['recall']:.4f})  TP={nn['token_level']['tp']} FP={nn['token_level']['fp']} FN={nn['token_level']['fn']}")
        print(f"    Neural  Span  F1 : {nn['exact_span']['f1']:.4f}  (P={nn['exact_span']['precision']:.4f}  R={nn['exact_span']['recall']:.4f})  TP={nn['exact_span']['tp']} FP={nn['exact_span']['fp']} FN={nn['exact_span']['fn']}")
        print(f"    Baseline Span F1 : {rb['exact_span']['f1']:.4f}  (P={rb['exact_span']['precision']:.4f}  R={rb['exact_span']['recall']:.4f})  TP={rb['exact_span']['tp']} FP={rb['exact_span']['fp']} FN={rb['exact_span']['fn']}")
        print(f"    Failures         : {r['failure_cases']['total']}")
    print("=" * 70)

    if output_path is None:
        output_path = os.path.join(checkpoints_dir, "dual_evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {output_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual evaluation: in-domain + out-of-frame.")
    parser.add_argument("--test-path", default="datasets/nlp/test.jsonl")
    parser.add_argument("--oof-path", default="datasets/nlp/out_of_frame_eval.jsonl")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dual_evaluation(
        test_path=args.test_path,
        oof_path=args.oof_path,
        checkpoints_dir=args.checkpoints_dir,
        output_path=args.output,
    )

