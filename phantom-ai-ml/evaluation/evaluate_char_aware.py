"""
evaluate_char_aware.py
PHANTOM AI 3.0 — Phase 3C.1 Controlled Evaluation

Rigorous three-way model comparison across three evaluation splits:
  Model A: Existing Phase 3B.2 Word-Only BiLSTM-CRF
  Model B: New Phase 3C.1 Character-Aware BiLSTM-CRF
  Model C: Deterministic Regex Pattern Matcher (Baseline)

Datasets evaluated:
  1. In-Domain Test Split (datasets/nlp/test.jsonl — 83 samples)
  2. Out-of-Frame Independent Corpus (datasets/nlp/out_of_frame_eval.jsonl — 106 samples)
  3. Controlled Development Corpus (30 samples, novel registers)

Calculates:
  - Exact-span Precision, Recall, and F1 (micro and macro)
  - Token-level Precision, Recall, and F1
  - Raw TP, FP, FN counts
  - Per-category exact-span metrics for all 10 attribute categories
  - OOV token rates per split
  - Parameter counts and runtime statistics

Identical span-matching and evaluation rules are enforced across all models.
"""

import os
import sys
import json
import re
import argparse
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
import torch
import torch.nn as nn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_PROJ_ROOT = os.path.abspath(os.path.join(_ML_ROOT, ".."))
sys.path.insert(0, _ML_ROOT)

from utils.device import resolve_device
from utils.preprocessing import (
    TAGS,
    TAG2IDX,
    IDX2TAG,
    tokenize_with_offsets,
    generate_bio_tags_from_spans,
    Vocabulary,
    CharVocabulary,
)
from models.nlp_model import BiLSTMAttributeLabeler, CharAwareBiLSTMAttributeLabeler
from data.dataset_loader import DatasetLoader
from schemas.attributes import AttributeCategory, EvidenceState
from schemas.dataset import DatasetSample
from services.nlp_service import OBSERVABLE_PATTERNS
from evaluation.evaluate_metrics import (
    compute_sequence_labeling_metrics,
    compute_exact_span_metrics,
    compute_per_attribute_metrics,
)
from scripts.phase3b3_audit import build_controlled_dev_corpus

ALL_CATEGORIES = [c.value for c in AttributeCategory]


def extract_spans_from_tags(
    tokens_with_offsets: List[Tuple[str, int, int]],
    tags: List[str],
    sample_id: str = "",
) -> List[Tuple[str, int, int, str]]:
    spans = []
    current_cat: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    for (tok, t0, t1), tag in zip(tokens_with_offsets, tags):
        if tag.startswith("B-"):
            if current_cat is not None:
                spans.append((sample_id, start_char, end_char, current_cat))
            current_cat = tag[2:].lower()
            start_char = t0
            end_char = t1
        elif tag.startswith("I-") and current_cat is not None and tag[2:].lower() == current_cat:
            end_char = t1
        else:
            if current_cat is not None:
                spans.append((sample_id, start_char, end_char, current_cat))
                current_cat = None
    if current_cat is not None:
        spans.append((sample_id, start_char, end_char, current_cat))
    return spans


def run_regex_baseline(sample: Any) -> List[Tuple[str, int, int, str]]:
    text = sample.witness_text if hasattr(sample, "witness_text") else sample["witness_text"]
    sid = sample.sample_id if hasattr(sample, "sample_id") else sample["sample_id"]
    spans = []
    for cat in AttributeCategory:
        patterns = OBSERVABLE_PATTERNS.get(cat, [])
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                spans.append((sid, m.start(), m.end(), cat.value))
    return spans


def extract_gold_spans(sample: Any) -> List[Tuple[str, int, int, str]]:
    spans = []
    annotations = sample.annotations if hasattr(sample, "annotations") else sample["annotations"]
    sid = sample.sample_id if hasattr(sample, "sample_id") else sample["sample_id"]
    for ann in annotations:
        c_val = ann.certainty if hasattr(ann, "certainty") else ann["certainty"]
        s_char = ann.start_char if hasattr(ann, "start_char") else ann["start_char"]
        e_char = ann.end_char if hasattr(ann, "end_char") else ann["end_char"]
        cat_obj = ann.category if hasattr(ann, "category") else ann["category"]
        if c_val not in (EvidenceState.UNKNOWN.value, "unknown", getattr(EvidenceState, "NEGATED", "negated"), "negated") and s_char is not None and e_char is not None:
            cat = (cat_obj.value if isinstance(cat_obj, AttributeCategory) else str(cat_obj).lower())
            spans.append((sid, s_char, e_char, cat))
    return spans


def compute_oov_rate(samples: List[Any], word_vocab: Vocabulary) -> float:
    vocab_words = set(word_vocab.word2idx.keys())
    total_tokens = 0
    oov_tokens = 0
    for s in samples:
        text = s.witness_text if hasattr(s, "witness_text") else s["witness_text"]
        tokens = [t[0] for t in tokenize_with_offsets(text)]
        for t in tokens:
            total_tokens += 1
            if t not in vocab_words:
                oov_tokens += 1
    return round(oov_tokens / max(total_tokens, 1), 4)


def evaluate_model_on_split(
    samples: List[Any],
    model: nn.Module,
    word_vocab: Vocabulary,
    char_vocab: Optional[CharVocabulary],
    device: torch.device,
    is_char_aware: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Tuple], List[Tuple]]:
    model.eval()
    all_true_tags: List[List[str]] = []
    all_pred_tags: List[List[str]] = []
    gold_spans: List[Tuple] = []
    pred_spans: List[Tuple] = []

    with torch.no_grad():
        for sample in samples:
            text = sample.witness_text if hasattr(sample, "witness_text") else sample["witness_text"]
            sid = sample.sample_id if hasattr(sample, "sample_id") else sample["sample_id"]
            annotations = sample.annotations if hasattr(sample, "annotations") else sample["annotations"]

            tok_offsets = tokenize_with_offsets(text)
            tokens, true_tags = generate_bio_tags_from_spans(
                text, annotations, tokens_with_offsets=tok_offsets
            )
            all_true_tags.append(true_tags)
            gold_spans.extend(extract_gold_spans(sample))

            if not tokens:
                all_pred_tags.append([])
                continue

            input_ids = torch.tensor([word_vocab.encode(tokens)], dtype=torch.long, device=device)
            lengths = torch.tensor([len(tokens)], dtype=torch.long)

            if is_char_aware and char_vocab is not None:
                char_ids_list = [char_vocab.encode_word(t) for t in tokens]
                max_c = max(max(len(c) for c in char_ids_list), 3)
                char_t = torch.zeros(1, len(tokens), max_c, dtype=torch.long, device=device)
                for w_i, c_seq in enumerate(char_ids_list):
                    c_len = min(len(c_seq), max_c)
                    if c_len > 0:
                        char_t[0, w_i, :c_len] = torch.tensor(c_seq[:c_len], dtype=torch.long, device=device)
                emissions = model(input_ids, lengths=lengths, char_ids=char_t)
            else:
                emissions = model(input_ids, lengths=lengths)

            decoded = model.decode_viterbi(emissions, lengths=lengths)[0]
            pred_tags = [IDX2TAG[p] for p in decoded]
            all_pred_tags.append(pred_tags)

            spans = extract_spans_from_tags(tok_offsets, pred_tags, sid)
            pred_spans.extend(spans)

    token_metrics = compute_sequence_labeling_metrics(all_true_tags, all_pred_tags, ignore_tag="O")
    span_metrics = compute_exact_span_metrics(gold_spans, pred_spans)
    per_cat = compute_per_attribute_metrics(gold_spans, pred_spans, ALL_CATEGORIES)

    # Macro span metrics
    macro_p = sum(v["precision"] for v in per_cat.values()) / max(len(per_cat), 1)
    macro_r = sum(v["recall"] for v in per_cat.values()) / max(len(per_cat), 1)
    macro_f = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0
    macro_metrics = {
        "precision": round(macro_p, 4),
        "recall": round(macro_r, 4),
        "f1": round(macro_f, 4),
    }

    return token_metrics, span_metrics, macro_metrics, per_cat, gold_spans, pred_spans


def evaluate_baseline_on_split(
    samples: List[Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Tuple], List[Tuple]]:
    gold_spans: List[Tuple] = []
    pred_spans: List[Tuple] = []

    for s in samples:
        gold_spans.extend(extract_gold_spans(s))
        pred_spans.extend(run_regex_baseline(s))

    span_metrics = compute_exact_span_metrics(gold_spans, pred_spans)
    per_cat = compute_per_attribute_metrics(gold_spans, pred_spans, ALL_CATEGORIES)

    macro_p = sum(v["precision"] for v in per_cat.values()) / max(len(per_cat), 1)
    macro_r = sum(v["recall"] for v in per_cat.values()) / max(len(per_cat), 1)
    macro_f = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0
    macro_metrics = {
        "precision": round(macro_p, 4),
        "recall": round(macro_r, 4),
        "f1": round(macro_f, 4),
    }

    return span_metrics, macro_metrics, per_cat, gold_spans, pred_spans


def run_full_controlled_evaluation(
    checkpoints_dir: str = "checkpoints",
    output_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    device = resolve_device()
    print("=" * 80)
    print("PHANTOM AI 3.0 — PHASE 3C.1 CONTROLLED EVALUATION (3-WAY COMPARISON)")
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Load Vocabularies
    vocab_file = os.path.join(checkpoints_dir, "vocab.json")
    char_vocab_file = os.path.join(checkpoints_dir, "char_vocab.json")

    word_vocab = Vocabulary.load(vocab_file)
    char_vocab = CharVocabulary.load(char_vocab_file)
    print(f"Loaded Word Vocabulary: {len(word_vocab):,} tokens")
    print(f"Loaded Character Vocabulary: {len(char_vocab):,} characters")

    # 2. Load Models
    # Model A: Word-Only (Phase 3B.2)
    bilstm_ckpt_path = os.path.join(checkpoints_dir, "nlp_bilstm.pt")
    bilstm_ckpt = torch.load(bilstm_ckpt_path, map_location=device)
    model_word = BiLSTMAttributeLabeler(
        vocab_size=bilstm_ckpt["vocab_size"],
        num_tags=bilstm_ckpt["num_tags"],
        embedding_dim=bilstm_ckpt["embedding_dim"],
        hidden_dim=bilstm_ckpt["hidden_dim"],
        num_layers=bilstm_ckpt["num_layers"],
        dropout=bilstm_ckpt["dropout"],
        padding_idx=0,
    )
    model_word.load_state_dict(bilstm_ckpt["model_state_dict"])
    model_word.to(device)
    model_word.eval()

    # Model B: Character-Aware (Phase 3C.1)
    char_ckpt_path = os.path.join(checkpoints_dir, "nlp_char_bilstm.pt")
    char_ckpt = torch.load(char_ckpt_path, map_location=device)
    model_char = CharAwareBiLSTMAttributeLabeler(
        vocab_size=char_ckpt["vocab_size"],
        char_vocab_size=char_ckpt["char_vocab_size"],
        num_tags=char_ckpt["num_tags"],
        embedding_dim=char_ckpt["embedding_dim"],
        char_embedding_dim=char_ckpt.get("char_embedding_dim", 32),
        char_out_dim=char_ckpt.get("char_out_dim", 64),
        char_kernel_size=char_ckpt.get("char_kernel_size", 3),
        hidden_dim=char_ckpt["hidden_dim"],
        num_layers=char_ckpt["num_layers"],
        dropout=char_ckpt["dropout"],
        padding_idx=0,
    )
    model_char.load_state_dict(char_ckpt["model_state_dict"])
    model_char.to(device)
    model_char.eval()

    word_params = model_word.count_parameters()
    char_params = model_char.count_parameters()
    print(f"Model A (Word-Only) Parameters:      {word_params:,}")
    print(f"Model B (Character-Aware) Parameters: {char_params:,} (+{char_params - word_params:,} char encoder weights)")

    # 3. Load Datasets
    loader = DatasetLoader()
    test_path = os.path.join(_ML_ROOT, "datasets", "nlp", "test.jsonl")
    oof_path = os.path.join(_ML_ROOT, "datasets", "nlp", "out_of_frame_eval.jsonl")

    in_domain_samples = loader.load_jsonl(test_path, validate=True)
    oof_samples = loader.load_jsonl(oof_path, validate=False)
    dev_samples = build_controlled_dev_corpus()

    dataset_splits = [
        ("In-Domain Test (Set A)", in_domain_samples, 83),
        ("Out-of-Frame Eval (Set B)", oof_samples, 106),
        ("Controlled Dev Corpus", dev_samples, 30),
    ]

    report = {
        "phase": "3C.1",
        "title": "Character-Aware BiLSTM-CRF Controlled Evaluation",
        "models": {
            "word_only_bilstm_crf": {
                "checkpoint": bilstm_ckpt_path,
                "phase": bilstm_ckpt.get("phase", "3B.2"),
                "parameters": word_params,
                "training_objective": bilstm_ckpt.get("training_objective"),
            },
            "char_aware_bilstm_crf": {
                "checkpoint": char_ckpt_path,
                "phase": char_ckpt.get("phase", "3C.1"),
                "parameters": char_params,
                "training_objective": char_ckpt.get("training_objective"),
            },
            "regex_baseline": {
                "type": "Deterministic Pattern Matcher",
            },
        },
        "splits_evaluated": {},
    }

    print("\n" + "=" * 80)
    print(f"{'Split / Model':<32} {'OOV%':>6} {'Tok F1':>8} {'Span P':>8} {'Span R':>8} {'Span F1':>8} {'Macro F1':>9} {'TP/FP/FN':>14}")
    print("-" * 98)

    for split_name, samples, expected_count in dataset_splits:
        oov_rate = compute_oov_rate(samples, word_vocab)

        # 1. Word-only model
        tok_w, span_w, macro_w, pcat_w, gold_w, pred_w = evaluate_model_on_split(
            samples, model_word, word_vocab, None, device, is_char_aware=False
        )

        # 2. Character-aware model
        tok_c, span_c, macro_c, pcat_c, gold_c, pred_c = evaluate_model_on_split(
            samples, model_char, word_vocab, char_vocab, device, is_char_aware=True
        )

        # 3. Baseline
        span_b, macro_b, pcat_b, gold_b, pred_b = evaluate_baseline_on_split(samples)

        counts_w = f"{span_w['tp']}/{span_w['fp']}/{span_w['fn']}"
        counts_c = f"{span_c['tp']}/{span_c['fp']}/{span_c['fn']}"
        counts_b = f"{span_b['tp']}/{span_b['fp']}/{span_b['fn']}"

        print(f"{split_name:<32} {oov_rate*100:>5.1f}%")
        print(f"  {'Word-Only BiLSTM-CRF':<30} {'':>6} {tok_w['f1']:>8.4f} {span_w['precision']:>8.4f} {span_w['recall']:>8.4f} {span_w['f1']:>8.4f} {macro_w['f1']:>9.4f} {counts_w:>14}")
        print(f"  {'Char-Aware BiLSTM-CRF':<30} {'':>6} {tok_c['f1']:>8.4f} {span_c['precision']:>8.4f} {span_c['recall']:>8.4f} {span_c['f1']:>8.4f} {macro_c['f1']:>9.4f} {counts_c:>14}")
        print(f"  {'Regex Baseline':<30} {'':>6} {'—':>8} {span_b['precision']:>8.4f} {span_b['recall']:>8.4f} {span_b['f1']:>8.4f} {macro_b['f1']:>9.4f} {counts_b:>14}")
        print("-" * 98)

        report["splits_evaluated"][split_name] = {
            "sample_count": len(samples),
            "gold_spans_count": len(gold_w),
            "oov_rate": oov_rate,
            "word_only_bilstm_crf": {
                "token_level": tok_w,
                "exact_span_micro": span_w,
                "exact_span_macro": macro_w,
                "per_category": pcat_w,
            },
            "char_aware_bilstm_crf": {
                "token_level": tok_c,
                "exact_span_micro": span_c,
                "exact_span_macro": macro_c,
                "per_category": pcat_c,
            },
            "regex_baseline": {
                "exact_span_micro": span_b,
                "exact_span_macro": macro_b,
                "per_category": pcat_b,
            },
        }

    if output_report_path is None:
        output_report_path = os.path.join(checkpoints_dir, "phase3c1_controlled_evaluation.json")

    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved full controlled evaluation JSON to: {output_report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Character-Aware BiLSTM-CRF.")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_full_controlled_evaluation(
        checkpoints_dir=args.checkpoints_dir,
        output_report_path=args.output,
    )
