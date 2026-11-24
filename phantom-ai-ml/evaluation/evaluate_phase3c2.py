"""
Phase 3C.2 — Controlled 4-way evaluation script.
Compares:
  1. Phase 3B.2 Word-Only BiLSTM-CRF (nlp_bilstm.pt)
  2. Phase 3C.1 Char-Aware BiLSTM-CRF (nlp_char_bilstm_phase3c1.pt)
  3. Phase 3C.2 Expanded Char-Aware BiLSTM-CRF (nlp_char_bilstm_phase3c2.pt)
  4. Regex Baseline

Evaluation sets:
  A. Frozen in-domain test set (test.jsonl, 83 samples, SHA-256 verified)
  B. Frozen out-of-frame corpus (out_of_frame_eval.jsonl, 106 samples, SHA-256 verified)
  C. Controlled dev corpus (30 samples, from Phase 3B.3)

Reports: exact-span precision/recall/F1, token F1, per-category, TP/FP/FN, negation accuracy, macro F1.
"""

import os, sys, json, re, hashlib
from typing import List, Any, Tuple, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from utils.preprocessing import (
    TAGS, TAG2IDX, IDX2TAG,
    tokenize_with_offsets, generate_bio_tags_from_spans,
    Vocabulary, CharVocabulary,
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

FROZEN_TEST_SHA256 = "d6787cee05be7881b08e89889d9b41abf70e836885c06405e3623c94c8da4c51"
FROZEN_OOF_SHA256  = "4cdc270629a87d1607ce85675b0a9683c1220637e01c1447e3168f218075b6e3"

CHECKPOINTS_DIR = "checkpoints"
ML_ROOT = "phantom-ai-ml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen(test_path: str, oof_path: str):
    assert sha256(test_path) == FROZEN_TEST_SHA256, "Frozen test.jsonl tampered!"
    assert sha256(oof_path)  == FROZEN_OOF_SHA256,  "Frozen out_of_frame_eval.jsonl tampered!"
    print("Frozen evaluation datasets verified (SHA-256 match).")


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
            start_char, end_char = t0, t1
        elif tag.startswith("I-") and current_cat is not None and tag[2:].lower() == current_cat:
            end_char = t1
        else:
            if current_cat is not None:
                spans.append((sample_id, start_char, end_char, current_cat))
                current_cat = None
    if current_cat is not None:
        spans.append((sample_id, start_char, end_char, current_cat))
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
        # Skip unknown AND negated — negated is NOT a positive span
        if c_val not in ("unknown", "negated") and s_char is not None and e_char is not None:
            cat = (cat_obj.value if isinstance(cat_obj, AttributeCategory) else str(cat_obj).lower())
            spans.append((sid, s_char, e_char, cat))
    return spans


def run_regex_baseline(sample: Any) -> List[Tuple[str, int, int, str]]:
    text = sample.witness_text if hasattr(sample, "witness_text") else sample["witness_text"]
    sid  = sample.sample_id   if hasattr(sample, "sample_id")   else sample["sample_id"]
    spans = []
    for cat in AttributeCategory:
        for pat in OBSERVABLE_PATTERNS.get(cat, []):
            for m in re.finditer(pat, text, re.IGNORECASE):
                spans.append((sid, m.start(), m.end(), cat.value))
    return spans


def evaluate_model_on_split(
    samples: List[Any],
    model: nn.Module,
    word_vocab: Vocabulary,
    char_vocab: Optional[CharVocabulary],
    device,
    is_char_aware: bool,
) -> Tuple[dict, dict, dict, dict, list, list]:
    all_gold_tokens, all_pred_tokens = [], []
    all_gold_spans, all_pred_spans = [], []

    model.eval()
    with torch.no_grad():
        for s in samples:
            text = s.witness_text if hasattr(s, "witness_text") else s["witness_text"]
            sid  = s.sample_id   if hasattr(s, "sample_id")   else s["sample_id"]
            annotations = s.annotations if hasattr(s, "annotations") else s["annotations"]

            toks = tokenize_with_offsets(text)
            tokens = [t[0] for t in toks]
            _, gold_tags = generate_bio_tags_from_spans(text, annotations, tokens_with_offsets=toks)

            word_ids = torch.tensor(word_vocab.encode(tokens), dtype=torch.long).unsqueeze(0).to(device)
            lengths  = torch.tensor([len(tokens)], dtype=torch.long).to(device)

            if is_char_aware and char_vocab is not None:
                char_ids_raw = [char_vocab.encode_word(t) for t in tokens]
                max_char = max(max(len(c) for c in char_ids_raw), 3)
                char_tensor = torch.zeros(1, len(tokens), max_char, dtype=torch.long, device=device)
                for wi, cids in enumerate(char_ids_raw):
                    char_tensor[0, wi, :len(cids)] = torch.tensor(cids, dtype=torch.long)
                emissions = model(word_ids, lengths=lengths, char_ids=char_tensor)
            else:
                emissions = model(word_ids, lengths=lengths)

            pred_ids = model.decode_viterbi(emissions, lengths=lengths)[0]
            pred_tags = [IDX2TAG.get(i, "O") for i in pred_ids]

            all_gold_tokens.append(gold_tags)
            all_pred_tokens.append(pred_tags)

            gold_spans = extract_gold_spans(s)
            pred_spans = extract_spans_from_tags(toks, pred_tags, sid)
            all_gold_spans.extend(gold_spans)
            all_pred_spans.extend(pred_spans)

    tok_metrics  = compute_sequence_labeling_metrics(all_gold_tokens, all_pred_tokens, TAGS)
    span_metrics = compute_exact_span_metrics(all_gold_spans, all_pred_spans)
    per_cat      = compute_per_attribute_metrics(all_gold_spans, all_pred_spans, ALL_CATEGORIES)
    macro_f1     = sum(per_cat[c]["f1"] for c in per_cat if per_cat[c]["ground_truth_count"] + per_cat[c]["predicted_count"] > 0) / max(1, sum(1 for c in per_cat if per_cat[c]["ground_truth_count"] + per_cat[c]["predicted_count"] > 0))
    macro = {"f1": macro_f1}
    return tok_metrics, span_metrics, macro, per_cat, all_gold_spans, all_pred_spans


def evaluate_baseline_on_split(
    samples: List[Any]
) -> Tuple[dict, dict, dict, list, list]:
    gold_spans, pred_spans = [], []
    for s in samples:
        gold_spans.extend(extract_gold_spans(s))
        pred_spans.extend(run_regex_baseline(s))
    span_metrics = compute_exact_span_metrics(gold_spans, pred_spans)
    per_cat      = compute_per_attribute_metrics(gold_spans, pred_spans, ALL_CATEGORIES)
    macro_f1     = sum(per_cat[c]["f1"] for c in per_cat if per_cat[c]["ground_truth_count"] + per_cat[c]["predicted_count"] > 0) / max(1, sum(1 for c in per_cat if per_cat[c]["ground_truth_count"] + per_cat[c]["predicted_count"] > 0))
    return span_metrics, {"f1": macro_f1}, per_cat, gold_spans, pred_spans


def compute_negation_accuracy(samples: List[Any], model: nn.Module, word_vocab: Vocabulary,
                               char_vocab: Optional[CharVocabulary], device, is_char_aware: bool) -> dict:
    """
    For samples containing negated annotations, measure whether the model correctly
    produces NO span for the negated attribute category.
    """
    total_negated = 0
    correct_no_span = 0
    false_positives_on_negated = 0

    model.eval()
    with torch.no_grad():
        for s in samples:
            text = s.witness_text if hasattr(s, "witness_text") else s["witness_text"]
            annotations = s.annotations if hasattr(s, "annotations") else s["annotations"]

            negated_anns = [
                ann for ann in annotations
                if (ann.certainty if hasattr(ann, "certainty") else ann["certainty"]) in ("negated", EvidenceState.NEGATED.value if hasattr(EvidenceState, "NEGATED") else "negated")
                and (ann.start_char if hasattr(ann, "start_char") else ann["start_char"]) is not None
            ]
            if not negated_anns:
                continue

            toks = tokenize_with_offsets(text)
            tokens = [t[0] for t in toks]
            word_ids = torch.tensor(word_vocab.encode(tokens), dtype=torch.long).unsqueeze(0).to(device)
            lengths  = torch.tensor([len(tokens)], dtype=torch.long).to(device)

            if is_char_aware and char_vocab is not None:
                char_ids_raw = [char_vocab.encode_word(t) for t in tokens]
                max_char = max(max(len(c) for c in char_ids_raw), 3)
                char_tensor = torch.zeros(1, len(tokens), max_char, dtype=torch.long, device=device)
                for wi, cids in enumerate(char_ids_raw):
                    char_tensor[0, wi, :len(cids)] = torch.tensor(cids, dtype=torch.long)
                emissions = model(word_ids, lengths=lengths, char_ids=char_tensor)
            else:
                emissions = model(word_ids, lengths=lengths)

            pred_ids  = model.decode_viterbi(emissions, lengths=lengths)[0]
            pred_tags = [IDX2TAG.get(i, "O") for i in pred_ids]
            pred_spans = extract_spans_from_tags(toks, pred_tags)

            for ann in negated_anns:
                cat_obj = ann.category if hasattr(ann, "category") else ann["category"]
                cat = (cat_obj.value if isinstance(cat_obj, AttributeCategory) else str(cat_obj).lower())
                neg_start = ann.start_char if hasattr(ann, "start_char") else ann["start_char"]
                neg_end   = ann.end_char   if hasattr(ann, "end_char")   else ann["end_char"]

                total_negated += 1
                # Check if model predicted any overlapping span for this category
                false_positive = any(
                    ps_cat == cat and max(ps[1], neg_start) < min(ps[2], neg_end)
                    for ps in pred_spans
                    for ps_cat in [ps[3]]
                )
                if false_positive:
                    false_positives_on_negated += 1
                else:
                    correct_no_span += 1

    return {
        "total_negated_annotations": total_negated,
        "correct_no_span_predicted": correct_no_span,
        "false_positives_on_negated": false_positives_on_negated,
        "negation_accuracy": correct_no_span / max(1, total_negated),
    }


def run_4way_evaluation(checkpoints_dir: str = "checkpoints", output_path: str = None) -> dict:
    device = torch.device("cpu")  # Use CPU for reproducible evaluation

    test_path = os.path.join(ML_ROOT, "datasets", "nlp", "test.jsonl")
    oof_path  = os.path.join(ML_ROOT, "datasets", "nlp", "out_of_frame_eval.jsonl")
    verify_frozen(test_path, oof_path)

    # -------- Load Models --------
    # A: Phase 3B.2 Word-Only
    ckpt_3b2 = torch.load(os.path.join(checkpoints_dir, "nlp_bilstm.pt"), map_location=device)
    vocab_3b2_data = json.load(open(os.path.join(checkpoints_dir, "vocab.json")))
    word_vocab_3b2 = Vocabulary.from_dict(vocab_3b2_data)
    model_3b2 = BiLSTMAttributeLabeler(
        vocab_size=ckpt_3b2["vocab_size"], num_tags=ckpt_3b2["num_tags"],
        embedding_dim=ckpt_3b2["embedding_dim"], hidden_dim=ckpt_3b2["hidden_dim"],
        num_layers=ckpt_3b2["num_layers"], dropout=ckpt_3b2["dropout"], padding_idx=0,
    )
    model_3b2.load_state_dict(ckpt_3b2["model_state_dict"])
    model_3b2.eval()
    print(f"Loaded Phase 3B.2 word-only model  ({model_3b2.count_parameters():,} params)")

    # B: Phase 3C.1 Char-Aware (trained on original 385 samples)
    ckpt_3c1 = torch.load(os.path.join(checkpoints_dir, "nlp_char_bilstm_phase3c1.pt"), map_location=device)
    char_vocab_3c1 = CharVocabulary()
    char_vocab_3c1.load(os.path.join(checkpoints_dir, "char_vocab_phase3c1.json"))
    word_vocab_3c1 = Vocabulary.from_dict(json.load(open(os.path.join(checkpoints_dir, "vocab.json"))))
    model_3c1 = CharAwareBiLSTMAttributeLabeler(
        vocab_size=ckpt_3c1["vocab_size"], char_vocab_size=ckpt_3c1["char_vocab_size"],
        num_tags=ckpt_3c1["num_tags"], embedding_dim=ckpt_3c1["embedding_dim"],
        char_embedding_dim=ckpt_3c1.get("char_embedding_dim", 32),
        char_out_dim=ckpt_3c1.get("char_out_dim", 64),
        char_kernel_size=ckpt_3c1.get("char_kernel_size", 3),
        hidden_dim=ckpt_3c1["hidden_dim"], num_layers=ckpt_3c1["num_layers"],
        dropout=ckpt_3c1["dropout"], padding_idx=0,
    )
    model_3c1.load_state_dict(ckpt_3c1["model_state_dict"])
    model_3c1.eval()
    print(f"Loaded Phase 3C.1 char-aware model ({model_3c1.count_parameters():,} params)")

    # C: Phase 3C.2 Expanded Char-Aware (trained on 1785 samples)
    ckpt_3c2 = torch.load(os.path.join(checkpoints_dir, "nlp_char_bilstm_phase3c2.pt"), map_location=device)
    char_vocab_3c2 = CharVocabulary()
    char_vocab_3c2.load(os.path.join(checkpoints_dir, "nlp_char_bilstm_phase3c2_char_vocab.json"))
    word_vocab_3c2 = Vocabulary.from_dict(
        json.load(open(os.path.join(checkpoints_dir, "nlp_char_bilstm_phase3c2_word_vocab.json")))
    )
    model_3c2 = CharAwareBiLSTMAttributeLabeler(
        vocab_size=ckpt_3c2["vocab_size"], char_vocab_size=ckpt_3c2["char_vocab_size"],
        num_tags=ckpt_3c2["num_tags"], embedding_dim=ckpt_3c2["embedding_dim"],
        char_embedding_dim=ckpt_3c2.get("char_embedding_dim", 32),
        char_out_dim=ckpt_3c2.get("char_out_dim", 64),
        char_kernel_size=ckpt_3c2.get("char_kernel_size", 3),
        hidden_dim=ckpt_3c2["hidden_dim"], num_layers=ckpt_3c2["num_layers"],
        dropout=ckpt_3c2["dropout"], padding_idx=0,
    )
    model_3c2.load_state_dict(ckpt_3c2["model_state_dict"])
    model_3c2.eval()
    print(f"Loaded Phase 3C.2 expanded model   ({model_3c2.count_parameters():,} params)")

    # -------- Load Datasets --------
    loader = DatasetLoader()
    test_samples = loader.load_jsonl(test_path, validate=True)
    oof_samples  = loader.load_jsonl(oof_path, validate=False)
    dev_samples  = build_controlled_dev_corpus()

    splits = [
        ("In-Domain Test (Set A)", test_samples),
        ("Out-of-Frame Eval (Set B)", oof_samples),
        ("Controlled Dev Corpus", dev_samples),
    ]

    results = {
        "phase": "3C.2",
        "title": "Phase 3C.2 — 4-Way Controlled Evaluation",
        "checkpoint_hashes": {
            "nlp_bilstm.pt (Phase 3B.2)": sha256(os.path.join(checkpoints_dir, "nlp_bilstm.pt")),
            "nlp_char_bilstm_phase3c1.pt (Phase 3C.1)": sha256(os.path.join(checkpoints_dir, "nlp_char_bilstm_phase3c1.pt")),
            "nlp_char_bilstm_phase3c2.pt (Phase 3C.2)": sha256(os.path.join(checkpoints_dir, "nlp_char_bilstm_phase3c2.pt")),
        },
        "dataset_hashes": {
            "test.jsonl": sha256(test_path),
            "out_of_frame_eval.jsonl": sha256(oof_path),
        },
        "splits": {},
    }

    DIVIDER = "=" * 110
    HEADER  = f"{'Split':<32} {'Tok-F1':>8} {'Span-P':>8} {'Span-R':>8} {'Span-F1':>9} {'Macro-F1':>10} {'TP/FP/FN':>14}"
    print("\n" + DIVIDER)
    print(HEADER)
    print("-" * 110)

    for split_name, samples in splits:
        print(f"\n{split_name} ({len(samples)} samples)")

        # 3B.2
        tok_b2, span_b2, macro_b2, pcat_b2, g_b2, p_b2 = evaluate_model_on_split(
            samples, model_3b2, word_vocab_3b2, None, device, is_char_aware=False)
        neg_b2 = compute_negation_accuracy(samples, model_3b2, word_vocab_3b2, None, device, is_char_aware=False)

        # 3C.1
        tok_c1, span_c1, macro_c1, pcat_c1, g_c1, p_c1 = evaluate_model_on_split(
            samples, model_3c1, word_vocab_3c1, char_vocab_3c1, device, is_char_aware=True)
        neg_c1 = compute_negation_accuracy(samples, model_3c1, word_vocab_3c1, char_vocab_3c1, device, is_char_aware=True)

        # 3C.2
        tok_c2, span_c2, macro_c2, pcat_c2, g_c2, p_c2 = evaluate_model_on_split(
            samples, model_3c2, word_vocab_3c2, char_vocab_3c2, device, is_char_aware=True)
        neg_c2 = compute_negation_accuracy(samples, model_3c2, word_vocab_3c2, char_vocab_3c2, device, is_char_aware=True)

        # Baseline
        span_bl, macro_bl, pcat_bl, g_bl, p_bl = evaluate_baseline_on_split(samples)

        def row(name, tok_f1, sm, macro_m):
            tp, fp, fn = sm["tp"], sm["fp"], sm["fn"]
            counts = f"{tp}/{fp}/{fn}"
            tok_str = f"{tok_f1:.4f}" if tok_f1 is not None else "  —   "
            return f"  {name:<30} {tok_str:>8} {sm['precision']:>8.4f} {sm['recall']:>8.4f} {sm['f1']:>9.4f} {macro_m['f1']:>10.4f} {counts:>14}"

        print(row("Word-Only BiLSTM-CRF (3B.2)",   tok_b2["f1"], span_b2, macro_b2))
        print(row("Char-Aware BiLSTM-CRF (3C.1)",  tok_c1["f1"], span_c1, macro_c1))
        print(row("Expanded Char-Aware (3C.2)",     tok_c2["f1"], span_c2, macro_c2))
        print(row("Regex Baseline",                 None,         span_bl, macro_bl))
        print(f"  Negation Accuracy  | 3B.2: {neg_b2['negation_accuracy']:.1%} | 3C.1: {neg_c1['negation_accuracy']:.1%} | 3C.2: {neg_c2['negation_accuracy']:.1%}")
        print("-" * 110)

        results["splits"][split_name] = {
            "sample_count": len(samples),
            "gold_span_count": len(g_b2),
            "phase_3b2_word_only": {"token_f1": tok_b2, "exact_span": span_b2, "macro": macro_b2, "per_category": pcat_b2, "negation": neg_b2},
            "phase_3c1_char_aware": {"token_f1": tok_c1, "exact_span": span_c1, "macro": macro_c1, "per_category": pcat_c1, "negation": neg_c1},
            "phase_3c2_expanded": {"token_f1": tok_c2, "exact_span": span_c2, "macro": macro_c2, "per_category": pcat_c2, "negation": neg_c2},
            "regex_baseline": {"exact_span": span_bl, "macro": macro_bl, "per_category": pcat_bl},
        }

    if output_path is None:
        output_path = os.path.join(checkpoints_dir, "phase3c2_evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved 4-way evaluation JSON to: {output_path}")
    return results


if __name__ == "__main__":
    run_4way_evaluation()
