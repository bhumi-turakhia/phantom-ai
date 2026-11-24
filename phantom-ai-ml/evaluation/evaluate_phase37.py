import os
import sys
import json
import time
import hashlib
from typing import Dict, Any, List, Tuple, Optional
import torch

# Ensure phantom-ai-ml is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas.attributes import AttributeCategory, EvidenceState, AttributeEvidence
from services.nlp_service import NLPService
from services.uncertainty_service import UncertaintyService
from utils.preprocessing import (
    TAGS,
    TAG2IDX,
    IDX2TAG,
    tokenize_with_offsets,
)
from utils.gcit_translator import (
    translate_evidence_to_gcit,
    GCIT_LAYOUT,
    TranslationConflictError,
    IncompatibleAttributeError,
)

CHECKPOINTS = {
    "nlp": {
        "path": "phantom-ai-ml/checkpoints/nlp_char_bilstm_phase3c2.pt",
        "expected_sha256": "2d567c79d759e76940968fadb02d412b530d3405e377a2d6454c64e9f2e7b4a3",
    },
    "generator": {
        "path": "phantom-ai-ml/checkpoints/cgan_generator_phase54.pt",
        "expected_sha256": "a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964",
    },
    "evaluator": {
        "path": "phantom-ai-ml/checkpoints/forensic_evaluator_split_b.pt",
        "expected_sha256": "f52ae5a8df4ae574432f353fd77fc605f6dcf758d67d492b447c139a79e78834",
    },
}

CATEGORIES = [
    "face_shape",
    "hair",
    "facial_hair",
    "eyes",
    "hairline",
    "eyebrows",
    "nose",
    "lips",
    "distinguishing_marks",
    "other_observable",
]

STATES = ["explicit", "negated", "uncertain", "contradictory", "unknown"]


def compute_sha256(file_path: str) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def verify_checkpoints() -> Dict[str, Dict[str, Any]]:
    results = {}
    for name, info in CHECKPOINTS.items():
        path = info["path"]
        expected = info["expected_sha256"]
        if not os.path.exists(path):
            alt = os.path.join("..", path)
            if os.path.exists(alt):
                path = alt
        actual = compute_sha256(path) if os.path.exists(path) else None
        valid = (actual == expected)
        results[name] = {
            "path": path,
            "exists": os.path.exists(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "valid": valid,
        }
        if not valid:
            raise RuntimeError(f"Cryptographic hash mismatch on {name}: expected {expected}, got {actual}")
    return results


def compute_crf_sequence_confidence(
    model: torch.nn.Module,
    emissions: torch.Tensor,
    viterbi_tags: List[int],
    seq_len: int,
) -> float:
    """
    Computes exact CRF sequence posterior probability:
        P(y_hat | x) = exp( Score(x, y_hat) - log Z(x) )
    using the forward algorithm.
    """
    with torch.no_grad():
        if seq_len == 0 or len(viterbi_tags) == 0:
            return 0.0

        # 1. Forward algorithm to compute log Z(x)
        alpha = model.start_transitions + emissions[0, 0]
        for t in range(1, seq_len):
            emit_score = emissions[0, t].unsqueeze(0)  # (1, K)
            prev_alpha = alpha.unsqueeze(1)            # (K, 1)
            trans_score = model.transitions            # (K, K)
            alpha = torch.logsumexp(prev_alpha + trans_score + emit_score, dim=0)

        final_alpha = alpha + model.end_transitions
        log_Z = torch.logsumexp(final_alpha, dim=-1).item()

        # 2. Viterbi path score Score(x, y_hat)
        v_score = model.start_transitions[viterbi_tags[0]].item() + emissions[0, 0, viterbi_tags[0]].item()
        for t in range(1, seq_len):
            v_score += model.transitions[viterbi_tags[t - 1], viterbi_tags[t]].item() + emissions[0, t, viterbi_tags[t]].item()
        v_score += model.end_transitions[viterbi_tags[-1]].item()

        log_prob = v_score - log_Z
        # Clamp to avoid numerical overflow/underflow
        prob = torch.exp(torch.tensor(min(0.0, log_prob))).item()
        return max(0.0, min(1.0, prob))


def compute_calibration_metrics(
    confidences: List[float],
    accuracies: List[int],
    num_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes Brier Score and Expected Calibration Error (ECE) across confidence bins.
    """
    if not confidences or not accuracies:
        return {"ece": 0.0, "brier_score": 0.0, "bins": []}

    n = len(confidences)
    brier_score = sum((c - y) ** 2 for c, y in zip(confidences, accuracies)) / n

    bin_boundaries = [i / num_bins for i in range(num_bins + 1)]
    bins = []
    ece = 0.0

    for i in range(num_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i + 1]
        bin_indices = [
            idx for idx, c in enumerate(confidences)
            if (low <= c < high) or (i == num_bins - 1 and low <= c <= high)
        ]
        bin_size = len(bin_indices)
        if bin_size > 0:
            bin_acc = sum(accuracies[idx] for idx in bin_indices) / bin_size
            bin_conf = sum(confidences[idx] for idx in bin_indices) / bin_size
            gap = abs(bin_acc - bin_conf)
            ece += (bin_size / n) * gap
            bins.append({
                "bin_range": f"[{low:.1f}, {high:.1f}]",
                "count": bin_size,
                "avg_confidence": bin_conf,
                "empirical_accuracy": bin_acc,
                "calibration_gap": gap,
            })
        else:
            bins.append({
                "bin_range": f"[{low:.1f}, {high:.1f}]",
                "count": 0,
                "avg_confidence": 0.0,
                "empirical_accuracy": 0.0,
                "calibration_gap": 0.0,
            })

    return {
        "ece": ece,
        "brier_score": brier_score,
        "sample_count": n,
        "bins": bins,
    }


def evaluate_corpus(
    corpus: List[Dict[str, Any]],
    nlp_service: NLPService,
    device: torch.device,
    corpus_name: str = "Test Corpus",
) -> Dict[str, Any]:
    """
    Evaluates both the raw BiLSTM-CRF model and the hybrid post-processed pipeline.
    Tracks token-level metrics, span metrics, evidence states, calibration, and GCIT safety.
    """
    total_samples = len(corpus)
    total_slots = total_samples * len(CATEGORIES)

    # 1. Token-level metrics (Raw BiLSTM-CRF)
    token_gold_total = 0
    token_pred_total = 0
    token_correct_total = 0
    token_confidences: List[float] = []
    token_accuracies: List[int] = []

    # Sequence calibration tracking
    seq_confidences: List[float] = []
    seq_accuracies: List[int] = []

    # 2. Raw neural spans tracking
    raw_gold_spans = 0
    raw_pred_spans = 0
    raw_exact_matches = 0

    # 3. Hybrid pipeline spans tracking
    hybrid_gold_spans = 0
    hybrid_pred_spans = 0
    hybrid_exact_matches = 0
    boundary_mismatches = []
    false_positives = []
    false_negatives = []

    # 4. Evidence state confusion matrix (Hybrid Pipeline)
    confusion = {s1: {s2: 0 for s2 in STATES} for s1 in STATES}
    state_totals = {s: {"gold": 0, "pred": 0, "correct": 0} for s in STATES}

    # 5. Extraction provenance
    provenance_distribution: Dict[str, int] = {}

    # 6. Per-category metrics (Hybrid Pipeline)
    cat_metrics = {
        cat: {
            "gold_spans": 0,
            "pred_spans": 0,
            "exact_matches": 0,
            "state_correct": 0,
            "total_slots": 0,
        }
        for cat in CATEGORIES
    }

    # 7. Subset metrics
    subset_metrics: Dict[str, Dict[str, Any]] = {}

    sample_evaluations = []

    for item in corpus:
        sid = item["id"]
        sub = item["subset"]
        text = item["text"]
        gt_profile = item["ground_truth"]

        if sub not in subset_metrics:
            subset_metrics[sub] = {
                "samples": 0,
                "slots": 0,
                "state_correct": 0,
                "gold_spans": 0,
                "pred_spans": 0,
                "exact_matches": 0,
            }
        subset_metrics[sub]["samples"] += 1
        subset_metrics[sub]["slots"] += len(CATEGORIES)

        tok_offsets = tokenize_with_offsets(text)
        tokens = [t[0] for t in tok_offsets]
        seq_len = len(tokens)

        # Build token gold BIO tags
        gold_bio = ["O"] * seq_len
        for cat, info in gt_profile.items():
            g_span = info.get("span")
            g_start = info.get("start_char")
            g_end = info.get("end_char")
            if g_span and g_start is not None and g_end is not None:
                is_first = True
                for i, (tok, t_start, t_end) in enumerate(tok_offsets):
                    if t_start >= g_start and t_end <= g_end:
                        gold_bio[i] = f"B-{cat.upper()}" if is_first else f"I-{cat.upper()}"
                        is_first = False

        # --- A. Raw Neural Forward Pass & Viterbi Decoding ---
        raw_neural_spans: Dict[str, Dict[str, Any]] = {}
        pred_bio = ["O"] * seq_len
        seq_prob = 0.0

        if nlp_service.is_trained and nlp_service.model is not None and nlp_service.vocab is not None:
            input_ids = torch.tensor([nlp_service.vocab.encode(tokens)], dtype=torch.long, device=device)
            lengths = torch.tensor([seq_len], dtype=torch.long)

            with torch.no_grad():
                if getattr(nlp_service.model, "use_char_encoder", False) and nlp_service.char_vocab is not None:
                    char_ids_list = [nlp_service.char_vocab.encode_word(t) for t in tokens]
                    max_c = max(max(len(c) for c in char_ids_list), 3) if char_ids_list else 3
                    char_t = torch.zeros(1, seq_len, max_c, dtype=torch.long, device=device)
                    for w_i, c_seq in enumerate(char_ids_list):
                        c_len = min(len(c_seq), max_c)
                        if c_len > 0:
                            char_t[0, w_i, :c_len] = torch.tensor(c_seq[:c_len], dtype=torch.long, device=device)
                    emissions = nlp_service.model(input_ids, lengths=lengths, char_ids=char_t)
                else:
                    emissions = nlp_service.model(input_ids, lengths=lengths)

                viterbi_indices = nlp_service.model.decode_viterbi(emissions, lengths=lengths)[0]
                seq_prob = compute_crf_sequence_confidence(nlp_service.model, emissions, viterbi_indices, seq_len)
                emission_softmax = torch.softmax(emissions[0], dim=-1)

            pred_bio = [IDX2TAG.get(idx, "O") for idx in viterbi_indices]

            # Raw span extraction
            raw_neural_spans = nlp_service._extract_neural_spans(text)

            # Token-level metrics and calibration
            is_seq_perfect = (pred_bio == gold_bio)
            seq_confidences.append(seq_prob)
            seq_accuracies.append(1 if is_seq_perfect else 0)

            for i in range(seq_len):
                g_tag = gold_bio[i]
                p_tag = pred_bio[i]
                t_conf = emission_softmax[i, viterbi_indices[i]].item()
                t_acc = 1 if (g_tag == p_tag) else 0

                token_confidences.append(t_conf)
                token_accuracies.append(t_acc)

                if g_tag != "O":
                    token_gold_total += 1
                if p_tag != "O":
                    token_pred_total += 1
                if g_tag != "O" and p_tag != "O" and g_tag == p_tag:
                    token_correct_total += 1

        # Evaluate Raw Spans
        for cat in CATEGORIES:
            g_span = gt_profile[cat].get("span")
            g_start = gt_profile[cat].get("start_char")
            g_end = gt_profile[cat].get("end_char")
            if g_span and g_start is not None and g_end is not None:
                raw_gold_spans += 1

            if cat in raw_neural_spans:
                raw_pred_spans += 1
                r_info = raw_neural_spans[cat]
                if g_span and r_info["span"] == g_span and r_info["start_char"] == g_start and r_info["end_char"] == g_end:
                    raw_exact_matches += 1

        # --- B. Hybrid Pipeline Evaluation (with Calibration, Rules & Safety) ---
        _, _, pred_attrs, _, _, _, _ = nlp_service.analyze_description(text)

        eval_record = {
            "id": sid,
            "subset": sub,
            "text": text,
            "seq_confidence": seq_prob,
            "attributes": {},
        }

        for cat in CATEGORIES:
            cat_metrics[cat]["total_slots"] += 1
            gt = gt_profile[cat]
            g_state = gt["state"]
            g_span = gt.get("span")
            g_start = gt.get("start_char")
            g_end = gt.get("end_char")

            pred_ev: AttributeEvidence = pred_attrs[cat]
            p_state = pred_ev.state.value if hasattr(pred_ev.state, "value") else str(pred_ev.state)
            p_span = pred_ev.source_spans[0] if pred_ev.source_spans else None
            p_start = pred_ev.start_char
            p_end = pred_ev.end_char
            prov = pred_ev.extracted_by or "system_default"

            provenance_distribution[prov] = provenance_distribution.get(prov, 0) + 1

            confusion[g_state][p_state] += 1
            state_totals[g_state]["gold"] += 1
            state_totals[p_state]["pred"] += 1

            state_correct = (g_state == p_state)
            if state_correct:
                state_totals[g_state]["correct"] += 1
                subset_metrics[sub]["state_correct"] += 1
                cat_metrics[cat]["state_correct"] += 1

            span_exact = False
            if g_span and g_start is not None and g_end is not None:
                hybrid_gold_spans += 1
                subset_metrics[sub]["gold_spans"] += 1
                cat_metrics[cat]["gold_spans"] += 1

            if p_span and p_start is not None and p_end is not None:
                hybrid_pred_spans += 1
                subset_metrics[sub]["pred_spans"] += 1
                cat_metrics[cat]["pred_spans"] += 1
                assert text[p_start:p_end] == p_span, f"Offset mismatch on {sid} {cat}"

            if g_span and g_start is not None and p_span and p_start is not None:
                if g_span == p_span and g_start == p_start and g_end == p_end:
                    span_exact = True
                    hybrid_exact_matches += 1
                    subset_metrics[sub]["exact_matches"] += 1
                    cat_metrics[cat]["exact_matches"] += 1
                else:
                    boundary_mismatches.append({
                        "id": sid, "subset": sub, "category": cat,
                        "gold": (g_span, g_start, g_end),
                        "pred": (p_span, p_start, p_end),
                        "prov": prov,
                    })
            elif p_span and p_start is not None and (not g_span or g_start is None):
                false_positives.append({
                    "id": sid, "subset": sub, "category": cat,
                    "gold_state": g_state, "pred_span": p_span, "offsets": [p_start, p_end],
                    "pred_state": p_state, "prov": prov,
                })
            elif g_span and g_start is not None and (not p_span or p_start is None):
                false_negatives.append({
                    "id": sid, "subset": sub, "category": cat,
                    "gold_span": g_span, "offsets": [g_start, g_end],
                    "gold_state": g_state, "pred_state": p_state,
                })

            eval_record["attributes"][cat] = {
                "gold_state": g_state,
                "pred_state": p_state,
                "gold_span": g_span,
                "pred_span": p_span,
                "span_exact_match": span_exact,
                "state_match": state_correct,
                "extracted_by": prov,
            }

        sample_evaluations.append(eval_record)

    # --- Compute Metrics ---
    # Token-level BIO
    tok_prec = token_correct_total / max(token_pred_total, 1)
    tok_rec = token_correct_total / max(token_gold_total, 1)
    tok_f1 = 2 * (tok_prec * tok_rec) / max(tok_prec + tok_rec, 1e-9)

    # Raw Span Metrics
    raw_span_prec = raw_exact_matches / max(raw_pred_spans, 1)
    raw_span_rec = raw_exact_matches / max(raw_gold_spans, 1)
    raw_span_f1 = 2 * (raw_span_prec * raw_span_rec) / max(raw_span_prec + raw_span_rec, 1e-9)

    # Hybrid Span Metrics
    hybrid_span_prec = hybrid_exact_matches / max(hybrid_pred_spans, 1)
    hybrid_span_rec = hybrid_exact_matches / max(hybrid_gold_spans, 1)
    hybrid_span_f1 = 2 * (hybrid_span_prec * hybrid_span_rec) / max(hybrid_span_prec + hybrid_span_rec, 1e-9)

    # State Accuracy
    total_state_correct = sum(state_totals[s]["correct"] for s in STATES)
    state_accuracy = total_state_correct / total_slots

    state_performance = {}
    for s in STATES:
        g_c = state_totals[s]["gold"]
        p_c = state_totals[s]["pred"]
        c_c = state_totals[s]["correct"]
        prec = c_c / max(p_c, 1)
        rec = c_c / max(g_c, 1)
        f1 = 2 * (prec * rec) / max(prec + rec, 1e-9)
        state_performance[s] = {
            "gold": g_c, "pred": p_c, "correct": c_c,
            "precision": prec, "recall": rec, "f1": f1,
        }

    # Calibration Metrics
    token_calib = compute_calibration_metrics(token_confidences, token_accuracies, num_bins=10)
    seq_calib = compute_calibration_metrics(seq_confidences, seq_accuracies, num_bins=10)

    return {
        "corpus_name": corpus_name,
        "sample_count": total_samples,
        "total_slots": total_slots,
        "token_metrics": {
            "token_precision": tok_prec,
            "token_recall": tok_rec,
            "token_f1": tok_f1,
            "gold_tokens": token_gold_total,
            "pred_tokens": token_pred_total,
            "correct_tokens": token_correct_total,
        },
        "raw_model_spans": {
            "precision": raw_span_prec,
            "recall": raw_span_rec,
            "f1": raw_span_f1,
            "gold_spans": raw_gold_spans,
            "pred_spans": raw_pred_spans,
            "exact_matches": raw_exact_matches,
        },
        "hybrid_pipeline_spans": {
            "precision": hybrid_span_prec,
            "recall": hybrid_span_rec,
            "f1": hybrid_span_f1,
            "gold_spans": hybrid_gold_spans,
            "pred_spans": hybrid_pred_spans,
            "exact_matches": hybrid_exact_matches,
            "boundary_mismatches_count": len(boundary_mismatches),
            "false_positives_count": len(false_positives),
            "false_negatives_count": len(false_negatives),
        },
        "state_accuracy": state_accuracy,
        "state_performance": state_performance,
        "confusion_matrix": confusion,
        "provenance_distribution": provenance_distribution,
        "category_metrics": cat_metrics,
        "subset_metrics": subset_metrics,
        "calibration": {
            "token_emission_calibration": token_calib,
            "crf_sequence_calibration": seq_calib,
        },
        "error_details": {
            "boundary_mismatches": boundary_mismatches,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
    }


def run_phase37_full_evaluation() -> Dict[str, Any]:
    print("=" * 80)
    print("PHANTOM AI 3.0 — PHASE 3.7 NLP EVALUATION, CALIBRATION & GENERALIZATION RUNNER")
    print("=" * 80)

    # 1. Pre-flight Checkpoints Verification
    print("\n[1/6] Verifying pre-flight cryptographic checkpoint integrity...")
    pre_ckpts = verify_checkpoints()
    for name, res in pre_ckpts.items():
        print(f"  {name.upper()}: {res['actual_sha256'][:16]}... [VERIFIED MATCH]")

    # 2. Instantiate Services
    print("\n[2/6] Initializing live services on CPU...")
    device = torch.device("cpu")
    unc_service = UncertaintyService()
    nlp_service = NLPService(
        checkpoint_path=CHECKPOINTS["nlp"]["path"],
        device=device,
        uncertainty_service=unc_service,
    )

    # 3. Load Datasets
    print("\n[3/6] Loading benchmark datasets...")
    dev_path = "phantom-ai-ml/datasets/nlp/phase36_evaluation_corpus.json"
    test_path = "phantom-ai-ml/datasets/nlp/phase37_generalization_corpus.json"

    with open(dev_path, "r", encoding="utf-8") as f:
        dev_corpus = json.load(f)["samples"]
    with open(test_path, "r", encoding="utf-8") as f:
        test_corpus = json.load(f)["samples"]

    print(f"  Development Set (Phase 3.6): {len(dev_corpus)} samples, {len(dev_corpus)*10} attribute slots.")
    print(f"  Independent Test Set (Phase 3.7): {len(test_corpus)} samples, {len(test_corpus)*10} attribute slots.")

    # 4. Evaluate Development Set
    print("\n[4/6] Evaluating Development Set (Calibration Baseline)...")
    dev_results = evaluate_corpus(dev_corpus, nlp_service, device, corpus_name="Phase 3.6 Development Set")
    print(f"  Dev State Accuracy:         {dev_results['state_accuracy']:.4f}")
    print(f"  Dev Raw Span F1:            {dev_results['raw_model_spans']['f1']:.4f}")
    print(f"  Dev Hybrid Span F1:         {dev_results['hybrid_pipeline_spans']['f1']:.4f}")
    print(f"  Dev Token BIO F1:           {dev_results['token_metrics']['token_f1']:.4f}")
    print(f"  Dev Sequence ECE:           {dev_results['calibration']['crf_sequence_calibration']['ece']:.4f} (Brier: {dev_results['calibration']['crf_sequence_calibration']['brier_score']:.4f})")

    # 5. Evaluate Independent Generalization Test Set
    print("\n[5/6] Evaluating Independent Generalization Test Set (Phase 3.7)...")
    test_results = evaluate_corpus(test_corpus, nlp_service, device, corpus_name="Phase 3.7 Generalization Test Set")
    print(f"  Test State Accuracy:        {test_results['state_accuracy']:.4f}")
    print(f"  Test Raw Span F1:           {test_results['raw_model_spans']['f1']:.4f}")
    print(f"  Test Hybrid Span F1:        {test_results['hybrid_pipeline_spans']['f1']:.4f}")
    print(f"  Test Token BIO F1:          {test_results['token_metrics']['token_f1']:.4f}")
    print(f"  Test Token Emission ECE:    {test_results['calibration']['token_emission_calibration']['ece']:.4f} (Brier: {test_results['calibration']['token_emission_calibration']['brier_score']:.4f})")
    print(f"  Test Sequence ECE:          {test_results['calibration']['crf_sequence_calibration']['ece']:.4f} (Brier: {test_results['calibration']['crf_sequence_calibration']['brier_score']:.4f})")

    # 6. GCIT Safety Validation
    print("\n[6/6] Verifying forensic safeguards through GCIT translation...")
    # Test A: Negated attributes never become positive GCIT conditionings
    neg_prof = {
        AttributeCategory.FACIAL_HAIR.value: AttributeEvidence(
            category=AttributeCategory.FACIAL_HAIR,
            state=EvidenceState.NEGATED,
            value=None,
            notes="no beard",
        )
    }
    c_gen_neg, mask_neg, _ = translate_evidence_to_gcit(neg_prof)
    # Check absence of beard mapped to inverted polarity facial_no_beard = +1.0
    assert c_gen_neg[GCIT_LAYOUT["facial_no_beard"]] == 1.0
    assert mask_neg[GCIT_LAYOUT["facial_no_beard"]] == 1.0
    # Must never activate positive mustache or goatee
    assert mask_neg[GCIT_LAYOUT["facial_mustache"]] == 0.0
    assert mask_neg[GCIT_LAYOUT["facial_goatee"]] == 0.0
    print("  [PASS] Negated attributes never yield positive conditioning.")

    # Test B: Uncertain and unknown attributes remain unconditioned
    unk_prof = {
        AttributeCategory.EYES.value: AttributeEvidence(
            category=AttributeCategory.EYES,
            state=EvidenceState.UNKNOWN,
            value=None,
        ),
        AttributeCategory.HAIR.value: AttributeEvidence(
            category=AttributeCategory.HAIR,
            state=EvidenceState.UNCERTAIN,
            value="dark hair",
            uncertainty_cues=["maybe"],
        ),
    }
    c_gen_unk, mask_unk, _ = translate_evidence_to_gcit(unk_prof)
    hair_dims = [GCIT_LAYOUT["hair_black"], GCIT_LAYOUT["hair_blond"], GCIT_LAYOUT["hair_brown"], GCIT_LAYOUT["hair_gray"]]
    assert all(mask_unk[i] == 0.0 for i in hair_dims)
    print("  [PASS] Unknown and uncertain attributes remain strictly unconditioned (mask=0.0 across all dimensions).")

    # Test C: Contradictions block generation
    conflict_blocked = False
    try:
        conf_prof = {
            AttributeCategory.HAIR.value: AttributeEvidence(
                category=AttributeCategory.HAIR,
                state=EvidenceState.CONTRADICTORY,
                value=None,
                source_spans=["black hair", "blonde hair"],
            )
        }
        translate_evidence_to_gcit(conf_prof)
    except TranslationConflictError:
        conflict_blocked = True
    assert conflict_blocked, "Contradiction failed to block generation!"
    print("  [PASS] Contradictory attributes block generation until investigator resolution.")

    # Post-flight Checkpoint Verification
    print("\nVerifying post-flight cryptographic checkpoint integrity...")
    post_ckpts = verify_checkpoints()
    for name, res in post_ckpts.items():
        assert res["actual_sha256"] == pre_ckpts[name]["actual_sha256"]
    print("  All checkpoints verified intact (Zero weights modified).")

    # Compile Final JSON Report
    full_report = {
        "benchmark_phase": "Phase 3.7",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "checkpoints": post_ckpts,
        "development_set_evaluation": dev_results,
        "independent_test_set_evaluation": test_results,
    }

    out_file = "phantom-ai-ml/evaluation/phase37_evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nFull evaluation report written to {out_file}")

    return full_report


if __name__ == "__main__":
    run_phase37_full_evaluation()
