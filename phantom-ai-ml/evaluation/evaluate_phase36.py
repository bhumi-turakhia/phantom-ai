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
from services.composite_generation_service import CompositeGenerationService

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
            # Try alternate path
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


def run_phase36_evaluation() -> Dict[str, Any]:
    print("=" * 80)
    print("PHANTOM AI 3.0 — PHASE 3.6 BENCHMARK EVALUATION RUNNER")
    print("Span-Boundary Calibration, Compound Descriptors & Category Protection")
    print("=" * 80)

    # 1. Cryptographic Checkpoint Verification
    print("\n[1/5] Verifying frozen checkpoint integrity...")
    ckpt_audit = verify_checkpoints()
    for name, res in ckpt_audit.items():
        print(f"  {name.upper()}: {res['actual_sha256'][:16]}... [VERIFIED MATCH]")

    # 2. Instantiate Services
    print("\n[2/5] Initializing NLP and Uncertainty Services...")
    device = torch.device("cpu")
    unc_service = UncertaintyService()
    nlp_service = NLPService(
        checkpoint_path=CHECKPOINTS["nlp"]["path"],
        device=device,
        uncertainty_service=unc_service,
    )
    if not nlp_service.is_trained:
        raise RuntimeError("NLPService failed to load trained Phase 3C.2 checkpoint!")

    # 3. Load Phase 3.6 Benchmark Corpus
    corpus_path = "phantom-ai-ml/datasets/nlp/phase36_evaluation_corpus.json"
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join("..", corpus_path)
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    samples = corpus_data["samples"]
    print(f"  Loaded {len(samples)} evaluation samples across {len(corpus_data['categories'])} categories.")

    # Tracking Metrics
    total_samples = len(samples)
    total_slots = total_samples * len(CATEGORIES)  # 600

    # Span metrics
    gold_spans_total = 0
    pred_spans_total = 0
    exact_case_matches = 0
    exact_lower_matches = 0

    # Evidence state metrics
    confusion_matrix = {s1: {s2: 0 for s2 in STATES} for s1 in STATES}
    state_totals = {s: {"gold": 0, "pred": 0, "correct": 0} for s in STATES}

    # Provenance tracking
    provenance_counts: Dict[str, int] = {}

    # Subset metrics
    subset_metrics: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        sub = s["subset"]
        if sub not in subset_metrics:
            subset_metrics[sub] = {
                "samples": 0,
                "slots": 0,
                "state_correct": 0,
                "gold_spans": 0,
                "pred_spans": 0,
                "exact_span_matches": 0,
            }
        subset_metrics[sub]["samples"] += 1

    # Category metrics
    category_metrics: Dict[str, Dict[str, Any]] = {
        cat: {
            "gold_spans": 0,
            "pred_spans": 0,
            "exact_span_matches": 0,
            "state_correct": 0,
            "total_slots": 0,
        }
        for cat in CATEGORIES
    }

    # Trace execution
    print("\n[3/5] Evaluating all 60 benchmark profiles (600 attribute slots)...")
    t0 = time.time()

    sample_evaluations = []

    for item in samples:
        s_id = item["id"]
        subset = item["subset"]
        text = item["text"]
        gt_profile = item["ground_truth"]

        status, is_trained, pred_attrs, contradictions, uncertainties, conf, notes = nlp_service.analyze_description(text)

        eval_record = {
            "id": s_id,
            "subset": subset,
            "text": text,
            "attributes": {},
        }

        subset_metrics[subset]["slots"] += len(CATEGORIES)

        for cat in CATEGORIES:
            category_metrics[cat]["total_slots"] += 1
            gt = gt_profile[cat]
            gt_state = gt["state"]
            gt_span = gt.get("span")
            gt_start = gt.get("start_char")
            gt_end = gt.get("end_char")

            pred_ev: AttributeEvidence = pred_attrs.get(cat)
            pred_state = pred_ev.state.value if hasattr(pred_ev.state, "value") else str(pred_ev.state)
            pred_span = pred_ev.source_spans[0] if pred_ev.source_spans else None
            pred_start = pred_ev.start_char
            pred_end = pred_ev.end_char
            extracted_by = pred_ev.extracted_by or "unknown"

            provenance_counts[extracted_by] = provenance_counts.get(extracted_by, 0) + 1

            # Update State Confusion Matrix
            confusion_matrix[gt_state][pred_state] += 1
            state_totals[gt_state]["gold"] += 1
            state_totals[pred_state]["pred"] += 1

            state_match = (gt_state == pred_state)
            if state_match:
                state_totals[gt_state]["correct"] += 1
                subset_metrics[subset]["state_correct"] += 1
                category_metrics[cat]["state_correct"] += 1

            # Span Evaluation
            span_exact = False
            span_ci = False

            if gt_span and gt_start is not None and gt_end is not None:
                gold_spans_total += 1
                subset_metrics[subset]["gold_spans"] += 1
                category_metrics[cat]["gold_spans"] += 1

            if pred_span and pred_start is not None and pred_end is not None:
                pred_spans_total += 1
                subset_metrics[subset]["pred_spans"] += 1
                category_metrics[cat]["pred_spans"] += 1

                # Verbatim check
                assert text[pred_start:pred_end] == pred_span, (
                    f"Verbatim offset mismatch on {s_id} {cat}: text[{pred_start}:{pred_end}]='{text[pred_start:pred_end]}' != '{pred_span}'"
                )

            if gt_span and pred_span:
                if gt_span == pred_span and gt_start == pred_start and gt_end == pred_end:
                    span_exact = True
                    exact_case_matches += 1
                    subset_metrics[subset]["exact_span_matches"] += 1
                    category_metrics[cat]["exact_span_matches"] += 1

                if gt_span.lower() == pred_span.lower():
                    span_ci = True
                    exact_lower_matches += 1

            eval_record["attributes"][cat] = {
                "gold_state": gt_state,
                "pred_state": pred_state,
                "state_match": state_match,
                "gold_span": gt_span,
                "pred_span": pred_span,
                "span_exact_match": span_exact,
                "extracted_by": extracted_by,
                "offsets": [pred_start, pred_end],
            }

        sample_evaluations.append(eval_record)

    eval_duration = time.time() - t0
    print(f"  Evaluation finished in {eval_duration:.2f}s ({eval_duration/total_samples*1000:.1f}ms per sample).")

    # Metrics Compilation
    strict_span_precision = exact_case_matches / max(pred_spans_total, 1)
    strict_span_recall = exact_case_matches / max(gold_spans_total, 1)
    strict_span_f1 = (
        2 * (strict_span_precision * strict_span_recall) / max(strict_span_precision + strict_span_recall, 1e-9)
    )

    ci_span_precision = exact_lower_matches / max(pred_spans_total, 1)
    ci_span_recall = exact_lower_matches / max(gold_spans_total, 1)
    ci_span_f1 = (
        2 * (ci_span_precision * ci_span_recall) / max(ci_span_precision + ci_span_recall, 1e-9)
    )

    state_accuracy = sum(state_totals[s]["correct"] for s in STATES) / total_slots

    state_metrics = {}
    for s in STATES:
        gold_c = state_totals[s]["gold"]
        pred_c = state_totals[s]["pred"]
        corr_c = state_totals[s]["correct"]
        prec = corr_c / max(pred_c, 1)
        rec = corr_c / max(gold_c, 1)
        f1 = 2 * (prec * rec) / max(prec + rec, 1e-9)
        state_metrics[s] = {
            "gold": gold_c,
            "pred": pred_c,
            "correct": corr_c,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

    # 4. Verification of 5 Unseen Failure Cases & 6 Documented Mismatches
    print("\n[4/5] Executing focused regression tests on documented failure cases...")
    unseen_tests = [
        ("Case 1", "He lacked any visible facial hair.", "facial_hair", "negated", "lacked any visible facial hair"),
        ("Case 2", "I cannot recall whether he wore glasses.", "other_observable", "unknown", "cannot recall whether he wore glasses"),
        ("Case 3", "His eyes were clearly blue, although I am unsure about his hair.", "eyes", "explicit", "clearly blue"),
        ("Case 4", "There were no scars, tattoos, or piercings.", "distinguishing_marks", "negated", "no scars, tattoos, or piercings"),
        ("Case 5", "He was not bald, but his hair was very short.", "hair", "explicit", "very short"),
    ]
    unseen_results = []
    for label, utext, cat, exp_state, exp_span in unseen_tests:
        _, _, uattrs, _, _, _, _ = nlp_service.analyze_description(utext)
        ev = uattrs[cat]
        act_state = ev.state.value if hasattr(ev.state, "value") else str(ev.state)
        act_span = ev.source_spans[0] if ev.source_spans else None
        s_pass = (act_state == exp_state and act_span == exp_span)
        unseen_results.append({
            "case": label,
            "text": utext,
            "target": cat,
            "expected_state": exp_state,
            "actual_state": act_state,
            "expected_span": exp_span,
            "actual_span": act_span,
            "passed": s_pass,
        })
        print(f"  {label}: {'PASS' if s_pass else 'FAIL'} (state={act_state}, span='{act_span}')")
        assert s_pass, f"Unseen case {label} failed!"

    mismatch_tests = [
        ("NEG-09", "He had no visible tattoos or piercings on his face.", "distinguishing_marks", "negated", "no visible tattoos or piercings"),
        ("NEG-15", "He had definitely no beard and wore a dark baseball cap.", "facial_hair", "negated", "definitely no beard"),
        ("UNC-02", "He appears to have narrow eyes.", "eyes", "uncertain", "narrow eyes"),
        ("UNC-09", "He was sort of bald on top.", "hair", "uncertain", "bald on top"),
        ("UNC-15", "He definitely had blue eyes, but possibly had a trimmed beard.", "eyes", "explicit", "blue eyes"),
        ("VIG-10", "The suspect had blond hair and blue eyes, but the second witness noted black hair.", "eyes", "explicit", "blue eyes"),
    ]
    mismatch_results = []
    for mid, mtext, cat, exp_state, exp_span in mismatch_tests:
        _, _, mattrs, _, _, _, _ = nlp_service.analyze_description(mtext)
        ev = mattrs[cat]
        act_state = ev.state.value if hasattr(ev.state, "value") else str(ev.state)
        act_span = ev.source_spans[0] if ev.source_spans else None
        m_pass = (act_state == exp_state and act_span == exp_span)
        mismatch_results.append({
            "id": mid,
            "text": mtext,
            "target": cat,
            "expected_state": exp_state,
            "actual_state": act_state,
            "expected_span": exp_span,
            "actual_span": act_span,
            "passed": m_pass,
        })
        print(f"  {mid}: {'PASS' if m_pass else 'FAIL'} (state={act_state}, span='{act_span}')")
        assert m_pass, f"Documented mismatch {mid} failed!"

    # 5. GCIT Generator Consistency Audit
    print("\n[5/5] Validating GCIT translation and conditioning isolation...")
    from utils.gcit_translator import (
        translate_evidence_to_gcit,
        GCIT_LAYOUT,
        TranslationConflictError,
        IncompatibleAttributeError,
    )

    # Test 1: Inverted polarity for negated facial hair
    neg_profile = {
        AttributeCategory.FACIAL_HAIR.value: AttributeEvidence(
            category=AttributeCategory.FACIAL_HAIR,
            state=EvidenceState.NEGATED,
            value=None,
            notes="Witness stated suspect had no beard.",
        )
    }
    c_gen1, mask1, warnings1 = translate_evidence_to_gcit(neg_profile)
    idx_no_beard = GCIT_LAYOUT["facial_no_beard"]
    assert c_gen1[idx_no_beard] == 1.0, f"Expected facial_no_beard to be 1.0, got {c_gen1[idx_no_beard]}"
    assert mask1[idx_no_beard] == 1.0, f"Expected mask to be 1.0, got {mask1[idx_no_beard]}"
    # Unconditioned isolation: unmentioned dimensions have mask 0.0 and value 0.0
    isolated_dims = sum(1 for m in mask1 if m == 0.0)
    print(f"  Negation polarity inversion: facial_no_beard=+1.0 (mask=1.0). {isolated_dims}/{len(mask1)} dims isolated.")

    # Test 2: Contradiction safety blocking
    conflict_detected = False
    try:
        conflict_profile = {
            AttributeCategory.HAIR.value: AttributeEvidence(
                category=AttributeCategory.HAIR,
                state=EvidenceState.CONTRADICTORY,
                value=None,
                source_spans=["blond hair", "black hair"],
                notes="Contradictory testimony detected.",
            )
        }
        c_gen2, mask2, warnings2 = translate_evidence_to_gcit(conflict_profile)
        # If not raising error, all hair dimensions must be strictly unconditioned (mask=0.0)
        hair_dims = [GCIT_LAYOUT["hair_black"], GCIT_LAYOUT["hair_blond"], GCIT_LAYOUT["hair_brown"]]
        assert all(mask2[idx] == 0.0 for idx in hair_dims), "Contradictory hair produced active conditioning!"
        print("  Contradiction safety: Contradictory attribute unconditioned (mask=0.0 across conflicting dims).")
    except (TranslationConflictError, IncompatibleAttributeError, ValueError) as e:
        print(f"  Contradiction safety: Translation properly blocked ({e})")

    # Output Summary Table
    print("\n" + "=" * 80)
    print("PHASE 3.6 BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Benchmark Samples:      {total_samples}")
    print(f"Total Attribute Slots:        {total_slots}")
    print(f"Overall Evidence Accuracy:    {state_accuracy:.4f} ({sum(state_totals[s]['correct'] for s in STATES)}/{total_slots})")
    print(f"Strict Span Precision:        {strict_span_precision:.4f} ({exact_case_matches}/{pred_spans_total})")
    print(f"Strict Span Recall:           {strict_span_recall:.4f} ({exact_case_matches}/{gold_spans_total})")
    print(f"Strict Span F1-Score:         {strict_span_f1:.4f}")
    print(f"Case-Insensitive Span F1:     {ci_span_f1:.4f}")
    print("\nEvidence State Performance:")
    for s, m in state_metrics.items():
        print(f"  {s:<14} Gold={m['gold']:<3} Pred={m['pred']:<3} Corr={m['correct']:<3} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")

    print("\nProvenance Distribution:")
    for prov, count in sorted(provenance_counts.items(), key=lambda x: -x[1]):
        pct = count / total_slots * 100
        print(f"  {prov:<32}: {count:>3} ({pct:>5.1f}%)")

    print("\nSubset Performance:")
    for sub, m in subset_metrics.items():
        s_acc = m["state_correct"] / m["slots"]
        s_prec = m["exact_span_matches"] / max(m["pred_spans"], 1)
        s_rec = m["exact_span_matches"] / max(m["gold_spans"], 1)
        s_f1 = 2 * (s_prec * s_rec) / max(s_prec + s_rec, 1e-9)
        print(f"  {sub:<24}: StateAcc={s_acc:.4f}  SpanP={s_prec:.4f}  SpanR={s_rec:.4f}  SpanF1={s_f1:.4f}")

    # Compile Final Report Dictionary
    final_report = {
        "benchmark_version": "Phase 3.6",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "checkpoint_audit": ckpt_audit,
        "sample_count": total_samples,
        "attribute_slots": total_slots,
        "metrics": {
            "evidence_state_accuracy": state_accuracy,
            "strict_span_precision": strict_span_precision,
            "strict_span_recall": strict_span_recall,
            "strict_span_f1": strict_span_f1,
            "case_insensitive_span_f1": ci_span_f1,
            "gold_spans_total": gold_spans_total,
            "predicted_spans_total": pred_spans_total,
            "exact_matches_total": exact_case_matches,
        },
        "state_metrics": state_metrics,
        "confusion_matrix": confusion_matrix,
        "provenance_distribution": provenance_counts,
        "subset_metrics": subset_metrics,
        "category_metrics": category_metrics,
        "unseen_cases_evaluation": unseen_results,
        "documented_mismatches_evaluation": mismatch_results,
    }

    out_file = "phantom-ai-ml/evaluation/phase36_evaluation_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nFull evaluation report saved to {out_file}")

    return final_report


if __name__ == "__main__":
    run_phase36_evaluation()
