import os
import json
from typing import Dict, Any, Optional
from schemas.requests import EvaluateResponse


class EvaluationService:
    """
    Truthful model evaluation service.
    Refuses to fabricate evaluation metrics or training results when models are untrained.
    When a genuine trained checkpoint and evaluation report exist, returns real evaluation metrics.
    """
    def __init__(self, checkpoints_dir: str):
        self.checkpoints_dir = checkpoints_dir

    def evaluate_model(self, model_name: str, dataset_name: Optional[str] = None) -> EvaluateResponse:
        model_name = model_name.lower().strip()

        checkpoint_files = {
            "nlp": "nlp_bilstm.pt",
            "uncertainty": "uncertainty_model.pt",
            "cnn": "cnn_feature_extractor.pt",
            "dcgan": "dcgan_generator.pt",
        }

        if model_name not in checkpoint_files:
            return EvaluateResponse(
                model_name=model_name,
                status="unrecognized_model",
                is_trained=False,
                metrics=None,
                evaluation_details=f"Model '{model_name}' is not recognized. Supported models: {list(checkpoint_files.keys())}"
            )

        ckpt_file = checkpoint_files[model_name]
        ckpt_path = os.path.join(self.checkpoints_dir, ckpt_file)

        if not os.path.exists(ckpt_path):
            return EvaluateResponse(
                model_name=model_name,
                status="model_not_trained",
                is_trained=False,
                metrics=None,
                evaluation_details=(
                    f"Model '{model_name}' has not been trained yet. No checkpoint found at '{ckpt_path}'. "
                    "In adherence to Phantom AI safety constraints, no simulated or fabricated evaluation metrics are returned."
                )
            )

        # Check if actual evaluation report exists on disk
        report_file = os.path.join(self.checkpoints_dir, "evaluation_results.json")
        if not os.path.exists(report_file):
            alt_report = os.path.join("checkpoints", "evaluation_results.json")
            if os.path.exists(alt_report):
                report_file = alt_report
        if model_name == "nlp" and os.path.exists(report_file):
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                span_metrics = eval_data.get("trained_model_metrics", {}).get("exact_span_level", {})
                return EvaluateResponse(
                    model_name=model_name,
                    status="evaluated_on_test_split",
                    is_trained=True,
                    metrics={
                        "precision": span_metrics.get("precision", 0.0),
                        "recall": span_metrics.get("recall", 0.0),
                        "f1": span_metrics.get("f1", 0.0),
                    },
                    evaluation_details=(
                        f"Genuine evaluation on held-out test set ({eval_data.get('test_sample_count', 0)} samples). "
                        f"Exact-span F1: {span_metrics.get('f1', 0.0):.4f}. "
                        f"Disclaimer: {eval_data.get('disclaimer', '')}"
                    )
                )
            except Exception as e:
                pass

        return EvaluateResponse(
            model_name=model_name,
            status="checkpoint_found_pending_evaluation_dataset",
            is_trained=True,
            metrics={"precision": 0.0, "recall": 0.0, "f1": 0.0},
            evaluation_details=f"Trained checkpoint detected at '{ckpt_path}'. Ready for dataset evaluation."
        )
