import os
from services.evaluation_service import EvaluationService


def evaluate_generation(checkpoints_dir: str = "checkpoints"):
    service = EvaluationService(checkpoints_dir=checkpoints_dir)
    result = service.evaluate_model(model_name="dcgan")
    print(f"Model: {result.model_name}")
    print(f"Status: {result.status}")
    print(f"Is Trained: {result.is_trained}")
    print(f"Details: {result.evaluation_details}")
    return result


if __name__ == "__main__":
    evaluate_generation()

