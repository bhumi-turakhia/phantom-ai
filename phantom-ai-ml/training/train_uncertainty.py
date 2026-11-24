import os
import argparse
import torch
import torch.nn as nn
from utils.device import resolve_device
from utils.seed import set_global_seed
from models.uncertainty_model import EvidentialUncertaintyClassifier


def evidential_mse_loss(alphas: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Evidential deep learning loss: Expected Mean Squared Error on Dirichlet distribution.
    """
    S = torch.sum(alphas, dim=-1, keepdim=True)
    p = alphas / S
    err = torch.sum((targets - p) ** 2, dim=-1, keepdim=True)
    var = torch.sum(alphas * (S - alphas) / (S * S * (S + 1.0)), dim=-1, keepdim=True)
    loss = err + var
    return torch.mean(loss)


def train_uncertainty_model(epochs: int = 5, lr: float = 1e-3, checkpoint_dir: str = "checkpoints"):
    set_global_seed(42)
    device = resolve_device()
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Initializing Evidential Uncertainty Classifier training on: {device}")
    model = EvidentialUncertaintyClassifier(input_dim=512, hidden_dim=256)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Evidential Model initialized with {model.count_parameters():,} trainable parameters.")
    print("Scaffold ready for evidential training on witness consistency sets in Phase 3.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_uncertainty_model(epochs=args.epochs, lr=args.lr)

