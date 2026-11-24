import os
import argparse
import torch
import torch.nn as nn
from utils.device import resolve_device
from utils.seed import set_global_seed
from models.cnn_model import ForensicFeatureCNN


def train_cnn_model(epochs: int = 5, lr: float = 2e-4, checkpoint_dir: str = "checkpoints"):
    set_global_seed(42)
    device = resolve_device()
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Initializing Forensic CNN training on: {device}")
    model = ForensicFeatureCNN(in_channels=3, feature_dim=256, image_size=64)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"Forensic CNN initialized with {model.count_parameters():,} trainable parameters.")
    print("Scaffold ready for supervised composite feature extraction in Phase 3.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train_cnn_model(epochs=args.epochs, lr=args.lr)

