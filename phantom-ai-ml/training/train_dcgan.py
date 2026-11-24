import os
import argparse
import torch
import torch.nn as nn
from utils.device import resolve_device
from utils.seed import set_global_seed
from models.dcgan_model import ConditionalDCGANGenerator, ConditionalDCGANDiscriminator


def train_dcgan_model(
    epochs: int = 5,
    latent_dim: int = 100,
    condition_dim: int = 32,
    lr: float = 2e-4,
    beta1: float = 0.5,
    checkpoint_dir: str = "checkpoints",
):
    set_global_seed(42)
    device = resolve_device()
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Initializing Conditional DCGAN training on: {device}")
    netG = ConditionalDCGANGenerator(latent_dim=latent_dim, condition_dim=condition_dim, image_size=64).to(device)
    netD = ConditionalDCGANDiscriminator(in_channels=3, condition_dim=condition_dim, image_size=64).to(device)

    criterion = nn.BCELoss()
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    print(f"Generator parameters: {netG.count_parameters():,}")
    print(f"Discriminator parameters: {netD.count_parameters():,}")
    print("Scaffold ready for conditional sketch/composite dataset training in Phase 3.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train_dcgan_model(epochs=args.epochs, lr=args.lr)

