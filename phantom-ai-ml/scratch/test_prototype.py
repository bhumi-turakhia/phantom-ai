import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath("phantom-ai-ml"))
from data.celeba_dataset import CelebAIdentityDataset
from config.gan_config import get_device

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
    def forward(self, x):
        return x + self.conv(x)

class HighResGenerator128(nn.Module):
    def __init__(self, latent_dim=100, condition_dim=20, condition_embed_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim * 2, condition_embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(condition_embed_dim, condition_embed_dim),
        )
        in_dim = latent_dim + condition_embed_dim
        # Initial projection to 4x4 x 512
        self.fc = nn.Linear(in_dim, 4 * 4 * 512)
        
        # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(256),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(128),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(64),
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, condition, mask):
        cond_input = torch.cat([condition, mask], dim=1)
        cond_embed = self.condition_embed(cond_input)
        x = torch.cat([z, cond_embed], dim=1)
        x = self.fc(x).view(-1, 512, 4, 4)
        x = self.up1(x) # 8x8
        x = self.up2(x) # 16x16
        x = self.up3(x) # 32x32
        x = self.up4(x) # 64x64
        x = self.up5(x) # 128x128
        return x

class FaceEncoder128(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), # 64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1), # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        feat = self.net(x)
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        return mu, logvar

def run_test():
    device = get_device("auto")
    print(f"Device: {device}")
    ds = CelebAIdentityDataset(
        metadata_dir="phantom-ai-ml/datasets/celeba",
        split="split_a1",
        image_dir="phantom-ai-ml/datasets/celeba/img_align_celeba",
        image_size=128,
        attribute_dropout=0.1,
        require_images=True,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    
    gen = HighResGenerator128().to(device)
    enc = FaceEncoder128().to(device)
    
    opt = torch.optim.AdamW(list(gen.parameters()) + list(enc.parameters()), lr=2e-4, weight_decay=1e-4)
    
    print("Training 150 prototype steps...")
    iter_loader = iter(loader)
    for step in range(150):
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            batch = next(iter_loader)
            
        x_real = batch["image"].to(device)
        cond = batch["condition"].to(device)
        mask = batch["mask"].to(device)
        
        opt.zero_grad()
        mu, logvar = enc(x_real)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        x_rec = gen(z, cond, mask)
        
        # Loss: L1 pixel reconstruction + KL divergence
        l1_loss = F.l1_loss(x_rec, x_real)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        loss = l1_loss + 0.001 * kl_loss
        loss.backward()
        opt.step()
        
        if (step + 1) % 50 == 0:
            print(f"Step [{step+1}/150] L1: {l1_loss.item():.4f} | KL: {kl_loss.item():.4f}")
            
    # Test generation from pure noise z ~ N(0, 1) and fixed conditions
    gen.eval()
    with torch.no_grad():
        fixed_z = torch.randn(4, 100, device=device)
        sample_rec = gen(fixed_z, cond[:4], mask[:4])
        
    imgs = ((sample_rec.clamp(-1, 1).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
    for i in range(4):
        im = Image.fromarray(np.transpose(imgs[i], (1, 2, 0)))
        im.save(f"phantom-ai-ml/scratch/sample_{i}.png")
    print("Saved 4 prototype samples to phantom-ai-ml/scratch/sample_*.png")

if __name__ == "__main__":
    run_test()

