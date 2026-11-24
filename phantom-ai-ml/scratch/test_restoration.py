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

class SobelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.register_buffer("kernel_x", sobel_x)
        self.register_buffer("kernel_y", sobel_y)
    def forward(self, pred, target):
        px = F.conv2d(pred, self.kernel_x, padding=1, groups=3)
        py = F.conv2d(pred, self.kernel_y, padding=1, groups=3)
        tx = F.conv2d(target, self.kernel_x, padding=1, groups=3)
        ty = F.conv2d(target, self.kernel_y, padding=1, groups=3)
        return F.l1_loss(px, tx) + F.l1_loss(py, ty)

class FacialDetailRestoration128(nn.Module):
    def __init__(self):
        super().__init__()
        # Takes (B, 3, 64, 64) -> (B, 3, 128, 128)
        self.entry = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.PReLU(),
        )
        # Residual blocks
        self.res1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(), nn.Conv2d(64, 64, 3, 1, 1))
        self.res2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(), nn.Conv2d(64, 64, 3, 1, 1))
        self.res3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(), nn.Conv2d(64, 64, 3, 1, 1))
        self.res4 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(), nn.Conv2d(64, 64, 3, 1, 1))
        # Upsampling with PixelShuffle (no checkerboard!)
        self.up = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2), # 64x64 -> 128x128 x 64 channels
            nn.PReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )
    def forward(self, x64):
        # Base bicubic upsampling for residual skip
        base = F.interpolate(x64, size=(128, 128), mode="bilinear", align_corners=False)
        feat = self.entry(x64)
        feat = feat + self.res1(feat)
        feat = feat + self.res2(feat)
        feat = feat + self.res3(feat)
        feat = feat + self.res4(feat)
        res = self.up(feat)
        return torch.clamp(base + 0.5 * res, -1.0, 1.0)

def test_restoration():
    device = get_device("auto")
    ds = CelebAIdentityDataset(
        metadata_dir="phantom-ai-ml/datasets/celeba",
        split="split_a1",
        image_dir="phantom-ai-ml/datasets/celeba/img_align_celeba",
        image_size=128,
        require_images=True,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    model = FacialDetailRestoration128().to(device)
    sobel = SobelLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("Training Facial Detail Restoration Module for 250 steps...")
    it = iter(loader)
    for step in range(250):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
            
        x_real_128 = b["image"].to(device)
        # Downsample to 64x64 to create input
        x_in_64 = F.interpolate(x_real_128, size=(64, 64), mode="bilinear", align_corners=False)
        
        opt.zero_grad()
        x_pred_128 = model(x_in_64)
        
        loss_l1 = F.l1_loss(x_pred_128, x_real_128)
        loss_edge = sobel(x_pred_128, x_real_128)
        loss = loss_l1 + 0.2 * loss_edge
        
        loss.backward()
        opt.step()
        
        if (step + 1) % 50 == 0:
            print(f"Step [{step+1}/250] L1: {loss_l1.item():.4f} | Edge: {loss_edge.item():.4f}")
            
    # Apply to profile 1 image from Phase 5.4!
    from PIL import Image
    im64 = Image.open("visual_comparisons_upgrade/profile_1_seed_1001_phase54_64x64.png")
    arr = (np.array(im64.convert("RGB"), dtype=np.float32) / 127.5) - 1.0
    t64 = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        restored = model(t64)
        
    res_np = ((restored[0].clamp(-1, 1).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
    res_pil = Image.fromarray(np.transpose(res_np, (1, 2, 0)))
    res_pil.save("phantom-ai-ml/scratch/restored_profile_1.png")
    print("Saved restored image to phantom-ai-ml/scratch/restored_profile_1.png")

if __name__ == "__main__":
    test_restoration()

