import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("phantom-ai-ml"))
from models.wgan_gp_model import ConditionalWGANGPGenerator
from utils.gcit_translator import GCIT_LAYOUT

class HighPassFrequencyRestoration(nn.Module):
    """
    Forensic High-Pass Frequency Restoration & Anatomical Edge Definition Layer.
    Restores high-frequency edge gradients (pupils, iris margins, eyebrows, nostrils, lips)
    attenuated by deconvolutional GAN architectures.
    """
    def __init__(self, channels: int = 3, kernel_size: int = 5, sigma: float = 1.2, amount: float = 1.4):
        super().__init__()
        self.channels = channels
        self.amount = amount

        # Create Gaussian kernel
        ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        self.register_buffer("gaussian_kernel", kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is in [-1.0, 1.0]
        low_pass = F.conv2d(x, self.gaussian_kernel, padding=self.gaussian_kernel.size(-1) // 2, groups=self.channels)
        high_pass = x - low_pass
        
        # High-frequency sharpening with soft edge preservation
        sharpened = x + self.amount * high_pass
        return torch.clamp(sharpened, -1.0, 1.0)


def compute_laplacian_var(img_pil: Image.Image) -> float:
    gray = img_pil.convert("L")
    arr = np.array(gray, dtype=np.float64)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    h, w = arr.shape
    lap = np.zeros((h - 2, w - 2))
    for i in range(3):
        for j in range(3):
            lap += kernel[i, j] * arr[i : h - 2 + i, j : w - 2 + j]
    return float(lap.var())


def test():
    device = torch.device("cpu")
    hd_ckpt = torch.load("phantom-ai-ml/checkpoints/cgan_generator_128_hd.pt", map_location=device, weights_only=False)
    gen_hd = ConditionalWGANGPGenerator(image_size=128)
    gen_hd.load_state_dict(hd_ckpt["generator_state_dict"])
    gen_hd.eval()

    enhancer = HighPassFrequencyRestoration(channels=3, amount=1.5)

    c = torch.zeros(1, 20)
    m = torch.zeros(1, 20)
    c[0, 0] = 1.0; m[0, 0] = 1.0  # hair_black
    c[0, 6] = 1.0; m[0, 6] = 1.0  # hair_straight
    c[0, 13] = 1.0; m[0, 13] = 1.0 # facial_no_beard

    with torch.no_grad():
        out_raw = gen_hd.generate_composite(condition=c, mask=m, seed=1001, device=device)
        out_enhanced = enhancer(out_raw)

        raw_np = ((out_raw[0].clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).to(torch.uint8).numpy()
        enh_np = ((out_enhanced[0].clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).to(torch.uint8).numpy()

        pil_raw = Image.fromarray(raw_np)
        pil_enh = Image.fromarray(enh_np)

        pil_raw.save("candidate_eval_samples/test_hd_raw.png")
        pil_enh.save("candidate_eval_samples/test_hd_enhanced.png")

        print(f"Raw 128 HD Laplacian Var: {compute_laplacian_var(pil_raw):.2f}")
        print(f"Enhanced 128 HD Laplacian Var: {compute_laplacian_var(pil_enh):.2f}")

if __name__ == "__main__":
    test()

