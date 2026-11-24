import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.abspath("phantom-ai-ml"))
from models.wgan_gp_model import ConditionalWGANGPGenerator
from utils.gcit_translator import GCIT_LAYOUT

def test_contrast_and_unsharp():
    device = torch.device("cpu")
    p54_ckpt = torch.load("phantom-ai-ml/checkpoints/cgan_generator_phase54.pt", map_location=device, weights_only=False)
    gen64 = ConditionalWGANGPGenerator(image_size=64)
    gen64.load_state_dict(p54_ckpt["generator_state_dict"])
    gen64.eval()

    hd_ckpt = torch.load("phantom-ai-ml/checkpoints/cgan_generator_128_hd.pt", map_location=device, weights_only=False)
    gen_hd = ConditionalWGANGPGenerator(image_size=128)
    gen_hd.load_state_dict(hd_ckpt["generator_state_dict"])
    gen_hd.eval()

    c = torch.zeros(1, 20)
    m = torch.zeros(1, 20)
    c[0, 0] = 1.0; m[0, 0] = 1.0  # hair_black
    c[0, 6] = 1.0; m[0, 6] = 1.0  # hair_straight
    c[0, 13] = 1.0; m[0, 13] = 1.0 # facial_no_beard

    with torch.no_grad():
        img_hd = gen_hd.generate_composite(condition=c, mask=m, seed=1001, device=device)
        img_np = ((img_hd[0].clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).to(torch.uint8).numpy()
        pil_raw = Image.fromarray(img_np)
        pil_raw.save("candidate_eval_samples/test_raw_hd.png")

        # Let's inspect what unsharp mask does
        pil_sharp1 = pil_raw.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=2))
        pil_sharp1.save("candidate_eval_samples/test_sharp1.png")

        pil_sharp2 = pil_raw.filter(ImageFilter.UnsharpMask(radius=2.0, percent=220, threshold=1))
        pil_sharp2.save("candidate_eval_samples/test_sharp2.png")

    def lap_var(img):
        arr = np.array(img.convert("L"), dtype=np.float64)
        k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        h, w = arr.shape
        lap = np.zeros((h - 2, w - 2))
        for i in range(3):
            for j in range(3):
                lap += k[i, j] * arr[i : h - 2 + i, j : w - 2 + j]
        return float(lap.var())

    print(f"Raw HD Output: Laplacian var = {lap_var(pil_raw):.2f}")
    print(f"Sharp 1 Output: Laplacian var = {lap_var(pil_sharp1):.2f}")
    print(f"Sharp 2 Output: Laplacian var = {lap_var(pil_sharp2):.2f}")

if __name__ == "__main__":
    test_contrast_and_unsharp()

