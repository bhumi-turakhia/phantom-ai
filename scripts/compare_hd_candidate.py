import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("phantom-ai-ml"))

from models.wgan_gp_model import ConditionalWGANGPGenerator
from config.gan_config import WGANConfig

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

def run_candidate_comparison():
    os.makedirs("candidate_eval_samples", exist_ok=True)
    device = torch.device("cpu")

    # 1. Load Phase 5.4 64x64 generator
    p54_ckpt = torch.load("phantom-ai-ml/checkpoints/cgan_generator_phase54.pt", map_location=device, weights_only=False)
    gen64 = ConditionalWGANGPGenerator(image_size=64)
    gen64.load_state_dict(p54_ckpt["generator_state_dict"])
    gen64.eval()

    # 2. Load Old 128x128 checkpoint
    old128_ckpt = torch.load("phantom-ai-ml/checkpoints/cgan_generator_128.pt", map_location=device, weights_only=False)
    # Temporary instantiate with previous refinement architecture if needed or check
    print("Testing candidate generator: cgan_generator_128_hd.pt...")
    hd_ckpt = torch.load("phantom-ai-ml/checkpoints/cgan_generator_128_hd.pt", map_location=device, weights_only=False)
    gen_hd = ConditionalWGANGPGenerator(image_size=128)
    gen_hd.load_state_dict(hd_ckpt["generator_state_dict"])
    gen_hd.eval()

    profiles = [
        {
            "id": 1,
            "name": "Male, Black Straight Hair, Clean-Shaven",
            "seed": 1001,
            "attrs": {"hair_black": 1.0, "hair_straight": 1.0, "facial_no_beard": 1.0},
        },
        {
            "id": 2,
            "name": "Female, Blond Wavy Hair, High Cheekbones, Big Lips",
            "seed": 2002,
            "attrs": {"hair_blond": 1.0, "hair_wavy": 1.0, "lips_big": 1.0, "face_cheekbones": 1.0},
        },
        {
            "id": 3,
            "name": "Male, Gray Hair, Glasses, Mustache",
            "seed": 3003,
            "attrs": {"hair_gray": 1.0, "glasses": 1.0, "facial_mustache": 1.0},
        },
    ]

    from utils.gcit_translator import GCIT_LAYOUT

    results = []

    for item in profiles:
        c = torch.zeros(1, 20)
        m = torch.zeros(1, 20)
        for k, v in item["attrs"].items():
            idx = GCIT_LAYOUT[k]
            c[0, idx] = v
            m[0, idx] = 1.0

        with torch.no_grad():
            # 64x64 Phase 5.4 output
            img64_tensor = gen64.generate_composite(condition=c, mask=m, seed=item["seed"], device=device)
            img64_np = ((img64_tensor[0].clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).to(torch.uint8).numpy()
            pil64 = Image.fromarray(img64_np)
            pil64_up = pil64.resize((128, 128), Image.Resampling.BILINEAR)
            lap_64_up = compute_laplacian_var(pil64_up)

            # HD 128x128 candidate output
            img_hd_tensor = gen_hd.generate_composite(condition=c, mask=m, seed=item["seed"], device=device)
            img_hd_np = ((img_hd_tensor[0].clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).to(torch.uint8).numpy()
            pil_hd = Image.fromarray(img_hd_np)
            lap_hd = compute_laplacian_var(pil_hd)

            # Save sample outputs
            p64_path = f"candidate_eval_samples/profile_{item['id']}_p54_64x64.png"
            phd_path = f"candidate_eval_samples/profile_{item['id']}_candidate_128x128_hd.png"
            pil64.save(p64_path)
            pil_hd.save(phd_path)

            # Pairwise side-by-side card
            card = Image.new("RGB", (270, 140), (20, 30, 45))
            card.paste(pil64_up, (8, 6))
            card.paste(pil_hd, (134, 6))
            card_path = f"candidate_eval_samples/comparison_profile_{item['id']}.png"
            card.save(card_path)

            gain = ((lap_hd - lap_64_up) / lap_64_up) * 100.0
            print(f"\nProfile {item['id']}: {item['name']} (Seed {item['seed']})")
            print(f"  P54 Bilinear 128 Laplacian Variance: {lap_64_up:.2f}")
            print(f"  HD Candidate 128 Laplacian Variance: {lap_hd:.2f}")
            print(f"  Sharpness Detail Gain: +{gain:.1f}%")
            print(f"  Saved comparison: {card_path}")

            results.append({
                "id": item["id"],
                "name": item["name"],
                "seed": item["seed"],
                "lap_64_bilinear": lap_64_up,
                "lap_hd": lap_hd,
                "detail_gain_pct": gain,
                "comparison_card": card_path
            })

    print("\n" + "=" * 80)
    print("ALL CANDIDATE COMPARISONS COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    run_candidate_comparison()

