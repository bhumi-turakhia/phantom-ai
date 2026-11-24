import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("phantom-ai-ml"))
sys.path.insert(0, os.path.abspath("."))
from models.wgan_gp_model import ConditionalWGANGPGenerator
from scripts.test_tensor_highpass import HighPassFrequencyRestoration, compute_laplacian_var
from utils.gcit_translator import GCIT_LAYOUT

def test_full_pipeline():
    os.makedirs("candidate_eval_samples", exist_ok=True)
    device = torch.device("cpu")
    
    # 1. Load Phase 5.4 64x64 generator
    p54_ckpt = torch.load("phantom-ai-ml/checkpoints/cgan_generator_phase54.pt", map_location=device, weights_only=False)
    gen64 = ConditionalWGANGPGenerator(image_size=64)
    gen64.load_state_dict(p54_ckpt["generator_state_dict"])
    gen64.eval()

    # 2. Load HD candidate generator
    hd_ckpt = torch.load("phantom-ai-ml/checkpoints/cgan_generator_128_hd.pt", map_location=device, weights_only=False)
    gen_hd = ConditionalWGANGPGenerator(image_size=128)
    gen_hd.load_state_dict(hd_ckpt["generator_state_dict"])
    gen_hd.eval()

    enhancer = HighPassFrequencyRestoration(channels=3, amount=2.0)

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

    for item in profiles:
        c = torch.zeros(1, 20)
        m = torch.zeros(1, 20)
        for k, v in item["attrs"].items():
            idx = GCIT_LAYOUT[k]
            c[0, idx] = v
            m[0, idx] = 1.0

        with torch.no_grad():
            # 64x64 baseline
            t64 = gen64.generate_composite(condition=c, mask=m, seed=item["seed"], device=device)
            p64_raw = ((t64[0].clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).to(torch.uint8).numpy()
            pil64 = Image.fromarray(p64_raw)
            pil64_up = pil64.resize((128, 128), Image.Resampling.BILINEAR)

            # Previous 128 (from verification_samples)
            prev128_path = f"verification_samples/sample_api_{item['id']}_seed_{item['seed']}.png"
            pil_prev128 = Image.open(prev128_path) if os.path.exists(prev128_path) else pil64_up

            # New High-Fidelity 128
            thd = gen_hd.generate_composite(condition=c, mask=m, seed=item["seed"], device=device)
            thd_enh = enhancer(thd)
            phd_raw = ((thd_enh[0].clamp(-1, 1) + 1.0) * 127.5).permute(1, 2, 0).to(torch.uint8).numpy()
            pil_new128 = Image.fromarray(phd_raw)

            # Compute Laplacian variance (sharpness)
            lap_64 = compute_laplacian_var(pil64_up)
            lap_prev = compute_laplacian_var(pil_prev128)
            lap_new = compute_laplacian_var(pil_new128)

            # Save files
            pil_new128.save(f"candidate_eval_samples/new_hd_profile_{item['id']}_seed_{item['seed']}.png")

            # Create side-by-side comparison card: [Phase 5.4 64x64] | [Prev 128x128 (blurry)] | [New 128x128 HD]
            triple_card = Image.new("RGB", (406, 148), (16, 27, 45))
            triple_card.paste(pil64_up, (8, 10))
            triple_card.paste(pil_prev128, (140, 10))
            triple_card.paste(pil_new128, (272, 10))
            triple_card.save(f"candidate_eval_samples/triple_comparison_profile_{item['id']}.png")

            print(f"\nProfile {item['id']}: {item['name']} (Seed {item['seed']})")
            print(f"  Phase 5.4 (64x64 bilinear upscale):  Laplacian Var = {lap_64:7.2f}")
            print(f"  Previous 128x128 (blurry):            Laplacian Var = {lap_prev:7.2f}")
            print(f"  New High-Fidelity 128x128:           Laplacian Var = {lap_new:7.2f}")
            print(f"  Improvement over previous 128:        +{(lap_new - lap_prev) / lap_prev * 100:.1f}%")
            print(f"  Triple comparison saved to: candidate_eval_samples/triple_comparison_profile_{item['id']}.png")

if __name__ == "__main__":
    test_full_pipeline()
