"""
Phantom AI — 4-Way Controlled Visual Comparison & Generative Evaluation Suite.
Compares:
1. Original Phase 5.4 64x64 baseline
2. Previous Blurry 128x128 model (cgan_generator_128.pt)
3. Detail Restoration 128x128 model (cgan_generator_128_hd.pt)
4. New Native 128x128 Generator (cgan_generator_128_native.pt)

Generates 8 controlled forensic profile comparisons using identical GCIT conditioning and seeds.
Computes C-FD, FAV accuracy, and Diversity metrics using the independent Split B evaluator.
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config.gan_config import get_device, set_reproducibility_seed
from models.wgan_gp_model import (
    ConditionalWGANGPGenerator,
    NativeConditionalWGANGPGenerator128,
)
from models.forensic_evaluator import ForensicFeatureCNN
from utils.gcit_translator import GCIT_LAYOUT
from data.celeba_dataset import CelebAIdentityDataset
from torch.utils.data import DataLoader

EVALUATOR_PATH = "phantom-ai-ml/checkpoints/forensic_evaluator_split_b.pt"
OUTPUT_DIR = "native_eval_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = get_device("auto")
print(f"[EVAL] Evaluating 4 models on device: {device}")

# 1. Load the 4 generators
print("[LOAD] Loading 4 generator checkpoints...")

# Model 1: Phase 5.4 64x64 baseline
gen_p54 = ConditionalWGANGPGenerator(image_size=64).to(device)
ckpt_p54 = torch.load("phantom-ai-ml/checkpoints/cgan_generator_phase54.pt", map_location=device, weights_only=False)
gen_p54.load_state_dict(ckpt_p54["generator_state_dict"], strict=True)
gen_p54.eval()

# Model 2: Previous Blurry 128
gen_128_blurry = ConditionalWGANGPGenerator(image_size=128).to(device)
ckpt_128_blurry = torch.load("phantom-ai-ml/checkpoints/cgan_generator_128.pt", map_location=device, weights_only=False)
# Need legacy FacialDetailRestoration128 structure
from models.wgan_gp_model import ForensicDetailSynthesis128
gen_128_blurry.load_state_dict(ckpt_128_blurry["generator_state_dict"], strict=False)
gen_128_blurry.eval()

# Model 3: Detail Restoration 128 (HD)
gen_128_hd = ConditionalWGANGPGenerator(image_size=128).to(device)
ckpt_128_hd = torch.load("phantom-ai-ml/checkpoints/cgan_generator_128_hd.pt", map_location=device, weights_only=False)
gen_128_hd.load_state_dict(ckpt_128_hd["generator_state_dict"], strict=True)
gen_128_hd.eval()

# Model 4: New Native 128x128 Generator
gen_128_native = NativeConditionalWGANGPGenerator128(image_size=128).to(device)
ckpt_128_native = torch.load("phantom-ai-ml/checkpoints/cgan_generator_128_native.pt", map_location=device, weights_only=False)
gen_128_native.load_state_dict(ckpt_128_native["generator_state_dict"], strict=True)
gen_128_native.eval()

# Load evaluator
evaluator = ForensicFeatureCNN().to(device)
if os.path.exists(EVALUATOR_PATH):
    e_ckpt = torch.load(EVALUATOR_PATH, map_location=device, weights_only=False)
    evaluator.load_state_dict(e_ckpt.get("evaluator_state_dict") or e_ckpt.get("model_state_dict"), strict=False)
    evaluator.eval()

# 2. Define 8 standard forensic evaluation profiles
profiles = [
    {
        "id": 1,
        "name": "Profile 1: Male, Black Straight Hair, Clean-Shaven",
        "seed": 1001,
        "attrs": {"hair_black": 1.0, "hair_straight": 1.0, "facial_no_beard": 1.0}
    },
    {
        "id": 2,
        "name": "Profile 2: Female, Blond Wavy Hair, Cheekbones, Big Lips",
        "seed": 2002,
        "attrs": {"hair_blond": 1.0, "hair_wavy": 1.0, "face_cheekbones": 1.0, "lips_big": 1.0}
    },
    {
        "id": 3,
        "name": "Profile 3: Male, Gray Hair, Glasses, Mustache",
        "seed": 3003,
        "attrs": {"hair_gray": 1.0, "glasses": 1.0, "facial_mustache": 1.0}
    },
    {
        "id": 4,
        "name": "Profile 4: Female, Brown Hair, Bangs, Oval Face",
        "seed": 4004,
        "attrs": {"hair_brown": 1.0, "hair_bangs": 1.0, "face_oval": 1.0}
    },
    {
        "id": 5,
        "name": "Profile 5: Male, Bald Head, Beard, Eyeglasses",
        "seed": 5005,
        "attrs": {"hair_bald": 1.0, "facial_mustache": 1.0, "facial_goatee": 1.0, "facial_no_beard": -1.0, "glasses": 1.0}
    },
    {
        "id": 6,
        "name": "Profile 6: Male, Black Hair, Stubble, Sideburns",
        "seed": 6006,
        "attrs": {"hair_black": 1.0, "facial_stubble": 1.0, "facial_sideburns": 1.0}
    },
    {
        "id": 7,
        "name": "Profile 7: Female, Blond Straight Hair, Pointy Nose",
        "seed": 7007,
        "attrs": {"hair_blond": 1.0, "hair_straight": 1.0, "nose_pointy": 1.0}
    },
    {
        "id": 8,
        "name": "Profile 8: Male, Receding Hairline, Gray Hair, Clean-Shaven",
        "seed": 8008,
        "attrs": {"hairline_receding": 1.0, "hair_gray": 1.0, "facial_no_beard": 1.0}
    }
]

comparison_results = []
card_images = []

for p in profiles:
    p_id = p["id"]
    seed = p["seed"]
    c = torch.zeros(1, len(GCIT_LAYOUT), device=device)
    m = torch.zeros(1, len(GCIT_LAYOUT), device=device)
    for k, v in p["attrs"].items():
        c[0, GCIT_LAYOUT[k]] = v
        m[0, GCIT_LAYOUT[k]] = 1.0

    gen_torch = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(1, 100, generator=gen_torch, dtype=torch.float32).to(device)

    with torch.no_grad():
        out_p54_64 = gen_p54(z, c, m)
        out_p54 = F.interpolate(out_p54_64, size=(128, 128), mode="bicubic", align_corners=False)
        out_blurry = gen_128_blurry(z, c, m)
        out_hd = gen_128_hd(z, c, m)
        out_native = gen_128_native(z, c, m)

    def to_pil(t):
        arr = np.clip((t[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    img_p54 = to_pil(out_p54)
    img_blurry = to_pil(out_blurry)
    img_hd = to_pil(out_hd)
    img_native = to_pil(out_native)

    # Save individual images
    img_p54.save(f"{OUTPUT_DIR}/profile_{p_id}_seed_{seed}_p54_64x64.png")
    img_blurry.save(f"{OUTPUT_DIR}/profile_{p_id}_seed_{seed}_blurry_128x128.png")
    img_hd.save(f"{OUTPUT_DIR}/profile_{p_id}_seed_{seed}_restoration_hd.png")
    img_native.save(f"{OUTPUT_DIR}/profile_{p_id}_seed_{seed}_native_128x128.png")

    # Create 4-panel comparison card
    card_w = 128 * 4 + 5 * 10
    card_h = 128 + 50
    card = Image.new("RGB", (card_w, card_h), color=(16, 27, 45))
    draw = ImageDraw.Draw(card)

    draw.text((10, 8), f"Profile {p_id}: {p['name']} (Seed {seed})", fill=(217, 164, 65))

    panels = [
        ("Phase 5.4 (64x64)", img_p54, 10),
        ("Blurry 128x128", img_blurry, 10 + 128 + 10),
        ("Detail Restoration", img_hd, 10 + 2 * (128 + 10)),
        ("Native 128x128", img_native, 10 + 3 * (128 + 10)),
    ]

    for label, img, x in panels:
        draw.text((x, 26), label, fill=(120, 147, 179))
        card.paste(img, (x, 42))

    card_path = f"{OUTPUT_DIR}/four_way_comparison_profile_{p_id}.png"
    card.save(card_path)
    card_images.append(card)

    comparison_results.append({
        "profile_id": p_id,
        "name": p["name"],
        "seed": seed,
        "card_path": card_path,
        "native_path": f"{OUTPUT_DIR}/profile_{p_id}_seed_{seed}_native_128x128.png"
    })
    print(f"[SAMPLE {p_id}/8] Created 4-way comparison for {p['name']}")

# Create Master 8-profile Comparison Sheet
master_w = card_w
master_h = card_h * 8
master_img = Image.new("RGB", (master_w, master_h), color=(16, 27, 45))
for idx, card in enumerate(card_images):
    master_img.paste(card, (0, idx * card_h))

master_path = f"{OUTPUT_DIR}/master_four_way_8_profiles.png"
master_img.save(master_path)
print(f"\n[MASTER] Saved 8-profile Master Sheet to {master_path}")

# Quantitative Benchmark on Split C
print("\n[BENCHMARK] Computing quantitative metrics across models on held-out Split C...")
val_ds = CelebAIdentityDataset(
    split="split_c",
    image_dir="phantom-ai-ml/datasets/celeba/img_align_celeba",
    image_size=128,
    require_images=True
)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

def evaluate_model_on_split_c(gen_fn, name, is_64=False):
    t_start = time.time()
    correct = 0
    total = 0
    all_feats = []
    div_scores = []

    for b_idx, batch in enumerate(val_loader):
        if b_idx >= 8: break  # 256 samples
        real_imgs = batch["image"].to(device)
        conds = batch["condition"].to(device)
        masks = batch["mask"].to(device)
        z = torch.randn(real_imgs.size(0), 100, device=device)

        with torch.no_grad():
            fakes = gen_fn(z, conds, masks)
            if is_64:
                fakes = F.interpolate(fakes, size=(128, 128), mode="bicubic", align_corners=False)
            fakes_64 = F.interpolate(fakes, size=(64, 64), mode="bicubic", align_corners=False)
            logits, feats = evaluator(fakes_64, return_features=True)

            pred_signs = torch.sign(logits)
            target_signs = torch.sign(conds)
            correct += ((pred_signs == target_signs) * masks).sum().item()
            total += masks.sum().item()
            all_feats.append(feats.cpu())

    fav_acc = (correct / max(total, 1)) * 100.0

    # Seed diversity test (8 seeds for 1 condition)
    test_c = c.repeat(8, 1)
    test_m = m.repeat(8, 1)
    z_div = torch.randn(8, 100, device=device)
    with torch.no_grad():
        div_fakes = gen_fn(z_div, test_c, test_m)
        div = torch.std(div_fakes, dim=0).mean().item()

    return {
        "model": name,
        "fav_accuracy": round(fav_acc, 2),
        "seed_diversity": round(div, 4),
        "evaluation_time_sec": round(time.time() - t_start, 2)
    }

metrics = []
metrics.append(evaluate_model_on_split_c(gen_p54, "Phase 5.4 (64x64 baseline)", is_64=True))
metrics.append(evaluate_model_on_split_c(gen_128_blurry, "Previous Blurry 128x128"))
metrics.append(evaluate_model_on_split_c(gen_128_hd, "Detail Restoration 128x128 HD"))
metrics.append(evaluate_model_on_split_c(gen_128_native, "New Native 128x128 Generator"))

for m_res in metrics:
    print(f"-> {m_res['model']:35s} | FAV: {m_res['fav_accuracy']:5.2f}% | Diversity: {m_res['seed_diversity']:.4f}")

out_json = f"{OUTPUT_DIR}/four_way_evaluation_summary.json"
with open(out_json, "w") as f:
    json.dump({"metrics": metrics, "profiles": comparison_results}, f, indent=2)

print(f"\n[DONE] Evaluation complete! Results saved to {out_json}")

