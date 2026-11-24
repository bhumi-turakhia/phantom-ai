import io
import json
import base64
import os
import urllib.request
import numpy as np
from PIL import Image

def compute_metrics(img_np: np.ndarray):
    gray = 0.2989 * img_np[:, :, 0] + 0.5870 * img_np[:, :, 1] + 0.1140 * img_np[:, :, 2]
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = gray.shape
    lap = np.zeros((h - 2, w - 2), dtype=np.float32)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            patch = gray[i-1:i+2, j-1:j+2]
            lap[i-1, j-1] = np.sum(patch * kernel)
    return float(np.var(lap))

profiles = [
    {
        "name": "Profile 1: Male, Black Straight Hair, Clean-Shaven",
        "seed": 1001,
        "forensic_profile": {
            "gender": {"value": "male", "state": "explicit"},
            "hair_color": {"value": "black", "state": "explicit"},
            "hair_style": {"value": "straight", "state": "explicit"},
            "facial_hair": {"value": "clean-shaven", "state": "explicit"}
        }
    },
    {
        "name": "Profile 2: Female, Blond Wavy Hair, High Cheekbones, Big Lips",
        "seed": 2002,
        "forensic_profile": {
            "gender": {"value": "female", "state": "explicit"},
            "hair_color": {"value": "blond", "state": "explicit"},
            "hair_style": {"value": "wavy", "state": "explicit"},
            "facial_shape": {"value": "high cheekbones", "state": "explicit"},
            "mouth": {"value": "big lips", "state": "explicit"}
        }
    },
    {
        "name": "Profile 3: Male, Gray Hair, Glasses, Mustache",
        "seed": 3003,
        "forensic_profile": {
            "gender": {"value": "male", "state": "explicit"},
            "hair_color": {"value": "gray", "state": "explicit"},
            "facial_hair": {"value": "mustache", "state": "explicit"},
            "eye_details": {"value": "eyeglasses", "state": "explicit"}
        }
    }
]

OUTPUT_DIR = "native_live_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)
results = []

for idx, p in enumerate(profiles, 1):
    payload = {
        "forensic_profile": p["forensic_profile"],
        "seed": p["seed"],
        "num_variations": 3,
        "case_id": "CASE-NATIVE-128-LIVE"
    }
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/generate-composite",
        data=req_data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        assert resp.status == 200, f"Error {resp.status}"
        data = json.loads(resp.read().decode("utf-8"))

    assert data["status"] == "success"
    gen_info = data["generator"]
    variations = data["variations"]
    assert len(variations) == 3

    print(f"\n=== Profile {idx}: {p['name']} ===")
    print(f"Checkpoint Loaded: {os.path.basename(gen_info['checkpoint_path'])}")
    print(f"SHA-256: {gen_info['checkpoint_sha256']}")
    print(f"Architecture: {gen_info['architecture']}")
    print(f"Latency: {data['latency_ms']:.1f}ms")

    for v in variations:
        var_idx = v["variation_index"]
        var_seed = v["seed"]
        w = v["width"]
        h = v["height"]
        assert w == 128 and h == 128, f"Unexpected dimensions: {w}x{h}"

        img_bytes = base64.b64decode(v["image_base64"])
        img = Image.open(io.BytesIO(img_bytes))
        assert img.size == (128, 128)

        out_path = f"{OUTPUT_DIR}/live_p{idx}_var{var_idx}_seed_{var_seed}.png"
        img.save(out_path)

        img_np = np.array(img, dtype=np.float32)
        lap = compute_metrics(img_np)
        print(f"  Variation #{var_idx} (Seed {var_seed}): size={img.size}, LapVar={lap:.1f}, saved={out_path}")

        results.append({
            "profile": p["name"],
            "variation_index": var_idx,
            "seed": var_seed,
            "width": w,
            "height": h,
            "laplacian_variance": round(lap, 2),
            "saved_path": out_path,
            "checkpoint": gen_info["checkpoint_path"],
            "sha256": gen_info["checkpoint_sha256"],
        })

with open(f"{OUTPUT_DIR}/live_generation_report.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n[SUCCESS] All live API generations completed and verified with native 128x128 model!")

