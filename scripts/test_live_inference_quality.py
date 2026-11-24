import io
import json
import base64
import os
import urllib.request
import numpy as np
from PIL import Image

def compute_laplacian_variance(img_np: np.ndarray) -> float:
    # Convert RGB to grayscale (float32 [0, 255])
    gray = 0.2989 * img_np[:, :, 0] + 0.5870 * img_np[:, :, 1] + 0.1140 * img_np[:, :, 2]
    # Standard 3x3 discrete Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = gray.shape
    lap = np.zeros((h - 2, w - 2), dtype=np.float32)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            patch = gray[i-1:i+2, j-1:j+2]
            lap[i-1, j-1] = np.sum(patch * kernel)
    return float(np.var(lap))

def compute_sobel_energy(img_np: np.ndarray) -> float:
    gray = 0.2989 * img_np[:, :, 0] + 0.5870 * img_np[:, :, 1] + 0.1140 * img_np[:, :, 2]
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    h, w = gray.shape
    gx = np.zeros((h - 2, w - 2), dtype=np.float32)
    gy = np.zeros((h - 2, w - 2), dtype=np.float32)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            patch = gray[i-1:i+2, j-1:j+2]
            gx[i-1, j-1] = np.sum(patch * kx)
            gy[i-1, j-1] = np.sum(patch * ky)
    mag = np.sqrt(gx**2 + gy**2)
    return float(np.mean(mag))

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

os.makedirs("verification_hd_samples", exist_ok=True)
results = []

for idx, p in enumerate(profiles, 1):
    payload = {
        "forensic_profile": p["forensic_profile"],
        "seed": p["seed"],
        "num_variations": 1,
        "case_id": "CASE-VERIFY-HD"
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
    generator_info = data["generator"]
    var = data["variations"][0]
    
    # Assert dimensions
    assert var["width"] == 128, f"Expected width 128, got {var['width']}"
    assert var["height"] == 128, f"Expected height 128, got {var['height']}"
    
    img_bytes = base64.b64decode(var["image_base64"])
    img = Image.open(io.BytesIO(img_bytes))
    assert img.size == (128, 128), f"Expected PIL size (128, 128), got {img.size}"
    
    out_path = f"verification_hd_samples/api_composite_profile_{idx}_seed_{p['seed']}.png"
    img.save(out_path)
    
    img_np = np.array(img, dtype=np.float32)
    lap_var = compute_laplacian_variance(img_np)
    sobel_mean = compute_sobel_energy(img_np)
    
    result = {
        "profile": p["name"],
        "seed": p["seed"],
        "width": img.width,
        "height": img.height,
        "checkpoint": generator_info["checkpoint_path"],
        "checkpoint_sha256": generator_info["checkpoint_sha256"],
        "laplacian_variance": round(lap_var, 2),
        "sobel_gradient_magnitude": round(sobel_mean, 2),
        "saved_path": out_path
    }
    results.append(result)
    print(f"[{idx}/3] {p['name']}: {img.size} | LapVar: {lap_var:.2f} | SobelMean: {sobel_mean:.2f} | Checkpoint: {os.path.basename(generator_info['checkpoint_path'])}")

with open("verification_hd_samples/live_api_generation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAll 3 profiles generated successfully via live FastAPI service!")
