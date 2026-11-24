import os
import sys
import json
import base64
from io import BytesIO
import urllib.request
from PIL import Image

def run_verification():
    os.makedirs("verification_samples", exist_ok=True)
    
    profiles = [
        {
            "id": 1,
            "title": "Male, Black Straight Hair, Clean-Shaven",
            "seed": 1001,
            "forensic_profile": {
                "hair": {
                    "category": "hair",
                    "state": "explicit",
                    "value": "black straight hair",
                    "confidence": 0.95,
                    "source_spans": ["black straight hair"],
                    "uncertainty_cues": []
                },
                "facial_hair": {
                    "category": "facial_hair",
                    "state": "negated",
                    "value": "clean-shaven",
                    "confidence": 0.95,
                    "source_spans": ["clean-shaven"],
                    "uncertainty_cues": []
                }
            }
        },
        {
            "id": 2,
            "title": "Female, Blond Wavy Hair, High Cheekbones, Full Lips",
            "seed": 2002,
            "forensic_profile": {
                "hair": {
                    "category": "hair",
                    "state": "explicit",
                    "value": "blond wavy hair",
                    "confidence": 0.92,
                    "source_spans": ["blond wavy hair"],
                    "uncertainty_cues": []
                },
                "face_shape": {
                    "category": "face_shape",
                    "state": "explicit",
                    "value": "high cheekbones",
                    "confidence": 0.88,
                    "source_spans": ["high cheekbones"],
                    "uncertainty_cues": []
                },
                "lips": {
                    "category": "lips",
                    "state": "explicit",
                    "value": "big lips",
                    "confidence": 0.85,
                    "source_spans": ["full lips"],
                    "uncertainty_cues": []
                }
            }
        },
        {
            "id": 3,
            "title": "Male, Gray Hair, Glasses, Mustache",
            "seed": 3003,
            "forensic_profile": {
                "hair": {
                    "category": "hair",
                    "state": "explicit",
                    "value": "gray hair",
                    "confidence": 0.90,
                    "source_spans": ["gray hair"],
                    "uncertainty_cues": []
                },
                "facial_hair": {
                    "category": "facial_hair",
                    "state": "explicit",
                    "value": "mustache",
                    "confidence": 0.92,
                    "source_spans": ["mustache"],
                    "uncertainty_cues": []
                },
                "other_observable": {
                    "category": "other_observable",
                    "state": "explicit",
                    "value": "glasses",
                    "confidence": 0.95,
                    "source_spans": ["wearing glasses"],
                    "uncertainty_cues": []
                }
            }
        }
    ]

    results = []

    for item in profiles:
        payload = {
            "forensic_profile": item["forensic_profile"],
            "seed": item["seed"],
            "num_variations": 1
        }
        
        req = urllib.request.Request(
            "http://127.0.0.1:8000/generate-composite",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\n--- Calling /generate-composite for Profile {item['id']}: {item['title']} (Seed: {item['seed']}) ---")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            
        generator_info = data.get("generator", {})
        variations = data.get("variations", [])
        
        print(f"Status: {data.get('status')}")
        print(f"Generator Model: {generator_info.get('model_name')}")
        print(f"Checkpoint Path: {generator_info.get('checkpoint_path')}")
        print(f"Checkpoint SHA-256: {generator_info.get('checkpoint_sha256')}")
        print(f"Latency: {data.get('latency_ms')} ms")
        print(f"Conditioned Attributes: {data.get('conditioned_attributes')}")
        
        if not variations:
            raise RuntimeError(f"No variations returned for Profile {item['id']}")
            
        var0 = variations[0]
        api_width = var0.get("width")
        api_height = var0.get("height")
        img_b64 = var0.get("image_base64")
        
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(BytesIO(img_bytes))
        
        out_path = f"verification_samples/sample_api_{item['id']}_seed_{item['seed']}.png"
        img.save(out_path)
        file_size = os.path.getsize(out_path)
        
        print(f"Returned API Dimensions: {api_width}x{api_height}")
        print(f"Saved Image: {out_path}")
        print(f"PIL Verified Dimensions: {img.width}x{img.height}")
        print(f"PIL Format: {img.format}, Mode: {img.mode}")
        print(f"File Size: {file_size} bytes ({file_size/1024:.2f} KB)")
        
        # Rigorous assertions
        assert img.width == 128, f"Expected width 128, got {img.width}"
        assert img.height == 128, f"Expected height 128, got {img.height}"
        assert api_width == 128, f"API width expected 128, got {api_width}"
        assert api_height == 128, f"API height expected 128, got {api_height}"
        
        results.append({
            "id": item["id"],
            "title": item["title"],
            "seed": item["seed"],
            "saved_path": out_path,
            "dimensions": f"{img.width}x{img.height}",
            "pixel_count": img.width * img.height,
            "file_size_bytes": file_size,
            "conditioned_attributes": data.get("conditioned_attributes"),
            "generator_checkpoint": os.path.basename(generator_info.get("checkpoint_path")),
            "checkpoint_sha256": generator_info.get("checkpoint_sha256")
        })

    with open("verification_samples/verification_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n================ ALL 3 INFERENCE SAMPLES SUCCESSFULLY VERIFIED ================")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_verification()

