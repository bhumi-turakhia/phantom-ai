"""
Generator Dataset Partitioner: Deterministic Identity-Disjoint Partitioning of CelebA Split A into Split A1 (80% Train) and Split A2 (20% Validation).
Ensures zero identity overlap between generator training and validation sets, and preserves complete isolation of Split B and Split C.
"""

import os
import json
import hashlib
from typing import Dict, List, Any
from pathlib import Path


CELEBA_DIR = "phantom-ai-ml/datasets/celeba"
MAIN_MANIFEST_PATH = os.path.join(CELEBA_DIR, "identity_split_manifest.json")
GEN_MANIFEST_OUT = os.path.join(CELEBA_DIR, "generator_split_a_manifest.json")


def hash_split_a_identity(identity_id: str) -> str:
    """
    Deterministically partitions a Split A identity into A1 (Train, 80%) or A2 (Val, 20%).
    Buckets [0..79] -> split_a1
    Buckets [80..99] -> split_a2
    """
    canonical_id = str(identity_id).strip()
    h = hashlib.sha256(f"generator_split_a_partition_{canonical_id}".encode("utf-8")).hexdigest()
    bucket = int(h, 16) % 100
    if bucket < 80:
        return "split_a1"
    else:
        return "split_a2"


def partition_split_a(
    main_manifest_path: str = MAIN_MANIFEST_PATH,
    output_path: str = GEN_MANIFEST_OUT,
) -> Dict[str, Any]:
    if not os.path.exists(main_manifest_path):
        raise FileNotFoundError(f"Main manifest not found at {main_manifest_path}")

    with open(main_manifest_path, "r", encoding="utf-8") as f:
        main_manifest = json.load(f)

    image_assignments = main_manifest["image_assignments"]
    image_to_identity = main_manifest["image_to_identity"]

    # Filter only Split A images and identities
    split_a_images = [img for img, sp in image_assignments.items() if sp == "split_a"]
    split_a_identities = sorted(list({image_to_identity[img] for img in split_a_images}))

    identity_assignments: Dict[str, str] = {}
    a1_identities = []
    a2_identities = []

    for ident in split_a_identities:
        sub_split = hash_split_a_identity(ident)
        identity_assignments[ident] = sub_split
        if sub_split == "split_a1":
            a1_identities.append(ident)
        else:
            a2_identities.append(ident)

    gen_image_assignments: Dict[str, str] = {}
    a1_images = []
    a2_images = []

    for img in sorted(split_a_images):
        ident = image_to_identity[img]
        sub_split = identity_assignments[ident]
        gen_image_assignments[img] = sub_split
        if sub_split == "split_a1":
            a1_images.append(img)
        else:
            a2_images.append(img)

    # Verification checks
    set_a1 = set(a1_identities)
    set_a2 = set(a2_identities)
    overlap = set_a1.intersection(set_a2)
    assert len(overlap) == 0, f"Identity leakage between Split A1 and A2: {len(overlap)}"

    # Check isolation from Split B and Split C
    split_b_images = [img for img, sp in image_assignments.items() if sp == "split_b"]
    split_b_identities = {image_to_identity[img] for img in split_b_images}
    split_c_images = [img for img, sp in image_assignments.items() if sp == "split_c"]
    split_c_identities = {image_to_identity[img] for img in split_c_images}

    leak_a1_b = set_a1.intersection(split_b_identities)
    leak_a2_b = set_a2.intersection(split_b_identities)
    leak_a1_c = set_a1.intersection(split_c_identities)
    leak_a2_c = set_a2.intersection(split_c_identities)

    assert len(leak_a1_b) == 0, f"Identity leakage between Split A1 and B: {len(leak_a1_b)}"
    assert len(leak_a2_b) == 0, f"Identity leakage between Split A2 and B: {len(leak_a2_b)}"
    assert len(leak_a1_c) == 0, f"Identity leakage between Split A1 and C: {len(leak_a1_c)}"
    assert len(leak_a2_c) == 0, f"Identity leakage between Split A2 and C: {len(leak_a2_c)}"

    total_identities = len(split_a_identities)
    total_images = len(split_a_images)

    manifest_data = {
        "dataset_name": "CelebA_Split_A_Generator",
        "parent_manifest_salt": main_manifest.get("hash_salt", ""),
        "hashing_algorithm": "SHA-256",
        "hash_salt": "generator_split_a_partition_",
        "total_split_a_identities": total_identities,
        "total_split_a_images": total_images,
        "partitions": {
            "split_a1": {
                "purpose": "generator_training",
                "identity_count": len(a1_identities),
                "image_count": len(a1_images),
                "identity_proportion": round(len(a1_identities) / total_identities, 4),
                "image_proportion": round(len(a1_images) / total_images, 4),
            },
            "split_a2": {
                "purpose": "generator_validation",
                "identity_count": len(a2_identities),
                "image_count": len(a2_images),
                "identity_proportion": round(len(a2_identities) / total_identities, 4),
                "image_proportion": round(len(a2_images) / total_images, 4),
            },
        },
        "identity_assignments": identity_assignments,
        "image_assignments": gen_image_assignments,
        "image_to_identity": {img: image_to_identity[img] for img in split_a_images},
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Partitioned Split A ({total_identities} identities, {total_images} images):")
    print(f"  Split A1 (Train): {len(a1_identities)} identities ({len(a1_images)} images)")
    print(f"  Split A2 (Val):   {len(a2_identities)} identities ({len(a2_images)} images)")
    print(f"Saved manifest to {output_path}")

    return manifest_data


if __name__ == "__main__":
    partition_split_a()

