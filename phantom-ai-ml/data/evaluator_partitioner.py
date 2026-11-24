"""
Evaluator Dataset Partitioner: Deterministic Identity-Disjoint Partitioning of CelebA Split B into Split B1 (80% Train) and Split B2 (20% Validation).
Ensures zero identity overlap between evaluator training and validation sets, and preserves complete isolation of Split C.
"""

import os
import json
import hashlib
from typing import Dict, List, Any


CELEBA_DIR = "phantom-ai-ml/datasets/celeba"
MAIN_MANIFEST_PATH = os.path.join(CELEBA_DIR, "identity_split_manifest.json")
EVAL_MANIFEST_OUT = os.path.join(CELEBA_DIR, "evaluator_split_b_manifest.json")


def hash_split_b_identity(identity_id: str) -> str:
    """
    Deterministically partitions a Split B identity into B1 (Train, 80%) or B2 (Val, 20%).
    Buckets [0..79] -> split_b1
    Buckets [80..99] -> split_b2
    """
    canonical_id = identity_id.strip()
    h = hashlib.sha256(f"evaluator_split_b_partition_{canonical_id}".encode("utf-8")).hexdigest()
    bucket = int(h, 16) % 100
    if bucket < 80:
        return "split_b1"
    else:
        return "split_b2"


def partition_split_b(
    main_manifest_path: str = MAIN_MANIFEST_PATH,
    output_path: str = EVAL_MANIFEST_OUT,
) -> Dict[str, Any]:
    with open(main_manifest_path, "r", encoding="utf-8") as f:
        main_manifest = json.load(f)

    image_assignments = main_manifest["image_assignments"]
    image_to_identity = main_manifest["image_to_identity"]

    # Filter only Split B images and identities
    split_b_images = [img for img, sp in image_assignments.items() if sp == "split_b"]
    split_b_identities = sorted(list({image_to_identity[img] for img in split_b_images}))

    identity_assignments: Dict[str, str] = {}
    b1_identities = []
    b2_identities = []

    for ident in split_b_identities:
        sub_split = hash_split_b_identity(ident)
        identity_assignments[ident] = sub_split
        if sub_split == "split_b1":
            b1_identities.append(ident)
        else:
            b2_identities.append(ident)

    eval_image_assignments: Dict[str, str] = {}
    b1_images = []
    b2_images = []

    for img in sorted(split_b_images):
        ident = image_to_identity[img]
        sub_split = identity_assignments[ident]
        eval_image_assignments[img] = sub_split
        if sub_split == "split_b1":
            b1_images.append(img)
        else:
            b2_images.append(img)

    # Verification checks
    set_b1 = set(b1_identities)
    set_b2 = set(b2_identities)
    overlap = set_b1.intersection(set_b2)
    assert len(overlap) == 0, f"Identity leakage between Split B1 and B2: {len(overlap)}"

    # Check isolation from Split C
    split_c_images = [img for img, sp in image_assignments.items() if sp == "split_c"]
    split_c_identities = {image_to_identity[img] for img in split_c_images}
    leak_b1_c = set_b1.intersection(split_c_identities)
    leak_b2_c = set_b2.intersection(split_c_identities)
    assert len(leak_b1_c) == 0, f"Identity leakage between Split B1 and C: {len(leak_b1_c)}"
    assert len(leak_b2_c) == 0, f"Identity leakage between Split B2 and C: {len(leak_b2_c)}"

    total_identities = len(split_b_identities)
    total_images = len(split_b_images)

    manifest_data = {
        "dataset_name": "CelebA_Split_B_Forensic_Evaluator",
        "parent_manifest_salt": main_manifest.get("hash_salt", ""),
        "hashing_algorithm": "SHA-256",
        "hash_salt": "evaluator_split_b_partition_",
        "total_split_b_identities": total_identities,
        "total_split_b_images": total_images,
        "partitions": {
            "split_b1": {
                "purpose": "evaluator_training",
                "identity_count": len(b1_identities),
                "image_count": len(b1_images),
                "identity_proportion": round(len(b1_identities) / total_identities, 4),
                "image_proportion": round(len(b1_images) / total_images, 4),
            },
            "split_b2": {
                "purpose": "evaluator_validation_checkpoint_selection",
                "identity_count": len(b2_identities),
                "image_count": len(b2_images),
                "identity_proportion": round(len(b2_identities) / total_identities, 4),
                "image_proportion": round(len(b2_images) / total_images, 4),
            },
        },
        "identity_assignments": identity_assignments,
        "image_assignments": eval_image_assignments,
        "image_to_identity": {img: image_to_identity[img] for img in split_b_images},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Evaluator Split B manifest written to: {output_path}")
    print(f"Split B1 (Train): {len(b1_identities)} identities ({len(b1_identities)/total_identities*100:.2f}%), {len(b1_images)} images ({len(b1_images)/total_images*100:.2f}%)")
    print(f"Split B2 (Val):   {len(b2_identities)} identities ({len(b2_identities)/total_identities*100:.2f}%), {len(b2_images)} images ({len(b2_images)/total_images*100:.2f}%)")
    print("VERIFIED: Zero identity leakage between Split B1 and B2, and zero leakage into Split C.")

    return manifest_data


if __name__ == "__main__":
    partition_split_b()

