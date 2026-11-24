"""
CelebA Identity-Disjoint Partitioning and Forensic Attribute Audit Module.
Executes deterministic SHA-256 identity hashing and calculates exact dataset statistics.
"""

import os
import json
import hashlib
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any


CELEBA_DIR = "phantom-ai-ml/datasets/celeba"
IDENTITY_FILE = os.path.join(CELEBA_DIR, "identity_CelebA.txt")
ATTR_FILE = os.path.join(CELEBA_DIR, "list_attr_celeba.txt")
PARTITION_FILE = os.path.join(CELEBA_DIR, "list_eval_partition.txt")
MANIFEST_OUT = os.path.join(CELEBA_DIR, "identity_split_manifest.json")
STATS_OUT = os.path.join(CELEBA_DIR, "attribute_statistics.json")


def hash_identity_to_split(identity_id: str) -> str:
    """
    Deterministically maps identity string to one of 3 disjoint splits using SHA-256.
    Bucket [0..79] -> split_a (80% generator training)
    Bucket [80..89] -> split_b (10% evaluator training)
    Bucket [90..99] -> split_c (10% held-out test evaluation)
    """
    canonical_id = identity_id.strip()
    h = hashlib.sha256(f"celeba_identity_partition_{canonical_id}".encode("utf-8")).hexdigest()
    bucket = int(h, 16) % 100
    if bucket < 80:
        return "split_a"
    elif bucket < 90:
        return "split_b"
    else:
        return "split_c"


def run_partitioning_and_audit():
    print(f"Reading CelebA metadata from: {CELEBA_DIR}")

    # 1. Parse identity file
    image_to_identity: Dict[str, str] = {}
    identity_to_images: Dict[str, List[str]] = defaultdict(list)
    malformed_records: List[str] = []

    with open(IDENTITY_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 2:
                malformed_records.append(f"Line {line_num}: {line.strip()}")
                continue
            img_name, ident = parts[0], parts[1]
            image_to_identity[img_name] = ident
            identity_to_images[ident].append(img_name)

    total_images = len(image_to_identity)
    unique_identities = len(identity_to_images)
    print(f"Total valid image records: {total_images}")
    print(f"Total unique identities: {unique_identities}")
    print(f"Malformed records: {len(malformed_records)}")

    # 2. Partition identities deterministically
    identity_to_split: Dict[str, str] = {}
    split_identities: Dict[str, List[str]] = {"split_a": [], "split_b": [], "split_c": []}
    split_images: Dict[str, List[str]] = {"split_a": [], "split_b": [], "split_c": []}
    image_to_split: Dict[str, str] = {}

    for ident in sorted(identity_to_images.keys()):
        split = hash_identity_to_split(ident)
        identity_to_split[ident] = split
        split_identities[split].append(ident)
        for img in identity_to_images[ident]:
            split_images[split].append(img)
            image_to_split[img] = split

    # Verify zero identity leakage
    set_a = set(split_identities["split_a"])
    set_b = set(split_identities["split_b"])
    set_c = set(split_identities["split_c"])

    assert len(set_a.intersection(set_b)) == 0, "Identity leakage between Split A and Split B!"
    assert len(set_a.intersection(set_c)) == 0, "Identity leakage between Split A and Split C!"
    assert len(set_b.intersection(set_c)) == 0, "Identity leakage between Split B and Split C!"
    print("VERIFIED: Zero identity leakage across Split A, Split B, and Split C.")

    # 3. Read official eval partition to demonstrate official identity leakage
    official_partition: Dict[str, int] = {}
    with open(PARTITION_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                official_partition[parts[0]] = int(parts[1])

    # Check identity overlap in official partition
    official_identities: Dict[int, set] = {0: set(), 1: set(), 2: set()}
    for img, ident in image_to_identity.items():
        if img in official_partition:
            official_identities[official_partition[img]].add(ident)

    overlap_0_1 = len(official_identities[0].intersection(official_identities[1]))
    overlap_0_2 = len(official_identities[0].intersection(official_identities[2]))
    overlap_1_2 = len(official_identities[1].intersection(official_identities[2]))
    print(f"Official partition identity overlap: Train-Val: {overlap_0_1}, Train-Test: {overlap_0_2}, Val-Test: {overlap_1_2}")

    # 4. Parse attributes
    with open(ATTR_FILE, "r", encoding="utf-8") as f:
        attr_count_header = f.readline().strip()
        attr_names = f.readline().strip().split()
        image_attributes: Dict[str, Dict[str, int]] = {}
        for line in f:
            parts = line.strip().split()
            if len(parts) != len(attr_names) + 1:
                continue
            img_name = parts[0]
            vals = [int(p) for p in parts[1:]]
            image_attributes[img_name] = {attr_names[i]: vals[i] for i in range(len(attr_names))}

    print(f"Parsed {len(attr_names)} attributes for {len(image_attributes)} images.")

    # 5. Compute attribute distributions
    attr_stats = {}
    for attr in attr_names:
        pos_total = sum(1 for img in image_attributes if image_attributes[img][attr] == 1)
        neg_total = sum(1 for img in image_attributes if image_attributes[img][attr] == -1)

        pos_a = sum(1 for img in split_images["split_a"] if image_attributes[img][attr] == 1)
        pos_b = sum(1 for img in split_images["split_b"] if image_attributes[img][attr] == 1)
        pos_c = sum(1 for img in split_images["split_c"] if image_attributes[img][attr] == 1)

        attr_stats[attr] = {
            "total_positive": pos_total,
            "total_negative": neg_total,
            "prevalence": round(pos_total / total_images, 5),
            "split_a_pos": pos_a,
            "split_a_prev": round(pos_a / len(split_images["split_a"]), 5),
            "split_b_pos": pos_b,
            "split_b_prev": round(pos_b / len(split_images["split_b"]), 5),
            "split_c_pos": pos_c,
            "split_c_prev": round(pos_c / len(split_images["split_c"]), 5),
        }

    # 6. Specific co-occurrence audits
    cooccurrences = {}

    # Bald combinations
    cooccurrences["bald_and_bangs"] = sum(
        1 for img, a in image_attributes.items() if a["Bald"] == 1 and a["Bangs"] == 1
    )
    cooccurrences["bald_and_straight"] = sum(
        1 for img, a in image_attributes.items() if a["Bald"] == 1 and a["Straight_Hair"] == 1
    )
    cooccurrences["bald_and_wavy"] = sum(
        1 for img, a in image_attributes.items() if a["Bald"] == 1 and a["Wavy_Hair"] == 1
    )

    # Hair color combinations
    cooccurrences["black_and_blond"] = sum(
        1 for img, a in image_attributes.items() if a["Black_Hair"] == 1 and a["Blond_Hair"] == 1
    )
    cooccurrences["black_and_brown"] = sum(
        1 for img, a in image_attributes.items() if a["Black_Hair"] == 1 and a["Brown_Hair"] == 1
    )
    cooccurrences["brown_and_blond"] = sum(
        1 for img, a in image_attributes.items() if a["Brown_Hair"] == 1 and a["Blond_Hair"] == 1
    )
    cooccurrences["no_hair_color_labeled"] = sum(
        1 for img, a in image_attributes.items()
        if a["Black_Hair"] == -1 and a["Blond_Hair"] == -1 and a["Brown_Hair"] == -1 and a["Gray_Hair"] == -1
    )

    # Facial hair combinations
    cooccurrences["mustache_and_no_beard_pos"] = sum(
        1 for img, a in image_attributes.items() if a["Mustache"] == 1 and a["No_Beard"] == 1
    )
    cooccurrences["mustache_and_no_beard_neg"] = sum(
        1 for img, a in image_attributes.items() if a["Mustache"] == 1 and a["No_Beard"] == -1
    )
    cooccurrences["goatee_and_no_beard_pos"] = sum(
        1 for img, a in image_attributes.items() if a["Goatee"] == 1 and a["No_Beard"] == 1
    )
    cooccurrences["goatee_and_no_beard_neg"] = sum(
        1 for img, a in image_attributes.items() if a["Goatee"] == 1 and a["No_Beard"] == -1
    )

    # Age and Gray Hair
    cooccurrences["young_and_gray_hair"] = sum(
        1 for img, a in image_attributes.items() if a["Young"] == 1 and a["Gray_Hair"] == 1
    )
    cooccurrences["not_young_and_gray_hair"] = sum(
        1 for img, a in image_attributes.items() if a["Young"] == -1 and a["Gray_Hair"] == 1
    )

    # 7. Demographic Intersectional Subgroups (Male x Young x Pale_Skin)
    demographic_subgroups = {}
    for m in [1, -1]:
        for y in [1, -1]:
            for p in [1, -1]:
                m_label = "Male" if m == 1 else "Female"
                y_label = "Young" if y == 1 else "Older"
                p_label = "Pale_Skin" if p == 1 else "Non_Pale"
                name = f"{m_label}_{y_label}_{p_label}"

                total_sub = sum(
                    1 for img, a in image_attributes.items()
                    if a["Male"] == m and a["Young"] == y and a["Pale_Skin"] == p
                )
                split_c_sub = sum(
                    1 for img in split_images["split_c"]
                    if image_attributes[img]["Male"] == m
                    and image_attributes[img]["Young"] == y
                    and image_attributes[img]["Pale_Skin"] == p
                )

                demographic_subgroups[name] = {
                    "total": total_sub,
                    "split_c": split_c_sub,
                    "proportion": round(total_sub / total_images, 5),
                }

    # 8. Save Split Manifest
    manifest_data = {
        "dataset_name": "CelebA",
        "hashing_algorithm": "SHA-256",
        "hash_salt": "celeba_identity_partition_",
        "total_images": total_images,
        "unique_identities": unique_identities,
        "partitions": {
            "split_a": {
                "purpose": "generator_training",
                "identity_count": len(split_identities["split_a"]),
                "image_count": len(split_images["split_a"]),
                "identity_proportion": round(len(split_identities["split_a"]) / unique_identities, 4),
                "image_proportion": round(len(split_images["split_a"]) / total_images, 4),
            },
            "split_b": {
                "purpose": "evaluator_training_validation",
                "identity_count": len(split_identities["split_b"]),
                "image_count": len(split_images["split_b"]),
                "identity_proportion": round(len(split_identities["split_b"]) / unique_identities, 4),
                "image_proportion": round(len(split_images["split_b"]) / total_images, 4),
            },
            "split_c": {
                "purpose": "held_out_benchmark",
                "identity_count": len(split_identities["split_c"]),
                "image_count": len(split_images["split_c"]),
                "identity_proportion": round(len(split_identities["split_c"]) / unique_identities, 4),
                "image_proportion": round(len(split_images["split_c"]) / total_images, 4),
            },
        },
        "identity_assignments": identity_to_split,
        "image_assignments": image_to_split,
        "identity_to_split": identity_to_split,
        "image_to_split": image_to_split,
        "image_to_identity": image_to_identity,
    }

    with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # 9. Save Stats Report
    stats_data = {
        "attributes": attr_stats,
        "cooccurrences": cooccurrences,
        "demographic_subgroups": demographic_subgroups,
    }

    with open(STATS_OUT, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)

    print(f"Manifest written to {MANIFEST_OUT}")
    print(f"Attribute stats written to {STATS_OUT}")
    return manifest_data, stats_data


if __name__ == "__main__":
    run_partitioning_and_audit()
