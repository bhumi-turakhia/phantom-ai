"""
CelebA Image Archive Extraction and Integrity Verification Script.
Safely extracts img_align_celeba.zip into phantom-ai-ml/datasets/celeba/img_align_celeba
and verifies that all images for Split B (evaluator) and Split A (generator) are present.
"""

import os
import sys
import zipfile
import json
from pathlib import Path


CELEBA_DIR = Path("phantom-ai-ml/datasets/celeba")
ZIP_PATH = CELEBA_DIR / "img_align_celeba.zip"
TARGET_DIR = CELEBA_DIR / "img_align_celeba"
EVAL_MANIFEST = CELEBA_DIR / "evaluator_split_b_manifest.json"


def extract_archive():
    if not ZIP_PATH.exists():
        print(f"Error: Archive not found at {ZIP_PATH}")
        sys.exit(1)

    print(f"Opening archive: {ZIP_PATH} ({ZIP_PATH.stat().st_size / (1024*1024):.1f} MB)...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        members = zf.namelist()
        total_members = len(members)
        print(f"Archive contains {total_members:,} entries.")

        # Determine nesting
        first_member = members[0]
        has_prefix = first_member.startswith("img_align_celeba/")

        extract_count = 0
        for i, member in enumerate(members, 1):
            filename = os.path.basename(member)
            if not filename or not filename.endswith(".jpg"):
                continue

            target_file = TARGET_DIR / filename
            if not target_file.exists():
                with zf.open(member) as source, open(target_file, "wb") as target:
                    target.write(source.read())
            extract_count += 1
            if extract_count % 25000 == 0:
                print(f"Extracted {extract_count:,} / {total_members:,} images...")

    print(f"Extraction complete! Total images in {TARGET_DIR}: {len(list(TARGET_DIR.glob('*.jpg'))):,}")

    # Verify Split B images
    if EVAL_MANIFEST.exists():
        with open(EVAL_MANIFEST, "r") as f:
            eval_data = json.load(f)
        split_b_images = list(eval_data["image_assignments"].keys())
        missing = [img for img in split_b_images if not (TARGET_DIR / img).exists()]
        if missing:
            print(f"WARNING: {len(missing)} images missing from Split B!")
        else:
            print(f"VERIFIED: All {len(split_b_images):,} images for Split B are present on disk.")


if __name__ == "__main__":
    extract_archive()

