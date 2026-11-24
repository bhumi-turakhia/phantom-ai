"""
Builds and validates the Phase 3C.2 Multi-Register Expanded Dataset.
Combines original 385 train + 82 val with 1,650 new multi-register samples (1,400 train + 250 val).
Total: ~1,785 training samples, ~332 validation samples.
Preserves test.jsonl and out_of_frame_eval.jsonl completely untouched.
"""

import hashlib
import json
import os
import shutil
from typing import List, Set
from data.generate_expanded_multiregister import generate_expanded_dataset
from data.validator import DatasetValidator
from schemas.dataset import DatasetSample

DATASET_DIR = "phantom-ai-ml/datasets/nlp"
TRAIN_PATH = os.path.join(DATASET_DIR, "train.jsonl")
VAL_PATH = os.path.join(DATASET_DIR, "val.jsonl")
TEST_PATH = os.path.join(DATASET_DIR, "test.jsonl")
OOF_PATH = os.path.join(DATASET_DIR, "out_of_frame_eval.jsonl")

TRAIN_BACKUP = os.path.join(DATASET_DIR, "train_phase3b_backup.jsonl")
VAL_BACKUP = os.path.join(DATASET_DIR, "val_phase3b_backup.jsonl")

FROZEN_TEST_SHA256 = "d6787cee05be7881b08e89889d9b41abf70e836885c06405e3623c94c8da4c51"
FROZEN_OOF_SHA256 = "4cdc270629a87d1607ce85675b0a9683c1220637e01c1447e3168f218075b6e3"


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_files():
    test_hash = compute_sha256(TEST_PATH)
    oof_hash = compute_sha256(OOF_PATH)
    assert test_hash == FROZEN_TEST_SHA256, f"Frozen test set tampered! Hash: {test_hash}"
    assert oof_hash == FROZEN_OOF_SHA256, f"Frozen out-of-frame set tampered! Hash: {oof_hash}"
    print(f"Verified frozen evaluation datasets integrity:")
    print(f"  test.jsonl SHA-256:             {test_hash}")
    print(f"  out_of_frame_eval.jsonl SHA-256: {oof_hash}")


def load_samples(path: str) -> List[DatasetSample]:
    samples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(DatasetSample.model_validate_json(line))
    return samples


def save_samples(samples: List[DatasetSample], path: str):
    with open(path, "w") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")


def build_and_validate():
    # 1. Verify frozen test sets
    verify_frozen_files()

    # 2. Backup original files if not already backed up
    if not os.path.exists(TRAIN_BACKUP):
        shutil.copyfile(TRAIN_PATH, TRAIN_BACKUP)
        print(f"Backed up original train set to {TRAIN_BACKUP}")
    if not os.path.exists(VAL_BACKUP):
        shutil.copyfile(VAL_PATH, VAL_BACKUP)
        print(f"Backed up original val set to {VAL_BACKUP}")

    orig_train = load_samples(TRAIN_BACKUP)
    orig_val = load_samples(VAL_BACKUP)
    print(f"Loaded original train: {len(orig_train)}, val: {len(orig_val)}")

    # 3. Generate 1,650 new multi-register samples
    new_samples = generate_expanded_dataset(
        target_count=1650,
        seed=1337,
        existing_eval_paths=[TEST_PATH, OOF_PATH]
    )

    # 4. Validate new samples with DatasetValidator
    validator = DatasetValidator()
    print("Validating new 1,650 samples with DatasetValidator...")
    val_summary = validator.validate_dataset(new_samples)
    print(f"New samples validation: is_valid={val_summary.is_valid}, errors={len(val_summary.errors)}, warnings={len(val_summary.warnings)}")
    if not val_summary.is_valid:
        raise ValueError(f"Validation failed on new samples: {val_summary.errors[:5]}")

    # 5. Split new samples: 1,400 for training, 250 for validation
    new_train = new_samples[:1400]
    new_val = new_samples[1400:1650]

    # Combine with original samples
    combined_train = orig_train + new_train
    combined_val = orig_val + new_val

    # 6. Check for split leakage across ALL splits
    train_texts = {s.witness_text.strip().lower() for s in combined_train}
    val_texts = {s.witness_text.strip().lower() for s in combined_val}
    test_texts = {s.witness_text.strip().lower() for s in load_samples(TEST_PATH)}
    oof_texts = {s.witness_text.strip().lower() for s in load_samples(OOF_PATH)}

    train_val_overlap = train_texts.intersection(val_texts)
    train_test_overlap = train_texts.intersection(test_texts)
    train_oof_overlap = train_texts.intersection(oof_texts)
    val_test_overlap = val_texts.intersection(test_texts)
    val_oof_overlap = val_texts.intersection(oof_texts)

    assert len(train_val_overlap) == 0, f"Train-Val leakage: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Train-Test leakage: {train_test_overlap}"
    assert len(train_oof_overlap) == 0, f"Train-OOF leakage: {train_oof_overlap}"
    assert len(val_test_overlap) == 0, f"Val-Test leakage: {val_test_overlap}"
    assert len(val_oof_overlap) == 0, f"Val-OOF leakage: {val_oof_overlap}"
    print("STRICT SPLIT LEAKAGE CHECK: PASSED (Zero overlap between train, val, test, and out_of_frame).")

    # 7. Write updated train and val files
    save_samples(combined_train, TRAIN_PATH)
    save_samples(combined_val, VAL_PATH)
    print(f"Saved expanded train dataset ({len(combined_train)} samples) to {TRAIN_PATH}")
    print(f"Saved expanded val dataset ({len(combined_val)} samples) to {VAL_PATH}")

    # 8. Re-validate combined datasets
    train_summary = validator.validate_dataset(combined_train)
    val_summary = validator.validate_dataset(combined_val)
    assert train_summary.is_valid, f"Train dataset invalid: {train_summary.errors[:5]}"
    assert val_summary.is_valid, f"Val dataset invalid: {val_summary.errors[:5]}"
    print("Full combined train & val dataset validation: 100% VALID.")

    # 9. Verify frozen files again
    verify_frozen_files()

    print("\n================ DATASET EXPANSION SUMMARY ================")
    print(f"Original Train:     {len(orig_train)}")
    print(f"New Train:          {len(new_train)}")
    print(f"Total Train:        {len(combined_train)}")
    print(f"Original Val:       {len(orig_val)}")
    print(f"New Val:            {len(new_val)}")
    print(f"Total Val:          {len(combined_val)}")
    print(f"Frozen Test:        83 (Untouched, SHA-256: {compute_sha256(TEST_PATH)})")
    print(f"Frozen OOF Eval:    106 (Untouched, SHA-256: {compute_sha256(OOF_PATH)})")
    print(f"Train File SHA-256: {compute_sha256(TRAIN_PATH)}")
    print(f"Val File SHA-256:   {compute_sha256(VAL_PATH)}")
    print("===========================================================")


if __name__ == "__main__":
    build_and_validate()

