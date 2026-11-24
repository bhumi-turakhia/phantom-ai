"""
CelebA Identity-Disjoint PyTorch Dataset Loader.
Integrates identity-disjoint splits, GCIT 20-dimensional conditioning, polarity handling,
configurable attribute dropout, and deterministic preprocessing.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.gcit_translator import GCIT_LAYOUT, GCIT_POLARITY, AnnotationPolarity


# Exact mapping from 20 GCIT dimension keys to CelebA attribute column names
GCIT_TO_CELEBA_ATTR: Dict[str, str] = {
    # Hair Color (Indices 0..4)
    "hair_black": "Black_Hair",
    "hair_brown": "Brown_Hair",
    "hair_blond": "Blond_Hair",
    "hair_gray": "Gray_Hair",
    "hair_bald": "Bald",
    # Hair Structure (Indices 5..8)
    "hair_bangs": "Bangs",
    "hair_straight": "Straight_Hair",
    "hair_wavy": "Wavy_Hair",
    "hairline_receding": "Receding_Hairline",
    # Facial Hair (Indices 9..13)
    "facial_mustache": "Mustache",
    "facial_goatee": "Goatee",
    "facial_sideburns": "Sideburns",
    "facial_stubble": "5_o_Clock_Shadow",
    "facial_no_beard": "No_Beard",  # Polarity is INVERTED (+1 = absent, -1 = present)
    # Facial Geometry (Indices 14..18)
    "nose_big": "Big_Nose",
    "nose_pointy": "Pointy_Nose",
    "lips_big": "Big_Lips",
    "face_oval": "Oval_Face",
    "face_cheekbones": "High_Cheekbones",
    # Accessories (Index 19)
    "glasses": "Eyeglasses",
}


class PreprocessingError(Exception):
    """Raised when an image file cannot be parsed or transformed correctly."""
    pass


class CelebAIdentityDataset(Dataset):
    """
    PyTorch Dataset providing identity-disjoint samples for Conditional WGAN-GP and Evaluator.

    Returns for each sample:
      - 'image': Tensor (3, image_size, image_size) normalized to [-1.0, 1.0].
      - 'condition': Tensor (20,) representing GCIT values in [-1.0, 1.0] (0.0 when dropped/unobserved).
      - 'mask': Tensor (20,) in {0.0, 1.0} indicating actively conditioned dimensions.
      - 'identity': String identity identifier.
      - 'image_name': String filename (e.g. '000001.jpg').
    """

    def __init__(
        self,
        metadata_dir: str = "phantom-ai-ml/datasets/celeba",
        split: str = "split_a",
        image_dir: Optional[str] = None,
        image_size: int = 64,
        attribute_dropout: float = 0.0,
        transform: Optional[Any] = None,
        require_images: bool = False,
    ):
        """
        Args:
            metadata_dir: Directory containing identity_split_manifest.json and list_attr_celeba.txt.
            split: One of 'split_a' (80% generator train), 'split_b' (10% evaluator train), 'split_c' (10% held-out test).
            image_dir: Optional path to directory with CelebA images (e.g. img_align_celeba).
                       If None and require_images=False, dummy tensors (3, image_size, image_size) are returned.
            image_size: Spatial target resolution (default: 64x64).
            attribute_dropout: Probability p in [0.0, 1.0] of dropping an observed attribute during training.
            transform: Optional custom torchvision transform pipeline.
            require_images: If True, enforces strict image file presence. Zero-image dummy fallback is blocked.
        """
        super().__init__()
        valid_splits = {"split_a", "split_b", "split_c", "split_b1", "split_b2", "split_a1", "split_a2"}
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Expected one of {valid_splits}")

        if not (0.0 <= attribute_dropout <= 1.0):
            raise ValueError(f"attribute_dropout must be in [0.0, 1.0], got {attribute_dropout}")

        self.metadata_dir = Path(metadata_dir)
        self.split = split
        self.image_dir = Path(image_dir) if image_dir else None
        self.image_size = image_size
        self.attribute_dropout = attribute_dropout
        self.transform = transform
        self.require_images = require_images

        if self.require_images:
            if self.image_dir is None:
                raise ValueError("require_images=True is set for training, but image_dir was not provided.")
            if not self.image_dir.exists():
                raise FileNotFoundError(f"require_images=True but image_dir does not exist: {self.image_dir}")

        # Load appropriate manifest
        if split in {"split_b1", "split_b2"}:
            manifest_path = self.metadata_dir / "evaluator_split_b_manifest.json"
        elif split in {"split_a1", "split_a2"}:
            manifest_path = self.metadata_dir / "generator_split_a_manifest.json"
        else:
            manifest_path = self.metadata_dir / "identity_split_manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. Please run partitioner first."
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Build list of images and identity mapping for this split
        image_assignments = manifest["image_assignments"]
        image_to_identity = manifest.get("image_to_identity")

        if not image_to_identity:
            # Fallback to reading identity_CelebA.txt
            identity_file = self.metadata_dir / "identity_CelebA.txt"
            image_to_identity = {}
            if identity_file.exists():
                with open(identity_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            image_to_identity[parts[0]] = parts[1]

        self.image_names: List[str] = [
            img for img, sp in sorted(image_assignments.items()) if sp == split
        ]
        self.image_to_identity: Dict[str, str] = {
            img: image_to_identity[img] for img in self.image_names
        }

        # Preload GCIT-mapped attributes for this split
        attr_file = self.metadata_dir / "list_attr_celeba.txt"
        if not attr_file.exists():
            raise FileNotFoundError(f"Attribute file not found at {attr_file}")

        self.attributes: torch.Tensor = self._load_and_map_attributes(attr_file)

    def _load_and_map_attributes(self, attr_file: Path) -> torch.Tensor:
        """
        Parses list_attr_celeba.txt and maps each image to a 20-dim tensor according to GCIT_LAYOUT.
        """
        split_set = set(self.image_names)
        raw_attrs: Dict[str, List[float]] = {}

        with open(attr_file, "r", encoding="utf-8") as f:
            f.readline()  # Total count line
            col_names = f.readline().split()
            col_to_idx = {name: i for i, name in enumerate(col_names)}

            # Validate all required GCIT columns exist
            for gcit_dim, celeba_col in GCIT_TO_CELEBA_ATTR.items():
                if celeba_col not in col_to_idx:
                    raise KeyError(f"Expected CelebA attribute '{celeba_col}' not found in header.")

            # Map from GCIT dimension index (0..19) to CelebA line token index
            gcit_index_to_col_idx: List[int] = [0] * len(GCIT_LAYOUT)
            for gcit_dim, dim_idx in GCIT_LAYOUT.items():
                celeba_col = GCIT_TO_CELEBA_ATTR[gcit_dim]
                gcit_index_to_col_idx[dim_idx] = col_to_idx[celeba_col]

            for line in f:
                parts = line.split()
                if not parts:
                    continue
                img_name = parts[0]
                if img_name in split_set:
                    # Values start at parts[1:]
                    vec = [0.0] * len(GCIT_LAYOUT)
                    for dim_idx, celeba_token_idx in enumerate(gcit_index_to_col_idx):
                        val_str = parts[1 + celeba_token_idx]
                        # CelebA raw values are +1 or -1
                        vec[dim_idx] = 1.0 if val_str == "1" else -1.0
                    raw_attrs[img_name] = vec

        # Convert to tensor ordered matching self.image_names
        attr_matrix = torch.empty((len(self.image_names), len(GCIT_LAYOUT)), dtype=torch.float32)
        for i, img_name in enumerate(self.image_names):
            attr_matrix[i] = torch.tensor(raw_attrs[img_name], dtype=torch.float32)

        return attr_matrix

    def __len__(self) -> int:
        return len(self.image_names)

    def _load_image(self, img_name: str) -> torch.Tensor:
        """
        Loads, resizes, and normalizes an image to [-1.0, 1.0].
        If image file is not available, returns a deterministic zero tensor.
        """
        if self.image_dir is not None:
            img_path = self.image_dir / img_name
            if img_path.exists():
                last_err = None
                for attempt in range(3):
                    try:
                        with Image.open(img_path) as pil_img:
                            pil_img = pil_img.convert("RGB")
                            if self.transform is not None:
                                return self.transform(pil_img)
                            # For 128x128+, apply face-centered square crop (178x178) to eliminate aspect ratio distortion
                            # and use bicubic resampling to preserve sharp facial geometry.
                            if self.image_size >= 128:
                                w, h = pil_img.size
                                crop_size = min(w, h)
                                top = (h - crop_size) // 2
                                box = (0, top, crop_size, top + crop_size)
                                pil_img = pil_img.crop(box).resize(
                                    (self.image_size, self.image_size),
                                    resample=Image.Resampling.BICUBIC,
                                )
                            else:
                                # Deterministic 64x64 baseline resize
                                pil_img = pil_img.resize(
                                    (self.image_size, self.image_size),
                                    resample=Image.Resampling.BILINEAR,
                                )
                            arr = np.array(pil_img, dtype=np.uint8)
                            tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # [0, 1]
                            tensor = (tensor - 0.5) / 0.5  # [-1, 1]
                            return tensor
                    except Exception as e:
                        last_err = e
                        import time
                        time.sleep(0.05)
                raise PreprocessingError(f"Failed to process image '{img_path}': {str(last_err)}")
            elif self.require_images:
                raise FileNotFoundError(
                    f"require_images=True: Image '{img_name}' not found in '{self.image_dir}'."
                )
        elif self.require_images:
            raise ValueError("require_images=True: image_dir is None; actual image files are required.")

        # Baseline fallback when images are not downloaded (e.g. metadata-only Phase 5.1 verification)
        return torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_name = self.image_names[idx]
        identity = self.image_to_identity[img_name]
        image_tensor = self._load_image(img_name)

        # Baseline condition vector from dataset (all values +1.0 or -1.0)
        condition = self.attributes[idx].clone()
        mask = torch.ones(len(GCIT_LAYOUT), dtype=torch.float32)

        # Apply attribute dropout if requested
        if self.attribute_dropout > 0.0:
            dropout_mask = torch.rand(len(GCIT_LAYOUT)) >= self.attribute_dropout
            # For dropped dimensions, set mask to 0.0 and condition value to 0.0
            condition = condition * dropout_mask.float()
            mask = mask * dropout_mask.float()

        return {
            "image": image_tensor,
            "condition": condition,
            "mask": mask,
            "identity": identity,
            "image_name": img_name,
        }

    def get_split_summary(self) -> Dict[str, Any]:
        """Returns statistics for this split."""
        unique_ids = set(self.image_to_identity.values())
        return {
            "split": self.split,
            "image_count": len(self.image_names),
            "identity_count": len(unique_ids),
            "image_size": self.image_size,
            "attribute_dropout": self.attribute_dropout,
            "has_images": self.image_dir is not None and self.image_dir.exists(),
        }
