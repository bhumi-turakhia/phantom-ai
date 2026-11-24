import json
import os
from typing import List, Dict, Any, Tuple, Optional
from schemas.dataset import DatasetSample, ForensicDataset
from utils.preprocessing import (
    tokenize_with_offsets,
    generate_bio_tags_from_spans,
    Vocabulary,
)
from data.validator import DatasetValidator


class DatasetLoader:
    """
    Dataset loader, exporter, and sequence converter for forensic witness datasets.
    Preserves original witness descriptions verbatim and exports training-ready BIO format.
    """

    def __init__(self, validator: Optional[DatasetValidator] = None):
        self.validator = validator or DatasetValidator()

    def load_json(self, file_path: str, validate: bool = True) -> ForensicDataset:
        """
        Loads a ForensicDataset from a JSON file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = ForensicDataset.model_validate(data)

        if validate:
            summary = self.validator.validate_dataset(dataset.samples)
            if not summary.is_valid:
                error_preview = "; ".join(summary.errors[:5])
                raise ValueError(
                    f"Dataset validation failed with {len(summary.errors)} error(s): {error_preview}"
                )

        return dataset

    def load_jsonl(self, file_path: str, validate: bool = True) -> List[DatasetSample]:
        """
        Loads dataset samples line-by-line from a JSONL file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        samples: List[DatasetSample] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample_dict = json.loads(line)
                    sample = DatasetSample.model_validate(sample_dict)
                    samples.append(sample)
                except Exception as e:
                    raise ValueError(f"Failed to parse JSONL line {line_num}: {e}")

        if validate:
            summary = self.validator.validate_dataset(samples)
            if not summary.is_valid:
                error_preview = "; ".join(summary.errors[:5])
                raise ValueError(
                    f"JSONL dataset validation failed with {len(summary.errors)} error(s): {error_preview}"
                )

        return samples

    def save_json(self, dataset: ForensicDataset, file_path: str, indent: int = 2) -> None:
        """
        Serializes and writes a ForensicDataset to a formatted JSON file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dataset.model_dump(), f, indent=indent, ensure_ascii=False)

    def save_jsonl(self, samples: List[DatasetSample], file_path: str) -> None:
        """
        Serializes and writes dataset samples to a JSONL file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for sample in samples:
                line = json.dumps(sample.model_dump(), ensure_ascii=False)
                f.write(line + "\n")

    def export_to_bio_format(
        self, samples: List[DatasetSample]
    ) -> List[Dict[str, Any]]:
        """
        Exports samples to training-ready sequence format:
        Returns a list of dicts:
        [
            {
                "sample_id": "SYNTH-001",
                "witness_text": "...",
                "tokens": ["the", "suspect", ...],
                "bio_tags": ["O", "O", ...],
                "offsets": [[0, 3], [4, 11], ...]
            },
            ...
        ]
        """
        exported = []
        for sample in samples:
            tokens_with_offsets = tokenize_with_offsets(sample.witness_text)
            tokens, tags = generate_bio_tags_from_spans(
                text=sample.witness_text,
                annotations=sample.annotations,
                tokens_with_offsets=tokens_with_offsets,
            )
            exported.append({
                "sample_id": sample.sample_id,
                "witness_text": sample.witness_text,
                "tokens": tokens,
                "bio_tags": tags,
                "token_offsets": [(t[1], t[2]) for t in tokens_with_offsets],
                "is_synthetic": sample.metadata.is_synthetic,
            })
        return exported

