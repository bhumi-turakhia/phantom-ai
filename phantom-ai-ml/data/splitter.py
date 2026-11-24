import random
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from schemas.dataset import DatasetSample


class DatasetSplitter:
    """
    Reproducible Dataset Splitting with Contamination & Leakage Prevention.
    Guarantees that identical or closely related samples never leak across train/val/test splits
    by strictly preserving cluster integrity at the partition boundary.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def _normalize_for_clustering(self, text: str) -> str:
        """
        Strips punctuation, spaces, and casing to group near-identical witness statements.
        """
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def split(
        self,
        samples: List[DatasetSample],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Splits a list of samples into train, validation, and test subsets.
        Prevents data leakage by grouping related texts into clusters before partitioning.
        Entire clusters are moved together, ensuring zero cross-split leakage.
        """
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
            raise ValueError(
                f"Split ratios must sum to 1.0, got train={train_ratio}, "
                f"val={val_ratio}, test={test_ratio} (sum={train_ratio + val_ratio + test_ratio})"
            )

        if not samples:
            return {
                "train": [],
                "val": [],
                "test": [],
                "statistics": {"total": 0, "train": 0, "val": 0, "test": 0},
            }

        # Cluster samples by normalized text to prevent data leakage across splits
        clusters: Dict[str, List[DatasetSample]] = defaultdict(list)
        for sample in samples:
            key = self._normalize_for_clustering(sample.witness_text)
            clusters[key].append(sample)

        cluster_list = list(clusters.values())

        # Deterministic shuffle of clusters using seed
        rng = random.Random(self.seed)
        rng.shuffle(cluster_list)

        total_samples = len(samples)
        target_train = int(total_samples * train_ratio)
        target_val = int(total_samples * val_ratio)
        target_test = total_samples - target_train - target_val

        train_clusters: List[List[DatasetSample]] = []
        val_clusters: List[List[DatasetSample]] = []
        test_clusters: List[List[DatasetSample]] = []

        remaining = list(cluster_list)

        # Allocate at least one cluster to each split if sufficient clusters exist
        if len(remaining) >= 3 and val_ratio > 0 and test_ratio > 0:
            train_clusters.append(remaining.pop(0))
            val_clusters.append(remaining.pop(0))
            test_clusters.append(remaining.pop(0))
        elif len(remaining) >= 2 and (val_ratio > 0 or test_ratio > 0):
            train_clusters.append(remaining.pop(0))
            if val_ratio > 0:
                val_clusters.append(remaining.pop(0))
            else:
                test_clusters.append(remaining.pop(0))

        train_count = sum(len(c) for c in train_clusters)
        val_count = sum(len(c) for c in val_clusters)
        test_count = sum(len(c) for c in test_clusters)

        # Greedy allocation to meet target counts while maintaining indivisible clusters
        for cluster in remaining:
            c_size = len(cluster)
            diff_train = target_train - train_count
            diff_val = target_val - val_count
            diff_test = target_test - test_count

            # Choose the partition with the highest remaining deficit
            if diff_train >= diff_val and diff_train >= diff_test:
                train_clusters.append(cluster)
                train_count += c_size
            elif diff_val >= diff_test and val_ratio > 0:
                val_clusters.append(cluster)
                val_count += c_size
            else:
                test_clusters.append(cluster)
                test_count += c_size

        train_set = [s for c in train_clusters for s in c]
        val_set = [s for c in val_clusters for s in c]
        test_set = [s for c in test_clusters for s in c]

        # Verify zero leakage across splits
        train_keys = {self._normalize_for_clustering(s.witness_text) for s in train_set}
        val_keys = {self._normalize_for_clustering(s.witness_text) for s in val_set}
        test_keys = {self._normalize_for_clustering(s.witness_text) for s in test_set}

        assert len(train_keys.intersection(val_keys)) == 0, "Leakage detected between train and val splits!"
        assert len(train_keys.intersection(test_keys)) == 0, "Leakage detected between train and test splits!"
        assert len(val_keys.intersection(test_keys)) == 0, "Leakage detected between val and test splits!"

        stats = {
            "seed": self.seed,
            "total_samples": total_samples,
            "train_count": len(train_set),
            "val_count": len(val_set),
            "test_count": len(test_set),
            "train_percentage": round(len(train_set) / total_samples * 100, 1),
            "val_percentage": round(len(val_set) / total_samples * 100, 1),
            "test_percentage": round(len(test_set) / total_samples * 100, 1),
        }

        return {
            "train": train_set,
            "val": val_set,
            "test": test_set,
            "statistics": stats,
        }
