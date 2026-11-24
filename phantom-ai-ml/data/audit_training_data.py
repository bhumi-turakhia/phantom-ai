import json
from collections import Counter, defaultdict
import numpy as np

def audit_dataset(path: str):
    samples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"==================================================")
    print(f"AUDIT REPORT FOR: {path}")
    print(f"Total samples: {len(samples)}")
    print(f"==================================================")

    cat_counts = Counter()
    certainty_counts = Counter()
    sample_certainty_distribution = Counter()
    lengths_words = []
    lengths_chars = []
    prefixes = Counter()
    vocab = Counter()
    co_occur = defaultdict(Counter)
    negation_keywords = ["no ", "not ", "clean-shaven", "never", "without", "none", "neither"]
    negated_sample_count = 0

    for s in samples:
        text = s["witness_text"]
        words = text.split()
        lengths_words.append(len(words))
        lengths_chars.append(len(text))
        for w in words:
            vocab[w.lower().strip(".,!?:;\"()")] += 1

        prefix = " ".join(words[:4]) if len(words) >= 4 else text
        prefixes[prefix] += 1

        if any(neg in text.lower() for neg in negation_keywords):
            negated_sample_count += 1

        sample_certainties = set()
        sample_cats = set()
        for ann in s.get("annotations", []):
            cert = ann.get("certainty", "unknown")
            certainty_counts[cert] += 1
            sample_certainties.add(cert)
            if cert != "unknown" and ann.get("source_span"):
                cat = ann["category"]
                cat_counts[cat] += 1
                sample_cats.add(cat)

        for cert in sample_certainties:
            sample_certainty_distribution[cert] += 1

        for c1 in sample_cats:
            for c2 in sample_cats:
                if c1 != c2:
                    co_occur[c1][c2] += 1

    print("\n--- Active Category Span Counts ---")
    for cat, count in cat_counts.most_common():
        print(f"  {cat:<22}: {count:>4} spans ({count/len(samples):.2f}/sample)")

    print("\n--- Annotation Certainty Breakdown (Total Annotations) ---")
    for cert, count in certainty_counts.most_common():
        print(f"  {cert:<15}: {count:>5}")

    print("\n--- Samples Containing Certainty Types ---")
    for cert, count in sample_certainty_distribution.most_common():
        print(f"  {cert:<15}: {count:>5} ({count/len(samples)*100:.1f}%)")

    print(f"\nSamples with negation cues: {negated_sample_count} ({negated_sample_count/len(samples)*100:.1f}%)")

    print("\n--- Sentence Length Statistics (Words) ---")
    print(f"  Mean:   {np.mean(lengths_words):.1f}")
    print(f"  Median: {np.median(lengths_words):.1f}")
    print(f"  Min:    {np.min(lengths_words)}")
    print(f"  Max:    {np.max(lengths_words)}")
    print(f"  Std:    {np.std(lengths_words):.1f}")

    print("\n--- Top 12 Repeated Sentence Structure Prefixes ---")
    for prefix, count in prefixes.most_common(12):
        pct = (count / len(samples)) * 100
        print(f"  [{count:>3} samples | {pct:4.1f}%]: {prefix}...")

    print(f"\n--- Vocabulary Overview ---")
    print(f"  Vocabulary Size (unique lower-cased tokens): {len(vocab)}")
    print(f"  Total Token Count: {sum(vocab.values())}")
    print(f"  Type-Token Ratio (TTR): {len(vocab)/sum(vocab.values()):.4f}")
    print(f"  Top 15 Most Common Tokens:")
    for token, freq in vocab.most_common(15):
        print(f"    {token:<15}: {freq}")

    print("\n--- Attribute Co-occurrence (Top pairs) ---")
    seen_pairs = set()
    pairs = []
    for c1, neighbors in co_occur.items():
        for c2, count in neighbors.items():
            pair_key = tuple(sorted([c1, c2]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                pairs.append((pair_key, count))
    pairs.sort(key=lambda x: x[1], reverse=True)
    for (c1, c2), count in pairs[:10]:
        print(f"  {c1} + {c2}: {count}")

    return {
        "total_samples": len(samples),
        "cat_counts": dict(cat_counts),
        "certainty_counts": dict(certainty_counts),
        "lengths_words": {
            "mean": float(np.mean(lengths_words)),
            "median": float(np.median(lengths_words)),
            "min": int(np.min(lengths_words)),
            "max": int(np.max(lengths_words)),
            "std": float(np.std(lengths_words)),
        },
        "vocab_size": len(vocab),
        "total_tokens": sum(vocab.values()),
        "prefixes": dict(prefixes.most_common(12)),
        "negated_samples": negated_sample_count,
    }

if __name__ == "__main__":
    audit_dataset("phantom-ai-ml/datasets/nlp/train.jsonl")

