import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.device import resolve_device
from utils.seed import set_global_seed
from utils.preprocessing import (
    TAGS,
    TAG2IDX,
    IDX2TAG,
    tokenize_with_offsets,
    generate_bio_tags_from_spans,
    Vocabulary,
)
from models.nlp_model import BiLSTMAttributeLabeler
from data.dataset_loader import DatasetLoader
from training.train_nlp import WitnessSequenceDataset
from evaluation.evaluate_metrics import compute_sequence_labeling_metrics, compute_exact_span_metrics
from evaluation.evaluate_nlp import extract_spans_from_bio_tags


def pad_collate_fn_fixed(batch):
    sequences, tags, lengths = zip(*batch)
    padded_seqs = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_tags = nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=-100)
    return padded_seqs, padded_tags, torch.tensor(lengths, dtype=torch.long)


def build_bio_transition_mask(tags: list) -> torch.Tensor:
    """
    Constructs a strict transition mask enforcing valid BIO transitions:
    - O -> I-CAT is forbidden (-10000.0)
    - B-CAT1 -> I-CAT2 (CAT1 != CAT2) is forbidden (-10000.0)
    - I-CAT1 -> I-CAT2 (CAT1 != CAT2) is forbidden (-10000.0)
    """
    num_tags = len(tags)
    mask = torch.zeros(num_tags, num_tags)
    for i, from_tag in enumerate(tags):
        for j, to_tag in enumerate(tags):
            if to_tag.startswith("I-"):
                to_cat = to_tag[2:]
                if from_tag == "O":
                    mask[i, j] = -10000.0
                elif from_tag.startswith("B-") or from_tag.startswith("I-"):
                    from_cat = from_tag[2:]
                    if from_cat != to_cat:
                        mask[i, j] = -10000.0
    return mask


def test_val_experiment():
    set_global_seed(42)
    device = resolve_device()

    train_path = "datasets/nlp/train.jsonl"
    val_path = "datasets/nlp/val.jsonl"
    if not os.path.exists(train_path):
        train_path = os.path.join("phantom-ai-ml", train_path)
        val_path = os.path.join("phantom-ai-ml", val_path)

    loader = DatasetLoader()
    train_samples = loader.load_jsonl(train_path, validate=True)
    val_samples = loader.load_jsonl(val_path, validate=True)

    # Build vocab on train only
    vocab = Vocabulary(min_freq=1)
    train_tokenized_sentences = [
        tokenize_with_offsets(s.witness_text) for s in train_samples
    ]
    vocab.build_vocab([[t[0] for t in tokens] for tokens in train_tokenized_sentences])

    train_ds = WitnessSequenceDataset(train_samples, vocab)
    val_ds = WitnessSequenceDataset(val_samples, vocab)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=pad_collate_fn_fixed)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=pad_collate_fn_fixed)

    model = BiLSTMAttributeLabeler(
        vocab_size=len(vocab),
        num_tags=len(TAGS),
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        padding_idx=0,
    ).to(device)

    # Set transition mask
    trans_mask = build_bio_transition_mask(TAGS).to(device)
    with torch.no_grad():
        model.transitions.copy_(trans_mask)

    # FIXED: ignore_index=-100 so genuine "O" (0) IS trained!
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)

    print("Training on train.jsonl (385 samples) with ignore_index=-100...")
    for epoch in range(1, 16):
        model.train()
        total_loss = 0.0
        for seqs, tags, lengths in train_loader:
            seqs, tags = seqs.to(device), tags.to(device)
            optimizer.zero_grad()
            emissions = model(seqs, lengths=lengths)
            loss = criterion(emissions.view(-1, len(TAGS)), tags.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate on validation set
        model.eval()
        all_true_tags = []
        all_pred_tags = []
        val_gt_spans = []
        val_pred_spans = []

        with torch.no_grad():
            for s in val_samples:
                tok_offsets = tokenize_with_offsets(s.witness_text)
                tokens, true_tags = generate_bio_tags_from_spans(s.witness_text, s.annotations, tokens_with_offsets=tok_offsets)
                all_true_tags.append(true_tags)
                for ann in s.annotations:
                    if ann.certainty != "unknown" and ann.start_char is not None:
                        val_gt_spans.append((s.sample_id, ann.start_char, ann.end_char, str(ann.category).lower()))

                inp = torch.tensor([vocab.encode(tokens)], device=device)
                lens = torch.tensor([len(tokens)])
                emissions = model(inp, lengths=lens)
                decoded = model.decode_viterbi(emissions, lengths=lens)[0]
                pred_tags = [IDX2TAG[t] for t in decoded]
                all_pred_tags.append(pred_tags)

                m_spans = extract_spans_from_bio_tags(tok_offsets, pred_tags, sample_id=s.sample_id)
                val_pred_spans.extend(m_spans)

        token_m = compute_sequence_labeling_metrics(all_true_tags, all_pred_tags, ignore_tag="O")
        span_m = compute_exact_span_metrics(val_gt_spans, val_pred_spans)
        if epoch % 3 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Val Token F1: {token_m['f1']:.4f} (P: {token_m['precision']:.4f}, R: {token_m['recall']:.4f}) | Val Exact-Span F1: {span_m['f1']:.4f} (P: {span_m['precision']:.4f}, R: {span_m['recall']:.4f}) | TP={span_m['tp']}, FP={span_m['fp']}, FN={span_m['fn']}")


if __name__ == "__main__":
    test_val_experiment()

