import os
import json
import time
import argparse
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.device import resolve_device, get_device_info
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
from schemas.dataset import DatasetSample
from evaluation.evaluate_metrics import compute_sequence_labeling_metrics


class WitnessSequenceDataset(Dataset):
    """
    Dataset loader for witness statement attribute sequence labeling.
    Compatible with both raw (tokens, tags) tuples and DatasetSample objects.
    """
    def __init__(self, samples: List[Any], vocab: Vocabulary):
        self.data: List[Tuple[List[str], List[str]]] = []
        self.vocab = vocab

        for s in samples:
            if isinstance(s, DatasetSample):
                tokens_with_offsets = tokenize_with_offsets(s.witness_text)
                tokens, tags = generate_bio_tags_from_spans(
                    s.witness_text,
                    s.annotations,
                    tokens_with_offsets=tokens_with_offsets,
                )
                self.data.append((tokens, tags))
            elif isinstance(s, dict) and "tokens" in s and "bio_tags" in s:
                self.data.append((s["tokens"], s["bio_tags"]))
            elif isinstance(s, tuple) and len(s) == 2:
                self.data.append(s)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        tokens, tags = self.data[idx]
        input_ids = self.vocab.encode(tokens)
        tag_ids = [TAG2IDX.get(t, TAG2IDX["O"]) for t in tags]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(tag_ids, dtype=torch.long),
            len(tokens),
        )


def pad_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, tags, lengths = zip(*batch)
    padded_seqs = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    # Use -100 for padding tags so index 0 ('O' tag) is actively trained and penalized
    padded_tags = nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=-100)
    return padded_seqs, padded_tags, torch.tensor(lengths, dtype=torch.long)


def evaluate_dataset(
    model: BiLSTMAttributeLabeler,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluates CRF sequence-level NLL loss and token-level metrics over a dataset
    split without gradient updates. Uses model.compute_crf_loss() so that
    evaluation loss is consistent with the training objective.
    """
    model.eval()
    total_loss = 0.0
    all_true_tags: List[List[str]] = []
    all_pred_tags: List[List[str]] = []

    with torch.no_grad():
        for seqs, tags, lengths in loader:
            seqs, tags = seqs.to(device), tags.to(device)
            emissions = model(seqs, lengths=lengths)

            # Sequence-level CRF NLL — consistent with training objective
            loss = model.compute_crf_loss(emissions, tags, lengths=lengths)
            total_loss += loss.item()

            # Viterbi decoding using learned CRF transitions
            decoded_preds = model.decode_viterbi(emissions, lengths=lengths)

            for b in range(seqs.size(0)):
                cur_len = lengths[b].item()
                true_seq = [IDX2TAG[t] for t in tags[b][:cur_len].cpu().tolist()]
                pred_seq = [IDX2TAG[p] for p in decoded_preds[b]]
                all_true_tags.append(true_seq)
                all_pred_tags.append(pred_seq)

    avg_loss = total_loss / max(len(loader), 1)
    metrics = compute_sequence_labeling_metrics(all_true_tags, all_pred_tags, ignore_tag="O")
    return avg_loss, metrics


def train_nlp_model(
    train_path: str = "datasets/nlp/train.jsonl",
    val_path: str = "datasets/nlp/val.jsonl",
    checkpoints_dir: str = "checkpoints",
    epochs: int = 25,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    patience: int = 6,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Trains the custom BiLSTM-CRF sequence labeling model from scratch.
    Adheres strictly to the rule: NO PRETRAINED WEIGHTS OR EMBEDDINGS.
    Uses training data exclusively for gradient updates.
    Uses validation data exclusively for model selection and early stopping.
    """
    set_global_seed(seed)
    device = resolve_device()
    device_info = get_device_info()
    os.makedirs(checkpoints_dir, exist_ok=True)

    print("=" * 70)
    print("PHANTOM AI 3.0 — BiLSTM-CRF TRAINING INITIALIZATION")
    print(f"Device: {device_info['device_name']} ({device})")
    print(f"Training split: {train_path}")
    print(f"Validation split: {val_path}")
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, "
          f"emb_dim={embedding_dim}, hidden_dim={hidden_dim}, layers={num_layers}, dropout={dropout}")
    print("=" * 70)

    # 1. Load Training and Validation Datasets
    loader_tool = DatasetLoader()
    if not os.path.exists(train_path):
        # Fallback to local path if in subfolder
        alt_train = os.path.join("phantom-ai-ml", train_path)
        alt_val = os.path.join("phantom-ai-ml", val_path)
        if os.path.exists(alt_train):
            train_path, val_path = alt_train, alt_val
        else:
            raise FileNotFoundError(f"Training dataset not found at {train_path}")

    train_samples = loader_tool.load_jsonl(train_path, validate=True)
    val_samples = loader_tool.load_jsonl(val_path, validate=True)

    print(f"Loaded {len(train_samples)} training samples and {len(val_samples)} validation samples.")

    # 2. Build Vocabulary strictly from training samples
    vocab = Vocabulary(min_freq=1)
    train_tokenized_sentences = [
        tokenize_with_offsets(s.witness_text) for s in train_samples
    ]
    vocab.build_vocab([[t[0] for t in tokens] for tokens in train_tokenized_sentences])
    print(f"Built vocabulary from training partition: {len(vocab):,} unique tokens.")

    # Save vocabulary
    vocab_file = os.path.join(checkpoints_dir, "vocab.json")
    vocab.save(vocab_file)
    print(f"Saved vocabulary to {vocab_file}")

    # 3. Create PyTorch DataLoaders
    train_dataset = WitnessSequenceDataset(train_samples, vocab)
    val_dataset = WitnessSequenceDataset(val_samples, vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
    )

    # 4. Instantiate Model Architecture (Random Initialization Only)
    num_tags = len(TAGS)
    model = BiLSTMAttributeLabeler(
        vocab_size=len(vocab),
        num_tags=num_tags,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        padding_idx=0,
    )
    # Initialize BIO transition constraints (forbid O -> I, forbid B/I-CAT1 -> I-CAT2)
    model.init_bio_transitions(TAGS)
    model.to(device)

    param_count = model.count_parameters()
    print(f"Model initialized from scratch with {param_count:,} trainable parameters.")

    # Save model configuration
    config = {
        "model_type": "BiLSTMAttributeLabeler",
        "training_objective": "sequence_level_crf_nll",
        "training_objective_note": (
            "Genuine Linear-Chain CRF trained with sequence-level negative log-likelihood. "
            "Loss = log Z(x) - Score(x, y). CRF transition, start_transition, and end_transition "
            "parameters all receive gradients via the forward algorithm."
        ),
        "vocab_size": len(vocab),
        "num_tags": num_tags,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "padding_idx": 0,
        "tags": TAGS,
        "categories": [cat.replace("B-", "") for cat in TAGS if cat.startswith("B-")],
        "random_initialization": True,
        "pretrained_weights": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    config_file = os.path.join(checkpoints_dir, "model_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 5. Optimizer — loss is computed directly via model.compute_crf_loss()
    # Genuine sequence-level CRF NLL: Loss = log Z(x) - Score(x, y)
    # This ensures CRF transition parameters (self.transitions, self.start_transitions,
    # self.end_transitions) receive gradients via backprop through log Z(x).
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # Dummy criterion kept for signature compatibility with evaluate_dataset (unused there too)
    criterion = None

    # 6. Training Loop with Early Stopping
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = os.path.join(checkpoints_dir, "nlp_bilstm.pt")
    history: List[Dict[str, Any]] = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for seqs, tags, lengths in train_loader:
            seqs, tags = seqs.to(device), tags.to(device)
            optimizer.zero_grad()

            emissions = model(seqs, lengths=lengths)
            # Genuine CRF sequence-level NLL — trains transitions jointly with emissions
            loss = model.compute_crf_loss(emissions, tags, lengths=lengths)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        # Validation Step — also uses CRF NLL for consistency
        avg_val_loss, val_metrics = evaluate_dataset(model, val_loader, criterion, device)
        val_f1 = val_metrics["f1"]

        scheduler.step(val_f1)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} (P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f})"
        )

        # Early Stopping & Checkpoint Saving
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            # Save genuine CRF-trained checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": len(vocab),
                "num_tags": num_tags,
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "val_loss": avg_val_loss,
                "val_f1": val_f1,
                "device": str(device),
                "is_trained": True,
                "training_objective": "sequence_level_crf_nll",
                "phase": "3B.2",
                "trained_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, checkpoint_path)
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Val F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    total_duration = round(time.time() - start_time, 2)
    print("=" * 70)
    print(f"Training completed in {total_duration}s. Best Epoch: {best_epoch} with Val F1: {best_val_f1:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print("=" * 70)

    # Save training history
    history_file = os.path.join(checkpoints_dir, "training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_duration_seconds": total_duration,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "device": str(device),
            "epochs_run": len(history),
            "history": history,
        }, f, indent=2)

    return {
        "checkpoint_path": checkpoint_path,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "total_duration": total_duration,
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom BiLSTM-CRF sequence labeler from scratch.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train-path", type=str, default="datasets/nlp/train.jsonl")
    parser.add_argument("--val-path", type=str, default="datasets/nlp/val.jsonl")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    train_nlp_model(
        train_path=args.train_path,
        val_path=args.val_path,
        checkpoints_dir=args.checkpoints_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
    )
