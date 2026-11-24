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
    CharVocabulary,
)
from models.nlp_model import CharAwareBiLSTMAttributeLabeler
from data.dataset_loader import DatasetLoader
from schemas.dataset import DatasetSample
from evaluation.evaluate_metrics import compute_sequence_labeling_metrics


class CharWitnessSequenceDataset(Dataset):
    """
    Dataset loader for Character-Aware BiLSTM-CRF sequence labeling.
    Provides word token IDs, character IDs per token, BIO tags, and sequence lengths.
    """
    def __init__(
        self,
        samples: List[Any],
        word_vocab: Vocabulary,
        char_vocab: CharVocabulary,
    ):
        self.data: List[Tuple[List[str], List[str]]] = []
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[List[int]], torch.Tensor, int, List[int]]:
        tokens, tags = self.data[idx]
        word_ids = self.word_vocab.encode(tokens)
        char_ids = [self.char_vocab.encode_word(t) for t in tokens]
        char_lens = [len(t) for t in tokens]
        tag_ids = [TAG2IDX.get(t, TAG2IDX["O"]) for t in tags]

        return (
            torch.tensor(word_ids, dtype=torch.long),
            char_ids,
            torch.tensor(tag_ids, dtype=torch.long),
            len(tokens),
            char_lens,
        )


def pad_char_collate_fn(
    batch: List[Tuple[torch.Tensor, List[List[int]], torch.Tensor, int, List[int]]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for character-aware batches.
    Produces:
        padded_word_ids: (B, max_word_len)
        char_tensor:     (B, max_word_len, max_char_len)
        padded_tags:     (B, max_word_len)
        word_lengths:    (B,)
        char_lengths:    (B, max_word_len)
        word_mask:       (B, max_word_len) - True for valid tokens
        char_mask:       (B, max_word_len, max_char_len) - True for valid characters
    """
    word_ids_list, char_ids_list, tags_list, word_lens, char_lens_list = zip(*batch)

    # Pad word sequences
    padded_word_ids = nn.utils.rnn.pad_sequence(word_ids_list, batch_first=True, padding_value=0)
    padded_tags = nn.utils.rnn.pad_sequence(tags_list, batch_first=True, padding_value=TAG2IDX["O"])
    word_lengths = torch.tensor(word_lens, dtype=torch.long)

    batch_size, max_word_len = padded_word_ids.size()

    # Determine max character length (bounded at minimum 3 for CNN kernel size 3)
    max_char_len = 3
    for char_seqs in char_ids_list:
        for c_seq in char_seqs:
            if len(c_seq) > max_char_len:
                max_char_len = len(c_seq)

    char_tensor = torch.zeros(batch_size, max_word_len, max_char_len, dtype=torch.long)
    char_lengths_tensor = torch.zeros(batch_size, max_word_len, dtype=torch.long)

    for b in range(batch_size):
        cur_w_len = word_lens[b]
        for w in range(cur_w_len):
            c_seq = char_ids_list[b][w]
            c_len = min(len(c_seq), max_char_len)
            if c_len > 0:
                char_tensor[b, w, :c_len] = torch.tensor(c_seq[:c_len], dtype=torch.long)
                char_lengths_tensor[b, w] = c_len

    # Masks
    word_mask = torch.arange(max_word_len).unsqueeze(0) < word_lengths.unsqueeze(1)
    char_mask = char_tensor != 0

    return (
        padded_word_ids,
        char_tensor,
        padded_tags,
        word_lengths,
        char_lengths_tensor,
        word_mask,
        char_mask,
    )


def evaluate_char_dataset(
    model: CharAwareBiLSTMAttributeLabeler,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluates loss and token-level metrics over a dataset split without gradient updates.
    Uses sequence-level CRF NLL loss.
    """
    model.eval()
    total_loss = 0.0
    all_true_tags: List[List[str]] = []
    all_pred_tags: List[List[str]] = []

    with torch.no_grad():
        for batch in loader:
            word_ids, char_ids, tags, word_lens, _, _, _ = batch
            word_ids = word_ids.to(device)
            char_ids = char_ids.to(device)
            tags = tags.to(device)
            word_lens = word_lens.to(device)

            emissions = model(word_ids, lengths=word_lens, char_ids=char_ids)

            loss = model.compute_crf_loss(emissions, tags, lengths=word_lens)
            total_loss += loss.item()

            decoded_preds = model.decode_viterbi(emissions, lengths=word_lens)

            for b in range(word_ids.size(0)):
                cur_len = word_lens[b].item()
                true_seq = [IDX2TAG[t] for t in tags[b][:cur_len].cpu().tolist()]
                pred_seq = [IDX2TAG[p] for p in decoded_preds[b]]
                all_true_tags.append(true_seq)
                all_pred_tags.append(pred_seq)

    avg_loss = total_loss / max(len(loader), 1)
    metrics = compute_sequence_labeling_metrics(all_true_tags, all_pred_tags, ignore_tag="O")
    return avg_loss, metrics


def train_char_nlp_model(
    train_path: str = "datasets/nlp/train.jsonl",
    val_path: str = "datasets/nlp/val.jsonl",
    checkpoints_dir: str = "checkpoints",
    epochs: int = 35,
    batch_size: int = 16,
    learning_rate: float = 5e-4,
    embedding_dim: int = 128,
    char_embedding_dim: int = 32,
    char_out_dim: int = 64,
    char_kernel_size: int = 3,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    patience: int = 8,
    seed: int = 42,
    checkpoint_name: str = "nlp_char_bilstm_phase3c2.pt",
    phase: str = "3C.2",
) -> Dict[str, Any]:
    """
    Trains the character-aware BiLSTM-CRF sequence labeling model from scratch.
    Adheres strictly to project constraints:
    - Zero pretrained weights or embeddings.
    - Character vocabulary built exclusively from training split.
    - Preserves Phase 3B.2 and 3C.1 checkpoints.
    """
    set_global_seed(seed)
    device = resolve_device()
    device_info = get_device_info()
    os.makedirs(checkpoints_dir, exist_ok=True)
    base_name = os.path.splitext(checkpoint_name)[0]

    print("=" * 70)
    print("PHANTOM AI 3.0 — CHARACTER-AWARE BiLSTM-CRF TRAINING INITIALIZATION")
    print(f"Device: {device_info['device_name']} ({device})")
    print(f"Training split: {train_path}")
    print(f"Validation split: {val_path}")
    print(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, "
          f"emb_dim={embedding_dim}, char_emb_dim={char_embedding_dim}, char_out_dim={char_out_dim}, "
          f"hidden_dim={hidden_dim}, layers={num_layers}, dropout={dropout}")
    print("=" * 70)

    # 1. Load Datasets
    loader_tool = DatasetLoader()
    if not os.path.exists(train_path):
        alt_train = os.path.join("phantom-ai-ml", train_path)
        alt_val = os.path.join("phantom-ai-ml", val_path)
        if os.path.exists(alt_train):
            train_path, val_path = alt_train, alt_val
        else:
            raise FileNotFoundError(f"Training dataset not found at {train_path}")

    train_samples = loader_tool.load_jsonl(train_path, validate=True)
    val_samples = loader_tool.load_jsonl(val_path, validate=True)
    print(f"Loaded {len(train_samples)} training samples and {len(val_samples)} validation samples.")

    # 2. Build Word Vocabulary from training partition
    word_vocab = Vocabulary(min_freq=1)
    train_tokens_list = [
        [t[0] for t in tokenize_with_offsets(s.witness_text)]
        for s in train_samples
    ]
    word_vocab.build_vocab(train_tokens_list)
    print(f"Built word vocabulary from training partition: {len(word_vocab):,} unique tokens.")

    # 3. Build Character Vocabulary strictly from training partition
    char_vocab = CharVocabulary()
    flat_train_tokens = [tok for sent in train_tokens_list for tok in sent]
    char_vocab.build_vocab(flat_train_tokens)
    print(f"Built character vocabulary from training partition: {len(char_vocab):,} unique characters.")

    # Save word vocabulary
    word_vocab_file = os.path.join(checkpoints_dir, f"{base_name}_word_vocab.json")
    word_vocab.save(word_vocab_file)
    print(f"Saved word vocabulary to {word_vocab_file}")

    # Save character vocabulary
    char_vocab_file = os.path.join(checkpoints_dir, f"{base_name}_char_vocab.json")
    char_vocab.save(char_vocab_file)
    print(f"Saved character vocabulary to {char_vocab_file}")

    # 4. DataLoaders
    train_dataset = CharWitnessSequenceDataset(train_samples, word_vocab, char_vocab)
    val_dataset = CharWitnessSequenceDataset(val_samples, word_vocab, char_vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_char_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_char_collate_fn,
    )

    # 5. Instantiate Character-Aware Model
    num_tags = len(TAGS)
    model = CharAwareBiLSTMAttributeLabeler(
        vocab_size=len(word_vocab),
        char_vocab_size=len(char_vocab),
        num_tags=num_tags,
        embedding_dim=embedding_dim,
        char_embedding_dim=char_embedding_dim,
        char_out_dim=char_out_dim,
        char_kernel_size=char_kernel_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        padding_idx=0,
    )
    model.init_bio_transitions(TAGS)
    model.to(device)

    param_count = model.count_parameters()
    print(f"Model initialized from scratch with {param_count:,} trainable parameters.")

    # Save configuration
    config = {
        "model_type": "CharAwareBiLSTMAttributeLabeler",
        "training_objective": "sequence_level_crf_nll",
        "phase": phase,
        "vocab_size": len(word_vocab),
        "char_vocab_size": len(char_vocab),
        "num_tags": num_tags,
        "embedding_dim": embedding_dim,
        "char_embedding_dim": char_embedding_dim,
        "char_out_dim": char_out_dim,
        "char_kernel_size": char_kernel_size,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "padding_idx": 0,
        "tags": TAGS,
        "random_initialization": True,
        "pretrained_weights": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    config_file = os.path.join(checkpoints_dir, f"{base_name}_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 6. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # 7. Training Loop with Early Stopping
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
    history: List[Dict[str, Any]] = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            word_ids, char_ids, tags, word_lens, _, _, _ = batch
            word_ids = word_ids.to(device)
            char_ids = char_ids.to(device)
            tags = tags.to(device)
            word_lens = word_lens.to(device)

            optimizer.zero_grad()

            emissions = model(word_ids, lengths=word_lens, char_ids=char_ids)
            loss = model.compute_crf_loss(emissions, tags, lengths=word_lens)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        # Validation Step
        avg_val_loss, val_metrics = evaluate_char_dataset(model, val_loader, device)
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

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            # Save character-aware model checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": len(word_vocab),
                "char_vocab_size": len(char_vocab),
                "num_tags": num_tags,
                "embedding_dim": embedding_dim,
                "char_embedding_dim": char_embedding_dim,
                "char_out_dim": char_out_dim,
                "char_kernel_size": char_kernel_size,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "val_loss": avg_val_loss,
                "val_f1": val_f1,
                "device": str(device),
                "is_trained": True,
                "model_type": "CharAwareBiLSTMAttributeLabeler",
                "training_objective": "sequence_level_crf_nll",
                "phase": phase,
                "trained_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, checkpoint_path)
            print(f"  --> Saved new best character-aware checkpoint at epoch {epoch} (Val F1: {val_f1:.4f})")
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
    history_file = os.path.join(checkpoints_dir, f"{base_name}_history.json")
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
    parser = argparse.ArgumentParser(description="Train custom Character-Aware BiLSTM-CRF sequence labeler.")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--train-path", type=str, default="datasets/nlp/train.jsonl")
    parser.add_argument("--val-path", type=str, default="datasets/nlp/val.jsonl")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt-name", type=str, default="nlp_char_bilstm_phase3c2.pt")
    args = parser.parse_args()

    train_char_nlp_model(
        train_path=args.train_path,
        val_path=args.val_path,
        checkpoints_dir=args.checkpoints_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        checkpoint_name=args.ckpt_name,
        phase="3C.2",
    )

