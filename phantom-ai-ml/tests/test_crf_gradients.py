import pytest
import torch
import torch.nn as nn
from models.nlp_model import BiLSTMAttributeLabeler
from utils.preprocessing import TAGS, TAG2IDX

def test_crf_transitions_receive_gradients():
    """
    Programmatic test verifying that CRF transitions, start_transitions,
    and end_transitions strictly receive finite, non-zero gradients on a variable-length batch.
    """
    model = BiLSTMAttributeLabeler(
        vocab_size=100,
        num_tags=len(TAGS),
        embedding_dim=32,
        hidden_dim=64,
        num_layers=1,
    )
    model.train()

    # Batch of 3 sequences with variable lengths
    seqs = torch.randint(0, 100, (3, 6))
    lengths = torch.tensor([6, 4, 3], dtype=torch.long)
    tags = torch.zeros(3, 6, dtype=torch.long)
    tags[0, :6] = torch.tensor([TAG2IDX["O"], TAG2IDX["B-HAIR"], TAG2IDX["I-HAIR"], TAG2IDX["O"], TAG2IDX["B-NOSE"], TAG2IDX["O"]])
    tags[1, :4] = torch.tensor([TAG2IDX["B-EYES"], TAG2IDX["I-EYES"], TAG2IDX["O"], TAG2IDX["O"]])
    tags[2, :3] = torch.tensor([TAG2IDX["O"], TAG2IDX["B-FACIAL_HAIR"], TAG2IDX["O"]])

    emissions = model(seqs, lengths=lengths)
    loss = model.compute_crf_loss(emissions, tags, lengths=lengths)

    assert torch.isfinite(loss), f"CRF loss must be finite, got {loss.item()}"
    assert loss.item() >= 0, f"CRF NLL must be non-negative, got {loss.item()}"

    loss.backward()

    # Verify transition parameters receive finite non-zero gradients
    assert model.transitions.grad is not None, "transitions.grad is None"
    assert torch.all(torch.isfinite(model.transitions.grad)), "transitions.grad has non-finite values"
    assert torch.any(model.transitions.grad != 0), "transitions.grad is all zeros"

    assert model.start_transitions.grad is not None, "start_transitions.grad is None"
    assert torch.all(torch.isfinite(model.start_transitions.grad)), "start_transitions.grad has non-finite values"
    assert torch.any(model.start_transitions.grad != 0), "start_transitions.grad is all zeros"

    assert model.end_transitions.grad is not None, "end_transitions.grad is None"
    assert torch.all(torch.isfinite(model.end_transitions.grad)), "end_transitions.grad has non-finite values"
    assert torch.any(model.end_transitions.grad != 0), "end_transitions.grad is all zeros"

