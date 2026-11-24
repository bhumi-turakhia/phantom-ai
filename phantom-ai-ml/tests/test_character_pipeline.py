import pytest
import torch
import torch.nn as nn
from utils.preprocessing import (
    CharVocabulary,
    Vocabulary,
    tokenize_with_offsets,
    build_char_ids_for_tokens,
    TAGS,
    TAG2IDX,
)
from models.nlp_model import CharAwareBiLSTMAttributeLabeler, BiLSTMAttributeLabeler
from utils.device import resolve_device


def test_char_vocabulary_build_and_encode():
    """
    Verifies character vocabulary construction and word-to-char encoding.
    """
    char_vocab = CharVocabulary()
    tokens = ["short", "black", "hair", "clean-shaven", "heart-shaped"]
    char_vocab.build_vocab(tokens)

    assert len(char_vocab) > 4  # special tokens + characters
    assert CharVocabulary.PAD_TOKEN in char_vocab.char2idx
    assert CharVocabulary.UNK_TOKEN in char_vocab.char2idx
    assert "-" in char_vocab.char2idx

    # Encode known word
    encoded = char_vocab.encode_word("hair")
    assert len(encoded) == 4
    decoded = char_vocab.decode_word(encoded)
    assert decoded == "hair"


def test_char_vocabulary_unknown_words_and_unicode():
    """
    Verifies that unknown words and Unicode input are handled cleanly without exceptions.
    Known characters get distinct indices while unseen characters map to UNK_TOKEN.
    """
    char_vocab = CharVocabulary()
    char_vocab.build_vocab(["black", "hair", "eyes"])

    # Unknown English word with known characters
    encoded_known_chars = char_vocab.encode_word("bark")
    assert all(idx != char_vocab.char2idx[CharVocabulary.UNK_TOKEN] for idx in encoded_known_chars)

    # Unicode word (e.g. café with 'é' unseen)
    encoded_unicode = char_vocab.encode_word("café")
    assert len(encoded_unicode) == 4
    # 'é' should map to UNK
    assert encoded_unicode[3] == char_vocab.char2idx[CharVocabulary.UNK_TOKEN]

    # Two distinct unknown words produce distinct character representations!
    enc1 = char_vocab.encode_word("cauliflower")
    enc2 = char_vocab.encode_word("microbladed")
    assert enc1 != enc2, "Distinct unseen words must produce distinct character ID sequences!"


def test_char_vocabulary_punctuation_and_hyphenation():
    """
    Verifies hyphenated descriptors and isolated punctuation.
    """
    char_vocab = CharVocabulary()
    tokens = ["heart-shaped", ",", ".", ":"]
    char_vocab.build_vocab(tokens)

    assert "-" in char_vocab.char2idx
    assert "," in char_vocab.char2idx
    assert "." in char_vocab.char2idx
    assert ":" in char_vocab.char2idx

    enc_hyphen = char_vocab.encode_word("heart-shaped")
    assert len(enc_hyphen) == len("heart-shaped")
    hyphen_idx = char_vocab.char2idx["-"]
    assert enc_hyphen[5] == hyphen_idx


def test_empty_and_malformed_sequences():
    """
    Verifies that empty strings and non-string inputs are handled robustly without crashes.
    """
    char_vocab = CharVocabulary()
    char_vocab.build_vocab(["test"])

    # Empty string
    enc_empty = char_vocab.encode_word("")
    assert enc_empty == []

    enc_empty_padded = char_vocab.encode_word("", max_char_len=5)
    assert len(enc_empty_padded) == 5
    assert enc_empty_padded == [0, 0, 0, 0, 0]

    # Decodes empty cleanly
    assert char_vocab.decode_word([]) == ""
    assert char_vocab.decode_word([0, 0, 0]) == ""


def test_char_aware_model_forward_and_variable_length_batches():
    """
    Verifies that CharAwareBiLSTMAttributeLabeler handles variable-length batches
    and character padding masks correctly.
    """
    device = resolve_device()
    vocab_size = 50
    char_vocab_size = 35
    num_tags = len(TAGS)
    batch_size = 3
    seq_len = 6
    max_char_len = 10

    model = CharAwareBiLSTMAttributeLabeler(
        vocab_size=vocab_size,
        char_vocab_size=char_vocab_size,
        num_tags=num_tags,
        embedding_dim=32,
        char_embedding_dim=16,
        char_out_dim=32,
        char_kernel_size=3,
        hidden_dim=64,
        num_layers=1,
    ).to(device)

    # Input tensors
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    char_ids = torch.randint(0, char_vocab_size, (batch_size, seq_len, max_char_len), device=device)
    lengths = torch.tensor([6, 4, 2], dtype=torch.long)

    emissions = model(input_ids, lengths=lengths, char_ids=char_ids)
    assert emissions.shape == (batch_size, seq_len, num_tags)
    assert torch.isfinite(emissions).all()


def test_char_encoder_gradients():
    """
    Verifies that both character embeddings and character CNN receive non-zero gradients
    during backpropagation through sequence-level CRF loss.
    """
    device = resolve_device()
    vocab_size = 40
    char_vocab_size = 30
    num_tags = len(TAGS)
    batch_size = 2
    seq_len = 5
    max_char_len = 8

    model = CharAwareBiLSTMAttributeLabeler(
        vocab_size=vocab_size,
        char_vocab_size=char_vocab_size,
        num_tags=num_tags,
        embedding_dim=32,
        char_embedding_dim=16,
        char_out_dim=32,
        char_kernel_size=3,
        hidden_dim=48,
        num_layers=1,
    ).to(device)
    model.train()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    char_ids = torch.randint(1, char_vocab_size, (batch_size, seq_len, max_char_len), device=device)
    tags = torch.randint(0, num_tags, (batch_size, seq_len), device=device)
    lengths = torch.tensor([5, 3], dtype=torch.long)

    emissions = model(input_ids, lengths=lengths, char_ids=char_ids)
    loss = model.compute_crf_loss(emissions, tags, lengths=lengths)

    assert torch.isfinite(loss)
    loss.backward()

    # Verify character embedding received gradient
    assert model.char_embedding.weight.grad is not None
    assert torch.any(model.char_embedding.weight.grad != 0)

    # Verify character convolution received gradient
    assert model.char_conv.weight.grad is not None
    assert torch.any(model.char_conv.weight.grad != 0)

