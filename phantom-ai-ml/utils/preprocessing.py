import re
from typing import List, Dict, Tuple, Optional, Any
from schemas.attributes import AttributeCategory, EvidenceState


# Observable forensic facial attribute tag categories (10 observable non-demographic categories)
ATTRIBUTE_CATEGORIES: List[str] = [
    "face_shape",
    "hair",
    "hairline",
    "eyes",
    "eyebrows",
    "nose",
    "lips",
    "facial_hair",
    "distinguishing_marks",
    "other_observable",
]

# Standard BIO Scheme sequence labeling tags
TAGS: List[str] = ["O"]
for cat in ATTRIBUTE_CATEGORIES:
    TAGS.append(f"B-{cat.upper()}")
    TAGS.append(f"I-{cat.upper()}")

TAG2IDX: Dict[str, int] = {tag: idx for idx, tag in enumerate(TAGS)}
IDX2TAG: Dict[int, str] = {idx: tag for idx, tag in enumerate(TAGS)}


def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokenizes text into lowercase tokens while preserving exact character offsets
    (start_char, end_char) in the original verbatim witness text.
    Punctuation characters are split as individual tokens to match standard NLP sequence labeling.
    Hyphenated compound descriptors (e.g. 'heart-shaped', 'clean-shaven') are preserved as single units.
    """
    tokens_with_offsets: List[Tuple[str, int, int]] = []
    # Pattern: words (including hyphens within words) or single non-whitespace punctuation characters
    token_pattern = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[^\s\w]")
    for match in token_pattern.finditer(text):
        token_str = match.group(0).lower()
        tokens_with_offsets.append((token_str, match.start(), match.end()))
    return tokens_with_offsets


def simple_tokenize(text: str) -> List[str]:
    """
    Splits text into lowercase word and punctuation tokens.
    Guaranteed to match the token strings produced by tokenize_with_offsets.
    """
    return [t[0] for t in tokenize_with_offsets(text)]


def generate_bio_tags_from_spans(
    text: str,
    annotations: List[Any],
    tokens_with_offsets: Optional[List[Tuple[str, int, int]]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Converts verbatim witness text and character span annotations into aligned
    token sequences and BIO sequence tags strictly compatible with BiLSTMAttributeLabeler.

    Rules:
    - UNKNOWN attributes have no span and remain 'O'.
    - Tokens within an active span receive 'B-CAT' (first token) or 'I-CAT' (subsequent).
    - Tokens outside any span receive 'O'.
    """
    if tokens_with_offsets is None:
        tokens_with_offsets = tokenize_with_offsets(text)

    tokens = [t[0] for t in tokens_with_offsets]
    tags = ["O"] * len(tokens)

    # Filter annotations that have valid character spans and are neither unknown nor negated
    valid_annotations = [
        ann for ann in annotations
        if getattr(ann, "certainty", None) not in (
            EvidenceState.UNKNOWN.value,
            "unknown",
            EvidenceState.NEGATED.value,
            "negated",
        )
        and getattr(ann, "start_char", None) is not None
        and getattr(ann, "end_char", None) is not None
        and getattr(ann, "start_char", 0) < getattr(ann, "end_char", 0)
    ]

    # Sort annotations by start_char
    valid_annotations.sort(key=lambda a: a.start_char)

    for ann in valid_annotations:
        cat_str = (
            ann.category.value
            if isinstance(ann.category, AttributeCategory)
            else str(ann.category).lower()
        )
        if cat_str not in ATTRIBUTE_CATEGORIES:
            continue

        b_tag = f"B-{cat_str.upper()}"
        i_tag = f"I-{cat_str.upper()}"

        first_token = True
        for idx, (_, tok_start, tok_end) in enumerate(tokens_with_offsets):
            # Check overlap between token [tok_start, tok_end] and annotation [ann.start_char, ann.end_char]
            if max(tok_start, ann.start_char) < min(tok_end, ann.end_char):
                if first_token:
                    tags[idx] = b_tag
                    first_token = False
                else:
                    tags[idx] = i_tag

    return tokens, tags


class Vocabulary:
    """
    Vocabulary mapping for custom BiLSTM sequence labeling models.
    """
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
        }
        self.idx2word: Dict[int, str] = {idx: w for w, idx in self.word2idx.items()}
        self.frequencies: Dict[str, int] = {}

    def build_vocab(self, sentences: List[List[str]]) -> None:
        for tokens in sentences:
            for token in tokens:
                self.frequencies[token] = self.frequencies.get(token, 0) + 1

        for token, count in self.frequencies.items():
            if count >= self.min_freq and token not in self.word2idx:
                new_idx = len(self.word2idx)
                self.word2idx[token] = new_idx
                self.idx2word[new_idx] = token

    def __len__(self) -> int:
        return len(self.word2idx)

    def encode(self, tokens: List[str]) -> List[int]:
        unk_idx = self.word2idx[self.UNK_TOKEN]
        return [self.word2idx.get(t, unk_idx) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2word.get(i, self.UNK_TOKEN) for i in indices]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word2idx": self.word2idx,
            "min_freq": self.min_freq,
            "frequencies": self.frequencies,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Vocabulary":
        vocab = cls(min_freq=data.get("min_freq", 1))
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
        vocab.frequencies = data.get("frequencies", {})
        return vocab

    def save(self, file_path: str) -> None:
        import json
        import os
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, file_path: str) -> "Vocabulary":
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class CharVocabulary:
    """
    Character-level vocabulary for character-aware sequence labeling models.
    Built exclusively from training partition tokens.
    Provides subword and morphological character indices for unknown word handling.
    """
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOW_TOKEN = "<BOW>"
    EOW_TOKEN = "<EOW>"

    def __init__(self):
        self.char2idx: Dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOW_TOKEN: 2,
            self.EOW_TOKEN: 3,
        }
        self.idx2char: Dict[int, str] = {idx: c for c, idx in self.char2idx.items()}
        self.frequencies: Dict[str, int] = {}

    def build_vocab(self, tokens_or_words: List[str]) -> None:
        """
        Populates character vocabulary strictly from a collection of tokens.
        """
        for token in tokens_or_words:
            for ch in str(token):
                self.frequencies[ch] = self.frequencies.get(ch, 0) + 1

        for ch in sorted(self.frequencies.keys()):
            if ch not in self.char2idx:
                new_idx = len(self.char2idx)
                self.char2idx[ch] = new_idx
                self.idx2char[new_idx] = ch

    def __len__(self) -> int:
        return len(self.char2idx)

    def encode_word(self, word: str, max_char_len: Optional[int] = None) -> List[int]:
        """
        Encodes a single word string into character IDs.
        Unseen characters map to UNK_TOKEN. Empty words produce an empty or padded list.
        """
        unk_idx = self.char2idx[self.UNK_TOKEN]
        chars = [self.char2idx.get(c, unk_idx) for c in str(word)]
        if max_char_len is not None:
            if len(chars) > max_char_len:
                chars = chars[:max_char_len]
            elif len(chars) < max_char_len:
                chars = chars + [self.char2idx[self.PAD_TOKEN]] * (max_char_len - len(chars))
        return chars

    def encode_tokens(
        self, tokens: List[str], max_char_len: Optional[int] = None
    ) -> List[List[int]]:
        """
        Encodes a list of tokens into lists of character IDs.
        """
        return [self.encode_word(t, max_char_len=max_char_len) for t in tokens]

    def decode_word(self, indices: List[int]) -> str:
        """
        Decodes a list of character indices back to string, omitting padding.
        """
        pad_idx = self.char2idx[self.PAD_TOKEN]
        return "".join(
            self.idx2char.get(i, self.UNK_TOKEN)
            for i in indices
            if i != pad_idx
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "char2idx": self.char2idx,
            "frequencies": self.frequencies,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharVocabulary":
        vocab = cls()
        vocab.char2idx = data["char2idx"]
        vocab.idx2char = {int(v) if isinstance(v, (int, str)) and str(v).isdigit() else v: k
                          for k, v in vocab.char2idx.items()}
        # Re-index properly
        vocab.idx2char = {v: k for k, v in vocab.char2idx.items()}
        vocab.frequencies = data.get("frequencies", {})
        return vocab

    def save(self, file_path: str) -> None:
        import json
        import os
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, file_path: str) -> "CharVocabulary":
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def build_char_ids_for_tokens(
    tokens: List[str],
    char_vocab: CharVocabulary,
    max_char_len: Optional[int] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Utility producing character ID sequences and lengths for a tokenized sentence.
    Returns: (char_ids_per_token, char_lengths)
    """
    char_ids = [char_vocab.encode_word(t, max_char_len=max_char_len) for t in tokens]
    char_lengths = [len(t) for t in tokens]
    return char_ids, char_lengths
