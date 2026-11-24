import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class BiLSTMAttributeLabeler(nn.Module):
    """
    Custom Bidirectional LSTM architecture with a fully trainable Linear-Chain CRF.
    Supports sequence-level negative log-likelihood (NLL) training via the forward algorithm
    and globally optimal Viterbi sequence decoding.
    Initializes exclusively with random weights (no pretrained embeddings or weights).
    """
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
        char_vocab_size: Optional[int] = None,
        char_embedding_dim: int = 32,
        char_out_dim: int = 64,
        char_kernel_size: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.char_vocab_size = char_vocab_size
        self.char_embedding_dim = char_embedding_dim
        self.char_out_dim = char_out_dim
        self.char_kernel_size = char_kernel_size
        self.use_char_encoder = char_vocab_size is not None and char_vocab_size > 0

        # Custom word embedding initialized from uniform distribution (no pretrained weights)
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # Character encoder components (if character vocabulary provided)
        if self.use_char_encoder:
            self.char_embedding = nn.Embedding(
                char_vocab_size,
                char_embedding_dim,
                padding_idx=0,
            )
            self.char_dropout = nn.Dropout(dropout)
            self.char_conv = nn.Conv1d(
                char_embedding_dim,
                char_out_dim,
                kernel_size=char_kernel_size,
                padding=char_kernel_size // 2,
            )
            lstm_input_dim = embedding_dim + char_out_dim
        else:
            lstm_input_dim = embedding_dim

        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Projection layer to tag emission scores
        # Bidirectional output dimension is hidden_dim * 2
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_tags)

        # Fully trainable Linear-Chain CRF parameters
        # transitions[i, j] represents score of transitioning from tag i to tag j
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initializes weights using Xavier uniform and normal distribution.
        """
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

        if self.use_char_encoder:
            nn.init.uniform_(self.char_embedding.weight, -0.1, 0.1)
            with torch.no_grad():
                self.char_embedding.weight[0].fill_(0)
            nn.init.kaiming_uniform_(self.char_conv.weight, nonlinearity="relu")
            if self.char_conv.bias is not None:
                nn.init.zeros_(self.char_conv.bias)

        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Set forget gate bias to 1.0 for LSTM stability
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)

        nn.init.xavier_uniform_(self.hidden2tag.weight)
        nn.init.zeros_(self.hidden2tag.bias)
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.zeros_(self.start_transitions)
        nn.init.zeros_(self.end_transitions)

    def init_bio_transitions(self, tags: Optional[List[str]] = None) -> None:
        """
        Initializes transition parameters with structural BIO sequence constraints:
        - Forbids transitioning into I-CAT from O or mismatched categories (-10000.0)
        - Forbids starting a sequence with an I- tag (-10000.0)
        """
        with torch.no_grad():
            self.transitions.fill_(0.0)
            self.start_transitions.fill_(0.0)
            self.end_transitions.fill_(0.0)

            if tags is not None and len(tags) == self.num_tags:
                for i, from_tag in enumerate(tags):
                    # Forbid starting sequence with an I- tag
                    if from_tag.startswith("I-"):
                        self.start_transitions[i] = -10000.0

                    for j, to_tag in enumerate(tags):
                        if to_tag.startswith("I-"):
                            to_cat = to_tag[2:]
                            if from_tag == "O":
                                self.transitions[i, j] = -10000.0
                            elif from_tag.startswith("B-") or from_tag.startswith("I-"):
                                from_cat = from_tag[2:]
                                if from_cat != to_cat:
                                    self.transitions[i, j] = -10000.0

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        char_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass computing tag emission scores.
        Supports both word-only mode and character-aware mode.
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            lengths: Optional Tensor containing valid sequence lengths
            char_ids: Optional Tensor of shape (batch_size, seq_len, max_char_len)
        Returns:
            emissions: Tensor of shape (batch_size, seq_len, num_tags)
        """
        word_embedded = self.embedding(input_ids)
        batch_size, seq_len, _ = word_embedded.size()

        if self.use_char_encoder:
            if char_ids is not None:
                # char_ids: (batch_size, seq_len, max_char_len)
                B, T, M = char_ids.size()
                char_flat = char_ids.view(B * T, M)
                # Embed: (B*T, M, char_emb_dim) -> transpose to (B*T, char_emb_dim, M)
                char_emb = self.char_dropout(self.char_embedding(char_flat)).transpose(1, 2)
                # Conv1d + ReLU: (B*T, char_out_dim, M)
                conv_out = torch.relu(self.char_conv(char_emb))
                # Max-pool across character sequence dimension: (B*T, char_out_dim)
                pooled, _ = torch.max(conv_out, dim=-1)
                char_repr = pooled.view(B, T, self.char_out_dim)

                # Zero out character representation for padding tokens if lengths provided
                if lengths is not None:
                    lengths_dev = lengths.to(char_repr.device)
                    word_mask = (torch.arange(T, device=char_repr.device).unsqueeze(0) < lengths_dev.unsqueeze(1)).unsqueeze(-1)
                    char_repr = char_repr * word_mask.float()
            else:
                char_repr = torch.zeros(batch_size, seq_len, self.char_out_dim, device=input_ids.device)

            combined = torch.cat([word_embedded, char_repr], dim=-1)
            embedded = self.embedding_dropout(combined)
        else:
            embedded = self.embedding_dropout(word_embedded)

        if lengths is not None:
            lengths_cpu = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embedded)

        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def compute_crf_loss(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the sequence-level negative log-likelihood (NLL) of the gold tag sequences
        using the linear-chain CRF forward algorithm and partition function Z(x):
            Loss = log Z(x) - Score(x, y)
        Guarantees that both emissions and CRF transition parameters receive gradients.
        """
        batch_size, seq_len, num_tags = emissions.size()

        if mask is None:
            if lengths is not None:
                lengths_dev = lengths.to(emissions.device)
                mask = torch.arange(seq_len, device=emissions.device).unsqueeze(0) < lengths_dev.unsqueeze(1)
            else:
                mask = tags.ne(-100)

        # 1. Forward Algorithm: Compute log Z(x)
        alpha = self.start_transitions + emissions[:, 0]  # (B, K)

        for t in range(1, seq_len):
            emit_score = emissions[:, t].unsqueeze(1)       # (B, 1, K)
            prev_alpha = alpha.unsqueeze(2)                 # (B, K, 1)
            trans_score = self.transitions.unsqueeze(0)      # (1, K, K)

            next_alpha = torch.logsumexp(prev_alpha + trans_score + emit_score, dim=1)  # (B, K)
            mask_t = mask[:, t].unsqueeze(1)
            alpha = torch.where(mask_t, next_alpha, alpha)

        final_alpha = alpha + self.end_transitions
        log_Z = torch.logsumexp(final_alpha, dim=-1)  # (B,)

        # 2. Compute Score of Gold Path: Score(x, y)
        safe_tags = tags.clamp(min=0, max=num_tags - 1)

        start_score = self.start_transitions[safe_tags[:, 0]]
        emit_scores = emissions.gather(2, safe_tags.unsqueeze(2)).squeeze(2)
        emit_score = (emit_scores * mask.float()).sum(dim=-1)

        trans_score = torch.zeros(batch_size, device=emissions.device)
        for t in range(seq_len - 1):
            t_from = safe_tags[:, t]
            t_to = safe_tags[:, t + 1]
            t_score = self.transitions[t_from, t_to]
            valid_t = (mask[:, t] & mask[:, t + 1]).float()
            trans_score = trans_score + (t_score * valid_t)

        if lengths is not None:
            last_idx = (lengths.to(emissions.device) - 1).clamp(min=0)
        else:
            last_idx = (mask.long().sum(dim=-1) - 1).clamp(min=0)

        last_tags = safe_tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        end_score = self.end_transitions[last_tags]

        gold_score = start_score + emit_score + trans_score + end_score

        # 3. Sequence NLL Loss
        nll = log_Z - gold_score
        return nll.mean()

    def decode_argmax(
        self,
        emissions: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        Greedy argmax decoding across emission scores.
        """
        predictions = emissions.argmax(dim=-1)
        batch_size = emissions.size(0)
        results = []
        for i in range(batch_size):
            seq_len = lengths[i].item() if lengths is not None else emissions.size(1)
            results.append(predictions[i, :seq_len].cpu().tolist())
        return results

    def decode_viterbi(
        self,
        emissions: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        Viterbi sequence decoding utilizing the trained CRF transition matrix,
        start_transitions, and end_transitions.
        Finds the globally optimal tag sequence: argmax_y Score(x, y).
        """
        batch_size, seq_len, num_tags = emissions.size()
        trans = self.transitions.detach()
        start_trans = self.start_transitions.detach()
        end_trans = self.end_transitions.detach()
        results = []

        for b in range(batch_size):
            cur_len = lengths[b].item() if lengths is not None else seq_len
            if cur_len <= 0:
                results.append([])
                continue

            viterbi = start_trans + emissions[b, 0]
            backpointers = []

            for t in range(1, cur_len):
                broadcast_viterbi = viterbi.unsqueeze(1).expand(num_tags, num_tags)
                scores = broadcast_viterbi + trans + emissions[b, t].unsqueeze(0)
                max_scores, best_prev = torch.max(scores, dim=0)
                viterbi = max_scores
                backpointers.append(best_prev.cpu().tolist())

            viterbi_final = viterbi + end_trans
            best_tag = torch.argmax(viterbi_final).item()
            best_path = [best_tag]

            for bp in reversed(backpointers):
                best_tag = bp[best_tag]
                best_path.append(best_tag)

            best_path.reverse()
            results.append(best_path)

        return results

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CharAwareBiLSTMAttributeLabeler(BiLSTMAttributeLabeler):
    """
    Character-Aware BiLSTM-CRF sequence labeling model.
    Combines word-level embeddings with character-level CNN representations to handle OOV words.
    All weights are initialized randomly from scratch without external or pretrained assets.
    """
    def __init__(
        self,
        vocab_size: int,
        char_vocab_size: int,
        num_tags: int,
        embedding_dim: int = 128,
        char_embedding_dim: int = 32,
        char_out_dim: int = 64,
        char_kernel_size: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__(
            vocab_size=vocab_size,
            num_tags=num_tags,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx,
            char_vocab_size=char_vocab_size,
            char_embedding_dim=char_embedding_dim,
            char_out_dim=char_out_dim,
            char_kernel_size=char_kernel_size,
        )
