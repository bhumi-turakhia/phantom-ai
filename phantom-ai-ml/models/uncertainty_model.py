import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class EvidentialUncertaintyClassifier(nn.Module):
    """
    Custom Evidential Deep Learning neural network for forensic attribute uncertainty estimation.
    Directly models Dirichlet distribution parameters to quantify epistemic vs aleatoric uncertainty
    across 4 forensic states:
    0: EXPLICIT
    1: UNCERTAIN
    2: UNKNOWN
    3: CONTRADICTORY
    """
    NUM_STATES = 4

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_categories: int = 7,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_categories = num_categories

        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output head produces evidence e_k >= 0 for each category and state
        # Shape: (batch_size, num_categories * 4)
        self.evidence_head = nn.Linear(hidden_dim // 2, num_categories * self.NUM_STATES)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computing Dirichlet parameters and uncertainty.
        Args:
            features: Tensor of shape (batch_size, input_dim)
        Returns:
            probabilities: Tensor of shape (batch_size, num_categories, 4)
            uncertainty: Tensor of shape (batch_size, num_categories) in range (0, 1]
            alphas: Dirichlet parameters of shape (batch_size, num_categories, 4)
        """
        batch_size = features.size(0)
        h = self.shared_encoder(features)

        # Evidence must be non-negative (ReLU or softplus activation)
        evidence = F.softplus(self.evidence_head(h))
        evidence = evidence.view(batch_size, self.num_categories, self.NUM_STATES)

        # Dirichlet alpha parameters: alpha_k = evidence_k + 1
        alphas = evidence + 1.0
        # Dirichlet strength S = sum(alpha_k)
        S = torch.sum(alphas, dim=-1, keepdim=True)

        # Expected categorical probabilities: p_k = alpha_k / S
        probabilities = alphas / S

        # Epistemic uncertainty: u = K / S, where K is number of classes (4)
        # When evidence is 0, u = 4 / 4 = 1.0 (maximal uncertainty)
        uncertainty = (float(self.NUM_STATES) / S.squeeze(-1)).clamp(0.0, 1.0)

        return probabilities, uncertainty, alphas

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

