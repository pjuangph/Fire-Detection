"""FireMLP — variable-depth MLP for fire detection from aggregate features."""

from __future__ import annotations

import torch
import torch.nn as nn


class FireMLP(nn.Module):
    """MLP fire detector from aggregate features.

    Variable-depth architecture: n_features -> [hidden layers] -> 1.
    Output is raw logits; use BCEWithLogitsLoss or sigmoid for probabilities.

    Attributes:
        hidden_layers (list[int]): Hidden layer sizes used to build the network.
        net (nn.Sequential): The neural network layers.
    """

    def __init__(
        self,
        n_features: int = 12,
        hidden_layers: list[int] | None = None,
    ) -> None:
        """Initialize FireMLP.

        Args:
            n_features: Number of input features. Default 12.
            hidden_layers: List of hidden layer sizes, e.g. [64, 32, 16, 8].
                Default [64, 32] for backward compatibility.
        """
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 32]
        self.hidden_layers = list(hidden_layers)

        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, n_features).

        Returns:
            torch.Tensor: Raw logits of shape (batch,).
        """
        return self.net(x).squeeze(-1)
