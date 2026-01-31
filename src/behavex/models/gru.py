import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal

"""
This class implements a GRU model for classification of behavioral events.
The model is a simple RNN model with a GRU layer and a fully connected layer.
The input is a tensor of shape (batch, window, input_size).

For binary classification (output_size=1):
    Output is (batch, 1), apply sigmoid for probabilities.

For multiclass classification (output_size=K, K > 1):
    Output is (batch, K), apply softmax for class probabilities.
"""
class GRUModel(nn.Module):
    """
    GRU model for classification of behavioral events.

    Supports:
    - Binary classification: output_size=1, use BCEWithLogitsLoss
    - Multiclass classification: output_size=K (num classes), use CrossEntropyLoss
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1, num_layers: int = 1) -> None:
        """
        Args:
            input_size: Number of input features per timestep
            hidden_size: GRU hidden state dimension
            output_size: Number of output classes (1 for binary, K for multiclass)
            num_layers: Number of GRU layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, window, input_size)

        Returns:
            Logits of shape (batch, output_size)
            - For binary (output_size=1): shape (batch, 1)
            - For multiclass (output_size=K): shape (batch, K)
        """
        # x shape: (batch, window, input_size)
        rnn_out, _ = self.rnn(x)  # rnn_out: (batch, window, hidden_size)
        last_output = rnn_out[:, -1, :]  # take last time step's output (batch, hidden_size)
        out = self.fc(last_output)  # (batch, output_size)
        return out