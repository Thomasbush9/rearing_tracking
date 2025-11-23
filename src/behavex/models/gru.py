import torch 
import torch.nn as nn
from torch import Tensor

"""
This class implements a GRU model for binary classification of behavioral events.
The model is a simple RNN model with a GRU layer and a fully connected layer.
The input is a tensor of shape (batch, window, input_size).
The output is a tensor of shape (batch, 1) with values between 0 and 1.
"""
class GRUModel(nn.Module):
    """
    GRU model for binary classification of behavioral events.
    """
    def __init__(self, input_size:int, hidden_size:int, output_size:int = 1, num_layers=1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        #TODO: add possibility for multiclass classification
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch, window, 4)
        rnn_out, _ = self.rnn(x)  # rnn_out: (batch, window, hidden_size)
        last_output = rnn_out[:, -1, :]  # take last time step's output for each sequence (batch, hidden_size)
        out = self.fc(last_output)       # (batch, 1)       # (batch, 1), probability between 0 and 1
        return out