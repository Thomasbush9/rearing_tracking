"""Contains the dataset class and the dataclass for the trianing arts for each model"""
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Dict, Any
import torch
import numpy as np

class CustomDataset(Dataset):
    """
    Dataset for windowed time series and associated binary (or multiclass) labels.
    Converts input arrays to torch tensors automatically, enforcing the correct shape.
    """

    def __init__(
        self, 
        X_windows: Union[np.ndarray, torch.Tensor], 
        y: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """
        Args:
            X_windows: Array or Tensor of shape (num_samples, window_len, num_features)
            y: Array or Tensor of shape (num_samples,) or (num_samples, 1)
        """
        # Convert to torch.Tensor if not already
        if not isinstance(X_windows, torch.Tensor):
            X_windows = torch.from_numpy(np.asarray(X_windows)).float()
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(np.asarray(y)).float()

        # Ensure labels are column vector (num_samples, 1)
        if y.ndim == 1:
            y = y.unsqueeze(1)
        elif y.ndim == 2 and y.shape[1] != 1:
            raise ValueError(f"Labels y should have shape (N,) or (N,1), got {y.shape}")

        self.X = X_windows
        self.y = y

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"Number of samples in X ({self.X.shape[0]}) and y ({self.y.shape[0]}) must match"
            )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def __len__(self) -> int:
        return self.X.shape[0]

@dataclass
class GRUTrainingArgs:
    """
    Arguments for training a GRU model.
    """
    batch_size: int
    device: str
    input_size: int
    hidden_size: int
    output_size: int
    learning_rate: float
    epochs: int
    weight_decay: float
    patience: int
    verbose: bool
    save_model: bool
    save_path: str
    test_every_n_epochs: int