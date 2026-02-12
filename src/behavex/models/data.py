"""Contains the dataset class and the dataclass for the training args for each model"""
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List, Dict, Any, Literal
import torch
import numpy as np


class CustomDataset(Dataset):
    """
    Dataset for windowed time series and associated labels.
    Converts input arrays to torch tensors automatically, enforcing the correct shape.

    Supports:
    - Binary classification: labels as (N,) or (N, 1) with values 0/1
    - Multiclass classification: labels as (N,) with class indices 0..K-1
    """

    def __init__(
        self,
        X_windows: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        task_type: Literal["binary", "multiclass"] = "binary"
    ) -> None:
        """
        Args:
            X_windows: Array or Tensor of shape (num_samples, window_len, num_features)
            y: Array or Tensor of shape (num_samples,) or (num_samples, 1)
               - For binary: float values 0.0/1.0
               - For multiclass: integer class indices 0..K-1
            task_type: "binary" or "multiclass"
        """
        self.task_type = task_type

        # Convert to torch.Tensor if not already
        if not isinstance(X_windows, torch.Tensor):
            X_windows = torch.from_numpy(np.asarray(X_windows)).float()
        if not isinstance(y, torch.Tensor):
            if task_type == "multiclass":
                # CrossEntropyLoss expects long tensor with class indices
                y = torch.from_numpy(np.asarray(y)).long()
            else:
                y = torch.from_numpy(np.asarray(y)).float()

        # Handle label shape based on task type
        if task_type == "multiclass":
            # CrossEntropyLoss expects 1D tensor of class indices (N,)
            if y.ndim == 2:
                y = y.squeeze(1)
        else:
            # Binary: ensure labels are column vector (num_samples, 1)
            if y.ndim == 1:
                y = y.unsqueeze(1)
            elif y.ndim == 2 and y.shape[1] != 1:
                raise ValueError(f"For binary task, labels y should have shape (N,) or (N,1), got {y.shape}")

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

class MaskedTemporalDataset(Dataset):
    """
    Dataset for masked temporal transformer pretraining.
    Returns windows only (no labels); target = input for reconstruction loss.
    Uses lazy tensor conversion to avoid loading full data into GPU-ready tensors at init.
    When event_masks is provided, __getitem__ returns (X, event_mask) so loss can ignore pre-event positions.
    """

    def __init__(
        self,
        X_windows: Union[np.ndarray, torch.Tensor],
        event_masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        """
        Args:
            X_windows: (num_samples, window_len, num_features) time series windows
            event_masks: optional (num_samples, window_len) bool; when set, __getitem__ returns (X, event_mask)
        """
        if isinstance(X_windows, torch.Tensor):
            self.X = X_windows.contiguous()
        else:
            self.X = torch.from_numpy(np.ascontiguousarray(X_windows, dtype=np.float32))
        if event_masks is not None:
            if event_masks.shape[0] != X_windows.shape[0]:
                raise ValueError("event_masks length must match X_windows length")
            if isinstance(event_masks, torch.Tensor):
                self.event_masks = event_masks.contiguous()
            else:
                self.event_masks = torch.from_numpy(np.ascontiguousarray(event_masks, dtype=bool))
        else:
            self.event_masks = None

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.X[idx]
        if self.event_masks is not None:
            return x, self.event_masks[idx]
        return x

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
    # Multiclass support
    task_type: Literal["binary", "multiclass"] = "binary"
    num_classes: int = 2  # Number of classes (2 for binary, K for multiclass)
    class_names: List[str] = field(default_factory=list)  # Names of classes for metrics/display
    use_class_weights: bool = True  # Use inverse frequency weighting for imbalanced classes