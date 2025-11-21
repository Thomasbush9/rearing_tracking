
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Literal
from behavex.models.data import CustomDataset, GRUTrainingArgs
from behavex.models.gru import GRUModel
from behavex.models.viz import Vizualizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import Tensor

class Trainer:
    def __init__(
        self, 
        project=None,
        model_name: Literal["gru"] = "gru",):

        self.project = project
        self.model_name = model_name 
        self.training_args = self._build_training_args()
        #test it 
        print(self.training_args)

    def _build_training_args(self):
        """Extract training config from project and build GRUTrainingArgs"""
        train_cfg = self.project.config.get("model_defaults", {}).get("training", {})
        debug_cfg = self.project.config.get("debug", {})
        
        return GRUTrainingArgs(
            batch_size=train_cfg.get("batch_size", 32),
            learning_rate=train_cfg.get("lr", 1e-3),
            epochs=train_cfg.get("epochs", 20),
            weight_decay=train_cfg.get("weight_decay", 1e-5),  # default
            patience=train_cfg.get("patience", 5),  # default
            verbose=debug_cfg.get("verbose", False),
            save_model=train_cfg.get("save_model", True),  # default
            save_path=str(self.project.project_dir / "models"),  # derive from project
        )
