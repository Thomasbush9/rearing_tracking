
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Trainer:
    def __init__(
        self, 
        project=None,
        model_name: Literal["gru"] = "gru",):

        self.project = project
        self.model_name = model_name 
        self.training_args = self._build_training_args()
        self.model = self._build_model()
        self.device = self.training_args.device
        self.model = self.model.to(self.device)
        train_dataset, test_dataset = self._build_dataset()
        self.train_loader = DataLoader(train_dataset, batch_size=self.training_args.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.training_args.batch_size, shuffle=False)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.training_args.learning_rate, weight_decay=self.training_args.weight_decay)
        self.train_losses = []
        self.test_losses = []
        self.epochs = []
        self.test_epochs = []
        self.test_every_n_epochs = self.training_args.test_every_n_epochs
    def _build_model(self):
        """Build the model based on the model name"""
        if self.model_name == "gru":
            return GRUModel(input_size=self.training_args.input_size, hidden_size=self.training_args.hidden_size, output_size=self.training_args.output_size)
        else:
            raise ValueError(f"Model name {self.model_name} not supported")

    def _build_training_args(self):
        """Extract training config from project and build GRUTrainingArgs"""
        train_cfg = self.project.config.get("model_defaults", {}).get("training", {})
        debug_cfg = self.project.config.get("debug", {})
        
        return GRUTrainingArgs(
            batch_size=train_cfg.get("batch_size", 32),
            input_size=len(self.project.config["data"]["feature_set"]),
            hidden_size=self.project.config["model_defaults"]["training"].get("hidden_size", 64),
            output_size=1,
            device=train_cfg.get("device", "mps"),
            learning_rate=train_cfg.get("lr", 1e-3),
            epochs=train_cfg.get("epochs", 20),
            weight_decay=train_cfg.get("weight_decay", 1e-5),  # default
            patience=train_cfg.get("patience", 5),  # default
            verbose=debug_cfg.get("verbose", False),
            save_model=train_cfg.get("save_model", True),  # default
            save_path=str(self.project.project_dir / "models"),  # derive from project
            test_every_n_epochs=train_cfg.get("test_every_n_epochs", 1),
        )
    def _build_dataset(self):
        """Build the training and test dataset based on the training args"""
        # Reshape windows from (N, T, F) to (N*T, F) for scaling
        original_shape = self.project.train_windows.shape
        windows_2d = self.project.train_windows.reshape(-1, original_shape[-1])
        
        # Fit scaler on training data only
        self.scaler = StandardScaler()
        windows_scaled_2d = self.scaler.fit_transform(windows_2d)
        
        # Reshape back to (N, T, F)
        self.project.train_windows = windows_scaled_2d.reshape(original_shape)
        
        # Don't scale labels - they're binary (0/1)
        X_train, X_test, y_train, y_test = train_test_split(
            self.project.train_windows, 
            self.project.train_labels, 
            test_size=0.1, 
            random_state=42
        )
        train_dataset = CustomDataset(
            X_windows=X_train,
            y=y_train
        )
        test_dataset = CustomDataset(
            X_windows=X_test,
            y=y_test
        )
        return train_dataset, test_dataset
    
    def learning_step(self, X, y):
        """Perform one learning step: forward, backward, optimizer step"""
        X = X.to(self.device)
        y = y.to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self):
        """Train the model"""
        self.model.train()
        for epoch in range(self.training_args.epochs):
            running_loss = 0.0
            for X, y in self.train_loader:
                loss = self.learning_step(X, y)
                if np.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch+1}, stopping training")
                    return
                running_loss += loss * X.size(0)
            avg_loss = running_loss / len(self.train_loader.dataset)
            self.train_losses.append(avg_loss)
            self.epochs.append(epoch + 1)
            
            print(f"Epoch {epoch+1}/{self.training_args.epochs} - Train Loss: {avg_loss:.4f}", end="")
            if (epoch + 1) % self.test_every_n_epochs == 0:
                test_loss = self.test()
                self.test_losses.append(test_loss)
                self.test_epochs.append(epoch + 1)
                print(f" | Test Loss: {test_loss:.4f}")
            else:
                print()
    
    def test(self):
        """Evaluate the model on test set"""
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y)
                running_loss += loss.item() * X.size(0)
        avg_loss = running_loss / len(self.test_loader.dataset)
        self.model.train()
        return avg_loss  
