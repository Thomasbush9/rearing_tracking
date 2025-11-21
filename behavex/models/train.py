
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
        model_name: Literal["gru"] = "gru",
        train_dataset: CustomDataset,
        test_dataset: CustomDataset,
        validation_data,
        training_args: GRUTrainingArgs,
        vizualizer: Vizualizer,
    ) -> None:
        self.model_name = model_name
        self.model = self.get_model(model_name)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_data = validation_data
        self.training_args = training_args
        self.vizualizer = vizualizer
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.training_args.batch_size,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.training_args.batch_size,
            shuffle=False,
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.training_args.learning_rate)
        #TODO: add support for other loss functions
        self.criterion = nn.BCEWithLogitsLoss()
        #TODO: add support for other devices
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Track loss history
        self.train_losses = []
        self.test_losses = []
        self.test_epochs = []
        self.epochs = []


    def get_model(self, model_name: Literal["gru"] = "gru") -> nn.Module:
        """Get the model."""
        if model_name == "gru":
            return GRUModel(input_size=self.train_dataset.X.shape[2], hidden_size=self.training_args.hidden_size, output_size=1, num_layers=self.training_args.num_layers)
        else:
            raise ValueError(f"Model {model_name} not supported.")
    def learning_step(self, X: Tensor, y: Tensor) -> float:
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.criterion(output.squeeze(), y.float())
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()
        
    def test(self) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output.squeeze(), y.float())
                total_loss += loss.item() * X.size(0)
        avg_loss = total_loss / len(self.test_loader.dataset)
        self.model.train()
        return avg_loss

    def train(self, ):
        """Train the model"""
        for epoch in tqdm(range(self.training_args.epochs)):
            unning_loss = 0.0
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
            test_loss = self.test(self.test_loader)
            self.test_losses.append(test_loss)
            self.test_epochs.append(epoch + 1)
            print(f" | Test Loss: {test_loss:.4f}")
