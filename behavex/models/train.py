
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
from datetime import datetime
import json
from pathlib import Path

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
        self.viz = Vizualizer()
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
            
            # Run validation every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.validation(epoch+1)
        self.plot_loss_history()
        if self.training_args.save_model:
            self.save_model()
    def plot_loss_history(self):
        plot_dir = self.project.project_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_path = plot_dir / f"loss_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.viz.plot_loss_history(self.train_losses, self.test_losses, save_path=save_path)

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
    
    def save_model(self, model_name: str = None):
        """Save the model with metadata for future loading
        
        Args:
            model_name: Optional custom name, otherwise auto-generated
        """
        models_dir = Path(self.training_args.save_path)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate descriptive filename
        if model_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"{self.model_name}_h{self.training_args.hidden_size}_e{self.training_args.epochs}_{timestamp}"
        
        model_path = models_dir / f"{model_name}.pth"
        metadata_path = models_dir / f"{model_name}_metadata.json"
        
        # Save model state dict
        torch.save(self.model.state_dict(), model_path)
        
        # Save metadata for reconstruction
        metadata = {
            "model_name": self.model_name,
            "model_path": str(model_path),
            "timestamp": datetime.now().isoformat(),
            "training_args": {
                "input_size": self.training_args.input_size,
                "hidden_size": self.training_args.hidden_size,
                "output_size": self.training_args.output_size,
                "batch_size": self.training_args.batch_size,
                "learning_rate": self.training_args.learning_rate,
                "epochs": self.training_args.epochs,
                "weight_decay": self.training_args.weight_decay,
                "device": self.training_args.device,
            },
            "feature_set": self.project.config["data"]["feature_set"],
            "window_size": self.project.config["data"]["window_size"],
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "final_test_loss": self.test_losses[-1] if self.test_losses else None,
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update model registry
        self._update_model_registry(model_name, metadata)
        
        print(f"Model saved: {model_path}")
        print(f"Metadata saved: {metadata_path}")
        return model_path

    def _update_model_registry(self, model_name: str, metadata: dict):
        """Update or create model registry index"""
        registry_path = Path(self.training_args.save_path) / "model_registry.json"
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {}
        
        registry[model_name] = {
            "model_path": metadata["model_path"],
            "metadata_path": str(Path(self.training_args.save_path) / f"{model_name}_metadata.json"),
            "timestamp": metadata["timestamp"],
            "hidden_size": metadata["training_args"]["hidden_size"],
            "epochs": metadata["training_args"]["epochs"],
            "final_test_loss": metadata["final_test_loss"],
        }
        
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2, sort_keys=True)

    def load_model(self, model_identifier: str = None):
        """Load a model by name or path
        
        Args:
            model_identifier: Model name (from registry) or full path to .pth file
                             If None, loads the most recent model
        """
        models_dir = Path(self.training_args.save_path)
        
        if model_identifier is None:
            # Load most recent model
            model_path = self._get_latest_model()
        elif Path(model_identifier).exists():
            # Direct path provided
            model_path = Path(model_identifier)
        else:
            # Model name from registry
            model_path = models_dir / f"{model_identifier}.pth"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load metadata if available
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            # Verify architecture matches
            if metadata["training_args"]["input_size"] != self.training_args.input_size:
                raise ValueError(f"Model input_size ({metadata['training_args']['input_size']}) "
                               f"doesn't match current config ({self.training_args.input_size})")
            if metadata["training_args"]["hidden_size"] != self.training_args.hidden_size:
                print(f"Warning: Model hidden_size ({metadata['training_args']['hidden_size']}) "
                      f"differs from current config ({self.training_args.hidden_size})")
                # Rebuild model with correct architecture
                self.training_args.hidden_size = metadata["training_args"]["hidden_size"]
                self.model = self._build_model().to(self.device)
        
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print(f"Model loaded: {model_path}")
        return metadata if metadata_path.exists() else None

    def _get_latest_model(self) -> Path:
        """Get the most recently saved model"""
        registry_path = Path(self.training_args.save_path) / "model_registry.json"
        if not registry_path.exists():
            raise FileNotFoundError("No models found. Train and save a model first.")
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Sort by timestamp and return most recent
        latest = max(registry.items(), key=lambda x: x[1]["timestamp"])
        return Path(latest[1]["model_path"])

    @staticmethod
    def list_models(project_dir: str) -> dict:
        """List all available models in a project
        
        Args:
            project_dir: Path to project directory
            
        Returns:
            Dictionary of model_name -> metadata
        """
        models_dir = Path(project_dir) / "models"
        registry_path = models_dir / "model_registry.json"
        
        if not registry_path.exists():
            return {}
        
        with open(registry_path, 'r') as f:
            return json.load(f)
    
    def validation(self, epoch:int):
        """Run validation over a temporal consistent timeseries for each session
        Saves plot for each run, session into session folder under validation folder
        """
        for session in self.project.sessions:
            # Skip if no validation data
            if not hasattr(session, 'val_windows') or session.val_windows is None:
                continue
                
            session_dir = session.session_root / "validation"
            session_dir.mkdir(parents=True, exist_ok=True)
            save_path = session_dir / f"{session.id}_{epoch}_validation.png"
            
            # Scale validation data using the same scaler as training data
            original_shape = session.val_windows.shape
            windows_2d = session.val_windows.reshape(-1, original_shape[-1])
            windows_scaled_2d = self.scaler.transform(windows_2d)
            val_windows_scaled = windows_scaled_2d.reshape(original_shape)
            
            # Run model over validation data
            with torch.no_grad():
                validation_data = Tensor(val_windows_scaled).float().to(self.device)
                predictions = self.model(validation_data)
                probs = torch.sigmoid(predictions).cpu().numpy().squeeze()
            
            # Ensure true_labels is 1D
            true_labels = np.array(session.val_labels).flatten()
            
            # Ensure probs is 1D and matches length
            if probs.ndim > 1:
                probs = probs.flatten()
            
            if len(probs) != len(true_labels):
                print(f"Warning: Shape mismatch for session {session.id}. Predictions: {probs.shape}, Labels: {true_labels.shape}")
                continue
            
            # Get actual frame indices (original frame numbers from temporal split)
            if hasattr(session, 'val_frame_indices') and session.val_frame_indices is not None:
                frame_indices = session.val_frame_indices  # Original frame indices from temporal split
            elif hasattr(session, 'val_indices') and session.val_indices is not None:
                frame_indices = session.val_indices  # Fallback to window indices
            else:
                frame_indices = np.arange(len(true_labels))
            
            # Debug: print label statistics
            num_positives = np.sum(true_labels)
            print(f"Session {session.id} - Validation: {len(true_labels)} frames, {num_positives} positive labels")
                
            self.viz.plot_validation_predictions(probs, true_labels, save_path=save_path, epoch=epoch, frame_indices=frame_indices)

        