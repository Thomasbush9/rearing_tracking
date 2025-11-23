import torch 
import torch.nn as nn
from pathlib import Path
import json
from typing import Literal
import numpy as np
import joblib

from behavex.models.gru import GRUModel

class Inference:
    def __init__(self, project=None, model_name: Literal["gru"] = "gru"):
        if project is None:
            raise ValueError("Project must be provided")
        
        self.project = project
        self.model_name = model_name
        self.model = None
        self.metadata = None
        self.scaler = None
        
        # Set up device (use same logic as Trainer)
        train_cfg = project.config.get("model_defaults", {}).get("training", {})
        self.device = train_cfg.get("device", "mps")
        
        # Model directory structure
        self.model_directory = project.project_dir / "models"

    def predict(self, x: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Predict probabilities on time series windows
        
        Args:
            x: Array of shape (TOT_FRAMES, window_size, features)
            batch_size: Batch size for inference
        """
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(x), batch_size):
                batch = x[i:i+batch_size]
                batch_tensor = torch.from_numpy(batch).float().to(self.device)
                output = self.model(batch_tensor)
                probs = torch.sigmoid(output).cpu().numpy().squeeze()
                all_probs.append(probs)
        
        return np.concatenate(all_probs)
    
    def warm_up(self, window_size: int, num_features: int, batch_size: int = 64):
        """Warm up the compiled model with a dummy forward pass
        
        Args:
            window_size: Window size used in the model
            num_features: Number of features
            batch_size: Batch size for inference
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        with torch.no_grad():
            # Create dummy input matching the expected shape (batch_size, window_size, num_features)
            dummy_input = torch.zeros(batch_size, window_size, num_features).float().to(self.device)
            _ = self.model(dummy_input)  # Triggers compilation if using torch.compile
        
        print(f"Model warmed up with shape: ({batch_size}, {window_size}, {num_features})")
    
    def _build_model(self, input_size: int, hidden_size: int, output_size: int = 1):
        """Build the model based on the model name and architecture parameters"""
        if self.model_name == "gru":
            return GRUModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        else:
            raise ValueError(f"Model name {self.model_name} not supported")
    
    def load_model(self, model_identifier: str = None):
        """Load a model by name or path
        
        Args:
            model_identifier: Model name (from registry) or full path to .pth file
                             If None, loads the most recent model
        
        Returns:
            dict: Model metadata if available, None otherwise
        """
        models_subdir = self.model_directory / "models"
        
        if model_identifier is None:
            # Load most recent model
            model_path = self._get_latest_model()
        elif Path(model_identifier).exists():
            # Direct path provided
            model_path = Path(model_identifier)
        else:
            # Model name from registry
            model_path = models_subdir / f"{model_identifier}.pth"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load metadata if available
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.metadata = metadata
            
            # Build model with architecture from metadata
            training_args = metadata["training_args"]
            self.model = self._build_model(
                input_size=training_args["input_size"],
                hidden_size=training_args["hidden_size"],
                output_size=training_args["output_size"]
            )
        else:
            raise FileNotFoundError(f"Metadata not found: {metadata_path}. Cannot determine model architecture.")
        
        # Load state dict
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load scaler using model name from path
        model_name = model_path.stem
        self.load_scaler(model_name)
        
        print(f"Model loaded: {model_path}")
        return metadata
    
    def load_scaler(self, model_name: str):
        """Load the scaler associated with a model
        
        Args:
            model_name: Model name (stem of .pth file, without extension)
        """
        # Load scaler
        scalers_dir = self.model_directory / "scalers"
        scaler_path = scalers_dir / f"scaler_{model_name}.pkl"
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        self.scaler = joblib.load(scaler_path)
        print(f"Scaler loaded: {scaler_path}")
    
    def _get_latest_model(self) -> Path:
        """Get the most recently saved model"""
        registry_path = self.model_directory / "model_registry.json"
        if not registry_path.exists():
            raise FileNotFoundError("No models found. Train and save a model first.")
        
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        # Sort by timestamp and return most recent
        latest = max(registry.items(), key=lambda x: x[1]["timestamp"])
        return Path(latest[1]["model_path"])