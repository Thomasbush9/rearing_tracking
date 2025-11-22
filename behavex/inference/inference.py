import torch
import json
from pathlib import Path
from datetime import datetime

# ... existing code ...

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

# Update train() to save model at the end
def train(self):
    """Train the model"""
    self.model.train()
    for epoch in range(self.training_args.epochs):
        # ... existing training loop ...
        
    self.plot_loss_history()
    
    # Save model after training completes
    if self.training_args.save_model:
        self.save_model()

Benefits:
-

